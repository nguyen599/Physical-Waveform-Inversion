import os
import time
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler

import torch.distributed as dist
from torch.utils.data import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch._dynamo import OptimizedModule

from config import cfg
from dataset import CustomDataset
from model import ModelEMA, Net
from vr_adam import VRAdam
from vel2seis import SeismicGeometry

import math

from torch.optim.lr_scheduler import _LRScheduler
from torch.nn.modules.batchnorm import _BatchNorm

from typing import (
    Any,
    Callable,
    Optional,
    TypeVar,
    Union,
    overload,
)

from transformers import get_cosine_schedule_with_warmup
import datetime

def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))

def disable_running_stats(model):
    def _disable(module):
        if isinstance(module, _BatchNorm):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)

def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, _BatchNorm) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum

    model.apply(_enable)

class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups

class ConstantCosineLR(_LRScheduler):
    """
    Constant learning rate followed by CosineAnnealing.
    """
    def __init__(
        self,
        optimizer,
        total_steps,
        pct_cosine,
        last_epoch=-1,
        ):
        self.total_steps = total_steps
        self.milestone = int(total_steps * (1 - pct_cosine))
        self.cosine_steps = max(total_steps - self.milestone, 1)
        self.min_lr = 0
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1
        if step <= self.milestone:
            factor = 1.0
        else:
            s = step - self.milestone
            factor = 0.5 * (1 + math.cos(math.pi * s / self.cosine_steps))
        return [lr * factor for lr in self.base_lrs]


def set_seed(seed=1234):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

def setup(rank, world_size):

    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    # this will make all .cuda() calls work properly
    try:
        torch.cuda.set_device(rank)
    except:
        print("error at", rank)

    # dist.barrier()
    setup_for_distributed(rank == 0)
    return

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def cleanup():
    dist.barrier()
    dist.destroy_process_group()
    return

def _unwrap_compiled(obj: Union[Any, OptimizedModule]) -> tuple[Union[Any, nn.Module], Optional[dict[str, Any]]]:
    """Removes the :class:`torch._dynamo.OptimizedModule` around the object if it is wrapped.

    Use this function before instance checks against e.g. :class:`_FabricModule`.

    """
    if isinstance(obj, OptimizedModule):
        if (compile_kwargs := getattr(obj, "_compile_kwargs", None)) is None:
            raise RuntimeError(
                "Failed to determine the arguments that were used to compile the module. Make sure to import"
                " lightning before `torch.compile` is used."
            )
        return obj._orig_mod, compile_kwargs
    return obj, None

def main(cfg, use_sam=False, use_physical_loss=False):

    # ========== Datasets / Dataloaders ==========
    if cfg.local_rank == 0:
        print("="*25)
        print("Loading data..")
    cfg.inv_scale = False
    # if 'conv' in cfg.backbone:
    #     cfg.batch_size = cfg.batch_size + 8
    train_ds = CustomDataset(cfg=cfg,
                             mode="train",
                            inv_scale=cfg.inv_scale,
                            )
    sampler= DistributedSampler(
        train_ds,
        num_replicas=cfg.world_size,
        rank=cfg.local_rank,
    )
    print('torch.cuda.device_count()', torch.cuda.device_count())
    train_dl = torch.utils.data.DataLoader(
        train_ds,
        sampler= sampler,
        batch_size= cfg.batch_size,
        num_workers= 2*torch.cuda.device_count(),
        pin_memory=True,
        prefetch_factor=2 if 'conv' in cfg.backbone else 4,
        drop_last=True,
        persistent_workers=True
    )

    valid_ds = CustomDataset(cfg=cfg,
                             mode="valid",
                             inv_scale=False
                            )

    sampler= DistributedSampler(
        valid_ds,
        num_replicas=cfg.world_size,
        rank=cfg.local_rank,
    )
    valid_dl = torch.utils.data.DataLoader(
        valid_ds,
        sampler= sampler,
        batch_size= cfg.batch_size_val,
        num_workers= 2*torch.cuda.device_count(),
        # pin_memory=True
    )
    print(len(valid_dl.dataset))

    geo = SeismicGeometry()

    # ========== Model / Optim ==========
    model = Net(backbone=cfg.backbone, pretrained=False,
                inv_scale=not cfg.inv_scale)

    print('Load finetuned model')
    if 'caformer' in cfg.backbone:
        f = cfg.pretrained_weights
    else:
        f = "/kaggle/input/simple-further-finetuned-bartley-open-models/bartley_unet2d_convnext_seed1_epochbest_FT.pth"
    state_dict= torch.load(f, map_location=cfg.device, weights_only=True)
    state_dict= {k.removeprefix("_orig_mod."):v for k,v in state_dict.items()} # Remove torch.compile() prefix
    # print(state_dict.keys())
    if 'large' not in cfg.backbone:
        model.load_state_dict(state_dict, strict=True)

    model = torch.compile(model,
                          fullgraph=True,
                          dynamic=False,           # Assume fixed input shapes
                          # mode="max-autotune"
                         )
    model = model.to(cfg.local_rank)

    if cfg.ema:
        if cfg.local_rank == 0:
            print("Initializing EMA model..")
        ema_model = ModelEMA(
            model,
            decay=cfg.ema_decay,
            device=cfg.local_rank,
        )
    else:
        ema_model = None
    model= DistributedDataParallel(
        model,
        device_ids=[cfg.local_rank],
        static_graph=True,               # Performance optimization if graph is static
        )

    criterion = nn.L1Loss(reduction='mean')
    criterion_l1 = nn.L1Loss(reduction='mean')
    criterion_sl1 = nn.SmoothL1Loss()
    criterion_mse = nn.MSELoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, fused=True, weight_decay=1e-3)
    optimizer = VRAdam(model.parameters(), lr=1e-3, weight_decay=1e-3)

    if use_sam:
        base_optimizer = torch.optim.AdamW
        optimizer = SAM(model.parameters(), base_optimizer, lr=1e-4, fused=True, weight_decay=1e-3)
        scheduler = get_cosine_schedule_with_warmup(optimizer.base_optimizer,
                                                    num_training_steps=len(train_dl) * cfg.epochs,
                                                    num_warmup_steps=0)
    else:

        # optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
        if 'large' in cfg.backbone:
            scheduler = ConstantCosineLR(optimizer, total_steps=len(train_dl)*cfg.epochs, pct_cosine=0.05)
        else:
            scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                        num_training_steps=len(train_dl) * cfg.epochs,
                                                        num_warmup_steps=100)
    scaler = GradScaler()


    # ========== Training ==========
    if cfg.local_rank == 0:
        print("="*25)
        print("Model params", sum(p.numel() for p in model.parameters()))
        print("Give me warp {}, Mr. Sulu.".format(cfg.world_size))
        print("="*25)

    best_loss= 1_000
    val_loss= 1_000
    if 'conv' in cfg.backbone:
        max_epochs = 25
    else:
        max_epochs = 9
    for epoch in range(0, min(cfg.epochs+1, max_epochs+1)):
        if epoch != 0:
            tstart= time.time()
            train_dl.sampler.set_epoch(epoch)

            # Train loop
            model.train()
            total_loss = []
            for i, (x, y, _) in enumerate(train_dl):
                x = x.to(cfg.local_rank)
                y = y.to(cfg.local_rank)

                with torch.autocast(cfg.device.type, dtype=torch.bfloat16):
                    if use_sam:
                        enable_running_stats(model)
                        loss = criterion(model(x), y)  # use this loss for any training statistics

                    else:
                        logits = model(x)

                        if use_physical_loss:
                            loss += 0.5 * criterion(logits, y) + 0.5 * criterion(geo.simulate(logits), y)
                        else:
                            loss = criterion(logits, y)
                        # if i%2==0:
                        #     loss = torch.sqrt(criterion_mse(logits, y)*0.5+1e-7)
                        # else:
                        #     loss = criterion(logits, y)
                        # loss_l1 = criterion_l1(logits, y)
                        # print('loss_l1.shape', loss_l1.shape)
                        # print('loss_l1', loss_l1[0].detach().cpu().numpy())
                        # print('logits', logits[0].detach().cpu().float().numpy())
                        # print('y.shape', y.shape)
                        # print('y', y[0].detach().cpu().float().numpy())
                        # assert 0==1
                        # loss_l2 = criterion_mse(logits, y)
                        # loss = loss_l1+loss_l2

                if use_sam:
                    scaler.scale(loss).backward()
                    optimizer.step = optimizer.first_step
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)

                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                    # second forward-backward pass
                    disable_running_stats(model)
                    with torch.autocast(cfg.device.type, dtype=torch.float16):
                        loss = criterion(model(x), y)
                    scaler.scale(loss).backward()
                    optimizer.step = optimizer.second_step
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)

                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                else:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)

                    if best_loss>100:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)

                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                scheduler.step()
                with torch.no_grad():
                    total_loss.append(loss.item())

                if ema_model is not None:
                    ema_model.update(model)

                if cfg.local_rank == 0 and (len(total_loss) >= cfg.logging_steps or i == 0):
                    train_loss = np.mean(total_loss)
                    total_loss = []
                    print("Epoch {}:     Train MAE: {:.2f}     Val MAE: {:.2f}     Time: {}     Step: {}/{}    lr: {}".format(
                        epoch,
                        train_loss,
                        val_loss,
                        format_time(time.time() - tstart),
                        i+1,
                        len(train_dl)+1,
                        scheduler._last_lr,
                    ))
                if i%800==0 and i>0:
                    # ========== Valid ==========
                    model.eval()
                    val_logits = []
                    val_targets = []
                    ds_idx = []
                    with torch.no_grad():
                        for x, y, d in tqdm(valid_dl, disable=cfg.local_rank != 0):
                            x = x.to(cfg.local_rank)
                            y = y.to(cfg.local_rank)

                            with autocast(cfg.device.type):
                                if ema_model is not None:
                                    out = ema_model.module(x)
                                    if cfg.inv_scale:
                                        out = out*1500 + 3000
                                else:
                                    out = model(x)
                                    if cfg.inv_scale:
                                        out = out*1500 + 3000

                            val_logits.append(out.detach())
                            val_targets.append(y.detach())

                            ds_idx.extend(d)

                        val_logits= torch.cat(val_logits, dim=0)
                        val_targets= torch.cat(val_targets, dim=0)

                        # Overall loss
                        loss = criterion(val_logits, val_targets).item()

                    # Gather loss for distributed training
                    v = torch.tensor([loss], device=cfg.local_rank)
                    torch.distributed.all_reduce(v, op=dist.ReduceOp.SUM)
                    val_loss = (v[0] / cfg.world_size).item()

                    # Gather all logits and targets from all GPUs for per-class evaluation
                    if cfg.local_rank == 0:
                        # Gather tensors from all ranks
                        gathered_logits = [torch.zeros_like(val_logits) for _ in range(cfg.world_size)]
                        gathered_targets = [torch.zeros_like(val_targets) for _ in range(cfg.world_size)]
                        ds_idxs_list = [None for _ in range(cfg.world_size)]
                    else:
                        gathered_logits = None
                        gathered_targets = None
                        ds_idxs_list= None

                    # Collect tensors from all ranks
                    torch.distributed.gather(val_logits, gathered_logits, dst=0)
                    torch.distributed.gather(val_targets, gathered_targets, dst=0)
                    # torch.distributed.all_gather(gathered_logits, val_logits, dst=0)
                    # torch.distributed.all_gather(gathered_targets, val_targets, dst=0)
                    torch.distributed.gather_object(ds_idx, ds_idxs_list, dst=0)

                    # Per-class loss calculation (only on rank 0)
                    if cfg.local_rank == 0:
                        # Concatenate all gathered tensors
                        all_logits = torch.cat(gathered_logits, dim=0)
                        all_targets = torch.cat(gathered_targets, dim=0)
                        ds_idxs = []
                        for x in ds_idxs_list:
                            ds_idxs.extend(x)
                        # Dataset Scores
                        # ds_idxs = np.array(valid_ds.records)  # Remove the extra brackets
                        # ds_idxs = np.repeat(ds_idxs, repeats=500)

                        # Ensure mask length matches tensor length
                        expected_length = all_logits.shape[0]
                        if len(ds_idxs) != expected_length:
                            print(f"Warning: ds_idxs length {len(ds_idxs)} != tensor length {expected_length}")
                            # Trim or pad ds_idxs to match
                            if len(ds_idxs) > expected_length:
                                ds_idxs = ds_idxs[:expected_length]
                            else:
                                # Repeat the pattern to match expected length
                                ds_idxs = np.tile(ds_idxs, (expected_length // len(ds_idxs) + 1))[:expected_length]

                        print("="*25)
                        # print(all_logits)
                        class_losses = {}
                        for idx in sorted(np.unique(ds_idxs)):
                            # Mask for current class
                            mask = np.array(ds_idxs) == idx
                            # Calculate loss for this class
                            class_loss = criterion(all_logits[mask], all_targets[mask]).item()
                            class_losses[idx] = class_loss
                            print("{:15} {:.2f}".format(idx, class_loss))

                        print("="*25)
                        print("Val MAE (Overall): {:.2f}".format(val_loss))
                        print("="*25)

                    # ========== Weights / Early stopping ==========
                    stop_train = torch.tensor([0], device=cfg.local_rank)
                    if cfg.local_rank == 0:
                        es= cfg.early_stopping
                        if val_loss < best_loss:
                            print("New best: {:.2f} -> {:.2f}".format(best_loss, val_loss))
                            print("Saving weights..")
                            best_loss = val_loss
                            if ema_model is not None:
                                torch.save(ema_model.module.state_dict(), f'best_model_{cfg.seed}.pt')
                                torch.save({'model': ema_model.module.state_dict(),
                                           'optimizer_state': optimizer.state_dict(),
                                            'scheduler_state': scheduler.state_dict(),
                                           }, f'best_model_{cfg.seed}_all.pt')
                            else:
                                torch.save(model.state_dict(), f'best_model_{cfg.seed}.pt')
                            print("Saved weights.")

                            es["streak"] = 0
                        else:
                            es= cfg.early_stopping
                            es["streak"] += 1
                            if es["streak"] > es["patience"]:
                                print("Ending training (early_stopping).")
                                stop_train = torch.tensor([1], device=cfg.local_rank)

                    model.train()

        # ========== Valid ==========
        model.eval()
        val_logits = []
        val_targets = []
        ds_idx = []
        with torch.no_grad():
            for x, y, d in tqdm(valid_dl, disable=cfg.local_rank != 0):
                x = x.to(cfg.local_rank)
                y = y.to(cfg.local_rank)

                with autocast(cfg.device.type):
                    if ema_model is not None:
                        out = ema_model.module(x)
                        if cfg.inv_scale:
                            out = out*1500 + 3000
                    else:
                        out = model(x)
                        if cfg.inv_scale:
                            out = out*1500 + 3000

                val_logits.append(out.detach())
                val_targets.append(y.detach())

                ds_idx.extend(d)

            val_logits= torch.cat(val_logits, dim=0)
            val_targets= torch.cat(val_targets, dim=0)

            # Overall loss
            loss = criterion(val_logits, val_targets).item()

        # Gather loss for distributed training
        v = torch.tensor([loss], device=cfg.local_rank)
        torch.distributed.all_reduce(v, op=dist.ReduceOp.SUM)
        # print('v', v)
        val_loss = (v[0] / cfg.world_size).item()

        # Gather all logits and targets from all GPUs for per-class evaluation
        if cfg.local_rank == 0:
            # Gather tensors from all ranks
            gathered_logits = [torch.zeros_like(val_logits) for _ in range(cfg.world_size)]
            gathered_targets = [torch.zeros_like(val_targets) for _ in range(cfg.world_size)]
            ds_idxs_list = [None for _ in range(cfg.world_size)]
        else:
            gathered_logits = None
            gathered_targets = None
            ds_idxs_list= None

        torch.distributed.gather(val_logits, gathered_logits, dst=0)
        torch.distributed.gather(val_targets, gathered_targets, dst=0)
        # torch.distributed.all_gather(gathered_logits, val_logits, dst=0)
        # torch.distributed.all_gather(gathered_targets, val_targets, dst=0)
        torch.distributed.gather_object(ds_idx, ds_idxs_list, dst=0)

        # Per-class loss calculation (only on rank 0)
        if cfg.local_rank == 0:
            # Concatenate all gathered tensors
            all_logits = torch.cat(gathered_logits, dim=0)
            all_targets = torch.cat(gathered_targets, dim=0)
            ds_idxs = []
            for x in ds_idxs_list:
                ds_idxs.extend(x)
            # Dataset Scores
            # ds_idxs = np.array(valid_ds.records)  # Remove the extra brackets
            # ds_idxs = np.repeat(ds_idxs, repeats=500)

            # Ensure mask length matches tensor length
            expected_length = all_logits.shape[0]
            if len(ds_idxs) != expected_length:
                print(f"Warning: ds_idxs length {len(ds_idxs)} != tensor length {expected_length}")
                # Trim or pad ds_idxs to match
                if len(ds_idxs) > expected_length:
                    ds_idxs = ds_idxs[:expected_length]
                else:
                    # Repeat the pattern to match expected length
                    ds_idxs = np.tile(ds_idxs, (expected_length // len(ds_idxs) + 1))[:expected_length]

            print("="*25)
            # print(all_logits)
            class_losses = {}
            for idx in sorted(np.unique(ds_idxs)):
                # Mask for current class
                mask = np.array(ds_idxs) == idx
                # print(idx)
                # print(mask)
                # print(all_logits[mask])
                # print(all_targets[mask])
                # assert 1==0
                # Calculate loss for this class
                class_loss = criterion(all_logits[mask], all_targets[mask]).item()
                class_losses[idx] = class_loss
                print("{:15} {:.2f}".format(idx, class_loss))

            print("="*25)
            print("Val MAE (Overall): {:.2f}".format(val_loss))
            print("="*25)

        # ========== Weights / Early stopping ==========
        stop_train = torch.tensor([0], device=cfg.local_rank)
        if cfg.local_rank == 0:
            es= cfg.early_stopping
            if val_loss < best_loss:
                print("New best: {:.2f} -> {:.2f}".format(best_loss, val_loss))
                print("Saving weights..")
                best_loss = val_loss
                if ema_model is not None:
                    torch.save(ema_model.module.state_dict(), f'best_model_{cfg.seed}.pt')
                    torch.save({'model': ema_model.module.state_dict(),
                               'optimizer_state': optimizer.state_dict(),
                                'scheduler_state': scheduler.state_dict(),
                               }, f'best_model_{cfg.seed}_all.pt')
                else:
                    torch.save(model.state_dict(), f'best_model_{cfg.seed}.pt')
                print("Saved weights.")

                es["streak"] = 0
            else:
                es= cfg.early_stopping
                es["streak"] += 1
                if es["streak"] > es["patience"]:
                    print("Ending training (early_stopping).")
                    stop_train = torch.tensor([1], device=cfg.local_rank)

    torch.save({'model': ema_model.module.state_dict(),
               'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
               }, f'best_model_{cfg.seed}_epoch_{max_epochs}_all.pt')
        # Exits training on all ranks
        # dist.broadcast(stop_train, src=0)
        # if stop_train.item() == 1:
        #     return

    return

if __name__ == "__main__":

    # GPU Specs
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    _, total = torch.cuda.mem_get_info(device=rank)

    # Init
    print(f"Rank: {rank}, World size: {world_size}, GPU memory: {total / 1024**3:.2f}GB", flush=True)
    setup(rank, world_size)
    time.sleep(rank)
    print(torch.__version__)
    time.sleep(world_size - rank)

    # Seed
    set_seed(cfg.seed+rank)

    # Run
    cfg.local_rank= rank
    cfg.world_size= world_size
    main(cfg, use_sam=False)
    cleanup()