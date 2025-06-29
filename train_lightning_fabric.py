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

from .config import cfg
from .dataset import CustomDataset
from .model import ModelEMA, Net
import math

from torch.optim.lr_scheduler import _LRScheduler
import lightning as L
from lightning.fabric import Fabric, seed_everything
import datetime

def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))

# from wandb.integration.lightning.fabric import WandbLogger
# import wandb

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
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def setup(rank, world_size):
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    return

def cleanup():
    dist.barrier()
    dist.destroy_process_group()
    return

class Net_ptl(L.LightningModule):

    def __init__(self, net):
        super().__init__()
        self.net = net

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, _ = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, x)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True, weight_decay=1e-3)
        return optimizer

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

def main(cfg):

    # ========== Datasets / Dataloaders ==========
    if cfg.local_rank == 0:
        print("="*25)
        print("Loading data..")
    train_ds = CustomDataset(cfg=cfg, mode="train")
    sampler= DistributedSampler(
        train_ds,
        num_replicas=cfg.world_size,
        rank=cfg.local_rank,
    )
    train_dl = torch.utils.data.DataLoader(
        train_ds,
        # sampler= sampler,
        batch_size= cfg.batch_size,
        num_workers= 3*torch.cuda.device_count(),
        pin_memory=True,
        drop_last=True,
        prefetch_factor=4,
    )

    valid_ds = CustomDataset(cfg=cfg, mode="valid")
    sampler= DistributedSampler(
        valid_ds,
        num_replicas=cfg.world_size,
        rank=cfg.local_rank,
    )
    valid_dl = torch.utils.data.DataLoader(
        valid_ds,
        # sampler= sampler,
        batch_size= cfg.batch_size_val,
        num_workers= 3*torch.cuda.device_count(),
        pin_memory=True
    )

    # wandb_logger = WandbLogger(project="Caformer")

    # ========== Model / Optim ==========
    model = Net(backbone=cfg.backbone, pretrained=False)
    f = f"{cfg.data_path}/unet2d_caformer_seed3_epochbest.pt"
    # state_dict= torch.load(f, map_location=cfg.device, weights_only=True)
    state_dict= torch.load(f, map_location='cpu', weights_only=True)
    state_dict= {k.removeprefix("_orig_mod."):v for k,v in state_dict.items()} # Remove torch.compile() prefix

    model.load_state_dict(state_dict)

    fabric = Fabric(precision='16-mixed')
    train_dl, valid_dl = fabric.setup_dataloaders(train_dl, valid_dl)
    model = torch.compile(model,
                          # fullgraph=True,
                          # mode="max-autotune"
                         )
    # model = model.to(cfg.local_rank)

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

    # model= DistributedDataParallel(
    #     model,
    #     device_ids=[cfg.local_rank],
    #     )

    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True, weight_decay=1e-3)
    # optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    model, optimizer = fabric.setup(model, optimizer)
    # ema_model = fabric.setup(ema_model)

    scheduler = ConstantCosineLR(optimizer, total_steps=len(train_dl)*cfg.epochs, pct_cosine=0.1)
    scaler = GradScaler()


    # ========== Training ==========
    if cfg.local_rank == 0:
        print("="*25)
        print("Model params", sum(p.numel() for p in model.parameters()))
        print("Give me warp {}, Mr. Sulu.".format(cfg.world_size))
        print("="*25)

    best_loss= 1_000_000
    val_loss= 1_000_000

    for epoch in range(0, cfg.epochs+1):
        if epoch != 0 or 1:
            tstart= time.time()
            train_dl.sampler.set_epoch(epoch)

            # Train loop
            model.train()
            total_loss = []
            for i, (x, y) in enumerate(train_dl):
                # x = x.to(cfg.local_rank)
                # y = y.to(cfg.local_rank)

                # with autocast(cfg.device.type):
                    # logits = model(x)
                logits = model(x)

                loss = criterion(logits, y)
                # loss = scaler.scale(criterion(logits, y))

                fabric.backward(loss)

                # scaler.scale(loss).backward()
                # scaler.unscale_(optimizer)

                # torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
                # fabric.clip_gradients(model, optimizer, max_norm=3.0, norm_type=2) # Gradient clipping is not implemented for optimizers handling the unscaling

                # scaler.step(optimizer)
                # scaler.update()
                optimizer.step()

                optimizer.zero_grad()
                scheduler.step()
                total_loss.append(loss.item())

                if ema_model is not None:
                    ema_model.update(model)

                if cfg.local_rank == 0 and (len(total_loss) >= cfg.logging_steps or i == 0):
                    train_loss = np.mean(total_loss)
                    total_loss = []
                    print("Epoch {}:     Train MAE: {:.2f}     Val MAE: {:.2f}     Time: {}     Step: {}/{}".format(
                        epoch,
                        train_loss,
                        val_loss,
                        format_time(time.time() - tstart),
                        i+1,
                        len(train_dl)+1,
                    ))
                if i%800==0 and i>0:
                    # ========== Valid ==========
                    model.eval()
                    val_logits = []
                    val_targets = []
                    with torch.no_grad():
                        for x, y in tqdm(valid_dl, disable=cfg.local_rank != 0):
            #                 x = x.to(cfg.local_rank)
            #                 y = y.to(cfg.local_rank)
            #
                            with autocast(cfg.device.type):
                            # with fabric.autocast():
                                if ema_model is not None:
                                    out = ema_model.module(x)
                                else:
                                    out = model(x)
                            # print(x[0])
                            # if ema_model is not None:
                            #     out = ema_model.module(x)
                            # else:
                            #     out = model(x)

                            val_logits.append(out.cpu())
                            val_targets.append(y.cpu())

                        val_logits= torch.cat(val_logits, dim=0)
                        val_targets= torch.cat(val_targets, dim=0)

                        loss = criterion(val_logits, val_targets).item()

                    # Gather loss
                    v = torch.tensor([loss], device=cfg.local_rank)
                    torch.distributed.all_reduce(v, op=dist.ReduceOp.SUM)
                    val_loss = (v[0] / cfg.world_size).item()

                    # ========== Weights / Early stopping ==========
                    stop_train = torch.tensor([0], device=cfg.local_rank)
                    if cfg.local_rank == 0:
                        es= cfg.early_stopping
                        if val_loss < best_loss:
                            print("New best: {:.2f} -> {:.2f}".format(best_loss, val_loss))
                            print("Start save weights..")
                            best_loss = val_loss
                            if ema_model is not None:
                                # print(ema_model.module.state_dict())
                                torch.save(ema_model.module.state_dict(), f'best_model_{cfg.seed}.pt')
                                # fabric.save(state=ema_model.module.state_dict(), path=f'best_model_{cfg.seed}.pt')
                            else:
                                torch.save(model.state_dict(), f'best_model_{cfg.seed}.pt')
                                # fabric.save(state=model.state_dict(), path=f'best_model_{cfg.seed}.pt')
                            print("Saved weights.")

        # ========== Valid ==========
        model.eval()
        val_logits = []
        val_targets = []
        with torch.no_grad():
            for x, y in tqdm(valid_dl, disable=cfg.local_rank != 0):
#                 x = x.to(cfg.local_rank)
#                 y = y.to(cfg.local_rank)
#
                with autocast(cfg.device.type):
                # with fabric.autocast():
                    if ema_model is not None:
                        out = ema_model.module(x)
                    else:
                        out = model(x)
                # print(x[0])
                # if ema_model is not None:
                #     out = ema_model.module(x)
                # else:
                #     out = model(x)

                val_logits.append(out.cpu())
                val_targets.append(y.cpu())

            val_logits= torch.cat(val_logits, dim=0)
            val_targets= torch.cat(val_targets, dim=0)

            loss = criterion(val_logits, val_targets).item()

        # Gather loss
        v = torch.tensor([loss], device=cfg.local_rank)
        torch.distributed.all_reduce(v, op=dist.ReduceOp.SUM)
        val_loss = (v[0] / cfg.world_size).item()

        # ========== Weights / Early stopping ==========
        stop_train = torch.tensor([0], device=cfg.local_rank)
        if cfg.local_rank == 0:
            es= cfg.early_stopping
            if val_loss < best_loss:
                print("New best: {:.2f} -> {:.2f}".format(best_loss, val_loss))
                print("Start save weights..")
                best_loss = val_loss
                if ema_model is not None:
                    # print(ema_model.module.state_dict())
                    torch.save(ema_model.module.state_dict(), f'best_model_{cfg.seed}.pt')
                    # fabric.save(state=ema_model.module.state_dict(), path=f'best_model_{cfg.seed}.pt')
                else:
                    torch.save(model.state_dict(), f'best_model_{cfg.seed}.pt')
                    # fabric.save(state=model.state_dict(), path=f'best_model_{cfg.seed}.pt')
                print("Saved weights.")

                es["streak"] = 0
            else:
                es= cfg.early_stopping
                es["streak"] += 1
                if es["streak"] > es["patience"]:
                    print("Ending training (early_stopping).")
                    stop_train = torch.tensor([1], device=cfg.local_rank)

        # Exits training on all ranks
        dist.broadcast(stop_train, src=0)
        if stop_train.item() == 1:
            return

    return

if __name__ == "__main__":

    # GPU Specs
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    _, total = torch.cuda.mem_get_info(device=rank)

    seed_everything(42)

    # Run
    cfg.local_rank= rank
    cfg.world_size= world_size
    main(cfg)
    # cleanup()
