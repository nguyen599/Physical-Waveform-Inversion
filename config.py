from types import SimpleNamespace
import torch

cfg= SimpleNamespace()
cfg.data_path = "path/to/your/data/"

cfg.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg.local_rank = 0
cfg.seed = 1
cfg.subsample = None

cfg.backbone = "convnext_small.fb_in22k_ft_in1k"
cfg.backbone = "convnext_large.fb_in22k_ft_in1k"
# cfg.backbone = "caformer_b36.sail_in22k_ft_in1k"

# caformer_b36.sail_in22k_ft_in1k
cfg.pretrained_weights = "./weights/unet2d_caformer_seed3_epochbest.pt"
# convnext_small.fb_in22k_ft_in1k
cfg.pretrained_weights = "./weights/bartley_unet2d_convnext_seed1_epochbest_FT.pth"

cfg.ema = True
cfg.ema_decay = 0.99

cfg.epochs = 100
cfg.batch_size = 32
cfg.batch_size_val = 16

cfg.early_stopping = {"patience": 3, "streak": 0}
cfg.logging_steps = 100