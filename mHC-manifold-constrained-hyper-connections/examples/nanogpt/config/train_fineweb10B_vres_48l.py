# FineWeb10B with Value Residual (attention-only, 48 layers)
# ~20M param GPT-2 style model
#
# Usage:
#   python train.py config/train_fineweb10B_vres_48l.py
#   torchrun --standalone --nproc_per_node=4 train.py config/train_fineweb10B_vres_48l.py

out_dir = "out-fineweb10B-vres-48l"
wandb_run_name = "vres-48l"
wandb_project = "mhc-nanogpt-48"

dataset = "fineweb10B"

# model
block_size = 1024
n_layer = 48
n_head = 6
n_embd = 150
dropout = 0.0
bias = False

# training
batch_size = 8
gradient_accumulation_steps = 4
max_iters = 5000
eval_interval = 500
log_interval = 10
eval_iters = 100

# optimizer
learning_rate = 6e-4
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# lr schedule
warmup_iters = 200
lr_decay_iters = 5000
min_lr = 6e-5

# dtype
dtype = "bfloat16"

# hyper-connections: DISABLED (baseline)
hc_num_streams = 1
hc_num_fracs = 1
hc_disable = True

# value residual
v_residual = True
v_residual_lamb_lr = 1e-2
