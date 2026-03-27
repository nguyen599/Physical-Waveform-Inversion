out_dir = "out-shakespeare-char-hc"

dataset = "shakespeare_char"

batch_size = 32
block_size = 256

gradient_accumulation_steps = 2

n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2
bias = False

max_iters = 5000
eval_interval = 200
learning_rate = 3e-4

hc_num_streams = 4
hc_num_fracs = 1
hc_disable = False
