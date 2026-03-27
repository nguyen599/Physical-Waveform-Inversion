out_dir = "out-shakespeare-char-20m-hc"

dataset = "shakespeare_char"

batch_size = 16
block_size = 256

gradient_accumulation_steps = 2

n_layer = 8
n_head = 8
n_embd = 448
dropout = 0.1
bias = False

max_iters = 20000
eval_interval = 500
learning_rate = 3e-4

hc_num_streams = 4
hc_num_fracs = 1
hc_disable = False
