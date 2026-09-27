# ----- output -----
out_dir = 'out-observe-fp32'

# ----- dataset -----
dataset = 'shakespeare_char'

# ----- reproducibility -----
seed = 1337

# ----- model -----
n_layer = 4
n_head = 4
n_embd = 128
block_size = 128
dropout = 0.1
bias = False

# ----- batch -----
batch_size = 32
gradient_accumulation_steps = 1

# ----- optimization -----
learning_rate = 1e-3
max_iters = 100
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# ----- learning-rate schedule -----
decay_lr = True
warmup_iters = 10
lr_decay_iters = 100
min_lr = 1e-4

# ----- evaluation -----
eval_interval = 20
eval_iters = 50
log_interval = 1
always_save_checkpoint = False

# ----- precision -----
device = 'cuda'
dtype = 'float32'
allow_tf32 = False
compile = False

# ----- W&B -----
wandb_log = True
wandb_project = 'task0-nanogpt'
wandb_run_name = 'observe-fp32'
