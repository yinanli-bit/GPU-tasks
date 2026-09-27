# Task 0 common configuration

# output / evaluation
out_dir = 'out-task0'
eval_interval = 100
eval_iters = 100
log_interval = 10
always_save_checkpoint = True

# dataset
dataset = 'shakespeare_char'

# reproducibility
seed = 1337

# model
n_layer = 4
n_head = 4
n_embd = 128
block_size = 128
dropout = 0.1
bias = False

# training batch
batch_size = 32
gradient_accumulation_steps = 1

# optimization
learning_rate = 1e-3
max_iters = 3000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# learning rate schedule
decay_lr = True
warmup_iters = 100
lr_decay_iters = 3000
min_lr = 1e-4

# runtime
device = 'cuda'
compile = False

# logging
wandb_log = True
wandb_project = 'task0-nanogpt'
