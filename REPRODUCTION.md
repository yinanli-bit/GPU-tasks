# Task 0 Reproduction Guide

This document describes how to reproduce the complete Task 0 workflow based on nanoGPT:

1. Environment setup
2. Data preparation
3. FP32 training
4. BF16 AMP training
5. Validation
6. Fixed-prompt inference
7. PyTorch Profiler

The formal FP32/BF16 training results in `results/task0_results.md` were produced on:

- GPU: NVIDIA GeForce RTX 3080 Ti
- PyTorch: 2.3.0+cu121
- CUDA runtime: 12.1
- cuDNN: 8.9.2
- Formal training Git commit: `68832f9`

The profiler support was added after the formal FP32/BF16 runs. The current `main` branch can be used to reproduce the complete workflow, including profiling.

---

# 1. Environment

## 1.1 Clone the repository

```bash
git clone https://github.com/yinanli-bit/GPU-tasks.git
cd GPU-tasks
```

Check the current repository status:

```bash
git status
git log -1 --oneline
```

For exact correspondence with the formal FP32/BF16 training runs:

```bash
git checkout 68832f9
```

> The formal FP32 and BF16 checkpoints/results in `results/task0_results.md` correspond to commit `68832f9`.
>
> The later profiler experiment requires the profiler-enabled version on `main`.

## 1.2 Create the Conda environment

Create the environment from the exported configuration:

```bash
conda env create -f environment.yml
```

Activate it:

```bash
conda activate task0
```

Verify Python and PyTorch:

```bash
which python
python --version
python -c "import torch; print('PyTorch:', torch.__version__)"
```

## 1.3 Verify CUDA and GPU

```bash
nvidia-smi
```

Then run:

```bash
python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("cuDNN:", torch.backends.cudnn.version())
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("BF16 supported:", torch.cuda.is_bf16_supported())
    print(
        "GPU memory GB:",
        torch.cuda.get_device_properties(0).total_memory / 1024**3
    )

    x = torch.randn(1024, 1024, device="cuda")
    y = x @ x
    print("CUDA matmul test:", y.shape)
PY
```

The formal experiments used a GPU with BF16 support.

## 1.4 Log in to Weights & Biases

Verify W&B installation:

```bash
wandb --version
```

Log in:

```bash
wandb login
```

Enter your own W&B API key when prompted.

Do not commit the API key to Git.

---

# 2. Data Preparation

Task 0 uses the character-level Shakespeare dataset supplied by nanoGPT.

Prepare the dataset with:

```bash
python data/shakespeare_char/prepare.py
```

This generates files including:

```text
data/shakespeare_char/train.bin
data/shakespeare_char/val.bin
data/shakespeare_char/meta.pkl
```

Verify them:

```bash
ls -lh \
  data/shakespeare_char/train.bin \
  data/shakespeare_char/val.bin \
  data/shakespeare_char/meta.pkl
```

`meta.pkl` contains the character vocabulary information.

The training and validation splits generated here are shared by the FP32 and BF16 experiments.

---

# 3. FP32 Training

The common Task 0 model configuration is stored in:

```text
config/task0.py
```

The main model configuration is:

```text
n_layer = 4
n_head = 4
n_embd = 128
block_size = 128
batch_size = 32
gradient_accumulation_steps = 1
dropout = 0.1
bias = False
max_iters = 3000
seed = 1337
compile = False
```

FP32 uses:

```text
dtype = float32
allow_tf32 = False
```

Disabling TF32 provides a stricter FP32 baseline.

Run the formal FP32 experiment:

```bash
python train_task0.py config/task0.py \
  --device=cuda \
  --dtype=float32 \
  --allow_tf32=False \
  --compile=False \
  --wandb_log=True \
  --wandb_run_name=task0-fp32-3080ti \
  --out_dir=out-task0-fp32
```

The run performs 3000 optimizer updates and periodically evaluates both the training and validation splits.

Expected checkpoints:

```text
out-task0-fp32/best.pt
out-task0-fp32/last.pt
out-task0-fp32/ckpt.pt
```

Check them with:

```bash
ls -lh out-task0-fp32
```

Inspect the final checkpoint metadata:

```bash
python - <<'PY'
import torch

ckpt = torch.load(
    "out-task0-fp32/last.pt",
    map_location="cpu"
)

print("iter_num:", ckpt["iter_num"])
print("best_val_loss:", float(ckpt["best_val_loss"]))
print("model_args:", ckpt["model_args"])
print("dtype:", ckpt["config"].get("dtype"))
print("seed:", ckpt["config"].get("seed"))
print("allow_tf32:", ckpt["config"].get("allow_tf32"))
PY
```

For the formal experiment, `iter_num` should be:

```text
3000
```

---

# 4. BF16 Training

The BF16 experiment uses the same:

- dataset split
- model architecture
- seed
- batch size
- block size
- optimizer settings
- learning-rate schedule
- number of training steps
- evaluation settings
- GPU

The primary experimental variable is the training precision.

Run:

```bash
python train_task0.py config/task0.py \
  --device=cuda \
  --dtype=bfloat16 \
  --allow_tf32=False \
  --compile=False \
  --wandb_log=True \
  --wandb_run_name=task0-bf16-3080ti \
  --out_dir=out-task0-bf16
```

With `dtype=bfloat16`, `train_task0.py` enters:

```python
torch.amp.autocast(
    device_type="cuda",
    dtype=torch.bfloat16
)
```

Therefore BF16 training uses PyTorch AMP autocast.

BF16 does not enable `GradScaler` in this implementation. `GradScaler` is enabled only for FP16.

Expected checkpoints:

```text
out-task0-bf16/best.pt
out-task0-bf16/last.pt
out-task0-bf16/ckpt.pt
```

Inspect the final BF16 checkpoint:

```bash
python - <<'PY'
import torch

ckpt = torch.load(
    "out-task0-bf16/last.pt",
    map_location="cpu"
)

print("iter_num:", ckpt["iter_num"])
print("best_val_loss:", float(ckpt["best_val_loss"]))
print("model_args:", ckpt["model_args"])
print("dtype:", ckpt["config"].get("dtype"))
print("seed:", ckpt["config"].get("seed"))
PY
```

The FP32 and BF16 formal runs should use the same fixed random seed:

```text
1337
```

The specific numeric value of the seed is arbitrary; consistency between experiments is what matters.

---

# 5. Validation

Validation is implemented inside `train_task0.py` through `estimate_loss()`.

During evaluation, the model does not perform backpropagation or parameter updates.

For each split, the script samples `eval_iters` batches:

```text
eval_iters batches
        ↓
compute loss for each batch
        ↓
average the losses
        ↓
reported train/validation loss
```

This produces a more stable estimate than using only one validation batch.

## 5.1 Validate FP32 checkpoint

```bash
python train_task0.py config/task0.py \
  --init_from=resume \
  --out_dir=out-task0-fp32 \
  --device=cuda \
  --dtype=float32 \
  --compile=False \
  --wandb_log=False \
  --eval_only=True
```

## 5.2 Validate BF16 checkpoint

```bash
python train_task0.py config/task0.py \
  --init_from=resume \
  --out_dir=out-task0-bf16 \
  --device=cuda \
  --dtype=bfloat16 \
  --compile=False \
  --wandb_log=False \
  --eval_only=True
```

Validation reports both loss and perplexity.

Perplexity is calculated as:

```text
PPL = exp(loss)
```

Lower validation loss and lower perplexity indicate that the model assigns higher probability to the validation data.

---

# 6. Inference

nanoGPT's `sample.py` is used as the inference entry point.

Both FP32 and BF16 checkpoints use the same fixed generation settings:

```text
Prompt: ROMEO:
num_samples = 1
max_new_tokens = 200
temperature = 0.8
top_k = 65
```

`ROMEO:` is the fixed input prompt.

`num_samples=1` generates one independent sample.

`max_new_tokens=200` allows the character-level model to generate up to 200 additional tokens/characters.

`temperature=0.8` controls sampling randomness. Values below 1.0 make sampling somewhat more conservative.

`top_k=65` allows sampling from the 65-character Shakespeare vocabulary.

## 6.1 FP32 inference

```bash
python sample.py \
  --out_dir=out-task0-fp32 \
  --device=cuda \
  --compile=False \
  --start="ROMEO:" \
  --num_samples=1 \
  --max_new_tokens=200 \
  --temperature=0.8 \
  --top_k=65
```

## 6.2 BF16 inference

```bash
python sample.py \
  --out_dir=out-task0-bf16 \
  --device=cuda \
  --compile=False \
  --start="ROMEO:" \
  --num_samples=1 \
  --max_new_tokens=200 \
  --temperature=0.8 \
  --top_k=65
```

The generated text is used to verify that the saved checkpoint can be loaded and used for autoregressive generation.

Generated text is not used as the primary quantitative comparison between FP32 and BF16 because sampling is stochastic.

---

# 7. Profiler

The profiler experiment uses PyTorch `torch.profiler`.

Its purpose is different from the normal throughput benchmark.

The normal training run answers:

```text
How fast is the complete training experiment?
```

The profiler answers:

```text
Where is the training time being spent?
```

It records information such as:

- PyTorch operators
- CUDA kernels
- CPU execution time
- CUDA execution time
- operator call counts
- tensor input shapes
- memory allocation behavior

Profiler measurements are not used as the official throughput numbers because profiling itself introduces runtime overhead.

## 7.1 Use the profiler-enabled code

If exact FP32/BF16 training reproduction was performed after checking out `68832f9`, return to the current profiler-enabled branch:

```bash
git switch main
git pull
```

Verify that profiler options exist:

```bash
grep -n "profile_run" train_task0.py
```

## 7.2 Profiling schedule

The configured schedule is:

```text
120 initial steps skipped
5 profiler warmup steps
30 consecutive active profiling steps
```

This avoids startup overhead and records a stable training region.

Validation is kept outside the active profiling region.

## 7.3 FP32 profiler

```bash
python train_task0.py config/task0.py \
  --device=cuda \
  --dtype=float32 \
  --allow_tf32=False \
  --compile=False \
  --wandb_log=False \
  --profile_run=True \
  --profile_dir=profiler/task0-fp32 \
  --out_dir=out-profiler-fp32 \
  --max_iters=160 \
  --eval_interval=1000 \
  --log_interval=10
```

Inspect the generated profiler summary:

```bash
cat profiler/task0-fp32/summary.txt
```

Check generated files:

```bash
find profiler/task0-fp32 -maxdepth 1 -type f -ls
```

Expected profiler outputs include:

```text
profiler/task0-fp32/summary.txt
profiler/task0-fp32/trace.json
```

## 7.4 BF16 profiler

Run the same workload with BF16:

```bash
python train_task0.py config/task0.py \
  --device=cuda \
  --dtype=bfloat16 \
  --allow_tf32=False \
  --compile=False \
  --wandb_log=False \
  --profile_run=True \
  --profile_dir=profiler/task0-bf16 \
  --out_dir=out-profiler-bf16 \
  --max_iters=160 \
  --eval_interval=1000 \
  --log_interval=10
```

Inspect:

```bash
cat profiler/task0-bf16/summary.txt
```

And:

```bash
find profiler/task0-bf16 -maxdepth 1 -type f -ls
```

The profiler runs use the same model dimensions and profiling schedule so FP32 and BF16 operator/kernel behavior can be compared.

Representative workload shapes include:

```text
Attention:
[32, 128, 4, 32]

MLP GEMM:
[4096, 128] x [128, 512]

MLP projection:
[4096, 512] x [512, 128]
```

The dimensions correspond to:

```text
batch_size = 32
sequence_length = 128
n_head = 4
head_dim = 32
n_embd = 128
MLP hidden dimension = 512
```

---

# Experiment Results

The formal FP32/BF16 measurements, checkpoint-to-W&B mapping, fixed-prompt outputs, and profiler observations are recorded in:

```text
results/task0_results.md
```

For the reported formal experiment:

```text
FP32 final validation loss: 1.6894
FP32 final PPL:             5.42
FP32 throughput:            415,554.86 tokens/s
FP32 peak GPU memory:       172.34 MB
FP32 training time:         52.39 s

BF16 final validation loss: 1.6927
BF16 final PPL:             approximately 5.43
BF16 throughput:            371,159.70 tokens/s
BF16 peak GPU memory:       116.51 MB
BF16 training time:         61.11 s
```

On this approximately 0.8M-parameter model, BF16 significantly reduced peak GPU memory but did not improve end-to-end training throughput.

Profiler results showed that individual BF16 GEMM kernels were faster, while non-GEMM work such as optimizer operations, LayerNorm, kernel launches, dispatch, and mixed-precision overhead remained significant for this small workload.

Therefore, AMP performance benefits depend on model size, tensor dimensions, hardware, and workload characteristics; mixed precision does not guarantee an end-to-end speedup.
