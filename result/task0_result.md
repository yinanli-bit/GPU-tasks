# Task 0 Experiment Results

## 1. Experiment Setup

- Repository: nanoGPT-based Task 0 implementation
- Training code commit: `68832f9`
- GPU: NVIDIA GeForce RTX 3080 Ti
- GPU memory: 12 GB
- PyTorch: 2.3.0+cu121
- CUDA runtime used by PyTorch: 12.1
- cuDNN: 8.9.2
- BF16 supported: Yes

### Model configuration

- Dataset: `shakespeare_char`
- `n_layer = 4`
- `n_head = 4`
- `n_embd = 128`
- `block_size = 128`
- `batch_size = 32`
- `gradient_accumulation_steps = 1`
- `dropout = 0.1`
- `bias = False`
- `max_iters = 3000`
- `seed = 1337`
- `compile = False`
- `allow_tf32 = False`

FP32 and BF16 experiments used the same dataset split, seed, model
configuration, batch size, training steps, evaluation settings, and GPU.
The main experimental variable was training precision.

For BF16, AMP is enabled through `torch.amp.autocast` with
`dtype=torch.bfloat16`. BF16 does not use GradScaler in this implementation.

---

## 2. FP32 vs BF16 Results

| Metric | FP32 | BF16 |
|---|---:|---:|
| Final validation loss | 1.6894 | 1.6927 |
| Final validation PPL | 5.42 | 5.43 |
| Best validation loss | 1.6894 | 1.6927 |
| Stable throughput (tokens/s) | 415,554.86 | 371,159.70 |
| Peak GPU memory (MB) | 172.34 | 116.51 |
| Training time (s) | 52.39 | 61.11 |
| Final learning rate | 0.0001 | 0.0001 |
| `best.pt` size | 9.4 MB | 9.4 MB |
| `last.pt` size | 9.4 MB | 9.4 MB |
| Git commit | `68832f9` | `68832f9` |
| W&B run ID | `tc22risq` | `y7ukcnh0` |

Checkpoint locations:

- FP32 best: `out-task0-fp32/best.pt`
- FP32 last: `out-task0-fp32/last.pt`
- BF16 best: `out-task0-bf16/best.pt`
- BF16 last: `out-task0-bf16/last.pt`

Stable throughput was taken from the stable non-evaluation region rather
than evaluation iterations, because validation work increases iteration
wall-clock time.

BF16 reduced peak allocated GPU memory by approximately **32.4%** relative
to FP32. In this experiment, BF16 throughput was approximately **10.7% lower**
and total training time approximately **16.6% higher** than FP32.

The validation losses are nearly identical, indicating no obvious numerical
stability degradation from BF16 for this workload.

---

## 3. Checkpoint Traceability

The FP32 and BF16 formal training runs were produced from Git commit
`68832f9`.

Each experiment has:

1. a distinct output directory;
2. a distinct W&B run;
3. `best.pt` and `last.pt` checkpoints;
4. checkpoint metadata containing model arguments, training configuration,
   iteration number, and best validation loss.

The experiment manifest records the mapping:

### FP32

- Git commit: `68832f9`
- W&B run ID: `tc22risq`
- Best checkpoint: `out-task0-fp32/best.pt`
- Last checkpoint: `out-task0-fp32/last.pt`

### BF16

- Git commit: `68832f9`
- W&B run ID: `y7ukcnh0`
- Best checkpoint: `out-task0-bf16/best.pt`
- Last checkpoint: `out-task0-bf16/last.pt`

---

## 4. Fixed-Prompt Inference

Inference settings were kept fixed for both trained models:

- Prompt: `ROMEO:`
- `num_samples = 1`
- `max_new_tokens = 200`
- `temperature = 0.8`
- `top_k = 65`

### FP32 generation

> ROMEO:
>
> Now thy brid own, which you that see but our take
>
> On the day taughter other us of his barderly
>
> Have away with hand a with must of the of it.
>
> ESCALUS:
>
> I long earther, by mine at Hereliove the death p

### BF16 generation

> ROMEO:
>
> You before will and is the true: and but our take
>
> On thy call and barthing out him to back:
>
> And ane away, my facest to my must off Lord Citizen!
>
> Alas now it, even we sentend latter in overs,
>
> He will

Both checkpoints successfully support autoregressive generation. Generated
text is not used to rank FP32 versus BF16 because sampling is stochastic;
validation loss/PPL are used for quantitative quality comparison.

---

## 5. PyTorch Profiler

Profiler configuration:

- Profiler: `torch.profiler`
- Stable active training steps recorded: 30
- Initial training steps skipped: 120
- Profiler warmup steps: 5
- Active steps: 30
- Validation was kept outside the active profiling window
- `record_shapes = True`
- `profile_memory = True`

Profiler summaries:

- FP32: `profiler/task0-fp32/summary.txt`
- BF16: `profiler/task0-bf16/summary.txt`

### Observed tensor/workload shapes

Attention input:

`[32, 128, 4, 32]`

corresponding to:

`[batch_size, sequence_length, n_head, head_dim]`

where `head_dim = 128 / 4 = 32`.

Representative MLP GEMMs:

- `[4096, 128] x [128, 512]`
- `[4096, 512] x [512, 128]`

where `4096 = 32 x 128` and `512 = 4 x n_embd`.

For 30 profiled training steps:

- Attention forward calls: 120 = 30 steps x 4 layers
- Attention backward calls: 120
- LayerNorm backward calls: 270 = 30 x (4 x 2 + 1)

### FP32 / BF16 profiler observations

FP32 used kernels such as:

- `ampere_sgemm...`
- FP32 efficient-attention / CUTLASS kernels

BF16 used kernels such as:

- `ampere_bf16...gemm...`
- BF16 / flash-attention kernels

Representative MLP GEMMs were individually faster under BF16. For example,
the `[4096,128] x [128,512]` GEMM decreased from approximately 9.15 ms
aggregated CUDA time in FP32 to approximately 3.84 ms in BF16 over the
profiled region.

However, the complete BF16 training step was not faster. Other work such as
AdamW, LayerNorm, kernel launches, elementwise operations, dispatch, and AMP
overhead remained significant. Because this model contains only about
0.8 million parameters, its matrix multiplications are relatively small and
the fixed overheads form a large fraction of the total training time.

Therefore, on this workload BF16 provides a clear memory benefit but does
not provide an end-to-end throughput benefit.

Profiler measurements are used for bottleneck analysis only. They are not
used as the official throughput measurements because profiling itself adds
runtime overhead.

---

## 6. Main Conclusions

1. FP32 and BF16 reached very similar validation loss and perplexity.
2. BF16 reduced peak GPU memory by approximately 32%.
3. BF16 did not accelerate this small model on the RTX 3080 Ti.
4. Profiler results show that BF16 GEMM kernels themselves are faster, but
   non-GEMM and launch/dispatch overheads are significant for this small
   workload.
5. AMP acceleration is therefore workload-dependent rather than guaranteed.
6. The full Task 0 pipeline was verified end-to-end:
   data preparation, training, validation, checkpointing, W&B logging,
   fixed-prompt inference, and profiler analysis.

