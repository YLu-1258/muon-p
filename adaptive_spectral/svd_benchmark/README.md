# Llama-1B SVD benchmark

Run from the repository root in the CUDA PyTorch environment on **one GPU**:

```bash
CUDA_VISIBLE_DEVICES=0 python adaptive_spectral/svd_benchmark/benchmark.py
```

Watch from another terminal (standard Python is sufficient):

```bash
watch --no-title -n 2 python adaptive_spectral/svd_benchmark/status.py
```

Both commands default to `adaptive_spectral/svd_benchmark/results/default`.
For another run, pass a fresh `--output DIRECTORY` to the benchmark and pass
that directory as a positional argument to the watcher. Existing runs are not
overwritten. CSV records are flushed after every configuration and progress JSON
is atomically replaced before each warmup, measurement and validation phase.

The assumed Llama-3.2-1B configuration is 16 blocks, width 2048, MLP width 8192,
32 query heads and 8 KV heads with head dimension 64. The four cases are:

| Case | Matrix shape | Matrices per step |
|---|---|---:|
| Attention query/output | 2048 x 2048 | 32 |
| Attention key/value | 512 x 2048 | 32 |
| MLP gate/up | 8192 x 2048 | 32 |
| MLP down | 2048 x 8192 | 16 |

Embeddings and LM head are excluded. Confirm these assumptions for the eventual
checkpoint and optimizer parameter groups. Sequential estimates multiply each
case's median latency by its count. DDP replicates this work on every rank;
additional GPUs do not automatically divide the optimizer cost.

Each configuration warms up once and measures five synchronized runs. Inputs are
synthetic BF16 Gaussian momentum matrices; operations include FP32 conversion,
Frobenius normalization, reduced FP32 SVD, spectral reconstruction at p=1/3, and
BF16 output conversion. `update_derivative` also forms the analytic derivative
with respect to p, reusing the same decomposition. It does not differentiate
through the SVD. Two CUDA drivers (`gesvdj`, `gesvd`) are tested; `gesvda` is
excluded because it is approximate. TF32 is disabled. No torch.compile or model
downloads are used. Only PyTorch is needed (same environment as the first benchmark).

Timings exclude input creation, progress writes and residual validation. CSV
reports min/median/max latency, peak allocated tensor memory, incremental peak
above the resident input, and relative SVD reconstruction residual. Memory is
PyTorch allocated memory, not total device or allocator-reserved memory.
Factors are released between configurations. Driver failures are recorded and
remaining configurations continue; the process exits nonzero if any fail.

This estimates **optimizer overhead**, excluding model forward/backward, fallback
optimizer, momentum accumulation, parameter application, meta-validation and
distributed communication. A full-model trial remains necessary. Random matrices
do not reproduce ill-conditioned momentum spectra, which can change SVD runtime.
Derivative mode represents its cost on every step; if used only occasionally,
combine the two mode estimates according to the intended meta-update frequency.

CPU functional smoke test (not a performance estimate):

```bash
python adaptive_spectral/svd_benchmark/benchmark.py --device cpu --drivers default --smoke --repeats 1 --output /tmp/svd-smoke
python adaptive_spectral/svd_benchmark/status.py /tmp/svd-smoke
```
