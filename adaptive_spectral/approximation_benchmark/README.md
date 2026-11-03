# Approximation benchmark

Requires Python 3.10+ and a CUDA-enabled PyTorch installation. Uses only PyTorch,
without importing the training stack. RTX A6000 supports FP32 and BF16; each run
uses one GPU and modest matrix sizes, well within its 48 GB memory.

From the repository root, in your CUDA PyTorch environment:

```bash
python adaptive_spectral/approximation_benchmark/benchmark.py --output adaptive_spectral/approximation_benchmark/results/a6000
```

Quick GPU smoke test:

```bash
python adaptive_spectral/approximation_benchmark/benchmark.py --shapes 32x64 --spectra log --steps 6 12 --coefficient-fractions 0.5 --repeats 1 --output /tmp/spectral-smoke
```

Use `--device cpu --dtypes float32` for a CPU functional check. Install a
CUDA-enabled build of `torch>=2.2` matching your driver if PyTorch is missing.
No model downloads are needed.

## Terminal monitor

In another terminal, from the repository root:

```bash
watch --no-title -n 2 python adaptive_spectral/approximation_benchmark/status.py results
```

Replace `results` with the benchmark's `--output` directory if customized.
The monitor follows the snapshot-under-`watch` pattern in `best-of-w`, showing
measurement progress, elapsed time, approximate ETA, recent errors/timings and
best observed directional errors. It requires no PyTorch and works with existing
CSV runs (elapsed/ETA need metadata from the updated benchmark). It is read-only;
an incomplete run may be running or stopped. For a single snapshot, omit `watch
--no-title -n 2`.

The default sweep covers the eight grid endpoints plus the paper's 1/3 power,
three coefficient fractions, five iteration budgets, both precisions, and flat,
six-decade log-spaced, and rank-deficient spectra in both rectangular orientations.
Matrices use common Frobenius normalization, with FP64 CPU SVD references.
Coefficients are fractions of conservative local-attraction bounds, not tuned
coefficients or guarantees of finite-step accuracy. The p=0 control uses the
repository's practical quintic Muon coefficients, which need not converge exactly
to the polar factor. The 1/3 recurrence is the paper's cubic form; its coefficient
is swept too. Even numerators use `x+c*(y^(2a)-x^(2b))*y`; this intentionally simple
valid construction may converge very slowly. It is not an optimized implementation.

`results.csv` records raw relative error, equal-norm directional error, cosine,
RMS ratio, projected relative error for nonzero singular modes below 1e-3 of the
maximum, matrix multiplication counts, synchronized latency, and CUDA peak total
allocated memory. Numerical null modes use a 1e-12 relative cutoff for the polar
reference. The fractional-power reference retains all computed singular values.
`metadata.json` records configuration, software and GPU. Timing excludes reference
SVD and data transfers, includes direction initialization and source powers, and
uses eager PyTorch with TF32 disabled. It measures individual endpoint transforms,
not the two-transform adaptive optimizer or training throughput. GPU memory includes
resident inputs; it is not an incremental allocation measurement.

To benchmark a real momentum matrix, save a single 2D tensor with `torch.save`,
then pass `--input /path/to/momentum.pt`. Synthetic sweeps alone cannot establish
useful iteration counts for training. Compare error/runtime tradeoffs before
choosing tolerances; no universal usefulness threshold is assumed.

For multiple A6000s, launch independent sweeps with `CUDA_VISIBLE_DEVICES=0` and
`CUDA_VISIBLE_DEVICES=1`, distinct `--seed` values and distinct output directories.
There is no distributed-process setup or cross-GPU communication.
