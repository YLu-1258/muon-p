# Exact-SVD delayed learned-p experiment

Four methods use the same exact-SVD geometry: fixed p=0 (Muon), fixed p=1/3,
global delayed learned p, and per-matrix delayed learned p. The per-matrix variant
lives in `matrix_p/`; its standalone trainer and detailed dashboard remain available. There is no exponent grid or polynomial
approximation. The learned method reuses each new training gradient to update p;
there is **no extra meta forward/backward pass**. Original repository trainers
are unchanged. The earlier held-out-meta implementation has been replaced.

## Install and run all four

Use the same environment Python for installation and execution:

```bash
/data/alexl/miniconda3/env/muonp/bin/python -m pip install -r adaptive_spectral/exact_learned_p/requirements.txt
```

From the repository root, start all four methods **in parallel on four GPUs**:

```bash
/data/alexl/miniconda3/env/muonp/bin/python adaptive_spectral/exact_learned_p/run_comparison.py \
  --gpus 0 1 2 3 --steps 600 --batch-size 128 --microbatch 1 --eval-every 25 \
  --output adaptive_spectral/exact_learned_p/runs/comparison-four
```

Watch all four methods in another terminal:

```bash
watch --color --no-title -n 2 /data/alexl/miniconda3/env/muonp/bin/python adaptive_spectral/exact_learned_p/suite_status.py adaptive_spectral/exact_learned_p/runs/comparison-four
```

A shorter first GPU trial uses `--steps 20 --batch-size 16 --eval-every 5` and a
fresh output directory, for example `runs/pilot-comparison`. Microbatch controls
memory; effective batch size controls exposure. Smaller microbatches preserve
effective batch size through response-token-weighted gradient accumulation.

The launcher forwards shared training arguments identically to each method,
uses the launcher's Python executable for every subprocess, writes individual
`muon.log`, `fixed_third.log`, `learned.log`, and `matrix_learned.log` files.
GPU identifiers are in that order. Each subprocess gets its own
`CUDA_VISIBLE_DEVICES` and uses local `cuda:0`; model parameters are not sharded.
The launcher checks all four CUDA contexts before loading models. If any worker
fails, it stops the remaining workers. Interrupting the launcher stops all workers;
interrupting the dashboard only stops the watcher. Every suite/run requires a
fresh output directory. Do not use torchrun/DDP.

Without `--gpus`, the launcher uses the four identifiers in an existing
`CUDA_VISIBLE_DEVICES`, or defaults to 0,1,2,3 when it is unset. An explicit
`--gpus` overrides that environment variable. Exactly four distinct identifiers
are required. Use comparable GPU hardware for wall-clock comparisons; otherwise
the per-method hardware difference is a confound. Four concurrent model loads
and checkpoint saves also share host RAM, CPU, and disk bandwidth.

The default model is `meta-llama/Llama-3.2-1B`, an explicit pilot assumption rather
than a verified reproduction of the paper checkpoint. Have access approved and
use `hf auth login` if needed. `--model` also accepts a local checkpoint. Pin
`--revision` and `--dataset-revision` to immutable commits for final runs. The
resolved model commit is recorded and checked across runs when available.

## Loss curves and comparison outputs

After all four methods complete, the launcher automatically creates:

- `comparison/loss_curves.png` and `.svg`: selection-validation and raw training
  loss against optimizer steps and elapsed training hours.
- `comparison/p_trajectory.png`: global/fixed p and per-matrix mean/min–max band.
- `comparison/matrix_p_heatmap.png`: each learned matrix exponent over time.
- `comparison/matrix_p_history.csv` and `final_matrix_p.csv`: named values.
  Scalar methods use a `global` entry in these combined tables.
- `comparison/curves.csv`: all plotted measurements with response-token exposure.
- `comparison/summary.csv` and `.json`: final/best selection loss, final p,
  response tokens, elapsed training seconds, and number of p updates.

The plotter checks matching configurations, model revisions, split indices,
response-token exposure and validation steps. It refuses incomplete or mismatched
comparisons. The step-zero validation loss should also be identical. Curves are
single-seed measurements, not uncertainty estimates; learning-rate sweeps and
multiple seeds are needed before concluding that adaptation helps. Fixed controls
here are exact-SVD versions, not the paper's practical polynomial implementations.

Regenerate the plots without training again:

```bash
python adaptive_spectral/exact_learned_p/plot_comparison.py adaptive_spectral/exact_learned_p/runs/comparison-four
```

Elapsed training time includes validation and checkpointing after setup, and
starts before initial validation. It excludes model loading/data preparation.
Each method has its own time origin. Batch training losses are noisy and are
measured before that step's update; selection CE is measured after the update.

## Delayed learning algorithm

At step t, weights W_t were produced by the previous update using p_previous.

1. Compute the next batch's ordinary training gradient g_t at W_t.
2. From step 2 onward, contract **unclipped** g_t with the saved previous update
   derivative J_previous: `dL/dp_previous = -sum(lr_previous * <g_t, J_previous>)`.
3. Multiply by `p_previous*(1-p_previous)` and use Adam to update the shared scalar
   a, where `p=sigmoid(a)`. The previous learning rate is saved per matrix, so
   model LR warmup is handled correctly. Clamp a to [-8,8] after updating.
4. Clip g_t for the model optimizer, update momentum, and apply the next model
   update using the **new** p. Save its derivative for the next training step.

This is a delayed local hypergradient on the next **training batch**, not held-out
meta-learning and not differentiation through the entire optimizer trajectory.
There is one normal training backward per optimizer step. A run of N steps has
N-1 p updates; the last saved derivative is only consumed if training continues.
Model-selection validation is isolated and never supplies a gradient to p.
The gradient used for the contraction is preserved for the normal model update.

Each eligible matrix uses momentum `B=0.95 B+grad` and Nesterov input
`grad+0.95 B`. Reduced FP32 SVD is applied after Frobenius normalization. Updates
`U diag(s**p) V.T` are rescaled to **unit RMS**, so p controls spectral geometry
rather than a trivial change of update magnitude. The derivative includes RMS
normalization: subtract the `s**(2p)`-weighted mean log singular value from each
`log(s)` before reconstructing J. Zero matrices yield zero updates. Exactly zero
singular values contribute zero, with no rank cutoff for positive FP32 values.

Embeddings/output heads (including tied weights), biases and other non-2D tensors
use AdamW with betas (0.9,0.95). Other 2D tensors use the SVD optimizer. Groups
are recorded in the manifest. This targets unfused Llama matrices; other models'
fused QKV parameters are not split. Momentum history is held fixed in each local
hypergradient. Neither SVD factors nor previous training steps are backpropagated
through. Derivative matrices remain on the GPU between steps in learned mode;
fixed modes do not construct them.

Parameters/momentum are FP32; CUDA model passes use BF16 autocast and PyTorch
SDPA. Non-reentrant gradient checkpointing is enabled by default. No FlashAttention
build is needed. `--driver gesvd` is default for its better measured reconstruction
residual; `--driver gesvdj` permits the faster alternative. Neither mathematical
exactness nor small-mode accuracy is guaranteed by finite-precision SVD.

Defaults: 600 steps, batch 128, microbatch 2, length 1024, model/fallback LR 2e-5,
20-step linear LR warmup followed by constant LR, no weight decay, clipping norm
1, initial p=1/3, p Adam LR `--p-lr 0.05`. These are starting settings, not tuned
results. Tune model and p learning rates; 599 scalar updates can move p much more
than the old held-out-meta cadence. Full-model GPU time/memory remains to be
measured; reducing microbatch does not reduce the stored derivative matrices.

## Data

Only the GSM8K **training split** is loaded. A deterministic shuffle reserves 256
examples for model-selection validation; all remaining examples train the model.
There is no meta split or official-test access. All four runs use identical split
indices, initialization seed and sampling order. Sampling is uniform with
replacement, so 600 steps is not described as ten epochs.

Formatting matches the repository's non-chat `Question: ...\nAnswer: ` prompt.
Only answer tokens plus EOS contribute to loss; BOS is not inserted. Examples
are right-truncated, and those with no response targets are dropped and recorded.
Losses/accumulation are weighted by unmasked shifted response-token counts.
This pilot measures CE rather than generation accuracy or pass@8, and is not an
exact reproduction of the paper's preprocessing.

## Individual runs and checkpoints

For just the learned experiment:

```bash
python adaptive_spectral/exact_learned_p/train.py --mode learned \
  --output adaptive_spectral/exact_learned_p/runs/learned
watch --color --no-title -n 2 python adaptive_spectral/exact_learned_p/status.py adaptive_spectral/exact_learned_p/runs/learned
```

Fixed runs use `--mode fixed --p-init 0` or `--mode fixed --p-init 0.3333333333333333`.
Both dashboard scripts need only standard Python and accept `--no-color`.

Outputs include `manifest.json`, `splits.json`, `metrics.jsonl`, and atomically
replaced `progress.json`. `checkpoint.pt` is replaced every 100 steps and on the
final step. It includes model, both model optimizers, momentum, p optimizer/a,
RNG states, counters, and the pending derivative matrices/previous p needed for
an exact delayed continuation. FP32 checkpoints are large; atomic replacement
needs temporary space for old and new copies. `--save-every 0` disables saving.
The best selection loss is tracked but no separate best-model checkpoint is saved.

Resume an individual trusted checkpoint into a fresh directory with matching
scientific arguments and a larger total `--steps`. The suite launcher does not
resume incomplete suites. Old held-out-meta checkpoints are incompatible.

## Offline verification

```bash
python -m unittest discover -s adaptive_spectral/exact_learned_p -p 'test_*.py'
python adaptive_spectral/exact_learned_p/run_comparison.py --smoke --output /tmp/delayed-p-smoke
```

Smoke uses a tiny model and synthetic data on CPU, downloads nothing, and still
exercises all four concurrent methods and generates plots. Tests check analytic derivatives,
no extra training backward passes, first-step timing, split disjointness, tied
parameter exclusion, and checkpoint continuation including pending derivatives.
This verifies plumbing, not GPU performance or math reasoning improvements.

Per-matrix implementation and standalone commands: [matrix_p/README.md](matrix_p/README.md).
The smoke model has two eligible matrices for every method, with identical initialization.
Old three-method suite outputs remain readable individually, but the unified plotter
requires all four matched runs in its suite directory.
