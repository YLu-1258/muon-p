# Per-matrix delayed learned p

The per-matrix variant now lives inside the parent `exact_learned_p` experiment.
It learns one independent exponent for every eligible matrix (normally 112 for
the assumed Llama-3.2-1B), with an independent Adam coordinate and moments.
Embeddings/output heads, biases and other non-2D parameters use fallback AdamW.

## Unified four-GPU comparison

From the repository root:

```bash
/data/alexl/miniconda3/env/muonp/bin/python adaptive_spectral/exact_learned_p/run_comparison.py \
  --gpus 0 1 2 3 --steps 600 --batch-size 128 --microbatch 1 --eval-every 25 \
  --output adaptive_spectral/exact_learned_p/runs/comparison-four
```

GPU order: fixed p=0, fixed p=1/3, global learned p, per-matrix learned p.
All four run concurrently, each seeing only its assigned GPU as local `cuda:0`.

```bash
watch --color --no-title -n 2 /data/alexl/miniconda3/env/muonp/bin/python \
  adaptive_spectral/exact_learned_p/suite_status.py
```

The parent comparison automatically exports all four loss curves, exponent
trajectories, the per-matrix heatmap, and CSV/JSON tables. The small launcher,
suite watcher and plotter scripts in this directory forward to the unified
parent suite; there is no separate three-method runner anymore. Full setup,
GPU-selection rules, shared model/data defaults, outputs and interpretation are
in [the parent README](../README.md).

## Just the per-matrix method

```bash
CUDA_VISIBLE_DEVICES=0 /data/alexl/miniconda3/env/muonp/bin/python \
  adaptive_spectral/exact_learned_p/matrix_p/train.py --mode learned \
  --output adaptive_spectral/exact_learned_p/matrix_p/runs/learned

watch --color --no-title -n 2 /data/alexl/miniconda3/env/muonp/bin/python \
  adaptive_spectral/exact_learned_p/matrix_p/status.py
```

The detailed dashboard displays mean/median/min/max/std, a histogram, mean-p
history, and the eight matrices farthest from initialization with their delayed
hypergradients. `--all-matrices` shows every matrix; `--no-color` disables ANSI.
Summaries are unweighted across matrices. The watcher uses standard Python only.

## Algorithm and checkpoints

Each matrix m has `p_m=sigmoid(a_m)`, initialized at 1/3. On each step from step 2,
the normal next training gradient, before clipping, gives
`dL/dp_m = -lr_previous_m * <grad_current_m, derivative_previous_m>`.
Its own sigmoid derivative maps this to a_m. Vector Adam maintains separate
coordinate moments; gradients are never summed into a shared scalar. Updated p
values are used immediately for the next weight update, and each derivative is
saved for the next normal training gradient. A run of N steps has N-1 vector
updates. No extra model passes or SVDs are needed compared with global learned p.
Opposing signals can move their exponents in opposite directions; tune `--p-lr`
because each coordinate receives less aggregated evidence.

Directions and derivatives use the parent's unit-RMS exact-SVD helpers, including
the derivative of normalization. Parameters/momentum/SVD are FP32; CUDA model
passes use BF16 autocast. History and SVD factors are held fixed in each local
hypergradient. Selection validation never supplies gradients to p.

Checkpoints include stable matrix-name ordering, exponent vector/Adam moments,
model/momentum/fallback optimizer state, RNGs, and pending per-matrix derivatives
and previous p values. Resume trusted checkpoints with `train.py --resume FILE`
into a fresh output directory with matching scientific settings and a larger
`--steps`. Global-p checkpoints are incompatible. `--save-every 0` disables large
FP32 checkpoints. The suite launcher itself does not resume incomplete suites.

## Verification

```bash
/data/alexl/miniconda3/env/muonp/bin/python -m unittest discover \
  -s adaptive_spectral/exact_learned_p/matrix_p -p 'test_*.py'
```

The offline two-matrix fixture verifies independent routing, coupled-loss finite
differences, no extra training passes, and exact checkpoint continuation. The
parent `run_comparison.py --smoke --output /tmp/four-p-smoke` verifies the full
four-worker pipeline on CPU without model downloads. Full-size GPU training and
four-GPU throughput must be measured on your machine.
