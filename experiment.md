# Experiment: Adaptive Spectral Interpolation for Muon

## Research question

Can an online-learned spectral interpolation over an evenly spaced `1/7` exponent grid improve supervised math fine-tuning versus standard Muon and fixed Muon-p (`p = 1/3`), without excessive runtime overhead?

**Hypothesis:** Adapting a continuous interpolation coordinate across a grid of fixed spectral powers can improve validation loss or convergence over fixed Muon and Muon-p. A negative result after accounting for computation is equally informative. **The learned coordinate `p` selects and interpolates grid updates; this is not the exact continuous spectral-power operator between grid points.**

## Setup

| Item | Initial plan |
|---|---|
| Model | Llama3-1B base checkpoint used in the Muon-p paper (verify exact checkpoint ID in authors' code before running) |
| Train | GSM8K training split |
| Validation | Fixed held-out subset of GSM8K *training* examples |
| Test | GSM8K official test set; optionally MATH-500 for out-of-distribution evaluation |
| Objective | Autoregressive supervised fine-tuning (cross-entropy; reproduce the paper's tokenization, formatting, and loss masking) |
| Parameters | Full-parameter fine-tuning; Muon-family optimizers for eligible 2D matrices, same fallback optimizer for embeddings, biases, normalization, and other ineligible tensors |
| Pilot budget | 600 steps for GSM8K (paper setting); shorter 50–100-step smoke tests first |
| Repetitions | 1 seed for debugging, then preferably 3 seeds for final curves |

**Paper-reproduction defaults (GSM8K):** The authors fine-tune Llama3-1B **base**, not a Qwen3 model; use **600 steps (10 epochs)**, tune learning rates in `{2e-5, 5e-5}` and batch sizes in `{128, 512}`, and use **6 polynomial iterations** for Muon and Muon-p. They evaluate GSM8K with **pass@8**, temperature **0.8**, estimated using **32 generations** over the **1,319 official test questions**. For faithful reproduction, first confirm the exact pretrained model repository/revision, training example formatting, loss masking, and parameter-group configuration from the authors' implementation rather than assuming these from the paper. Maintain a separate development split for meta-learning and model selection, so strict numerical replication of their published results may not be exact.

**Important:** MATH-500 is an evaluation benchmark, not a replacement for a sufficiently sized training set. Do not train on the held-out test examples.

## Main experiments

1. **Standard Muon (control):** polynomial spectral flattening corresponding to `p = 0`.
2. **Fixed Muon-p (paper baseline):** the authors' efficient polynomial optimizer at `p = 1/3` (not approximated by our grid).
3. **Learned grid-interpolated Muon-p (our primary algorithm):** an evenly spaced rational-exponent grid, with online optimization of one continuous interpolation coordinate `p_t` across the entire interval `[0,1]`.

**Primary grid:**

`P = {0, 1/7, 2/7, 3/7, 4/7, 5/7, 6/7, 1}`.

**Recommended ablations:** fixed grid coordinates (`p=0`, `p=3/7`, `p=1`, or several grid points), a fixed off-grid interpolation coordinate (`p=1/3` *within our method*, distinct from the paper's exact fixed `1/3` baseline), and optionally AdamW. A fixed-coordinate sweep is important: improvements must be attributable to *learning*, not simply to selecting a better fixed spectral geometry.

## Primary algorithm: learned interpolation on a `1/7` grid

For each eligible momentum matrix `G_t`, let `D_j(G_t)` denote the paper-style **SVD-free polynomial approximation** to the spectral power `j/7`, for `j=1,...,6`. Define `D_0` using standard Muon and `D_7` using the `p=1` (identity/momentum-direction) endpoint. Use consistent matrix orientation, normalization, update magnitude, and learning-rate scaling across endpoints. The grid does **not** require computing an SVD during training.

Maintain one shared learnable scalar `a_t`; set `p_t = sigmoid(a_t)`, constraining `0 < p_t < 1`. Initialize `p_t = 1/3` as a reasonable interior starting point (or use `p_t = 0.5` for a symmetric initialization). For every training step:

1. Locate the interval containing `p_t`: `j = min(floor(7 * p_t), 6)`, `p_lo = j/7`, `p_hi = (j+1)/7`.
2. Compute **only two** polynomial directions on the same momentum matrix: `D_lo = D_j(G_t)` and `D_hi = D_(j+1)(G_t)`. Both use the *same* input `G_t` and momentum state; do not maintain separate momentum histories for grid endpoints.
3. Set `alpha_t = 7 * p_t - j`, so `0 <= alpha_t <= 1`.
4. Interpolate `D(p_t) = (1 - alpha_t) * D_lo + alpha_t * D_hi`.
5. Apply `W_(t+1) = W_t - eta_t * D(p_t)` with a consistent parameter-group scaling rule.

At exact grid points, use that endpoint direction without ambiguity. The interpolation is **continuous** across grid points if neighboring polynomial routines agree on their shared endpoint, but its derivative can jump at grid boundaries. The discrete interval index `j` is not differentiated through; gradients are computed within the active interval. With sigmoid, the algorithm remains inside `(0,1)`; for exact endpoint experiments, specify a fixed exponent rather than relying on saturation.

### Why these exponents need new polynomial routines

For a rational power `p = a/b`, the paper constructs a scalar fixed-point iteration with a target `x = y^(a/b)`. When both reduced numerator and denominator are odd, one possible form is `x_next = x + c * (y^a - x^b)`. In our grid, the odd numerators `1/7`, `3/7`, and `5/7` admit this form. The even-numerator cases `2/7`, `4/7`, and `6/7` require a suitable odd-polynomial construction instead. Their constants, iteration counts, and normalization must be chosen and validated; **the paper's tuned `p=1/3` coefficients cannot simply be reused for the new grid**.

A suggested implementation milestone is to build and test all six interior grid transformations on small matrices before Llama fine-tuning. Compare against an **exact SVD reference in tests only** to measure directional relative error, stability near small singular values, and runtime. Use a common accuracy tolerance, and report actual matrix-multiplication counts rather than treating polynomial degree as a runtime proxy. Standard Muon (`0`) and the `1` identity endpoint should also be included in the numerical checks.

**Clarification:** For `p` strictly between grid points, `D(p)` is generally **not** `U diag(sigma^p) V^T`, even if the two endpoint routines were exact. Call the method **learned grid-interpolated spectral updates**, not exact learned Muon-p.

### How the paper’s polynomial works: meaning of `x` and `y`

The paper first describes the update **one singular value at a time**. In that scalar recurrence:

- `y` is an **original, fixed, normalized singular value** of the gradient/momentum matrix; it is **not** the polynomial.
- `x_k` is the **current approximation** to the desired transformed singular value `y^p`; it changes as the polynomial is iterated.
- `f(x_k, y)` is the **polynomial update rule**, such that `x_(k+1) = f(x_k, y)`.

For `p = 1/3`, the scalar iteration is

`x_(k+1) = x_k + c * (y - x_k^3)`, starting at `x_0 = y` after normalization.

At a fixed point, `y - x^3 = 0`, so the nonnegative fixed point is `x = y^(1/3)`. A fixed point alone does **not** guarantee convergence; the scaling, coefficient `c`, and iteration count determine numerical behavior.

For the corresponding matrix recurrence, `G` is the **fixed, normalized momentum matrix** and `X_k` is the **changing matrix approximation**:

`X_(k+1) = X_k + c * (G - X_k X_k^T X_k)`, with `X_0 = G`.

In the SVD notation `G = U diag(sigma) V^T`, this iteration aims for `X_k ≈ U diag(sigma^(1/3)) V^T`, without computing that SVD during training. The product `X_k X_k^T X_k` is the matrix analogue of cubing each singular value, while preserving the singular-vector directions in exact arithmetic. For a rectangular matrix, check the implementation’s orientation/transposition convention. The polynomial is the **rule `f`**, not `y`, `G`, or `X_k`.

**Note:** This illustrates the paper’s `p=1/3` baseline. Our primary method uses analogous pre-calibrated routines for the `1/7` grid and interpolates the two active neighboring outputs, rather than differentiating through a newly constructed polynomial at each changing `p`.

### Learning `p` with a one-step hypergradient

Use a *separate held-out meta-validation minibatch* to learn the shared unconstrained scalar `a` (with `p=sigmoid(a)`). During any one hypergradient step, keep the current interval `j`, the two endpoint directions, and their polynomial computations fixed:

1. On `B_train`, compute gradients/momentum and the two adjacent grid directions for the current `p_t`.
2. Form tentative weights `W'(p) = W - eta * D(p)` using the active interval.
3. Compute `L_meta(W'(p))` on disjoint `B_meta` and its gradient with respect to the tentative weights.
4. Inside the active interval, `dD/dp = 7 * (D_hi - D_lo)`. Thus, summing eligible matrices,

   `dL_meta/dp = -7 * eta * sum_matrices <grad_W' L_meta, D_hi - D_lo>`.

   For `a`, multiply by `dp/da = p * (1-p)` and take an Adam/SGD meta-step. If the learning-rate multiplier, direction rescaling, or normalization depends on `p`, differentiate through that dependence too; for the first implementation, normalize endpoints *before* mixing and avoid `p`-dependent post-mix rescaling.
5. Commit the **model update computed with the pre-meta-update `p`**. Do not use the meta-gradient to update the model weights directly.

Start by adapting `p` every `K=25` optimization steps rather than every step. The extra meta forward/backward pass and two polynomial transforms must count toward wall-clock comparisons. Because derivatives may be discontinuous at grid knots, log interval transitions; if updates oscillate at boundaries, reduce the meta-learning rate or add an explicit smoothness/stability penalty. Finite-difference-check the within-interval hypergradient on a tiny deterministic model before scaling up.

### Computational and numerical considerations

- **No per-step SVD:** grid directions use polynomial/matrix-multiplication recurrences; exact SVD is reserved for small-scale validation.
- **Two active endpoints:** the main implementation evaluates at most two grid transforms per eligible matrix in each optimizer step, not all eight. A transition to another interval changes which pair is evaluated.
- **Calibration is essential:** the `1/7` interior exponents are not drop-in variants of the paper's tuned `1/3` polynomial. Validate each polynomial's coefficient choice, convergence rate, and numerical error *before* training.
- **Shared `p` first:** one global scalar controls all eligible matrix parameters. Per-layer `p_l = sigmoid(a_l)` is a later ablation; different layers may occupy different grid intervals and incur additional kernel diversity.
- **Control update scaling:** track each endpoint's update RMS, the interpolated update RMS, endpoint cosine similarity, and gradient norms. Normalizing endpoints prevents trivial scale differences, but mixing can still change magnitude through cancellation.
- **Boundary behavior:** interpolation is piecewise differentiable, not globally smooth. At a knot use a documented one-sided derivative or a consistent interval-selection convention.

## Optional extension: exact learned exponent

On a **small model or selected small matrices only**, compare against an exact SVD reference `G = U diag(sigma) V^T` with `D_p = U diag(sigma^p) V^T`, differentiating the singular-value powers with respect to a continuous `p`. This provides a reference for the grid interpolation error and for whether adaptive spectral powers help at all. Full SVD on every Llama3-1B matrix per step is not the primary approach.

Constructing a new rational recurrence at every FP32 `p` is not practical: rational representations can have large numerators/denominators and can demand expensive or unstable high-degree iterations. A continuously parameterized polynomial is a separate, higher-risk numerical-methods project.

## Fair comparison protocol

- Identical checkpoint, tokenizer/chat template, dataset splits, sequence lengths, global effective batch size, data order, and seed for paired comparisons.
- Same choice of parameter groups and fallback optimizer for non-matrix parameters.
- Normalize grid endpoint update RMS consistently, report interpolation update norms, and check whether apparent gains arise from effective step-size changes.
- Tune or sweep learning rate for each baseline within the same stated tuning budget; record momentum, weight decay, warmup, scheduler, and learned-coordinate initialization `p_0`.
- Separate training, meta-validation (for adapting `p`), model-selection validation, and final test partitions. Account for the extra meta-batch forward/backward passes and two active polynomial transforms.
- Report both *steps/tokens* and *wall-clock time* (and peak GPU memory).
- Avoid selecting hyperparameters or choosing checkpoints using test accuracy.

## Measurements and plots

**Primary:** held-out *model-selection* validation response-token cross-entropy versus training steps and versus elapsed wall-clock time; do not use the meta-validation minibatches for the primary comparison.

**Secondary:** GSM8K exact-answer accuracy and paper-comparable pass@8 (temperature 0.8, 32 samples, where feasible); optionally MATH-500 accuracy (same decoding settings and answer extraction for all runs); training loss; peak memory; optimizer-step latency; learned `p_t` and active grid interval trajectories; endpoint and interpolated update norms; optional endpoint cosine similarity.

**Plots:**

1. Validation loss vs. optimization steps (all three methods).
2. Validation loss vs. wall-clock time (accounts for learned-p overhead).
3. Learned grid coordinate `p_t` vs. steps, with grid knots at `j/7` marked (label as an **interpolation coordinate**, not an exact spectral-power exponent).
4. Math accuracy vs. steps; add uncertainty bands across seeds if possible.

## Execution order

1. **Reproduce paper baselines:** GSM8K fine-tuning of the paper's Llama3-1B base checkpoint with standard Muon and the authors' exact fixed `p = 1/3` polynomial, using identical optimizer-eligible groups and fallback optimizer.
2. **Implement the primary `1/7` grid:** add SVD-free polynomial routines for `1/7` through `6/7`, plus `0` (standard Muon) and `1` (identity/momentum endpoint). Calibrate convergence coefficients and iteration counts for each; benchmark against SVD references on small synthetic matrices.
3. **Test grid interpolation:** confirm `p=j/7` yields the corresponding endpoint, intermediate `p` interpolates neighboring directions, adjacent intervals meet continuously, and only two transformations are computed per step. Check update magnitude and run 50–100-step smoke tests.
4. **Validate the hypergradient:** implement shared `p=sigmoid(a)` with a piecewise-linear interpolation rule. Compare its analytic one-step meta-gradient against finite differences away from grid boundaries on a tiny deterministic network.
5. **Run learned-grid fine-tuning:** run 600 matched Llama3-1B optimizer steps; adapt `a` using meta-validation roughly every 25 steps to start. Log `p`, interval transitions, norms, latency, and loss.
6. **Ablations and reporting:** compare learned grid interpolation against fixed coordinates (including at least a fixed midpoint and a short grid sweep), paper's fixed `p=1/3`, standard Muon, and optionally AdamW. Repeat with multiple seeds if feasible.

## Success criteria

A useful positive result is improved model-selection validation loss or math accuracy relative to standard Muon, the paper's fixed `p=1/3`, and strong **fixed-coordinate grid** baselines, at matched tokens **and** wall-clock time. A useful negative result is weak polynomial accuracy, unstable meta-learning across interval boundaries, excessive overhead, or a learned coordinate that matches a fixed configuration. Do not claim to have learned the exact continuously parameterized Muon-p operator between grid points.

## References

- Dong & Sawin, *Muon^p: Muon with Fractional Spectral Powers* (arXiv v1, 2026): https://arxiv.org/html/2606.13867v1
- Implementation: https://github.com/princeton-pli/muon-p
