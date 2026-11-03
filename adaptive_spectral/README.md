# Adaptive spectral interpolation

Isolated experiments for the proposal in `../experiment.md`.

Start with [approximation_benchmark](approximation_benchmark/README.md) to
measure numerical accuracy and cost before implementing adaptive fine-tuning.
Nothing here changes the repository's training optimizers.

The [SVD benchmark](svd_benchmark/README.md) measures exact spectral updates at
Llama-1B matrix sizes and projects sequential per-step optimizer overhead.

The [exact learned-p pilot](exact_learned_p/README.md) runs single-GPU GSM8K
fine-tuning with an exact-SVD fixed or delayed learned exponent, a matched
parallel four-method comparison runner, exported loss curves, and colored monitoring.

The [per-matrix learned-p variant](exact_learned_p/matrix_p/README.md) assigns an independent
exponent to every eligible matrix, with distribution monitoring, matrix histories,
and participates in the unified four-method comparison.
