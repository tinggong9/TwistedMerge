# TwistedMerge Experimental Summary

Generated from local artifacts in `reports/` on 2026-06-29.

## Exact commands run

```bash
/Users/tinggong/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-synthetic.txt
.venv/bin/python experiments/synthetic_mu2_obstruction.py
.venv/bin/python experiments/synthetic_u1_obstruction.py
.venv/bin/python experiments/rank_lift_ablation.py
.venv/bin/python experiments/mnist_model_merging.py --check-only || true
.venv/bin/python experiments/cifar_model_merging.py --check-only || true
.venv/bin/python experiments/generate_summary.py
```

An earlier run failed before producing all outputs because Pandas 3.x required `jinja2` for LaTeX table generation. `jinja2>=3.1` was added to the requirements and the synthetic experiments were rerun successfully.

## Hardware and software environment

- Platform: macOS-26.0.1-arm64-arm-64bit
- Machine: arm64
- Processor: arm
- Python: 3.12.13 (main, Mar  3 2026, 15:35:03) [Clang 21.1.4 ]
- NumPy: 2.5.0
- Pandas: 3.0.4
- Matplotlib: 3.11.0
- Torch: not installed
- Torchvision: not installed

## Metrics

- `obstruction_score`: normalized triangle cocycle inconsistency.
- `oracle_accuracy`: test accuracy of the unmerged local synthetic models.
- `naive_accuracy`: test accuracy after descended single-gauge global merge.
- `rank_lift_accuracy`: test accuracy after rank-lifted branch merge with validation branch selection.
- `naive_failure`: `oracle_accuracy - naive_accuracy`.
- `rank_lift_gain`: `rank_lift_accuracy - naive_accuracy`.

## mu_2 synthetic obstruction

CSV: `reports/csv/synthetic_mu2_results.csv`  
Plot: `reports/plots/synthetic_mu2_obstruction.png`  
LaTeX table: `reports/tables/synthetic_mu2_summary.tex`

Correlation between obstruction score and descended-merge failure: `0.584`.

| flip_prob | obstruction_score_mean | naive_accuracy_mean | rank_lift_accuracy_mean | oracle_accuracy_mean | rank_lift_gain_mean |
| --- | --- | --- | --- | --- | --- |
| 0.000 | 0.000 | 0.933 | 0.933 | 0.980 | 0.000 |
| 0.020 | 0.076 | 0.933 | 0.933 | 0.980 | 0.000 |
| 0.050 | 0.148 | 0.933 | 0.933 | 0.980 | 0.000 |
| 0.100 | 0.283 | 0.933 | 0.933 | 0.980 | 0.000 |
| 0.200 | 0.399 | 0.921 | 0.934 | 0.980 | 0.012 |
| 0.300 | 0.480 | 0.810 | 0.935 | 0.980 | 0.125 |
| 0.400 | 0.495 | 0.720 | 0.936 | 0.980 | 0.216 |

## U(1) synthetic obstruction

CSV: `reports/csv/synthetic_u1_results.csv`  
Plot: `reports/plots/synthetic_u1_obstruction.png`  
LaTeX table: `reports/tables/synthetic_u1_summary.tex`

Correlation between obstruction score and descended-merge failure: `0.839`.

| noise_std | obstruction_score_mean | naive_accuracy_mean | rank_lift_accuracy_mean | oracle_accuracy_mean | rank_lift_gain_mean |
| --- | --- | --- | --- | --- | --- |
| 0.000 | 0.000 | 0.931 | 0.931 | 0.980 | 0.000 |
| 0.050 | 0.022 | 0.931 | 0.931 | 0.980 | 0.000 |
| 0.100 | 0.045 | 0.931 | 0.931 | 0.980 | 0.000 |
| 0.200 | 0.089 | 0.929 | 0.929 | 0.980 | 0.000 |
| 0.400 | 0.179 | 0.921 | 0.921 | 0.980 | 0.000 |
| 0.800 | 0.349 | 0.898 | 0.898 | 0.980 | 0.000 |
| 1.200 | 0.456 | 0.850 | 0.886 | 0.980 | 0.036 |

## Rank-lift ablation

CSV: `reports/csv/rank_lift_ablation.csv`  
Plot: `reports/plots/rank_lift_ablation.png`  
LaTeX table: `reports/tables/rank_lift_ablation.tex`

| experiment | rank | effective_rank | mean | std |
| --- | --- | --- | --- | --- |
| mu2 | 1 | 1 | 0.846 | 0.072 |
| mu2 | 2 | 2 | 0.934 | 0.005 |
| mu2 | 4 | 2 | 0.934 | 0.005 |
| mu2 | 8 | 2 | 0.934 | 0.005 |
| u1 | 1 | 1 | 0.903 | 0.011 |
| u1 | 2 | 2 | 0.903 | 0.011 |
| u1 | 4 | 4 | 0.903 | 0.011 |
| u1 | 8 | 8 | 0.918 | 0.004 |

## Claim status

| Claim | Status | Evidence |
| --- | --- | --- |
| Cocycle obstruction predicts descended/global merge failure on synthetic mu_2. | Supported, moderate. | Obstruction/failure correlation is `0.584`. At flip probability `0.40`, mean descended accuracy is `0.720` versus `0.933` at no flips. |
| Cocycle obstruction predicts descended/global merge failure on synthetic U(1). | Supported. | Obstruction/failure correlation is `0.839`. At noise std `1.20`, mean descended accuracy is `0.850` versus `0.931` at zero noise. |
| Rank-lifted merging fixes mu_2 synthetic failure. | Partially supported. | At flip probability `0.40`, rank-lift accuracy is `0.936` versus descended `0.720`. It does not reach oracle `0.980`. |
| Rank-lifted merging fixes U(1) synthetic failure. | Weakly supported at high obstruction only. | At noise std `1.20`, rank-lift accuracy is `0.886` versus descended `0.850`. Lower-noise settings show no measurable gain in this implementation. |
| MNIST model-merging claims. | Unsupported. | Status: `unsupported`. Reason: PyTorch/torchvision are not installed in this environment. |
| CIFAR model-merging claims. | Unsupported. | Status: `unsupported`. Reason: PyTorch/torchvision are not installed in this environment. |
| Comparisons against Git Re-Basin, C2M3, Model Soups, RegMean, TIES, mergekit/MergeBench. | Unsupported. | Baselines are listed in the README as future wrappers; no external baseline code was run. |

## Reproducibility notes

- Fixed seeds are encoded in each experiment config under `reports/configs/`.
- Synthetic runs do not require external repositories.
- PyTorch and torchvision were not installed for this run, so image experiments remain scaffolds only.
- The rank-lifted merge uses validation labels for branch selection; future comparisons should account for this when comparing against baselines that do not use validation branch selection.
