# Paper Evidence Summary

This summary separates implemented artifacts, completed runs, and claims supported by data. It is synchronized with `experiments/generate_claim_audit.py`.

## Implemented

- `experiments/model_merging_fixed_setting_verification.py` is implemented as the paper-grade real verification script.
- `experiments/controlled_twisted_overlap_benchmark.py` is implemented as the paper-grade controlled central-twist benchmark.
- `experiments/train_quality_sweep.py` is implemented for model-quality setting selection.
- `src/monomial_gauge_alignment.py` and `tests/test_monomial_gauge_alignment.py` implement and test exact ReLU-compatible monomial gauges.
- `experiments/generate_claim_audit.py` generates the Prompt 10 audit section plus CSV/TeX/status artifacts.

## Run Or Descriptive

- Controlled twisted-overlap outputs are present and support controlled rank-lift wording only.
- Training-quality sweep outputs are present and support design-choice wording only.
- Historical `experiments/model_merging_benchmark.py --mode verification` outputs are descriptive context only.
- `reports/monomial_gauge_alignment_report.md` is implementation/descriptive until full repeated-seed runs exist.
- Official external-code integration was attempted and documented, but no official external baseline metrics were produced.

## Supported By Data Or Tests

| claim_id | status | safe_wording |
| --- | --- | --- |
| controlled_twisted_overlap_rank_lift | supported controlled | In the controlled central-twist benchmark, rank-lifted branches are supported as controlled obstruction-structured evidence. |
| training_quality_sweep | supported design choice | The training-quality sweep supports choosing model-quality settings before the paper-grade verification run. |
| real_fixed_setting_obstruction_prediction | not yet supported unless full runs pass gates | The fixed-setting script is the paper-grade real verification entry point, but real obstruction-prediction claims remain gated until full observed repeated-seed runs pass the predefined statistical criteria. |
| monomial_gauge_functional_preservation | supported implementation | Positive monomial ReLU MLP gauges are implemented and tested as function-preserving transformations. |
| monomial_gauge_performance | not yet supported | Monomial gauge performance remains an open empirical question in this audit layer. |
| greedy_soup_win | not supported | Greedy soup remains a strong boundary baseline that exact-gauge methods do not robustly beat under the current evidence. |
| official_external_baseline_win | not supported | Official external-code integration was attempted and documented, but no official baseline win is claimed. |
| real_brauer_projective_residual | not supported | Real residuals remain non-Brauer under tested diagnostics; controlled period-index examples support the mathematics. |

## Unsupported Or Gated

- Real fixed-setting obstruction prediction is not yet supported unless full observed repeated-seed runs pass gates.
- Monomial gauge performance is not yet supported.
- A greedy soup win is not supported.
- An official external baseline win is not supported.
- Real Brauer/projective residual claims are not supported.
- Fake-MNIST smoke rows are diagnostic only and never empirical support.
