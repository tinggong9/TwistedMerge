# Verification Pipeline Status

This report fixes the current pipeline roles before additional experiments are run. It names exactly which script should be used for the next paper-grade real run.

## Script Roles

| artifact | status | role |
| --- | --- | --- |
| experiments/model_merging_benchmark.py --mode verification | historical/descriptive | Retained for continuity with earlier benchmark plumbing. It is not the next paper-grade real verification run. |
| experiments/model_merging_fixed_setting_verification.py | paper-grade real verification script | Use this script for the next paper-grade real run measuring obstruction predictors, merge degradation targets, rank-lift comparisons, and confidence intervals. |
| experiments/controlled_twisted_overlap_benchmark.py | paper-grade controlled central-twist benchmark | Use this for controlled central-twist obstruction and rank-lift claims, not for natural real-model Brauer claims. |
| experiments/train_quality_sweep.py | only for choosing model-quality settings | This supports choosing train/width/epoch settings before the fixed-setting run; it is not a merge-claim experiment. |
| reports/monomial_gauge_alignment_report.md | implementation/descriptive until full repeated-seed runs exist | This supports exact ReLU-compatible gauge implementation and functional preservation, not performance. |

## Next Paper-Grade Real Run

The next paper-grade real run should use `experiments/model_merging_fixed_setting_verification.py`. That script is the current real-model verification entry point for obstruction predictors, alignment-conditioned targets, ordinary merge degradation, cycle-consistent merge, rank-lift comparisons, ensemble bounds, and bootstrap confidence intervals.

## Claim Boundary Notes

- `experiments/model_merging_benchmark.py --mode verification` is historical/descriptive and should not be cited as the final paper-grade real verification run.
- `experiments/model_merging_fixed_setting_verification.py` is the paper-grade real verification script.
- `experiments/controlled_twisted_overlap_benchmark.py` is the paper-grade controlled central-twist benchmark.
- `experiments/train_quality_sweep.py` is only for choosing model-quality settings.
- `reports/monomial_gauge_alignment_report.md` is implementation/descriptive until full repeated-seed runs exist.
- Fake-MNIST smoke runs are diagnostic only and never empirical support for paper claims.
