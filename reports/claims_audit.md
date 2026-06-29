# Claims Audit

This file tracks which claims are supported by current repository artifacts and which claims remain unsupported.

## Supported

| Claim | Status | Evidence |
| --- | --- | --- |
| TwistedMerge can detect failed gauge synchronization in a controlled `mu_2` central-twist example. | Supported | `tests/test_twisted_merge_algorithm.py` checks that the finite central twist has `status == "failed"` for `q=1` and `gauge.success == False`. |
| `q=2` branch lift can recover perfect prediction in the controlled finite central `mu_2` example. | Supported | `tests/test_twisted_merge_algorithm.py` checks `status == "twisted_rank_lifted"`, `twist_residual == 0`, and twisted zero-one loss `0`. |
| The code separates finite central coboundary twists from the nonzero `H^2(mu_2)` tetrahedral obstruction. | Supported | `tests/test_twisted_merge_algorithm.py` checks the H2 tetrahedral twist is non-coboundary and is not absorbed by the current `TwistedMerge` algorithm. |
| Lifted transition maps are no longer trivial placeholders for the finite central coboundary example. | Supported | `test_lifted_transition_maps_encode_nontrivial_edge_sign` checks that a nontrivial lifted edge uses `rho(-1)` rather than `rho(+1)`. |
| The model-merging benchmark now includes fixed-`N` repeated-seed MNIST checks and controlled injected-alignment negative controls. | Supported | `reports/model_merging_verification_report.md` and `reports/csv/model_merging_verification.csv` cover MNIST MLP, `N=3,4`, widths `16,32`, five seeds, and injected pairwise alignment noise. |
| CIFAR should remain plumbing-only in the current benchmark artifacts. | Supported | `reports/model_merging_verification_report.md` records that prior smoke-run CIFAR individual accuracy maxed at `0.1328`, below the `0.20` threshold for non-plumbing claims. |

## Not Yet Supported

| Claim | Status | Reason |
| --- | --- | --- |
| TwistedMerge beats external model-merging baselines. | Not yet supported | No external Git Re-Basin, C2M3, Model Soups, RegMean, TIES, or mergekit/MergeBench implementation has been run. |
| TwistedMerge solves natural MNIST/CIFAR merging. | Not yet supported | `reports/model_merging_verification_report.md` is stronger for MNIST than the smoke run, but it still reports descriptive prototype baselines and excludes CIFAR as near chance. |
| TwistedMerge fully trivializes a nonzero `H^2(mu_2)` class as an ordinary untwisted vector bundle. | Not yet supported | The nonzero tetrahedral `H^2` class is explicitly non-coboundary. Current `TwistedMerge` does not construct an edge-level untwisted descent for it. |
| The branch-prediction lift is a complete transition-map-level twisted sheaf descent implementation. | Not yet supported | The q=2 branch result is a controlled prediction-level sanity check. It is not a proof of full sheaf-level descent in the non-coboundary case. |
| Cycle obstruction score predicts weight-average merge degradation beyond the trivial number-of-models confound. | Not yet supported | In `reports/csv/model_merging_stats.csv`, fixed-`N` observed correlations are marked unsupported: `N=3` Pearson `-0.0347`, `N=4` Pearson `-0.3622`, and bootstrap intervals cross zero. |
| C2M3-style alignment or TwistedMerge/rank-lifted branching gives a statistically meaningful improvement over weight averaging on MNIST. | Not yet supported | Mean deltas in `reports/csv/model_merging_stats.csv` are descriptive; fixed setting rows have only five seeds and no significance claim. |
| TwistedMerge/rank-lifted branching beats greedy soup as a single-model baseline. | Not yet supported | `reports/model_merging_verification_report.md` labels `twisted_rank_lift_2` as a branch ensemble with extra capacity; fixed-`N` deltas versus greedy soup are mixed or negative. |

## Current Artifact Map

| Artifact | Role |
| --- | --- |
| `tests/test_twisted_merge_algorithm.py` | Regression tests for TwistedMerge success/failure modes. |
| `reports/twisted_merge_algorithm_verification.md` | Verification report for the hardened prototype. |
| `reports/twisted_merge_algorithm_report.md` | Demo report for the prototype algorithm. |
| `reports/synthetic_obstruction_report.md` | Separate nonzero `H^2(mu_2)` obstruction witness. |
| `reports/model_merging_report.md` | Small MNIST/CIFAR model-merging benchmark smoke report. |
| `reports/model_merging_verification_report.md` | Fixed-`N`, repeated-seed MNIST model-merging verification report. |
| `reports/csv/model_merging_verification.csv` | Per-baseline rows for observed and injected-alignment verification settings. |
| `reports/csv/model_merging_stats.csv` | Correlations, bootstrap intervals, deltas, and negative-result labels for verification settings. |
