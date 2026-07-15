# Baseline Regime Audit

Generated for benchmark series 28. This is a reproducibility and claim-boundary artifact, not paper prose.

## Scope

This audit separates two experimental regimes that should not be collapsed:

- Independent-seed/rebasin regime: local models are trained from different random initializations. Permutation, synchronization, and ReLU-gauge methods are meaningful because they address coordinate mismatch.
- Same-base task-vector regime: models are fine-tuned from one common base checkpoint. SLERP, Task Arithmetic, TIES, and DARE are meaningful because parameter deltas live in a shared coordinate system.

The current fixed-setting fixed-setting data are independent-seed small-network runs. They are not a same-base task-vector benchmark. Therefore Task Arithmetic, TIES, and DARE should not be judged from those rows unless a new common-base setup is created.

## Audit Table

| method | intended regime | common init? | validation? | capacity | output | fair on current fixed-setting fixed-setting data? | fair on new same-base task-vector benchmark? |
| --- | --- | --- | --- | --- | --- | --- | --- |
| weight averaging | same-base task-vector; same-dataset fine-tune; independent seed as weak/unprotected baseline | no | no | yes | single averaged model | yes as weak unaligned baseline | yes |
| greedy soup / Model Soups | same-base task-vector; same-dataset fine-tune; checkpoint soup over a common candidate pool | preferred | yes | yes | single averaged soup | yes as validation-descent baseline over candidate checkpoints | yes |
| SLERP | same-base task-vector; related checkpoints | yes in practice | optional | yes | single interpolated model | no | yes |
| Task Arithmetic | same-base task-vector; multi-task fine-tuning from one base | yes | optional | yes | single task-vector-composed model | no | yes |
| TIES | same-base task-vector; multi-task fine-tuning from one base | yes | optional | yes | single merged model | no | yes |
| DARE | same-base task-vector; dropout/rescale of fine-tuned deltas | yes | optional | yes | single merged model | no | yes |
| Git-ReBasin-style pairwise rebasin | independent seed; same architecture; same task/dataset | no | no | yes | single aligned average | yes | partly, as secondary diagnostic |
| C2M3-style synchronization | independent seed; same architecture; same task/dataset | no | no | yes | single synchronized aligned average | yes | partly, as secondary diagnostic |
| monomial gauge alignment | independent-seed ReLU networks with compatible positive scaling/permutation gauges | no | optional | yes | single same-capacity aligned average | yes | partly, architecture dependent |
| Twisted/rank-lifted branch candidates | controlled obstruction; central twist; optional real-model diagnostic | no | yes for selection/controls | no | branch or extra-capacity model | limited extra-capacity diagnostic only | limited, only with branch-capacity controls |

Machine-readable details are in `reports/csv/baseline_regime_audit.csv`.

## Key Claim Boundaries

Task Arithmetic, TIES, and DARE should not be judged from independent random-initialization model merging. Without a common base checkpoint, their task-vector or delta semantics are absent.

SLERP should not be treated as a fair failure or success case on arbitrary independently initialized checkpoints. It is best interpreted when endpoints are already in a comparable weight coordinate system, usually from a shared base or a separately aligned setup.

Git-ReBasin-style pairwise alignment, C2M3-style synchronization, and monomial gauge alignment are fair in the current independent-seed/rebasin regime because they explicitly address coordinate mismatch.

Greedy soup is fair in both regimes when the candidate pool and validation split are shared. Its interpretation changes: in same-base fine-tuning it is the canonical Model Soups-style baseline; in independent-seed runs it remains an empirical validation-descent candidate-selection boundary.

Twisted/rank-lifted branch candidates are not same-capacity single-model baselines unless distilled or parameter-matched. They require branch-capacity matched controls and should remain separated from SLERP/Task Arithmetic/TIES/DARE single-model comparisons.

## Allowed Claims

- The current fixed-setting independent-seed benchmark is fair for weight averaging, greedy soup over the same candidate pool, Git-ReBasin-style pairwise rebasin, C2M3-style synchronization, and monomial ReLU-gauge diagnostics.
- SLERP, Task Arithmetic, TIES, and DARE require a same-base or otherwise aligned task-vector benchmark before they can be evaluated as fair baselines.
- Greedy soup remains the empirical validation-descent boundary baseline in both regimes, but it does not make the independent-seed benchmark a task-vector benchmark.

## Forbidden Claims

- Do not claim Task Arithmetic, TIES, or DARE fail based on current independent random-initialization fixed-setting fixed-setting rows.
- Do not claim a broad win over SLERP unless SLERP is run in a same-base or otherwise alignment-safe regime.
- Do not compare rank-lifted branch candidates directly against same-capacity task-vector baselines without labeling extra capacity and including branch-capacity matched controls.
- Do not claim official external-baseline wins from faithful internal implementations or regime-audit rows.

## Recommended Next Benchmark

1. Create a same-base task-vector benchmark with one base `mlp2` checkpoint per dataset/setting, then fine-tune `N=3,4` task variants from that base using fixed train/validation/test splits.
2. Start with MNIST and Fashion-MNIST `mlp2`, width `128`, seeds `20+` per fixed setting, and task variants such as label-preserving domain shifts or class-balanced subsets. Keep architecture and base checkpoint identical across task vectors.
3. Compare same-base methods: weight averaging, uniform soup, greedy soup, SLERP, Task Arithmetic, TIES, and DARE. Tune any coefficients, densities, drop rates, or interpolation weights on validation only.
4. Include independent-seed methods only as secondary diagnostics in that benchmark: Git-ReBasin-style rebasin, C2M3 synchronization, and monomial gauge alignment. Mark them as rebasin diagnostics, not primary task-vector baselines.
5. Report two tables rather than one mixed leaderboard: same-base task-vector baselines in one table, independent-seed/rebasin baselines in another.

No new experiments were run for this prompt.
