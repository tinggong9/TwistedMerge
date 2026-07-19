# Controlled LoRA gauge-invariance smoke

## Scope

This is one fixed synthetic four-adapter group with rank 3, not four trained adapters and not four independent seeds. Each well-conditioned gauge family has `20` dependent scrambles of the same effective updates. No model or hyperparameter was selected, no validation metric was consulted, and test labels were used only after the protocol was fixed. A paired bootstrap CI is intentionally absent because the scramble rows are not independent training groups.

## Gauge preservation

The maximum relative effective-delta error over the orthogonal, positive-diagonal, and dense families is `1.797e-15`. The ill-conditioned family is reported separately as a numerical boundary and is excluded from the primary claim.

Alignment solvers recorded `40` numerical failures across all method/representation rows, of which `0` occurred in the three well-conditioned primary families. Each is retained in `per_run.csv`; the affected method used the declared full-delta SVD safety fallback rather than silently dropping the setting.

## Primary representation-stability result

For moderately conditioned dense gauges:

- naive factor averaging has test-accuracy range `0.252930`, maximum relative merged-delta change `5.967e+00`, and maximum absolute logit change `9.367e+00`;
- global synchronization has test-accuracy range `0.000000`, maximum relative merged-delta change `1.463e-14`, and maximum absolute logit change `1.910e-14`;
- full-delta SVD has maximum relative merged-delta change `1.603e-15` and maximum absolute logit change `2.220e-15`.

The controlled result supports gauge stability only for the planted shared-B rank space and well-conditioned transforms. Full-delta SVD, Task Arithmetic-style effective-delta averaging, internal TIES-style and DARE-style controls, fixed-rank compression, canonical SVD factors, the ensemble, and separate-adapter controls are gauge-invariant baselines because they operate on effective updates or predictions. TwistedMerge is not the only invariant method.

## Cycle-aware boundary

A separately labeled injected transition inconsistency has maximum normalized cycle defect `2.021e-01`. The cycle-aware method chose `fallback_full_delta_svd` and returned the gauge-invariant full-delta SVD fallback. This is a diagnostic-only controlled inconsistency, not a natural Brauer or period-index class.

## Gate decisions

| Gate | Passed |
|---|---:|
| `twenty_scrambles_per_family` | `True` |
| `well_conditioned_gauges_preserve_effective_updates` | `True` |
| `well_conditioned_gauges_preserve_predictions` | `True` |
| `twistedmerge_methods_are_representation_stable` | `True` |
| `effective_delta_baselines_are_representation_stable` | `True` |
| `naive_factor_average_is_representation_dependent` | `True` |
| `exact_transition_cycles_close` | `True` |
| `cycle_aware_method_abstains_on_injected_inconsistency` | `True` |

## Negative boundaries and next step

- No real adapter was trained or evaluated.
- No method is claimed to improve ordinary merging accuracy or beat TIES, DARE, soups, Task Arithmetic, or SVD broadly.
- The single adapter group cannot support confidence intervals or a performance-generalization claim.
- Ill-conditioned `GL(r)` transforms are a numerical boundary rather than part of the successful invariance scope.
- The real-adapter pilot remains blocked until dataset licensing is resolved and a frozen independent-training-group protocol is written.
