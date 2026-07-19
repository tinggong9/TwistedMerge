# Real-adapter gauge stability

Evidence stage: **pilot**. Independent training groups: [0, 1, 2]. The trained holonomy factors were reused without modification; no adapter was retrained.

## Gate decision

- `every_primary_scramble_preserves_individual_adapters`: `True`
- `naive_factor_average_representation_dependent_in_every_group`: `True`
- `global_synchronization_invariant_in_every_group`: `True`
- `global_synchronization_substantially_more_stable_than_naive`: `True`
- `global_synchronization_validation_noninferior_to_unscrambled`: `True`
- `same_rank_cap_for_compared_methods`: `True`
- `at_least_three_independent_groups`: `True`

## Primary method summary

- `canonical_svd_factor_average`: maximum relative merged-delta change `3.195e-15`, maximum test-logit change `5.329e-15`, pooled test-accuracy range `0.011000`.
- `cycle_aware_alignment`: maximum relative merged-delta change `1.982e-15`, maximum test-logit change `1.155e-14`, pooled test-accuracy range `0.009625`.
- `full_delta_svd`: maximum relative merged-delta change `1.982e-15`, maximum test-logit change `1.155e-14`, pooled test-accuracy range `0.009625`.
- `global_synchronization`: maximum relative merged-delta change `2.710e-13`, maximum test-logit change `9.865e-13`, pooled test-accuracy range `0.026750`.
- `naive_factor_average`: maximum relative merged-delta change `1.034e+01`, maximum test-logit change `1.157e+01`, pooled test-accuracy range `0.248000`.
- `oracle_alignment`: maximum relative merged-delta change `1.359e-15`, maximum test-logit change `3.109e-15`, pooled test-accuracy range `0.010375`.
- `pairwise_reference_alignment`: maximum relative merged-delta change `8.053e-14`, maximum test-logit change `1.560e-13`, pooled test-accuracy range `0.014375`.

## Boundaries

Gauge scrambles are dependent representations and are never bootstrapped as independent observations. Paired intervals resample training groups after scramble-level metrics are averaged within group. Full-delta SVD is an invariant baseline. The ill-conditioned family is excluded from the primary claim. Accuracy is reported as a preservation boundary, not a superiority claim. Holonomy, Brauer, period-index, invariant-pooling, linter, and broad baseline experiments were not rerun.
