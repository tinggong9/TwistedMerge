# Clean Central Reproduction Report

## Decisions

- Controlled mu2 reproduction: **supported** as an executed controlled construction.
- Finite-Heisenberg period-index reproduction: **supported** as a checked representation-theoretic construction.

## Exact command

```bash
/Users/tinggong/Documents/GitHub/TwistedMerge/.venv/bin/python experiments/central_reproduction_next.py --seeds 0:29 --widths 32,64 --samples-per-chart 500 --samples-per-overlap 2000
```

- Git commit at execution: `0a41f76d3c8a77acc3a47514c2639b81fbc5b280`
- Controlled families: `mu2_coboundary, mu2_nontrivial_h2, random_noncentral`
- Widths: `32, 64`
- Seeds: `0:29`
- Label-permutation regression: `True`
- Saved candidate logits: `reports/next_benchmarks/logits/central_mu2_logits.npz`

## Controlled mu2

All candidate predictions are executed MLP, soup, branch, router, distilled-model, or ensemble operations. The `supplied_context_q2_branch_predictor` receives the exact face identity. The `validation_face_table_router` is labeled as a face-table diagnostic. `ensemble_reference` is extra-capacity and is not called an upper bound.

| family | width | method | n_seeds | mean_test_accuracy | mean_test_loss | parameter_multiplier | branch_count | inference_multiplier |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mu2_coboundary | 32 | c2m3_synchronized | 30 | 1 | 0.407134 | 1 | 1 | 1 |
| mu2_coboundary | 32 | distilled_single_model_control | 30 | 1 | 0.407134 | 1 | 1 | 1 |
| mu2_coboundary | 32 | ensemble_reference | 30 | 1 | 0.407134 | 4 | 4 | 4 |
| mu2_coboundary | 32 | git_rebasin_pairwise | 30 | 1 | 0.407134 | 1 | 1 | 1 |
| mu2_coboundary | 32 | no_twist_branch_control | 30 | 1 | 0.407134 | 2 | 2 | 2 |
| mu2_coboundary | 32 | ordinary_weight_average | 30 | 0.500975 | 0.693147 | 1 | 1 | 1 |
| mu2_coboundary | 32 | parameter_matched_wide_control | 30 | 0.971479 | 0.476119 | 2.06066 | 1 | 1 |
| mu2_coboundary | 32 | random_branch_control | 30 | 0.4 | 0.885258 | 2 | 2 | 2 |
| mu2_coboundary | 32 | supplied_context_q2_branch_predictor | 30 | 1 | 0.407134 | 2 | 2 | 2 |
| mu2_coboundary | 32 | validation_face_table_router | 30 | 1 | 0.407134 | 2 | 2 | 2 |
| mu2_coboundary | 32 | validation_global_branch_selector | 30 | 1 | 0.407134 | 2 | 2 | 2 |
| mu2_coboundary | 32 | wrong_context_control | 30 | 1 | 0.407134 | 2 | 2 | 2 |
| mu2_coboundary | 32 | wrong_twist_control | 30 | 0 | 1.20431 | 2 | 2 | 2 |
| mu2_coboundary | 64 | c2m3_synchronized | 30 | 1 | 0.407374 | 1 | 1 | 1 |
| mu2_coboundary | 64 | distilled_single_model_control | 30 | 1 | 0.407374 | 1 | 1 | 1 |
| mu2_coboundary | 64 | ensemble_reference | 30 | 1 | 0.407374 | 4 | 4 | 4 |
| mu2_coboundary | 64 | git_rebasin_pairwise | 30 | 1 | 0.407374 | 1 | 1 | 1 |
| mu2_coboundary | 64 | no_twist_branch_control | 30 | 1 | 0.407374 | 2 | 2 | 2 |
| mu2_coboundary | 64 | ordinary_weight_average | 30 | 0.501079 | 0.693147 | 1 | 1 | 1 |
| mu2_coboundary | 64 | parameter_matched_wide_control | 30 | 0.977021 | 0.477544 | 2.03078 | 1 | 1 |
| mu2_coboundary | 64 | random_branch_control | 30 | 0.5 | 0.806415 | 2 | 2 | 2 |
| mu2_coboundary | 64 | supplied_context_q2_branch_predictor | 30 | 1 | 0.407374 | 2 | 2 | 2 |
| mu2_coboundary | 64 | validation_face_table_router | 30 | 1 | 0.407374 | 2 | 2 | 2 |
| mu2_coboundary | 64 | validation_global_branch_selector | 30 | 1 | 0.407374 | 2 | 2 | 2 |
| mu2_coboundary | 64 | wrong_context_control | 30 | 1 | 0.407374 | 2 | 2 | 2 |
| mu2_coboundary | 64 | wrong_twist_control | 30 | 0 | 1.2042 | 2 | 2 | 2 |
| mu2_nontrivial_h2 | 32 | c2m3_synchronized | 30 | 0.75 | 0.607019 | 1 | 1 | 1 |
| mu2_nontrivial_h2 | 32 | distilled_single_model_control | 30 | 0.75 | 0.623908 | 1 | 1 | 1 |
| mu2_nontrivial_h2 | 32 | ensemble_reference | 30 | 0.75 | 0.607019 | 4 | 4 | 4 |
| mu2_nontrivial_h2 | 32 | git_rebasin_pairwise | 30 | 0.75 | 0.607019 | 1 | 1 | 1 |
| mu2_nontrivial_h2 | 32 | no_twist_branch_control | 30 | 0.75 | 0.607019 | 2 | 2 | 2 |
| mu2_nontrivial_h2 | 32 | ordinary_weight_average | 30 | 0.500096 | 0.693147 | 1 | 1 | 1 |
| mu2_nontrivial_h2 | 32 | parameter_matched_wide_control | 30 | 0.739021 | 0.613006 | 2.06066 | 1 | 1 |
| mu2_nontrivial_h2 | 32 | random_branch_control | 30 | 0.45 | 0.84749 | 2 | 2 | 2 |
| mu2_nontrivial_h2 | 32 | supplied_context_q2_branch_predictor | 30 | 1 | 0.406317 | 2 | 2 | 2 |
| mu2_nontrivial_h2 | 32 | validation_face_table_router | 30 | 1 | 0.406317 | 2 | 2 | 2 |
| mu2_nontrivial_h2 | 32 | validation_global_branch_selector | 30 | 0.75 | 0.607019 | 2 | 2 | 2 |
| mu2_nontrivial_h2 | 32 | wrong_context_control | 30 | 0.5 | 0.806603 | 2 | 2 | 2 |
| mu2_nontrivial_h2 | 32 | wrong_twist_control | 30 | 0.5 | 0.80716 | 2 | 2 | 2 |
| mu2_nontrivial_h2 | 64 | c2m3_synchronized | 30 | 0.75 | 0.60757 | 1 | 1 | 1 |
| mu2_nontrivial_h2 | 64 | distilled_single_model_control | 30 | 0.75 | 0.623992 | 1 | 1 | 1 |
| mu2_nontrivial_h2 | 64 | ensemble_reference | 30 | 0.75 | 0.60757 | 4 | 4 | 4 |
| mu2_nontrivial_h2 | 64 | git_rebasin_pairwise | 30 | 0.75 | 0.60757 | 1 | 1 | 1 |
| mu2_nontrivial_h2 | 64 | no_twist_branch_control | 30 | 0.75 | 0.60757 | 2 | 2 | 2 |
| mu2_nontrivial_h2 | 64 | ordinary_weight_average | 30 | 0.499612 | 0.693147 | 1 | 1 | 1 |
| mu2_nontrivial_h2 | 64 | parameter_matched_wide_control | 30 | 0.740354 | 0.613541 | 2.03078 | 1 | 1 |
| mu2_nontrivial_h2 | 64 | random_branch_control | 30 | 0.5 | 0.807164 | 2 | 2 | 2 |
| mu2_nontrivial_h2 | 64 | supplied_context_q2_branch_predictor | 30 | 1 | 0.407573 | 2 | 2 | 2 |
| mu2_nontrivial_h2 | 64 | validation_face_table_router | 30 | 1 | 0.407573 | 2 | 2 | 2 |
| mu2_nontrivial_h2 | 64 | validation_global_branch_selector | 30 | 0.75 | 0.60757 | 2 | 2 | 2 |
| mu2_nontrivial_h2 | 64 | wrong_context_control | 30 | 0.5 | 0.806095 | 2 | 2 | 2 |
| mu2_nontrivial_h2 | 64 | wrong_twist_control | 30 | 0.5 | 0.807383 | 2 | 2 | 2 |
| random_noncentral | 32 | c2m3_synchronized | 30 | 1 | 0.406159 | 1 | 1 | 1 |
| random_noncentral | 32 | distilled_single_model_control | 30 | 1 | 0.406159 | 1 | 1 | 1 |
| random_noncentral | 32 | ensemble_reference | 30 | 1 | 0.406159 | 4 | 4 | 4 |
| random_noncentral | 32 | git_rebasin_pairwise | 30 | 1 | 0.406159 | 1 | 1 | 1 |
| random_noncentral | 32 | no_twist_branch_control | 30 | 1 | 0.406159 | 2 | 2 | 2 |
| random_noncentral | 32 | ordinary_weight_average | 30 | 0.500179 | 0.693147 | 1 | 1 | 1 |
| random_noncentral | 32 | parameter_matched_wide_control | 30 | 0.970879 | 0.475532 | 2.06066 | 1 | 1 |
| random_noncentral | 32 | random_branch_control | 30 | 0.4 | 0.887519 | 2 | 2 | 2 |
| random_noncentral | 32 | supplied_context_q2_branch_predictor | 30 | 1 | 0.406159 | 2 | 2 | 2 |
| random_noncentral | 32 | validation_face_table_router | 30 | 1 | 0.406159 | 2 | 2 | 2 |
| random_noncentral | 32 | validation_global_branch_selector | 30 | 1 | 0.406159 | 2 | 2 | 2 |
| random_noncentral | 32 | wrong_context_control | 30 | 1 | 0.406159 | 2 | 2 | 2 |
| random_noncentral | 32 | wrong_twist_control | 30 | 0 | 1.2066 | 2 | 2 | 2 |
| random_noncentral | 64 | c2m3_synchronized | 30 | 1 | 0.406889 | 1 | 1 | 1 |
| random_noncentral | 64 | distilled_single_model_control | 30 | 1 | 0.406889 | 1 | 1 | 1 |
| random_noncentral | 64 | ensemble_reference | 30 | 1 | 0.406889 | 4 | 4 | 4 |
| random_noncentral | 64 | git_rebasin_pairwise | 30 | 1 | 0.406889 | 1 | 1 | 1 |
| random_noncentral | 64 | no_twist_branch_control | 30 | 1 | 0.406889 | 2 | 2 | 2 |
| random_noncentral | 64 | ordinary_weight_average | 30 | 0.498992 | 0.693147 | 1 | 1 | 1 |
| random_noncentral | 64 | parameter_matched_wide_control | 30 | 0.972725 | 0.476374 | 2.03078 | 1 | 1 |
| random_noncentral | 64 | random_branch_control | 30 | 0.5 | 0.804579 | 2 | 2 | 2 |
| random_noncentral | 64 | supplied_context_q2_branch_predictor | 30 | 1 | 0.406889 | 2 | 2 | 2 |
| random_noncentral | 64 | validation_face_table_router | 30 | 1 | 0.406889 | 2 | 2 | 2 |
| random_noncentral | 64 | validation_global_branch_selector | 30 | 1 | 0.406889 | 2 | 2 | 2 |
| random_noncentral | 64 | wrong_context_control | 30 | 1 | 0.406889 | 2 | 2 | 2 |
| random_noncentral | 64 | wrong_twist_control | 30 | 0 | 1.20519 | 2 | 2 | 2 |

Structural fields include exact local functional equivalence, pairwise residual, centrality residual, exact coboundary flag, and negative-face rate. The random-noncentral family is a negative control and is not promoted as central evidence.

## Controlled finite-Heisenberg period-index benchmark

| case_id | d | k | scalar_commutator_order | period | certified_representation_threshold | minimal_successful_rank | matrix_relation_residual | direct_sum_multiple_realized |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| d2_k1 | 2 | 1 | 2 | 2 | 2 | 2 | 1.73191e-16 | True |
| d2_k2 | 2 | 2 | 2 | 2 | 4 | 4 | 1.73191e-16 | True |
| d2_k3 | 2 | 3 | 2 | 2 | 8 | 8 | 1.73191e-16 | True |
| d3_k1 | 3 | 1 | 3 | 3 | 3 | 3 | 3.04047e-16 | True |
| d3_k2 | 3 | 2 | 3 | 3 | 9 | 9 | 3.04047e-16 | True |
| d4_k1 | 4 | 1 | 4 | 4 | 4 | 4 | 1.22465e-16 | True |
| d4_k2 | 4 | 2 | 4 | 4 | 16 | 16 | 1.22465e-16 | True |

For every case, the scalar commutator order is `d`, the nondegenerate `k`-pair matrix relations are checked, the representation-theoretic threshold is explicitly `d^k`, ranks below or not divisible by that threshold fail, and direct sums realize multiples. The empirical rank sweep is not identified with a classical index without this checked theorem.

## Negative boundaries

- The supplied-context q=2 result is not a learned practical router.
- The validation face table is not a generalizing router.
- The central construction does not show that natural MNIST/CIFAR residuals are Brauer classes.
