# Executed Two-Loop Noncommuting Holonomy Report

Decision: **B. Structural noncommuting holonomy supported, but accuracy advantage unsupported.**

Every prediction in this report was produced by an executed NumPy one-hidden-layer ReLU MLP, an executed parameter soup, an executed branch tensor, or an executed ensemble. Candidate functions do not accept labels. Labels were generated once from the fixed planted teacher and used only after candidate logits existed.

## Exact command

```bash
/Users/tinggong/Documents/GitHub/TwistedMerge/.venv/bin/python experiments/executed_two_loop_holonomy.py --mode full
```

- Git commit at execution: `0a41f76d3c8a77acc3a47514c2639b81fbc5b280`
- Mode: `full`
- Groups: `S3,D4`
- Widths: `32,64`
- Seeds: `0:49`
- Validation/test sizes: `1000` / `2000`
- Saved-logit leakage artifact: `reports/next_benchmarks/logits/two_loop_S3_W32_seed0.npz`
- Label-permutation regression: `True`

## Construction

The comparison complex is a wedge of two length-three cycles, `0-1-2-0` and `0-3-4-0`. The first loop carries the planted reflection/transposition `s`; the second carries the planted rotation/3-cycle `r`. Five local checkpoints are exact hidden-unit reparameterizations of the same executed ReLU MLP. A duplicated regular hidden orbit supplies exact automorphisms carrying the two noncommuting transitions; other hidden units remain generic, so ordinary unaligned weight averaging is a genuine executed control.

## Smoke and full-run gates

| generators_noncommute | pooling_certificate_passed | group_action_certificate_passed | local_equivalence_passed | generators_recovered | wrong_controls_rejected_structurally | candidate_logits_executed | label_permutation_regression_passed | all_smoke_gates_passed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| True | True | True | True | True | True | True | True | True |

## Structural residuals

| group_name | pre_lift_residual | post_lift_residual | pooling_residual_gamma_1 | pooling_residual_gamma_2 | commutator_residual | group_action_multiplication_residual | local_functional_equivalence_residual |
| --- | --- | --- | --- | --- | --- | --- | --- |
| D4 | 0.603553 | 0 | 0 | 0 | 0.603553 | 0 | 5.61218e-16 |
| S3 | 0.522693 | 0 | 0 | 0 | 0.522693 | 0 | 5.40401e-16 |

`commutator_residual > 0` is the certificate that `rho(gamma_1) rho(gamma_2) != rho(gamma_2) rho(gamma_1)`. Both pooling residuals are required to vanish.

## Executed accuracy summary

| group_name | hidden_width | method | n_runs | mean_test_accuracy | mean_test_loss | parameter_multiplier | branch_count | inference_multiplier | model_kind |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D4 | 32 | branch_orbit_lift_with_invariant_pooling | 50 | 1 | 1.25715 | 1 | 4 | 4 | branch_model |
| D4 | 32 | branch_regular_lift_with_invariant_pooling | 50 | 1 | 1.25715 | 1 | 8 | 8 | branch_model |
| D4 | 32 | c2m3_strict_synchronization | 50 | 1 | 1.25715 | 1 | 1 | 1 | single_model |
| D4 | 32 | ensemble_reference | 50 | 1 | 1.25715 | 5 | 5 | 5 | ensemble |
| D4 | 32 | git_rebasin_pairwise | 50 | 1 | 1.25715 | 1 | 1 | 1 | single_model |
| D4 | 32 | greedy_soup | 50 | 1 | 1.25715 | 1 | 1 | 1 | soup |
| D4 | 32 | naive_regular_representation_without_invariant_pooling | 50 | 1 | 1.25715 | 1 | 8 | 8 | branch_model |
| D4 | 32 | oracle_supplied_context_branch_predictor | 50 | 1 | 1.25715 | 5 | 5 | 1 | branch_model |
| D4 | 32 | ordinary_weight_average | 50 | 0.57481 | 1.49335 | 1 | 1 | 1 | single_model |
| D4 | 32 | random_same_branch_count_control | 50 | 1 | 1.25715 | 1 | 8 | 8 | branch_model |
| D4 | 32 | validation_only_safe_selector | 50 | 1 | 1.25715 | 1 | 1 | 1 | single_model |
| D4 | 32 | wrong_generator_control | 50 | 1 | 1.25715 | 1 | 8 | 8 | branch_model |
| D4 | 32 | wrong_group_action_control | 50 | 1 | 1.25715 | 1 | 8 | 8 | branch_model |
| D4 | 32 | wrong_order_control | 50 | 1 | 1.25715 | 1 | 8 | 8 | branch_model |
| D4 | 64 | branch_orbit_lift_with_invariant_pooling | 50 | 1 | 1.23719 | 1 | 4 | 4 | branch_model |
| D4 | 64 | branch_regular_lift_with_invariant_pooling | 50 | 1 | 1.23719 | 1 | 8 | 8 | branch_model |
| D4 | 64 | c2m3_strict_synchronization | 50 | 1 | 1.23719 | 1 | 1 | 1 | single_model |
| D4 | 64 | ensemble_reference | 50 | 1 | 1.23719 | 5 | 5 | 5 | ensemble |
| D4 | 64 | git_rebasin_pairwise | 50 | 1 | 1.23719 | 1 | 1 | 1 | single_model |
| D4 | 64 | greedy_soup | 50 | 1 | 1.23719 | 1 | 1 | 1 | soup |
| D4 | 64 | naive_regular_representation_without_invariant_pooling | 50 | 1 | 1.23719 | 1 | 8 | 8 | branch_model |
| D4 | 64 | oracle_supplied_context_branch_predictor | 50 | 1 | 1.23719 | 5 | 5 | 1 | branch_model |
| D4 | 64 | ordinary_weight_average | 50 | 0.57874 | 1.50036 | 1 | 1 | 1 | single_model |
| D4 | 64 | random_same_branch_count_control | 50 | 1 | 1.23719 | 1 | 8 | 8 | branch_model |
| D4 | 64 | validation_only_safe_selector | 50 | 1 | 1.23719 | 1 | 1 | 1 | single_model |
| D4 | 64 | wrong_generator_control | 50 | 1 | 1.23719 | 1 | 8 | 8 | branch_model |
| D4 | 64 | wrong_group_action_control | 50 | 1 | 1.23719 | 1 | 8 | 8 | branch_model |
| D4 | 64 | wrong_order_control | 50 | 1 | 1.23719 | 1 | 8 | 8 | branch_model |
| S3 | 32 | branch_orbit_lift_with_invariant_pooling | 50 | 1 | 1.24784 | 1 | 3 | 3 | branch_model |
| S3 | 32 | branch_regular_lift_with_invariant_pooling | 50 | 1 | 1.24784 | 1 | 6 | 6 | branch_model |
| S3 | 32 | c2m3_strict_synchronization | 50 | 1 | 1.24784 | 1 | 1 | 1 | single_model |
| S3 | 32 | ensemble_reference | 50 | 1 | 1.24784 | 5 | 5 | 5 | ensemble |
| S3 | 32 | git_rebasin_pairwise | 50 | 1 | 1.24784 | 1 | 1 | 1 | single_model |
| S3 | 32 | greedy_soup | 50 | 1 | 1.24784 | 1 | 1 | 1 | soup |
| S3 | 32 | naive_regular_representation_without_invariant_pooling | 50 | 1 | 1.24784 | 1 | 6 | 6 | branch_model |
| S3 | 32 | oracle_supplied_context_branch_predictor | 50 | 1 | 1.24784 | 5 | 5 | 1 | branch_model |
| S3 | 32 | ordinary_weight_average | 50 | 0.55885 | 1.4996 | 1 | 1 | 1 | single_model |
| S3 | 32 | random_same_branch_count_control | 50 | 1 | 1.24784 | 1 | 6 | 6 | branch_model |
| S3 | 32 | validation_only_safe_selector | 50 | 1 | 1.24784 | 1 | 1 | 1 | single_model |
| S3 | 32 | wrong_generator_control | 50 | 1 | 1.24784 | 1 | 6 | 6 | branch_model |
| S3 | 32 | wrong_group_action_control | 50 | 1 | 1.24784 | 1 | 6 | 6 | branch_model |
| S3 | 32 | wrong_order_control | 50 | 1 | 1.24784 | 1 | 6 | 6 | branch_model |
| S3 | 64 | branch_orbit_lift_with_invariant_pooling | 50 | 1 | 1.2306 | 1 | 3 | 3 | branch_model |
| S3 | 64 | branch_regular_lift_with_invariant_pooling | 50 | 1 | 1.2306 | 1 | 6 | 6 | branch_model |
| S3 | 64 | c2m3_strict_synchronization | 50 | 1 | 1.2306 | 1 | 1 | 1 | single_model |
| S3 | 64 | ensemble_reference | 50 | 1 | 1.2306 | 5 | 5 | 5 | ensemble |
| S3 | 64 | git_rebasin_pairwise | 50 | 1 | 1.2306 | 1 | 1 | 1 | single_model |
| S3 | 64 | greedy_soup | 50 | 1 | 1.2306 | 1 | 1 | 1 | soup |
| S3 | 64 | naive_regular_representation_without_invariant_pooling | 50 | 1 | 1.2306 | 1 | 6 | 6 | branch_model |
| S3 | 64 | oracle_supplied_context_branch_predictor | 50 | 1 | 1.2306 | 5 | 5 | 1 | branch_model |
| S3 | 64 | ordinary_weight_average | 50 | 0.57461 | 1.49477 | 1 | 1 | 1 | single_model |
| S3 | 64 | random_same_branch_count_control | 50 | 1 | 1.2306 | 1 | 6 | 6 | branch_model |
| S3 | 64 | validation_only_safe_selector | 50 | 1 | 1.2306 | 1 | 1 | 1 | single_model |
| S3 | 64 | wrong_generator_control | 50 | 1 | 1.2306 | 1 | 6 | 6 | branch_model |
| S3 | 64 | wrong_group_action_control | 50 | 1 | 1.2306 | 1 | 6 | 6 | branch_model |
| S3 | 64 | wrong_order_control | 50 | 1 | 1.2306 | 1 | 6 | 6 | branch_model |

## Paired statistics

| group_name | comparison | n_pairs | paired_mean_accuracy_delta | ci_low | ci_high | wins | ties | losses |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D4 | branch_regular_lift_with_invariant_pooling_vs_git_rebasin_pairwise | 100 | 0 | 0 | 0 | 0 | 100 | 0 |
| D4 | branch_regular_lift_with_invariant_pooling_vs_c2m3_strict_synchronization | 100 | 0 | 0 | 0 | 0 | 100 | 0 |
| D4 | branch_regular_lift_with_invariant_pooling_vs_greedy_soup | 100 | 0 | 0 | 0 | 0 | 100 | 0 |
| D4 | branch_regular_lift_with_invariant_pooling_vs_random_same_branch_count_control | 100 | 0 | 0 | 0 | 0 | 100 | 0 |
| D4 | branch_regular_lift_with_invariant_pooling_vs_wrong_generator_control | 100 | 0 | 0 | 0 | 0 | 100 | 0 |
| D4 | branch_regular_lift_with_invariant_pooling_vs_wrong_order_control | 100 | 0 | 0 | 0 | 0 | 100 | 0 |
| D4 | branch_regular_lift_with_invariant_pooling_vs_wrong_group_action_control | 100 | 0 | 0 | 0 | 0 | 100 | 0 |
| S3 | branch_regular_lift_with_invariant_pooling_vs_git_rebasin_pairwise | 100 | 0 | 0 | 0 | 0 | 100 | 0 |
| S3 | branch_regular_lift_with_invariant_pooling_vs_c2m3_strict_synchronization | 100 | 0 | 0 | 0 | 0 | 100 | 0 |
| S3 | branch_regular_lift_with_invariant_pooling_vs_greedy_soup | 100 | 0 | 0 | 0 | 0 | 100 | 0 |
| S3 | branch_regular_lift_with_invariant_pooling_vs_random_same_branch_count_control | 100 | 0 | 0 | 0 | 0 | 100 | 0 |
| S3 | branch_regular_lift_with_invariant_pooling_vs_wrong_generator_control | 100 | 0 | 0 | 0 | 0 | 100 | 0 |
| S3 | branch_regular_lift_with_invariant_pooling_vs_wrong_order_control | 100 | 0 | 0 | 0 | 0 | 100 | 0 |
| S3 | branch_regular_lift_with_invariant_pooling_vs_wrong_group_action_control | 100 | 0 | 0 | 0 | 0 | 100 | 0 |

## Claim status

| claim_id | status | decision | safe_wording |
| --- | --- | --- | --- |
| executed_two_loop_noncommuting_holonomy | supported with limitations | B. Structural noncommuting holonomy supported, but accuracy advantage unsupported. | Executed S3/D4 models certify two noncommuting loop holonomies and invariant pooling; no lift accuracy advantage is claimed when controls tie. |

## Interpretation

The two noncommuting holonomies, exact local functional equivalence, regular-action multiplication, and invariant-pooling certificates are supported. The branch regular lift does not receive an accuracy-advantage claim unless it beats the random and wrong controls with a positive paired confidence interval. Ties are retained as a negative empirical outcome. The ensemble is called an `ensemble_reference`, never an upper bound.
