# Sequential Quotient Lift Report

## Correction Notice

The controlled accuracy tables from commit `9e743a0fd2cefced2c155e47e64466c23c4c9128` are invalid as empirical evidence.  The old script used method-dependent label injection (`signal_for`, `logits_from_signal`, and `metric_row`) to prescribe accuracies from labels.  The corrected run removes those functions, recomputes quotient-chain residuals, uses resampled bootstrap recovery, and sets `lift_implemented=True` only for branch tensors actually constructed and evaluated.

## Exact Command

```bash
.venv/bin/python experiments/sequential_quotient_lift_benchmark.py --mode smoke --groups C2xC2,C4,D4,S3 --seeds 0:2 --noise-levels 0.0,0.25 --bootstrap-samples 200 --max-train-samples 512 --max-test-samples 384 --n-val 128 --width 32 --epochs 1 --batch-size 128 --device cpu
```

## Commit And Environment

- Commit: `9e743a0`
- Dirty status before writing artifacts: `M experiments/sequential_quotient_lift_benchmark.py
 M reports/plots/sequential_quotient_accuracy_by_depth.pdf
 M reports/plots/sequential_quotient_delta_vs_controls.pdf
 M reports/plots/sequential_quotient_residual_by_depth.pdf
 M src/sequential_quotient_lift.py
 M tests/test_sequential_quotient_lift.py`
- Dataset: `mnist`
- Architecture: `mlp`
- Width: `32`
- Seeds: `0:2`
- Noise levels: `0.0,0.25`

## Evidence Decision

D. The genuine consecutive lift could not be implemented or evaluated completely; list exact blockers.

## Smoke Gates

| gate | passed | reason |
| --- | --- | --- |
| actual_model_logits | True | all candidate logits came from executed probe models and branch tensors |
| label_leakage_regression | True | permuting labels after branch-logit production did not change logits |
| expected_chains_recovered | True | controlled quotient signatures match preregistered C2/C3 chains |
| coset_actions_verified | True | coset action multiplication residual is zero where stages exist |
| correct_lift_beats_wrong_or_random_controls | False | failed honestly: exact gauge-copy branches tie controls, so this is only a functional sanity check |

## Probe Model Quality

| seed | dataset | architecture | width | train_accuracy_last_loader | validation_accuracy | test_accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | mnist | mlp | 32 | 0.3125 | 0.359375 | 0.347656 |
| 1 | mnist | mlp | 32 | 0.285156 | 0.296875 | 0.269531 |
| 2 | mnist | mlp | 32 | 0.457031 | 0.359375 | 0.347656 |

## Controlled Group Diagnostics

| group_name | noise_level | expected_chain_signature | observed_chain_signature | chain_matches_expected | stage_depth | quotient_order | kernel_order | residual_group_order | coset_action_law_residual | final_regular_representation_verified | bootstrap_stability |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C2xC2 | 0 | C2->C2 | C2->C2 | True | 1 | 2 | 2 | 2 | 0 | False | 0.835 |
| C2xC2 | 0 | C2->C2 | C2->C2 | True | 2 | 2 | 1 | 1 | 0 | True | 0.835 |
| C2xC2 | 0.25 | C2->C2 | C2->C2 | True | 1 | 2 | 2 | 2 | 0 | False | 0.49 |
| C2xC2 | 0.25 | C2->C2 | C2->C2 | True | 2 | 2 | 1 | 1 | 0 | True | 0.49 |
| C4 | 0 | C2->C2 | C2->C2 | True | 1 | 2 | 2 | 2 | 0 | False | 0.925 |
| C4 | 0 | C2->C2 | C2->C2 | True | 2 | 2 | 1 | 1 | 0 | True | 0.925 |
| C4 | 0.25 | C2->C2 | C2->C2 | True | 1 | 2 | 2 | 2 | 0 | False | 0.94 |
| C4 | 0.25 | C2->C2 | C2->C2 | True | 2 | 2 | 1 | 1 | 0 | True | 0.94 |
| D4 | 0 | C2->C2->C2 | C2->C2->C2 | True | 1 | 2 | 4 | 4 | 0 | False | 0.985 |
| D4 | 0 | C2->C2->C2 | C2->C2->C2 | True | 2 | 2 | 2 | 2 | 0 | False | 0.985 |
| D4 | 0 | C2->C2->C2 | C2->C2->C2 | True | 3 | 2 | 1 | 1 | 0 | True | 0.985 |
| D4 | 0.25 | C2->C2->C2 | C2->C2->C2 | True | 1 | 2 | 4 | 4 | 0 | False | 0.985 |
| D4 | 0.25 | C2->C2->C2 | C2->C2->C2 | True | 2 | 2 | 2 | 2 | 0 | False | 0.985 |
| D4 | 0.25 | C2->C2->C2 | C2->C2->C2 | True | 3 | 2 | 1 | 1 | 0 | True | 0.985 |
| S3 | 0 | C2->C3 | C2->C3 | True | 1 | 2 | 3 | 3 | 0 | False | 0.97 |
| S3 | 0 | C2->C3 | C2->C3 | True | 2 | 3 | 1 | 1 | 0 | True | 0.97 |
| S3 | 0.25 | C2->C3 | C2->C3 | True | 1 | 2 | 3 | 3 | 0 | False | 0.95 |
| S3 | 0.25 | C2->C3 | C2->C3 | True | 2 | 3 | 1 | 1 | 0 | True | 0.95 |

## Controlled Accuracy Summary

| group_name | method | n | mean_validation_accuracy | mean_test_accuracy | max_functional_preservation_error |
| --- | --- | --- | --- | --- | --- |
| C2xC2 | base_model | 6 | 0.338542 | 0.321615 | 0 |
| C2xC2 | c2_fourier_plus_minus | 6 | 0.338542 | 0.321615 | 2.98023e-08 |
| C2xC2 | one_shot_regular_lift | 6 | 0.338542 | 0.321615 | 2.98023e-08 |
| C2xC2 | random_same_branch_count_control | 6 | 0.338542 | 0.321615 | 2.98023e-08 |
| C2xC2 | sequential_quotient_lift_validation_router | 6 | 0.338542 | 0.321615 | 2.98023e-08 |
| C2xC2 | uniform_pool | 6 | 0.338542 | 0.321615 | 2.98023e-08 |
| C2xC2 | wrong_quotient_control | 6 | 0.338542 | 0.321615 | 2.98023e-08 |
| C4 | base_model | 6 | 0.338542 | 0.321615 | 0 |
| C4 | c2_fourier_plus_minus | 6 | 0.338542 | 0.321615 | 2.98023e-08 |
| C4 | one_shot_regular_lift | 6 | 0.338542 | 0.321615 | 2.98023e-08 |
| C4 | random_same_branch_count_control | 6 | 0.338542 | 0.321615 | 2.98023e-08 |
| C4 | sequential_quotient_lift_validation_router | 6 | 0.338542 | 0.321615 | 2.98023e-08 |
| C4 | uniform_pool | 6 | 0.338542 | 0.321615 | 2.98023e-08 |
| C4 | wrong_quotient_control | 6 | 0.338542 | 0.321615 | 2.98023e-08 |
| D4 | base_model | 6 | 0.338542 | 0.321615 | 0 |
| D4 | c2_fourier_plus_minus | 6 | 0.338542 | 0.321615 | 2.98023e-08 |
| D4 | one_shot_regular_lift | 6 | 0.338542 | 0.321615 | 2.98023e-08 |
| D4 | random_same_branch_count_control | 6 | 0.338542 | 0.321615 | 2.98023e-08 |
| D4 | sequential_quotient_lift_validation_router | 6 | 0.338542 | 0.321615 | 2.98023e-08 |
| D4 | uniform_pool | 6 | 0.338542 | 0.321615 | 2.98023e-08 |
| D4 | wrong_quotient_control | 6 | 0.338542 | 0.321615 | 2.98023e-08 |
| S3 | base_model | 6 | 0.338542 | 0.321615 | 0 |
| S3 | c2_fourier_plus_minus | 6 | 0.338542 | 0.321615 | 2.98023e-08 |
| S3 | one_shot_regular_lift | 6 | 0.338542 | 0.321615 | 2.98023e-08 |
| S3 | random_same_branch_count_control | 6 | 0.338542 | 0.321615 | 2.98023e-08 |
| S3 | sequential_quotient_lift_validation_router | 6 | 0.338542 | 0.321615 | 2.98023e-08 |
| S3 | uniform_pool | 6 | 0.338542 | 0.321615 | 2.98023e-08 |
| S3 | wrong_quotient_control | 6 | 0.338542 | 0.321615 | 2.98023e-08 |

## Paired Stats

| group_name | noise_level | comparison | n_paired_seeds | mean_delta | ci_low | ci_high | wins | ties | losses | claim_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C2xC2 | 0 | sequential_quotient_lift_validation_router - base_model | 3 | 0 | 0 | 0 | 0 | 3 | 0 | unsupported_smoke |
| C2xC2 | 0 | sequential_quotient_lift_validation_router - uniform_pool | 3 | 0 | 0 | 0 | 0 | 3 | 0 | unsupported_smoke |
| C2xC2 | 0 | sequential_quotient_lift_validation_router - random_same_branch_count_control | 3 | 0 | 0 | 0 | 0 | 3 | 0 | unsupported_smoke |
| C2xC2 | 0 | sequential_quotient_lift_validation_router - wrong_quotient_control | 3 | 0 | 0 | 0 | 0 | 3 | 0 | unsupported_smoke |
| C2xC2 | 0 | sequential_quotient_lift_validation_router - one_shot_regular_lift | 3 | 0 | 0 | 0 | 0 | 3 | 0 | unsupported_smoke |
| C2xC2 | 0.25 | sequential_quotient_lift_validation_router - base_model | 3 | 0 | 0 | 0 | 0 | 3 | 0 | unsupported_smoke |
| C2xC2 | 0.25 | sequential_quotient_lift_validation_router - uniform_pool | 3 | 0 | 0 | 0 | 0 | 3 | 0 | unsupported_smoke |
| C2xC2 | 0.25 | sequential_quotient_lift_validation_router - random_same_branch_count_control | 3 | 0 | 0 | 0 | 0 | 3 | 0 | unsupported_smoke |
| C2xC2 | 0.25 | sequential_quotient_lift_validation_router - wrong_quotient_control | 3 | 0 | 0 | 0 | 0 | 3 | 0 | unsupported_smoke |
| C2xC2 | 0.25 | sequential_quotient_lift_validation_router - one_shot_regular_lift | 3 | 0 | 0 | 0 | 0 | 3 | 0 | unsupported_smoke |
| C4 | 0 | sequential_quotient_lift_validation_router - base_model | 3 | 0 | 0 | 0 | 0 | 3 | 0 | unsupported_smoke |
| C4 | 0 | sequential_quotient_lift_validation_router - uniform_pool | 3 | 0 | 0 | 0 | 0 | 3 | 0 | unsupported_smoke |
| C4 | 0 | sequential_quotient_lift_validation_router - random_same_branch_count_control | 3 | 0 | 0 | 0 | 0 | 3 | 0 | unsupported_smoke |
| C4 | 0 | sequential_quotient_lift_validation_router - wrong_quotient_control | 3 | 0 | 0 | 0 | 0 | 3 | 0 | unsupported_smoke |
| C4 | 0 | sequential_quotient_lift_validation_router - one_shot_regular_lift | 3 | 0 | 0 | 0 | 0 | 3 | 0 | unsupported_smoke |
| C4 | 0.25 | sequential_quotient_lift_validation_router - base_model | 3 | 0 | 0 | 0 | 0 | 3 | 0 | unsupported_smoke |
| C4 | 0.25 | sequential_quotient_lift_validation_router - uniform_pool | 3 | 0 | 0 | 0 | 0 | 3 | 0 | unsupported_smoke |
| C4 | 0.25 | sequential_quotient_lift_validation_router - random_same_branch_count_control | 3 | 0 | 0 | 0 | 0 | 3 | 0 | unsupported_smoke |
| C4 | 0.25 | sequential_quotient_lift_validation_router - wrong_quotient_control | 3 | 0 | 0 | 0 | 0 | 3 | 0 | unsupported_smoke |
| C4 | 0.25 | sequential_quotient_lift_validation_router - one_shot_regular_lift | 3 | 0 | 0 | 0 | 0 | 3 | 0 | unsupported_smoke |
| D4 | 0 | sequential_quotient_lift_validation_router - base_model | 3 | 0 | 0 | 0 | 0 | 3 | 0 | unsupported_smoke |
| D4 | 0 | sequential_quotient_lift_validation_router - uniform_pool | 3 | 0 | 0 | 0 | 0 | 3 | 0 | unsupported_smoke |
| D4 | 0 | sequential_quotient_lift_validation_router - random_same_branch_count_control | 3 | 0 | 0 | 0 | 0 | 3 | 0 | unsupported_smoke |
| D4 | 0 | sequential_quotient_lift_validation_router - wrong_quotient_control | 3 | 0 | 0 | 0 | 0 | 3 | 0 | unsupported_smoke |
| D4 | 0 | sequential_quotient_lift_validation_router - one_shot_regular_lift | 3 | 0 | 0 | 0 | 0 | 3 | 0 | unsupported_smoke |
| D4 | 0.25 | sequential_quotient_lift_validation_router - base_model | 3 | 0 | 0 | 0 | 0 | 3 | 0 | unsupported_smoke |
| D4 | 0.25 | sequential_quotient_lift_validation_router - uniform_pool | 3 | 0 | 0 | 0 | 0 | 3 | 0 | unsupported_smoke |
| D4 | 0.25 | sequential_quotient_lift_validation_router - random_same_branch_count_control | 3 | 0 | 0 | 0 | 0 | 3 | 0 | unsupported_smoke |
| D4 | 0.25 | sequential_quotient_lift_validation_router - wrong_quotient_control | 3 | 0 | 0 | 0 | 0 | 3 | 0 | unsupported_smoke |
| D4 | 0.25 | sequential_quotient_lift_validation_router - one_shot_regular_lift | 3 | 0 | 0 | 0 | 0 | 3 | 0 | unsupported_smoke |
| S3 | 0 | sequential_quotient_lift_validation_router - base_model | 3 | 0 | 0 | 0 | 0 | 3 | 0 | unsupported_smoke |
| S3 | 0 | sequential_quotient_lift_validation_router - uniform_pool | 3 | 0 | 0 | 0 | 0 | 3 | 0 | unsupported_smoke |
| S3 | 0 | sequential_quotient_lift_validation_router - random_same_branch_count_control | 3 | 0 | 0 | 0 | 0 | 3 | 0 | unsupported_smoke |
| S3 | 0 | sequential_quotient_lift_validation_router - wrong_quotient_control | 3 | 0 | 0 | 0 | 0 | 3 | 0 | unsupported_smoke |
| S3 | 0 | sequential_quotient_lift_validation_router - one_shot_regular_lift | 3 | 0 | 0 | 0 | 0 | 3 | 0 | unsupported_smoke |
| S3 | 0.25 | sequential_quotient_lift_validation_router - base_model | 3 | 0 | 0 | 0 | 0 | 3 | 0 | unsupported_smoke |
| S3 | 0.25 | sequential_quotient_lift_validation_router - uniform_pool | 3 | 0 | 0 | 0 | 0 | 3 | 0 | unsupported_smoke |
| S3 | 0.25 | sequential_quotient_lift_validation_router - random_same_branch_count_control | 3 | 0 | 0 | 0 | 0 | 3 | 0 | unsupported_smoke |
| S3 | 0.25 | sequential_quotient_lift_validation_router - wrong_quotient_control | 3 | 0 | 0 | 0 | 0 | 3 | 0 | unsupported_smoke |
| S3 | 0.25 | sequential_quotient_lift_validation_router - one_shot_regular_lift | 3 | 0 | 0 | 0 | 0 | 3 | 0 | unsupported_smoke |

## Natural MNIST Status

| source | method | lift_implemented | prediction_level_lift | claim_boundary |
| --- | --- | --- | --- | --- |
| natural_mnist | sequential_quotient_lift_validation_router | False | False | natural_skipped_until_controlled_smoke_gates_pass: correct_lift_beats_wrong_or_random_controls |

## What This Proves

- The finite-group quotient-chain code now certifies C2/C3 stages by homomorphism checks, not order heuristics.
- The coset action on `Gamma/K_j` is built and checked at every exact stage.
- Exact hidden-unit permutation gauge copies preserve executed MLP logits up to numerical tolerance.
- Labels do not affect branch logits before validation selection.

## What This Does Not Prove

- It does not prove a destructive controlled merge is repaired by the sequential lift.
- It does not prove natural MNIST quotient-routed prediction tensors work.
- It does not prove a parameter-level sequential lift.
- It does not justify Brauer/H2 language for real neural residuals.
- It does not rely on the older controlled nonabelian benchmark as empirical support; that code path still contains prescribed target-accuracy/synthetic-teacher logic and remains a separate artifact boundary.

## Required Questions

1. Are old commit `9e743a0fd2cefced2c155e47e64466c23c4c9128` controlled accuracy tables valid?  No.
2. Were `signal_for` and `logits_from_signal` removed from the corrected empirical pipeline?  Yes.
3. Are quotient chains certified from exact homomorphisms instead of element-order heuristics?  Yes.
4. Are coset action permutation representations constructed and checked?  Yes.
5. Does truncated sign-character handling avoid recursive fake kernels?  Yes.
6. Is bootstrap stability now resampled rather than fixed at one?  Yes.
7. Do label-leakage regression tests pass?  Yes in the focused test run recorded by this task.
8. Are branch tensors built from actual executed model logits?  Yes for the exact-gauge smoke rows.
9. Does the corrected smoke show quotient lifting beats wrong/random controls?  No; the exact gauge-copy branches tie controls.
10. Was natural MNIST attempted after the failed smoke gate?  No; it is explicitly skipped.

## Blockers

- Level-2 destructive planted holonomy merging with actual overlap maps is not implemented in this corrected run.
- Correct lift versus wrong/random controls is not positive on exact gauge-copy branches.
- Full controlled 30-seed runs and natural N=6/N=8 MNIST are gated behind a passing destructive smoke.

Final decision: D. The genuine consecutive lift could not be implemented or evaluated completely; list exact blockers.
