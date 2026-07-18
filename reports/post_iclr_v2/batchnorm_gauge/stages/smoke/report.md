# BatchNorm-aware channel-gauge exactness report

## Verdict

Compatible ResNet-18 channel permutations are **exact within the preregistered float32 numerical tolerance** in eval and train modes. Positive channel scaling is eval-exact only under explicitly frozen-statistic parameterizations (`affine` or `running_affine`). Scaling running mean and variance alone is not exact for nonzero BatchNorm epsilon. Arbitrary channelwise positive scaling is not claimed train-mode exact.

## Protocol

- Stage: `smoke`; seeds: `0`; epsilons: `0.001`.
- Random pretrained-free ResNet-18 parameter states are used because this is a functional identity test, not a performance benchmark.
- Each row covers `1` independent input batches of size `4`.
- Permutations cover Conv outputs, following Conv inputs, residual branches, projected shortcuts, BatchNorm affine parameters and buffers, and classifier inputs.
- Identity shortcuts enforce equal input/output bases.
- No parameters, branches, width, or inference operations are added.
- Failures: `0`.

## Primary exactness summary

| strategy | mode | epsilon | n_seeds | max_overall_logit_error | mean_absolute_logit_error | max_prediction_disagreement | all_exactness_checks_passed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| permutation | eval | 0.001 | 1 | 1.08406e-06 | 3.46638e-07 | 0 | True |
| permutation | train | 0.001 | 1 | 3.76105e-05 | 1.09062e-05 | 0 | True |
| running | eval | 0.001 | 1 | 0.0150818 | 0.00547526 | 0 | False |
| running | train | 0.001 | 1 | 0.0263632 | 0.0102188 | 0 | False |
| affine | eval | 0.001 | 1 | 1.2517e-06 | 3.41889e-07 | 0 | True |
| affine | train | 0.001 | 1 | 1.63175 | 0.479613 | 0.75 | False |
| running_affine | eval | 0.001 | 1 | 1.01328e-06 | 3.80259e-07 | 0 | True |
| running_affine | train | 0.001 | 1 | 0.215255 | 0.0815574 | 0.25 | False |
| no_batchnorm_control | eval | 0.001 | 1 | 5.96046e-08 | 5.75092e-09 | 0 | True |
| no_batchnorm_control | train | 0.001 | 1 | 5.96046e-08 | 5.75092e-09 | 0 | True |

## Parameterization comparisons

| comparison | n_pairs | n_seeds | paired_mean_max_error_delta | paired_delta_ci_low | paired_delta_ci_high | wins_lower_error | ties | losses_higher_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| running_minus_frozen | 1 | 1 | -0.591133 | -0.591133 | -0.591133 | 1 | 0 | 0 |
| affine_minus_frozen | 1 | 1 | -0.606214 | -0.606214 | -0.606214 | 1 | 0 | 0 |
| running_affine_minus_running | 1 | 1 | -0.0150808 | -0.0150808 | -0.0150808 | 1 | 0 | 0 |
| complete_recomputation_minus_post_merge_recalibration | 1 | 1 | -0.373833 | -0.373833 | -0.373833 | 1 | 0 | 0 |

## Interpretation

- `permutation`: exact graph-wide basis change, including residual addition and shortcut projections.
- `affine`: exact only in eval mode with original frozen running statistics; the stored statistics no longer describe the scaled Conv output.
- `running_affine`: exact only in eval mode after transforming running statistics and applying the epsilon-aware affine correction.
- `running`: approximate because `sqrt(s^2 v + epsilon)` differs from `s sqrt(v + epsilon)`.
- `post_merge_recalibration` and `complete_recomputation`: approximate unless followed by the frozen-statistic epsilon-aware affine correction.
- `no_batchnorm_control`: exact positive ReLU gauge in eval and train modes.

Train-mode errors for the static scaling parameterizations are negative evidence and remain in `exactness.csv`. Per-location canonicalized activation errors before and after residual blocks are in `activations.csv`. Deep train-mode activation comparisons accumulate float32 reduction-order differences (maximum `0.00028038`); the preregistered exactness decision is logit-level, whose train-mode maximum remains below `2e-4` with zero prediction disagreement.

## Commands

Smoke:

```bash
/Users/tinggong/Documents/GitHub/TwistedMerge/.venv/bin/python experiments/batchnorm_gauge_exactness.py --stage smoke
```

Confirmatory:

```bash
/Users/tinggong/Documents/GitHub/TwistedMerge/.venv/bin/python experiments/batchnorm_gauge_exactness.py --stage confirmatory
```

![Scaling exactness](plots/batchnorm_scaling_exactness.png)

![Permutation exactness](plots/permutation_exactness.png)
