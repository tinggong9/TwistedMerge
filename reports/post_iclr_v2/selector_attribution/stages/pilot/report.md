# Post-ICLR selector-attribution report

## Verdict

**TwistedMerge-specific selector gain supported.**

The confirmatory unit is the independent training-group seed. Settings sharing a seed across model count and width are averaged before the paired bootstrap. The study contains `2` independent groups and `4` exact checkpoint settings. Failed official candidates: `0`.

## Frozen protocol

- Stage: `pilot`.
- Seeds: `9100,9101`; model counts: `3,4`; widths: `32`.
- Recipe: Adam, learning rate `0.001`, `2` epochs, `2000` sampled MNIST training examples, validation fraction `0.2`.
- Every selector sees the identical checkpoint group in a setting.
- A0--A5 and B0 are frozen from validation metrics before test evaluation. A6 alone uses test metrics and is an oracle upper bound.
- B0 exactly matches A5's candidate count and selector validation-evaluation count. Candidate-generation kernels and compute are not exactly equal; `budget_audit.csv` reports the difference.
- A4 uses the frozen residual threshold `0`: above threshold it falls back to A0; otherwise it chooses from A1.
- A5 includes no lift: the current claim ledger does not certify a natural-MNIST lift candidate.
- Git Re-Basin and C2M3 are adapter-assisted official cores, not unmodified end-to-end runs.

Smoke command:

```bash
/Users/tinggong/Documents/GitHub/TwistedMerge/.venv/bin/python experiments/post_iclr_selector_attribution.py --stage smoke --data-dir /Users/tinggong/Documents/GitHub/TwistedMerge/data --official-root /Users/tinggong/Documents/Codex/2026-07-17/files-mentioned-by-the-user-you/work/official-baseline-sources --jax-python /Users/tinggong/Documents/Codex/2026-07-17/files-mentioned-by-the-user-you/work/official-git-rebasin-py312-venv/bin/python
```

Confirmatory command:

```bash
/Users/tinggong/Documents/GitHub/TwistedMerge/.venv/bin/python experiments/post_iclr_selector_attribution.py --stage confirmatory --data-dir /Users/tinggong/Documents/GitHub/TwistedMerge/data --official-root /Users/tinggong/Documents/Codex/2026-07-17/files-mentioned-by-the-user-you/work/official-baseline-sources --jax-python /Users/tinggong/Documents/Codex/2026-07-17/files-mentioned-by-the-user-you/work/official-git-rebasin-py312-venv/bin/python
```

## Selector summary

| selector | n_exact_settings | n_training_groups | mean_test_accuracy | median_test_accuracy | std_test_accuracy | mean_test_loss | mean_regret_vs_greedy_soup | mean_regret_vs_oracle | worst_setting_accuracy | tm_specific_selection_frequency | validation_test_best_agreement |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A0 | 4 | 2 | 0.7147 | 0.7060 | 0.0197 | 1.6496 | 0.0000 | 0.0165 | 0.7030 | 0.0000 | 0.2500 |
| A1 | 4 | 2 | 0.7312 | 0.7310 | 0.0256 | 1.6507 | -0.0165 | 0.0000 | 0.7030 | 0.0000 | 0.7500 |
| A2 | 4 | 2 | 0.7147 | 0.7060 | 0.0197 | 1.6496 | 0.0000 | 0.0165 | 0.7030 | 0.0000 | 0.2500 |
| A3 | 4 | 2 | 0.7123 | 0.7045 | 0.0219 | 1.6919 | 0.0025 | 0.0190 | 0.6960 | 0.7500 | 0.0000 |
| A4 | 4 | 2 | 0.7147 | 0.7060 | 0.0197 | 1.6496 | 0.0000 | 0.0165 | 0.7030 | 0.0000 | 0.2500 |
| A5 | 4 | 2 | 0.7295 | 0.7310 | 0.0283 | 1.6602 | -0.0148 | 0.0018 | 0.6960 | 0.5000 | 0.5000 |
| B0 | 4 | 2 | 0.7147 | 0.7060 | 0.0197 | 1.6496 | 0.0000 | 0.0165 | 0.7030 | 0.0000 | 0.2500 |
| A6 | 4 | 2 | 0.7312 | 0.7310 | 0.0256 | 1.6507 | -0.0165 | 0.0000 | 0.7030 | 0.2500 | 1.0000 |

## Paired attribution

| comparison | n_exact_settings | n_training_groups | paired_mean_accuracy_delta | paired_accuracy_delta_ci_low | paired_accuracy_delta_ci_high | wins | ties | losses | paired_effect_size_dz |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A5_minus_A0 | 4 | 2 | 0.0148 | 0.0010 | 0.0285 | 2 | 1 | 1 | 0.5101 |
| A5_minus_A1 | 4 | 2 | -0.0018 | -0.0035 | 0.0000 | 0 | 3 | 1 | -0.5000 |
| A5_minus_A2 | 4 | 2 | 0.0148 | 0.0010 | 0.0285 | 2 | 1 | 1 | 0.5101 |
| A5_minus_A3 | 4 | 2 | 0.0173 | 0.0045 | 0.0300 | 2 | 2 | 0 | 0.5987 |
| A5_minus_A4 | 4 | 2 | 0.0148 | 0.0010 | 0.0285 | 2 | 1 | 1 | 0.5101 |
| A5_minus_B0 | 4 | 2 | 0.0148 | 0.0010 | 0.0285 | 2 | 1 | 1 | 0.5101 |
| A4_minus_A1_same_pool_diagnostic_rule | 4 | 2 | -0.0165 | -0.0285 | -0.0045 | 0 | 2 | 2 | NA |

The primary controlled comparison A5 - B0 is `0.0148` with group-bootstrap 95% CI `[0.0010, 0.0285]`.

## A5 selections

| selected_family | selected_candidate | count | frequency | tm_specific | conditional_mean_gain_vs_a0 |
| --- | --- | --- | --- | --- | --- |
| official_synchronization | official_c2m3 | 2 | 0.5000 | False | 0.0330 |
| permutation_gauge_soup | permutation_gauge_soup | 1 | 0.2500 | True | 0.0000 |
| union_gauge_soup | union_gauge_soup | 1 | 0.2500 | True | -0.0070 |

TwistedMerge-specific selection frequency is `0.500`. Conditional rows are reported even when negative or absent.

## Preregistered success gates

```json
[
  {
    "ci_high": 0.028500000000000025,
    "ci_low": 0.0010000000000000009,
    "criterion": "A5 beats B0 with positive group-bootstrap CI",
    "passed": true,
    "value": 0.014750000000000013
  },
  {
    "conditional_gain": -0.003500000000000003,
    "criterion": "TM-specific choice has nontrivial frequency and positive conditional gain",
    "passed": false,
    "value": 0.5
  },
  {
    "ci_high": -0.004500000000000004,
    "ci_low": -0.028500000000000025,
    "criterion": "A4 residual rule reduces regret with same pool",
    "passed": false,
    "value": -0.016500000000000015
  },
  {
    "a5_worst": 0.696,
    "b0_worst": 0.703,
    "criterion": "A5 improves worst setting without material mean loss",
    "passed": false
  }
]
```

## Budget and stability interpretation

Pool size and final selector evaluation count are exactly controlled by B0. Generation compute cannot be made identical because official synchronization, exact gauges, and greedy soup construction use different kernels; it is timed and counted rather than hidden. Validation-resampling stability is computed from per-example validation losses and correctness without touching test labels.

## Capacity and cost

Every A0--A6 candidate is a single, same-width MLP at inference. Soup candidates are materialized as one averaged model. There are no ensembles, wider models, branch lifts, or rank lifts. Per-candidate parameters, stored bytes, latency, merge time, training time, peak process memory, branches, and validation evaluations are in `resource_accounting.csv`.

## Reproducibility and negative results

`config.json`, `checkpoint_manifest.csv`, `failure_log.csv`, and `artifact_manifest.csv` record the recipe, split and dataset checksums, checkpoint provenance, external source commits, environment, commands, and output hashes. Official failures are never replaced by internal methods. Per-setting unfavorable results remain in `runs.csv` and `selectors.csv`.

![Selector accuracy](plots/selector_accuracy.png)

![A5 versus budget-matched ordinary control](plots/paired_delta_a5_vs_budget_matched.png)
