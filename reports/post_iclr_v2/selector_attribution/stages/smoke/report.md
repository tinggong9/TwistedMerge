# Post-ICLR selector-attribution report

## Verdict

**TwistedMerge-specific selector gain supported.**

The confirmatory unit is the independent training-group seed. Settings sharing a seed across model count and width are averaged before the paired bootstrap. The study contains `1` independent groups and `1` exact checkpoint settings. Failed official candidates: `0`.

## Frozen protocol

- Stage: `smoke`.
- Seeds: `9100`; model counts: `3`; widths: `32`.
- Recipe: Adam, learning rate `0.001`, `1` epochs, `512` sampled MNIST training examples, validation fraction `0.2`.
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
| A0 | 1 | 1 | 0.1445 | 0.1445 | NA | 2.2173 | 0.0000 | 0.1816 | 0.1445 | 0.0000 | 0.0000 |
| A1 | 1 | 1 | 0.3262 | 0.3262 | NA | 2.2422 | -0.1816 | 0.0000 | 0.3262 | 0.0000 | 1.0000 |
| A2 | 1 | 1 | 0.2676 | 0.2676 | NA | 2.2433 | -0.1230 | 0.0586 | 0.2676 | 1.0000 | 0.0000 |
| A3 | 1 | 1 | 0.2598 | 0.2598 | NA | 2.2212 | -0.1152 | 0.0664 | 0.2598 | 1.0000 | 0.0000 |
| A4 | 1 | 1 | 0.1445 | 0.1445 | NA | 2.2173 | 0.0000 | 0.1816 | 0.1445 | 0.0000 | 0.0000 |
| A5 | 1 | 1 | 0.3262 | 0.3262 | NA | 2.2422 | -0.1816 | 0.0000 | 0.3262 | 0.0000 | 1.0000 |
| B0 | 1 | 1 | 0.2363 | 0.2363 | NA | 2.2625 | -0.0918 | 0.0898 | 0.2363 | 0.0000 | 0.0000 |
| A6 | 1 | 1 | 0.3262 | 0.3262 | NA | 2.2422 | -0.1816 | 0.0000 | 0.3262 | 0.0000 | 1.0000 |

## Paired attribution

| comparison | n_exact_settings | n_training_groups | paired_mean_accuracy_delta | paired_accuracy_delta_ci_low | paired_accuracy_delta_ci_high | wins | ties | losses | paired_effect_size_dz |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A5_minus_A0 | 1 | 1 | 0.1816 | 0.1816 | 0.1816 | 1 | 0 | 0 | NA |
| A5_minus_A1 | 1 | 1 | 0.0000 | 0.0000 | 0.0000 | 0 | 1 | 0 | NA |
| A5_minus_A2 | 1 | 1 | 0.0586 | 0.0586 | 0.0586 | 1 | 0 | 0 | NA |
| A5_minus_A3 | 1 | 1 | 0.0664 | 0.0664 | 0.0664 | 1 | 0 | 0 | NA |
| A5_minus_A4 | 1 | 1 | 0.1816 | 0.1816 | 0.1816 | 1 | 0 | 0 | NA |
| A5_minus_B0 | 1 | 1 | 0.0898 | 0.0898 | 0.0898 | 1 | 0 | 0 | NA |
| A4_minus_A1_same_pool_diagnostic_rule | 1 | 1 | -0.1816 | -0.1816 | -0.1816 | 0 | 0 | 1 | NA |

The primary controlled comparison A5 - B0 is `0.0898` with group-bootstrap 95% CI `[0.0898, 0.0898]`.

## A5 selections

| selected_family | selected_candidate | count | frequency | tm_specific | conditional_mean_gain_vs_a0 |
| --- | --- | --- | --- | --- | --- |
| official_synchronization | official_c2m3 | 1 | 1.0000 | False | 0.1816 |

TwistedMerge-specific selection frequency is `0.000`. Conditional rows are reported even when negative or absent.

## Preregistered success gates

```json
[
  {
    "ci_high": 0.08984375,
    "ci_low": 0.08984375,
    "criterion": "A5 beats B0 with positive group-bootstrap CI",
    "passed": true,
    "value": 0.08984375
  },
  {
    "conditional_gain": NaN,
    "criterion": "TM-specific choice has nontrivial frequency and positive conditional gain",
    "passed": false,
    "value": 0.0
  },
  {
    "ci_high": -0.181640625,
    "ci_low": -0.181640625,
    "criterion": "A4 residual rule reduces regret with same pool",
    "passed": false,
    "value": -0.181640625
  },
  {
    "a5_worst": 0.326171875,
    "b0_worst": 0.236328125,
    "criterion": "A5 improves worst setting without material mean loss",
    "passed": true
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
