# Obstruction/noise bound calibration

## Exact command

`PYTHONPYCACHEPREFIX=/private/tmp/codex-pycache MPLCONFIGDIR=/private/tmp/codex-mpl /Users/tinggong/Documents/GitHub/TwistedMerge/.venv/bin/python experiments/obstruction_noise_bound_calibration.py --bootstrap-samples 1000`

## Environment

- Commit: `a5bf2fa`
- Python: `3.12.13 (main, Mar  3 2026, 15:35:03) [Clang 21.1.4 ]`
- Platform: `macOS-26.0.1-arm64-arm-64bit`
- Packages: `{'numpy': '2.5.0', 'pandas': '3.0.4', 'matplotlib': '3.11.0', 'torch': '2.12.1', 'torchvision': '0.27.1'}`

## Outputs

- Calibration rows: `/Users/tinggong/Documents/GitHub/TwistedMerge/reports/csv/obstruction_noise_bound_calibration.csv`
- Summary rows: `/Users/tinggong/Documents/GitHub/TwistedMerge/reports/csv/obstruction_noise_bound_summary.csv`
- Plot: `/Users/tinggong/Documents/GitHub/TwistedMerge/reports/plots/def_observed_vs_noise_floor.pdf`
- Config: `/Users/tinggong/Documents/GitHub/TwistedMerge/reports/configs/obstruction_noise_bound_calibration_config.json`

## Calibration definitions

- `Def(c)` is source-specific and explicitly recorded in the CSV: synthetic cochain toys use mean normalized distance from identity; controlled tetrahedral mu2 rows use normalized distance to mu2 coboundaries; real rows use the saved fixed-setting cycle defect because the true cochain is unknown.
- `||c_hat - c||` is the same mean normalized distance between the noisy and true triangle cochains when a true cochain is available.
- The practical threshold is `stable_obstruction = Def(c_hat) > 3 * noise_floor`.
- For real fixed-setting rows the true cochain is unknown, so the script estimates a proxy noise floor from saved triangle-defect bootstrap samples. It does not recompute activations because overlap activations are not saved in the current artifacts.

## Bound checks

- Controlled rows with known true cochains: 840
- Direct check `|Def(c_hat)-Def(c)| <= ||c_hat-c||` pass rate: 1.0000
- Minimum `actual_synchronization_residual - predicted_lower_bound` on controlled rows: 0.0000
- Lower-bound violation count on controlled rows: 0
- Real rows set `real_brauer_claim_allowed = False`; finite-central gates are only marked on controlled finite-central rows that already carry that gate.

## Summary table

| source | family | n_rows | mean_def_observed | mean_noise_floor | direct_lipschitz_pass_rate | stable_obstruction_fraction |
| --- | --- | --- | --- | --- | --- | --- |
| controlled_twisted_overlap | mu2_coboundary | 60 | 0.0000 | 0.0000 | 1.0000 | 0.0000 |
| controlled_twisted_overlap | mu2_nontrivial_h2 | 60 | 0.2500 | 0.0000 | 1.0000 | 1.0000 |
| real_fixed_setting_triangle_proxy | real_alignment_unknown_true_cochain | 480 | 0.9768 | 0.0025 | nan | 1.0000 |
| real_fixed_setting_triangle_proxy | real_alignment_unknown_true_cochain | 480 | 0.9467 | 0.0038 | nan | 1.0000 |
| synthetic_mu2 | half_nontrivial | 120 | 0.5091 | 0.0768 | 1.0000 | 0.8250 |
| synthetic_mu2 | sparse_nontrivial | 120 | 0.2070 | 0.0768 | 1.0000 | 0.6083 |
| synthetic_mu2 | trivial | 120 | 0.0768 | 0.0768 | 1.0000 | 0.0000 |
| synthetic_u1 | smooth_high_phase | 120 | 0.3485 | 0.0596 | 1.0000 | 0.8167 |
| synthetic_u1 | smooth_low_phase | 120 | 0.1313 | 0.0596 | 1.0000 | 0.4917 |
| synthetic_u1 | trivial | 120 | 0.0596 | 0.0596 | 1.0000 | 0.0000 |

## Real fixed-setting proxy table

| dataset | architecture | n_models | domain_shift | matching | alignment_source | n_rows | mean_def_observed | mean_noise_floor | stable_fraction | mean_sync_residual |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fashion_mnist | mlp2 | 3.0000 | input_noise | activation | injected_noise | 30 | 0.9593 | 0.0019 | 1.0000 | 0.3241 |
| fashion_mnist | mlp2 | 3.0000 | input_noise | activation | observed | 30 | 0.9035 | 0.0031 | 1.0000 | 0.2219 |
| fashion_mnist | mlp2 | 3.0000 | input_noise | weight | injected_noise | 30 | 0.9910 | 0.0012 | 1.0000 | 0.3103 |
| fashion_mnist | mlp2 | 3.0000 | input_noise | weight | observed | 30 | 0.9854 | 0.0016 | 1.0000 | 0.2035 |
| fashion_mnist | mlp2 | 3.0000 | none | activation | injected_noise | 30 | 0.9618 | 0.0021 | 1.0000 | 0.3187 |
| fashion_mnist | mlp2 | 3.0000 | none | activation | observed | 30 | 0.9036 | 0.0033 | 1.0000 | 0.2127 |
| fashion_mnist | mlp2 | 3.0000 | none | weight | injected_noise | 30 | 0.9939 | 0.0008 | 1.0000 | 0.3096 |
| fashion_mnist | mlp2 | 3.0000 | none | weight | observed | 30 | 0.9896 | 0.0012 | 1.0000 | 0.2027 |
| fashion_mnist | mlp2 | 4.0000 | input_noise | activation | injected_noise | 30 | 0.9595 | 0.0046 | 1.0000 | 0.4444 |
| fashion_mnist | mlp2 | 4.0000 | input_noise | activation | observed | 30 | 0.9043 | 0.0078 | 1.0000 | 0.3378 |
| fashion_mnist | mlp2 | 4.0000 | input_noise | weight | injected_noise | 30 | 0.9928 | 0.0020 | 1.0000 | 0.4419 |
| fashion_mnist | mlp2 | 4.0000 | input_noise | weight | observed | 30 | 0.9884 | 0.0026 | 1.0000 | 0.3422 |

## Claim boundary

- Supported by this report: the controlled mu2 and U(1) definitions satisfy the direct Lipschitz-style defect stability check under injected cochain noise.
- Supported as a diagnostic artifact: fixed-setting real rows can be assigned a saved-triangle bootstrap noise floor and threshold flag.
- Not supported here: real Brauer/projective class detection, activation-level bootstrap stability, or any claim that obstruction scores prove real model-merging failure by themselves.
