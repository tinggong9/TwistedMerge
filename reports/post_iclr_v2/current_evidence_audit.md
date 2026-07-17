# Post-ICLR v2 Current Evidence Audit

Generated from tracked source artifacts at `origin/main` `a89b7f0486a6ec71e2885f25c386461bd19bd279`. The audit ran in an isolated Codex worktree and did not modify manuscript, bibliography, or existing paper-figure files.

## Protected collaborator work

The authoritative main checkout was clean at the preflight. The following manuscript-like uncommitted paths exist in other worktrees and are explicitly out of scope:

- `/Users/tinggong/Documents/Codex/2026-07-14/ru-3/work/TwistedMerge-stage2/reports/overnight_program/tables/lora_holonomy.tex`
- `/Users/tinggong/Documents/Codex/2026-07-14/ru-3/work/TwistedMerge-stage3/reports/overnight_program/tables/federated_frame.tex`
- `/Users/tinggong/Documents/Codex/2026-07-14/ru-3/work/TwistedMerge-stage4/reports/overnight_program/tables/transformer_merging.tex`

## Verified official-baseline starting point

- Adapter-assisted official Git Re-Basin: `20` exact independent-initialization settings.
- Adapter-assisted official C2M3: `20` exact independent-initialization settings.
- Adapter-assisted official TIES: `3` exact common-base settings and zero paired difference from the internal TIES-style implementation.
- Evaluated official-core rows with recorded runtime failure: `0`.
- Official Model Soups remains interface-blocked. Task Arithmetic and DARE remain license-blocked in the pinned author repositories. No blocked metric is substituted.

## Verified narrow positive result

- Existing validation-only selector minus official Git Re-Basin: `0.014345`, 95% CI `[0.007435, 0.022095]`.
- Existing validation-only selector minus official C2M3: `0.008145`, 95% CI `[0.003030, 0.013580]`.

These are supported-narrow exact-family comparisons, not broad external-baseline or SOTA claims.

## Verified negative boundaries

- Official C2M3 minus the pure TwistedMerge monomial gauge: `0.008730`, 95% CI `[0.004994, 0.012320]`; the proposed pure-gauge win has the wrong direction.
- Existing improved selector minus greedy soup: `-0.002415`, 95% CI `[-0.004175, -0.000860]`.
- Soup-based selections: `19/20` (`95.0%`). The existing selector advantage over matching cores is therefore not attributable to residual geometry without a new budget-matched experiment.
- Biomedical inferred retransport, TwistedMerge-specific benefit, multidomain benefit, and realistic residual correction are all false under their recorded gates.
- Any inferred spatial-output method on a measured quality-cost Pareto frontier: `False`.
- Full independently initialized ResNet-18 CIFAR-10/CIFAR-100 and BatchNorm-aware gauge experiments remain absent.

## Claim classification

| claim_id | regime | status | value | ci_low | ci_high |
| --- | --- | --- | --- | --- | --- |
| official_git_rebasin_exact_family | independent-initialization/rebasin | supported-narrow | 20 |  |  |
| official_c2m3_exact_family | independent-initialization/rebasin | supported-narrow | 20 |  |  |
| official_ties_matches_internal | common-base task-vector | supported-narrow | 0.0 | 0.0 | 0.0 |
| selector_over_official_git_rebasin | independent-initialization/rebasin | supported-narrow | 0.014345 | 0.0074349999999999 | 0.022095 |
| selector_over_official_c2m3 | independent-initialization/rebasin | supported-narrow | 0.008145 | 0.0030297499999999 | 0.013580125 |
| pure_gauge_over_official_c2m3 | independent-initialization/rebasin | negative | -0.00873 | -0.012320125 | -0.004994 |
| selector_over_greedy_soup | greedy-soup validation descent | negative | -0.0024149999999999 | -0.0041751249999999 | -0.0008598749999999 |
| selector_choices_are_soup_dominated | greedy-soup validation descent | descriptive | 0.95 |  |  |
| controlled_rank_lift | controlled planted obstruction | supported-narrow | 0.25 | 0.25 | 0.25 |
| raw_weight_average_prediction | diagnostic prediction | negative |  |  |  |
| biomedical_inferred_retransport | biomedical site/domain heterogeneity | negative | False |  |  |
| biomedical_twistedmerge_specific_benefit | biomedical site/domain heterogeneity | negative | False |  |  |
| biomedical_multidomain_benefit | biomedical site/domain heterogeneity | negative | False |  |  |
| biomedical_residual_correction | biomedical site/domain heterogeneity | negative | False |  |  |
| resnet18_cifar10_independent_merge | independent-initialization/rebasin | pending |  |  |  |
| batchnorm_aware_exact_gauge | exact gauge construction | pending |  |  |  |
| broad_sota | claim boundary | forbidden |  |  |  |
| natural_brauer_classes | claim boundary | forbidden |  |  |  |

Machine-readable details are in `current_claim_matrix.csv`. Integrity and row counts are in `current_artifact_manifest.csv`.
