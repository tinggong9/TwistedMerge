# Post-ICLR Official Baseline Report

Exact command: `/Users/tinggong/Documents/GitHub/TwistedMerge/.venv/bin/python experiments/post_iclr_official_baseline_integration.py --official-root /Users/tinggong/Documents/Codex/2026-07-17/files-mentioned-by-the-user-you/work/official-baseline-sources --jax-python /Users/tinggong/Documents/Codex/2026-07-17/files-mentioned-by-the-user-you/work/official-git-rebasin-py312-venv/bin/python --data-dir /Users/tinggong/Documents/GitHub/TwistedMerge/data --independent-checkpoint-root /Users/tinggong/Documents/GitHub/TwistedMerge/reports/checkpoints/external_baselines --common-checkpoint-root /Users/tinggong/Documents/GitHub/TwistedMerge/reports/checkpoints/same_base_task_vector`

Execution commit: `b19ead8fec5886533052af404fc9f460adc2b8a2`; worktree dirty before execution: `False`.

## Successful adapter-assisted official cores

| Regime | Baseline | Rows | Seeds | Failed | Mean score | Median | SD | 95% CI | Mean delta vs internal same method | Delta 95% CI | Wins/ties/losses |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- | --- |
| common_base_task_vector | official_ties | 3 | 3 | 0 | 0.8215 | 0.8228 | 0.0142 | [0.8067, 0.8350] | 0.0000 | [0.0000, 0.0000] | 0/3/0 |
| independent_initialization | official_c2m3 | 20 | 5 | 0 | 0.8608 | 0.8643 | 0.0185 | [0.8526, 0.8683] | 0.0128 | [0.0088, 0.0185] | 20/0/0 |
| independent_initialization | official_git_rebasin | 20 | 5 | 0 | 0.8546 | 0.8649 | 0.0228 | [0.8448, 0.8634] | 0.0086 | [0.0039, 0.0134] | 18/0/2 |

The bootstrap unit is the exact checkpoint setting; the seed column reports unique training-group seeds. Same-method deltas compare official Git Re-Basin with the internal pairwise implementation, official C2M3 with the internal C2M3-style implementation, and official TIES with the internal TIES-style implementation.

## Independent-regime comparison context

| Official core | Delta vs greedy soup | Delta vs TwistedMerge gauge | Delta vs TwistedMerge selector | Delta vs prediction ensemble upper bound |
| --- | --- | --- | --- | --- |
| official_c2m3 | -0.0106 [-0.0153, -0.0060] | 0.0087 [0.0050, 0.0123] | -0.0081 [-0.0136, -0.0030] | -0.0142 [-0.0190, -0.0098] |
| official_git_rebasin | -0.0168 [-0.0239, -0.0102] | 0.0025 [-0.0036, 0.0081] | -0.0143 [-0.0221, -0.0074] | -0.0204 [-0.0277, -0.0140] |

Greedy soup, TwistedMerge gauge/selector, and the prediction ensemble are pre-existing internal controls evaluated on the same checkpoint settings. The ensemble uses N branches and N-times inference and is an upper bound, not a same-cost candidate.

## Blocked methods and negative integrations

- `official_model_soups`: `blocked_incompatible_interface` -- Official main.py is inseparable from its CLIP/ImageNet model loader and evaluator; replacing those components would be a reimplementation, so the faithful internal greedy soup remains separately labeled.
- `official_task_arithmetic`: `blocked_license` -- The pinned author repository contains no LICENSE or COPYING file; author code was not used for a publishable metric.
- `official_dare`: `blocked_license` -- The pinned author MergeLM repository contains no LICENSE or COPYING file; author code was not used for a publishable metric.

## Claim decision

The run establishes that official Git Re-Basin and C2M3 matching cores, and the official TIES merge core, can be connected to exact TwistedMerge checkpoint families through explicit adapters. Whether any paired delta is positive is reported numerically and is not generalized beyond these checkpoints. No official Model Soups, Task Arithmetic, or DARE performance result is claimed.

All evaluated outputs are same-capacity single models with 1x inference. No lift or ensemble is included in the official-core score table.
