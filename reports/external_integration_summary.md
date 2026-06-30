# External Integration Summary

This reproducibility appendix consolidates the official external-baseline and
Neural Sheaf Diffusion integration attempts. It is based on:

- `external_baselines/OFFICIAL_INTEGRATION.md`
- `reports/official_external_baseline_attempt.md`
- `external_baselines/NSD_INTEGRATION.md`
- `reports/nsd_official_integration_report.md`

## Claim Boundary

Allowed: Official integrations are documented; main comparisons remain faithful
in-repo unless official runs succeeded.

Not supported: Official external code confirms the method.

## Integration Table

| External project | Official repository | License | Attempted environment | Official code ran? | Blocker or scope limit | Faithful in-repo surrogate | Exact claim boundary |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Git Re-Basin | <https://github.com/samuela/git-re-basin> | MIT | Current TwistedMerge Python `3.12.13`, Torch `2.12.1`; official shallow clone at commit `ef40098257ab97243930eba737d6dcb8edd5863e`; no `nix`/`nix-shell` | No | Missing JAX/Flax/TFDS/W&B; official scripts expect Flax/W&B artifacts and a three-hidden-layer `Dense(512)` MNIST MLP, not TwistedMerge PyTorch one-hidden-layer checkpoints | Faithful Git-ReBasin-style pairwise permutation alignment in the in-repo external-baseline comparison | No official Git Re-Basin baseline numbers; do not claim TwistedMerge beats official Git Re-Basin |
| C2M3 | <https://github.com/crisostomi/cycle-consistent-model-merging> | MIT | Current TwistedMerge Python `3.12.13`, Torch `2.12.1`; official shallow clone at commit `ea1eca76b19c5d57ed97b1ef396368189e864eee`; no `uv` or `python3.9` | No | Official project pins Python `3.9.19`, Torch `1.13.0`, Hydra, PyTorch Lightning, TorchMetrics, and W&B; official MLP config uses a deeper `layer0..layer4` model rather than TwistedMerge `hidden.*`/`classifier.*` state dicts | Faithful C2M3-style cycle-consistent permutation synchronization in the in-repo comparison | No official C2M3 baseline numbers; do not claim TwistedMerge beats official C2M3 |
| Model Soups | <https://github.com/mlfoundations/model-soups> | MIT | Current TwistedMerge Python `3.12.13`, Torch `2.12.1`; official shallow clone at commit `d5398f181ea51c5cd9d95ebacc6ea7132bb108ec`; no `conda` or `python3.6` | No | Official repo targets CLIP/ViT ImageNet soups with Python `3.6.13`, Torch `1.7.1`, CLIP, ImageNet-family datasets, and `classification_head.*` keys; current import also hit missing `wget` before CLIP/ImageNet setup | Faithful greedy and uniform soup implementations over the same in-repo MNIST MLP checkpoints | No official Model Soups baseline numbers; do not claim TwistedMerge beats official Model Soups |
| Neural Sheaf Diffusion | <https://github.com/twitter-research/neural-sheaf-diffusion> | Apache-2.0 | Separate Python `3.9` venv at `/private/tmp/nsd-pyg-py39`, official clone at commit `11e21b561d884713ab1a18a521a7dc2fb26b9361`, Torch `1.11.0`, PyG `2.0.4`, CPU, `TORCH_EXTENSIONS_DIR=/private/tmp/torch_extensions` | Yes, as a tiny smoke test | Official WebKB Texas BundleSheaf run completed for one fold and three requested epochs; this is not a fair GNN benchmark. Cycle regularizer was not applied because the official discrete cache is detached via `clone().detach()` | In-repo PyTorch synthetic sheaf/GNN diagnostic remains supplementary; `experiments/nsd_official_cycle_diagnostics.py` is a non-vendored wrapper for official learned transport-cache diagnostics | May claim official NSD can run a tiny external smoke test and can be post-processed for triangle diagnostics; do not claim NSD confirms TwistedMerge model merging or that cycle regularization improves GNNs |

## Model-Merging Baseline Outcome

The official Git Re-Basin, C2M3, and Model Soups repositories were cloned,
licensed, inspected, and probed in the current environment. None produced direct
official-code baseline metrics on the exact TwistedMerge MNIST MLP checkpoint
set. Therefore no `official_external_baseline_comparison.csv` or official
LaTeX table was generated.

The main model-merging comparisons remain the documented faithful in-repo
surrogates: Git-ReBasin-style pairwise alignment, C2M3-style cycle-consistent
synchronization, and Model-Soups-style greedy/uniform soups. These comparisons
are fair for the paper's current internal-checkpoint claims, but they are not
official external-code wins.

## NSD Outcome

Official Neural Sheaf Diffusion is the only external code path in this appendix
that ran successfully. The run was a tiny WebKB Texas BundleSheaf smoke test in
a separate PyG environment, reporting test accuracy `0.6486` and best
validation accuracy `0.5254`. TwistedMerge then computed post-hoc triangle
cycle diagnostics from the official learned transport cache.

The unweighted connection-cache diagnostic had 67 triangles and mean cycle
score about `3.438e-7`. The weighted cache had mean score about `0.9997`, but
that row includes learned scalar edge weights and should not be interpreted as
a pure cohomological obstruction. The cycle-regularizer attempt was not applied
inside official NSD because the exposed cache is detached.

## Reproducibility Boundary

This appendix is a reproducibility record, not a performance win section.
Official integrations are documented, license-clean, and non-vendored. Main
comparisons remain faithful in-repo unless official runs succeeded. The current
artifacts do not support the claim that official external code confirms the
method.
