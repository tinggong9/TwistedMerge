# Official External Baseline Attempt

This report summarizes the official-code integration attempt documented in
`external_baselines/OFFICIAL_INTEGRATION.md`.

## Target Benchmark

The attempted target was the same MNIST MLP checkpoint protocol used by
`reports/external_baseline_comparison.md`:

- dataset: MNIST;
- architecture: one-hidden-layer ReLU MLP;
- model counts: `N=3,4`;
- widths: `32,64`;
- seeds: `1800,1801,1802,1803,1804`;
- checkpoint root: `reports/checkpoints/external_baselines/`;
- validation/test split and checkpoint reuse as in the existing external
  baseline report.

## Official Source Status

| Baseline | Official repository | Commit inspected | License | Attempt result |
| --- | --- | --- | --- | --- |
| Git Re-Basin | <https://github.com/samuela/git-re-basin> | `ef40098257ab97243930eba737d6dcb8edd5863e` | MIT | Import/run blocked |
| C2M3 | <https://github.com/crisostomi/cycle-consistent-model-merging> | `ea1eca76b19c5d57ed97b1ef396368189e864eee` | MIT | Import/run blocked |
| Model Soups | <https://github.com/mlfoundations/model-soups> | `d5398f181ea51c5cd9d95ebacc6ea7132bb108ec` | MIT | Import/run blocked |

## Why No Official Results Were Generated

Git Re-Basin is JAX/Flax/TFDS/W&B-based and its MNIST MLP uses three hidden
`Dense(512)` layers plus W&B/Flax artifacts. The current project venv lacks
`jax`, `flax`, `tensorflow_datasets`, and `wandb`, and no `nix`/`nix-shell` is
available to recreate the official environment. The saved TwistedMerge
checkpoints are PyTorch one-hidden-layer MLP state dicts.

C2M3 pins Python `3.9.19`, `torch==1.13.0`, Hydra, PyTorch Lightning,
TorchMetrics, W&B, and uv-managed dependencies. The current environment is
Python 3.12/Torch 2.12, with no `uv` or `python3.9` available. Its official MLP
config/source also uses a deeper `layer0..layer4` model rather than the
TwistedMerge `hidden.*`/`classifier.*` state dict.

Model Soups targets CLIP/ViT ImageNet soups with a conda Python 3.6/Torch 1.7
environment, CLIP checkpoints, ImageNet-family datasets, and
`classification_head.*` state dict keys. The current environment lacks conda,
Python 3.6, `clip`, and supporting packages, and the official loader does not
accept the MNIST MLP checkpoints.

## Decision

No `reports/csv/official_external_baseline_comparison.csv` or
`reports/tables/official_external_baseline_comparison.tex` was generated,
because no direct official-code run succeeded on the exact checkpoint set.

The faithful in-repository implementations remain the comparison layer for the
current paper draft:

- Git-ReBasin-style pairwise permutation alignment;
- C2M3-style cycle-consistent synchronization;
- Model-Soups-style greedy soup.

This report supports only the integration-attempt claim. It does not support
any claim that TwistedMerge beats official external baselines.
