# Official External-Code Integration Attempt

This note records the official-code integration attempt for Git Re-Basin,
C2M3, and Model Soups. The attempt used the same target benchmark protocol as
`reports/external_baseline_comparison.md`: MNIST, one-hidden-layer ReLU MLP,
`N=3,4`, widths `32,64`, seeds `1800..1804`, validation fraction `0.2`, and
saved checkpoints under `reports/checkpoints/external_baselines/`.

No official-code baseline numbers were produced in this pass. The existing
faithful in-repository baselines therefore remain the comparison layer. The
project must not claim that TwistedMerge beats official Git Re-Basin, official
C2M3, or official Model Soups from the current artifacts.

## Clone Status

Official repositories were shallow-cloned outside the tracked repository into
`work/official_external_baselines/`:

| Baseline | Official repository | Cloned commit | License | Direct same-checkpoint run |
| --- | --- | --- | --- | --- |
| Git Re-Basin | <https://github.com/samuela/git-re-basin> | `ef40098257ab97243930eba737d6dcb8edd5863e` | MIT | Not run |
| C2M3 | <https://github.com/crisostomi/cycle-consistent-model-merging> | `ea1eca76b19c5d57ed97b1ef396368189e864eee` | MIT | Not run |
| Model Soups | <https://github.com/mlfoundations/model-soups> | `d5398f181ea51c5cd9d95ebacc6ea7132bb108ec` | MIT | Not run |

The license classification comes from each repository's `LICENSE` file.

## Local Environment Probes

Current TwistedMerge environment:

- Python `3.12.13`
- Torch `2.12.1`
- Torchvision `0.27.1`

Import probes in the current project venv:

| Package or entry point | Result |
| --- | --- |
| `jax`, `flax`, `tensorflow_datasets`, `wandb` | Missing |
| `hydra`, `pytorch_lightning`, `torchmetrics` | Missing |
| `clip`, `open_clip` | Missing |
| `torch`, `torchvision` | Present |
| Git Re-Basin `weight_matching.py` | Fails at import: missing `jax` |
| Git Re-Basin `mnist_mlp_weight_matching.py` | Fails at import: missing `jax` |
| C2M3 `ccmm` / `FrankWolfeSynchronizedMerger` | Fails at import: missing `omegaconf` |
| Model Soups `utils.py` | Imports |
| Model Soups `main.py` | Fails at import: missing `wget` before reaching CLIP/ImageNet setup |

Environment manager probes:

| Tool | Available here |
| --- | --- |
| `nix`, `nix-shell` | No |
| `uv` | No |
| `conda` | No |
| `python3.9` | No |
| `python3.6` | No |

## Official Git Re-Basin

Repository: <https://github.com/samuela/git-re-basin>

The official repo is JAX/Flax-based. Its `shell.nix` declares dependencies
including `flax`, `jax`, `jaxlib`, TensorFlow, `tensorflow-datasets`, `wandb`,
and plotting packages. The repo's MNIST MLP script loads/saves Flax parameters
through W&B artifacts.

The official MNIST MLP architecture in `src/mnist_mlp_train.py` is a Flax model
with three hidden `Dense(512)` ReLU layers followed by `Dense(10)`. The
TwistedMerge external-baseline checkpoints are PyTorch state dicts for a
one-hidden-layer MLP:

```text
hidden.weight        (32 or 64, 784)
hidden.bias          (32 or 64,)
classifier.weight    (10, 32 or 64)
classifier.bias      (10,)
```

Direct official-code execution on the exact checkpoint set is therefore blocked
by both environment and interface mismatch:

- the current environment lacks JAX/Flax/TFDS/W&B;
- no `nix`/`nix-shell` is available to recreate the official environment;
- the official MNIST script expects its own Flax/W&B artifacts and architecture;
- converting TwistedMerge PyTorch checkpoints into Flax parameter dictionaries
  and evaluating them in TwistedMerge would be adapter code, not a direct
  official-code baseline run.

The faithful in-repo Git-ReBasin-style pairwise permutation baseline remains
the comparison for the current MNIST MLP benchmark.

## Official C2M3

Repository: <https://github.com/crisostomi/cycle-consistent-model-merging>

The official C2M3 repo is a Hydra/uv/PyTorch-Lightning project. Its
`pyproject.toml` pins:

- `requires-python = "==3.9.19"`
- `torch==1.13.0`
- `pytorch-lightning==1.7.7`
- `torchmetrics==0.10.3`
- `hydra-core>=1.3.2`
- `wandb>=0.21.1`

The local environment is Python 3.12/Torch 2.12, and neither `uv` nor
`python3.9` is available. Importing the package in the current venv fails before
any benchmark code can run.

The official MNIST MLP config also does not match the TwistedMerge checkpoint
set. `conf/model/mlp.yaml` uses `num_hidden_layers: 4`, and the source model
has layers `layer0` through `layer4`, with default hidden dimension `512`.
TwistedMerge's checkpoint set is a one-hidden-layer model with parameter names
`hidden.*` and `classifier.*`.

Direct official C2M3 execution on the exact saved checkpoint set is therefore
infeasible in this pass. Recreating the official Python 3.9 environment and
writing adapters for the TwistedMerge checkpoint format would be a separate
integration project; any such adapter must be reported as adapter-assisted, not
as unmodified official-code output.

The faithful in-repo C2M3-style cycle-consistent synchronization baseline
remains the comparison for the current MNIST MLP benchmark.

## Official Model Soups

Repository: <https://github.com/mlfoundations/model-soups>

The official Model Soups repo targets CLIP/ViT ImageNet soups. Its
`environment.yml` pins an old conda/CUDA stack, including:

- Python `3.6.13`
- PyTorch `1.7.1`
- Torchvision `0.8.2`
- `clip==1.0`
- ImageNet/ImageNetV2/ObjectNet dependencies

No `conda` or `python3.6` is available here. Importing `main.py` in the current
venv fails on missing `wget`, and the script subsequently expects `clip`, CLIP
ViT-B/32 checkpoints, ImageNet-family datasets, and state dict keys such as
`classification_head.weight`.

This is incompatible with the exact TwistedMerge MNIST MLP checkpoints, whose
state dict keys are `hidden.*` and `classifier.*`. Running the official greedy
soup loop on MNIST MLPs would require replacing the official model loader,
dataset classes, evaluation function, and checkpoint naming scheme, which would
amount to a reimplementation/adaptation rather than direct official-code
execution.

The faithful in-repo greedy soup baseline remains the comparison for the current
MNIST MLP benchmark.

## Outcome

The official-code integration attempt is complete as a documented negative
result:

- official repositories were cloned and inspected;
- license and commit hashes were recorded;
- required environments were identified;
- import/entry-point probes were attempted in the current environment;
- the exact saved MNIST MLP checkpoint format was compared against official
  model/checkpoint expectations;
- no official baseline CSV/table was generated because no official-code method
  ran successfully on the exact checkpoint set.

Allowed claim from this artifact:

- "Official external-code integration was attempted and documented; no official
  baseline results are available in the current repository artifacts."

Forbidden claim from this artifact:

- "TwistedMerge beats official external baselines."
