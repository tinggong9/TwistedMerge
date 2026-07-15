# Official Neural Sheaf Diffusion Integration

This note records the local attempt to run the official Neural Sheaf Diffusion
implementation as an optional related experiment for TwistedMerge.

- Official repository: <https://github.com/twitter-research/neural-sheaf-diffusion>
- Paper: "Neural Sheaf Diffusion: A Topological Perspective on Heterophily and Oversmoothing in GNNs"
- License: Apache-2.0
- Integration mode: external clone only, no vendored code
- Local clone used for this attempt: `/private/tmp/neural-sheaf-diffusion`
- Official commit tested: `11e21b561d884713ab1a18a521a7dc2fb26b9361`

## Environment Outcome

The main TwistedMerge venv is still PyTorch-only for normal repo work. It has
PyTorch installed but does not include `torch_geometric`, `torch_sparse`,
`torch_scatter`, or `torchdiffeq`.

A separate PyG-compatible environment was feasible on this machine:

```bash
git clone --depth=1 https://github.com/twitter-research/neural-sheaf-diffusion.git /private/tmp/neural-sheaf-diffusion
python3 -m venv /private/tmp/nsd-pyg-py39
/private/tmp/nsd-pyg-py39/bin/python -m pip install --upgrade pip setuptools wheel
/private/tmp/nsd-pyg-py39/bin/python -m pip install torch==1.11.0
/private/tmp/nsd-pyg-py39/bin/python -m pip install 'setuptools<81' numpy==1.26.4
/private/tmp/nsd-pyg-py39/bin/python -m pip install --force-reinstall --no-build-isolation torch-scatter==2.0.9 torch-sparse==0.6.13 -f https://data.pyg.org/whl/torch-1.11.0+cpu.html
/private/tmp/nsd-pyg-py39/bin/python -m pip install --force-reinstall numpy==1.26.4
/private/tmp/nsd-pyg-py39/bin/python -m pip install torch-geometric==2.0.4 torchdiffeq==0.2.2 wandb==0.13.1 torch-householder==1.0.1 networkx==2.6.3 tqdm==4.62.3 pytest==6.2.5 GitPython
```

Important environment notes:

- No `conda`, `mamba`, or `micromamba` was available, so the official
  `environment_cpu.yml` was translated to a Python 3.9 venv.
- Installing all PyG packages in one pip invocation failed because
  `torch-scatter` tried to build before `torch` was importable.
- Latest `torch-sparse` was incompatible with `torch==1.11.0` on this stack
  because it expected `torch.sparse_csc_tensor`, so the working pair is
  `torch-scatter==2.0.9` and `torch-sparse==0.6.13`.
- `torch-householder` attempted to JIT-build in
  `<user-cache>/torch_extensions`, which was not writable
  from this session. The working invocation sets:

```bash
PATH=/private/tmp/nsd-pyg-py39/bin:$PATH
TORCH_EXTENSIONS_DIR=/private/tmp/torch_extensions
```

Working package versions:

| Package | Version |
| --- | --- |
| `torch` | `1.11.0` |
| `torch_geometric` | `2.0.4` |
| `torch_scatter` | `2.0.9` |
| `torch_sparse` | `0.6.13` |
| `torchdiffeq` | `0.2.2` |
| `torch_householder` | `1.0.1` |
| `numpy` | `1.26.4` |
| `scipy` | `1.13.1` |

## Official Smoke Run

The official code ran a tiny WebKB Texas BundleSheaf configuration:

```bash
PATH=/private/tmp/nsd-pyg-py39/bin:$PATH TORCH_EXTENSIONS_DIR=/private/tmp/torch_extensions WANDB_MODE=disabled /private/tmp/nsd-pyg-py39/bin/python -m exp.run --dataset=texas --model=BundleSheaf --folds=1 --epochs=3 --early_stopping=3 --d=2 --layers=1 --hidden_channels=8 --left_weights=True --right_weights=True --lr=0.02 --weight_decay=5e-3 --input_dropout=0.0 --dropout=0.0 --use_act=True --normalised=True --sparse_learner=True
```

Observed output:

| Metric | Value |
| --- | ---: |
| Dataset | `texas` |
| Model | `BundleSheaf` |
| Folds | `1` |
| Epochs requested | `3` |
| Test accuracy | `0.6486` |
| Best validation accuracy | `0.5254` |

This is a smoke run only. It does not establish a GNN performance claim.

## Cycle Diagnostic Integration

TwistedMerge adds a non-vendored diagnostic wrapper:

```bash
PATH=/private/tmp/nsd-pyg-py39/bin:$PATH TORCH_EXTENSIONS_DIR=/private/tmp/torch_extensions WANDB_MODE=disabled PYTHONPYCACHEPREFIX=/private/tmp/codex-pycache /private/tmp/nsd-pyg-py39/bin/python experiments/nsd_official_cycle_diagnostics.py --nsd-root /private/tmp/neural-sheaf-diffusion --dataset texas --fold 0 --seed 43 --epochs 3 --edge-weight-modes false true --cycle-lambda-attempt 1.0
```

Outputs:

- `reports/csv/nsd_cycle_diagnostics.csv`
- `reports/configs/nsd_official_integration_config.json`
- `reports/nsd_official_integration_report.md`

The diagnostic reads `model.sheaf_learners[layer].L` after a forward pass and
maps the cached `saved_tril_maps` back to undirected graph triangles. In the
official code this tensor is created with `clone().detach()`, so it is valid
for post-hoc diagnostics but not for a non-invasive differentiable cycle
regularizer.

Tiny-run diagnostic summary:

| Variant | Edge weights | Triangles | Mean cycle score | Test accuracy | Regularizer applied |
| --- | --- | ---: | ---: | ---: | --- |
| BundleSheaf connection cache | no | 67 | `3.44e-7` | `0.5135` | no |
| BundleSheaf weighted cache | yes | 67 | `0.9997` | `0.6486` | no |
| Cycle-regularizer attempt | no | 67 | `3.44e-7` | `0.5135` | no, cache detached |

The unweighted connection cache is the cleaner holonomy diagnostic. The default
weighted cache is not a pure connection map, so its near-one cycle score should
not be interpreted as a cohomological obstruction.

## Inclusion Decision

The official Neural Sheaf Diffusion side can be included as an optional related
experiment with a separate PyG environment. The current status supports only:

- official NSD can run on a tiny WebKB Texas configuration in the separate env;
- learned official transport caches can be post-processed for triangle cycle
  diagnostics;
- the existing PyTorch-only synthetic sheaf/GNN run remains supplementary.

The current status does not support:

- sheaf regularization generally improves GNNs;
- cycle regularization improves official Neural Sheaf Diffusion accuracy;
- the official NSD diagnostic is evidence for TwistedMerge model-merging
  performance.
