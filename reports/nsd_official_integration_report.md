# Official Neural Sheaf Diffusion Integration Report

## Summary

The official Neural Sheaf Diffusion code can run as an optional external
baseline in a separate PyG environment. A tiny WebKB Texas BundleSheaf run
completed, and TwistedMerge computed triangle cycle diagnostics from the
official learned transport cache.

The cycle diagnostic is post-hoc only. The official discrete model stores
`model.sheaf_learners[layer].L` via `clone().detach()`, so this attempt did not
test a differentiable cycle regularizer inside the official code path.

## Exact Commands Run

Dependency check in the main TwistedMerge venv:

```bash
.venv/bin/python - <<'PY'
mods = ['torch','torch_geometric','torch_sparse','torch_scatter','torchdiffeq']
for mod in mods:
    try:
        imported = __import__(mod)
        print(mod, getattr(imported, '__version__', 'ok'))
    except Exception as exc:
        print(mod, 'MISSING', type(exc).__name__, exc)
PY
```

Result: `torch 2.12.1` was present, but `torch_geometric`, `torch_sparse`,
`torch_scatter`, and `torchdiffeq` were missing.

Official clone and separate environment setup:

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

Official smoke run:

```bash
PATH=/private/tmp/nsd-pyg-py39/bin:$PATH TORCH_EXTENSIONS_DIR=/private/tmp/torch_extensions WANDB_MODE=disabled /private/tmp/nsd-pyg-py39/bin/python -m exp.run --dataset=texas --model=BundleSheaf --folds=1 --epochs=3 --early_stopping=3 --d=2 --layers=1 --hidden_channels=8 --left_weights=True --right_weights=True --lr=0.02 --weight_decay=5e-3 --input_dropout=0.0 --dropout=0.0 --use_act=True --normalised=True --sparse_learner=True
```

TwistedMerge diagnostic wrapper:

```bash
PATH=/private/tmp/nsd-pyg-py39/bin:$PATH TORCH_EXTENSIONS_DIR=/private/tmp/torch_extensions WANDB_MODE=disabled PYTHONPYCACHEPREFIX=/private/tmp/codex-pycache /private/tmp/nsd-pyg-py39/bin/python experiments/nsd_official_cycle_diagnostics.py --nsd-root /private/tmp/neural-sheaf-diffusion --dataset texas --fold 0 --seed 43 --epochs 3 --edge-weight-modes false true --cycle-lambda-attempt 1.0
```

Generated:

- `reports/csv/nsd_cycle_diagnostics.csv`
- `reports/configs/nsd_official_integration_config.json`

## Environment

| Item | Value |
| --- | --- |
| TwistedMerge commit at run time | `401ae6977404ba296c2ec63420c59079a6a0640a` |
| Official NSD commit | `11e21b561d884713ab1a18a521a7dc2fb26b9361` |
| OS/platform | `macOS-26.0.1-arm64-arm-64bit` |
| Python | `3.9.6` |
| Device | CPU |
| `torch` | `1.11.0` |
| `torch_geometric` | `2.0.4` |
| `torch_scatter` | `2.0.9` |
| `torch_sparse` | `0.6.13` |
| `torchdiffeq` | `0.2.2` |
| `torch_householder` | `1.0.1` |
| `numpy` | `1.26.4` |
| `scipy` | `1.13.1` |
| `wandb` | `0.13.1` |

The official `environment_cpu.yml` pins Python `3.9.9`, PyTorch `1.11.0`,
and PyG `2.0.4`. No `conda`, `mamba`, or `micromamba` executable was available,
so the local run used a Python 3.9 venv.

## Setup Blockers and Fixes

| Issue | Outcome |
| --- | --- |
| Main TwistedMerge venv lacks PyG dependencies | Kept unchanged; created separate `/private/tmp/nsd-pyg-py39` env. |
| One-shot pip install of PyG stack failed while building extensions | Installed `torch==1.11.0` first, then PyG extensions. |
| Latest `torch_sparse` expected `torch.sparse_csc_tensor` absent in PyTorch 1.11 | Used `torch_sparse==0.6.13` and `torch_scatter==2.0.9`. |
| `torch_householder` tried to write under an unwritable user cache | Set `TORCH_EXTENSIONS_DIR=/private/tmp/torch_extensions` and put venv `bin` on `PATH` for `ninja`. |

## Official Smoke Result

The official `exp.run` command completed on WebKB Texas:

| Metric | Value |
| --- | ---: |
| Test accuracy | `0.6486` |
| Best validation accuracy | `0.5254` |
| Epochs reported | `2` as final zero-indexed epoch |
| Laplacian abs avg | `0.0323` |

This is a smoke test, not a benchmark result.

## Cycle Diagnostic Results

From `reports/csv/nsd_cycle_diagnostics.csv`:

| Variant | Edge weights | Triangles | Mean cycle score | Max cycle score | Final test accuracy | Cycle regularizer |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `bundle_sheaf_connection_cache` | false | 67 | `3.438e-7` | `1.014e-6` | `0.5135` | not requested |
| `bundle_sheaf_weighted_cache` | true | 67 | `0.9997` | `0.9997` | `0.6486` | not requested |
| `bundle_sheaf_connection_cache_cycle_regularizer_attempt` | false | 67 | `3.438e-7` | `1.014e-6` | `0.5135` | not applied, cache detached |

The unweighted connection-cache row gives the cleanest holonomy diagnostic:
the cached maps are orthogonal transports, and triangle products are essentially
identity on this tiny run.

The default weighted row should not be read as a cohomology obstruction. With
edge weights enabled, the cached off-diagonal terms include learned scalar
weights and are not pure transition maps.

## What This Proves

- A separate PyG-compatible environment can run the official Neural Sheaf
  Diffusion code on a tiny WebKB Texas configuration.
- TwistedMerge can compute a triangle cycle diagnostic from the official
  discrete BundleSheaf learned transport cache.
- The official sheaf/GNN side is feasible as an optional related experiment,
  with dependency setup documented outside the main PyTorch-only repo path.

## What This Does Not Prove

- It does not prove that sheaf regularization generally improves GNNs.
- It does not test a differentiable cycle regularizer in official NSD, because
  the exposed cache is detached.
- It does not provide a fair NSD benchmark against GCN/GAT/MLP baselines.
- It does not support any claim about TwistedMerge model-merging performance.

## Decision

Include official Neural Sheaf Diffusion as an optional external related
experiment. Keep `experiments/sheaf_gnn_cycle_diagnostics.py` as a supplementary
PyTorch-only synthetic diagnostic, not as a replacement for official NSD.
