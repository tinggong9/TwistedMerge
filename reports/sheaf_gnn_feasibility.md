# Sheaf/GNN Feasibility Report

This is a pre-implementation feasibility check for benchmark series 6.  The external reference is the official Neural Sheaf Diffusion repository:

- Repository: <https://github.com/twitter-research/neural-sheaf-diffusion>
- Paper: "Neural Sheaf Diffusion: A Topological Perspective on Heterophily and Oversmoothing in GNNs" (NeurIPS 2022)
- License: Apache-2.0
- Integration decision: do not vendor the external repository.  Use it as a documented design reference and keep any optional local run self-contained.

## Supported Datasets

The official Neural Sheaf Diffusion code supports these datasets through `utils/heterophilic.py` and bundled fixed split files:

| Dataset family | Dataset names | Local split files in official repo | Loading status |
| --- | --- | ---: | --- |
| WebKB | `texas`, `wisconsin`, `cornell` | 10 each | Downloaded on the fly by the official loader. |
| Actor/Film | `film` | 10 | Downloaded on the fly by the official loader. |
| WikipediaNetwork | `chameleon`, `squirrel` | 10 each | Loader expects Geom-GCN raw files under `datasets/<name>/raw/`; the official README says these must be downloaded separately. |
| Planetoid/Geom-GCN | `cora`, `citeseer`, `pubmed` | 10 each | Loader expects Geom-GCN-style raw files under `datasets/<name>/raw/`. |

The TwistedMerge environment has PyTorch but does not have `torch_geometric`, `torch_sparse`, or `torch_scatter`, so the official data loaders and model classes cannot run here without a new PyG environment.  This should not block the model-merging line of experiments.

## Baselines Without Major Refactoring

Official Neural Sheaf Diffusion exposes these baselines with minimal code changes once the PyG environment is available:

| Baseline | Official class/script path | Notes |
| --- | --- | --- |
| Discrete diagonal sheaf diffusion | `models/disc_models.py::DiscreteDiagSheafDiffusion` | Example used for Chameleon. |
| Discrete bundle/orthogonal sheaf diffusion | `models/disc_models.py::DiscreteBundleSheafDiffusion` | Example used for Texas/Wisconsin/Cornell/Squirrel. |
| Discrete general sheaf diffusion | `models/disc_models.py::DiscreteGeneralSheafDiffusion` | More flexible restriction maps; likely heavier. |
| Continuous/ODE diagonal, bundle, and general sheaf diffusion | `models/cont_models.py` | Requires `torchdiffeq`; not the minimal local path. |
| Graph Laplacian diffusion | `models/cont_models.py::GraphLaplacianDiffusion` | GCN-like diffusion control, but not exposed through `exp/run.py` as a normal model choice. |

The official repo does not provide a plain GCN training script as a first-class `--model` choice in `exp/parser.py`; a fair GCN control would need a small wrapper or a separate local implementation.

## Learned Restriction Maps

In the official discrete models, learned edge maps are available after a forward pass:

- `models/sheaf_models.py::SheafLearner` stores a detached tensor in `self.L` through `set_L`.
- `models/disc_models.py` calls `self.sheaf_learners[layer].set_L(trans_maps)` after each Laplacian builder call.
- `models/laplacian_builders.py` constructs `saved_tril_maps` in `DiagLaplacianBuilder`, `NormConnectionLaplacianBuilder`, and `GeneralLaplacianBuilder`.

Therefore a cycle diagnostic can be added without rewriting the official model: run a forward pass, read `model.sheaf_learners[layer].L`, map the stored transport matrices back to graph edges, and compose them around triangles or short cycles.  ODE models cache Laplacians inside the ODE function, but transport maps are less directly exposed; discrete models are the minimal target.

## Cycles and Triangles

The official split files alone do not include graph raw data, so triangle counts for WebKB/Film/Wikipedia/Planetoid were not verified locally in this pass.  For the smallest local experiment, the safest feasible choice is a synthetic heterophilic stochastic-block graph with explicitly retained triangles:

- two labels/classes;
- target edge heterophily levels around `0.25`, `0.55`, and `0.85`;
- average degree around `8`;
- a minimum triangle floor during generation so cycle holonomy is measurable.

This synthetic path is not an official Neural Sheaf Diffusion benchmark.  It is a modest diagnostic smoke test for whether learned sheaf transports expose measurable cycle inconsistency on heterophilic graphs.

## Minimal Local Run

Feasible local run:

```bash
.venv/bin/python experiments/sheaf_gnn_cycle_diagnostics.py --epochs 120 --seeds 0,1,2 --heterophily-targets 0.25,0.55,0.85
```

This run should finish on CPU because it avoids PyG and uses only PyTorch dense/small-graph operations.  It compares:

- a plain dense GCN baseline;
- a small rotation-sheaf GNN baseline with learned orthogonal edge transports;
- the same rotation-sheaf GNN with a weak cycle-consistency regularizer.

Expected outputs if run:

- `reports/sheaf_gnn_optional_report.md`
- `reports/csv/sheaf_gnn_cycle_diagnostics.csv`
- `reports/plots/sheaf_gnn_cycle_vs_accuracy.pdf`

## Feasibility Decision

Official Neural Sheaf Diffusion is feasible only after installing the PyG stack specified by `environment_cpu.yml` or `environment_gpu.yml`.  That dependency work is intentionally out of scope here.

The smallest local experiment is feasible as an Apache-clean, self-contained PyTorch implementation inspired by the official bundle-sheaf idea.  Any result from that run should be labeled as a synthetic diagnostic smoke test, not as an official NSD result and not as evidence that twisted sheaf regularization improves GNNs in general.
