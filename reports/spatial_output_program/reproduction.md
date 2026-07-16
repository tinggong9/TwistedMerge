# Spatial-output program reproduction

```bash
PYTHONPYCACHEPREFIX=/private/tmp/codex-pycache .venv/bin/python experiments/fetch_kvasir_subset.py
PYTHONPYCACHEPREFIX=/private/tmp/codex-pycache .venv/bin/python experiments/run_spatial_output_program.py --tier all --force
```

The runner writes stage state, exact commands, failures, tests, manifests, and checksums under `reports/spatial_output_program/`.
