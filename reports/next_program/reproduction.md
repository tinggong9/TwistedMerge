# Reproduction instructions

Run `PYTHON_BIN=python scripts/reproduce_next_program.sh` from the repository root. Set `TWISTEDMERGE_DATA_ROOT` when datasets are stored outside `data/`. The runner supports `--tier immediate|iclr|extended|all`, `--resume`, and `--force-stage STAGE_ID`. Saved prediction tensors and checkpoints remain under ignored `reports/tmp/next_program/`; their hashes are recorded in committed ledgers.
