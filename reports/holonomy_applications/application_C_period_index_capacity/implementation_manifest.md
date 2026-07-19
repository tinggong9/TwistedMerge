# Application C Implementation Manifest

- Implementation: `experiments/holonomy_application_C.py`
- Controlled carrier library: `src/holonomy_period_index_capacity.py`
- Focused tests: `tests/test_holonomy_period_index_capacity.py`
- Configuration: `reports/holonomy_applications/application_C_period_index_capacity/config.json`
- Smoke command: `/Users/tinggong/Documents/GitHub/TwistedMerge/.venv/bin/python experiments/holonomy_application_C.py --mode smoke`
- Confirmatory command: `/Users/tinggong/Documents/GitHub/TwistedMerge/.venv/bin/python experiments/holonomy_application_C.py --mode confirmatory`
- Confirmatory result: controlled structural threshold supported; practical-superiority gate negative.

The finite-Heisenberg layer is explicitly labeled `controlled_on_real_features`. It consumes actual saved local-model logits and does not retrain the shared adapters.
