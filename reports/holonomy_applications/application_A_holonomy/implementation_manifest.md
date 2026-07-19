# Application A Implementation Manifest

- Implementation: `experiments/holonomy_application_A.py`
- Structural library: `src/holonomy_application_transitions.py`
- Focused tests: `tests/test_holonomy_application_transitions.py`
- Shared-corpus tests: `tests/test_holonomy_application_corpus.py`
- Configuration: `reports/holonomy_applications/application_A_holonomy/config.json`
- Smoke command: `/Users/tinggong/Documents/GitHub/TwistedMerge/.venv/bin/python experiments/holonomy_application_A.py --mode smoke --data-dir /Users/tinggong/Documents/GitHub/TwistedMerge/data`
- Confirmatory command: `/Users/tinggong/Documents/GitHub/TwistedMerge/.venv/bin/python experiments/holonomy_application_A.py --mode confirmatory --data-dir /Users/tinggong/Documents/GitHub/TwistedMerge/data`
- Confirmatory result: negative; no additional chart family or dataset opened.

The implementation loads and verifies the shared checkpoints, reconstructs four transition families, saves all candidate logits before label access, and does not retrain chart adapters.
