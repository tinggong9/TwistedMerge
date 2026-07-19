# Application B Implementation Manifest

- Implementation: `experiments/holonomy_application_B.py`
- Certificate library: `src/holonomy_brauer_certificate.py`
- Transition library reused from A: `src/holonomy_application_transitions.py`
- Focused tests: `tests/test_holonomy_brauer_certificate.py`
- Configuration: `reports/holonomy_applications/application_B_brauer_certificate/config.json`
- Smoke command: `/Users/tinggong/Documents/GitHub/TwistedMerge/.venv/bin/python experiments/holonomy_application_B.py --mode smoke`
- Confirmatory command: `/Users/tinggong/Documents/GitHub/TwistedMerge/.venv/bin/python experiments/holonomy_application_B.py --mode confirmatory`
- Confirmatory result: no natural Brauer-like candidate certified.

This phase reconstructs the exact deterministic A transitions, accesses no classification test labels, and trains no model.
