# Selector-Attribution Verification

- Focused command: `python -m pytest -q tests/test_post_iclr_selector_attribution.py tests/test_post_iclr_v2_current_evidence_audit.py`
- Focused result: `7 passed in 1.36s`.
- Smoke stage: 1 exact setting completed; zero candidate failures.
- Pilot stage: 4 exact settings completed; zero candidate failures; A4 threshold frozen from diagnostics only.
- Confirmatory stage: 10 independent training groups and 40 exact settings completed; both official candidates present in every setting; zero candidate failures.
- Artifact checksum verification: 56 phase artifacts verified against `artifact_manifest.csv`.
- Checkpoint checksum verification: all 140 confirmatory checkpoints verified against `checkpoint_manifest.csv`.

The repository-wide `python -m pytest -q` check was also attempted. It reached `tests/test_block_gauge_branch_closure.py::BlockGaugeBranchClosureTests::test_smoke_closure_outputs_and_threshold_columns`, whose unrelated `experiments/block_gauge_phase_diagram.py` subprocess remained idle in a socket connection with no output or generated files for more than six minutes. The stalled test process was terminated; it did not emit an assertion failure. No unrelated source was changed.
