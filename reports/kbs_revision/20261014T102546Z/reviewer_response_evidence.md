# Reviewer evidence

| Concern | Status | Evidence |
| --- | --- | --- |
| Missing actual language backend | addressed at software level | backend_smoke.json; software_integration.json; loader/training/forward tests |
| Seven real data roles and leakage machinery | partially addressed | software_only_splits; task_data tests; real benchmark sources blocked |
| Efficient/strong comparator definitions | partially addressed | baseline_parity_neural.json; pinned tensor parity; target-scale runs blocked |
| Useful incremental policy action | blocked | pilot_decision.md; no real held-out population |
| Complete policy cost | partially addressed | software_stage_costs.csv; attribution scope; real standalone costs blocked |
| Honest provenance/negative evidence | addressed | provenance.json; manifest.csv; old immutable ledger retained |
