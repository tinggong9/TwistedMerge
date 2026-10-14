# KBS revision execution

This package executes finite algebra, effective-update baseline correctness,
architecture-shaped CPU profiling, historical CSV reanalysis, and readiness
auditing. It does not implement modern language-model training or evaluation.
The language configuration is a **blocked planning record**, not a frozen
confirmatory protocol. `freeze` and `run` refuse unmet prerequisites rather
than writing a successful neural result. No paper prose is changed.

Use the existing isolated environment, whose exact packages are captured in
each run's `environment.lock.txt`. The receipt records the actual environment;
the older repository requirements file is not assumed to match it.

```sh
RUN=reports/kbs_revision/$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$RUN"
cp configs/kbs_revision/budget.yaml "$RUN/budget.yaml"
.venv/bin/python experiments/kbs_revision.py inventory --out "$RUN"
.venv/bin/python experiments/kbs_revision.py verify --out "$RUN"
.venv/bin/python experiments/kbs_revision.py pilot --config configs/kbs_revision/pilot.yaml --out "$RUN"
# The blocked pilot returns 2 and preserves its readiness record.
.venv/bin/python experiments/kbs_revision.py report --out "$RUN"
.venv/bin/python experiments/kbs_revision.py pilot --config configs/kbs_revision/pilot.yaml --out "$RUN" --dry-run
```

Each artifact uses exclusive creation. Use a fresh timestamp for a rerun;
existing outputs are never overwritten. `RUN_LOG.md` is an append-only event
log. Run the tests with:

```sh
.venv/bin/python -m pytest -q tests/test_kbs_revision_*.py
```

Language continuation requires an implemented, tested language backend, a
compatible local pinned base and LoRA cohort, reviewed adapter licenses,
immutable split manifests for all seven data roles, exact and near-duplicate
audits, complete baseline definitions and parity, and a measured feasible
pilot. The `pilot.yaml` resource entries are planning caps, not paid-compute
authorization. Do not treat a GPU absence, memory shortage, or a blocked
backend as an experimental neural outcome.

After prerequisites exist, the intended interface is:

```sh
.venv/bin/python experiments/kbs_revision.py freeze --config <resolved-protocol.yaml> --out <fresh-run>
.venv/bin/python experiments/kbs_revision.py run --config <resolved-protocol.yaml> --resume --out <same-run>
```

Those two commands currently fail explicitly because the neural backend is
unimplemented. Resume matching and per-episode completion are remaining work,
not features claimed by this CPU package. Training-group inference, practical
noninferiority, policy performance, risk-coverage, and sequential cost savings
remain unmeasured. Policy unit fixtures test rules only. P3G and vector P4 use
the same information and gates and are expected to agree. Their tolerances
are fixture values and are not calibrated deployment thresholds.
