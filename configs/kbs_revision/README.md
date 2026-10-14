# KBS revision execution

This package executes finite algebra, effective-update baseline correctness,
architecture-shaped CPU profiling, historical CSV reanalysis, safe local Qwen/
PEFT training and scored evaluation, split/leakage auditing, and atomic resume.
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

Those two commands require a real pilot GO, validated manifests, compatible
trained cohort, source-matched freeze, explicit final-test access and suitable
resources. The local-only backend now exists; the current real cohort/resources
remain unavailable. Training-group inference, practical
noninferiority, policy performance, risk-coverage, and sequential cost savings
remain unmeasured. Policy unit fixtures test rules only. P3G and vector P4 use
the same information and gates and are expected to agree. Their tolerances
are fixture values and are not calibrated deployment thresholds.

The resource-constrained next-step execution uses `next_pilot.yaml` and a fresh
report directory. Its generated two-block fixture is software-only, not an
alternative research backbone:

```sh
.venv/bin/python experiments/kbs_revision.py backend-smoke --out <fresh-run>
.venv/bin/python experiments/kbs_revision.py pilot --config configs/kbs_revision/next_pilot.yaml --out <fresh-run>
.venv/bin/python experiments/kbs_revision.py report --out <fresh-run>
```

`backend-smoke` trains/safely reloads real multilayer PEFT factors, prepares all
seven JSONL roles, scores all four native task formats, exercises eleven merge
methods and five policies, then validates completed-episode resume. It downloads
no weights/data and uses CPU. Real benchmark CSVs contain explicit blocked rows;
software metrics/stages are saved separately. KnOTS/Core Space are reviewed
tensor adaptations with pinned reference excerpts and independent fixture parity,
not official end-to-end benchmark runs. Policy cost totals are attributed from
a shared matched-candidate panel; inference latency multipliers are unmeasured.

For supplied local data and a resolved, reviewed real protocol:

```sh
.venv/bin/python experiments/kbs_revision.py prepare-splits --config <data-plan.yaml> --out <fresh-run>
.venv/bin/python experiments/kbs_revision.py train-cohort --config <training-plan.yaml> --out <fresh-run>
```

The data plan pins a JSONL `dataset_records_path`/SHA256, per-task
`source_metadata`, `split_seed` and counts/ratios. The training plan provides
the base directory/manifest, split manifest, tasks, explicit group/task seeds
and fixed training settings. Loader manifests record all config/tokenizer/
safetensors hashes, reviewed licenses, shared frozen causal head and Qwen pin.
The episode protocol records adapters, tasks, output rank, explicit merge
weights, method density/seed/weight semantics, candidate/policy family and
resource ceilings. Paths resolve relative to the repository. Standard factors
are merged on CPU float64; model inference uses the explicitly chosen device.
Cooperative budget checks run between operations; a native kernel itself is
not preempted and cross-invocation receipts do not reconstruct an interrupted
process's unobserved usage.
