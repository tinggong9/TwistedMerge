"""Evidence-qualified report for backend validation with a blocked real pilot."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from .external_methods import external_method_definitions
from .provenance import write_once
from .reporting import SCHEMA


def table(path, rows, fields):
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def report(root, out):
    root, out = Path(root), Path(out)
    software = json.loads((out / "software_integration.json").read_text())
    smoke = json.loads((out / "backend_smoke.json").read_text())
    parity = json.loads((out / "external_method_parity.json").read_text())
    initial = json.loads((out / "initial_environment.json").read_text())
    provenance = json.loads((out / "provenance.json").read_text())
    blocked = json.loads((out / "pilot/readiness.json").read_text())
    episode = software["episodes"][0]["result"]
    common = {"run_id": out.name, "source_commit": provenance["source_commit"]}
    base = "https://huggingface.co/Qwen/Qwen2.5-1.5B/tree/8faed761d45a263340a0528343f099c05c9a4323"
    write_once(out / "hardware_and_license_audit.md", f"""# Hardware and license audit

Initial total RAM: {initial['ram_total_bytes']} bytes; available RAM snapshot:
{initial['ram_available_bytes']} bytes. Accelerator inventory is recorded in
initial_environment.json; CUDA is unavailable and MPS shares system memory.
This execution used CPU only. The user reported no known GPU/checkpoint host
and requested conservation of resources. No paid compute, weights download,
license acceptance, remote upload, or push was performed.

The pinned [Qwen file listing]({base}) displays model.safetensors as 3.09 GB;
its [model card]({base.replace('/tree/', '/blob/')}/README.md) records Apache-2.0,
1.54B parameters and 28 layers. A download exceeds both the previous 2 GB
ceiling and this run's stricter 200 MB allowance, before data or checkpoints.
The local free-memory snapshot also falls below weight storage alone. These
are admission blockers, not measured full-model OOM or training benchmarks.

KnOTS tensor source is MIT; Core Space is Apache-2.0, with exact license text
and source pins in configs/kbs_revision/external_parity_sources.json. The
reference harness runs reviewed, hash-locked tensor excerpts offline, without
external model loaders. Generated fixture weights are marked CC0/software-only.
Real dataset and adapter licensing remains unresolved because no cohort was
downloaded or supplied. No gated or proprietary tensors are in the review bundle.
""")
    write_once(out / "readiness_diagnosis.md", """# Readiness diagnosis

| Dependency | Current state | Smallest remaining requirement |
| --- | --- | --- |
| A. Base/cohort/licensing | blocked for real Qwen; generated software cohort works | compatible local pinned Qwen weights/tokenizer, four task adapters or an approved fixed training recipe, reviewed real dataset/adapter licenses |
| B. Backend/baselines | software integration addressed | target-hardware dtype and performance validation; official end-to-end launchers are not claimed |
| C. Splits/statistics | seven actual software JSONL roles pass recomputed leakage audit; real roles unavailable | local pinned benchmark records and final frozen statistical protocol; no noninferiority claim without justification |
| D. Resources | real pilot infeasible under current local availability and cap | suitable already-authorized GPU/memory and storage; user knows of no such host |

The missing backend is now implemented and exercised, rather than retained as
a placeholder. The software cohort is not a substitute Qwen research cohort.
No final benchmark labels were loaded/evaluated. No research GO or frozen
confirmatory manifest is emitted. Resource checks are cooperative between
operations; native kernels are not preempted, and budget receipts account for
each observed invocation rather than unobserved interrupted-process work.

No model-size/dataset substitution or protocol revision was made. Pilot
throughput, confirmatory time, GPU memory peaks and cloud prices remain unknown.
""")
    definitions = external_method_definitions()
    write_once(out / "baseline_parity_neural.json", {"status": "software_only", "real_qwen_status": "not_run",
        "source_commit": provenance["source_commit"], "source_hashes": provenance["code_hashes"],
        "trained_fixture_target_modules": smoke["multilayer_target_count"],
        "safe_reload_logits_max_absolute_error": smoke["safe_reload_logits_max_absolute_error"],
        "compact_dense_effective_delta_max_absolute_error": smoke["compact_dense_effective_delta_max_absolute_error"],
        "same_seed_factor_max_absolute_error": smoke["same_seed_factor_max_absolute_error"],
        "shared_head_preserved": smoke["shared_head_preserved"], "methods": episode["candidate_metadata"],
        "external_fixture_parity": parity, "external_definitions": definitions,
        "scope": "trained random two-block Qwen interface/forward tests; not modern-scale efficacy"})
    write_once(out / "external_method_parity.md", f"""# External comparator fidelity

{len(parity['checks'])} primary small-fixture checks passed. Exact reviewed
upstream tensor excerpts are executed with a hash-locked manifest, independently
of the adaptations. Full upstream repositories, model loaders and pickle
weights are not executed. Source/license/hash details are recorded in
external_method_parity.json and configs/kbs_revision/external_parity_sources.json.

KnOTS applies pinned dense effective-update SVD with upstream cutoff/promotion,
then globally flattened multilayer TIES in aligned coordinates. Core Space
uses separate factor bases and nonlinear core-coordinate merging. It is not
the compact weighted-sum SVD operator. Heterogeneous weights require the named
corrected_per_task adaptation; official parity covers common/equal weights.
Fixed output-rank compression is an added adaptation and is labelled explicitly.

Conditioning stress records, including failures, remain in the parity JSON.
KnOTS inherits upstream float32 precision and cannot meet a universal strict
1e-10 claim even in ordinary conditions; primary parity tolerances account for
source precision. No official end-to-end neural benchmark result is claimed.
""")
    blocked_row = {**common, "phase": "blocked", "status": "blocked", "raw_artifact_path": "pilot/readiness.json",
                   "failure_reason": "Real Qwen pilot unavailable under current model/cohort/split/resource constraints"}
    for name in ("pilot_results.csv", "pilot_costs.csv", "per_episode.csv", "per_task.csv", "gate_decisions.csv", "gate_ablations.csv", "costs.csv"):
        table(out / name, [blocked_row], SCHEMA)
    failures = [{**blocked_row, "failure_reason": reason} for reason in blocked["blockers"]]
    failures += [{**common, "phase": "controlled_algebraic", "status": "stress_failure_retained",
                 "failure_reason": json.dumps(row, sort_keys=True), "raw_artifact_path": "external_method_parity.json"}
                for row in parity.get("stress_panel", []) if row.get("status") == "failed" or row.get("passed") is False]
    table(out / "failures.csv", failures, SCHEMA)
    rows = []
    for method, metric in episode["candidate_metrics"].items():
        for row in metric["rows"]:
            rows.append({**common, **row, "phase": "software_only", "status": "software_only",
                         "method_id": method, "natural_or_controlled": "software_fixture",
                         "raw_artifact_path": "software_integration.json"})
    table(out / "software_task_metrics.csv", rows, SCHEMA + ["prediction", "correct"])
    stages = [{**common, **row, "phase": "software_only", "status": "software_only",
               "total_seconds": row["seconds"], "raw_artifact_path": "software_integration.json"} for row in episode["stage_costs"]]
    stages += [{**common, **row, "phase": "software_only", "status": "software_only", "total_seconds": row["seconds"]}
               for row in software["training_stage_costs"]]
    table(out / "software_stage_costs.csv", stages, SCHEMA + ["stage", "device", "role"])
    write_once(out / "pilot_decision.md", f"""# STOP: current practical pilot is resource-blocked

No real Qwen pilot ran. No research GO, REDESIGN based on measured real outcomes,
noninferiority claim or confirmatory freeze is justified. The user supplied no
GPU host and requested limited resource use. Preserve the planned backbone and
task families rather than replacing them with the generated smoke model.

The software integration ran in {software['wall_seconds']:.2f} wall seconds,
with zero accelerator hours, four generated task specialists, 11 candidate
operators and five action rules. It reuses completed episodes after checksum
verification without duplicate inference. All action rules selected the compact
fallback in this fixture. Those invented tasks do not supply research pilot
numbers or a negative modern-model utility result.
""")
    write_once(out / "primary_policy_summary.md", """# Primary policy status

P4-P0 and P4-P1 on real original Qwen adapters: blocked. Software action
selection is tested and is frozen before candidate evaluation. P3 is routed
to the fully gated Hodge admission rule; P3/P4 share the same information.
No neural coefficient/confidence model is independently justified here, so the
structural branch is inapplicable. The timed structural_applicability stage
is an abstention check, not Hodge projection or certificate computation.

Policy costs are attributed from a shared matched-candidate panel. They count
common factor preparation, candidate generation, validation scans, applicable
diagnostic checks and rule time, with loading and inference separately exposed.
They are not measured standalone-policy latency or sequential-generation savings.
Actual factor alignment fits remain inside generation and are not counted twice.
Full parameter counts and exact rank-sum capacity are retained; inference latency
multipliers remain unmeasured. Compressed Task Arithmetic duplicates compact SVD
under matched weights/rank and is retained as correctness reference.
""")
    for name in ("risk_coverage.md", "confidence_and_power.md"):
        write_once(out / name, "# Not run for the research population\n\nNo real independently trained Qwen groups or held-out outcomes exist. Software rows do not support coverage/risk bounds, variance estimates, power, significance or noninferiority. No zero-event confidence bound is calculated from correlated or generated fixtures. A real 1-3 group pilot would itself provide fragile uncertainty, not publication-grade inference.\n")
    write_once(out / "claims.md", """# Evidence boundaries

Supported software claims: safe local Qwen/PEFT loading, multilayer LoRA-only
training from a fixed frozen base, native four-task prompts/scoring, immutable
split/leakage checks, saved-weight reload, actual scored candidate inference,
atomic episode completion and checksum/source-matched resume. KnOTS/Core Space
are parity-tested tensor adaptations, with conditioning limitations retained.

Blocked research claims: modern Qwen efficacy, useful incremental certification,
complete standalone policy resource advantages, neural Hodge/certificate gates,
natural central classes, risk control, power or confirmatory significance.

Historical negatives remain unchanged: cycle held-out R2=-0.205534; A5-B0
accuracy delta=-0.001865; single-layer dense SVD minus global synchronization
accuracy=+0.051725. Their experiment-specific provenance remains in the previous
run's inventory/claim ledger. Source 625a15d generated those reports; dd0aac5
was the later evidence commit. This run does not reclassify them as new evidence.
""")
    write_once(out / "paper_update_recommendations.md", "# Factual recommendations only\n\nThe new backend and comparator parity may be described as reproducibility/software work. No modern model-merging results table, decision-utility superiority, natural-cohomology claim or significance statement should be added. Retain the original controlled torus/no-go/Hodge mathematics and historical negative neural evidence with their original scope. A practical-method KBS claim remains unsupported; a narrower diagnostic framing fits the available evidence. Manuscript prose, figures and LaTeX were not edited.\n")
    write_once(out / "reviewer_response_evidence.md", """# Reviewer evidence

| Concern | Status | Evidence |
| --- | --- | --- |
| Missing actual language backend | addressed at software level | backend_smoke.json; software_integration.json; loader/training/forward tests |
| Seven real data roles and leakage machinery | partially addressed | software_only_splits; task_data tests; real benchmark sources blocked |
| Efficient/strong comparator definitions | partially addressed | baseline_parity_neural.json; pinned tensor parity; target-scale runs blocked |
| Useful incremental policy action | blocked | pilot_decision.md; no real held-out population |
| Complete policy cost | partially addressed | software_stage_costs.csv; attribution scope; real standalone costs blocked |
| Honest provenance/negative evidence | addressed | provenance.json; manifest.csv; old immutable ledger retained |
""")
    tables = out / "tables"
    tables.mkdir(exist_ok=True)
    table(tables / "software_capacity.csv", [{"method": method, "rank_by_module": json.dumps(value["output_ranks"], sort_keys=True),
          "adapter_parameters": value["output_parameters"], "full_model_parameters": value["full_model_parameter_count"],
          "status": "software_only"} for method, value in episode["candidate_metadata"].items()],
          ["method", "rank_by_module", "adapter_parameters", "full_model_parameters", "status"])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    totals = {}
    for row in stages:
        totals[row["stage"]] = totals.get(row["stage"], 0) + row["seconds"]
    figures = out / "figures"
    figures.mkdir(exist_ok=True)
    fig, axis = plt.subplots(figsize=(10, 4))
    axis.bar(list(totals), list(totals.values()))
    axis.set_ylabel("Measured CPU wall seconds")
    axis.tick_params(axis="x", rotation=35)
    for label in axis.get_xticklabels():
        label.set_ha("right")
    fig.tight_layout()
    fig.savefig(figures / "software_stage_costs.png", dpi=140)
    plt.close(fig)
    write_once(figures / "captions.md", "Software-only generated two-block Qwen cohort: one integration episode, four invented task schemas, zero independent research groups; no uncertainty interval. Stage sums include all candidate/reference work. Real performance-cost, risk-coverage and paired-policy research figures are blocked.\n")
    write_once(out / "execution_summary.md", f"""# Backend execution summary

Code source: `{provenance['source_commit']}`. The real local-only language backend,
native task scoring, split audits and atomic resume are implemented. The CPU
integration executed {smoke['multilayer_target_count']} LoRA modules across two
blocks, four generated specialists, 11 merge methods and five policy rules;
completed episodes were reused without duplication. Same-seed and reload errors
were {smoke['same_seed_factor_max_absolute_error']} and {smoke['safe_reload_logits_max_absolute_error']}.

Software integration wall time: {software['wall_seconds']:.2f} seconds; accelerator
compute: 0 hours. Tests and preparation are additional CPU work recorded in logs.
No large weights/datasets were downloaded. The real Qwen pilot and confirmatory
campaign remain blocked by resources/cohort/data availability. No new practical
neural advantage or natural central-class conclusion is supported. Changes are
local; previous artifacts and the manuscript are preserved.
""")
