"""Factual artifacts from CPU measurements and legacy source rows."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import statistics

from .protocol import file_hash
from .provenance import write_once

SCHEMA = "run_id phase source_commit config_sha256 group_id base_model_revision training_seed task_subset task_id adapter_rank output_rank method_id policy_id natural_or_controlled gauge_family gauge_seed complex_id coefficient_model certificate_kind structural_status selection_status selected_action fallback_action n_train_groups n_test_examples raw_task_metric bounded_loss delta_vs_fallback delta_vs_validation fit_seconds certificate_seconds merge_seconds selection_seconds total_seconds gpu_peak_allocated_bytes gpu_peak_reserved_bytes cpu_peak_rss_bytes parameter_count inference_multiplier status failure_reason raw_artifact_path".split()


def read(out, name):
    return json.loads((out / name).read_text())


def csv_write(path, rows, fields):
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def tex_write(path, columns, rows):
    def safe(value):
        return str(value).replace("\\", "\\textbackslash{}").replace("_", "\\_").replace("%", "\\%").replace("&", "\\&")
    lines = ["% Generated from measured records; no planned neural values.", "\\begin{tabular}{" + "l" * len(columns) + "}", "\\hline",
             " & ".join(map(safe, columns)) + " \\\\", "\\hline"]
    lines += [" & ".join(safe(v) for v in row) + " \\\\" for row in rows]
    lines += ["\\hline", "\\end{tabular}"]
    write_once(path, "\n".join(lines) + "\n")


def report(root, out):
    root, out = Path(root), Path(out)
    provenance = read(out, "provenance.json")
    inventory = read(out, "inventory.json")
    algebra, baselines = read(out, "algebraic_checks.json"), read(out, "baseline_parity.json")
    shapes = read(out, "cpu_shape_profile.json")
    readiness = read(out, "pilot/readiness.json")
    cost = read(out, "verification_cost.json")
    run_id, source = out.name, provenance["source_commit"]
    tables = out / "tables"
    tables.mkdir(exist_ok=True)
    lines = ["# Evidence inventory", "", "| Item | Existing artifact | Provenance complete? | Can reuse? | Action needed |",
             "| --- | --- | --- | --- | --- |"]
    for row in inventory["items"]:
        paths = "; ".join(a["path"] for a in row["existing_artifact"] if a.get("exists"))
        lines.append(f"| {row['item']} | {paths} | {row['provenance_complete']} | {row['can_reuse']} | {row['action_needed']} |")
    lines += ["", "Detailed raw-row indices, commands, experiment execution commits, artifact commits and hashes are in `inventory.json`.",
              "Dirty historical execution is disclosed. A later artifact commit does not reconstruct every dirty execution source.",
              "Historical single-layer image-feature results are excluded from the planned language cohort.",
              "", "## Snapshot roles", "", inventory["provenance_resolution"]["conclusion"],
              inventory["provenance_resolution"]["bibliography_limitation"]]
    write_once(out / "inventory.md", "\n".join(lines) + "\n")

    torus = algebra["torus"]
    torus_rows = [(r["construction"]["size"], r["construction"]["trivial_control"], len(r["construction"]["coordinates"]),
                   len(r["construction"]["simplices_by_degree"][1]), len(r["face_cochain"]), r["negative_faces"],
                   r["rank_delta1_f2"], r["rank_augmented_f2"], r["exact_trivial"]) for r in torus]
    tex_write(tables / "controlled_algebra.tex", ["Grid", "Control", "V", "E", "F", "Negative faces", "rank d1", "rank augmented", "Exact trivial"], torus_rows)
    tex_write(tables / "baseline_correctness.tex", ["Check", "Relative error", "Tolerance", "Pass"],
              [(r["name"], f"{r['observed_relative_error']:.3g}", r["threshold"], r["passed"]) for r in baselines["checks"]])

    # Neural schema tables contain no invented metric for the blocked campaign.
    csv_write(out / "per_episode.csv", [], SCHEMA)
    csv_write(out / "per_task.csv", [], SCHEMA)
    controls = read(out, "gate_controls.json")
    gate_rows = [{**row, "run_id": run_id, "source_commit": source, "status": "controlled_algebraic",
                  "natural_or_controlled": "controlled", "raw_artifact_path": "gate_controls.json"} for row in controls]
    csv_write(out / "gate_decisions.csv", gate_rows, SCHEMA + ["control_id", "removed_gates", "admitted_candidate_count", "fixture_validation_losses"])
    failure_rows = [{"run_id": run_id, "phase": "blocked", "source_commit": source, "status": "blocked",
                     "failure_reason": reason, "raw_artifact_path": "pilot/readiness.json"} for reason in readiness["blockers"]]
    for row in baselines["stress_panel"]:
        if row.get("status") == "failed" or not row.get("passed", row.get("primary_tolerance_passed", True)):
            failure_rows.append({"run_id": run_id, "phase": "controlled_algebraic", "source_commit": source,
                                 "status": "stress_failure_retained", "failure_reason": json.dumps(row, sort_keys=True),
                                 "raw_artifact_path": "baseline_parity.json"})

    all_timings = [("baseline_parity.json", row) for row in baselines.get("cpu_timings", {}).get("rows", [])]
    all_timings += [("cpu_shape_profile.json", row) for row in shapes.get("rows", [])]
    cost_rows = []
    for artifact, row in all_timings:
        if row.get("status") in ("failed", "timeout"):
            failure_rows.append({"run_id": run_id, "phase": "controlled_algebraic", "source_commit": source,
                                 "method_id": row["method"], "status": row["status"], "failure_reason": row.get("error") or row.get("reason") or row.get("stderr") or "unknown worker failure",
                                 "raw_artifact_path": artifact})
        cost_rows.append({"run_id": run_id, "phase": "controlled_algebraic", "source_commit": source,
                          "method_id": row["method"], "natural_or_controlled": "controlled", "status": row["status"],
                          "merge_seconds": row.get("merge_seconds"), "total_seconds": row.get("full_process_wall_seconds"),
                          "cpu_peak_rss_bytes": row.get("peak_process_rss_bytes"), "parameter_count": row.get("output_parameters"),
                          "output_rank": row.get("output_rank"), "raw_artifact_path": artifact,
                          "shape": "x".join(map(str, row.get("ambient_shape", []))), "repeat": row["repeat"],
                          "warmup_seconds": row.get("warmup_seconds"), "fixture_seconds": row.get("fixture_seconds"),
                          "cost_scope": "CPU layer merge fixture; not complete model/policy/inference cost"})
    csv_write(out / "costs.csv", cost_rows, SCHEMA + ["shape", "repeat", "warmup_seconds", "fixture_seconds", "cost_scope"])
    csv_write(out / "failures.csv", failure_rows, SCHEMA)
    profile_groups = {}
    for row in cost_rows:
        if row["status"] == "controlled_algebraic" and row["raw_artifact_path"] == "cpu_shape_profile.json":
            profile_groups.setdefault((row["shape"], row["method_id"]), []).append(row)
    cost_table = [(shape, method, len(rows), f"{statistics.median(r['merge_seconds'] for r in rows):.6f}",
                   f"{statistics.median(r['total_seconds'] for r in rows):.3f}", max(r["cpu_peak_rss_bytes"] for r in rows),
                   rows[0]["output_rank"], rows[0]["parameter_count"])
                  for (shape, method), rows in sorted(profile_groups.items())]
    tex_write(tables / "cpu_cost.tex", ["Shape", "Method", "Repeats", "Median merge seconds", "Median process seconds", "Peak RSS bytes", "Output rank", "Layer adapter parameters"], cost_table)

    historical = {r["claim_id"]: r for r in inventory["historical"]}
    selector = historical["legacy_matched_selector_attribution"]
    diagnostic = historical["legacy_natural_diagnostic_prediction"]
    lora = historical["legacy_single_feature_layer_lora"]
    tex_write(tables / "legacy_reproduced.tex", ["Legacy quantity", "Value", "Population"], [
        ("Cycle held-out R2", diagnostic["recomputation"]["cycle_residual"]["heldout_r2"], "120 dependent checkpoint collections"),
        ("Pairwise held-out R2", diagnostic["recomputation"]["pairwise_alignment_loss"]["heldout_r2"], "120 dependent checkpoint collections"),
        ("A5-B0 mean accuracy difference", selector["recomputation"]["paired_mean_accuracy_delta"], "10 seed clusters, 40 settings"),
        ("Dense SVD minus global accuracy", lora["recomputation"]["full_delta_svd_mean_accuracy_advantage_over_global"], "5 single-layer adapter groups")])
    write_once(out / "claim_ledger.json", {"run_id": run_id, "source_commit": source,
        "new_code_hashes": provenance["code_hashes"], "historical_numbers": inventory["historical"],
        "new_measured": {"status": "controlled_algebraic", "algebra": "algebraic_checks.json", "baselines": "baseline_parity.json", "cpu_cost": "cpu_shape_profile.json"},
        "unmeasured": {"status": "blocked", "modern_neural_efficacy": None, "modern_policy_utility": None,
                       "neural_complete_cost": None, "source": "pilot/readiness.json"}})
    write_once(out / "claims.md", """# Supported claims and limitations

New outputs are controlled algebra and CPU systems checks. Stored-edge torus
examples supply exact F2 certificates, and simply-connected construction
controls support the no-go under the stated constant-edge model. No natural
neural obstruction is asserted. The vector Hodge identities use the same
complex and inner product; mu2 is checked over F2. Other mu_n are inapplicable.

Compact weighted-sum QR/core-SVD matches the dense reference in the tested
float64/complex128 cases. Dispatch shapes and Torch profiler records support
no ambient-shaped Torch output in the efficient regime. Opaque LAPACK
workspace is outside that shape claim; saturated cores lose the memory claim.
Ill-conditioned failures are retained separately. Exact rank-sum and fixed-rank
outputs have different capacity. CPU measurements use generated factors on
verified Qwen layer shapes, not trained language adapters. Adapter parameter
counts are not full-model counts or inference costs.

PEFT TIES/DARE kernels are applied to effective updates. Kernel parity is not
an official end-to-end TIES, KnOTS, or Core Space campaign. Those important
comparators remain unavailable for model evaluation. P3G/vector-P4 rule
agreement is tested; fixture validation losses are invented unit inputs and
never reported as task results. Actual control residuals are recorded in
gate_ablation_controls.json. Neural policy ablations and sequential savings
are blocked. Historical figures preserve dirty-execution and population
limitations. The old pilot-plus-extension LoRA cohort is not a new confirmatory
study, and scrambles do not add independent groups.

The package does not support a practical-method efficacy claim for KBS.
A narrower coefficient-disciplined diagnostic/algebra framing is consistent
with these checks and retained negative legacy evidence; publication fitness
still depends on novelty and the reviewers' full concerns.
""")
    reviewer = [
        ("Effective update/scaling baseline correctness", "addressed", "baseline_parity.json; profile dispatch and measured identities", "limited to tested algebra fixtures"),
        ("Edge-realizable finite certificate and simply-connected no-go", "addressed", "algebraic_checks.json", "constant-edge model only"),
        ("Coefficient/Hodge/exact-versus-approximate discipline", "addressed", "algebraic_checks.json; protocol/policy tests", "general mu_n unimplemented"),
        ("Historical attribution and provenance", "partially addressed", "inventory.json; claim_ledger.json", "dirty source snapshots and missing historical checkpoints disclosed"),
        ("Strong external literature comparators", "partially addressed", "inventory.json external pinned sources", "official model runs/parity integrations blocked"),
        ("Useful structural decision advantage", "negative result", "legacy selector and diagnostic source rows", "legacy image studies only; no new neural evidence"),
        ("Modern multilayer language experiment", "blocked", "pilot/readiness.json", "model/cohort/backend/splits/parity/resources unresolved"),
        ("Complete neural policy cost and sequential savings", "blocked", "costs.csv CPU layer fixtures only", "model/policy/inference costs unmeasured"),
        ("Neural gate ablations and confidence calibration", "blocked", "gate_ablation_controls.json controlled only", "no independent language episodes or calibrated thresholds"),
        ("Reproducibility package", "partially addressed", "manifest.csv; environment.lock.txt; source/tests", "optional sanitized export; no neural resume backend")]
    csv_write(out / "reviewer_response_evidence.csv", [{"criticism": c, "status": s, "evidence": e, "limitation": l} for c, s, e, l in reviewer], ["criticism", "status", "evidence", "limitation"])
    write_once(out / "reviewer_response_evidence.md", "# Reviewer evidence\n\n| Criticism | Status | Evidence | Remaining boundary |\n| --- | --- | --- | --- |\n" + "\n".join(f"| {c} | {s} | {e} | {l} |" for c, s, e, l in reviewer) + "\n")
    write_once(out / "next_steps.md", """# Remaining blockers

1. Implement and test the real language training/evaluation backend and
   matching-artifact resume, including model, selection and inference costs.
2. Obtain compatible pinned trained language adapters and local base weights.
   Official Llama-3-8B is gated and its terms were not accepted. Public adapter
   cards lack license/seed/head details; one cohort cannot supply five groups.
3. On a suitable available accelerator, measure a small pilot before assigning
   training settings, dtype tolerances or a confirmatory compute estimate. This
   Mac has 8 GB shared RAM; the recorded free-memory snapshot and unavailable
   cohort do not support the proposed full campaign. No cloud price is quoted
   because no provider or pilot throughput was selected or measured. The
   initial campaign cap remains 48 accelerator-hours, not an estimate.
4. Create and hash actual disjoint split/example manifests; implement near-
   duplicate checks, fixed prompts/scoring and justified confidence assumptions.
5. Integrate pinned KnOTS and Core Space with parity on the same checkpoints;
   freeze all primary baselines, gates, finite candidate families and endpoints.
6. Restore the disconnected historical data drive only if checkpoint-level
   legacy reruns are required. Historical CSV reanalysis already succeeds.

Current evidence supports coefficient-disciplined diagnostics and algebraic
verification, not a new practical-method superiority claim for KBS. Do not
start another broad holonomy search to replace these missing prerequisites.
""")
    write_once(out / "execution_summary.md", f"""# KBS experiment execution

Run `{run_id}`, code source `{source}`.

Executed exact stored-edge torus/no-go/Hodge certificates, {len(baselines['checks'])}
primary baseline checks, PEFT effective-update kernel parity, measured gate
controls, {len(all_timings)} isolated CPU timing repeats, and historical CSV
reanalysis. Verification/profile wall time: {cost['wall_seconds']:.2f} seconds;
accelerator compute: 0 hours. Inventory, tests and report generation are
additional CPU work recorded in RUN_LOG.md and test_results.txt.

The language pilot is blocked; specialist training, final test evaluation,
modern policy utility, risk-coverage and full-model cost did not run.
`per_episode.csv` and `per_task.csv` intentionally contain headers only.
No new neural superiority or natural central-class claim is supported.
No manuscript edit, model upload, paid service or remote push was performed.

Reproduce with configs/kbs_revision/README.md. Source rows, hashes, commands,
snapshot roles and limitations appear in inventory.json and claim_ledger.json.
""")
    # Plot measured CPU stage costs only. No risk-coverage or performance-cost
    # plot can be drawn without held-out model outcomes.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure_dir = out / "figures"
    figure_dir.mkdir(exist_ok=True)
    fig, axis = plt.subplots(figsize=(10, 5))
    labels, cold, warm, merged = [], [], [], []
    for (shape, method), rows in sorted(profile_groups.items()):
        labels.append(f"{shape}\n{method}")
        merged.append(statistics.median(r["merge_seconds"] for r in rows))
        warm.append(statistics.median(r.get("warmup_seconds") or 0 for r in rows))
        cold.append(statistics.median(r["total_seconds"] - r["merge_seconds"] - (r.get("warmup_seconds") or 0) for r in rows))
    axis.bar(range(len(labels)), cold, label="process/import/fixture overhead")
    axis.bar(range(len(labels)), warm, bottom=cold, label="warmup")
    axis.bar(range(len(labels)), merged, bottom=[a+b for a,b in zip(cold,warm)], label="measured merge")
    axis.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    axis.set_ylabel("CPU wall seconds, median of three isolated repeats")
    axis.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "cpu_stage_cost.png", dpi=140)
    plt.close(fig)
    write_once(figure_dir / "captions.md", "CPU stage costs: random-factor fixtures on pinned Qwen architecture shapes, three isolated repeats per shape/method, medians with no inferential CI; zero independent neural training groups. Includes cold process and warmup costs. Performance-cost, risk-coverage and new paired-policy plots are blocked because no held-out language outcomes exist.\n")
