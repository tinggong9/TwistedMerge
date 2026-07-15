#!/usr/bin/env python3
"""Reaudit historical compression claims from retained executable evidence."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.chart_followup_common import OUT, factual_report, provenance
from experiments.next_program_common import latex_table, write_csv

SCRIPT = Path(__file__).resolve()
DEST = OUT / "compression"
SOURCE = ROOT / "reports" / "next_program" / "iclr" / "compression_runs.csv"
COMMAND = "python experiments/compression_claim_reaudit.py"


def finite(value: str) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def storage_gate(reduction: float, retained_gain: float | None) -> bool:
    return reduction >= 0.25 and retained_gain is not None and retained_gain >= 0.95


def audit_row(row: dict[str, str]) -> tuple[dict[str, object], dict[str, object]]:
    teacher_bytes = int(row["teacher_storage_bytes"])
    executable_bytes = int(row["student_storage_bytes"])
    dense_tensor_bytes = int(row["stored_parameters"]) * 4
    sparse_bytes: int | str = executable_bytes if "pruned" in row["method"] else ""
    quantized_bytes: int | str = executable_bytes if "quantized" in row["method"] else ""
    reduction = 1.0 - executable_bytes / teacher_bytes
    retained = finite(row["retained_teacher_gain_fraction"])
    storage = {
        "setting_id": row["setting_id"],
        "group": row["group"],
        "seed": int(row["seed"]),
        "method": row["method"],
        "target_storage_reduction": float(row["target_storage_reduction"]),
        "teacher_executable_bytes": teacher_bytes,
        "student_dense_tensor_bytes": dense_tensor_bytes,
        "student_sparse_serialized_bytes": sparse_bytes,
        "student_quantized_serialized_bytes": quantized_bytes,
        "student_executable_model_bytes": executable_bytes,
        "actual_storage_reduction": reduction,
        "teacher_accuracy": float(row["teacher_accuracy"]),
        "student_accuracy": float(row["accuracy"]),
        "accuracy_delta": float(row["accuracy"]) - float(row["teacher_accuracy"]),
        "retained_teacher_gain_fraction": "" if retained is None else retained,
        "storage_claim_passed": storage_gate(reduction, retained),
        "teacher_artifact_sha256": row["teacher_artifact_sha256"],
        "student_artifact_sha256": row["student_artifact_sha256"],
        **provenance(SCRIPT, COMMAND, int(row["seed"])),
    }
    # The retained ledger contains a student latency only.  It has neither a
    # matching teacher timing nor the historical teacher executable, so a
    # teacher-to-student latency reduction cannot be reconstructed honestly.
    latency = {
        "setting_id": row["setting_id"],
        "group": row["group"],
        "seed": int(row["seed"]),
        "method": row["method"],
        "batch_size": 128,
        "student_latency_ms": float(row["latency_ms_batch128"]),
        "teacher_latency_ms": "",
        "actual_latency_reduction": "",
        "latency_evaluable": False,
        "latency_claim_passed": False,
        "unevaluable_reason": "matching teacher executable and synchronized teacher timing were not retained",
        **provenance(SCRIPT, COMMAND, int(row["seed"])),
    }
    return storage, latency


def summarize(storage_rows: list[dict[str, object]], latency_rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    grouped: dict[tuple[str, str, float], list[dict[str, object]]] = defaultdict(list)
    for row in storage_rows:
        grouped[(str(row["group"]), str(row["method"]), float(row["target_storage_reduction"]))].append(row)
    claims: list[dict[str, object]] = []
    pareto: list[dict[str, object]] = []
    for (group, method, target), rows in sorted(grouped.items()):
        reductions = [float(row["actual_storage_reduction"]) for row in rows]
        accuracies = [float(row["student_accuracy"]) for row in rows]
        passes = [bool(row["storage_claim_passed"]) for row in rows]
        evaluable_latency = any(
            bool(row["latency_evaluable"])
            for row in latency_rows
            if row["group"] == group and row["method"] == method
        )
        claims.append(
            {
                "group": group,
                "method": method,
                "target_storage_reduction": target,
                "runs": len(rows),
                "median_actual_storage_reduction": statistics.median(reductions),
                "mean_student_accuracy": statistics.mean(accuracies),
                "storage_gate_pass_rate": statistics.mean(passes),
                "storage_claim_confirmed": all(passes),
                "latency_evaluable": evaluable_latency,
                "latency_claim_confirmed": False,
                "pareto_claim_confirmed": False,
                "overall_claim_confirmed": all(passes),
                "claim_basis": "storage_only" if all(passes) else "no_confirmed_compression_claim",
            }
        )
        pareto.append(
            {
                "group": group,
                "method": method,
                "target_storage_reduction": target,
                "mean_student_accuracy": statistics.mean(accuracies),
                "median_executable_bytes": statistics.median(int(row["student_executable_model_bytes"]) for row in rows),
                "median_student_latency_ms": statistics.median(
                    float(row["student_latency_ms"])
                    for row in latency_rows
                    if row["group"] == group and row["method"] == method and float(row.get("student_latency_ms", 0)) > 0
                ),
                "storage_accuracy_pareto_evaluable": True,
                "latency_accuracy_pareto_evaluable": False,
                "latency_accuracy_pareto_optimal": False,
            }
        )
    for group in sorted({str(row["group"]) for row in pareto}):
        block = [row for row in pareto if row["group"] == group]
        for row in block:
            row["storage_accuracy_pareto_optimal"] = not any(
                float(other["mean_student_accuracy"]) >= float(row["mean_student_accuracy"])
                and float(other["median_executable_bytes"]) <= float(row["median_executable_bytes"])
                and (
                    float(other["mean_student_accuracy"]) > float(row["mean_student_accuracy"])
                    or float(other["median_executable_bytes"]) < float(row["median_executable_bytes"])
                )
                for other in block
                if other is not row
            )
    return claims, pareto


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    if not SOURCE.exists():
        raise FileNotFoundError(f"retained compression ledger is missing: {SOURCE}")
    with SOURCE.open(encoding="utf-8", newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    if arguments.smoke:
        source_rows = source_rows[:24]
    storage_rows: list[dict[str, object]] = []
    latency_rows: list[dict[str, object]] = []
    for row in source_rows:
        storage, latency = audit_row(row)
        storage_rows.append(storage)
        latency_rows.append(latency)
    claims, pareto = summarize(storage_rows, latency_rows)
    write_csv(DEST / "storage.csv", storage_rows)
    write_csv(DEST / "latency.csv", latency_rows)
    write_csv(DEST / "pareto.csv", pareto)
    write_csv(DEST / "claims.csv", claims)
    latex_table(
        DEST / "tables" / "compression_audit.tex",
        ["group", "method", "target_storage_reduction", "median_actual_storage_reduction", "storage_claim_confirmed", "latency_evaluable"],
        claims,
        "Compression claim reaudit",
    )
    confirmed = sum(bool(row["overall_claim_confirmed"]) for row in claims)
    factual_report(
        DEST / "report.md",
        "Compression claim reaudit",
        [
            f"Reaudited {len(storage_rows)} retained historical runs and {len(claims)} grouped claims. {confirmed} grouped claims satisfy the preregistered storage-and-retained-gain gate.",
            "Executable student bytes are taken from the retained artifact ledger. Dense tensor bytes are recomputed as stored parameter count times four bytes and are reported separately from sparse or quantized executable bytes.",
            "The historical ledger does not retain a matching teacher executable or synchronized teacher latency. Latency reduction and latency-Pareto claims are therefore marked unevaluable, rather than inferred from student-only timings.",
        ],
    )


if __name__ == "__main__":
    main()
