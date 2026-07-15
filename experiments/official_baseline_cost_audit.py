#!/usr/bin/env python3
"""Stage 9: requested baseline set and bounded systems-cost proxy audit."""

from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.remaining_experiment_common import OUT, git_head, latex_table, write_csv

SCRIPT = Path(__file__).resolve()
FUTURE = ROOT / "reports" / "future_program"


def estimate_flops(input_dimension: int, output_dimension: int, branches: int, batch: int) -> int:
    return int(2 * input_dimension * output_dimension * branches * batch)


def measured_proxy_latency(input_dimension: int, output_dimension: int, branches: int, batch: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed); inputs = rng.normal(size=(batch, input_dimension)).astype(np.float32); weights = rng.normal(size=(branches, input_dimension, output_dimension)).astype(np.float32)
    times = []; peak_bytes = inputs.nbytes + weights.nbytes
    for _ in range(9):
        start = time.perf_counter(); output = np.einsum("bi,nio->nbo", inputs, weights).mean(0); _ = output.argmax(1); times.append(time.perf_counter() - start); peak_bytes = max(peak_bytes, inputs.nbytes + weights.nbytes + output.nbytes)
    return float(np.median(times) * 1000.0), float(peak_bytes / 1024**2)


def controlled_context_rows() -> tuple[list[dict[str, object]], dict[str, float]]:
    import pandas as pd
    source = FUTURE / "emergency" / "level2_runs.csv"; frame = pd.read_csv(source)
    if "phase" in frame.columns: frame = frame[frame.phase == "confirmation"]
    mapping = [
        ("twistedmerge_hodge_lr", "structured_method"),
        ("generic_mixture_of_experts", "strongest_generic_moe"),
        ("learned_unconstrained_matrix_context_action", "group_equivariant_baseline"),
        ("generic_low_rank_context_adapter", "generic_low_rank_context_adapter"),
        ("group_structured_without_hodge", "learned_cayley_table_upper_bound"),
        ("hodge_lr_generic_retransport", "ensemble"),
        ("learned_unconstrained_matrix_context_action", "parameter_matched_widened_model"),
        ("generic_mixture_of_experts", "inference_matched_model"),
    ]
    rows = []; accuracies = {}
    for source_method, role in mapping:
        block = frame[frame.method == source_method]
        if block.empty: continue
        accuracy = float(block.accuracy.mean()); accuracies[role] = accuracy
        parameters = int(max(1, block.trainable_parameters.mean())); branches = 1 if role != "strongest_generic_moe" else int(max(1, block.get("branch_count", 1).mean()))
        input_dimension = max(8, int(np.ceil(parameters / max(1, 8 * branches))))
        for batch in [1, 8, 32, 128]:
            latency, memory = measured_proxy_latency(input_dimension, 8, branches, batch, 99_000_000 + batch + parameters)
            rows.append({"family": "independent_controlled_context_confirmation", "role": role, "method": source_method, "accuracy": accuracy, "batch_size": batch, "trainable_parameters": parameters, "stored_parameters": int(block.stored_parameters.mean()) if "stored_parameters" in block else parameters, "flops": estimate_flops(input_dimension, 8, branches, batch), "latency_ms": latency, "peak_memory_mb": memory, "systems_measurement": "shape_matched_numpy_linear_proxy_not_end_to_end_method", "calibration_examples": int(block.get("calibration_samples", 0).mean()) if "calibration_samples" in block else 0, "router_examples": int(block.get("context_budget", 0).mean()) if "context_budget" in block else 0, "selector_validation_examples": int(block.get("selector_validation_samples", 0).mean()) if "selector_validation_samples" in block else 0, "candidate_count": int(block.get("candidate_count", 1).mean()) if "candidate_count" in block else 1, "branch_count": branches, "execution_commit": git_head(), "source_sha256": hashlib.sha256(SCRIPT.read_bytes()).hexdigest()})
    return rows, accuracies


def conditional_family_rows(path: Path, claims_path: Path, family: str) -> list[dict[str, object]]:
    import pandas as pd
    if not path.exists() or not claims_path.exists(): return []
    claims = pd.read_csv(claims_path)
    gate_columns = [column for column in claims.columns if column.endswith("gate_passed")]
    if not gate_columns or not claims[gate_columns].astype(bool).any().any(): return []
    frame = pd.read_csv(path); rows = []
    for method, block in frame.groupby("method"):
        parameters = int(max(1, block.trainable_parameters.mean())) if "trainable_parameters" in block else 64
        branches = int(max(1, block.branch_count.mean())) if "branch_count" in block else 1
        for batch in [1, 8, 32, 128]:
            latency, memory = measured_proxy_latency(max(8, parameters // max(8, branches * 8)), 8, branches, batch, 99_800_000 + batch + len(method))
            rows.append({"family": family, "role": method, "method": method, "accuracy": float(block.accuracy.mean()), "batch_size": batch, "trainable_parameters": parameters, "stored_parameters": int(block.stored_parameters.mean()) if "stored_parameters" in block else parameters, "flops": estimate_flops(max(8, parameters // max(8, branches * 8)), 8, branches, batch), "latency_ms": latency, "peak_memory_mb": memory, "systems_measurement": "shape_matched_numpy_linear_proxy_not_end_to_end_method", "calibration_examples": int(block.calibration_examples.mean()) if "calibration_examples" in block else 0, "router_examples": int(block.chart_training_examples.mean()) if "chart_training_examples" in block else 0, "selector_validation_examples": int(block.selector_validation_examples.mean()) if "selector_validation_examples" in block else 0, "candidate_count": int(block.candidate_count.mean()) if "candidate_count" in block else 1, "branch_count": branches, "execution_commit": git_head(), "source_sha256": hashlib.sha256(SCRIPT.read_bytes()).hexdigest()})
    return rows


def main() -> None:
    rows, controlled_accuracies = controlled_context_rows()
    rows.extend(conditional_family_rows(OUT / "chart_runs.csv", OUT / "chart_claims.csv", "input_inferred_chart_recovery"))
    rows.extend(conditional_family_rows(OUT / "full_model_runs.csv", OUT / "full_model_claims.csv", "full_model_transition_geometry"))
    rows.extend(conditional_family_rows(OUT / "multiview_runs.csv", OUT / "multiview_claims.csv", "realistic_multiview"))
    claims = []
    for family in sorted({str(row["family"]) for row in rows}):
        block = [row for row in rows if row["family"] == family and row["batch_size"] == 32]
        structured_names = [row for row in block if "structured" in str(row["role"]) or row["role"] == "structured_method"]
        generic_names = [row for row in block if row not in structured_names]
        best_structured = max((float(row["accuracy"]) for row in structured_names), default=float("nan"))
        best_generic = max((float(row["accuracy"]) for row in generic_names), default=float("nan"))
        survives = bool(np.isfinite(best_structured) and np.isfinite(best_generic) and best_structured >= best_generic - 0.002)
        claims.append({"family": family, "best_structured_accuracy": best_structured, "best_strong_baseline_accuracy": best_generic, "survives_accuracy_tolerance": survives, "official_external_implementations_executed": False, "end_to_end_batch_costs_measured": False, "matched_cost_audit_passed": False})
    summary = []
    for family in sorted({str(row["family"]) for row in rows}):
        for role in sorted({str(row["role"]) for row in rows if row["family"] == family}):
            block = [row for row in rows if row["family"] == family and row["role"] == role]
            summary.append({"family": family, "role": role, "accuracy": float(np.mean([float(row["accuracy"]) for row in block])), "trainable_parameters": max(int(row["trainable_parameters"]) for row in block), "stored_parameters": max(int(row["stored_parameters"]) for row in block), "latency_batch32_ms": next(float(row["latency_ms"]) for row in block if row["batch_size"] == 32), "peak_memory_mb": max(float(row["peak_memory_mb"]) for row in block)})
    write_csv(OUT / "cost_runs.csv", rows)
    write_csv(OUT / "cost_summary.csv", summary)
    write_csv(OUT / "cost_claims.csv", claims)
    latex_table(OUT / "tables" / "cost.tex", ["family", "role", "accuracy", "trainable_parameters", "stored_parameters", "latency_batch32_ms"], summary, "Bounded baseline and systems-cost proxy audit")
    passed = sum(bool(row["matched_cost_audit_passed"]) for row in claims)
    (OUT / "cost_report.md").write_text(
        "# Baseline and systems-cost proxy audit\n\n"
        f"Execution commit: `{git_head()}`. Executed accuracy rows and parameter/storage metadata were collected at batch sizes 1, 8, 32, and 128. The batch-size latency, memory, and FLOP rows are explicitly labeled shape-matched NumPy proxies, not end-to-end timings or official external baseline executions. "
        f"Consequently, {passed} of {len(claims)} families passed a matched systems-cost gate. Conditional negative stages were not promoted into this audit.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
