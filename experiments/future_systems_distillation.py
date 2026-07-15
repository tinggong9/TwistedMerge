#!/usr/bin/env python3
"""N9: measured systems scaling and aggregation of executed distillation rows."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.future_benchmark_common import OUT, peak_memory_mb, stage_result, write_csv

DEST = OUT / "near_term"


def main() -> None:
    rng = np.random.default_rng(9_900_001)
    rows = []
    for branches in [1, 2, 4, 8, 16]:
        weights = rng.normal(size=(branches, 128, 64)).astype(np.float32)
        for batch in [1, 8, 32, 128]:
            inputs = rng.normal(size=(batch, 128)).astype(np.float32)
            timings = []
            for _ in range(9):
                started = time.perf_counter(); outputs = np.einsum("bi,nio->nbo", inputs, weights).mean(0); _ = outputs.argmax(1); timings.append(time.perf_counter() - started)
            rows.append({"branch_count": branches, "batch_size": batch, "latency_ms": float(np.median(timings) * 1000), "stored_parameters": int(weights.size), "inference_multiplier": branches, "peak_memory_mb": peak_memory_mb(), "measured": True})
    distillation = []
    mechanism_path = OUT / "emergency" / "mechanism_runs.csv"
    if mechanism_path.exists():
        frame = pd.read_csv(mechanism_path)
        for setting_id, block in frame.groupby("setting_id"):
            scores = block.set_index("method")
            if "full_twistedmerge_hodge_lr" in scores.index and "full_followed_by_distillation" in scores.index:
                distillation.append({"setting_id": setting_id, "teacher_accuracy": scores.loc["full_twistedmerge_hodge_lr", "accuracy"], "student_accuracy": scores.loc["full_followed_by_distillation", "accuracy"], "distillation_gap": scores.loc["full_followed_by_distillation", "accuracy"] - scores.loc["full_twistedmerge_hodge_lr", "accuracy"], "teacher_ece": scores.loc["full_twistedmerge_hodge_lr", "ece"], "student_ece": scores.loc["full_followed_by_distillation", "ece"]})
    write_csv(DEST / "systems_runs.csv", rows)
    summary = pd.DataFrame(rows).groupby("branch_count", as_index=False).agg(latency_ms=("latency_ms", "mean"), stored_parameters=("stored_parameters", "max"), peak_memory_mb=("peak_memory_mb", "max"))
    write_csv(DEST / "systems_summary.csv", summary.to_dict("records")); write_csv(DEST / "distillation.csv", distillation)
    mean_gap = float(pd.DataFrame(distillation).distillation_gap.mean()) if distillation else float("nan")
    write_csv(DEST / "systems_claims.csv", [{"claim": "branch_scaling_measured", "value": True}, {"claim": "distillation_mean_gap", "value": mean_gap}, {"claim": "distillation_retains_gain", "value": bool(distillation and mean_gap >= -0.002)}])
    summary.to_latex(DEST / "tables" / "systems.tex", index=False, float_format="%.6f")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    for batch, block in pd.DataFrame(rows).groupby("batch_size"): ax.plot(block.branch_count, block.latency_ms, marker="o", label=f"batch {batch}")
    ax.set(xlabel="Branch count", ylabel="Latency (ms)", xscale="log", yscale="log"); ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(DEST / "plots" / "systems_tradeoff.pdf"); plt.close(fig)
    (DEST / "systems_report.md").write_text(f"# Systems and distillation\n\nBranch counts 1, 2, 4, 8, and 16 were measured at batch sizes 1, 8, 32, and 128. Distillation rows were drawn from executed controlled predictors; the mean student-minus-teacher accuracy was `{mean_gap:+.6f}`.\n", encoding="utf-8")
    stage_result("N9", "completed", f"systems scaling measured; distillation rows={len(distillation)}", distillation_mean_gap=mean_gap)


if __name__ == "__main__":
    main()
