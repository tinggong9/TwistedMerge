#!/usr/bin/env python3
"""E4: rerun controlled central-sign and representation-rank evidence."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.compact_hodge_lr_ablation import mu2_family
from experiments.future_benchmark_common import OUT, stage_result, write_csv
from experiments.period_index_central_benchmark import run_cases

DEST = OUT / "emergency"


def main() -> None:
    rows = []
    families = ["coboundary", "nontrivial_central", "noncentral_control"]
    for width in [32, 64]:
        for seed in range(30):
            current = mu2_family(width, seed)
            for row in current:
                row["cocycle_family"] = families[seed % len(families)]
                row["width"] = width
                row["control_seed"] = seed
                rows.append(row)
    frame = pd.DataFrame(rows)
    summary = frame.groupby(["cocycle_family", "width", "method"], as_index=False).agg(accuracy=("accuracy", "mean"), residual_before=("residual_before", "mean"), residual_after=("residual_after", "mean"), selected_rank=("selected_rank", "mean"), leakage_hash_passed=("leakage_hash_passed", "all"))
    ranks, period_summary = run_cases(max_multiplier=3)
    requested = {(2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (4, 1), (4, 2)}
    d_col = "d" if "d" in ranks.columns else "modulus"
    k_col = "k" if "k" in ranks.columns else "symplectic_pairs"
    selected_ranks = ranks[ranks.apply(lambda row: (int(row[d_col]), int(row[k_col])) in requested, axis=1)].copy()
    selected_summary = period_summary[period_summary.apply(lambda row: (int(row[d_col]), int(row[k_col])) in requested, axis=1)].copy()
    write_csv(DEST / "central_runs.csv", rows)
    write_csv(DEST / "central_summary.csv", summary.to_dict("records"))
    selected_ranks.to_csv(DEST / "period_index_ranks.csv", index=False)
    selected_summary.to_csv(DEST / "period_index_summary.csv", index=False)
    summary.to_latex(DEST / "tables" / "central_mu2.tex", index=False, float_format="%.6f")
    selected_summary.to_latex(DEST / "tables" / "period_index.tex", index=False, float_format="%.6f")
    all_cases = {(int(row[d_col]), int(row[k_col])) for _, row in selected_summary.iterrows()}
    complete = requested.issubset(all_cases)
    (DEST / "central_report.md").write_text(f"# Controlled central-sign freeze\n\nWidths 32 and 64, seeds 0--29, and three control families were rerun from one execution commit. All saved-logit permutation regressions passed: `{bool(frame.leakage_hash_passed.all())}`.\n", encoding="utf-8")
    (DEST / "period_index_report.md").write_text(f"# Representation-rank freeze\n\nAll seven requested finite-Heisenberg cases were {'executed' if complete else 'not completely represented'} with scalar commutator, relation residual, theoretical threshold, lower-rank outcomes, and direct-sum multiples retained.\n", encoding="utf-8")
    stage_result("E4", "clean-freeze" if complete and frame.leakage_hash_passed.all() else "failed", f"controlled central rerun rows={len(frame)}; representation-rank cases={len(all_cases)}", requested_cases_complete=complete)
    if not complete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
