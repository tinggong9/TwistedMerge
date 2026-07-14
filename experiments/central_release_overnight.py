#!/usr/bin/env python3
"""Run and validate the Stage 4 central/period-index evidence release."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "overnight_program"

REQUIRED_METHODS = {
    "ordinary_weight_average",
    "git_rebasin_pairwise",
    "c2m3_synchronized",
    "supplied_context_q2_branch_predictor",
    "validation_face_table_router",
    "wrong_context_control",
    "wrong_twist_control",
    "no_twist_branch_control",
    "random_branch_control",
    "parameter_matched_wide_control",
    "distilled_single_model_control",
    "ensemble_reference",
}
REQUIRED_PERIOD_CASES = {"d2_k1", "d2_k2", "d2_k3", "d3_k1", "d3_k2", "d4_k1", "d4_k2"}


def validate_outputs(out: Path = OUT) -> dict[str, object]:
    runs = pd.read_csv(out / "central_mu2_runs.csv")
    period = pd.read_csv(out / "period_index_summary.csv")
    missing_methods = REQUIRED_METHODS - set(runs["method"])
    missing_period = REQUIRED_PERIOD_CASES - set(period["case_id"])
    checks = {
        "required_methods_present": not missing_methods,
        "required_period_cases_present": not missing_period,
        "all_predictions_executed": bool(runs["candidate_logits_executed"].all()),
        "all_leakage_regressions_passed": bool(runs["label_permutation_regression_passed"].all()),
        "all_scalar_orders_match_d": bool((period["scalar_commutator_order"] == period["d"]).all()),
        "all_thresholds_match_d_power_k": bool(period["threshold_equals_d_power_k"].all()),
        "all_direct_sums_realized": bool(period["direct_sum_multiple_realized"].all()),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Stage 4 validation failed: {checks}; missing methods={missing_methods}; missing period={missing_period}")
    return checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "full"), default="full")
    args = parser.parse_args()
    execution_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty_worktree_at_execution = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    command = [
        sys.executable,
        str(ROOT / "experiments" / "central_reproduction_next.py"),
        "--out-dir",
        str(OUT),
    ]
    if args.mode == "smoke":
        command.extend(["--seeds", "0:1", "--widths", "32", "--samples-per-chart", "100", "--samples-per-overlap", "200"])
    subprocess.run(command, cwd=ROOT, check=True)
    checks = validate_outputs()
    shutil.copy2(OUT / "central_reproduction_report.md", OUT / "central_release_report.md")
    manifest = json.loads((OUT / "central_reproduction_manifest.json").read_text(encoding="utf-8"))
    config = {
        "stage": 4,
        "mode": args.mode,
        "execution_commit": execution_commit,
        "dirty_worktree_at_execution": dirty_worktree_at_execution,
        "command": " ".join(command),
        "validation": checks,
    }
    (OUT / "central_release_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
