#!/usr/bin/env python3
"""Bounded runner for the exact and biomedical spatial-output program."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "spatial_output_program"

STAGES: dict[str, tuple[str, str]] = {
    "S1": ("sanity", "exact_mask_retransport.py"),
    "S2": ("sanity", "exact_spatial_output_actions.py"),
    "S3": ("sanity", "trivial_vs_spatial_output_action.py"),
    "B0": ("biomedical", "biomedical_dataset_audit.py"),
    "B1": ("biomedical", "biomedical_segmentation_discovery.py"),
    "B2": ("biomedical", "biomedical_zeroshot_segmentation.py"),
    "B3": ("biomedical", "biomedical_chart_uncertainty.py"),
    "B4": ("biomedical", "biomedical_segmentation_cost.py"),
    "C1": ("biomedical", "multidomain_biomedical_experts.py"),
    "C2": ("biomedical", "biomedical_missing_expert_robustness.py"),
    "D1": ("biomedical", "segmentation_transition_geometry.py"),
    "D2": ("biomedical", "residual_aware_segmentation.py"),
    "E1": ("confirmation", "second_biomedical_segmentation.py"),
    "E2": ("confirmation", "biomedical_landmark_retransport.py"),
    "F1": ("extended", "medical_3d_retransport.py"),
    "F2": ("extended", "microscopy_multiview_retransport.py"),
    "Z0": ("all", "spatial_output_finalize.py"),
}

SENTINEL_DIRS = {
    "S1": "sanity", "S2": "sanity", "S3": "sanity", "B0": "data",
    "B1": "biomedical/discovery", "B2": "biomedical/zeroshot", "B3": "biomedical/uncertainty", "B4": "biomedical/cost",
    "C1": "multidomain", "C2": "robustness", "D1": "transitions", "D2": "transitions", "E1": "confirmation", "E2": "landmarks", "F1": "extended_3d", "F2": "microscopy", "Z0": ".",
}


def stages_for(tier: str, stage: str | None) -> list[str]:
    if stage:
        value = stage.upper()
        if value not in STAGES:
            raise ValueError(f"unknown stage {stage}")
        return [value]
    if tier == "all":
        return list(STAGES)
    values = [name for name, (owner, _) in STAGES.items() if owner == tier]
    return values


def complete(stage: str) -> bool:
    path = OUT / SENTINEL_DIRS[stage] / f".{stage}.complete.json"
    if not path.exists():
        return False
    state = str(json.loads(path.read_text(encoding="utf-8")).get("state"))
    return state in {"completed", "gate_closed", "blocked"}


def append_failure(row: dict[str, Any]) -> None:
    path = OUT / "failures.csv"
    fields = ("stage", "command", "exit_code", "stderr", "time")
    rows = []
    if path.exists():
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    rows.append({key: row.get(key, "") for key in fields})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run_stage(stage: str, smoke: bool, force: bool) -> int:
    script = STAGES[stage][1]
    command = [str(ROOT / ".venv" / "bin" / "python"), str(ROOT / "experiments" / script)]
    if smoke and stage not in {"S1", "S2", "S3", "B0"}:
        command.append("--smoke")
    if force and stage == "B1":
        command.append("--force")
    result = subprocess.run(command, cwd=ROOT, text=True, check=False)
    if result.returncode:
        append_failure({"stage": stage, "command": " ".join(command), "exit_code": result.returncode, "stderr": "see command output", "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    return result.returncode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", choices=("sanity", "biomedical", "confirmation", "extended", "all"), default="all")
    parser.add_argument("--stage")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.force and (OUT / "failures.csv").exists():
        (OUT / "failures.csv").unlink()
    for stage in stages_for(args.tier, args.stage):
        if args.resume and not args.force and complete(stage):
            continue
        code = run_stage(stage, args.smoke, args.force)
        if code:
            raise SystemExit(code)


if __name__ == "__main__":
    main()
