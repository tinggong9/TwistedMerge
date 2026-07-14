#!/usr/bin/env python
"""Resumable sequential runner for the overnight TwistedMerge program."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "overnight_program"


STAGES = [
    (0, "provenance_audit", ["experiments/overnight_stage0_audit.py", "--run-tests"]),
    (1, "fresh_practical_selector", ["experiments/final_practical_selector.py", "--mode", "full"]),
    (2, "hodge_lr", ["experiments/hodge_lr_smoke.py"]),
    (3, "context_two_loop", ["experiments/context_dependent_two_loop_holonomy.py", "--mode", "full"]),
    (4, "central_release", ["experiments/central_release_overnight.py"]),
    (5, "arxiv_release", ["experiments/build_overnight_release.py", "--release", "arxiv"]),
    (6, "quaternion_pose", ["experiments/quaternion_projective_pose_merge.py", "--mode", "smoke"]),
    (7, "natural_twist", ["experiments/natural_twist_discovery.py", "--mode", "smoke"]),
    (8, "pretrained_vision", ["experiments/full_pretrained_vision_merging.py", "--mode", "smoke"]),
    (9, "lora_holonomy", ["experiments/lora_holonomy_merging.py", "--mode", "smoke"]),
    (10, "federated_frame", ["experiments/federated_sensor_frame_merge.py", "--mode", "smoke"]),
    (11, "transformer", ["experiments/shared_base_transformer_merging.py", "--mode", "smoke"]),
    (12, "capacity_latency", ["experiments/capacity_latency_robustness.py"]),
]


def load_status() -> dict:
    path = OUT / "status.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"schema_version": 1, "stages": {}}


def save_status(status: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    lines = ["# Overnight TwistedMerge Program Status", "", "| stage | name | status | updated |", "| --- | --- | --- | --- |"]
    for number, name, _ in STAGES:
        entry = status["stages"].get(str(number), {})
        lines.append(f"| {number} | {name} | {entry.get('status', 'pending')} | {entry.get('updated_at', '')} |")
    (OUT / "status.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def stamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-stage", type=int, default=0)
    parser.add_argument("--through-stage", type=int, default=12)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    status = load_status()
    save_status(status)

    for number, name, command in STAGES:
        if not (args.from_stage <= number <= args.through_stage):
            continue
        existing = status["stages"].get(str(number), {})
        if existing.get("status") == "completed" and not args.force:
            continue
        full = [args.python, *command]
        status["stages"][str(number)] = {
            "name": name,
            "status": "running",
            "command": full,
            "updated_at": stamp(),
        }
        save_status(status)
        completed = subprocess.run(full, cwd=ROOT, text=True)
        entry = status["stages"][str(number)]
        entry["updated_at"] = stamp()
        entry["return_code"] = completed.returncode
        entry["status"] = "completed" if completed.returncode == 0 else "failed"
        save_status(status)
        if completed.returncode:
            raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
