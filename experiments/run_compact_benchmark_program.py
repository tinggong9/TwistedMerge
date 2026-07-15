#!/usr/bin/env python3
"""Resumable runner for the compact, decision-focused benchmark program."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "compact_program"
LOCAL_LOGS = ROOT / "reports" / "tmp" / "compact_program" / "runner_logs"

STAGES = {
    0: ("environment and provenance", "full", "compact_environment.py"),
    1: ("context fairness", "discovery", "compact_context_fairness.py"),
    2: ("Hodge and low-rank ablation", "discovery", "compact_hodge_lr_ablation.py"),
    3: ("natural checkpoint discovery", "discovery", "compact_natural_twist.py"),
    4: ("pretrained vision", "discovery", "compact_pretrained_vision.py"),
    5: ("federated frames", "discovery", "compact_federated_frame.py"),
    6: ("systems and distillation", "full", "compact_systems_audit.py"),
    7: ("claim decision and evidence packet", "full", "compact_evidence_packet.py"),
}


def source_hash(script: Path) -> str:
    return hashlib.sha256(script.read_bytes()).hexdigest()


def load_status() -> dict:
    path = OUT / "status.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"schema_version": 1, "run_id": str(uuid.uuid4()), "stages": {}}


def write_status(status: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "status.json").write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# Compact benchmark status", "", f"Run ID: `{status['run_id']}`", "", "| Stage | Kind | Status | Runtime (s) |", "|---:|---|---|---:|"]
    for number in STAGES:
        item = status["stages"].get(str(number), {})
        lines.append(
            f"| {number} | {STAGES[number][1]} | {item.get('status', 'pending')} | {item.get('runtime_seconds', '')} |"
        )
    lines.extend(["", "Detailed stdout and stderr are retained in the local ignored run cache.", ""])
    (OUT / "status.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-stage", type=int, choices=STAGES)
    parser.add_argument("--start-stage", type=int, choices=STAGES, default=0)
    parser.add_argument("--stop-stage", type=int, choices=STAGES, default=7)
    parser.add_argument("--new-run", action="store_true", help="start a fresh status ledger and execute all selected stages")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    LOCAL_LOGS.mkdir(parents=True, exist_ok=True)
    status = {"schema_version": 1, "run_id": str(uuid.uuid4()), "stages": {}} if args.new_run else load_status()
    status["runner_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    for number, (name, kind, filename) in STAGES.items():
        if number < args.start_stage or number > args.stop_stage:
            continue
        if args.force_stage is not None and number != args.force_stage:
            continue
        script = ROOT / "experiments" / filename
        digest = source_hash(script)
        previous = status["stages"].get(str(number), {})
        if args.force_stage is None and previous.get("status") == "completed" and previous.get("source_sha256") == digest:
            continue
        command = [sys.executable, str(script)]
        started = time.time()
        completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
        runtime = time.time() - started
        log_prefix = LOCAL_LOGS / f"stage{number}"
        log_prefix.with_suffix(".stdout.txt").write_text(completed.stdout, encoding="utf-8")
        log_prefix.with_suffix(".stderr.txt").write_text(completed.stderr, encoding="utf-8")
        item = {
            "name": name,
            "kind": kind,
            "status": "completed" if completed.returncode == 0 else "failed",
            "exit_code": completed.returncode,
            "runtime_seconds": round(runtime, 3),
            "command": f"python experiments/{filename}",
            "source_sha256": digest,
            "execution_commit": status["runner_commit"],
        }
        status["stages"][str(number)] = item
        write_status(status)
        print(f"stage {number}: {item['status']} ({runtime:.1f}s)", flush=True)
        if completed.returncode:
            print(completed.stderr[-4000:], file=sys.stderr)
            raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
