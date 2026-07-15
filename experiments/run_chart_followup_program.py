#!/usr/bin/env python3
"""Resumable orchestrator for the focused chart follow-up program."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "chart_followup"
TMP = ROOT / "reports" / "tmp" / "chart_followup"
STATUS_JSON = OUT / "status.json"
STATUS_MD = OUT / "status.md"
COMMANDS = OUT / "commands.csv"
FAILURES = OUT / "failures.csv"

STAGES = {
    "ABLATION": "experiments/chart_component_ablation.py",
    "ZEROSHOT": "experiments/strict_zeroshot_chart_generalization.py",
    "CIFAR": "experiments/cifar10_chart_retransport.py",
    "COST": "experiments/fashion_complete_cost_audit.py",
    "COMPRESSION": "experiments/compression_claim_reaudit.py",
    "SAMPLE_EFFICIENCY": "experiments/chart_sample_efficiency.py",
}
ORDER = tuple(STAGES)
SEEDS = {
    "ABLATION": "20:29",
    "ZEROSHOT": "30:39",
    "CIFAR": "0:4 discovery; 5:9 conditional confirmation",
    "COST": "20:29 checkpoints",
    "COMPRESSION": "retained executed-artifact ledger",
    "SAMPLE_EFFICIENCY": "40:44",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def execution_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def source_sha(path: str) -> str:
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, object]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def update_status(payload: dict[str, object]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    payload["updated_at"] = now()
    STATUS_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    stages = payload.get("stages", {})
    lines = ["# Chart follow-up status", "", f"Overall: **{payload.get('overall', 'unknown')}**", ""]
    for stage in ORDER:
        value = stages.get(stage, {}) if isinstance(stages, dict) else {}
        lines.append(f"- {stage}: {value.get('status', 'not_selected')}")
    STATUS_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def marker(stage: str) -> Path:
    return OUT / STAGES[stage].removeprefix("experiments/").removesuffix(".py").replace("chart_component_ablation", "ablation").replace("strict_zeroshot_chart_generalization", "zeroshot").replace("cifar10_chart_retransport", "cifar").replace("fashion_complete_cost_audit", "cost").replace("compression_claim_reaudit", "compression").replace("chart_sample_efficiency", "sample_efficiency") / ".complete.json"


def clear_stage(stage: str) -> None:
    stage_dir = marker(stage).parent
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    checkpoint_dir = TMP / "checkpoints" / stage_dir.name
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)


def run_command(command: list[str], environment: dict[str, str]) -> tuple[int, float]:
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=ROOT, env=environment, check=False)
    return completed.returncode, time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=(*ORDER, "ALL"), default="ALL")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    selected = list(ORDER if arguments.stage == "ALL" else (arguments.stage,))
    environment = os.environ.copy()
    environment.setdefault("PYTHONPYCACHEPREFIX", "/private/tmp/twistedmerge-chart-followup-pycache")
    status: dict[str, object] = {
        "overall": "running",
        "selected": selected,
        "resume": arguments.resume,
        "force": arguments.force,
        "smoke": arguments.smoke,
        "started_at": now(),
        "stages": {name: {"status": "pending" if name in selected else "not_selected"} for name in ORDER},
    }
    command_rows: list[dict[str, object]] = read_csv(COMMANDS) if arguments.resume else []
    failure_rows: list[dict[str, object]] = read_csv(FAILURES) if arguments.resume else []
    update_status(status)
    for stage in selected:
        current_marker = marker(stage)
        if arguments.force:
            clear_stage(stage)
        if arguments.resume and current_marker.exists():
            status["stages"][stage] = {"status": "completed", "resumed_from_marker": True, "marker": str(current_marker.relative_to(ROOT))}
            update_status(status)
            continue
        command = [sys.executable, STAGES[stage]] + (["--smoke"] if arguments.smoke else [])
        status["stages"][stage] = {"status": "running", "started_at": now(), "command": " ".join(command)}
        update_status(status)
        started_at = now()
        try:
            returncode, seconds = run_command(command, environment)
            command_rows.append(
                {
                    "stage": stage,
                    "exact_command": " ".join(command),
                    "execution_commit": execution_commit(),
                    "source_sha256": source_sha(STAGES[stage]),
                    "seed": "smoke" if arguments.smoke else SEEDS[stage],
                    "start_time": started_at,
                    "runtime_seconds": seconds,
                    "exit_code": returncode,
                    "final_state": "completed" if returncode == 0 else "failed",
                    "factual_summary": "stage artifacts written" if returncode == 0 else f"stage exited with status {returncode}",
                }
            )
            if returncode:
                raise RuntimeError(f"stage exited with status {returncode}")
            current_marker.parent.mkdir(parents=True, exist_ok=True)
            current_marker.write_text(json.dumps({"stage": stage, "completed_at": now(), "command": command, "smoke": arguments.smoke}, indent=2) + "\n", encoding="utf-8")
            status["stages"][stage] = {"status": "completed", "duration_seconds": seconds, "marker": str(current_marker.relative_to(ROOT))}
            update_status(status)
        except Exception as error:
            failure_rows.append({"stage": stage, "exact_command": " ".join(command), "execution_commit": execution_commit(), "source_sha256": source_sha(STAGES[stage]), "seed": "smoke" if arguments.smoke else SEEDS[stage], "error_type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()})
            status["stages"][stage] = {"status": "failed", "message": str(error)}
            status["overall"] = "failed"
            write_csv(COMMANDS, command_rows)
            write_csv(FAILURES, failure_rows)
            update_status(status)
            raise
    if arguments.stage == "ALL":
        test_command = [sys.executable, "-m", "pytest", "-q", "tests"]
        started_at = now()
        started = time.perf_counter()
        with (OUT / "test_results.txt").open("w", encoding="utf-8") as handle:
            completed = subprocess.run(test_command, cwd=ROOT, env=environment, stdout=handle, stderr=subprocess.STDOUT, check=False, text=True)
        seconds = time.perf_counter() - started
        command_rows.append(
            {
                "stage": "TESTS",
                "exact_command": " ".join(test_command),
                "execution_commit": execution_commit(),
                "source_sha256": "multiple_test_sources",
                "seed": "not_applicable",
                "start_time": started_at,
                "runtime_seconds": seconds,
                "exit_code": completed.returncode,
                "final_state": "completed" if completed.returncode == 0 else "failed",
                "factual_summary": "full repository test suite passed" if completed.returncode == 0 else "full repository test suite failed",
            }
        )
        if completed.returncode:
            status["overall"] = "failed"
            failure_rows.append({"stage": "TESTS", "exact_command": " ".join(test_command), "execution_commit": execution_commit(), "source_sha256": "multiple_test_sources", "seed": "not_applicable", "error_type": "TestFailure", "message": f"pytest exited with status {completed.returncode}", "traceback": ""})
            write_csv(COMMANDS, command_rows)
            write_csv(FAILURES, failure_rows)
            update_status(status)
            raise SystemExit(completed.returncode)
    write_csv(COMMANDS, command_rows)
    write_csv(FAILURES, failure_rows)
    status["overall"] = "completed"
    status["completed_at"] = now()
    update_status(status)
    if arguments.stage == "ALL":
        final_command = [sys.executable, "experiments/chart_followup_finalize.py"] + (["--smoke"] if arguments.smoke else [])
        returncode, _ = run_command(final_command, environment)
        if returncode:
            raise SystemExit(returncode)


if __name__ == "__main__":
    main()
