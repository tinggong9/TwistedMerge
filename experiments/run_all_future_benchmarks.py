#!/usr/bin/env python3
"""Resumable, resource-aware runner for all public benchmark tiers."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.future_benchmark_common import LOCAL, OUT, ROOT, ensure_dirs, git_head, safe_path, write_csv, write_json


@dataclass(frozen=True)
class Stage:
    stage_id: str
    tier: str
    name: str
    script: str
    kind: str = "discovery"
    args: tuple[str, ...] = ()


STAGES = [
    Stage("E0", "emergency", "clean provenance and evidence freeze", "future_e0_provenance.py", "clean-freeze"),
    Stage("E1", "emergency", "independent controlled confirmation", "emergency_level2_confirmation.py", "confirmation"),
    Stage("E2", "emergency", "calibration and uncertainty", "context_calibration_audit.py"),
    Stage("E3", "emergency", "mechanistic component attribution", "emergency_mechanism_ablation.py"),
    Stage("E4", "emergency", "central and representation-rank freeze", "future_controlled_freeze.py", "clean-freeze"),
    Stage("E5", "emergency", "practical selector freeze", "future_practical_selector_freeze.py", "clean-freeze"),
    Stage("E6", "emergency", "emergency evidence packet", "future_emergency_evidence_packet.py"),
    Stage("N1", "near-term", "targeted realistic confirmation", "targeted_realistic_level3.py"),
    Stage("N2", "near-term", "real-image chart inference", "real_image_chart_inference.py"),
    Stage("N3", "near-term", "compositional context generalization", "compositional_context_generalization.py"),
    Stage("N4", "near-term", "pretrained vision", "pretrained_vision_near_term.py"),
    Stage("N5", "near-term", "federated sensor frames", "federated_sensor_frame_near_term.py"),
    Stage("N6", "near-term", "real adapter benchmark", "real_lora_adapter_near_term.py"),
    Stage("N7", "near-term", "shared-base transformer merging", "pretrained_transformer_near_term.py"),
    Stage("N8", "near-term", "projective pose", "quaternion_pose_near_term.py"),
    Stage("N9", "near-term", "systems and distillation", "future_systems_distillation.py"),
    Stage("N10", "near-term", "claim decision", "future_claim_decision.py"),
    Stage("X1", "extended", "broader pretrained vision", "broader_vision_extended.py"),
    Stage("X2", "extended", "broader language and adapters", "broader_language_extended.py"),
    *[Stage(f"X{index}", "extended", f"extended discovery topic {index}", "extended_benchmark_suite.py", args=("--stage", f"X{index}")) for index in range(3, 13)],
    Stage("F", "extended", "global evidence report", "future_final_report.py"),
]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_status(new: bool) -> dict:
    path = OUT / "status.json"
    if not new and path.exists():
        status = json.loads(path.read_text(encoding="utf-8"))
        status.get("stages", {}).pop("X1-X12", None)
        return status
    return {"schema_version": 1, "run_id": str(uuid.uuid4()), "stages": {}}


def render(status: dict) -> None:
    write_json(OUT / "status.json", status)
    lines = ["# Future benchmark status", "", f"Run ID: `{status['run_id']}`", "", "| Stage | Tier | Kind | State | Runtime (s) |", "|---|---|---|---|---:|"]
    for stage in STAGES:
        item = status["stages"].get(stage.stage_id, {})
        lines.append(f"| {stage.stage_id} | {stage.tier} | {stage.kind} | {item.get('state', 'pending')} | {item.get('runtime_seconds', '')} |")
    lines.extend(["", "Raw stdout and stderr are retained in the ignored local run cache.", ""])
    (OUT / "status.md").write_text("\n".join(lines), encoding="utf-8")
    commands = []
    failures = []
    for stage in STAGES:
        item = status["stages"].get(stage.stage_id)
        if not item:
            continue
        commands.append({key: item.get(key, "") for key in ["stage_id", "tier", "command", "execution_commit", "source_sha256", "started_at", "runtime_seconds", "exit_code", "state"]})
        if item.get("state") in {"failed", "blocked"}:
            failures.append({"stage_id": stage.stage_id, "state": item.get("state"), "exit_code": item.get("exit_code"), "error": item.get("summary", "")})
    write_csv(OUT / "commands.csv", commands, ["stage_id", "tier", "command", "execution_commit", "source_sha256", "started_at", "runtime_seconds", "exit_code", "state"])
    write_csv(OUT / "failures.csv", failures, ["stage_id", "state", "exit_code", "error"])


def append_attempt(run_id: str, item: dict) -> None:
    path = OUT / "attempt_history.csv"
    fields = ["run_id", "stage_id", "tier", "command", "execution_commit", "source_sha256", "started_at", "runtime_seconds", "exit_code", "state", "summary"]
    existing = []
    if path.exists():
        with path.open(newline="", encoding="utf-8") as handle:
            existing = list(csv.DictReader(handle))
    existing.append({"run_id": run_id, **{key: item.get(key, "") for key in fields if key != "run_id"}})
    write_csv(path, existing, fields)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", choices=["emergency", "near-term", "extended", "all"], default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force-stage", choices=[stage.stage_id for stage in STAGES])
    parser.add_argument("--new-run", action="store_true")
    args = parser.parse_args()
    ensure_dirs()
    status = load_status(args.new_run)
    selected = STAGES if args.tier == "all" else [stage for stage in STAGES if stage.tier == args.tier]
    if args.force_stage:
        selected = [stage for stage in STAGES if stage.stage_id == args.force_stage]
    for stage in selected:
        script = ROOT / "experiments" / stage.script
        source_sha = digest(script)
        previous = status["stages"].get(stage.stage_id, {})
        if args.resume and previous.get("state") in {"completed", "confirmation", "clean-freeze", "negative", "blocked"} and previous.get("source_sha256") == source_sha:
            continue
        command = [sys.executable, str(script), *stage.args]
        started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        started = time.time()
        completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
        runtime = time.time() - started
        (LOCAL / "logs").mkdir(parents=True, exist_ok=True)
        (LOCAL / "logs" / f"{stage.stage_id}.stdout.txt").write_text(completed.stdout, encoding="utf-8")
        (LOCAL / "logs" / f"{stage.stage_id}.stderr.txt").write_text(completed.stderr, encoding="utf-8")
        result_path = LOCAL / "stage_results" / f"{stage.stage_id}.json"
        result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {}
        state = result.get("state", "completed" if completed.returncode == 0 else "failed")
        if completed.returncode and state not in {"blocked", "failed"}:
            state = "failed"
        summary = result.get("summary") or safe_path(completed.stderr[-2000:] or completed.stdout[-1000:])
        item = {
            "stage_id": stage.stage_id,
            "tier": stage.tier,
            "kind": stage.kind,
            "name": stage.name,
            "state": state,
            "summary": summary,
            "command": " ".join([Path(sys.executable).name, f"experiments/{stage.script}", *stage.args]),
            "execution_commit": git_head(),
            "source_sha256": source_sha,
            "started_at": started_at,
            "runtime_seconds": round(runtime, 3),
            "exit_code": completed.returncode,
        }
        status["stages"][stage.stage_id] = item
        append_attempt(status["run_id"], item)
        render(status)
        print(f"{stage.stage_id}: {state} ({runtime:.1f}s)", flush=True)
    render(status)


if __name__ == "__main__":
    main()
