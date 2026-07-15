#!/usr/bin/env python3
"""Gated, resumable runner for the complete next TwistedMerge program."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.next_program_common import OUT, TMP, git_head, sha256_file, write_csv, write_json

SCRIPT = Path(__file__).resolve()
COMMAND_FIELDS = ("stage", "tier", "exact_command", "execution_commit", "source_hash", "seed", "start_time", "runtime_seconds", "exit_code", "final_state", "summary", "log_sha256")
FAILURE_FIELDS = ("stage", "tier", "time", "exit_code", "error_type", "summary", "log_path", "execution_commit")


@dataclass(frozen=True)
class Stage:
    stage_id: str
    tier: str
    script: str
    summary: str


STAGES = (
    Stage("A1", "immediate", "trained_chart_inference.py", "trained Fashion-MNIST chart inference"),
    Stage("A2", "immediate", "end_to_end_controlled_cost.py", "end-to-end controlled systems audit"),
    Stage("A3", "immediate", "nontrivial_refinement_invariance.py", "nontrivial refinement invariance"),
    Stage("B1", "iclr", "full_model_hidden_geometry.py", "full-model hidden-layer geometry"),
    Stage("B2", "iclr", "learned_compositional_baselines.py", "strong learned compositional baselines"),
    Stage("B3", "iclr", "genuine_multiview_retransport.py", "genuine multiview coordinate retransport"),
    Stage("B4", "iclr", "new_realistic_residual_search.py", "new realistic residual search"),
    Stage("B5", "iclr", "structured_compression.py", "structure-preserving compression"),
    Stage("B6", "iclr", "noncyclic_central_extensions.py", "noncyclic central extensions"),
    Stage("B7", "iclr", "official_baseline_integration.py", "conditional official baseline integration"),
    Stage("C1_C2", "extended", "conditional_extended_families.py", "conditional extended vision and adapter families"),
    Stage("C3", "extended", "language_checkpoint_transition_geometry.py", "language checkpoint transition geometry"),
    Stage("C4_C5", "extended", "comparison_alignment_robustness.py", "comparison-complex and alignment robustness"),
    Stage("C6", "extended", "selective_activation_diagnostics.py", "selective activation diagnostics"),
    Stage("C7", "extended", "real_scaling_audit.py", "real runtime and memory scaling"),
    Stage("C8", "extended", "reproducibility_release.py", "reproducibility and release manifests"),
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists(): return []
    with path.open(encoding="utf-8", newline="") as handle: return list(csv.DictReader(handle))


def write_status(payload: dict[str, object]) -> None:
    write_json(OUT / "status.json", payload)
    lines = ["# Next-program live status", "", f"Execution commit: `{payload['execution_commit']}`", "", f"Updated: `{payload['updated_at']}`", "", "| Stage | Tier | State | Runtime (s) | Summary |", "|---|---|---:|---:|---|"]
    stages = payload.get("stages", {})
    for stage in STAGES:
        record = stages.get(stage.stage_id, {})
        lines.append(f"| {stage.stage_id} | {stage.tier} | {record.get('state', 'pending')} | {record.get('runtime_seconds', '')} | {record.get('summary', stage.summary)} |")
    (OUT / "status.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def stage_selected(stage: Stage, tier: str, forced: set[str]) -> bool:
    return stage.stage_id in forced or tier == "all" or stage.tier == tier


def execution_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("TWISTEDMERGE_DATA_ROOT", str(ROOT / "data"))
    env.setdefault("HF_HOME", str(Path(env["TWISTEDMERGE_DATA_ROOT"]) / "huggingface"))
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    return env


def run_tests(env: dict[str, str]) -> int:
    command = [sys.executable, "-m", "pytest", "-q"]
    started = time.perf_counter(); result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
    payload = [f"command: {' '.join(command)}", f"execution_commit: {git_head()}", f"runtime_seconds: {time.perf_counter() - started:.6f}", f"exit_code: {result.returncode}", "", result.stdout, result.stderr]
    (OUT / "test_results.txt").write_text("\n".join(payload), encoding="utf-8")
    return result.returncode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", choices=("immediate", "iclr", "extended", "all"), default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force-stage", action="append", default=[], metavar="STAGE_ID")
    arguments = parser.parse_args(); forced = set(arguments.force_stage)
    known = {stage.stage_id for stage in STAGES}
    unknown = forced - known
    if unknown: parser.error("unknown stage(s): " + ", ".join(sorted(unknown)))
    existing = json.loads((OUT / "status.json").read_text()) if arguments.resume and (OUT / "status.json").exists() else {}
    status = {"program": "next_twistedmerge_program", "execution_commit": git_head(), "requested_tier": arguments.tier, "resume": arguments.resume, "updated_at": now(), "stages": existing.get("stages", {})}
    command_rows = read_rows(OUT / "commands.csv") if arguments.resume else []
    failure_rows = read_rows(OUT / "failures.csv") if arguments.resume else []
    env = execution_env()
    write_status(status); failures = 0
    for stage in STAGES:
        if not stage_selected(stage, arguments.tier, forced): continue
        prior = status["stages"].get(stage.stage_id, {})
        if arguments.resume and stage.stage_id not in forced and prior.get("state") in {"completed", "completed_gate_closed"}:
            continue
        if stage.stage_id == "C8":
            test_exit = run_tests(env)
            if test_exit: failures += 1
        script = ROOT / "experiments" / stage.script
        command = [sys.executable, str(script)]
        start_time = now(); started = time.perf_counter()
        status["stages"][stage.stage_id] = {"state": "running", "summary": stage.summary, "start_time": start_time}; status["updated_at"] = now(); write_status(status)
        result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
        runtime = time.perf_counter() - started; log_path = TMP / "logs" / f"{stage.stage_id}.log"; log_path.parent.mkdir(parents=True, exist_ok=True); log_path.write_text(result.stdout + result.stderr, encoding="utf-8")
        state = "completed" if result.returncode == 0 else "failed"
        summary = stage.summary if result.returncode == 0 else (result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "stage command failed")[:400]
        if result.returncode == 0 and stage.stage_id in {"B7", "C1_C2"}: state = "completed_gate_closed"
        status["stages"][stage.stage_id] = {"state": state, "summary": summary, "start_time": start_time, "runtime_seconds": runtime, "exit_code": result.returncode}; status["updated_at"] = now(); write_status(status)
        command_rows.append({"stage": stage.stage_id, "tier": stage.tier, "exact_command": " ".join(command), "execution_commit": git_head(), "source_hash": sha256_file(script), "seed": "stage_internal_preregistered", "start_time": start_time, "runtime_seconds": runtime, "exit_code": result.returncode, "final_state": state, "summary": summary, "log_sha256": sha256_file(log_path)})
        write_csv(OUT / "commands.csv", command_rows, COMMAND_FIELDS)
        if result.returncode:
            failures += 1; failure_rows.append({"stage": stage.stage_id, "tier": stage.tier, "time": now(), "exit_code": result.returncode, "error_type": "subprocess_nonzero", "summary": summary, "log_path": str(log_path.relative_to(ROOT)), "execution_commit": git_head()}); write_csv(OUT / "failures.csv", failure_rows, FAILURE_FIELDS)
    if not (OUT / "failures.csv").exists(): write_csv(OUT / "failures.csv", [], FAILURE_FIELDS)
    status["final_state"] = "completed" if failures == 0 else "completed_with_failures"; status["updated_at"] = now(); write_status(status)
    # C8 first runs inside the stage loop so its own execution is measured like
    # every other stage.  Regenerate its manifests after the final status and
    # command ledgers are durable, otherwise the release report would capture
    # C8 as still running and omit C8 from commands.csv.
    if stage_selected(next(stage for stage in STAGES if stage.stage_id == "C8"), arguments.tier, forced) and status["stages"].get("C8", {}).get("state") == "completed":
        finalize = subprocess.run([sys.executable, str(ROOT / "experiments" / "reproducibility_release.py")], cwd=ROOT, env=env, capture_output=True, text=True)
        if finalize.returncode:
            failures += 1
            summary = (finalize.stderr.strip().splitlines()[-1] if finalize.stderr.strip() else "C8 final manifest regeneration failed")[:400]
            failure_rows.append({"stage": "C8", "tier": "extended", "time": now(), "exit_code": finalize.returncode, "error_type": "final_manifest_regeneration", "summary": summary, "log_path": str((TMP / "logs" / "C8-finalize.log").relative_to(ROOT)), "execution_commit": git_head()})
            (TMP / "logs" / "C8-finalize.log").write_text(finalize.stdout + finalize.stderr, encoding="utf-8")
            write_csv(OUT / "failures.csv", failure_rows, FAILURE_FIELDS)
            status["final_state"] = "completed_with_failures"; status["updated_at"] = now(); write_status(status)
    if failures: raise SystemExit(1)


if __name__ == "__main__": main()
