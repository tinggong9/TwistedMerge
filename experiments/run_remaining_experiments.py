#!/usr/bin/env python3
"""Run and compile the nine-stage remaining-experiments program."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.remaining_experiment_common import OUT, git_head, sha256_file, write_csv, write_json

STAGES = {
    "1": "gauge_invariance_and_refinement.py",
    "2": "input_inferred_chart_recovery.py",
    "3": "full_model_transition_geometry.py",
    "4": "strong_compositional_baselines.py",
    "5": "complete_natural_stability_gate.py",
    "6": "equivariant_distillation.py",
    "7": "realistic_multiview_twist.py",
    "8": "projective_representation_expansion.py",
    "9": "official_baseline_cost_audit.py",
}

# Status describes protocol coverage, not whether the script returned zero.  A
# stage is marked partial whenever the bounded implementation did not execute
# every method or data condition in the requested protocol.
PROTOCOL_COVERAGE = {
    "1": ("partial", "Exact gauge/lift checks ran, but the refinement audit is a finite-dimensional certificate surrogate."),
    "2": ("partial", "Fashion-MNIST discovery ran; the bounded router uses ridge and orbit-moment features, not a trained equivariant CNN."),
    "3": ("partial", "Cached pretrained features and synthetic feature adapters ran; residual-block and full-backbone fine-tuning did not run."),
    "4": ("partial", "Exact algebra and ridge/random-feature sequence controls ran; neural Transformer and differentiable-automaton baselines did not run."),
    "5": ("complete", "Both fixed natural families ran with five calibration resamples and four 200-draw null families."),
    "6": ("complete", "Both controlled teachers, seven bounded student families, and five objectives ran; the Fashion branch was correctly gated off."),
    "7": ("partial", "ModelNet10 mesh features and four fitted view experts ran; the graph and retransport methods are bounded linear proxies."),
    "8": ("partial", "Exact cyclic and finite-Heisenberg constructions ran; S3, D4, and Q8 were covered only by normalized trivial cocycles."),
    "9": ("partial", "Accuracy was read from executed controls, but batch-size systems costs are shape-matched NumPy proxies, not end-to-end method timings."),
}


def stage_outputs(stage: str) -> list[Path]:
    prefixes = {"1": ["invariance_", "refinement", "outside_gauge"], "2": ["chart_"], "3": ["full_model_"], "4": ["composition_"], "5": ["natural_"], "6": ["distillation_"], "7": ["multiview_"], "8": ["central_extensions", "projective_"], "9": ["cost_"]}
    return [path for path in OUT.rglob("*") if path.is_file() and any(path.name.startswith(prefix) for prefix in prefixes[stage])]


def run_stage(stage: str, python: str, env: dict[str, str]) -> dict[str, object]:
    command = [python, str(ROOT / "experiments" / STAGES[stage])]
    start = time.perf_counter(); process = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True); elapsed = time.perf_counter() - start
    result = {"stage": stage, "script": STAGES[stage], "command": f"python experiments/{STAGES[stage]}", "execution_commit": git_head(), "returncode": process.returncode, "runtime_seconds": elapsed, "stdout": process.stdout[-4000:], "stderr": process.stderr[-4000:]}
    if process.returncode != 0:
        raise RuntimeError(json.dumps(result, indent=2))
    return result


def compile_final(commands: list[dict[str, object]]) -> None:
    coverage = [
        {"stage": stage, "status": PROTOCOL_COVERAGE[stage][0], "limitation": PROTOCOL_COVERAGE[stage][1]}
        for stage in STAGES
    ]
    write_csv(OUT / "protocol_coverage.csv", coverage)
    stage_lines = []
    for stage, script in STAGES.items():
        outputs = stage_outputs(stage)
        status, limitation = PROTOCOL_COVERAGE[stage]
        stage_lines.append(f"| {stage} | {status} | {len(outputs)} | `experiments/{script}` | {limitation} |")
    reports = {
        "1": "invariance_report.md", "2": "chart_report.md", "3": "full_model_report.md", "4": "composition_report.md",
        "5": "natural_report.md", "6": "distillation_report.md", "7": "multiview_report.md", "8": "projective_report.md", "9": "cost_report.md",
    }
    conclusions = []
    for stage, report in reports.items():
        path = OUT / report
        final_line = path.read_text(encoding="utf-8").strip().splitlines()[-1] if path.exists() else "report missing"
        conclusions.append(f"- Stage {stage}: {final_line}")
    (OUT / "final_experimental_report.md").write_text(
        "# Final experimental report\n\n"
        f"Execution commit: `{git_head()}`.\n\n"
        "Status is protocol coverage, not merely process completion. Every stage script returned zero; partial means that at least one requested method or condition remained a bounded proxy or was not executed.\n\n"
        "| Stage | Protocol status | Artifact count | Script | Limitation |\n|---|---|---:|---|---|\n" + "\n".join(stage_lines) + "\n\n"
        "## Factual outcomes\n\n" + "\n".join(conclusions) + "\n\n"
        "All failed gates and negative findings remain in the stage claims and numerical ledgers. Exact commands and runtimes are stored in the experiment manifest. Protocol limitations are also stored in `reports/remaining_experiments/protocol_coverage.csv`.\n",
        encoding="utf-8",
    )
    excluded = {OUT / "experiment_manifest.json", OUT / "experiment_manifest.csv"}
    artifacts = []
    for path in sorted(OUT.rglob("*")):
        if path.is_file() and path not in excluded:
            artifacts.append({"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path), "bytes": path.stat().st_size, "execution_commit": git_head()})
    write_csv(OUT / "experiment_manifest.csv", artifacts)
    write_json(OUT / "experiment_manifest.json", {"execution_commit": git_head(), "python": sys.version, "platform": platform.platform(), "stages": commands, "protocol_coverage": coverage, "artifacts": artifacts})


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--stage", choices=list(STAGES) + ["all"], default="all"); parser.add_argument("--resume", action="store_true"); parser.add_argument("--python", default=sys.executable); args = parser.parse_args()
    selected = list(STAGES) if args.stage == "all" else [args.stage]
    env = os.environ.copy(); commands = []
    for stage in selected:
        if args.resume and stage_outputs(stage):
            commands.append({"stage": stage, "script": STAGES[stage], "command": "resume-existing", "execution_commit": git_head(), "returncode": 0, "runtime_seconds": 0.0, "stdout": "", "stderr": ""})
        else:
            commands.append(run_stage(stage, args.python, env))
    if args.stage == "all": compile_final(commands)


if __name__ == "__main__":
    main()
