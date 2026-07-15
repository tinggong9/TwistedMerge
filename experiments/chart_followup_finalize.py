#!/usr/bin/env python3
"""Build manifests, checksums, and a factual final report for chart follow-up."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.chart_followup_common import DATA, OUT, TMP
from experiments.next_program_common import git_head, sha256_file, write_json

STAGES = ("ablation", "zeroshot", "cifar", "cost", "compression", "sample_efficiency")


def write_csv(path: Path, values: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not values:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for value in values:
        for key in value:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(values)


def normalize_stage_csv_line_endings() -> None:
    for path in OUT.rglob("*.csv"):
        content = path.read_bytes()
        normalized = content.replace(b"\r\n", b"\n")
        if normalized != content:
            path.write_bytes(normalized)


def stage_claims(stage: str) -> list[dict[str, str]]:
    path = OUT / stage / "claims.csv"
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def rows(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def factual_final_report(smoke: bool) -> None:
    status = json.loads((OUT / "status.json").read_text(encoding="utf-8"))
    command_rows = rows(OUT / "commands.csv")
    lines = [
        "# Focused chart follow-up program",
        "",
        f"Mode: {'smoke' if smoke else 'full'}. Overall status: {status['overall']}. Execution commit: `{git_head()}`.",
        "",
        "## Stage status and protocol coverage",
        "",
    ]
    for stage in STAGES:
        key = "SAMPLE_EFFICIENCY" if stage == "sample_efficiency" else stage.upper()
        stage_status = status.get("stages", {}).get(key, {}).get("status", "not_recorded")
        ledger_name = "storage.csv" if stage == "compression" else "runs.csv"
        run_rows = rows(OUT / stage / ledger_name)
        seeds = sorted({row.get("seed", "") for row in run_rows})
        methods = sorted({row.get("method", "") for row in run_rows})
        lines.append(f"- `{stage}`: {stage_status}; {len(run_rows)} run rows, {len(seeds)} seeds, {len(methods)} methods; artifacts: `reports/chart_followup/{stage}/`.")
    lines.extend(["", "## Exact commands and execution commits", ""])
    for row in command_rows:
        lines.append(
            f"- `{row.get('exact_command', '')}` — commit `{row.get('execution_commit', '')}`, seed scope `{row.get('seed', '')}`, exit {row.get('exit_code', '')}, runtime {row.get('runtime_seconds', '')} s, state {row.get('final_state', '')}."
        )
    lines.extend(["", "## Numerical results", ""])
    display_columns = {
        "ablation": ("method", "task_accuracy", "ece", "complete_latency_ms_batch128", "stored_bytes"),
        "zeroshot": ("method", "element_role", "task_accuracy", "chart_accuracy", "ece"),
        "cifar": ("phase", "method", "task_accuracy", "ece", "stored_bytes"),
        "cost": ("method", "task_accuracy", "complete_path_latency_ms_batch128", "stored_bytes", "chart_training_examples"),
        "sample_efficiency": ("chart_label_budget", "method", "mean_task_accuracy", "mean_worst_condition_task_accuracy", "mean_chart_accuracy", "mean_ece"),
    }
    for stage, columns in display_columns.items():
        summary = rows(OUT / stage / "summary.csv")
        if not summary:
            continue
        lines.extend([f"### {stage}", "", "| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"])
        for row in summary:
            lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
        lines.append("")
    compression = rows(OUT / "compression" / "claims.csv")
    if compression:
        storage_confirmed = sum(row.get("storage_claim_confirmed") == "True" for row in compression)
        latency_evaluable = sum(row.get("latency_evaluable") == "True" for row in compression)
        pareto_confirmed = sum(row.get("pareto_claim_confirmed") == "True" for row in compression)
        lines.append(f"Compression: {storage_confirmed}/{len(compression)} grouped storage claims confirmed; {latency_evaluable}/{len(compression)} latency claims evaluable; {pareto_confirmed}/{len(compression)} Pareto claims confirmed.")
        lines.append("")
    lines.extend(["## Paired confidence intervals and component attribution", ""])
    for stage in ("ablation", "zeroshot", "cifar", "sample_efficiency"):
        for row in rows(OUT / stage / "paired.csv"):
            lines.append(
                f"- `{stage}/{row.get('comparison', '')}`: mean delta {row.get('mean_delta', '')}, 95% paired bootstrap CI [{row.get('ci_low', '')}, {row.get('ci_high', '')}], collections {row.get('collections', '')}."
            )
    for row in rows(OUT / "ablation" / "component_deltas.csv"):
        if row.get("comparison") == "equivariant_chart_minus_best_ordinary":
            lines.append(f"- `ablation/{row['comparison']}`: mean delta {row['mean_delta']}, 95% paired bootstrap CI [{row['ci_low']}, {row['ci_high']}].")
    lines.extend(["", "## Gate status and negative findings", ""])
    for stage in ("ablation", "zeroshot", "cifar", "cost"):
        for row in stage_claims(stage):
            lines.append(f"- `{stage}/{row.get('claim', '')}`: {row.get('value', '')}.")
    for row in compression:
        if row.get("overall_claim_confirmed") != "True":
            lines.append(f"- `compression/{row.get('group')}/{row.get('method')}/{row.get('target_storage_reduction')}`: no overall compression claim confirmed.")
    lines.extend(
        [
            "",
            "Strict zero-shot status is the recorded `zeroshot/strict_zeroshot_gate_passed` value. CIFAR transfer status is the recorded discovery gate and conditional-confirmation status. End-to-end cost status is determined only from complete-path timing in `cost/`. Storage, latency, and Pareto compression statuses are separate fields in `compression/claims.csv`.",
            "",
            "## Artifact index",
            "",
            "Machine-readable paths and SHA-256 values are in `reports/chart_followup/experiment_manifest.csv`, `experiment_manifest.json`, and `artifact_checksums.csv`. Local unpublished checkpoints are listed in `checkpoint_manifest.csv`.",
        ]
    )
    (OUT / "final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    normalize_stage_csv_line_endings()
    datasets = []
    for name in ("FashionMNIST", "cifar-10-batches-py"):
        path = DATA / name
        files = [candidate for candidate in path.rglob("*") if candidate.is_file()] if path.exists() else []
        datasets.append(
            {
                "dataset": "FashionMNIST" if name == "FashionMNIST" else "CIFAR10",
                "path": f"data/{name}",
                "exists": path.exists(),
                "files": len(files),
                "bytes": sum(candidate.stat().st_size for candidate in files),
                "root_configuration": "TWISTEDMERGE_DATA_ROOT or repository data directory",
                "role": "local_read_only_dataset",
            }
        )
    write_csv(OUT / "dataset_manifest.csv", datasets)
    write_json(OUT / "dataset_manifest.json", {"datasets": datasets})
    checkpoints = []
    checkpoint_root = TMP / "checkpoints"
    if checkpoint_root.exists():
        for path in sorted(checkpoint_root.rglob("*.pt")):
            checkpoints.append(
                {
                    "path": str(path.relative_to(ROOT)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "published": False,
                    "role": "local_trained_model_state",
                }
            )
    write_csv(OUT / "checkpoint_manifest.csv", checkpoints)
    write_json(
        OUT / "environment.json",
        {
            "python": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "execution_commit": git_head(),
            "smoke": arguments.smoke,
        },
    )
    factual_final_report(arguments.smoke)
    artifacts: list[dict[str, object]] = []
    for path in sorted(OUT.rglob("*")):
        if not path.is_file() or path.name in {"artifact_checksums.csv", "experiment_manifest.csv", "experiment_manifest.json"}:
            continue
        artifacts.append(
            {
                "path": str(path.relative_to(ROOT)),
                "stage": path.relative_to(OUT).parts[0],
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "role": path.suffix.lstrip(".") or "artifact",
                "execution_commit": git_head(),
            }
        )
    write_csv(OUT / "experiment_manifest.csv", artifacts)
    write_json(
        OUT / "experiment_manifest.json",
        {
            "execution_commit": git_head(),
            "smoke": arguments.smoke,
            "artifacts": artifacts,
        },
    )
    # Checksums are produced after all other manifest files so every published
    # evidence artifact except this self-referential ledger is covered.
    checksums = []
    for path in sorted(OUT.rglob("*")):
        if path.is_file() and path.name != "artifact_checksums.csv":
            checksums.append({"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    write_csv(OUT / "artifact_checksums.csv", checksums)


if __name__ == "__main__":
    main()
