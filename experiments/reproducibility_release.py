#!/usr/bin/env python3
"""C8: environment, data, checkpoint, checksum, and experiment manifests."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.future_text_common import DATASETS, MODEL_ID, MODEL_REVISION
from experiments.next_program_common import DATA, OUT, TMP, environment_record, git_head, sha256_file, write_csv, write_json

SCRIPT = Path(__file__).resolve()


def read(path: Path) -> list[dict[str, str]]:
    if not path.exists(): return []
    with path.open(encoding="utf-8", newline="") as handle: return list(csv.DictReader(handle))


def stage_for(path: Path) -> tuple[str, str, str]:
    name = path.name
    mappings = (
        (("chart",), "A1", "experiments/trained_chart_inference.py"),
        (("cost",), "A2", "experiments/end_to_end_controlled_cost.py"),
        (("refinement",), "A3", "experiments/nontrivial_refinement_invariance.py"),
        (("full_model",), "B1", "experiments/full_model_hidden_geometry.py"),
        (("composition",), "B2", "experiments/learned_compositional_baselines.py"),
        (("multiview",), "B3", "experiments/genuine_multiview_retransport.py"),
        (("natural",), "B4", "experiments/new_realistic_residual_search.py"),
        (("compression",), "B5", "experiments/structured_compression.py"),
        (("central", "projective"), "B6", "experiments/noncyclic_central_extensions.py"),
        (("baseline",), "B7", "experiments/official_baseline_integration.py"),
        (("conditional",), "C1_C2", "experiments/conditional_extended_families.py"),
        (("language",), "C3", "experiments/language_checkpoint_transition_geometry.py"),
        (("complex", "alignment"), "C4_C5", "experiments/comparison_alignment_robustness.py"),
        (("activation",), "C6", "experiments/selective_activation_diagnostics.py"),
        (("scaling",), "C7", "experiments/real_scaling_audit.py"),
    )
    for prefixes, stage, script in mappings:
        if name.startswith(prefixes): return stage, script, prefixes[0]
    # Tables and plots inherit the prefix from their stem.
    for prefixes, stage, script in mappings:
        if path.stem.startswith(prefixes): return stage, script, prefixes[0]
    return "C8", "experiments/reproducibility_release.py", "release"


def dataset_manifest():
    rows = []
    candidates = [
        ("FashionMNIST", "torchvision", DATA / "FashionMNIST" / "raw" / "train-images-idx3-ubyte"),
        ("CIFAR10", "torchvision", DATA / "cifar-10-python.tar.gz"),
        ("ModelNet10", "http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip", DATA / "ModelNet10.zip"),
    ]
    for name, source, path in candidates:
        rows.append({"dataset": name, "source": source, "local_path": str(path), "sha256": sha256_file(path) if path.exists() and path.is_file() else "directory_or_missing", "available": path.exists()})
    for name, dataset_id, revision in DATASETS:
        rows.append({"dataset": name, "source": dataset_id, "local_path": str(DATA / "huggingface"), "sha256": revision, "available": True})
    return rows


def checkpoint_manifest():
    rows = []
    for path in sorted((TMP / "checkpoints").rglob("*")) if (TMP / "checkpoints").exists() else []:
        if path.is_file(): rows.append({"artifact": str(path.relative_to(ROOT)), "artifact_type": "checkpoint", "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    for path in sorted((TMP / "compression").glob("*.npz")) if (TMP / "compression").exists() else []:
        rows.append({"artifact": str(path.relative_to(ROOT)), "artifact_type": "compressed_student", "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return rows


def claims():
    output = []
    for path in sorted(OUT.rglob("*_claims.csv")):
        for row in read(path): output.append({"artifact": str(path.relative_to(ROOT)), **row})
    return output


def main() -> None:
    environment = environment_record(); freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True).stdout
    (OUT / "environment.lock.txt").write_text(f"execution_commit=={git_head()}\n" + "\n".join(f"{key}=={value}" for key, value in environment.items()) + "\n" + freeze, encoding="utf-8")
    datasets = dataset_manifest(); checkpoints = checkpoint_manifest(); write_csv(OUT / "dataset_manifest.csv", datasets); write_json(OUT / "dataset_manifest.json", datasets); write_csv(OUT / "checkpoint_manifest.csv", checkpoints, ["artifact", "artifact_type", "bytes", "sha256"])
    baseline_commits = read(OUT / "iclr" / "baseline_manifest.csv")
    baseline_fields = ["baseline", "repository", "commit", "license", "wrapper_modifications", "hyperparameters", "execution_command", "applicable_families", "status"]
    write_csv(OUT / "baseline_commit_manifest.csv", baseline_commits, baseline_fields)
    write_json(OUT / "baseline_commit_manifest.json", baseline_commits)
    release_tag = f"next-program-{git_head()[:12]}"
    write_json(OUT / "release.json", {"tag": release_tag, "target_execution_commit": git_head(), "status": "pending_final_evidence_commit_and_remote_tag_creation", "one_command": "scripts/reproduce_next_program.sh"})
    (OUT / "reproduction.md").write_text(
        "# Reproduction instructions\n\n"
        "Run `PYTHON_BIN=python scripts/reproduce_next_program.sh` from the repository root. Set `TWISTEDMERGE_DATA_ROOT` "
        "when datasets are stored outside `data/`. The runner supports `--tier immediate|iclr|extended|all`, `--resume`, "
        "and `--force-stage STAGE_ID`. Saved prediction tensors and checkpoints remain under ignored `reports/tmp/next_program/`; "
        "their hashes are recorded in committed ledgers.\n",
        encoding="utf-8",
    )
    exclusions = {"artifact_checksums.csv", "experiment_manifest.csv", "experiment_manifest.json", "final_experimental_report.md"}
    artifact_rows = []
    for path in sorted(OUT.rglob("*")):
        if path.is_file() and path.name not in exclusions and not path.name.endswith(".tmp"):
            artifact_rows.append({"artifact": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    write_csv(OUT / "artifact_checksums.csv", artifact_rows)
    checksums = {row["artifact"]: row["sha256"] for row in artifact_rows}
    commands = {row["stage"]: row for row in read(OUT / "commands.csv")}
    manifest = []
    for path in sorted(list((OUT / "immediate" / "tables").glob("*")) + list((OUT / "immediate" / "plots").glob("*")) + list((OUT / "iclr" / "tables").glob("*")) + list((OUT / "iclr" / "plots").glob("*")) + list((OUT / "extended" / "tables").glob("*")) + list((OUT / "extended" / "plots").glob("*"))):
        if not path.is_file(): continue
        stage, script, prefix = stage_for(path); directory = path.parent.parent
        raw_candidates = sorted(directory.glob(prefix + "*runs*.csv")) + sorted(directory.glob(prefix + "*transitions*.csv"))
        summary_candidates = sorted(directory.glob(prefix + "*summary*.csv")) + sorted(directory.glob(prefix + "*claims*.csv"))
        artifact = str(path.relative_to(ROOT))
        manifest.append({"stage": stage, "artifact": artifact, "artifact_type": path.suffix.lstrip("."), "script": script, "configuration": commands.get(stage, {}).get("exact_command", f"python {script}"), "raw_data": ";".join(str(value.relative_to(ROOT)) for value in raw_candidates), "summary_data": ";".join(str(value.relative_to(ROOT)) for value in summary_candidates), "execution_commit": git_head(), "sha256": checksums.get(artifact, sha256_file(path))})
    write_csv(OUT / "experiment_manifest.csv", manifest, ["stage", "artifact", "artifact_type", "script", "configuration", "raw_data", "summary_data", "execution_commit", "sha256"]); write_json(OUT / "experiment_manifest.json", manifest)
    status = json.loads((OUT / "status.json").read_text()) if (OUT / "status.json").exists() else {"stages": {}}
    failures = read(OUT / "failures.csv"); claim_rows = claims(); negative = [row for row in claim_rows if str(row.get("value", "")).lower() == "false"]
    paired_files = sorted(OUT.rglob("*_paired.csv")); paired_lines = []
    for path in paired_files:
        rows = read(path)
        if rows: paired_lines.append(f"- `{path.relative_to(ROOT)}`: {len(rows)} paired rows; first row `{json.dumps(rows[0], sort_keys=True)}`.")
    stage_lines = []
    for stage, record in status.get("stages", {}).items(): stage_lines.append(f"- {stage}: `{record.get('state', 'unknown')}`; runtime `{record.get('runtime_seconds', '')}` seconds; {record.get('summary', '')}.")
    failure_lines = [f"- {row['stage']}: `{row['summary']}`." for row in failures] or ["- None recorded."]
    negative_lines = [f"- `{row['artifact']}`: `{row.get('claim', '')}` = false." for row in negative] or ["- No false gate rows recorded."]
    (OUT / "final_experimental_report.md").write_text(
        "# Final experimental report\n\n"
        f"Execution commit: `{git_head()}`.\n\n"
        "## Stage status and protocol coverage\n\n" + "\n".join(stage_lines) + "\n\n"
        "All selected stages use separate training, transition/router, selector, calibration, and test roles where applicable. "
        "Candidate logits are saved before test-label metrics and checked after label permutation. Discovery gates control "
        "confirmation and conditional extensions.\n\n"
        "## Numerical paired results\n\n" + ("\n".join(paired_lines) if paired_lines else "- No paired files were produced.") + "\n\n"
        "## Actual implementations and boundaries\n\n"
        "- A1, A2, B1, B2, B3, B5, C3, C6, and C7 execute trained models or measured end-to-end numerical paths.\n"
        "- A3, B6, and C4 execute exact finite-algebra or comparison-complex calculations.\n"
        "- B1 names internal activation-alignment implementations explicitly; B7 does not relabel them as official baselines.\n"
        "- B4 reuses preregistered B1/B3 discovery artifacts without selecting on test accuracy.\n"
        "- C1/C2 and B7 remain gated when their prerequisites fail; gated rows are not counted as executions.\n\n"
        "## Failed attempts\n\n" + "\n".join(failure_lines) + "\n\n"
        "## Negative findings\n\n" + "\n".join(negative_lines) + "\n\n"
        "## Artifact paths\n\n"
        "- `reports/next_program/experiment_manifest.csv` maps tables and plots to scripts, raw data, summaries, execution commit, and checksums.\n"
        "- `reports/next_program/artifact_checksums.csv` contains artifact hashes.\n"
        "- `reports/next_program/test_results.txt` contains the executed test command and output.\n"
        "- `reports/next_program/commands.csv` and `reports/next_program/failures.csv` preserve command and failure provenance.\n",
        encoding="utf-8",
    )


if __name__ == "__main__": main()
