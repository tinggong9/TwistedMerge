#!/usr/bin/env python3
"""B7: conditional manifest and execution boundary for official baselines."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.next_program_common import OUT, TMP, git_head, write_csv

DEST = OUT / "iclr"
BASELINES = (
    ("Model Soups", "https://github.com/mlfoundations/model-soups.git", "Apache-2.0"),
    ("Git Re-Basin", "https://github.com/samuela/git-re-basin.git", "MIT"),
    ("Task Arithmetic", "https://github.com/mlfoundations/task_vectors.git", "MIT"),
    ("TIES", "https://github.com/prateeky2806/ties-merging.git", "MIT"),
    ("DARE and RegMean", "https://github.com/yule-BUAA/MergeLM.git", "MIT"),
    ("RegMean", "https://github.com/bloomberg/dataless-model-merging.git", "Apache-2.0"),
)


def claim(path: Path, name: str) -> bool:
    if not path.exists():
        return False
    with path.open(encoding="utf-8", newline="") as handle:
        return any(row.get("claim") == name and row.get("value", "").lower() == "true" for row in csv.DictReader(handle))


def positive_families() -> list[str]:
    output = []
    if claim(DEST / "full_model_claims.csv", "complete_realistic_gate_passed"): output.append("CIFAR10_ResNet18")
    if claim(DEST / "multiview_claims.csv", "complete_multiview_gate_passed"): output.append("ModelNet10_multiview")
    if claim(DEST / "natural_claims.csv", "complete_new_family_gate_passed"): output.append("selected_new_natural_family")
    return output


def remote_head(repository: str) -> str:
    result = subprocess.run(["git", "ls-remote", repository, "HEAD"], capture_output=True, text=True, check=True)
    return result.stdout.split()[0]


def main() -> None:
    families = positive_families(); manifest = []
    for name, repository, license_name in BASELINES:
        if families:
            commit = remote_head(repository)
            status = "pinned_upstream_requires_family_specific_wrapper_execution"
        else:
            commit = "not_fetched_gate_closed"
            status = "gated_off_no_positive_realistic_family"
        manifest.append({"baseline": name, "repository": repository, "commit": commit, "license": license_name, "wrapper_modifications": "none" if not families else "not_executed", "hyperparameters": "not_applicable_gate_closed" if not families else "not_selected", "execution_command": "not_applicable_gate_closed" if not families else "", "applicable_families": ";".join(families), "status": status})
    write_csv(DEST / "baseline_manifest.csv", manifest)
    write_csv(DEST / "baseline_runs.csv", [], ["family", "baseline", "setting_id", "accuracy", "execution_commit"])
    write_csv(DEST / "baseline_paired.csv", [], ["family", "baseline", "mean_delta", "ci_low", "ci_high"])
    if families:
        # This is intentionally a hard evidence boundary: pinning a repository
        # is not an official baseline execution.
        raise RuntimeError("a realistic family passed; official family-specific wrappers must execute before B7 can complete")
    (DEST / "baseline_report.md").write_text(
        "# Official baseline integration\n\n"
        f"Execution commit: `{git_head()}`. No realistic discovery family passed its complete gate, so official baseline "
        "integration was not applicable. Upstream repositories are listed without fabricated run rows; no internal "
        "approximation is labeled as an official execution.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
