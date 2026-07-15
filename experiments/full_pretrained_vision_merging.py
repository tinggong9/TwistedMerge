#!/usr/bin/env python3
"""Stage 8 wrapper for a resource-bounded pretrained ResNet-18 smoke."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "overnight_program"
SOURCE = OUT / "pretrained_vision_source"
DEFAULT_DATA = ROOT / "data"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    args = parser.parse_args()
    if args.mode == "full":
        raise RuntimeError("full mode requires partial/full backbone fine-tuning and official external baseline integration; see blocker report")
    command = [
        sys.executable,
        str(ROOT / "experiments" / "pretrained_merge_smoke.py"),
        "--seed", "0",
        "--train-samples", "512",
        "--validation-samples", "256",
        "--test-samples", "512",
        "--head-epochs", "30",
        "--data-dir", str(DEFAULT_DATA),
        "--out-dir", str(SOURCE),
    ]
    subprocess.run(command, cwd=ROOT, check=True)
    runs = pd.read_csv(SOURCE / "pretrained_merge_runs.csv")
    summary = runs.copy()
    paired_rows = []
    baseline = float(runs.loc[runs.method == "weight_average", "average_accuracy"].iloc[0])
    for row in runs.itertuples():
        paired_rows.append({"method": row.method, "baseline": "weight_average", "n_pairs": 1, "mean_accuracy_delta": row.average_accuracy - baseline, "ci_low": row.average_accuracy - baseline, "ci_high": row.average_accuracy - baseline})
    paired = pd.DataFrame(paired_rows)
    metadata = pd.read_csv(SOURCE / "pretrained_merge_baseline_metadata.csv")
    choices = runs[runs.selected_by_validation][["seed", "method", "selector_source_method", "selection_budget"]]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tables").mkdir(exist_ok=True)
    (OUT / "plots").mkdir(exist_ok=True)
    runs.to_csv(OUT / "pretrained_vision_runs.csv", index=False)
    summary.to_csv(OUT / "pretrained_vision_summary.csv", index=False)
    paired.to_csv(OUT / "pretrained_vision_paired_stats.csv", index=False)
    metadata.to_csv(OUT / "pretrained_vision_baseline_metadata.csv", index=False)
    choices.to_csv(OUT / "pretrained_vision_choices.csv", index=False)
    summary[["method", "average_accuracy", "worst_task_accuracy", "calibration_ece", "forgetting_interference"]].to_latex(OUT / "tables" / "pretrained_vision.tex", index=False, float_format="%.4f")
    fig, ax = plt.subplots(figsize=(8, 5))
    ordered = summary.sort_values("average_accuracy")
    ax.barh(ordered.method, ordered.average_accuracy, color="#356c8d")
    ax.set_xlabel("CIFAR-10 smoke accuracy")
    ax.set_title("Frozen pretrained ResNet-18 head-merging smoke")
    fig.tight_layout()
    fig.savefig(OUT / "plots" / "pretrained_vision.pdf")
    plt.close(fig)
    source_config = json.loads((SOURCE / "pretrained_merge_config.json").read_text())
    config = {
        "stage": 8,
        "mode": "smoke",
        "execution_commit": source_config["git_commit"],
        "command": " ".join(command),
        "backbone": "torchvision ResNet-18 ImageNet weights",
        "dataset": "CIFAR-10",
        "seeds_completed": 1,
        "full_required_scale_completed": False,
        "label_permutation_regression_passed": source_config["label_permutation_regression_passed"],
    }
    (OUT / "pretrained_vision_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    report = f"""# Stage 8: pretrained vision merging smoke

One frozen-backbone ResNet-18/CIFAR-10 smoke completed with four specialized heads, a separate validation split, saved-logit leakage regression, Task Arithmetic, TIES, DARE, SLERP, greedy soup, weight averaging, and the validation-only TwistedMerge selector. It is feasibility evidence only.

Exact blockers: one seed rather than five; CIFAR-100 absent; the backbone is frozen rather than partially/fully fine-tuned; only class-group specialization is run; official Git Re-Basin, C2M3, RegMean, representation-alignment, and low-rank implementations are not integrated; official external code is pinned in metadata but internal implementations are used for Task Arithmetic/TIES/DARE. Full command is deliberately refused until those conditions are supplied. No obstruction certificate passed and no branch candidate activated.
"""
    (OUT / "pretrained_vision_report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"smoke_completed": True, "rows": len(runs), "selector": choices.selector_source_method.tolist()}, indent=2))


if __name__ == "__main__":
    main()
