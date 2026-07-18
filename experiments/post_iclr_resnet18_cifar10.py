#!/usr/bin/env python3
"""Staged base-quality gate for the post-ICLR ResNet-18/CIFAR-10 phase."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cifar_resnet_benchmark import (  # noqa: E402
    TrainingRecipe,
    checkpoint_manifest,
    cifar10_train_val_loaders,
    dataset_archive_metadata,
    resolve_device,
    train_resnet18,
)


PHASE = "resnet18_cifar10"
PILOT_SEEDS = (25100, 25101, 25102)
BASE_MEAN_GATE = 0.92
BASE_MIN_GATE = 0.90
BASE_STD_GATE = 0.015


def write_csv(path: Path, rows: list[dict], fields: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def parse_seeds(text: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in text.split(",") if item.strip())


def environment() -> dict:
    packages = {}
    for name in ("numpy", "pandas", "matplotlib", "torch", "torchvision"):
        module = __import__(name)
        packages[name] = getattr(module, "__version__", "unknown")
    return {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "packages": packages,
    }


def stage_recipe(args) -> tuple[TrainingRecipe, tuple[int, ...], int, int]:
    if args.stage == "smoke":
        recipe = TrainingRecipe(
            epochs=args.epochs or 1,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            warmup_epochs=0,
            validation_size=5000,
            split_seed=args.split_seed,
            num_workers=args.num_workers,
        )
        return recipe, parse_seeds(args.seeds or "25100"), args.train_limit or 2048, args.validation_limit or 512
    if args.stage == "pilot":
        recipe = TrainingRecipe(
            epochs=args.epochs or 150,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            warmup_epochs=args.warmup_epochs,
            validation_size=5000,
            split_seed=args.split_seed,
            num_workers=args.num_workers,
        )
        return recipe, parse_seeds(args.seeds or ",".join(map(str, PILOT_SEEDS))), 0, 0
    raise ValueError("confirmatory merging remains gated on a successful frozen pilot")


def gate_status(runs: pd.DataFrame, expected_seeds: int, stage: str) -> dict:
    if runs.empty:
        return {
            "stage": stage,
            "status": "implementation_or_compute_failure",
            "reason": "no completed model rows",
            "gate_interpretable": False,
        }
    values = runs["validation_accuracy"].to_numpy(float)
    mean = float(values.mean())
    minimum = float(values.min())
    standard_deviation = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    complete = len(runs) == expected_seeds
    passes = (
        stage == "pilot"
        and complete
        and mean >= BASE_MEAN_GATE
        and minimum >= BASE_MIN_GATE
        and standard_deviation <= BASE_STD_GATE
    )
    return {
        "stage": stage,
        "status": "base_quality_gate_passed" if passes else "smoke_only" if stage == "smoke" else "base_quality_gate_failed",
        "gate_interpretable": stage == "pilot" and complete,
        "completed_models": len(runs),
        "expected_models": expected_seeds,
        "validation_accuracy_mean": mean,
        "validation_accuracy_min": minimum,
        "validation_accuracy_std": standard_deviation,
        "mean_gate": BASE_MEAN_GATE,
        "minimum_gate": BASE_MIN_GATE,
        "standard_deviation_gate": BASE_STD_GATE,
        "test_evaluations": 0,
    }


def make_plot(output: Path, history: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = output / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))
    for seed, group in history.groupby("seed"):
        axes[0].plot(group["epoch"], group["train_accuracy"], marker="o", alpha=0.75, label=f"train {seed}")
        axes[0].plot(group["epoch"], group["validation_accuracy"], marker="o", linewidth=2, label=f"val {seed}")
        axes[1].plot(group["epoch"], group["train_loss"], marker="o", label=str(seed))
    axes[0].axhline(BASE_MEAN_GATE, color="#dc2626", linestyle="--", linewidth=1, label="0.92 gate")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend(fontsize=7, ncol=2)
    axes[1].set_title("Training loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Cross-entropy")
    axes[1].legend(fontsize=8, title="seed")
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.suptitle("CIFAR-10 ResNet-18 base-quality stage")
    fig.tight_layout()
    fig.savefig(plot_dir / "training_curves.png", dpi=220)
    fig.savefig(plot_dir / "training_curves.pdf")
    plt.close(fig)


def markdown_table(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for _, row in frame[columns].iterrows():
        rendered = []
        for column in columns:
            value = row[column]
            rendered.append(f"{value:.6g}" if isinstance(value, (float, np.floating)) else str(value))
        lines.append("| " + " | ".join(rendered) + " |")
    return "\n".join(lines)


def report_text(
    args,
    recipe: TrainingRecipe,
    runs: pd.DataFrame,
    gate: dict,
    failures: pd.DataFrame,
    loader_metadata: dict | None,
) -> str:
    table = markdown_table(
        runs,
        [
            "seed",
            "validation_accuracy",
            "validation_nll",
            "validation_ece",
            "validation_brier",
            "validation_worst_class_accuracy",
            "best_epoch",
        ],
    ) if not runs.empty else "No model completed."
    training_examples = loader_metadata.get("training_examples", 0) if loader_metadata else 0
    validation_examples = loader_metadata.get("validation_examples", 0) if loader_metadata else 0
    return f"""# CIFAR-10 ResNet-18 base-quality {args.stage}

## Verdict

Status: **{gate['status']}**.

This stage evaluates individual-model training quality only. It is not a model-merging result, and the CIFAR-10 test partition was not loaded or evaluated. A confirmatory merge run remains forbidden unless the validation-only pilot gate passes and the recipe is frozen.

## Protocol

- Architecture: torchvision ResNet-18 with `3x3`, stride-1 CIFAR stem and no max-pool.
- Training: SGD, momentum `{recipe.momentum}`, Nesterov, weight decay `{recipe.weight_decay}`, cosine decay, `{recipe.warmup_epochs}` warmup epochs.
- Data: deterministic train/validation split with `{training_examples}` training and `{validation_examples}` validation examples in this stage; random crop with padding 4, horizontal flip, and channel normalization on training only.
- Epochs: `{recipe.epochs}`; batch size: `{recipe.batch_size}`; initial learning rate: `{recipe.learning_rate}`.
- Model seeds: `{','.join(map(str, runs['seed'].tolist())) if not runs.empty else ''}`.
- Failures: `{len(failures)}`.
- Base gate: mean validation accuracy >= `{BASE_MEAN_GATE}`, minimum >= `{BASE_MIN_GATE}`, and seed standard deviation <= `{BASE_STD_GATE}`.

## Individual models

{table}

## Aggregate gate

```json
{json.dumps(gate, indent=2, sort_keys=True)}
```

## Leakage boundary

The pilot chooses and freezes the training recipe using the held-out validation partition only. There are zero test evaluations in the epoch log and resource ledger. Final test evaluation is reserved for a later frozen confirmatory protocol.

## Reproduction

```bash
{sys.executable} experiments/post_iclr_resnet18_cifar10.py --stage {args.stage}
```

![Training curves](plots/training_curves.png)
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("smoke", "pilot", "confirmatory"), required=True)
    parser.add_argument("--seeds", default="")
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--split-seed", type=int, default=24680)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--validation-limit", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--output-root", type=Path, default=ROOT / "reports" / "post_iclr_v2" / PHASE)
    parser.add_argument("--checkpoint-root", type=Path, default=ROOT / "checkpoints" / "post_iclr_resnet18_cifar10")
    args = parser.parse_args()
    recipe, seeds, train_limit, validation_limit = stage_recipe(args)
    output = args.output_root / "stages" / args.stage
    checkpoint_stage = args.checkpoint_root / args.stage
    output.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    run_rows: list[dict] = []
    history_rows: list[dict] = []
    resource_rows: list[dict] = []
    failures: list[dict] = []
    checkpoint_paths: list[Path] = []
    loader_metadata = None
    for seed in seeds:
        try:
            train_loader, validation_loader, loader_metadata = cifar10_train_val_loaders(
                args.data_root,
                recipe,
                model_seed=seed,
                train_limit=train_limit,
                validation_limit=validation_limit,
                download=True,
            )
            _model, history, result = train_resnet18(
                seed=seed,
                recipe=recipe,
                train_loader=train_loader,
                validation_loader=validation_loader,
                device=device,
                checkpoint_dir=checkpoint_stage / f"seed_{seed}",
            )
            history_rows.extend(history)
            validation = result["validation"]
            best = max(history, key=lambda row: row["validation_accuracy"])
            run_rows.append(
                {
                    "seed": seed,
                    "validation_accuracy": validation["accuracy"],
                    "validation_nll": validation["nll"],
                    "validation_ece": validation["ece"],
                    "validation_brier": validation["brier"],
                    "validation_worst_class_accuracy": validation["worst_class_accuracy"],
                    "best_epoch": best["epoch"],
                    "training_examples": loader_metadata["training_examples"],
                    "validation_examples": loader_metadata["validation_examples"],
                    "test_evaluations": 0,
                }
            )
            resource_rows.append(result["resources"])
            checkpoint_paths.extend(
                [checkpoint_stage / f"seed_{seed}" / "best.pt", checkpoint_stage / f"seed_{seed}" / "last.pt"]
            )
        except Exception as error:
            failures.append(
                {
                    "seed": seed,
                    "stage": args.stage,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            print(f"seed={seed} failed: {type(error).__name__}: {error}", file=sys.stderr, flush=True)

    runs = pd.DataFrame(run_rows)
    history = pd.DataFrame(history_rows)
    failure_frame = pd.DataFrame(failures, columns=["seed", "stage", "error_type", "error"])
    gate = gate_status(runs, len(seeds), args.stage)
    write_csv(output / "runs.csv", run_rows)
    write_csv(output / "epoch_log.csv", history_rows)
    write_csv(output / "resource_accounting.csv", resource_rows)
    write_csv(output / "failure_log.csv", failure_frame.to_dict(orient="records"), list(failure_frame.columns))
    manifest_rows = checkpoint_manifest(path for path in checkpoint_paths if path.exists())
    for row in manifest_rows:
        path = Path(row["path"])
        row["path"] = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
    write_csv(output / "checkpoint_manifest.csv", manifest_rows)
    if not history.empty:
        make_plot(output, history)
    (output / "gate_status.json").write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    config = {
        "phase": PHASE,
        "stage": args.stage,
        "command": " ".join([sys.executable, *sys.argv]),
        "git_commit_at_execution": git_output("rev-parse", "HEAD"),
        "git_worktree_dirty_at_execution": bool(git_output("status", "--porcelain")),
        "device": str(device),
        "recipe": recipe.to_dict(),
        "seeds": list(seeds),
        "train_limit": train_limit,
        "validation_limit": validation_limit,
        "loader_metadata": loader_metadata,
        "dataset_archive": dataset_archive_metadata(args.data_root),
        "test_partition_loaded": False,
        "environment": environment(),
        "gates": {"mean": BASE_MEAN_GATE, "minimum": BASE_MIN_GATE, "standard_deviation": BASE_STD_GATE},
    }
    (output / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "report.md").write_text(
        report_text(args, recipe, runs, gate, failure_frame, loader_metadata), encoding="utf-8"
    )
    if gate["status"] == "base_quality_gate_passed":
        frozen = {
            "frozen_after_stage": "pilot",
            "recipe": recipe.to_dict(),
            "pilot_seeds": list(seeds),
            "pilot_gate": gate,
            "confirmatory_group_seeds": [
                [26000 + 10 * group + model for model in range(4)] for group in range(5)
            ],
            "model_counts": [3, 4],
            "test_evaluation_policy": "once per frozen model and merge candidate after all validation-only choices",
            "git_commit_at_pilot_execution": git_output("rev-parse", "HEAD"),
        }
        args.output_root.mkdir(parents=True, exist_ok=True)
        (args.output_root / "frozen_config.json").write_text(
            json.dumps(frozen, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    artifact_paths = [
        path for path in output.rglob("*") if path.is_file() and path.name != "artifact_manifest.csv"
    ]
    artifact_paths.extend(
        [
            ROOT / "experiments" / "post_iclr_resnet18_cifar10.py",
            ROOT / "src" / "cifar_resnet_benchmark.py",
            ROOT / "tests" / "test_cifar_resnet_benchmark.py",
        ]
    )
    artifact_rows = checkpoint_manifest(artifact_paths)
    for row in artifact_rows:
        path = Path(row["path"])
        row["path"] = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
    write_csv(output / "artifact_manifest.csv", artifact_rows)
    phase_paths = [
        path
        for path in args.output_root.rglob("*")
        if path.is_file() and path != args.output_root / "artifact_manifest.csv"
    ]
    phase_rows = checkpoint_manifest(phase_paths)
    for row in phase_rows:
        path = Path(row["path"])
        row["path"] = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
    write_csv(args.output_root / "artifact_manifest.csv", phase_rows)
    print(json.dumps({"stage": args.stage, "completed": len(runs), "failures": len(failures), "gate": gate}, sort_keys=True))


if __name__ == "__main__":
    main()
