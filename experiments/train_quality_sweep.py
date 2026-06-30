#!/usr/bin/env python
"""Sweep individual model training quality for small mergeable networks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.model_merging_benchmark import parse_csv, split_train_val  # noqa: E402
from src.metrics import capture_environment, save_json  # noqa: E402
from src.model_merging_benchmark import (  # noqa: E402
    device_from_arg,
    evaluate_model,
    format_markdown_table,
    load_dataset,
    make_loader,
    make_model,
    set_seed,
    train_model,
)


def setting_id(row: dict) -> str:
    return (
        f"{row['dataset']}_{row['architecture']}_W{row['width']}_E{row['epochs']}"
        f"_LR{row['lr']}_N{row['max_train_samples']}_{row['optimizer']}_{row['scheduler']}_{row['augmentation']}"
    )


def run_one(args, dataset_name: str, architecture: str, width: int, epochs: int, lr: float, max_train_samples: int, seed: int) -> dict:
    device = device_from_arg(args.device)
    dataset_seed = args.dataset_seed + 17 * max_train_samples
    spec, train_base, test_base = load_dataset(
        dataset_name,
        args.data_dir,
        max_train_samples,
        args.max_test_samples,
        dataset_seed,
        augmentation=args.augmentation,
    )
    train_subset, val_subset = split_train_val(train_base, args.val_fraction, seed)
    train_loader = make_loader(train_subset, args.batch_size, shuffle=True, seed=seed + 101)
    val_loader = make_loader(val_subset, args.batch_size, shuffle=False, seed=seed + 202)
    test_loader = make_loader(test_base, args.batch_size, shuffle=False, seed=seed + 303)
    set_seed(seed)
    model = make_model(architecture, spec, width)
    train_model(
        model,
        train_loader,
        epochs,
        lr,
        device,
        optimizer=args.optimizer,
        weight_decay=args.weight_decay,
        scheduler=args.scheduler,
        step_size=args.step_size,
        gamma=args.gamma,
    )
    train_metrics = evaluate_model(model, train_loader, device)
    val_metrics = evaluate_model(model, val_loader, device)
    test_metrics = evaluate_model(model, test_loader, device)
    row = {
        "dataset": spec.name,
        "requested_dataset": dataset_name,
        "architecture": architecture,
        "width": width,
        "epochs": epochs,
        "lr": lr,
        "max_train_samples": max_train_samples,
        "max_test_samples": args.max_test_samples,
        "val_fraction": args.val_fraction,
        "batch_size": args.batch_size,
        "optimizer": args.optimizer,
        "weight_decay": args.weight_decay,
        "scheduler": args.scheduler,
        "step_size": args.step_size,
        "gamma": args.gamma,
        "augmentation": args.augmentation,
        "seed": seed,
        "device": str(device),
        "train_loss": train_metrics["loss"],
        "train_accuracy": train_metrics["accuracy"],
        "val_loss": val_metrics["loss"],
        "val_accuracy": val_metrics["accuracy"],
        "test_loss": test_metrics["loss"],
        "test_accuracy": test_metrics["accuracy"],
    }
    row["setting_id"] = setting_id(row)
    return row


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "dataset",
        "architecture",
        "width",
        "epochs",
        "lr",
        "max_train_samples",
        "optimizer",
        "scheduler",
        "augmentation",
    ]
    return (
        rows.groupby(group_cols, dropna=False)["test_accuracy"]
        .agg(["count", "mean", "std", "min", "max"])
        .reset_index()
        .sort_values(["dataset", "mean", "min"], ascending=[True, False, False])
    )


def row_identity_columns() -> list[str]:
    return [
        "dataset",
        "architecture",
        "width",
        "epochs",
        "lr",
        "max_train_samples",
        "max_test_samples",
        "optimizer",
        "weight_decay",
        "scheduler",
        "augmentation",
        "seed",
    ]


def recommendation(summary: pd.DataFrame, dataset: str, threshold: float | None = None) -> dict:
    part = summary[summary["dataset"] == dataset].copy()
    if part.empty:
        return {
            "dataset": dataset,
            "status": "not_evaluated",
            "recommendation": "No rows were produced for this dataset.",
        }
    best = part.sort_values(["mean", "min", "count"], ascending=[False, False, False]).iloc[0].to_dict()
    status = "recommended"
    if threshold is not None and float(best["mean"]) < threshold:
        status = "below_threshold"
    best["dataset"] = dataset
    best["status"] = status
    return best


def verification_command(best: dict) -> str:
    return (
        ".venv/bin/python experiments/model_merging_benchmark.py --mode verification "
        f"--datasets {best['dataset']} --architecture {best['architecture']} "
        "--model-counts 3,4 "
        f"--widths {int(best['width'])} --domain-shifts none "
        "--seeds 1000,1001,1002,1003,1004 "
        f"--epochs {int(best['epochs'])} --max-train-samples {int(best['max_train_samples'])} "
        "--max-test-samples 1000 --batch-size 128 "
        f"--lr {best['lr']} --optimizer {best['optimizer']} --scheduler {best['scheduler']} "
        "--device cpu --alignment-noise-levels 0.05,0.15,0.30 --bootstrap-samples 500"
    )


def write_report(args, rows: pd.DataFrame, summary: pd.DataFrame, report_path: Path) -> None:
    mnist = recommendation(summary, "mnist", threshold=0.90)
    fashion = recommendation(summary, "fashion_mnist", threshold=None)
    rec_rows = [mnist, fashion]
    rec_table = pd.DataFrame(rec_rows).fillna("")
    rec_cols = [
        "dataset",
        "status",
        "architecture",
        "width",
        "epochs",
        "lr",
        "max_train_samples",
        "optimizer",
        "scheduler",
        "augmentation",
        "count",
        "mean",
        "min",
        "max",
    ]
    rec_cols = [col for col in rec_cols if col in rec_table.columns]
    summary_cols = [
        "dataset",
        "architecture",
        "width",
        "epochs",
        "lr",
        "max_train_samples",
        "optimizer",
        "scheduler",
        "augmentation",
        "count",
        "mean",
        "std",
        "min",
        "max",
    ]
    if mnist.get("status") == "recommended":
        default_note = (
            "MNIST clears the 0.90 mean individual-accuracy gate in this sweep. "
            "The verification defaults may use this measured setting."
        )
        command = verification_command(mnist)
    else:
        default_note = (
            "No MNIST setting in this sweep cleared the 0.90 mean individual-accuracy gate. "
            "Do not update verification defaults from this run."
        )
        command = ""
    fashion_note = (
        "Fashion-MNIST is reported as best feasible in this bounded sweep; no threshold is forced."
        if fashion.get("status") != "not_evaluated"
        else "Fashion-MNIST was not evaluated."
    )
    actual_datasets = ",".join(sorted(rows["dataset"].astype(str).unique()))
    actual_architectures = ",".join(sorted(rows["architecture"].astype(str).unique()))
    actual_widths = ",".join(str(item) for item in sorted(rows["width"].astype(int).unique()))
    actual_epochs = ",".join(str(item) for item in sorted(rows["epochs"].astype(int).unique()))
    actual_lrs = ",".join(str(item) for item in sorted(rows["lr"].astype(float).unique()))
    actual_max_train = ",".join(str(item) for item in sorted(rows["max_train_samples"].astype(int).unique()))
    actual_seeds = ",".join(str(item) for item in sorted(rows["seed"].astype(int).unique()))
    append_note = (
        "This report was regenerated with `--append`, so the tables summarize the existing CSV plus the last command above."
        if args.append
        else "This report summarizes exactly the command above."
    )
    report = f"""# Training Quality Sweep Report

Generated by `experiments/train_quality_sweep.py`.

## Last Command

```bash
{args.command_string}
```

{append_note}

## Scope

- Datasets in CSV: `{actual_datasets}`
- Architectures in CSV: `{actual_architectures}`
- Widths in CSV: `{actual_widths}`
- Epoch grid in CSV: `{actual_epochs}`
- LR grid in CSV: `{actual_lrs}`
- Max-train-sample grid in CSV: `{actual_max_train}`
- Seeds in CSV: `{actual_seeds}`
- Optimizer: `{args.optimizer}`
- Scheduler: `{args.scheduler}`
- Augmentation: `{args.augmentation}`
- Device: `{args.device}`

This sweep reports individual-model quality only. It does not evaluate model
merging success and should not be cited as a merge-improvement result.

## Outputs

- CSV: `reports/csv/training_quality_sweep.csv`
- Config: `reports/configs/training_quality_sweep_config.json`
- Report: `reports/training_quality_sweep_report.md`

## Recommended Settings

{format_markdown_table(rec_table.to_dict("records"), rec_cols)}

{default_note}

{fashion_note}

## Recommended Verification Command

```bash
{command}
```

## Summary Table

{format_markdown_table(summary.to_dict("records"), summary_cols)}

## Accuracy Gate

- MNIST acceptance target: at least one CPU-feasible setting with mean individual
  test accuracy preferably above `0.90`.
- Fashion-MNIST: report the best feasible setting; do not force a success claim
  if accuracy is weak.

## Environment

```json
{json.dumps(capture_environment(), indent=2)}
```
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="mnist,fashion_mnist")
    parser.add_argument("--architectures", default="mlp2,small_cnn")
    parser.add_argument("--widths", default="64,128")
    parser.add_argument("--epochs-grid", default="5")
    parser.add_argument("--lrs", default="0.001")
    parser.add_argument("--max-train-samples-grid", default="5000")
    parser.add_argument("--max-test-samples", type=int, default=1000)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--optimizer", default="adamw", choices=["adam", "adamw", "sgd"])
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--scheduler", default="cosine", choices=["none", "cosine", "step"])
    parser.add_argument("--step-size", type=int, default=3)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--augmentation", default="none", choices=["none", "light"])
    parser.add_argument("--seeds", default="1000,1001,1002")
    parser.add_argument("--dataset-seed", type=int, default=8128)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "reports")
    parser.add_argument("--append", action="store_true")
    args = parser.parse_args()
    args.command_string = " ".join([sys.executable, *sys.argv])

    rows = []
    for dataset_name in parse_csv(args.datasets, str):
        for architecture in parse_csv(args.architectures, str):
            for width in parse_csv(args.widths, int):
                for epochs in parse_csv(args.epochs_grid, int):
                    for lr in parse_csv(args.lrs, float):
                        for max_train_samples in parse_csv(args.max_train_samples_grid, int):
                            for seed in parse_csv(args.seeds, int):
                                row = run_one(args, dataset_name, architecture, width, epochs, lr, max_train_samples, seed)
                                print(
                                    f"{row['setting_id']} seed={seed} test_acc={row['test_accuracy']:.4f}",
                                    flush=True,
                                )
                                rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = args.reports_dir / "csv" / "training_quality_sweep.csv"
    if args.append and csv_path.exists():
        previous = pd.read_csv(csv_path)
        df = pd.concat([previous, df], ignore_index=True)
        identity = [col for col in row_identity_columns() if col in df.columns]
        df = df.drop_duplicates(subset=identity, keep="last")
    summary = summarize(df)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    save_json(
        args.reports_dir / "configs" / "training_quality_sweep_config.json",
        {
            "argv": sys.argv,
            "args": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
                if key != "command_string"
            },
            "environment": capture_environment(),
        },
    )
    report_path = args.reports_dir / "training_quality_sweep_report.md"
    write_report(args, df, summary, report_path)
    print(f"wrote {csv_path}")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
