#!/usr/bin/env python
"""Small MLP/CNN model-merging benchmark on MNIST and CIFAR-10."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.metrics import capture_environment, save_json  # noqa: E402
from src.model_merging_benchmark import (  # noqa: E402
    DomainShiftDataset,
    average_models,
    compute_pairwise_permutations,
    cycle_score,
    device_from_arg,
    evaluate_ensemble,
    evaluate_model,
    format_markdown_table,
    greedy_soup,
    load_dataset,
    make_loader,
    make_model,
    permute_model_to_reference,
    rank_lifted_branch_models,
    require_torch,
    save_checkpoint,
    set_seed,
    synchronize_permutations,
    train_model,
)


def parse_csv(text: str, cast=str):
    return [cast(item.strip()) for item in text.split(",") if item.strip()]


def default_architecture(dataset: str) -> str:
    return "mlp" if dataset in {"mnist", "fake-mnist"} else "cnn"


def split_train_val(dataset, val_fraction: float, seed: int):
    torch, _, _ = require_torch()
    n_val = max(1, int(len(dataset) * val_fraction))
    n_train = len(dataset) - n_val
    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.utils.data.random_split(dataset, [n_train, n_val], generator=generator)


def run_setting(args, dataset_name: str, n_models: int, width: int, domain_shift: str) -> tuple[list[dict], list[dict]]:
    torch, _, _ = require_torch()
    device = device_from_arg(args.device)
    architecture = args.architecture if args.architecture != "auto" else default_architecture(dataset_name)
    setting_id = f"{dataset_name}_{architecture}_N{n_models}_W{width}_{domain_shift}"
    spec, train_base, test_base = load_dataset(
        dataset_name,
        args.data_dir,
        args.max_train_samples,
        args.max_test_samples,
        args.seed,
    )
    test_loader = make_loader(test_base, args.batch_size, shuffle=False, seed=args.seed + 999)

    models = []
    per_model_rows = []
    for model_idx in range(n_models):
        set_seed(args.seed + 1000 * model_idx + 17 * width + n_models)
        shifted = DomainShiftDataset(train_base, domain_shift, model_idx, n_models)
        train_subset, val_subset = split_train_val(shifted, args.val_fraction, args.seed + model_idx)
        train_loader = make_loader(train_subset, args.batch_size, shuffle=True, seed=args.seed + model_idx)
        val_loader_model = make_loader(val_subset, args.batch_size, shuffle=False, seed=args.seed + 100 + model_idx)
        model = make_model(architecture, spec, width)
        train_model(model, train_loader, args.epochs, args.lr, device)
        test_metrics = evaluate_model(model, test_loader, device)
        val_metrics = evaluate_model(model, val_loader_model, device)
        model.to("cpu")
        ckpt_path = args.reports_dir / "checkpoints" / setting_id / f"model_{model_idx}.pt"
        save_checkpoint(
            model,
            ckpt_path,
            {
                "dataset": dataset_name,
                "architecture": architecture,
                "model_index": model_idx,
                "n_models": n_models,
                "width": width,
                "domain_shift": domain_shift,
                "epochs": args.epochs,
                "seed": args.seed + 1000 * model_idx,
            },
        )
        models.append(model)
        per_model_rows.append(
            {
                "setting_id": setting_id,
                "dataset": dataset_name,
                "architecture": architecture,
                "n_models": n_models,
                "width": width,
                "domain_shift": domain_shift,
                "model_index": model_idx,
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "test_loss": test_metrics["loss"],
                "test_accuracy": test_metrics["accuracy"],
                "checkpoint": str(ckpt_path),
            }
        )

    val_loader = make_loader(train_base, args.batch_size, shuffle=False, seed=args.seed + 500)
    match_loader = make_loader(train_base, args.batch_size, shuffle=False, seed=args.seed + 501)
    pairwise = compute_pairwise_permutations(models, architecture, match_loader, device, args.matching)
    score, cycle_rows = cycle_score(pairwise, n_models, width)
    ref, synced, sync_disagreement = synchronize_permutations(pairwise, n_models)
    aligned_to_zero = [
        permute_model_to_reference(model, architecture, spec, width, pairwise[(0, idx)])
        for idx, model in enumerate(models)
    ]
    aligned_synced = [
        permute_model_to_reference(model, architecture, spec, width, synced[idx])
        for idx, model in enumerate(models)
    ]

    baseline_metrics: list[dict] = []
    model_metrics = [row["test_accuracy"] for row in per_model_rows]
    single_best = max(model_metrics)

    def add_baseline(name: str, metrics: dict, extra: dict | None = None) -> None:
        merged_degradation = single_best - metrics["accuracy"]
        row = {
            "setting_id": setting_id,
            "dataset": dataset_name,
            "architecture": architecture,
            "n_models": n_models,
            "width": width,
            "domain_shift": domain_shift,
            "matching": args.matching,
            "baseline": name,
            "loss": metrics["loss"],
            "accuracy": metrics["accuracy"],
            "single_best_accuracy": single_best,
            "merge_degradation": merged_degradation,
            "cycle_score": score,
            "sync_disagreement": sync_disagreement,
            "sync_reference": ref,
        }
        if extra:
            row.update(extra)
        baseline_metrics.append(row)

    weight_avg = average_models(models, architecture, spec, width)
    add_baseline("weight_average", evaluate_model(weight_avg, test_loader, device))

    soup, soup_indices, soup_metrics = greedy_soup(models, val_loader, test_loader, device, architecture, spec, width)
    add_baseline("greedy_soup", soup_metrics, {"soup_indices": json.dumps(soup_indices)})

    git_rebasin = average_models(aligned_to_zero, architecture, spec, width)
    add_baseline("git_rebasin_pairwise", evaluate_model(git_rebasin, test_loader, device))

    c2m3 = average_models(aligned_synced, architecture, spec, width)
    add_baseline("c2m3_cycle_consistent", evaluate_model(c2m3, test_loader, device))

    branches = rank_lifted_branch_models(
        aligned_synced,
        pairwise,
        args.rank_lift_branches,
        architecture,
        spec,
        width,
    )
    add_baseline(
        f"twisted_rank_lift_{len(branches)}",
        evaluate_ensemble(branches, test_loader, device),
        {"rank_lift_branches": len(branches)},
    )

    add_baseline("ensemble_upper_bound", evaluate_ensemble(models, test_loader, device))

    for name, model in [
        ("weight_average", weight_avg),
        ("greedy_soup", soup),
        ("git_rebasin_pairwise", git_rebasin),
        ("c2m3_cycle_consistent", c2m3),
    ]:
        save_checkpoint(
            model.to("cpu"),
            args.reports_dir / "checkpoints" / setting_id / f"{name}.pt",
            {"setting_id": setting_id, "baseline": name, "cycle_score": score},
        )
    for branch_idx, branch in enumerate(branches):
        save_checkpoint(
            branch.to("cpu"),
            args.reports_dir / "checkpoints" / setting_id / f"twisted_branch_{branch_idx}.pt",
            {"setting_id": setting_id, "baseline": "twisted_rank_lift", "branch_index": branch_idx},
        )

    cycle_csv = args.reports_dir / "csv" / "model_merging_cycle_defects.csv"
    cycle_rows = [
        {
            "setting_id": setting_id,
            "dataset": dataset_name,
            "architecture": architecture,
            "n_models": n_models,
            "width": width,
            "domain_shift": domain_shift,
            **row,
        }
        for row in cycle_rows
    ]
    if cycle_rows:
        cycle_df = pd.DataFrame(cycle_rows)
        if cycle_csv.exists():
            old = pd.read_csv(cycle_csv)
            cycle_df = pd.concat([old[old["setting_id"] != setting_id], cycle_df], ignore_index=True)
        cycle_csv.parent.mkdir(parents=True, exist_ok=True)
        cycle_df.to_csv(cycle_csv, index=False)

    return baseline_metrics, per_model_rows


def plot_cycle_scatter(df: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    part = df[df["baseline"].isin(["weight_average", "git_rebasin_pairwise", "c2m3_cycle_consistent"])]
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for baseline, group in part.groupby("baseline"):
        ax.scatter(group["cycle_score"], group["merge_degradation"], label=baseline, alpha=0.8)
    ax.set_xlabel("cycle obstruction score")
    ax.set_ylabel("single-best accuracy minus merged accuracy")
    ax.set_title("Cycle score versus merge degradation")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_ablation(df: pd.DataFrame, x_col: str, path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    part = df[df["baseline"].isin(["weight_average", "c2m3_cycle_consistent", "twisted_rank_lift_2", "ensemble_upper_bound"])]
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for baseline, group in part.groupby("baseline"):
        summary = group.groupby(x_col)["accuracy"].mean().reset_index()
        ax.plot(summary[x_col], summary["accuracy"], marker="o", label=baseline)
    ax.set_xlabel(x_col)
    ax.set_ylabel("accuracy")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_report(args, results: pd.DataFrame, per_model: pd.DataFrame, report_path: Path) -> None:
    columns = [
        "dataset",
        "architecture",
        "n_models",
        "width",
        "domain_shift",
        "baseline",
        "accuracy",
        "loss",
        "cycle_score",
        "merge_degradation",
    ]
    rows = results[columns].sort_values(["dataset", "architecture", "domain_shift", "width", "n_models", "baseline"]).to_dict("records")
    best_rows = (
        results.sort_values("accuracy", ascending=False)
        .groupby("setting_id")
        .head(1)[["setting_id", "baseline", "accuracy", "cycle_score"]]
        .to_dict("records")
    )
    corr = float(results[results["baseline"] == "weight_average"][["cycle_score", "merge_degradation"]].corr().iloc[0, 1]) if len(results[results["baseline"] == "weight_average"]) > 1 else float("nan")
    pivot = results.pivot_table(index="setting_id", columns="baseline", values="accuracy", aggfunc="mean")
    c2m3_delta = float((pivot["c2m3_cycle_consistent"] - pivot["weight_average"]).mean()) if {"c2m3_cycle_consistent", "weight_average"}.issubset(pivot.columns) else float("nan")
    twisted_delta = float((pivot.filter(like="twisted_rank_lift").iloc[:, 0] - pivot["weight_average"]).mean()) if "weight_average" in pivot.columns and any(col.startswith("twisted_rank_lift") for col in pivot.columns) else float("nan")
    single_summary = (
        per_model.groupby(["dataset", "architecture"])["test_accuracy"]
        .agg(["mean", "max"])
        .reset_index()
        .to_dict("records")
    )
    report = f"""# Model Merging Benchmark Report

This report is generated by `experiments/model_merging_benchmark.py`.

## Commands

```bash
{args.command_string}
```

## Scope

This is a small controlled benchmark for one-hidden-layer MLPs and one-block
CNNs.  It compares ordinary weight averaging, greedy model soup, pairwise
Git-Re-Basin-style permutation alignment, a C2M3-style cycle-consistent
synchronization, an ensemble upper bound, and a cycle-aware rank-lifted branch
ensemble.  The rank-lifted branch result is not a single merged model; it is
reported separately to avoid hiding that extra capacity.

## Outputs

- Baseline CSV: `reports/csv/model_merging_benchmark.csv`
- Per-model CSV: `reports/csv/model_merging_individual_models.csv`
- Cycle defects CSV: `reports/csv/model_merging_cycle_defects.csv`
- Scatter plot: `reports/plots/model_merging_cycle_score_vs_degradation.png`
- N ablation plot: `reports/plots/model_merging_ablation_n_models.png`
- Width ablation plot: `reports/plots/model_merging_ablation_width.png`
- Domain-shift table: `reports/csv/model_merging_domain_shift_summary.csv`
- Checkpoints: `reports/checkpoints/`

## Main Table

{format_markdown_table(rows, columns)}

## Best Baseline Per Setting

{format_markdown_table(best_rows, ["setting_id", "baseline", "accuracy", "cycle_score"])}

## Individual Model Accuracy

{format_markdown_table(single_summary, ["dataset", "architecture", "mean", "max"])}

## Diagnostic Claim Status

- Cycle score versus weight-average degradation correlation in this run:
  `{corr:.4f}`. Treat this as descriptive only for small smoke runs.
- C2M3-style synchronization average accuracy delta versus weight averaging:
  `{c2m3_delta:.4f}`.
- Rank-lifted branch ensemble average accuracy delta versus weight averaging:
  `{twisted_delta:.4f}`.
- The models are intentionally undertrained (`{args.epochs}` epoch, at most
  `{args.max_train_samples}` train samples per setting). CIFAR-10 accuracies are
  near chance in this smoke run, so CIFAR rows mainly test the plumbing rather
  than a publishable image-model claim.
- A positive result here means the score co-varies with merge degradation in
  controlled small settings. It is not yet a claim that the method beats all
  external baselines at scale.
- This run supports a weak diagnostic claim for weight averaging, but it does
  not support a general claim that cycle-consistent or rank-lifted merging
  improves accuracy across all settings.
- Negative results should be preserved in the CSV; the report is regenerated
  from all rows produced by the command above.

## Environment

```json
{json.dumps(capture_environment(), indent=2)}
```
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="mnist,cifar10")
    parser.add_argument("--architecture", default="auto", choices=["auto", "mlp", "cnn"])
    parser.add_argument("--model-counts", default="3")
    parser.add_argument("--widths", default="16,32")
    parser.add_argument("--domain-shifts", default="none,input_noise")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-train-samples", type=int, default=512)
    parser.add_argument("--max-test-samples", type=int, default=512)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--matching", default="activation", choices=["activation", "weight"])
    parser.add_argument("--rank-lift-branches", type=int, default=2)
    parser.add_argument("--seed", type=int, default=8128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "reports")
    args = parser.parse_args()
    args.command_string = " ".join([sys.executable, *sys.argv])

    all_results = []
    all_models = []
    for dataset_name in parse_csv(args.datasets, str):
        for n_models in parse_csv(args.model_counts, int):
            for width in parse_csv(args.widths, int):
                for domain_shift in parse_csv(args.domain_shifts, str):
                    results, model_rows = run_setting(args, dataset_name, n_models, width, domain_shift)
                    all_results.extend(results)
                    all_models.extend(model_rows)

    results_df = pd.DataFrame(all_results)
    models_df = pd.DataFrame(all_models)
    csv_dir = args.reports_dir / "csv"
    plot_dir = args.reports_dir / "plots"
    csv_dir.mkdir(parents=True, exist_ok=True)
    results_path = csv_dir / "model_merging_benchmark.csv"
    models_path = csv_dir / "model_merging_individual_models.csv"
    domain_path = csv_dir / "model_merging_domain_shift_summary.csv"
    results_df.to_csv(results_path, index=False)
    models_df.to_csv(models_path, index=False)
    domain_summary = (
        results_df.groupby(["dataset", "architecture", "domain_shift", "baseline"])[["accuracy", "merge_degradation", "cycle_score"]]
        .mean()
        .reset_index()
    )
    domain_summary.to_csv(domain_path, index=False)
    plot_cycle_scatter(results_df, plot_dir / "model_merging_cycle_score_vs_degradation.png")
    plot_ablation(results_df, "n_models", plot_dir / "model_merging_ablation_n_models.png", "Ablation over number of models")
    plot_ablation(results_df, "width", plot_dir / "model_merging_ablation_width.png", "Ablation over width")
    save_json(
        args.reports_dir / "configs" / "model_merging_benchmark_config.json",
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
    write_report(args, results_df, models_df, args.reports_dir / "model_merging_report.md")
    print(f"wrote {results_path}")
    print(f"wrote {models_path}")
    print(f"wrote {domain_path}")
    print(f"wrote {args.reports_dir / 'model_merging_report.md'}")


if __name__ == "__main__":
    main()
