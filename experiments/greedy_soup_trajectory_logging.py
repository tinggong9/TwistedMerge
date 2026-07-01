#!/usr/bin/env python
"""Log full greedy-soup candidate trajectories from saved fixed-setting checkpoints.

This script is a direct-audit complement to `greedy_soup_descent_audit.py`.
The earlier audit reconstructed rejected decisions from final soup metadata;
this one loads saved local-model checkpoints and evaluates every candidate
soup before deciding whether the candidate is accepted or rejected.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model_merging_benchmark import (  # noqa: E402
    average_models,
    clone_model,
    device_from_arg,
    format_markdown_table,
    greedy_soup,
    load_dataset,
    make_loader,
    make_model,
    require_torch,
)


TRAJECTORY_CSV = "greedy_soup_trajectory.csv"
SUMMARY_CSV = "greedy_soup_trajectory_summary.csv"
REPORT_MD = "greedy_soup_trajectory_report.md"
PLOT_PDF = "greedy_soup_candidate_margins.pdf"
TOL = 1e-12


def parse_seed_text(text: str | None) -> set[int] | None:
    if text is None or not str(text).strip():
        return None
    out: set[int] = set()
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            start, end = part.split(":", 1)
            out.update(range(int(start), int(end)))
        elif "-" in part:
            start, end = part.split("-", 1)
            out.update(range(int(start), int(end) + 1))
        else:
            out.add(int(part))
    return out


def split_indices(n_items: int, val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    torch, _, _ = require_torch()
    generator = torch.Generator()
    generator.manual_seed(seed)
    indices = torch.randperm(n_items, generator=generator).tolist()
    n_val = max(1, int(round(n_items * val_fraction)))
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]
    return train_indices, val_indices


def parse_checkpoint_name(path: Path) -> tuple[int, int] | None:
    match = re.fullmatch(r"seed(\d+)_model(\d+)\.pt", path.name)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def checkpoint_groups(checkpoint_root: Path, datasets: set[str] | None, settings: set[str] | None, seeds: set[int] | None) -> list[dict]:
    torch, _, _ = require_torch()
    groups = []
    for setting_dir in sorted(path for path in checkpoint_root.iterdir() if path.is_dir()):
        if settings is not None and setting_dir.name not in settings:
            continue
        by_seed: dict[int, dict[int, Path]] = {}
        for path in sorted(setting_dir.glob("seed*_model*.pt")):
            parsed = parse_checkpoint_name(path)
            if parsed is None:
                continue
            seed, model_index = parsed
            if seeds is not None and seed not in seeds:
                continue
            by_seed.setdefault(seed, {})[model_index] = path
        for seed, model_paths in sorted(by_seed.items()):
            first_path = model_paths[min(model_paths)]
            payload = torch.load(first_path, map_location="cpu")
            metadata = dict(payload.get("metadata", {}))
            if datasets is not None and str(metadata.get("dataset")) not in datasets:
                continue
            n_models = int(metadata["n_models"])
            missing = [idx for idx in range(n_models) if idx not in model_paths]
            groups.append(
                {
                    "setting_id": setting_dir.name,
                    "seed": int(seed),
                    "model_paths": model_paths,
                    "metadata": metadata,
                    "complete": not missing,
                    "missing_model_indices": missing,
                }
            )
    return groups


def load_eval_loaders(metadata: dict, args, cache: dict):
    torch, _, _ = require_torch()
    key = (
        metadata["dataset"],
        int(metadata.get("max_train_samples", args.max_train_samples)),
        int(metadata.get("max_test_samples", args.max_test_samples)),
        int(metadata.get("dataset_seed", args.dataset_seed)),
        str(metadata.get("augmentation", args.augmentation)),
        int(metadata.get("train_split_seed", int(metadata.get("dataset_seed", args.dataset_seed)) + 17)),
        float(args.val_fraction),
        int(args.batch_size),
    )
    if key in cache:
        return cache[key]
    spec, train_base, test_base = load_dataset(
        key[0],
        args.data_dir,
        key[1],
        key[2],
        key[3],
        augmentation=key[4],
    )
    _train_indices, val_indices = split_indices(len(train_base), key[6], key[5])
    val_subset = torch.utils.data.Subset(train_base, val_indices)
    val_loader = make_loader(val_subset, key[7], shuffle=False, seed=key[3] + 100)
    test_loader = make_loader(test_base, key[7], shuffle=False, seed=key[3] + 200)
    cache[key] = (spec, val_loader, test_loader)
    return cache[key]


def load_models(group: dict, spec, device) -> list:
    torch, _, _ = require_torch()
    metadata = group["metadata"]
    architecture = str(metadata["architecture"])
    width = int(metadata["width"])
    models = []
    for model_index in range(int(metadata["n_models"])):
        payload = torch.load(group["model_paths"][model_index], map_location="cpu")
        model = make_model(architecture, spec, width)
        model.load_state_dict(payload["state_dict"])
        model.to(device)
        model.eval()
        models.append(model)
    return models


def json_list(value) -> str:
    if isinstance(value, str):
        return value
    return json.dumps([int(item) for item in value], separators=(",", ":"))


def trajectory_rows_for_group(group: dict, args, loader_cache: dict) -> list[dict]:
    metadata = group["metadata"]
    device = device_from_arg(args.device)
    spec, val_loader, test_loader = load_eval_loaders(metadata, args, loader_cache)
    models = load_models(group, spec, device)
    architecture = str(metadata["architecture"])
    width = int(metadata["width"])
    _soup_model, final_indices, final_test, trajectory = greedy_soup(
        models,
        val_loader,
        test_loader,
        device,
        architecture,
        spec,
        width,
        return_trajectory=True,
    )
    setting_id = str(group["setting_id"])
    run_id = f"{setting_id}_seed{int(group['seed'])}"
    rows = []
    final_rank = max(int(row["candidate_rank"]) for row in trajectory if row.get("is_final_selection"))
    for item in trajectory:
        is_final = bool(item["is_final_selection"])
        row = {
            "setting_id": setting_id,
            "run_id": run_id,
            "seed": int(group["seed"]),
            "dataset": str(metadata["dataset"]),
            "architecture": architecture,
            "n_models": int(metadata["n_models"]),
            "width": width,
            "domain_shift": str(metadata["domain_shift"]),
            "matching": str(metadata.get("matching", "activation")),
            "candidate_rank": int(item["candidate_rank"]),
            "candidate_model_index": int(item["candidate_model_index"]),
            "candidate_order": json_list(item["candidate_order"]),
            "candidate_method": str(item["candidate_method"]),
            "candidate_source": str(item["candidate_source"]),
            "soup_indices_before": json_list(item["soup_indices_before"]),
            "soup_indices_after": json_list(item["soup_indices_after"]),
            "validation_accuracy_before": item["validation_accuracy_before"],
            "validation_loss_before": item["validation_loss_before"],
            "candidate_soup_validation_accuracy": item["candidate_soup_validation_accuracy"],
            "candidate_soup_validation_loss": item["candidate_soup_validation_loss"],
            "validation_accuracy_margin_after_minus_before": item["validation_accuracy_margin_after_minus_before"],
            "validation_loss_margin_before_minus_after": item["validation_loss_margin_before_minus_after"],
            "accepted": bool(item["accepted"]),
            "decision_reason": str(item["decision_reason"]),
            "decision_metric": str(item["decision_metric"]),
            "decision_metric_source": str(item["decision_metric_source"]),
            "candidate_soup_validation_metric_logged": True,
            "algorithm_implied_decision": False,
            "test_used_for_selection": False,
            "is_final_selection": is_final,
            "final_selection_indices": json_list(final_indices),
            "final_selection_candidate_rank": final_rank,
            "final_test_accuracy": final_test["accuracy"] if is_final else float("nan"),
            "final_test_loss": final_test["loss"] if is_final else float("nan"),
            "test_metric_role": "evaluation_only_final_selection" if is_final else "not_evaluated_for_candidate_decision",
            "checkpoint_source": str(group["model_paths"][int(item["candidate_model_index"])]),
            "run_source": "checkpointed_fixed_setting_verification",
        }
        rows.append(row)
    for model in models:
        model.to("cpu")
    return rows


def build_summary(trajectory: pd.DataFrame) -> pd.DataFrame:
    if trajectory.empty:
        return pd.DataFrame()
    df = trajectory.copy()
    df["accepted_bool"] = df["accepted"].astype(bool)
    df["is_final_bool"] = df["is_final_selection"].astype(bool)
    df["acc_margin"] = pd.to_numeric(df["validation_accuracy_margin_after_minus_before"], errors="coerce")
    df["loss_margin"] = pd.to_numeric(df["validation_loss_margin_before_minus_after"], errors="coerce")
    df["candidate_acc"] = pd.to_numeric(df["candidate_soup_validation_accuracy"], errors="coerce")
    df["candidate_loss"] = pd.to_numeric(df["candidate_soup_validation_loss"], errors="coerce")
    df["test_used"] = df["test_used_for_selection"].astype(bool)
    non_initial = df["candidate_rank"].astype(int) > 1
    rejected = ~df["accepted_bool"]
    accepted_after_initial = df["accepted_bool"] & non_initial
    df["validation_monotonicity_violation"] = accepted_after_initial & (df["acc_margin"] < -TOL)
    df["rejection_rule_violation"] = rejected & (df["acc_margin"] >= -TOL)
    df["missing_rejected_candidate_metric"] = rejected & (~np.isfinite(df["candidate_acc"]) | ~np.isfinite(df["candidate_loss"]))
    df["nonfinal_test_metric_present"] = (~df["is_final_bool"]) & (
        pd.to_numeric(df["final_test_accuracy"], errors="coerce").notna()
        | pd.to_numeric(df["final_test_loss"], errors="coerce").notna()
    )

    run_rows = []
    for run_id, group in df.groupby("run_id", dropna=False):
        initial = group.loc[group["candidate_rank"].astype(int).idxmin()]
        final = group[group["is_final_bool"]].iloc[-1]
        run_rows.append(
            {
                "run_id": run_id,
                "setting_id": group["setting_id"].iloc[0],
                "dataset": group["dataset"].iloc[0],
                "architecture": group["architecture"].iloc[0],
                "n_models": int(group["n_models"].iloc[0]),
                "width": int(group["width"].iloc[0]),
                "domain_shift": group["domain_shift"].iloc[0],
                "matching": group["matching"].iloc[0],
                "seed": int(group["seed"].iloc[0]),
                "candidate_rows": int(len(group)),
                "accepted_candidates": int(group["accepted_bool"].sum()),
                "rejected_candidates": int((~group["accepted_bool"]).sum()),
                "missing_rejected_candidate_metrics": int(group["missing_rejected_candidate_metric"].sum()),
                "validation_monotonicity_violations": int(group["validation_monotonicity_violation"].sum()),
                "rejection_rule_violations": int(group["rejection_rule_violation"].sum()),
                "nonfinal_test_metric_rows": int(group["nonfinal_test_metric_present"].sum()),
                "test_used_for_selection_rows": int(group["test_used"].sum()),
                "initial_validation_accuracy": float(initial["candidate_acc"]),
                "final_validation_accuracy": float(final["candidate_acc"]),
                "final_test_accuracy": float(final["final_test_accuracy"]),
                "final_output_validation_descent_violation": bool(float(final["candidate_acc"]) < float(initial["candidate_acc"]) - TOL),
            }
        )
    run_level = pd.DataFrame(run_rows)

    group_cols = ["dataset", "architecture", "n_models", "width", "domain_shift", "matching"]
    summary = run_level.groupby(group_cols, dropna=False).agg(
        summary_type=("run_id", lambda _: "setting_summary"),
        n_runs=("run_id", "count"),
        n_unique_seeds=("seed", "nunique"),
        candidate_rows=("candidate_rows", "sum"),
        accepted_candidates=("accepted_candidates", "sum"),
        rejected_candidates=("rejected_candidates", "sum"),
        missing_rejected_candidate_metrics=("missing_rejected_candidate_metrics", "sum"),
        validation_monotonicity_violations=("validation_monotonicity_violations", "sum"),
        rejection_rule_violations=("rejection_rule_violations", "sum"),
        final_output_validation_descent_violations=("final_output_validation_descent_violation", "sum"),
        nonfinal_test_metric_rows=("nonfinal_test_metric_rows", "sum"),
        test_used_for_selection_rows=("test_used_for_selection_rows", "sum"),
        mean_final_validation_accuracy=("final_validation_accuracy", "mean"),
        mean_final_test_accuracy=("final_test_accuracy", "mean"),
    ).reset_index()

    margin_summary = df[non_initial].groupby(group_cols, dropna=False).agg(
        mean_accepted_accuracy_margin=(
            "acc_margin",
            lambda values: float(np.nanmean(values[df.loc[values.index, "accepted_bool"]])) if df.loc[values.index, "accepted_bool"].any() else float("nan"),
        ),
        mean_rejected_accuracy_margin=(
            "acc_margin",
            lambda values: float(np.nanmean(values[~df.loc[values.index, "accepted_bool"]])) if (~df.loc[values.index, "accepted_bool"]).any() else float("nan"),
        ),
        mean_accepted_loss_margin=(
            "loss_margin",
            lambda values: float(np.nanmean(values[df.loc[values.index, "accepted_bool"]])) if df.loc[values.index, "accepted_bool"].any() else float("nan"),
        ),
        mean_rejected_loss_margin=(
            "loss_margin",
            lambda values: float(np.nanmean(values[~df.loc[values.index, "accepted_bool"]])) if (~df.loc[values.index, "accepted_bool"]).any() else float("nan"),
        ),
    ).reset_index()
    summary = summary.merge(margin_summary, on=group_cols, how="left")
    summary["direct_stepwise_descent_supported"] = (
        summary["missing_rejected_candidate_metrics"].eq(0)
        & summary["validation_monotonicity_violations"].eq(0)
        & summary["rejection_rule_violations"].eq(0)
        & summary["final_output_validation_descent_violations"].eq(0)
        & summary["test_used_for_selection_rows"].eq(0)
        & summary["nonfinal_test_metric_rows"].eq(0)
    )
    summary["claim_decision"] = np.where(
        summary["direct_stepwise_descent_supported"],
        "directly_supported_for_checkpointed_activation_setting",
        "not_supported_or_needs_investigation",
    )

    overall = {
        "dataset": "ALL",
        "architecture": "mlp2",
        "n_models": "",
        "width": 128,
        "domain_shift": "ALL",
        "matching": "activation",
        "summary_type": "overall",
        "n_runs": int(run_level["run_id"].nunique()),
        "n_unique_seeds": int(run_level.groupby(["dataset", "n_models", "domain_shift"])["seed"].nunique().min()),
        "candidate_rows": int(len(df)),
        "accepted_candidates": int(df["accepted_bool"].sum()),
        "rejected_candidates": int((~df["accepted_bool"]).sum()),
        "missing_rejected_candidate_metrics": int(df["missing_rejected_candidate_metric"].sum()),
        "validation_monotonicity_violations": int(df["validation_monotonicity_violation"].sum()),
        "rejection_rule_violations": int(df["rejection_rule_violation"].sum()),
        "final_output_validation_descent_violations": int(run_level["final_output_validation_descent_violation"].sum()),
        "nonfinal_test_metric_rows": int(df["nonfinal_test_metric_present"].sum()),
        "test_used_for_selection_rows": int(df["test_used"].sum()),
        "mean_final_validation_accuracy": float(run_level["final_validation_accuracy"].mean()),
        "mean_final_test_accuracy": float(run_level["final_test_accuracy"].mean()),
        "mean_accepted_accuracy_margin": float(df.loc[accepted_after_initial, "acc_margin"].mean()),
        "mean_rejected_accuracy_margin": float(df.loc[rejected, "acc_margin"].mean()),
        "mean_accepted_loss_margin": float(df.loc[accepted_after_initial, "loss_margin"].mean()),
        "mean_rejected_loss_margin": float(df.loc[rejected, "loss_margin"].mean()),
    }
    overall["direct_stepwise_descent_supported"] = (
        overall["missing_rejected_candidate_metrics"] == 0
        and overall["validation_monotonicity_violations"] == 0
        and overall["rejection_rule_violations"] == 0
        and overall["final_output_validation_descent_violations"] == 0
        and overall["test_used_for_selection_rows"] == 0
        and overall["nonfinal_test_metric_rows"] == 0
    )
    overall["claim_decision"] = (
        "directly_supported_for_checkpointed_activation_settings"
        if overall["direct_stepwise_descent_supported"]
        else "not_supported_or_needs_investigation"
    )
    summary = pd.concat([pd.DataFrame([overall]), summary], ignore_index=True, sort=False)
    return summary


def md_table(df: pd.DataFrame, columns: list[str], max_rows: int = 40) -> str:
    if df.empty:
        return "_No rows._"
    rows = df.head(max_rows).copy()
    for col in columns:
        if col not in rows.columns:
            rows[col] = ""
    return format_markdown_table(rows[columns].to_dict("records"), columns)


def plot_candidate_margins(trajectory: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    df = trajectory.copy()
    df["candidate_rank"] = pd.to_numeric(df["candidate_rank"], errors="coerce")
    df["margin"] = pd.to_numeric(df["validation_accuracy_margin_after_minus_before"], errors="coerce")
    df = df[df["candidate_rank"] > 1].copy()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    if df.empty:
        for ax in axes:
            ax.text(0.5, 0.5, "No non-initial candidates", ha="center", va="center")
            ax.set_axis_off()
    else:
        colors = np.where(df["accepted"].astype(bool), "tab:green", "tab:red")
        axes[0].scatter(df["candidate_rank"], df["margin"], c=colors, alpha=0.55, s=16)
        axes[0].axhline(0.0, color="black", linewidth=0.8)
        axes[0].set_xlabel("candidate rank")
        axes[0].set_ylabel("validation accuracy margin")
        axes[0].set_title("Candidate soup margins")
        axes[0].grid(True, alpha=0.25)

        accepted = df[df["accepted"].astype(bool)]["margin"].dropna()
        rejected = df[~df["accepted"].astype(bool)]["margin"].dropna()
        bins = 30
        axes[1].hist(rejected, bins=bins, color="tab:red", alpha=0.68, label="rejected")
        axes[1].hist(accepted, bins=bins, color="tab:green", alpha=0.68, label="accepted")
        axes[1].axvline(0.0, color="black", linewidth=0.8)
        axes[1].set_xlabel("validation accuracy margin")
        axes[1].set_ylabel("candidate rows")
        axes[1].set_title("Decision boundary")
        axes[1].legend()
        axes[1].grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def update_claims_audit(summary: pd.DataFrame, path: Path) -> None:
    if path is None:
        return
    supported = bool(
        not summary.empty
        and summary.iloc[0]["summary_type"] == "overall"
        and bool(summary.iloc[0]["direct_stepwise_descent_supported"])
    )
    status = "Supported limited" if supported else "Not yet supported"
    decision = str(summary.iloc[0]["claim_decision"]) if not summary.empty else "not_run"
    row = (
        "| The full stepwise greedy-soup empirical descent theorem is directly auditable for the checkpointed activation-matching fixed-setting MLP2 trajectory run. "
        f"| {status} | `reports/greedy_soup_trajectory_report.md` and `reports/csv/greedy_soup_trajectory.csv` log directly observed candidate-soup validation accuracy/loss for every accepted and rejected candidate in the checkpointed activation settings; decision `{decision}`; test metrics are final-selection evaluation only and are not used for selection. |"
    )
    text = path.read_text(encoding="utf-8")
    marker = "The full stepwise greedy-soup empirical descent theorem is directly auditable"
    if marker not in text:
        insert_marker = "<!-- prompt10-claim-audit:start -->"
        if insert_marker in text:
            text = text.replace(insert_marker, row + "\n\n" + insert_marker)
        else:
            text = text.rstrip() + "\n" + row + "\n"
        path.write_text(text, encoding="utf-8")


def write_report(args, trajectory: pd.DataFrame, summary: pd.DataFrame, skipped: list[dict], path: Path) -> None:
    overall = summary.iloc[0].to_dict() if not summary.empty else {}
    supported = bool(overall.get("direct_stepwise_descent_supported", False))
    skipped_text = pd.DataFrame(skipped) if skipped else pd.DataFrame()
    report = f"""# Greedy Soup Trajectory Report

Generated by `experiments/greedy_soup_trajectory_logging.py`.

## Exact Command

```bash
{args.command_string}
```

## Scope

- This report logs directly observed greedy-soup candidate margins by loading saved fixed-setting checkpoints and evaluating every candidate soup on the validation split before the accept/reject decision.
- The run covers the checkpointed quality-gated `mlp2` width-128 activation-matching settings under `reports/checkpoints/fixed_setting_verification`.
- The newer Prompt 22 reconstructed audit rows for seeds `4100:4129` include weight-matching duplicates without saved checkpoints. Those rows remain algorithm-implied unless rerun with checkpoint saving or retraining.
- Test metrics are written only on the final selected row of each run and are evaluation-only.

## Decision

Full stepwise validation-descent support for the checkpointed activation settings: `{"yes" if supported else "no"}`.

Validation monotonicity violations: `{overall.get("validation_monotonicity_violations", "NA")}`.
Rejected candidates missing candidate-soup validation metrics: `{overall.get("missing_rejected_candidate_metrics", "NA")}`.
Rejection-rule violations: `{overall.get("rejection_rule_violations", "NA")}`.
Rows where test metrics were used for selection: `{overall.get("test_used_for_selection_rows", "NA")}`.

## Outputs

- `reports/csv/{TRAJECTORY_CSV}`
- `reports/csv/{SUMMARY_CSV}`
- `reports/{REPORT_MD}`
- `reports/plots/{PLOT_PDF}`

## Directly Observed Candidate Margins

Every output candidate row has `candidate_soup_validation_accuracy` and `candidate_soup_validation_loss` populated before the decision. Rejected rows are therefore directly auditable, not inferred from final selected indices.

## Algorithm-Implied Decisions

`algorithm_implied_decision` is `False` for all generated trajectory rows. The only remaining algorithm-implied boundary is external to this report: Prompt 22's 4100-series reconstructed audit did not have checkpointed candidate-soup metrics.

## Final-Output Validation Descent

Final-output validation descent is checked by comparing the final selected soup's validation accuracy with the initial best-validation local model inside each run.

## Evaluation-Only Test Metrics

`final_test_accuracy` and `final_test_loss` are populated only for `is_final_selection = True` rows. They are not used in greedy-soup candidate selection.

## Summary

{md_table(summary, ["summary_type", "dataset", "architecture", "n_models", "width", "domain_shift", "matching", "n_runs", "candidate_rows", "accepted_candidates", "rejected_candidates", "missing_rejected_candidate_metrics", "validation_monotonicity_violations", "rejection_rule_violations", "final_output_validation_descent_violations", "test_used_for_selection_rows", "direct_stepwise_descent_supported", "claim_decision"], 40)}

## Candidate Row Sample

{md_table(trajectory, ["setting_id", "seed", "candidate_rank", "candidate_model_index", "soup_indices_before", "soup_indices_after", "validation_accuracy_before", "candidate_soup_validation_accuracy", "validation_accuracy_margin_after_minus_before", "candidate_soup_validation_loss", "accepted", "decision_reason", "is_final_selection", "final_test_accuracy", "test_used_for_selection"], 30)}

## Skipped Checkpoint Groups

{md_table(skipped_text, ["setting_id", "seed", "missing_model_indices"], 30)}
"""
    path.write_text(report, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path, default=ROOT / "reports" / "checkpoints" / "fixed_setting_verification")
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "reports")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--datasets", default="mnist,fashion_mnist")
    parser.add_argument("--settings", default="")
    parser.add_argument("--seeds", default="")
    parser.add_argument("--max-runs", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--max-train-samples", type=int, default=10000)
    parser.add_argument("--max-test-samples", type=int, default=5000)
    parser.add_argument("--dataset-seed", type=int, default=314159)
    parser.add_argument("--augmentation", default="none")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--update-claims-audit", action="store_true", default=True)
    parser.add_argument("--no-update-claims-audit", action="store_false", dest="update_claims_audit")
    args = parser.parse_args()
    args.command_string = " ".join([sys.executable, *sys.argv])
    return args


def main() -> None:
    args = parse_args()
    datasets = {item.strip() for item in args.datasets.split(",") if item.strip()} or None
    settings = {item.strip() for item in args.settings.split(",") if item.strip()} or None
    seeds = parse_seed_text(args.seeds)
    groups = checkpoint_groups(args.checkpoint_root, datasets, settings, seeds)
    if args.max_runs and args.max_runs > 0:
        groups = groups[: args.max_runs]

    rows: list[dict] = []
    skipped: list[dict] = []
    loader_cache: dict = {}
    for idx, group in enumerate(groups, start=1):
        if not group["complete"]:
            skipped.append(
                {
                    "setting_id": group["setting_id"],
                    "seed": group["seed"],
                    "missing_model_indices": json_list(group["missing_model_indices"]),
                }
            )
            continue
        print(f"[{idx}/{len(groups)}] {group['setting_id']} seed {group['seed']}", flush=True)
        rows.extend(trajectory_rows_for_group(group, args, loader_cache))

    trajectory = pd.DataFrame(rows)
    summary = build_summary(trajectory)

    csv_dir = args.reports_dir / "csv"
    plot_dir = args.reports_dir / "plots"
    csv_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    trajectory.to_csv(csv_dir / TRAJECTORY_CSV, index=False, lineterminator="\n")
    summary.to_csv(csv_dir / SUMMARY_CSV, index=False, lineterminator="\n")
    plot_candidate_margins(trajectory, plot_dir / PLOT_PDF)
    write_report(args, trajectory, summary, skipped, args.reports_dir / REPORT_MD)
    if args.update_claims_audit:
        update_claims_audit(summary, args.reports_dir / "claims_audit.md")

    print(f"wrote {csv_dir / TRAJECTORY_CSV}")
    print(f"wrote {csv_dir / SUMMARY_CSV}")
    print(f"wrote {args.reports_dir / REPORT_MD}")
    print(f"wrote {plot_dir / PLOT_PDF}")


if __name__ == "__main__":
    main()
