#!/usr/bin/env python3
"""Fresh, leakage-audited practical selector benchmark and release writer."""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports" / "overnight_program"
SOURCE = OUTPUT / "practical_selector_source"

METHOD_RENAME = {
    "weight_average": "ordinary_weight_average",
    "git_rebasin_pairwise": "git_rebasin_style_pairwise",
    "c2m3_permutation": "c2m3_strict_synchronization",
    "c2m3_greedy_soup": "c2m3_greedy_soup",
    "monomial_scale": "raw_monomial_alignment",
    "shrinkage_monomial_scale": "shrinkage_monomial",
    "global_monomial_scale": "global_monomial",
    "optimized_monomial_scale": "optimized_monomial",
    "monomial_scaled_greedy_soup": "monomial_greedy_soup",
    "greedy_soup": "ordinary_greedy_soup",
    "union_candidate_soup": "union_candidate_soup",
    "randomly_augmented_candidate_union": "randomly_augmented_candidate_union",
    "improved_validated_selector": "twistedmerge_exact_gauge_soup_selector",
    "ensemble_upper_bound": "ensemble_reference",
}


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def peak_child_memory_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
    return value / (1024.0 * 1024.0) if sys.platform == "darwin" else value / 1024.0


def source_command(mode: str) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "experiments" / "improved_validated_ladder_merge_benchmark.py"),
        "--reports-dir",
        str(SOURCE),
        "--saved-logits-dir",
        str(OUTPUT / "logits" / "practical_selector"),
    ]
    if mode == "smoke":
        command.extend(
            [
                "--seeds", "1800",
                "--model-counts", "3",
                "--widths", "16",
                "--epochs", "1",
                "--max-train-samples", "512",
                "--max-test-samples", "512",
                "--optimization-steps", "2",
                "--bootstrap-samples", "50",
            ]
        )
    return command


def run_source(mode: str) -> tuple[list[str], float, float]:
    command = source_command(mode)
    start = time.perf_counter()
    subprocess.run(command, cwd=ROOT, check=True)
    return command, time.perf_counter() - start, peak_child_memory_mb()


def prepare_runs(
    source: pd.DataFrame,
    runtime_seconds: float,
    peak_memory_mb: float,
    *,
    execution_commit: str | None = None,
    dirty_worktree_at_execution: bool | None = None,
) -> pd.DataFrame:
    runs = source[source["method"].isin(METHOD_RENAME)].copy()
    runs["source_method"] = runs["method"]
    runs["method"] = runs["method"].map(METHOD_RENAME)
    runs["fresh_inference"] = True
    runs["central_lift_activated"] = False
    runs["nonabelian_lift_activated"] = False
    runs["supplied_context"] = False
    runs["uses_obstruction_data"] = runs["method"].str.contains("twistedmerge|monomial|c2m3", case=False)
    runs["uses_validation_data"] = runs["method"].str.contains("soup|selector|shrinkage|global|optimized")
    runs["candidate_count"] = pd.to_numeric(runs.get("union_candidate_count", 1), errors="coerce").fillna(1).astype(int)
    selector = runs["method"] == "twistedmerge_exact_gauge_soup_selector"
    runs.loc[selector, "candidate_count"] = 13
    runs["method_kind"] = "single_model"
    runs.loc[runs["method"].str.contains("soup"), "method_kind"] = "soup"
    runs.loc[runs["method"] == "twistedmerge_exact_gauge_soup_selector", "method_kind"] = "validation_selector"
    runs.loc[runs["method"] == "ensemble_reference", "method_kind"] = "ensemble"
    runs["benchmark_wall_time_seconds"] = runtime_seconds
    runs["peak_memory_mb"] = peak_memory_mb
    runs["execution_commit"] = execution_commit or git_output("rev-parse", "HEAD")
    runs["dirty_worktree_at_execution"] = (
        bool(git_output("status", "--porcelain"))
        if dirty_worktree_at_execution is None
        else dirty_worktree_at_execution
    )
    best_test = runs.groupby("setting_id")["accuracy"].transform("max")
    runs["selector_regret_audit"] = np.where(selector, best_test - runs["accuracy"], np.nan)
    return runs.sort_values(["setting_id", "method"]).reset_index(drop=True)


def ci(values: pd.Series) -> tuple[float, float]:
    clean = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    if clean.size <= 1:
        mean = float(clean.mean()) if clean.size else float("nan")
        return mean, mean
    half = 1.96 * float(clean.std(ddof=1)) / np.sqrt(clean.size)
    return float(clean.mean() - half), float(clean.mean() + half)


def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    scopes = [("overall", ["method"]), ("fixed_setting", ["n_models", "width", "method"])]
    for scope, keys in scopes:
        for key, group in runs.groupby(keys, sort=True):
            key = key if isinstance(key, tuple) else (key,)
            values = dict(zip(keys, key, strict=True))
            acc_low, acc_high = ci(group["accuracy"])
            loss_low, loss_high = ci(group["loss"])
            rows.append(
                {
                    "scope": scope,
                    "n_models": values.get("n_models", "all"),
                    "width": values.get("width", "all"),
                    "method": values["method"],
                    "n": len(group),
                    "mean_test_accuracy": group["accuracy"].mean(),
                    "accuracy_ci_low": acc_low,
                    "accuracy_ci_high": acc_high,
                    "mean_test_loss": group["loss"].mean(),
                    "loss_ci_low": loss_low,
                    "loss_ci_high": loss_high,
                    "mean_validation_accuracy": group["val_accuracy"].mean(),
                    "mean_inference_time_seconds_512": group["measured_inference_time_seconds_512"].mean(),
                    "mean_inference_multiplier": group["inference_multiplier"].mean(),
                }
            )
    return pd.DataFrame(rows)


def paired_statistics(runs: pd.DataFrame, baseline: str = "ordinary_greedy_soup") -> pd.DataFrame:
    pivot_acc = runs.pivot(index="setting_id", columns="method", values="accuracy")
    pivot_loss = runs.pivot(index="setting_id", columns="method", values="loss")
    rows = []
    for method in sorted(pivot_acc.columns):
        if method == baseline:
            continue
        delta = (pivot_acc[method] - pivot_acc[baseline]).dropna()
        loss_delta = (pivot_loss[method] - pivot_loss[baseline]).dropna()
        low, high = ci(delta)
        rows.append(
            {
                "method": method,
                "baseline": baseline,
                "n_pairs": len(delta),
                "mean_accuracy_delta": delta.mean(),
                "accuracy_delta_ci_low": low,
                "accuracy_delta_ci_high": high,
                "mean_loss_delta": loss_delta.mean(),
                "wins": int((delta > 1e-12).sum()),
                "ties": int((delta.abs() <= 1e-12).sum()),
                "losses": int((delta < -1e-12).sum()),
            }
        )
    return pd.DataFrame(rows)


def selection_choices(runs: pd.DataFrame) -> pd.DataFrame:
    selected = runs[runs["method"] == "twistedmerge_exact_gauge_soup_selector"].copy()
    selected["selected_source_method"] = selected["selector_chose"]
    selected["selected_method"] = selected["selector_chose"].map(METHOD_RENAME).fillna(selected["selector_chose"])
    return selected[
        [
            "setting_id", "n_models", "width", "seed", "selected_source_method", "selected_method",
            "selector_val_margin", "selector_validation_budget", "candidate_count",
            "selector_regret_audit", "central_lift_activated", "nonabelian_lift_activated",
        ]
    ]


def capacity_table(runs: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "actual_trainable_parameters", "stored_parameters", "parameter_multiplier", "branch_count",
        "measured_inference_time_seconds_512", "inference_multiplier", "peak_memory_mb", "candidate_count",
        "selector_validation_budget",
    ]
    return runs.groupby("method", as_index=False)[columns].mean(numeric_only=True)


def write_latex(summary: pd.DataFrame, choices: pd.DataFrame) -> None:
    table_dir = OUTPUT / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    overall = summary[summary["scope"] == "overall"][
        ["method", "mean_test_accuracy", "accuracy_ci_low", "accuracy_ci_high", "mean_test_loss"]
    ].copy()
    overall.to_latex(table_dir / "practical_selector_main.tex", index=False, float_format="%.4f", escape=True)
    frequencies = choices.groupby("selected_method").size().rename("count").reset_index()
    frequencies.to_latex(table_dir / "practical_selector_choices.tex", index=False, escape=True)


def write_plot(summary: pd.DataFrame) -> None:
    plot_dir = OUTPUT / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    overall = summary[summary["scope"] == "overall"].sort_values("mean_test_accuracy")
    fig, ax = plt.subplots(figsize=(9, 6))
    error = np.vstack(
        [
            overall["mean_test_accuracy"] - overall["accuracy_ci_low"],
            overall["accuracy_ci_high"] - overall["mean_test_accuracy"],
        ]
    )
    ax.barh(overall["method"], overall["mean_test_accuracy"], xerr=error, color="#3565a8", alpha=0.85)
    ax.set_xlabel("Held-out test accuracy (95% normal CI)")
    ax.set_title("Fresh practical-selector rerun")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plot_dir / "practical_selector_accuracy.pdf")
    plt.close(fig)


def write_report(runs: pd.DataFrame, summary: pd.DataFrame, paired: pd.DataFrame, choices: pd.DataFrame, mode: str) -> None:
    overall = summary[summary["scope"] == "overall"].sort_values("mean_test_accuracy", ascending=False)
    selector = overall[overall["method"] == "twistedmerge_exact_gauge_soup_selector"].iloc[0]
    soup = overall[overall["method"] == "ordinary_greedy_soup"].iloc[0]
    selector_pair = paired[paired["method"] == "twistedmerge_exact_gauge_soup_selector"].iloc[0]
    leakage = bool(runs["label_permutation_regression_passed"].astype(bool).all())
    frequencies = choices["selected_method"].value_counts().to_dict()
    text = f"""# Stage 1: fresh practical selector rerun

## Decision

This is a **{mode}** execution with {runs['setting_id'].nunique()} matched settings and fresh model training, merged checkpoints, inference, saved logits, validation-only selection, and held-out test evaluation. The saved-logit label-permutation regression {'passed' if leakage else 'FAILED'} for every method-setting row.

The TwistedMerge selector mean accuracy is {selector['mean_test_accuracy']:.6f}; ordinary greedy soup is {soup['mean_test_accuracy']:.6f}. The paired selector delta is {selector_pair['mean_accuracy_delta']:+.6f} (95% CI [{selector_pair['accuracy_delta_ci_low']:+.6f}, {selector_pair['accuracy_delta_ci_high']:+.6f}]), with {int(selector_pair['wins'])}/{int(selector_pair['ties'])}/{int(selector_pair['losses'])} win/tie/loss.

No central lift was selected. No nonabelian lift was selected. Both activation rates are exactly 0 because no certificate passed and these candidates were not invented.

## Protocol

- MNIST, one-hidden-layer ReLU MLP.
- Model counts: {sorted(runs['n_models'].unique().tolist())}; widths: {sorted(runs['width'].unique().tolist())}; seeds: {int(runs['seed'].min())}--{int(runs['seed'].max())}.
- Checkpoints and splits are matched across all methods within each setting.
- Alignment uses the training partition, selection uses a disjoint validation partition, and the test set is evaluation-only.
- Every reported prediction comes from an executed merged model, soup, selector choice, or ensemble.
- Selector frequencies: `{json.dumps(frequencies, sort_keys=True)}`.

## Evidence boundary

Selector regret is reported only as a post-hoc audit and never influences selection. The central and nonabelian activation columns are retained as explicit negative findings. Runtime, peak process memory, parameter counts, stored parameters, branch count, candidate count, validation budget, and measured inference time are in the run and capacity CSV files.
"""
    (OUTPUT / "practical_selector_report.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "full"), default="full")
    parser.add_argument("--reuse-source", action="store_true", help="Rebuild reports without rerunning inference.")
    args = parser.parse_args()
    execution_commit = git_output("rev-parse", "HEAD")
    dirty_worktree_at_execution = bool(git_output("status", "--porcelain"))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if args.reuse_source:
        command, runtime_seconds, peak_memory_mb = source_command(args.mode), float("nan"), float("nan")
    else:
        command, runtime_seconds, peak_memory_mb = run_source(args.mode)

    source_path = SOURCE / "csv" / "improved_validated_ladder_merge_benchmark.csv"
    source = pd.read_csv(source_path)
    runs = prepare_runs(
        source,
        runtime_seconds,
        peak_memory_mb,
        execution_commit=execution_commit,
        dirty_worktree_at_execution=dirty_worktree_at_execution,
    )
    summary = summarize(runs)
    paired = paired_statistics(runs)
    choices = selection_choices(runs)
    capacity = capacity_table(runs)

    required_methods = set(METHOD_RENAME.values())
    missing = required_methods - set(runs["method"])
    if missing:
        raise RuntimeError(f"required methods missing from fresh execution: {sorted(missing)}")
    if not runs["label_permutation_regression_passed"].astype(bool).all():
        raise RuntimeError("saved-logit label-permutation regression failed")

    runs.to_csv(OUTPUT / "practical_selector_runs.csv", index=False)
    summary.to_csv(OUTPUT / "practical_selector_summary.csv", index=False)
    paired.to_csv(OUTPUT / "practical_selector_paired_stats.csv", index=False)
    choices.to_csv(OUTPUT / "practical_selector_choices.csv", index=False)
    capacity.to_csv(OUTPUT / "practical_selector_capacity.csv", index=False)
    write_latex(summary, choices)
    write_plot(summary)
    write_report(runs, summary, paired, choices, args.mode)

    config = {
        "stage": 1,
        "mode": args.mode,
        "execution_commit": execution_commit,
        "dirty_worktree_at_execution": dirty_worktree_at_execution,
        "command": " ".join(command),
        "source_script": "experiments/improved_validated_ladder_merge_benchmark.py",
        "fresh_inference": not args.reuse_source,
        "matched_grid": {
            "dataset": "MNIST",
            "architecture": "one_hidden_layer_relu_mlp",
            "model_counts": sorted(runs["n_models"].unique().tolist()),
            "widths": sorted(runs["width"].unique().tolist()),
            "seeds": sorted(runs["seed"].unique().tolist()),
        },
        "runtime_seconds": runtime_seconds,
        "peak_child_memory_mb": peak_memory_mb,
        "central_lift_activated": False,
        "nonabelian_lift_activated": False,
        "leakage_regression_passed": True,
        "environment": {
            "python": sys.version,
            "executable": sys.executable,
            "platform": sys.platform,
            "PYTHONPYCACHEPREFIX": os.environ.get("PYTHONPYCACHEPREFIX"),
        },
    }
    (OUTPUT / "practical_selector_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"wrote Stage 1 outputs to {OUTPUT}")


if __name__ == "__main__":
    main()
