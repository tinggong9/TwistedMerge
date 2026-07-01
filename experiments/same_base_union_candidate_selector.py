#!/usr/bin/env python
"""Validation-only union selector for same-base task-vector candidates.

The Prompt 31 same-base benchmark already evaluated a family of candidates on
validation and test loaders.  This script treats those rows as a single union
candidate pool and asks whether validation selection over the union matches or
improves the best existing method family in each exact setting.

No training is performed here.  Test metrics are used only after the
validation-selected candidate is fixed.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model_merging_benchmark import format_markdown_table  # noqa: E402


RUN_CSV = "same_base_union_candidate_selector.csv"
SUMMARY_CSV = "same_base_union_candidate_selector_summary.csv"
REPORT_MD = "same_base_union_candidate_selector.md"
PLOT_PDF = "same_base_union_candidate_selector_deltas.pdf"
TABLE_TEX = "same_base_union_candidate_selector.tex"

BASELINE_METHODS = ["greedy_soup", "task_arithmetic", "dare", "ties_merging"]
DEFAULT_POOL_METHODS = [
    "base_model",
    "weight_average",
    "greedy_soup",
    "task_arithmetic",
    "dare",
    "ties_merging",
    "slerp_sequential",
]
TOL = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-csv", type=Path, default=ROOT / "reports" / "csv" / "same_base_task_vector_benchmark.csv")
    parser.add_argument("--candidate-grid-csv", type=Path, default=ROOT / "reports" / "csv" / "same_base_task_vector_candidate_grid.csv")
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "reports")
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--exclude-slerp", action="store_true", help="Do not include the SLERP candidate family.")
    args = parser.parse_args()
    args.command_string = " ".join([sys.executable, *sys.argv])
    return args


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def safe_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def safe_json(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, float) and not math.isfinite(value):
        return default
    try:
        return json.loads(str(value))
    except Exception:
        return default


def compact_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def bootstrap_mean_ci(values: pd.Series | np.ndarray, samples: int, seed: int) -> tuple[float, float]:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1 or samples <= 0:
        value = float(arr.mean())
        return value, value
    rng = np.random.default_rng(seed)
    draws = np.empty(int(samples), dtype=float)
    for idx in range(int(samples)):
        draws[idx] = float(rng.choice(arr, size=arr.size, replace=True).mean())
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def candidate_family(method: str, role: str) -> str:
    if method == "base_model":
        return "base_model"
    if method.startswith("individual_finetuned_") and role == "individual_task_model":
        return "individual_finetuned_checkpoint"
    if method == "weight_average":
        return "weight_average"
    if method == "greedy_soup":
        return "greedy_soup_final_candidate"
    if method == "task_arithmetic":
        return "task_arithmetic_selected"
    if method == "dare":
        return "dare_selected"
    if method == "ties_merging":
        return "ties_selected"
    if method == "slerp_sequential":
        return "slerp_selected"
    return method


def candidate_label(row: pd.Series) -> str:
    method = str(row["method"])
    if method.startswith("individual_finetuned_"):
        params = safe_json(row.get("selected_hyperparameters_json"), {})
        task = params.get("task_name")
        if task:
            return f"{method}:{task}"
    return method


def selected_params_for_method(row: pd.Series) -> dict:
    payload = safe_json(row.get("selected_hyperparameters_json"), {})
    if not isinstance(payload, dict):
        return {}
    return {key: value for key, value in payload.items() if key != "selection_trace"}


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"true", "1", "yes"}


def pool_methods(include_slerp: bool) -> list[str]:
    methods = list(DEFAULT_POOL_METHODS)
    if not include_slerp:
        methods = [method for method in methods if method != "slerp_sequential"]
    return methods


def candidate_pool_for_run(group: pd.DataFrame, include_slerp: bool) -> pd.DataFrame:
    methods = set(pool_methods(include_slerp))
    base = group[
        group["status"].astype(str).eq("ok")
        & (
            group["method"].isin(methods)
            | (
                group["method"].astype(str).str.startswith("individual_finetuned_")
                & group["method_role"].astype(str).eq("individual_task_model")
            )
        )
    ].copy()
    if base.empty:
        return base
    base["candidate_family"] = [
        candidate_family(str(row["method"]), str(row["method_role"])) for _, row in base.iterrows()
    ]
    base["candidate_label"] = [candidate_label(row) for _, row in base.iterrows()]
    base["selected_hyperparameters_compact_json"] = [
        compact_json(selected_params_for_method(row)) for _, row in base.iterrows()
    ]
    for col in [
        "validation_selected_accuracy",
        "validation_selected_loss",
        "average_test_accuracy",
        "worst_task_accuracy",
        "average_test_loss",
    ]:
        base[col] = pd.to_numeric(base[col], errors="coerce")
    # Deterministic tie-break: prefer simpler existing families, then stable label.
    family_order = {
        "base_model": 0,
        "individual_finetuned_checkpoint": 1,
        "weight_average": 2,
        "greedy_soup_final_candidate": 3,
        "task_arithmetic_selected": 4,
        "dare_selected": 5,
        "ties_selected": 6,
        "slerp_selected": 7,
    }
    base["tie_break_order"] = base["candidate_family"].map(family_order).fillna(100).astype(int)
    return base


def select_union_candidate(pool: pd.DataFrame) -> pd.Series | None:
    if pool.empty:
        return None
    eligible = pool.dropna(subset=["validation_selected_accuracy", "validation_selected_loss"]).copy()
    if eligible.empty:
        return None
    eligible = eligible.sort_values(
        ["validation_selected_accuracy", "validation_selected_loss", "tie_break_order", "candidate_label"],
        ascending=[False, True, True, True],
    )
    return eligible.iloc[0]


def best_existing_by_setting(benchmark: pd.DataFrame) -> dict[tuple, str]:
    setting_cols = ["dataset", "task_preset", "architecture", "width", "n_tasks"]
    out = {}
    for key, group in benchmark.groupby(setting_cols, dropna=False):
        comparable = group[group["method"].isin(BASELINE_METHODS) & group["status"].astype(str).eq("ok")].copy()
        if comparable.empty:
            continue
        summary = comparable.groupby("method", dropna=False).agg(
            mean_average_test_accuracy=("average_test_accuracy", "mean"),
            mean_worst_task_accuracy=("worst_task_accuracy", "mean"),
            n_unique_seeds=("seed", "nunique"),
        ).reset_index()
        summary = summary.sort_values(
            ["mean_average_test_accuracy", "mean_worst_task_accuracy", "method"],
            ascending=[False, False, True],
        )
        out[key] = str(summary.iloc[0]["method"])
    return out


def grid_diagnostics(grid_group: pd.DataFrame) -> dict:
    if grid_group.empty:
        return {
            "candidate_grid_available": False,
            "greedy_soup_grid_candidates_count": 0,
            "greedy_soup_grid_final_is_validation_max": False,
        }
    greedy = grid_group[grid_group["method"].eq("greedy_soup")].copy()
    if greedy.empty:
        return {
            "candidate_grid_available": True,
            "greedy_soup_grid_candidates_count": 0,
            "greedy_soup_grid_final_is_validation_max": False,
        }
    selected = greedy[greedy["selected"].map(bool_value)]
    if selected.empty:
        final_is_max = False
    else:
        final_acc = safe_float(selected.sort_values("candidate_rank").iloc[-1]["validation_accuracy"])
        max_acc = safe_float(pd.to_numeric(greedy["validation_accuracy"], errors="coerce").max())
        final_is_max = math.isfinite(final_acc) and math.isfinite(max_acc) and final_acc + 1e-12 >= max_acc
    return {
        "candidate_grid_available": True,
        "greedy_soup_grid_candidates_count": int(len(greedy)),
        "greedy_soup_grid_final_is_validation_max": bool(final_is_max),
    }


def build_run_rows(benchmark: pd.DataFrame, candidate_grid: pd.DataFrame, include_slerp: bool) -> pd.DataFrame:
    rows = []
    setting_best = best_existing_by_setting(benchmark)
    grid_by_run = {run_id: group for run_id, group in candidate_grid.groupby("run_id", dropna=False)} if not candidate_grid.empty else {}
    setting_cols = ["dataset", "task_preset", "architecture", "width", "n_tasks"]
    for run_id, group in benchmark.groupby("run_id", dropna=False):
        meta_row = group.iloc[0]
        key = tuple(meta_row[col] for col in setting_cols)
        best_setting_method = setting_best.get(key, "")
        best_setting_row = group[group["method"].eq(best_setting_method)].iloc[0] if best_setting_method else None
        baseline_group = group[group["method"].isin(BASELINE_METHODS) & group["status"].astype(str).eq("ok")].copy()
        if not baseline_group.empty:
            baseline_group = baseline_group.sort_values(
                ["average_test_accuracy", "worst_task_accuracy", "method"],
                ascending=[False, False, True],
            )
            run_oracle = baseline_group.iloc[0]
        else:
            run_oracle = None
        pool = candidate_pool_for_run(group, include_slerp)
        selected = select_union_candidate(pool)
        if selected is None:
            selected = pd.Series(dtype=object)
        candidate_families = pool["candidate_family"].tolist() if not pool.empty else []
        candidate_methods = pool["method"].tolist() if not pool.empty else []
        selected_test = safe_float(selected.get("average_test_accuracy"))
        selected_worst = safe_float(selected.get("worst_task_accuracy"))
        best_setting_test = safe_float(best_setting_row.get("average_test_accuracy")) if best_setting_row is not None else float("nan")
        best_setting_worst = safe_float(best_setting_row.get("worst_task_accuracy")) if best_setting_row is not None else float("nan")
        run_oracle_test = safe_float(run_oracle.get("average_test_accuracy")) if run_oracle is not None else float("nan")
        run_oracle_worst = safe_float(run_oracle.get("worst_task_accuracy")) if run_oracle is not None else float("nan")
        row = {
            "setting_id": meta_row["setting_id"],
            "run_id": run_id,
            "dataset": meta_row["dataset"],
            "task_preset": meta_row["task_preset"],
            "architecture": meta_row["architecture"],
            "width": int(meta_row["width"]),
            "n_tasks": int(meta_row["n_tasks"]),
            "seed": int(meta_row["seed"]),
            "selector": "same_base_union_validation_selector",
            "validation_protocol": "replay_existing_validation_split",
            "nested_validation_available": False,
            "test_metrics_used_for_selection": False,
            "candidate_pool_count": int(len(pool)),
            "candidate_pool_methods_json": compact_json(sorted(set(str(item) for item in candidate_methods))),
            "candidate_pool_families_json": compact_json(sorted(set(str(item) for item in candidate_families))),
            "selected_candidate_family": selected.get("candidate_family", ""),
            "selected_method": selected.get("method", ""),
            "selected_method_role": selected.get("method_role", ""),
            "selected_candidate_label": selected.get("candidate_label", ""),
            "selected_hyperparameters_json": selected.get("selected_hyperparameters_compact_json", "{}"),
            "selected_validation_accuracy": safe_float(selected.get("validation_selected_accuracy")),
            "selected_validation_loss": safe_float(selected.get("validation_selected_loss")),
            "selected_average_test_accuracy": selected_test,
            "selected_worst_task_accuracy": selected_worst,
            "selected_average_test_loss": safe_float(selected.get("average_test_loss")),
            "selected_uses_validation_data": bool_value(selected.get("uses_validation_data", False)),
            "selected_single_model": bool_value(selected.get("single_model", True)),
            "selected_capacity_matched": bool_value(selected.get("capacity_matched", True)),
            "best_existing_setting_method": best_setting_method,
            "best_existing_setting_average_test_accuracy": best_setting_test,
            "best_existing_setting_worst_task_accuracy": best_setting_worst,
            "delta_vs_best_existing_setting_method": selected_test - best_setting_test,
            "worst_task_delta_vs_best_existing_setting_method": selected_worst - best_setting_worst,
            "run_oracle_best_existing_method": run_oracle.get("method", "") if run_oracle is not None else "",
            "run_oracle_best_existing_average_test_accuracy": run_oracle_test,
            "run_oracle_best_existing_worst_task_accuracy": run_oracle_worst,
            "delta_vs_run_oracle_best_existing": selected_test - run_oracle_test,
            "worst_task_delta_vs_run_oracle_best_existing": selected_worst - run_oracle_worst,
            "selected_matches_best_existing_setting_method": bool(selected.get("method", "") == best_setting_method),
            "selected_matches_run_oracle_best_existing": bool(selected.get("method", "") == (run_oracle.get("method", "") if run_oracle is not None else "")),
            "include_slerp": bool(include_slerp),
            "claim_boundary": "same-base exact-setting validation selector; no independent-seed rebasin or Brauer/projective claim",
        }
        row.update(grid_diagnostics(grid_by_run.get(run_id, pd.DataFrame())))
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_runs(runs: pd.DataFrame, bootstrap_samples: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    setting_cols = ["dataset", "task_preset", "architecture", "width", "n_tasks"]
    for key, group in runs.groupby(setting_cols, dropna=False):
        dataset, task_preset, architecture, width, n_tasks = key
        deltas = pd.to_numeric(group["delta_vs_best_existing_setting_method"], errors="coerce")
        worst_deltas = pd.to_numeric(group["worst_task_delta_vs_best_existing_setting_method"], errors="coerce")
        oracle_deltas = pd.to_numeric(group["delta_vs_run_oracle_best_existing"], errors="coerce")
        low, high = bootstrap_mean_ci(deltas, bootstrap_samples, seed=36000 + len(rows) * 101)
        worst_low, worst_high = bootstrap_mean_ci(worst_deltas, bootstrap_samples, seed=36500 + len(rows) * 101)
        oracle_low, oracle_high = bootstrap_mean_ci(oracle_deltas, bootstrap_samples, seed=37000 + len(rows) * 101)
        n_unique = int(group["seed"].nunique())
        mean_delta = float(deltas.mean())
        if n_unique < 20:
            claim = "descriptive_below_20_seed_gate"
        elif math.isfinite(low) and low > 0.0:
            claim = "supported_exact_setting_union_improves_best_existing"
        elif abs(mean_delta) <= TOL and int((deltas.abs() <= TOL).sum()) == len(group):
            claim = "matched_best_existing_no_positive_gain"
        elif mean_delta >= 0.0:
            claim = "descriptive_nonnegative_mean_ci_crosses_zero"
        else:
            claim = "unsupported_union_below_best_existing"
        selected_counts = group["selected_method"].value_counts(dropna=False)
        family_counts = group["selected_candidate_family"].value_counts(dropna=False)
        rows.append(
            {
                "summary_type": "setting_summary",
                "scope": "fixed_setting",
                "dataset": dataset,
                "task_preset": task_preset,
                "architecture": architecture,
                "width": int(width),
                "n_tasks": int(n_tasks),
                "n_runs": int(len(group)),
                "n_unique_seeds": n_unique,
                "best_existing_setting_method": str(group["best_existing_setting_method"].iloc[0]),
                "selected_method_mode": str(selected_counts.index[0]) if len(selected_counts) else "",
                "selected_method_mode_count": int(selected_counts.iloc[0]) if len(selected_counts) else 0,
                "selected_candidate_family_mode": str(family_counts.index[0]) if len(family_counts) else "",
                "selected_candidate_family_mode_count": int(family_counts.iloc[0]) if len(family_counts) else 0,
                "mean_selected_average_test_accuracy": float(pd.to_numeric(group["selected_average_test_accuracy"], errors="coerce").mean()),
                "mean_best_existing_setting_average_test_accuracy": float(pd.to_numeric(group["best_existing_setting_average_test_accuracy"], errors="coerce").mean()),
                "mean_delta_vs_best_existing_setting_method": mean_delta,
                "delta_vs_best_existing_ci_low": low,
                "delta_vs_best_existing_ci_high": high,
                "wins_vs_best_existing_setting_method": int((deltas > TOL).sum()),
                "ties_vs_best_existing_setting_method": int((deltas.abs() <= TOL).sum()),
                "losses_vs_best_existing_setting_method": int((deltas < -TOL).sum()),
                "mean_worst_task_delta_vs_best_existing_setting_method": float(worst_deltas.mean()),
                "worst_task_delta_ci_low": worst_low,
                "worst_task_delta_ci_high": worst_high,
                "mean_delta_vs_run_oracle_best_existing": float(oracle_deltas.mean()),
                "delta_vs_run_oracle_ci_low": oracle_low,
                "delta_vs_run_oracle_ci_high": oracle_high,
                "match_best_existing_setting_method_fraction": float(group["selected_matches_best_existing_setting_method"].astype(bool).mean()),
                "match_run_oracle_best_existing_fraction": float(group["selected_matches_run_oracle_best_existing"].astype(bool).mean()),
                "mean_candidate_pool_count": float(pd.to_numeric(group["candidate_pool_count"], errors="coerce").mean()),
                "greedy_soup_grid_final_is_validation_max_fraction": float(group["greedy_soup_grid_final_is_validation_max"].astype(bool).mean()),
                "claim_decision": claim,
                "claim_boundary": "exact-setting only; requires paired CI lower bound > 0 for improvement claim",
            }
        )
        total = max(len(group), 1)
        for method, count in selected_counts.items():
            rows.append(
                {
                    "summary_type": "selection_frequency",
                    "scope": "fixed_setting",
                    "dataset": dataset,
                    "task_preset": task_preset,
                    "architecture": architecture,
                    "width": int(width),
                    "n_tasks": int(n_tasks),
                    "selected_method": method,
                    "selected_count": int(count),
                    "selected_fraction": float(count / total),
                    "n_runs": int(total),
                    "claim_boundary": "validation-selected frequency; descriptive",
                }
            )
        for baseline in BASELINE_METHODS:
            baseline_delta = []
            for _, run in group.iterrows():
                # The run CSV already stores the best-setting method and run oracle;
                # per-baseline deltas are recovered from source benchmark below when
                # available in the appended columns.
                col = f"delta_vs_{baseline}"
                if col in group:
                    baseline_delta.append(safe_float(run.get(col)))
            if baseline_delta:
                b_low, b_high = bootstrap_mean_ci(pd.Series(baseline_delta), bootstrap_samples, seed=37500 + len(rows) * 17)
                rows.append(
                    {
                        "summary_type": "baseline_comparison",
                        "scope": "fixed_setting",
                        "dataset": dataset,
                        "task_preset": task_preset,
                        "architecture": architecture,
                        "width": int(width),
                        "n_tasks": int(n_tasks),
                        "baseline_method": baseline,
                        "mean_delta_vs_baseline": float(np.nanmean(baseline_delta)),
                        "delta_ci_low": b_low,
                        "delta_ci_high": b_high,
                        "n_runs": int(len(baseline_delta)),
                        "claim_boundary": "secondary comparison; primary gate uses best existing setting method",
                    }
                )
    rows.append(
        {
            "summary_type": "claim_boundary",
            "scope": "overall",
            "dataset": "ALL",
            "task_preset": "ALL",
            "claim_decision": "same_base_union_selector_audit_only",
            "claim_boundary": "This is same-base validation selection over existing candidates; it does not write paper prose or certify independent-seed/rebasin obstruction claims.",
        }
    )
    return pd.DataFrame(rows)


def append_per_baseline_deltas(runs: pd.DataFrame, benchmark: pd.DataFrame) -> pd.DataFrame:
    out = runs.copy()
    lookup = {}
    for _, row in benchmark[benchmark["method"].isin(BASELINE_METHODS)].iterrows():
        lookup[(row["run_id"], row["method"])] = safe_float(row["average_test_accuracy"])
    for baseline in BASELINE_METHODS:
        out[f"delta_vs_{baseline}"] = [
            safe_float(row["selected_average_test_accuracy"]) - lookup.get((row["run_id"], baseline), float("nan"))
            for _, row in out.iterrows()
        ]
    return out


def write_plot(runs: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    data = runs.copy()
    data["setting_label"] = data.apply(
        lambda row: f"{row['dataset']} {row['task_preset']} W{int(row['width'])}", axis=1
    )
    labels = []
    values = []
    for label, group in data.groupby("setting_label", sort=False):
        labels.append(label)
        values.append(pd.to_numeric(group["delta_vs_best_existing_setting_method"], errors="coerce").dropna().to_numpy())
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5))
    if values:
        axes[0].boxplot(values, tick_labels=labels, showmeans=True)
        axes[0].axhline(0.0, color="black", linewidth=0.8)
        axes[0].tick_params(axis="x", rotation=35, labelsize=8)
        axes[0].set_ylabel("union selector test accuracy delta")
        axes[0].set_title("Delta vs best existing exact-setting method")
        axes[0].grid(True, axis="y", alpha=0.25)
    else:
        axes[0].text(0.5, 0.5, "No rows", ha="center", va="center")
        axes[0].set_axis_off()
    counts = data["selected_method"].value_counts()
    if not counts.empty:
        axes[1].bar(counts.index.astype(str), counts.values, color="#2563eb", alpha=0.8)
        axes[1].tick_params(axis="x", rotation=35, labelsize=8)
        axes[1].set_ylabel("selected runs")
        axes[1].set_title("Union-selected method frequency")
        axes[1].grid(True, axis="y", alpha=0.25)
    else:
        axes[1].text(0.5, 0.5, "No selections", ha="center", va="center")
        axes[1].set_axis_off()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def tex_escape(value: Any) -> str:
    text = str(value)
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("$", "\\$")
        .replace("#", "\\#")
        .replace("_", "\\_")
        .replace("{", "\\{")
        .replace("}", "\\}")
    )


def write_latex_table(summary: pd.DataFrame, path: Path) -> None:
    settings = summary[summary["summary_type"].eq("setting_summary")].copy()
    columns = [
        ("dataset", "Dataset"),
        ("task_preset", "Task preset"),
        ("width", "Width"),
        ("best_existing_setting_method", "Best existing"),
        ("selected_method_mode", "Union mode"),
        ("mean_delta_vs_best_existing_setting_method", "Mean delta"),
        ("delta_vs_best_existing_ci_low", "CI low"),
        ("delta_vs_best_existing_ci_high", "CI high"),
        ("claim_decision", "Claim decision"),
    ]
    lines = [
        "\\begin{tabular}{llrllrrrl}",
        "\\toprule",
        " & ".join(label for _col, label in columns) + " \\\\",
        "\\midrule",
    ]
    for _, row in settings.iterrows():
        cells = []
        for col, _label in columns:
            value = row.get(col, "")
            if isinstance(value, (float, np.floating)):
                cells.append(f"{float(value):.4f}" if math.isfinite(float(value)) else "")
            else:
                cells.append(tex_escape(value))
        lines.append(" & ".join(cells) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def md_table(df: pd.DataFrame, columns: list[str], max_rows: int = 40) -> str:
    if df.empty:
        return "_No rows._"
    view = df.head(max_rows).copy()
    rows = []
    for row in view.to_dict("records"):
        clean = {}
        for col in columns:
            value = row.get(col, "")
            if isinstance(value, (float, np.floating)):
                clean[col] = "" if not math.isfinite(float(value)) else f"{float(value):.4f}"
            else:
                clean[col] = value
        rows.append(clean)
    text = format_markdown_table(rows, columns)
    if len(df) > max_rows:
        text += f"\n\n_Showing {max_rows} of {len(df)} rows._"
    return text


def write_report(args: argparse.Namespace, runs: pd.DataFrame, summary: pd.DataFrame, path: Path) -> None:
    settings = summary[summary["summary_type"].eq("setting_summary")].copy()
    frequency = summary[summary["summary_type"].eq("selection_frequency")].copy()
    positive = settings[settings["claim_decision"].eq("supported_exact_setting_union_improves_best_existing")]
    if positive.empty:
        decision = (
            "No exact setting passes the positive paired-CI gate versus the best existing method. "
            "The result is negative/diagnostic: simple validation selection among existing same-base methods is already hard to improve."
        )
    else:
        decision = f"{len(positive)} exact setting(s) pass the positive paired-CI gate versus the best existing method."
    report = f"""# Same-Base Union Candidate Selector

Generated by `experiments/same_base_union_candidate_selector.py`.

## Exact Command

```bash
{args.command_string}
```

## Inputs

- `reports/csv/same_base_task_vector_benchmark.csv`
- `reports/csv/same_base_task_vector_candidate_grid.csv`

## Outputs

- `reports/csv/{RUN_CSV}`
- `reports/csv/{SUMMARY_CSV}`
- `reports/{REPORT_MD}`
- `reports/plots/{PLOT_PDF}`
- `reports/tables/{TABLE_TEX}`

## Selector

The union pool contains the common base model, individual fine-tuned checkpoints, weight average, the final greedy-soup candidate, selected Task Arithmetic, selected DARE, selected TIES, and {'SLERP' if not args.exclude_slerp else 'no SLERP'} candidates. Selection uses validation accuracy with validation loss as a tie-breaker. Test metrics are read only after the selected candidate is fixed.

Nested validation was not available in the Prompt 31 artifacts, so this is a replay validation audit rather than a nested-selection result. Greedy-soup intermediate candidates are represented by the final validation-selected greedy soup; the candidate grid verifies whether that final greedy candidate was also the maximum-validation greedy prefix.

## Claim Decision

{decision}

Positive claim gate: paired CI lower bound greater than zero versus the best of `greedy_soup`, `task_arithmetic`, `dare`, and `ties_merging` in the exact setting, with at least 20 seeds.

## Setting Summary

{md_table(settings, ["dataset", "task_preset", "width", "n_tasks", "n_unique_seeds", "best_existing_setting_method", "selected_method_mode", "mean_selected_average_test_accuracy", "mean_best_existing_setting_average_test_accuracy", "mean_delta_vs_best_existing_setting_method", "delta_vs_best_existing_ci_low", "delta_vs_best_existing_ci_high", "wins_vs_best_existing_setting_method", "ties_vs_best_existing_setting_method", "losses_vs_best_existing_setting_method", "match_best_existing_setting_method_fraction", "claim_decision"], 50)}

## Selection Frequencies

{md_table(frequency, ["dataset", "task_preset", "width", "selected_method", "selected_count", "selected_fraction", "n_runs"], 80)}

## Run-Level Sample

{md_table(runs, ["dataset", "task_preset", "width", "seed", "selected_method", "selected_candidate_family", "selected_validation_accuracy", "selected_average_test_accuracy", "best_existing_setting_method", "delta_vs_best_existing_setting_method", "run_oracle_best_existing_method", "delta_vs_run_oracle_best_existing"], 30)}

## Claim Boundary

- This is a same-base task-vector candidate-family selector, not an independent-seed/rebasin benchmark.
- It does not certify Brauer/projective obstruction or any natural-model-merging superiority claim.
- If no setting passes the positive gate, the supported interpretation is that the existing validation-selected method families already cover the useful same-base candidates in this artifact set.
"""
    path.write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    benchmark = pd.read_csv(args.benchmark_csv)
    candidate_grid = pd.read_csv(args.candidate_grid_csv) if args.candidate_grid_csv.exists() else pd.DataFrame()
    runs = build_run_rows(benchmark, candidate_grid, include_slerp=not args.exclude_slerp)
    runs = append_per_baseline_deltas(runs, benchmark)
    summary = summarize_runs(runs, args.bootstrap_samples)

    csv_dir = args.reports_dir / "csv"
    plot_dir = args.reports_dir / "plots"
    table_dir = args.reports_dir / "tables"
    csv_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    runs.to_csv(csv_dir / RUN_CSV, index=False, lineterminator="\n")
    summary.to_csv(csv_dir / SUMMARY_CSV, index=False, lineterminator="\n")
    write_plot(runs, plot_dir / PLOT_PDF)
    write_latex_table(summary, table_dir / TABLE_TEX)
    write_report(args, runs, summary, args.reports_dir / REPORT_MD)
    print(f"wrote {csv_dir / RUN_CSV}")
    print(f"wrote {csv_dir / SUMMARY_CSV}")
    print(f"wrote {args.reports_dir / REPORT_MD}")
    print(f"wrote {plot_dir / PLOT_PDF}")
    print(f"wrote {table_dir / TABLE_TEX}")
    print(f"commit {git_commit()}")


if __name__ == "__main__":
    main()
