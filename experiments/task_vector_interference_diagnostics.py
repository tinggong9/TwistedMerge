#!/usr/bin/env python
"""Task-vector interference diagnostics for the same-base benchmark.

This script separates same-base task-vector geometry from independent-seed
rebasin/cycle obstruction diagnostics.  The inputs are the Prompt 29
same-base benchmark CSVs; no training is performed here.
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


RUNS_CSV = "task_vector_interference_diagnostics.csv"
SUMMARY_CSV = "task_vector_interference_summary.csv"
REPORT_MD = "task_vector_interference_report.md"
PLOT_PDF = "task_vector_interference_vs_delta.pdf"

GENERATED_METHODS = [
    "weight_average",
    "greedy_soup",
    "slerp_sequential",
    "task_arithmetic",
    "ties_merging",
    "dare",
]
TARGET_METHODS = {
    "task_arithmetic": "task_arithmetic",
    "ties_merging": "ties",
    "dare": "dare",
}
TARGET_COLUMNS = [
    "task_arithmetic_delta_vs_greedy",
    "ties_delta_vs_greedy",
    "dare_delta_vs_greedy",
    "best_generated_delta_vs_greedy",
    "best_generated_worst_task_delta_vs_greedy",
]
PREDICTORS = [
    ("task_vector_sign_conflict_fraction", "task_vector_interference"),
    ("task_vector_active_fraction", "task_vector_interference"),
    ("task_vector_mean_pairwise_cosine", "task_vector_interference"),
    ("task_vector_min_pairwise_cosine", "task_vector_interference"),
    ("task_vector_mean_delta_norm", "delta_norm"),
    ("task_vector_min_delta_norm", "delta_norm"),
    ("task_vector_max_delta_norm", "delta_norm"),
    ("task_vector_std_delta_norm", "delta_norm"),
    ("task_vector_delta_norm_cv", "delta_norm"),
    ("task_vector_merged_delta_norm", "delta_norm"),
    ("task_vector_mean_pairwise_delta_distance", "delta_norm"),
    ("greedy_soup_accepted_count", "selector_behavior"),
    ("task_arithmetic_selected_scale", "selector_behavior"),
    ("ties_selected_density", "selector_behavior"),
    ("ties_selected_scale", "selector_behavior"),
    ("dare_selected_drop_rate", "selector_behavior"),
    ("dare_selected_scale", "selector_behavior"),
    ("triangle_cycle_score", "secondary_rebasin_cycle"),
    ("sync_disagreement", "secondary_rebasin_cycle"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-csv", type=Path, default=ROOT / "reports" / "csv" / "same_base_task_vector_benchmark.csv")
    parser.add_argument("--summary-csv", type=Path, default=ROOT / "reports" / "csv" / "same_base_task_vector_summary.csv")
    parser.add_argument("--candidate-grid-csv", type=Path, default=ROOT / "reports" / "csv" / "same_base_task_vector_candidate_grid.csv")
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "reports")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--skip-delta-norms", action="store_true", help="Skip checkpoint loading and leave delta-norm fields empty.")
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


def safe_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def bootstrap_mean_ci(values: pd.Series | np.ndarray, samples: int, seed: int) -> tuple[float, float]:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1 or samples <= 0:
        value = float(arr.mean())
        return value, value
    rng = np.random.default_rng(seed)
    means = np.empty(int(samples), dtype=float)
    for idx in range(int(samples)):
        means[idx] = float(rng.choice(arr, size=arr.size, replace=True).mean())
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def safe_corr(x: pd.Series, y: pd.Series, method: str) -> float:
    x_num = pd.to_numeric(x, errors="coerce")
    y_num = pd.to_numeric(y, errors="coerce")
    mask = x_num.notna() & y_num.notna()
    if int(mask.sum()) < 3:
        return float("nan")
    if x_num[mask].nunique() < 2 or y_num[mask].nunique() < 2:
        return float("nan")
    return float(x_num[mask].corr(y_num[mask], method=method))


def state_vector_from_checkpoint(path: str | Path) -> np.ndarray:
    import torch

    checkpoint = torch.load(Path(path), map_location="cpu")
    state = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    pieces = []
    for key in sorted(state):
        tensor = state[key]
        if hasattr(tensor, "detach"):
            arr = tensor.detach().cpu().numpy()
        else:
            arr = np.asarray(tensor)
        pieces.append(arr.reshape(-1).astype(np.float64))
    return np.concatenate(pieces)


def delta_norm_stats(row: pd.Series, vector_cache: dict[str, np.ndarray]) -> dict[str, Any]:
    base_path = str(row.get("base_checkpoint", ""))
    task_paths = safe_json(row.get("task_checkpoints_json", "[]"), [])
    if not base_path or not task_paths:
        return {"delta_norms_available": False, "delta_norms_missing_reason": "missing_checkpoint_paths"}
    missing = [path for path in [base_path, *task_paths] if not Path(path).exists()]
    if missing:
        return {
            "delta_norms_available": False,
            "delta_norms_missing_reason": "checkpoint_missing",
            "delta_norm_missing_count": int(len(missing)),
        }
    try:
        if base_path not in vector_cache:
            vector_cache[base_path] = state_vector_from_checkpoint(base_path)
        base = vector_cache[base_path]
        deltas = []
        for path in task_paths:
            path = str(path)
            if path not in vector_cache:
                vector_cache[path] = state_vector_from_checkpoint(path)
            deltas.append(vector_cache[path] - base)
    except Exception as exc:
        return {
            "delta_norms_available": False,
            "delta_norms_missing_reason": f"checkpoint_load_failed:{type(exc).__name__}",
        }
    matrix = np.stack(deltas, axis=0)
    norms = np.linalg.norm(matrix, axis=1)
    pairwise_distances = []
    for idx in range(len(deltas)):
        for jdx in range(idx + 1, len(deltas)):
            pairwise_distances.append(float(np.linalg.norm(matrix[idx] - matrix[jdx])))
    merged = matrix.mean(axis=0)
    mean_norm = float(norms.mean())
    return {
        "delta_norms_available": True,
        "delta_norms_missing_reason": "",
        "delta_norm_missing_count": 0,
        "task_vector_mean_delta_norm": mean_norm,
        "task_vector_min_delta_norm": float(norms.min()),
        "task_vector_max_delta_norm": float(norms.max()),
        "task_vector_std_delta_norm": float(norms.std(ddof=0)),
        "task_vector_delta_norm_cv": float(norms.std(ddof=0) / mean_norm) if mean_norm > 0 else float("nan"),
        "task_vector_merged_delta_norm": float(np.linalg.norm(merged)),
        "task_vector_mean_pairwise_delta_distance": float(np.mean(pairwise_distances)) if pairwise_distances else 0.0,
        "task_vector_max_pairwise_delta_distance": float(np.max(pairwise_distances)) if pairwise_distances else 0.0,
    }


def selected_grid_row(grid_group: pd.DataFrame, method: str) -> pd.Series | None:
    subset = grid_group[grid_group["method"].eq(method)].copy()
    if subset.empty:
        return None
    selected = subset[safe_bool_series(subset["selected"])]
    if selected.empty:
        return None
    return selected.sort_values(["candidate_rank"]).iloc[0]


def greedy_acceptance_stats(grid_group: pd.DataFrame, benchmark_group: pd.DataFrame) -> dict[str, Any]:
    greedy_grid = grid_group[grid_group["method"].eq("greedy_soup")].copy()
    if not greedy_grid.empty:
        accepted = safe_bool_series(greedy_grid["accepted"])
        selected = safe_bool_series(greedy_grid["selected"])
        final_size = float("nan")
        if selected.any():
            payload = safe_json(greedy_grid[selected].sort_values("candidate_rank").iloc[-1].get("candidate_params_json"), {})
            final_size = len(payload.get("soup_indices_after", [])) if isinstance(payload, dict) else float("nan")
        return {
            "greedy_soup_candidate_count": int(len(greedy_grid)),
            "greedy_soup_accepted_count": int(accepted.sum()),
            "greedy_soup_rejected_count": int((~accepted).sum()),
            "greedy_soup_acceptance_fraction": float(accepted.mean()) if len(accepted) else float("nan"),
            "greedy_soup_final_size": final_size,
        }
    greedy_row = benchmark_group[benchmark_group["method"].eq("greedy_soup")]
    if greedy_row.empty:
        return {
            "greedy_soup_candidate_count": 0,
            "greedy_soup_accepted_count": float("nan"),
            "greedy_soup_rejected_count": float("nan"),
            "greedy_soup_acceptance_fraction": float("nan"),
            "greedy_soup_final_size": float("nan"),
        }
    trajectory = safe_json(greedy_row.iloc[0].get("greedy_soup_acceptance_json"), [])
    accepted_count = sum(bool(item.get("accepted")) for item in trajectory if isinstance(item, dict))
    candidate_count = len(trajectory)
    return {
        "greedy_soup_candidate_count": int(candidate_count),
        "greedy_soup_accepted_count": int(accepted_count),
        "greedy_soup_rejected_count": int(candidate_count - accepted_count),
        "greedy_soup_acceptance_fraction": float(accepted_count / candidate_count) if candidate_count else float("nan"),
        "greedy_soup_final_size": int(accepted_count),
    }


def method_row(group: pd.DataFrame, method: str) -> pd.Series | None:
    subset = group[group["method"].eq(method)]
    if subset.empty:
        return None
    return subset.iloc[0]


def build_diagnostics(benchmark: pd.DataFrame, grid: pd.DataFrame, compute_delta_norms: bool) -> pd.DataFrame:
    rows = []
    vector_cache: dict[str, np.ndarray] = {}
    grid_by_run = {run_id: group.copy() for run_id, group in grid.groupby("run_id", dropna=False)} if not grid.empty else {}
    run_groups = benchmark.groupby("run_id", dropna=False)
    for run_idx, (run_id, group) in enumerate(run_groups, start=1):
        base = group.iloc[0]
        grid_group = grid_by_run.get(run_id, pd.DataFrame(columns=grid.columns))
        greedy = method_row(group, "greedy_soup")
        greedy_avg = safe_float(greedy.get("average_test_accuracy")) if greedy is not None else float("nan")
        greedy_worst = safe_float(greedy.get("worst_task_accuracy")) if greedy is not None else float("nan")
        row = {
            "setting_id": base.get("setting_id", ""),
            "run_id": run_id,
            "dataset": base.get("dataset", ""),
            "task_preset": base.get("task_preset", ""),
            "architecture": base.get("architecture", ""),
            "width": int(base.get("width")),
            "n_tasks": int(base.get("n_tasks")),
            "seed": int(base.get("seed")),
            "base_epochs": safe_float(base.get("base_epochs")),
            "finetune_epochs": safe_float(base.get("finetune_epochs")),
            "task_vector_sign_conflict_fraction": safe_float(base.get("task_vector_sign_conflict_fraction")),
            "task_vector_active_fraction": safe_float(base.get("task_vector_active_fraction")),
            "task_vector_mean_pairwise_cosine": safe_float(base.get("task_vector_mean_pairwise_cosine")),
            "task_vector_min_pairwise_cosine": safe_float(base.get("task_vector_min_pairwise_cosine")),
            "triangle_cycle_score": safe_float(base.get("triangle_cycle_score")),
            "sync_disagreement": safe_float(base.get("sync_disagreement")),
            "greedy_soup_average_test_accuracy": greedy_avg,
            "greedy_soup_worst_task_accuracy": greedy_worst,
        }
        row.update(greedy_acceptance_stats(grid_group, group))
        if compute_delta_norms:
            row.update(delta_norm_stats(base, vector_cache))
        else:
            row.update({"delta_norms_available": False, "delta_norms_missing_reason": "skipped_by_flag"})

        generated = group[group["method"].isin(GENERATED_METHODS) & group["status"].eq("ok")].copy()
        if not generated.empty:
            generated["average_test_accuracy_num"] = pd.to_numeric(generated["average_test_accuracy"], errors="coerce")
            generated["worst_task_accuracy_num"] = pd.to_numeric(generated["worst_task_accuracy"], errors="coerce")
            best = generated.sort_values(["average_test_accuracy_num", "worst_task_accuracy_num"], ascending=False).iloc[0]
            best_worst = generated.sort_values(["worst_task_accuracy_num", "average_test_accuracy_num"], ascending=False).iloc[0]
            row.update(
                {
                    "best_generated_candidate_family": best["method"],
                    "best_generated_average_test_accuracy": safe_float(best["average_test_accuracy_num"]),
                    "best_generated_delta_vs_greedy": safe_float(best["average_test_accuracy_num"]) - greedy_avg,
                    "best_generated_worst_task_accuracy": safe_float(best["worst_task_accuracy_num"]),
                    "best_generated_worst_task_delta_vs_greedy": safe_float(best["worst_task_accuracy_num"]) - greedy_worst,
                    "best_worst_task_candidate_family": best_worst["method"],
                    "best_worst_task_accuracy": safe_float(best_worst["worst_task_accuracy_num"]),
                    "best_worst_task_delta_vs_greedy": safe_float(best_worst["worst_task_accuracy_num"]) - greedy_worst,
                    "greedy_soup_is_best_generated": bool(best["method"] == "greedy_soup"),
                }
            )
        else:
            row.update(
                {
                    "best_generated_candidate_family": "",
                    "best_generated_average_test_accuracy": float("nan"),
                    "best_generated_delta_vs_greedy": float("nan"),
                    "best_generated_worst_task_accuracy": float("nan"),
                    "best_generated_worst_task_delta_vs_greedy": float("nan"),
                    "best_worst_task_candidate_family": "",
                    "best_worst_task_accuracy": float("nan"),
                    "best_worst_task_delta_vs_greedy": float("nan"),
                    "greedy_soup_is_best_generated": False,
                }
            )

        for method, prefix in TARGET_METHODS.items():
            mrow = method_row(group, method)
            selected = selected_grid_row(grid_group, method)
            method_avg = safe_float(mrow.get("average_test_accuracy")) if mrow is not None else float("nan")
            method_worst = safe_float(mrow.get("worst_task_accuracy")) if mrow is not None else float("nan")
            row[f"{prefix}_average_test_accuracy"] = method_avg
            row[f"{prefix}_worst_task_accuracy"] = method_worst
            row[f"{prefix}_delta_vs_greedy"] = method_avg - greedy_avg
            row[f"{prefix}_worst_task_delta_vs_greedy"] = method_worst - greedy_worst
            row[f"{prefix}_validation_selected_accuracy"] = safe_float(mrow.get("validation_selected_accuracy")) if mrow is not None else float("nan")
            if selected is not None:
                row[f"{prefix}_selected_scale"] = safe_float(selected.get("scale"))
                row[f"{prefix}_selected_density"] = safe_float(selected.get("density"))
                row[f"{prefix}_selected_drop_rate"] = safe_float(selected.get("drop_rate"))
                row[f"{prefix}_selected_validation_accuracy"] = safe_float(selected.get("validation_accuracy"))
                row[f"{prefix}_selected_validation_loss"] = safe_float(selected.get("validation_loss"))
            else:
                row[f"{prefix}_selected_scale"] = float("nan")
                row[f"{prefix}_selected_density"] = float("nan")
                row[f"{prefix}_selected_drop_rate"] = float("nan")
                row[f"{prefix}_selected_validation_accuracy"] = float("nan")
                row[f"{prefix}_selected_validation_loss"] = float("nan")
        # Friendly aliases requested in the prompt.
        row["task_arithmetic_selected_scale"] = row.get("task_arithmetic_selected_scale", float("nan"))
        row["ties_selected_density"] = row.get("ties_selected_density", float("nan"))
        row["ties_selected_scale"] = row.get("ties_selected_scale", float("nan"))
        row["dare_selected_drop_rate"] = row.get("dare_selected_drop_rate", float("nan"))
        row["dare_selected_scale"] = row.get("dare_selected_scale", float("nan"))
        rows.append(row)
        if run_idx % 25 == 0:
            print(f"processed {run_idx}/{len(run_groups)} runs", flush=True)
    return pd.DataFrame(rows)


def association_rows(diagnostics: pd.DataFrame) -> list[dict]:
    rows = []
    scope_groups: list[tuple[str, dict[str, Any], pd.DataFrame]] = [
        ("overall", {"dataset": "ALL", "task_preset": "ALL", "width": ""}, diagnostics)
    ]
    for key, group in diagnostics.groupby(["dataset", "task_preset", "architecture", "width", "n_tasks"], dropna=False):
        dataset, task_preset, architecture, width, n_tasks = key
        scope_groups.append(
            (
                "fixed_setting",
                {
                    "dataset": dataset,
                    "task_preset": task_preset,
                    "architecture": architecture,
                    "width": width,
                    "n_tasks": n_tasks,
                },
                group,
            )
        )
    for scope, meta, group in scope_groups:
        for target in TARGET_COLUMNS:
            for predictor, predictor_role in PREDICTORS:
                pearson = safe_corr(group[predictor], group[target], "pearson") if predictor in group and target in group else float("nan")
                spearman = safe_corr(group[predictor], group[target], "spearman") if predictor in group and target in group else float("nan")
                n = int((pd.to_numeric(group.get(predictor), errors="coerce").notna() & pd.to_numeric(group.get(target), errors="coerce").notna()).sum()) if predictor in group and target in group else 0
                x_unique = int(pd.to_numeric(group.get(predictor), errors="coerce").nunique()) if predictor in group else 0
                y_unique = int(pd.to_numeric(group.get(target), errors="coerce").nunique()) if target in group else 0
                if predictor_role == "secondary_rebasin_cycle" and x_unique < 2:
                    boundary = "secondary cycle diagnostic has no variance in same-base regime"
                elif n < 10 and scope == "overall":
                    boundary = "too few paired rows for association claim"
                elif x_unique < 2 or y_unique < 2:
                    boundary = "descriptive only; predictor or target has no usable variance"
                else:
                    boundary = "descriptive association; same-base task-vector regime only"
                rows.append(
                    {
                        "summary_type": "association",
                        "scope": scope,
                        **meta,
                        "target": target,
                        "predictor": predictor,
                        "predictor_role": predictor_role,
                        "n_rows": n,
                        "predictor_unique_values": x_unique,
                        "target_unique_values": y_unique,
                        "pearson": pearson,
                        "spearman": spearman,
                        "claim_boundary": boundary,
                    }
                )
    return rows


def setting_rows(diagnostics: pd.DataFrame, bootstrap_samples: int) -> list[dict]:
    rows = []
    for key, group in diagnostics.groupby(["dataset", "task_preset", "architecture", "width", "n_tasks"], dropna=False):
        dataset, task_preset, architecture, width, n_tasks = key
        base = {
            "summary_type": "setting_summary",
            "scope": "fixed_setting",
            "dataset": dataset,
            "task_preset": task_preset,
            "architecture": architecture,
            "width": width,
            "n_tasks": n_tasks,
            "n_runs": int(len(group)),
            "n_unique_seeds": int(group["seed"].nunique()),
            "mean_sign_conflict_fraction": float(group["task_vector_sign_conflict_fraction"].mean()),
            "mean_active_fraction": float(group["task_vector_active_fraction"].mean()),
            "mean_pairwise_task_vector_cosine": float(group["task_vector_mean_pairwise_cosine"].mean()),
            "mean_min_pairwise_task_vector_cosine": float(group["task_vector_min_pairwise_cosine"].mean()),
            "mean_delta_norm": float(pd.to_numeric(group.get("task_vector_mean_delta_norm"), errors="coerce").mean()),
            "mean_pairwise_delta_distance": float(pd.to_numeric(group.get("task_vector_mean_pairwise_delta_distance"), errors="coerce").mean()),
            "mean_greedy_soup_accepted_count": float(pd.to_numeric(group["greedy_soup_accepted_count"], errors="coerce").mean()),
            "mean_task_arithmetic_selected_scale": float(pd.to_numeric(group["task_arithmetic_selected_scale"], errors="coerce").mean()),
            "mean_ties_selected_density": float(pd.to_numeric(group["ties_selected_density"], errors="coerce").mean()),
            "mean_ties_selected_scale": float(pd.to_numeric(group["ties_selected_scale"], errors="coerce").mean()),
            "mean_dare_selected_drop_rate": float(pd.to_numeric(group["dare_selected_drop_rate"], errors="coerce").mean()),
            "mean_dare_selected_scale": float(pd.to_numeric(group["dare_selected_scale"], errors="coerce").mean()),
            "mean_triangle_cycle_score": float(pd.to_numeric(group["triangle_cycle_score"], errors="coerce").mean()),
            "mean_sync_disagreement": float(pd.to_numeric(group["sync_disagreement"], errors="coerce").mean()),
            "delta_norm_availability_fraction": float(group["delta_norms_available"].astype(bool).mean()) if "delta_norms_available" in group else 0.0,
        }
        for target in TARGET_COLUMNS:
            values = pd.to_numeric(group[target], errors="coerce")
            low, high = bootstrap_mean_ci(values, bootstrap_samples, seed=33000 + len(rows) * 31)
            base[f"mean_{target}"] = float(values.mean())
            base[f"{target}_ci_low"] = low
            base[f"{target}_ci_high"] = high
        winners = group["best_generated_candidate_family"].value_counts(dropna=False)
        base["best_generated_candidate_family_mode"] = str(winners.index[0]) if len(winners) else ""
        base["best_generated_candidate_family_mode_count"] = int(winners.iloc[0]) if len(winners) else 0
        rows.append(base)
    return rows


def winner_rows(diagnostics: pd.DataFrame) -> list[dict]:
    rows = []
    for key, group in diagnostics.groupby(["dataset", "task_preset", "architecture", "width", "n_tasks"], dropna=False):
        dataset, task_preset, architecture, width, n_tasks = key
        total = max(len(group), 1)
        counts = group["best_generated_candidate_family"].value_counts(dropna=False)
        for method, count in counts.items():
            rows.append(
                {
                    "summary_type": "winner_frequency",
                    "scope": "fixed_setting",
                    "dataset": dataset,
                    "task_preset": task_preset,
                    "architecture": architecture,
                    "width": width,
                    "n_tasks": n_tasks,
                    "candidate_family": method,
                    "win_count": int(count),
                    "win_fraction": float(count / total),
                    "n_runs": int(total),
                    "claim_boundary": "winner is by test metric after validation selection; descriptive audit only",
                }
            )
    return rows


def build_summary(diagnostics: pd.DataFrame, bootstrap_samples: int) -> pd.DataFrame:
    rows: list[dict] = []
    rows.extend(setting_rows(diagnostics, bootstrap_samples))
    rows.extend(winner_rows(diagnostics))
    rows.extend(association_rows(diagnostics))
    # A compact global interpretation row.
    cycle_unique = int(pd.to_numeric(diagnostics["triangle_cycle_score"], errors="coerce").nunique())
    sync_unique = int(pd.to_numeric(diagnostics["sync_disagreement"], errors="coerce").nunique())
    rows.append(
        {
            "summary_type": "claim_boundary",
            "scope": "overall",
            "dataset": "ALL",
            "task_preset": "ALL",
            "n_runs": int(len(diagnostics)),
            "mean_sign_conflict_fraction": float(diagnostics["task_vector_sign_conflict_fraction"].mean()),
            "mean_triangle_cycle_score": float(pd.to_numeric(diagnostics["triangle_cycle_score"], errors="coerce").mean()),
            "triangle_cycle_unique_values": cycle_unique,
            "sync_disagreement_unique_values": sync_unique,
            "claim_decision": "same_base_task_vector_interference_audit_only",
            "claim_boundary": "Explains same-base task-vector behavior; does not certify Brauer/projective obstruction or replace independent-seed rebasin diagnostics.",
        }
    )
    return pd.DataFrame(rows)


def write_plot(diagnostics: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.2))
    panels = [
        ("task_vector_sign_conflict_fraction", "task_arithmetic_delta_vs_greedy", "sign conflict vs Task Arithmetic delta"),
        ("task_vector_sign_conflict_fraction", "ties_delta_vs_greedy", "sign conflict vs TIES delta"),
        ("task_vector_sign_conflict_fraction", "dare_delta_vs_greedy", "sign conflict vs DARE delta"),
        ("task_vector_mean_pairwise_cosine", "best_generated_delta_vs_greedy", "task-vector cosine vs best generated delta"),
    ]
    colors = {"mnist": "#2563eb", "fashion_mnist": "#dc2626"}
    for ax, (x_col, y_col, title) in zip(axes.flat, panels, strict=False):
        for dataset, group in diagnostics.groupby("dataset", dropna=False):
            x = pd.to_numeric(group[x_col], errors="coerce")
            y = pd.to_numeric(group[y_col], errors="coerce")
            ax.scatter(x, y, s=28, alpha=0.72, label=str(dataset), color=colors.get(str(dataset), None), edgecolor="none")
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=max(1, len(labels)))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path)
    plt.close(fig)


def md_table(df: pd.DataFrame, columns: list[str], max_rows: int = 40) -> str:
    if df.empty:
        return "_No rows._"
    view = df.head(max_rows).copy()
    records = []
    for row in view.to_dict("records"):
        out = {}
        for col in columns:
            value = row.get(col, "")
            if isinstance(value, (float, np.floating)):
                out[col] = "" if not math.isfinite(float(value)) else f"{float(value):.4f}"
            else:
                out[col] = value
        records.append(out)
    table = format_markdown_table(records, columns)
    if len(df) > max_rows:
        table += f"\n\n_Showing {max_rows} of {len(df)} rows._"
    return table


def top_association_table(summary: pd.DataFrame) -> pd.DataFrame:
    assoc = summary[summary["summary_type"].eq("association")].copy()
    if assoc.empty:
        return assoc
    assoc = assoc[assoc["scope"].eq("overall")].copy()
    assoc["abs_spearman"] = pd.to_numeric(assoc["spearman"], errors="coerce").abs()
    assoc = assoc.sort_values(["target", "abs_spearman"], ascending=[True, False])
    return assoc.groupby("target", dropna=False).head(6).reset_index(drop=True)


def write_report(args: argparse.Namespace, diagnostics: pd.DataFrame, summary: pd.DataFrame, path: Path) -> None:
    settings = summary[summary["summary_type"].eq("setting_summary")].copy()
    winners = summary[summary["summary_type"].eq("winner_frequency")].copy()
    associations = top_association_table(summary)
    boundary = summary[summary["summary_type"].eq("claim_boundary")].copy()
    boundary_row = boundary.iloc[0].to_dict() if not boundary.empty else {}
    cycle_zero = (
        pd.to_numeric(diagnostics["triangle_cycle_score"], errors="coerce").fillna(0.0).abs().max() <= 1e-12
        and pd.to_numeric(diagnostics["sync_disagreement"], errors="coerce").fillna(0.0).abs().max() <= 1e-12
    )
    report = f"""# Task-Vector Interference Diagnostics

Generated by `experiments/task_vector_interference_diagnostics.py`.

## Exact Command

```bash
{args.command_string}
```

## Inputs

- `reports/csv/same_base_task_vector_benchmark.csv`
- `reports/csv/same_base_task_vector_summary.csv`
- `reports/csv/same_base_task_vector_candidate_grid.csv`

## Outputs

- `reports/csv/{RUNS_CSV}`
- `reports/csv/{SUMMARY_CSV}`
- `reports/{REPORT_MD}`
- `reports/plots/{PLOT_PDF}`

## Interpretation Boundary

This is a same-base task-vector diagnostic. It separates task-vector interference from rebasin/cycle obstruction. It does not certify Brauer/projective obstruction, and it does not replace independent-seed rebasin diagnostics.

Cycle diagnostics are secondary in this regime. Observed triangle cycle score and synchronization disagreement are {'zero across all rows' if cycle_zero else 'not uniformly zero; inspect secondary rows'}.

Overall mean sign conflict fraction: `{boundary_row.get("mean_sign_conflict_fraction", float("nan")):.4f}`.
Overall mean triangle cycle score: `{boundary_row.get("mean_triangle_cycle_score", float("nan")):.4f}`.
Claim decision: `{boundary_row.get("claim_decision", "not_run")}`.

## Setting Summary

{md_table(settings, ["dataset", "task_preset", "width", "n_tasks", "n_runs", "mean_sign_conflict_fraction", "mean_active_fraction", "mean_pairwise_task_vector_cosine", "mean_min_pairwise_task_vector_cosine", "mean_delta_norm", "mean_greedy_soup_accepted_count", "mean_task_arithmetic_selected_scale", "mean_ties_selected_density", "mean_dare_selected_drop_rate", "mean_task_arithmetic_delta_vs_greedy", "mean_ties_delta_vs_greedy", "mean_dare_delta_vs_greedy", "best_generated_candidate_family_mode"], 40)}

## Best Generated Candidate Families

{md_table(winners, ["dataset", "task_preset", "width", "candidate_family", "win_count", "win_fraction", "n_runs", "claim_boundary"], 80)}

## Strongest Overall Associations

Associations are descriptive. Cycle predictors are included as secondary diagnostics; if they have no variance, no cycle-based association claim is available.

{md_table(associations, ["target", "predictor", "predictor_role", "n_rows", "predictor_unique_values", "pearson", "spearman", "claim_boundary"], 80)}

## Run-Level Diagnostics

{md_table(diagnostics, ["dataset", "task_preset", "width", "seed", "task_vector_sign_conflict_fraction", "task_vector_mean_pairwise_cosine", "task_vector_mean_delta_norm", "greedy_soup_accepted_count", "task_arithmetic_selected_scale", "ties_selected_density", "dare_selected_drop_rate", "best_generated_candidate_family", "task_arithmetic_delta_vs_greedy", "ties_delta_vs_greedy", "dare_delta_vs_greedy"], 30)}

## Claim Boundary

- Same-base task-vector behavior can be analyzed through sign conflict, active fraction, task-vector cosines, delta norms, and validation-selected hyperparameters.
- Triangle cycle score and synchronization disagreement are secondary here because the benchmark intentionally avoids independent initialization/permutation mismatch.
- Positive Task Arithmetic/TIES/DARE deltas in this report are exact-setting, validation-selected same-base findings, not general model-merging superiority claims.
"""
    path.write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    benchmark = pd.read_csv(args.benchmark_csv)
    if not args.summary_csv.exists():
        raise FileNotFoundError(args.summary_csv)
    # Read the summary to make the input dependency explicit. The diagnostics are
    # computed from run-level benchmark and validation-grid rows.
    _same_base_summary = pd.read_csv(args.summary_csv)
    grid = pd.read_csv(args.candidate_grid_csv) if args.candidate_grid_csv.exists() else pd.DataFrame()
    diagnostics = build_diagnostics(benchmark, grid, compute_delta_norms=not args.skip_delta_norms)
    summary = build_summary(diagnostics, args.bootstrap_samples)

    csv_dir = args.reports_dir / "csv"
    plot_dir = args.reports_dir / "plots"
    csv_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    diagnostics.to_csv(csv_dir / RUNS_CSV, index=False, lineterminator="\n")
    summary.to_csv(csv_dir / SUMMARY_CSV, index=False, lineterminator="\n")
    write_plot(diagnostics, plot_dir / PLOT_PDF)
    write_report(args, diagnostics, summary, args.reports_dir / REPORT_MD)
    print(f"wrote {csv_dir / RUNS_CSV}")
    print(f"wrote {csv_dir / SUMMARY_CSV}")
    print(f"wrote {args.reports_dir / REPORT_MD}")
    print(f"wrote {plot_dir / PLOT_PDF}")
    print(f"commit {git_commit()}")


if __name__ == "__main__":
    main()
