#!/usr/bin/env python
"""Validation-safe diagnostic method selection for fixed-setting merge runs."""

from __future__ import annotations

import argparse
import math
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

RUNS_CSV = "fixed_setting_verification_runs.csv"
INDIVIDUALS_CSV = "fixed_setting_individual_models.csv"
PREDICTOR_STATS_CSV = "obstruction_predictor_target_stats.csv"
BARRIER_CSV = "alignment_barrier_targets.csv"
SELECTOR_CSV = "diagnostic_method_selector.csv"
SUMMARY_CSV = "diagnostic_method_selector_summary.csv"
REPORT_MD = "diagnostic_method_selector_report.md"
PLOT_PDF = "diagnostic_method_selector_deltas.pdf"

KEY_COLS = [
    "setting_id",
    "run_id",
    "dataset",
    "architecture",
    "n_models",
    "width",
    "domain_shift",
    "matching",
    "seed",
]

SETTING_COLS = ["dataset", "architecture", "n_models", "width", "domain_shift", "matching"]

DEFAULT_CANDIDATES = (
    "weight_average",
    "git_rebasin_pairwise_ref0",
    "c2m3_synchronized",
    "greedy_soup",
    "c2m3_cluster_branch_ensemble_2",
    "twisted_rank_lift_2",
    "single_best_model",
)

REQUESTED_CANDIDATES = (
    "weight_average",
    "git_rebasin_pairwise_ref0",
    "c2m3_synchronized",
    "greedy_soup",
    "c2m3_cluster_branch_ensemble_2",
    "twisted_rank_lift_2",
    "monomial_gauge",
    "single_best_model",
)

SELECTORS = (
    "greedy_baseline_selector",
    "best_validation_selector",
    "obstruction_rule_selector",
    "logistic_or_tree_selector",
    "conservative_lcb_selector",
)

BARRIER_TARGET_COLS = (
    "linear_mode_connectivity_barrier",
    "c2m3_barrier_delta_vs_git_rebasin",
    "c2m3_barrier_delta_vs_weight_average",
    "monomial_barrier_delta_vs_c2m3",
)

DIAGNOSTIC_FEATURES = (
    "mean_cycle_score",
    "max_cycle_score",
    "nonidentity_triangle_fraction",
    "sync_disagreement",
    "pairwise_alignment_residual_mean",
    "activation_assignment_similarity_mean",
    "combined_obstruction_score",
    "monomial_defect_score",
    *BARRIER_TARGET_COLS,
)

METHOD_PRIORITY = {
    "greedy_soup": 0,
    "single_best_model": 1,
    "c2m3_synchronized": 2,
    "git_rebasin_pairwise_ref0": 3,
    "weight_average": 4,
    "c2m3_cluster_branch_ensemble_2": 5,
    "twisted_rank_lift_2": 6,
}


@dataclass
class LogisticModel:
    weights: np.ndarray
    feature_names: list[str]
    means: np.ndarray
    stds: np.ndarray
    fill_values: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-csv", type=Path, default=ROOT / "reports" / "csv" / RUNS_CSV)
    parser.add_argument("--individuals-csv", type=Path, default=ROOT / "reports" / "csv" / INDIVIDUALS_CSV)
    parser.add_argument("--predictor-stats-csv", type=Path, default=ROOT / "reports" / "csv" / PREDICTOR_STATS_CSV)
    parser.add_argument("--barriers-csv", type=Path, default=ROOT / "reports" / "csv" / BARRIER_CSV)
    parser.add_argument("--monomial-runs-csv", type=Path, default=ROOT / "reports" / "csv" / "monomial_fixed_setting_runs.csv")
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "reports")
    parser.add_argument("--candidate-methods", default=",".join(DEFAULT_CANDIDATES))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--selector-seed", type=int, default=1729)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--lcb-z", type=float, default=1.96)
    parser.add_argument("--rule-margin", type=float, default=0.002)
    parser.add_argument("--logistic-steps", type=int, default=700)
    parser.add_argument("--logistic-lr", type=float, default=0.15)
    parser.add_argument("--logistic-l2", type=float, default=0.01)
    return parser.parse_args()


def parse_csv(text: str) -> list[str]:
    return [item.strip() for item in str(text).split(",") if item.strip()]


def safe_float(value) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y"}


def safe_mean(values: Iterable[float]) -> float:
    arr = np.asarray([safe_float(value) for value in values], dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def bootstrap_mean_ci(values: Iterable[float], n_bootstrap: int, seed: int) -> tuple[float, float]:
    arr = np.asarray([safe_float(value) for value in values], dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0 or int(n_bootstrap) <= 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(int(n_bootstrap), arr.size))
    means = arr[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def setting_id_from_row(row: pd.Series | dict) -> str:
    return (
        f"{row['dataset']}_{row['architecture']}_N{int(row['n_models'])}_"
        f"W{int(row['width'])}_{row['domain_shift']}_{row['matching']}"
    )


def method_priority(method: str) -> int:
    return METHOD_PRIORITY.get(str(method), 100)


def method_is_monomial(method: str) -> bool:
    return str(method).startswith("monomial_gauge")


def md_table(df: pd.DataFrame, cols: list[str], max_rows: int = 40) -> str:
    if df.empty:
        return "_None._"
    view = df[[col for col in cols if col in df.columns]].head(max_rows).copy()
    integer_cols = {
        "n_rows",
        "n_unique_seeds",
        "n_selected",
        "count",
        "n_challenger_selections",
        "false_challenger_count",
        "n_pairs",
        "fold",
        "n_models",
        "width",
    }
    for col in view.columns:
        if pd.api.types.is_bool_dtype(view[col]):
            view[col] = view[col].map(lambda value: "true" if bool(value) else "false")
        elif col in integer_cols and pd.api.types.is_numeric_dtype(view[col]):
            view[col] = view[col].map(lambda value: "" if pd.isna(value) else str(int(round(float(value)))))
        elif pd.api.types.is_numeric_dtype(view[col]):
            precision = 6 if any(token in col for token in ("delta", "ci_low", "ci_high")) else 4
            view[col] = view[col].map(lambda value: "" if pd.isna(value) else f"{float(value):.{precision}f}")
        else:
            view[col] = view[col].fillna("").astype(str)
    header = "| " + " | ".join(view.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(view.columns)) + " |"
    rows = ["| " + " | ".join(str(value) for value in row) + " |" for row in view.to_numpy()]
    suffix = f"\n\n_Showing {max_rows} of {len(df)} rows._" if len(df) > max_rows else ""
    return "\n".join([header, sep, *rows]) + suffix


def observed_run_rows(runs: pd.DataFrame) -> pd.DataFrame:
    out = runs[
        (runs["alignment_source"].astype(str) == "observed")
        & (pd.to_numeric(runs["alignment_noise_fraction"], errors="coerce") == 0.0)
    ].copy()
    return out


def merge_barrier_features(candidates: pd.DataFrame, barriers_csv: Path) -> pd.DataFrame:
    if not barriers_csv.exists():
        return candidates
    barriers = pd.read_csv(barriers_csv)
    key_cols = [col for col in ("setting_id", "run_id", "seed") if col in barriers.columns and col in candidates.columns]
    if not key_cols:
        return candidates
    if "status" in barriers.columns:
        barriers = barriers[barriers["status"].astype(str) == "ok"].copy()
    target_cols = [col for col in BARRIER_TARGET_COLS if col in barriers.columns]
    pieces = []
    if target_cols:
        pieces.append(barriers[key_cols + target_cols].drop_duplicates(subset=key_cols, keep="first"))
    if "val_max_loss_barrier" in barriers.columns and "method" in barriers.columns:
        pivot = barriers.pivot_table(index=key_cols, columns="method", values="val_max_loss_barrier", aggfunc="first")
        pivot = pivot.rename(columns={col: f"val_max_loss_barrier_{col}" for col in pivot.columns}).reset_index()
        pieces.append(pivot)
    if not pieces:
        return candidates
    merged = pieces[0]
    for piece in pieces[1:]:
        merged = merged.merge(piece, on=key_cols, how="outer")
    existing = [col for col in merged.columns if col in candidates.columns and col not in key_cols]
    if existing:
        candidates = candidates.drop(columns=existing)
    return candidates.merge(merged, on=key_cols, how="left")


def validation_best_individuals(individuals: pd.DataFrame, run_ids: set[str]) -> pd.DataFrame:
    if individuals.empty:
        return pd.DataFrame()
    ind = individuals[individuals["run_id"].astype(str).isin(run_ids)].copy()
    if ind.empty:
        return ind
    ind["_val_acc"] = pd.to_numeric(ind["val_accuracy"], errors="coerce")
    ind["_val_loss"] = pd.to_numeric(ind["val_loss"], errors="coerce")
    ind = ind.sort_values(["run_id", "_val_acc", "_val_loss", "model_index"], ascending=[True, False, True, True])
    return ind.groupby("run_id", as_index=False, sort=False).head(1).copy()


def make_single_best_rows(individuals: pd.DataFrame, base_rows: pd.DataFrame) -> pd.DataFrame:
    best = validation_best_individuals(individuals, set(base_rows["run_id"].astype(str)))
    if best.empty:
        return pd.DataFrame()
    meta_cols = [col for col in base_rows.columns if col not in {"method", "test_accuracy", "test_loss", "val_accuracy", "val_loss"}]
    meta = base_rows[meta_cols].drop_duplicates(subset=["run_id"], keep="first")
    out = meta.merge(
        best[["run_id", "model_index", "val_accuracy", "val_loss", "test_accuracy", "test_loss"]],
        on="run_id",
        how="inner",
    )
    out["method"] = "single_best_model"
    out["validation_accuracy_used_for_selection"] = out["val_accuracy"]
    out["selection_indices"] = out["model_index"].map(lambda value: str(int(value)) if pd.notna(value) else "")
    out["uses_validation_data"] = True
    out["is_single_model"] = True
    out["exact_relu_symmetry"] = False
    out["is_soup"] = False
    out["is_ensemble_or_extra_capacity"] = False
    out["capacity_matched_to_weight_average"] = True
    out["capacity_matched_to_rank_lift"] = False
    out["branch_count"] = 1
    out["parameter_count_multiplier"] = 1.0
    out["inference_time_multiplier"] = 1.0
    out["parameter_multiplier"] = 1.0
    out["inference_multiplier"] = 1.0
    out["method_note"] = "validation-best local model; no-merge fallback"
    return out


def load_monomial_candidate_rows(args: argparse.Namespace, base_run_ids: set[str]) -> tuple[pd.DataFrame, str]:
    if not args.monomial_runs_csv.exists():
        return pd.DataFrame(), "not_found"
    mono = pd.read_csv(args.monomial_runs_csv)
    if "run_id" not in mono or "method" not in mono:
        return pd.DataFrame(), "malformed"
    mono = observed_run_rows(mono) if "alignment_source" in mono and "alignment_noise_fraction" in mono else mono.copy()
    mono = mono[mono["method"].astype(str).map(method_is_monomial)].copy()
    matched = mono[mono["run_id"].astype(str).isin(base_run_ids)].copy()
    if matched.empty:
        return pd.DataFrame(), "present_but_no_run_id_overlap_with_quality_gated_inputs"
    return matched, "included_matching_run_ids"


def load_candidates(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    runs = pd.read_csv(args.runs_csv)
    individuals = pd.read_csv(args.individuals_csv)
    requested = parse_csv(args.candidate_methods)
    observed = observed_run_rows(runs)
    base_rows = observed[observed["method"].astype(str) == "weight_average"].copy()
    method_rows = observed[observed["method"].astype(str).isin([m for m in requested if m != "single_best_model"])].copy()
    single = make_single_best_rows(individuals, base_rows) if "single_best_model" in requested else pd.DataFrame()
    candidates = pd.concat([method_rows, single], ignore_index=True, sort=False)
    mono_rows, mono_status = load_monomial_candidate_rows(args, set(base_rows["run_id"].astype(str)))
    if not mono_rows.empty:
        candidates = pd.concat([candidates, mono_rows], ignore_index=True, sort=False)
    candidates = merge_barrier_features(candidates, args.barriers_csv)
    candidates["setting_id"] = candidates.apply(setting_id_from_row, axis=1)
    candidates["selector_fold"] = assign_folds(candidates, args.folds)
    candidates["method_priority"] = candidates["method"].astype(str).map(method_priority)
    candidates["candidate_is_no_merge"] = candidates["method"].astype(str) == "single_best_model"
    candidates["candidate_is_extra_capacity"] = candidates["is_ensemble_or_extra_capacity"].map(to_bool)
    candidates["candidate_is_single_model"] = candidates["is_single_model"].map(to_bool)
    candidates["candidate_is_soup"] = candidates["is_soup"].map(to_bool)
    candidates["candidate_exact_relu_symmetry"] = candidates["exact_relu_symmetry"].map(to_bool)

    availability_rows = []
    for method in REQUESTED_CANDIDATES:
        if method == "monomial_gauge":
            count = int(candidates["method"].astype(str).map(method_is_monomial).sum())
            availability_rows.append(
                {
                    "summary_type": "candidate_availability",
                    "method": method,
                    "included": count > 0,
                    "n_rows": count,
                    "reason": mono_status,
                }
            )
            continue
        count = int((candidates["method"].astype(str) == method).sum())
        availability_rows.append(
            {
                "summary_type": "candidate_availability",
                "method": method,
                "included": count > 0,
                "n_rows": count,
                "reason": "included" if count > 0 else "not_available_in_inputs",
            }
        )
    return candidates, pd.DataFrame(availability_rows), runs


def assign_folds(candidates: pd.DataFrame, folds: int) -> pd.Series:
    folds = max(2, int(folds))
    fold_by_run: dict[str, int] = {}
    base = candidates[["setting_id", "seed", "run_id"]].drop_duplicates().copy()
    for _setting, group in base.groupby("setting_id", sort=True):
        seeds = sorted(int(seed) for seed in group["seed"].dropna().unique())
        for idx, seed in enumerate(seeds):
            run_ids = group[group["seed"].astype(int) == seed]["run_id"].astype(str)
            for run_id in run_ids:
                fold_by_run[run_id] = idx % folds
    return candidates["run_id"].astype(str).map(fold_by_run).fillna(0).astype(int)


def validation_sample_count(row: pd.Series) -> int:
    n = safe_float(row.get("max_train_samples", np.nan))
    frac = safe_float(row.get("val_fraction", np.nan))
    if not math.isfinite(n) or not math.isfinite(frac):
        return 1
    return max(1, int(round(n * frac)))


def candidate_sort_frame(frame: pd.DataFrame, score_col: str = "val_accuracy") -> pd.DataFrame:
    out = frame.copy()
    out["_score"] = pd.to_numeric(out[score_col], errors="coerce").fillna(-np.inf)
    out["_loss"] = pd.to_numeric(out["val_loss"], errors="coerce").fillna(np.inf)
    out["_extra"] = out["candidate_is_extra_capacity"].astype(int)
    out["_priority"] = out["method"].astype(str).map(method_priority)
    return out.sort_values(["_score", "_loss", "_extra", "_priority"], ascending=[False, True, True, True])


def best_validation_row(frame: pd.DataFrame) -> pd.Series:
    return candidate_sort_frame(frame).iloc[0]


def method_row(frame: pd.DataFrame, method: str) -> pd.Series | None:
    rows = frame[frame["method"].astype(str) == method]
    if rows.empty:
        return None
    return rows.iloc[0]


def lcb_delta(candidate: pd.Series, baseline: pd.Series, z: float) -> float:
    p1 = min(max(safe_float(candidate.get("val_accuracy")), 0.0), 1.0)
    p0 = min(max(safe_float(baseline.get("val_accuracy")), 0.0), 1.0)
    n = min(validation_sample_count(candidate), validation_sample_count(baseline))
    se = math.sqrt(max(p1 * (1.0 - p1), 0.0) / n + max(p0 * (1.0 - p0), 0.0) / n)
    return float((p1 - p0) - float(z) * se)


def obstruction_rule_select(frame: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.Series, str]:
    greedy = method_row(frame, "greedy_soup")
    if greedy is None:
        return best_validation_row(frame), "greedy_missing_use_best_validation"
    single = method_row(frame, "single_best_model")
    if single is not None and safe_float(single["val_accuracy"]) >= safe_float(greedy["val_accuracy"]) + float(args.rule_margin):
        return single, "validation_best_single_beats_greedy_margin"

    diagnostics = greedy
    cycle = safe_float(diagnostics.get("mean_cycle_score"))
    sync = safe_float(diagnostics.get("sync_disagreement"))
    residual = safe_float(diagnostics.get("pairwise_alignment_residual_mean"))
    c2m3_delta_weight_barrier = safe_float(diagnostics.get("c2m3_barrier_delta_vs_weight_average"))
    c2m3 = method_row(frame, "c2m3_synchronized")
    if c2m3 is not None:
        low_obstruction = (
            (math.isfinite(cycle) and cycle <= 0.903)
            or (math.isfinite(sync) and sync <= 0.19)
            or (math.isfinite(residual) and residual <= 0.29)
        )
        low_barrier_for_c2m3 = math.isfinite(c2m3_delta_weight_barrier) and c2m3_delta_weight_barrier > 0.05
        if (low_obstruction or low_barrier_for_c2m3) and safe_float(c2m3["val_accuracy"]) >= safe_float(greedy["val_accuracy"]) - float(args.rule_margin):
            return c2m3, "low_obstruction_or_low_barrier_c2m3_within_validation_margin"

    branch_pool = frame[frame["method"].astype(str).isin(["c2m3_cluster_branch_ensemble_2", "twisted_rank_lift_2"])].copy()
    if not branch_pool.empty:
        branch_best = best_validation_row(branch_pool)
        high_obstruction = (
            (math.isfinite(cycle) and cycle >= 0.906)
            or (math.isfinite(sync) and sync >= 0.22)
            or (math.isfinite(residual) and residual >= 0.33)
        )
        if high_obstruction and safe_float(branch_best["val_accuracy"]) >= safe_float(greedy["val_accuracy"]) + float(args.rule_margin):
            return branch_best, "high_obstruction_branch_beats_greedy_validation_margin"

    return greedy, "default_greedy_boundary"


def conservative_lcb_select(frame: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.Series, str]:
    greedy = method_row(frame, "greedy_soup")
    if greedy is None:
        return best_validation_row(frame), "greedy_missing_use_best_validation"
    challengers = frame[frame["method"].astype(str) != "greedy_soup"].copy()
    if challengers.empty:
        return greedy, "no_challengers"
    rows = []
    for row in challengers.itertuples(index=False):
        row_series = pd.Series(row._asdict())
        rows.append((lcb_delta(row_series, greedy, float(args.lcb_z)), row_series))
    rows.sort(key=lambda item: (item[0], safe_float(item[1].get("val_accuracy")), -method_priority(str(item[1].get("method")))), reverse=True)
    best_lcb, best_row = rows[0]
    if math.isfinite(best_lcb) and best_lcb > 0.0:
        return best_row, f"positive_validation_lcb={best_lcb:.6f}"
    return greedy, "no_positive_validation_lcb"


def numeric_feature(row: pd.Series, name: str) -> float:
    if name == "candidate_val_delta_vs_greedy":
        return safe_float(row.get("candidate_val_delta_vs_greedy"))
    if name == "candidate_loss_delta_vs_greedy":
        return safe_float(row.get("candidate_loss_delta_vs_greedy"))
    if name == "candidate_is_extra_capacity":
        return float(bool(row.get("candidate_is_extra_capacity", False)))
    if name == "candidate_is_single_model":
        return float(bool(row.get("candidate_is_single_model", False)))
    if name == "candidate_is_no_merge":
        return float(bool(row.get("candidate_is_no_merge", False)))
    if name == "candidate_parameter_count_multiplier":
        return safe_float(row.get("parameter_count_multiplier"))
    if name == "candidate_inference_time_multiplier":
        return safe_float(row.get("inference_time_multiplier"))
    return safe_float(row.get(name))


def enrich_candidate_features(candidates: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _run_id, group in candidates.groupby("run_id", sort=False):
        greedy = method_row(group, "greedy_soup")
        greedy_val = safe_float(greedy.get("val_accuracy")) if greedy is not None else float("nan")
        greedy_loss = safe_float(greedy.get("val_loss")) if greedy is not None else float("nan")
        best = best_validation_row(group)
        for row in group.itertuples(index=False):
            item = row._asdict()
            item["candidate_val_delta_vs_greedy"] = safe_float(item.get("val_accuracy")) - greedy_val
            item["candidate_loss_delta_vs_greedy"] = greedy_loss - safe_float(item.get("val_loss"))
            item["validation_best_method_label"] = str(best["method"])
            item["is_validation_best_method"] = str(item.get("method")) == str(best["method"])
            rows.append(item)
    return pd.DataFrame(rows)


def logistic_feature_names(methods: list[str]) -> list[str]:
    base = [
        "candidate_val_delta_vs_greedy",
        "candidate_loss_delta_vs_greedy",
        "candidate_is_extra_capacity",
        "candidate_is_single_model",
        "candidate_is_no_merge",
        "candidate_parameter_count_multiplier",
        "candidate_inference_time_multiplier",
        *DIAGNOSTIC_FEATURES,
    ]
    return base + [f"method_is_{method}" for method in methods]


def feature_matrix(rows: pd.DataFrame, feature_names: list[str], fit: bool, model: LogisticModel | None = None) -> tuple[np.ndarray, LogisticModel | None]:
    values = []
    for row in rows.itertuples(index=False):
        item = pd.Series(row._asdict())
        method = str(item.get("method"))
        values.append(
            [
                1.0 if name == f"method_is_{method}" else numeric_feature(item, name)
                for name in feature_names
            ]
        )
    x = np.asarray(values, dtype=float)
    if fit:
        finite = np.isfinite(x)
        fill = np.zeros(x.shape[1], dtype=float)
        for col_idx in range(x.shape[1]):
            if finite[:, col_idx].any():
                fill[col_idx] = float(np.median(x[finite[:, col_idx], col_idx]))
        x = np.where(np.isfinite(x), x, fill)
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        std = np.where(std > 1e-12, std, 1.0)
        x = (x - mean) / std
        x = np.column_stack([np.ones(len(x)), x])
        return x, LogisticModel(weights=np.zeros(x.shape[1]), feature_names=feature_names, means=mean, stds=std, fill_values=fill)
    if model is None:
        raise ValueError("model required when fit=False")
    x = np.where(np.isfinite(x), x, model.fill_values)
    x = (x - model.means) / model.stds
    x = np.column_stack([np.ones(len(x)), x])
    return x, None


def train_logistic_selector(train_rows: pd.DataFrame, args: argparse.Namespace) -> LogisticModel | None:
    if train_rows.empty or train_rows["run_id"].nunique() < 4:
        return None
    y = train_rows["is_validation_best_method"].astype(float).to_numpy()
    if float(y.sum()) <= 0.0 or float(y.sum()) >= float(len(y)):
        return None
    methods = sorted(str(method) for method in train_rows["method"].dropna().unique())
    feature_names = logistic_feature_names(methods)
    x, model = feature_matrix(train_rows, feature_names, fit=True)
    assert model is not None
    weights = model.weights.copy()
    lr = float(args.logistic_lr)
    l2 = float(args.logistic_l2)
    for _ in range(max(1, int(args.logistic_steps))):
        logits = np.clip(x @ weights, -40.0, 40.0)
        pred = 1.0 / (1.0 + np.exp(-logits))
        grad = (x.T @ (pred - y)) / max(len(y), 1)
        grad[1:] += l2 * weights[1:]
        weights -= lr * grad
    model.weights = weights
    return model


def logistic_select(frame: pd.DataFrame, model: LogisticModel | None) -> tuple[pd.Series, str]:
    if model is None:
        greedy = method_row(frame, "greedy_soup")
        if greedy is not None:
            return greedy, "insufficient_training_default_greedy"
        return best_validation_row(frame), "insufficient_training_greedy_missing_use_best_validation"
    eval_rows = frame.copy()
    usable_methods = {name.removeprefix("method_is_") for name in model.feature_names if name.startswith("method_is_")}
    eval_rows = eval_rows[eval_rows["method"].astype(str).isin(usable_methods)].copy()
    if eval_rows.empty:
        return best_validation_row(frame), "no_methods_seen_in_training_use_best_validation"
    x, _ = feature_matrix(eval_rows, model.feature_names, fit=False, model=model)
    scores = 1.0 / (1.0 + np.exp(-np.clip(x @ model.weights, -40.0, 40.0)))
    eval_rows = eval_rows.assign(_selector_score=scores)
    eval_rows["_val_acc"] = pd.to_numeric(eval_rows["val_accuracy"], errors="coerce").fillna(-np.inf)
    eval_rows["_priority"] = eval_rows["method"].astype(str).map(method_priority)
    selected = eval_rows.sort_values(["_selector_score", "_val_acc", "_priority"], ascending=[False, False, True]).iloc[0]
    return selected, f"logistic_score={safe_float(selected['_selector_score']):.6f}"


def evaluate_selectors(candidates: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    candidates = enrich_candidate_features(candidates)
    results = []
    base_runs = candidates[["setting_id", "run_id", *SETTING_COLS, "seed", "selector_fold"]].drop_duplicates().copy()
    grouped = {run_id: group.copy() for run_id, group in candidates.groupby("run_id", sort=False)}
    for setting_id, setting_runs in base_runs.groupby("setting_id", sort=True):
        setting_candidates = candidates[candidates["setting_id"].astype(str) == str(setting_id)].copy()
        for fold in sorted(setting_runs["selector_fold"].unique()):
            eval_run_ids = set(setting_runs[setting_runs["selector_fold"].astype(int) == int(fold)]["run_id"].astype(str))
            train_rows = setting_candidates[~setting_candidates["run_id"].astype(str).isin(eval_run_ids)].copy()
            logistic_model = train_logistic_selector(train_rows, args)
            for run_id in sorted(eval_run_ids):
                frame = grouped[run_id]
                greedy = method_row(frame, "greedy_soup")
                single = method_row(frame, "single_best_model")
                if greedy is None or single is None:
                    continue
                selections = {
                    "greedy_baseline_selector": (greedy, "fixed_greedy_boundary"),
                    "best_validation_selector": (best_validation_row(frame), "best_validation_accuracy_no_test"),
                    "obstruction_rule_selector": obstruction_rule_select(frame, args),
                    "logistic_or_tree_selector": logistic_select(frame, logistic_model),
                    "conservative_lcb_selector": conservative_lcb_select(frame, args),
                }
                for selector_name, (selected, reason) in selections.items():
                    row = selected.to_dict()
                    selected_test = safe_float(row.get("test_accuracy"))
                    greedy_test = safe_float(greedy.get("test_accuracy"))
                    single_test = safe_float(single.get("test_accuracy"))
                    selected_loss = safe_float(row.get("test_loss"))
                    greedy_loss = safe_float(greedy.get("test_loss"))
                    single_loss = safe_float(single.get("test_loss"))
                    selected_method = str(row.get("method"))
                    challenger = selected_method != "greedy_soup"
                    results.append(
                        {
                            **{col: row.get(col) for col in KEY_COLS if col in row},
                            "selector": selector_name,
                            "selector_fold": int(fold),
                            "selected_method": selected_method,
                            "selection_reason": reason,
                            "selected_val_accuracy": safe_float(row.get("val_accuracy")),
                            "selected_val_loss": safe_float(row.get("val_loss")),
                            "selected_test_accuracy": selected_test,
                            "selected_test_loss": selected_loss,
                            "greedy_val_accuracy": safe_float(greedy.get("val_accuracy")),
                            "greedy_test_accuracy": greedy_test,
                            "greedy_test_loss": greedy_loss,
                            "single_best_val_accuracy": safe_float(single.get("val_accuracy")),
                            "single_best_test_accuracy": single_test,
                            "single_best_test_loss": single_loss,
                            "paired_delta_vs_greedy_soup": selected_test - greedy_test,
                            "paired_delta_vs_single_best_model": selected_test - single_test,
                            "paired_loss_delta_vs_greedy_soup": selected_loss - greedy_loss,
                            "paired_loss_delta_vs_single_best_model": selected_loss - single_loss,
                            "selected_is_single_model": bool(row.get("candidate_is_single_model", False)),
                            "selected_is_extra_capacity": bool(row.get("candidate_is_extra_capacity", False)),
                            "selected_is_no_merge": selected_method == "single_best_model",
                            "selected_is_soup": bool(row.get("candidate_is_soup", False)),
                            "selected_exact_relu_symmetry": bool(row.get("candidate_exact_relu_symmetry", False)),
                            "selected_parameter_count_multiplier": safe_float(row.get("parameter_count_multiplier")),
                            "selected_inference_time_multiplier": safe_float(row.get("inference_time_multiplier")),
                            "false_challenger": bool(challenger and selected_test < greedy_test),
                            "is_challenger_selection": bool(challenger),
                            "validation_safe": True,
                            "test_used_for_selection": False,
                            **{col: row.get(col, np.nan) for col in DIAGNOSTIC_FEATURES if col in row},
                        }
                    )
    return pd.DataFrame(results)


def summarize_selector_group(group: pd.DataFrame, scope: str, selector: str, n_bootstrap: int, seed_offset: int) -> dict:
    delta_g = pd.to_numeric(group["paired_delta_vs_greedy_soup"], errors="coerce")
    delta_s = pd.to_numeric(group["paired_delta_vs_single_best_model"], errors="coerce")
    low_g, high_g = bootstrap_mean_ci(delta_g, n_bootstrap, seed=991 + seed_offset)
    low_s, high_s = bootstrap_mean_ci(delta_s, n_bootstrap, seed=1991 + seed_offset)
    challenger = group["is_challenger_selection"].astype(bool)
    false_challenger = group["false_challenger"].astype(bool)
    extra_rate = float(group["selected_is_extra_capacity"].astype(bool).mean()) if len(group) else float("nan")
    mean_delta_g = safe_mean(delta_g)
    if math.isfinite(low_g) and low_g > 0.0 and extra_rate <= 1e-12:
        decision = "supported_capacity_matched_selector_beats_greedy"
    elif math.isfinite(low_g) and low_g > 0.0 and extra_rate > 0.0:
        decision = "supported_only_with_extra_capacity_or_mixed_capacity"
    elif math.isfinite(mean_delta_g) and mean_delta_g > 0.0:
        decision = "descriptive_positive_mean_ci_crosses_zero"
    else:
        decision = "negative_does_not_beat_greedy"
    return {
        "summary_type": "selector_overall" if scope == "overall" else "selector_fixed_setting",
        "scope": scope,
        "selector": selector,
        "n_rows": int(len(group)),
        "n_unique_seeds": int(group["seed"].nunique()) if "seed" in group else 0,
        "mean_test_accuracy": safe_mean(group["selected_test_accuracy"]),
        "mean_test_loss": safe_mean(group["selected_test_loss"]),
        "paired_mean_accuracy_delta_vs_greedy_soup": mean_delta_g,
        "paired_accuracy_delta_vs_greedy_soup_ci_low": low_g,
        "paired_accuracy_delta_vs_greedy_soup_ci_high": high_g,
        "paired_mean_accuracy_delta_vs_single_best_model": safe_mean(delta_s),
        "paired_accuracy_delta_vs_single_best_model_ci_low": low_s,
        "paired_accuracy_delta_vs_single_best_model_ci_high": high_s,
        "false_challenger_count": int(false_challenger.sum()),
        "n_challenger_selections": int(challenger.sum()),
        "false_challenger_rate": float(false_challenger.sum() / max(challenger.sum(), 1)),
        "false_challenger_rate_all_rows": float(false_challenger.mean()) if len(group) else float("nan"),
        "no_merge_selection_rate": float(group["selected_is_no_merge"].astype(bool).mean()) if len(group) else float("nan"),
        "extra_capacity_selection_rate": extra_rate,
        "single_model_selection_rate": float(group["selected_is_single_model"].astype(bool).mean()) if len(group) else float("nan"),
        "greedy_selection_rate": float((group["selected_method"].astype(str) == "greedy_soup").mean()) if len(group) else float("nan"),
        "claim_decision": decision,
    }


def build_summary(results: pd.DataFrame, availability: pd.DataFrame, predictor_stats: pd.DataFrame, n_bootstrap: int) -> pd.DataFrame:
    rows: list[dict] = []
    rows.extend(availability.to_dict("records"))
    supported = predictor_stats[predictor_stats.get("claim_supported", False) == True].copy() if not predictor_stats.empty else pd.DataFrame()  # noqa: E712
    for item in supported.head(50).to_dict("records"):
        rows.append(
            {
                "summary_type": "prior_supported_diagnostic",
                "scope": "predictor_target_stats",
                "target": item.get("target"),
                "predictor": item.get("predictor"),
                "claim_status": item.get("claim_status"),
                "support_scope": item.get("support_scope"),
                "note": "audited only; not used as selector-training labels",
            }
        )
    for idx, (selector, group) in enumerate(results.groupby("selector", sort=True)):
        rows.append(summarize_selector_group(group, "overall", selector, n_bootstrap, seed_offset=idx * 1000))
    setting_cols = ["dataset", "architecture", "n_models", "width", "domain_shift", "matching", "selector"]
    for idx, (key, group) in enumerate(results.groupby(setting_cols, sort=True)):
        meta = dict(zip(setting_cols, key))
        scope = (
            f"{meta['dataset']}_{meta['architecture']}_N{int(meta['n_models'])}_"
            f"W{int(meta['width'])}_{meta['domain_shift']}_{meta['matching']}"
        )
        row = summarize_selector_group(group, scope, str(meta["selector"]), n_bootstrap, seed_offset=10000 + idx * 1000)
        row.update({col: meta[col] for col in setting_cols if col != "selector"})
        rows.append(row)
    for (selector, method), group in results.groupby(["selector", "selected_method"], sort=True):
        rows.append(
            {
                "summary_type": "selected_method_frequency",
                "scope": "overall",
                "selector": selector,
                "selected_method": method,
                "n_selected": int(len(group)),
                "selection_fraction": float(len(group) / max((results["selector"] == selector).sum(), 1)),
                "mean_test_accuracy": safe_mean(group["selected_test_accuracy"]),
                "extra_capacity": bool(group["selected_is_extra_capacity"].astype(bool).any()),
                "no_merge": method == "single_best_model",
            }
        )
    return pd.DataFrame(rows)


def load_predictor_stats(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def write_plot(results: pd.DataFrame, summary: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    overall = summary[summary["summary_type"].astype(str) == "selector_overall"].copy()
    fig, ax = plt.subplots(figsize=(10.8, 5.8))
    if overall.empty:
        ax.text(0.5, 0.5, "No selector rows", ha="center", va="center")
        ax.set_axis_off()
    else:
        overall = overall.sort_values("selector")
        x = np.arange(len(overall))
        width = 0.36
        means_g = pd.to_numeric(overall["paired_mean_accuracy_delta_vs_greedy_soup"], errors="coerce").to_numpy()
        lows_g = pd.to_numeric(overall["paired_accuracy_delta_vs_greedy_soup_ci_low"], errors="coerce").to_numpy()
        highs_g = pd.to_numeric(overall["paired_accuracy_delta_vs_greedy_soup_ci_high"], errors="coerce").to_numpy()
        means_s = pd.to_numeric(overall["paired_mean_accuracy_delta_vs_single_best_model"], errors="coerce").to_numpy()
        lows_s = pd.to_numeric(overall["paired_accuracy_delta_vs_single_best_model_ci_low"], errors="coerce").to_numpy()
        highs_s = pd.to_numeric(overall["paired_accuracy_delta_vs_single_best_model_ci_high"], errors="coerce").to_numpy()
        err_g = np.vstack([np.maximum(means_g - lows_g, 0.0), np.maximum(highs_g - means_g, 0.0)])
        err_s = np.vstack([np.maximum(means_s - lows_s, 0.0), np.maximum(highs_s - means_s, 0.0)])
        ax.bar(x - width / 2, means_g, width, yerr=err_g, label="vs greedy_soup", color="#5078a8", capsize=3)
        ax.bar(x + width / 2, means_s, width, yerr=err_s, label="vs single_best_model", color="#c26d3d", capsize=3)
        ax.axhline(0.0, color="black", linewidth=0.8)
        labels = [str(selector).replace("_selector", "").replace("_", "\n") for selector in overall["selector"]]
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("paired test accuracy delta")
        ax.set_title("Validation-safe method selector deltas")
        ax.legend(frameon=False)
        ax.text(
            0.0,
            -0.22,
            "Selectors use validation metrics/diagnostics only; test accuracy is read after selection is frozen.",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_report(args: argparse.Namespace, results: pd.DataFrame, summary: pd.DataFrame, availability: pd.DataFrame, predictor_stats: pd.DataFrame) -> None:
    report_path = args.reports_dir / REPORT_MD
    overall = summary[summary["summary_type"].astype(str) == "selector_overall"].copy()
    fixed = summary[summary["summary_type"].astype(str) == "selector_fixed_setting"].copy()
    freq = summary[summary["summary_type"].astype(str) == "selected_method_frequency"].copy()
    supported = predictor_stats[predictor_stats.get("claim_supported", False) == True].copy() if not predictor_stats.empty else pd.DataFrame()  # noqa: E712
    positive = overall[pd.to_numeric(overall["paired_accuracy_delta_vs_greedy_soup_ci_low"], errors="coerce") > 0.0].copy()
    if positive.empty:
        conclusion = "No selector beats greedy soup with a positive paired bootstrap lower bound; this is a negative selector result."
    elif (pd.to_numeric(positive["extra_capacity_selection_rate"], errors="coerce") > 0.0).any():
        conclusion = "A selector has positive paired evidence only with mixed or extra-capacity selections; label this as extra capacity, not a same-capacity merge win."
    else:
        conclusion = "A capacity-matched selector has positive paired evidence over greedy soup in this offline analysis."

    command = " ".join(sys.argv)
    report = f"""# Diagnostic Method Selector

Generated by `experiments/diagnostic_method_selector.py`.

## Exact Command

```bash
{command}
```

## Scope And Leakage Control

- Inputs are completed fixed-setting CSVs; no model is retrained here.
- Only observed alignment rows are used for primary selector evaluation; injected-noise rows are excluded.
- The learned `logistic_or_tree_selector` is trained with seed-fold cross-validation inside each fixed setting. Training labels are validation-best methods in selector-train folds only.
- `best_validation_selector`, `obstruction_rule_selector`, and `conservative_lcb_selector` use validation metrics and diagnostics from the evaluated seed but never inspect test metrics.
- Test accuracy and test loss are joined only after each selector has chosen a method.
- The original validation examples are not resplit because the saved inputs expose aggregate validation metrics rather than per-example validation predictions; this artifact performs nested seed-fold validation over saved validation metrics.
- Greedy soup remains the primary boundary baseline.
- Extra-capacity branch selections are labeled with `selected_is_extra_capacity=true`.

## Inputs

- `{args.runs_csv}`
- `{args.individuals_csv}`
- `{args.predictor_stats_csv}`
- `{args.barriers_csv}`
- `{args.monomial_runs_csv}`

## Candidate Availability

{md_table(availability, ["method", "included", "n_rows", "reason"], 20)}

## Overall Selector Results

{md_table(overall, ["selector", "n_rows", "mean_test_accuracy", "paired_mean_accuracy_delta_vs_greedy_soup", "paired_accuracy_delta_vs_greedy_soup_ci_low", "paired_accuracy_delta_vs_greedy_soup_ci_high", "paired_mean_accuracy_delta_vs_single_best_model", "paired_accuracy_delta_vs_single_best_model_ci_low", "paired_accuracy_delta_vs_single_best_model_ci_high", "false_challenger_rate", "no_merge_selection_rate", "extra_capacity_selection_rate", "claim_decision"], 20)}

## Selected Method Frequency

{md_table(freq, ["selector", "selected_method", "n_selected", "selection_fraction", "mean_test_accuracy", "extra_capacity", "no_merge"], 80)}

## Fixed-Setting Results

{md_table(fixed, ["scope", "selector", "n_rows", "mean_test_accuracy", "paired_mean_accuracy_delta_vs_greedy_soup", "paired_accuracy_delta_vs_greedy_soup_ci_low", "paired_accuracy_delta_vs_greedy_soup_ci_high", "false_challenger_rate", "no_merge_selection_rate", "extra_capacity_selection_rate", "claim_decision"], 80)}

## Prior Diagnostic Support Audit

The predictor-target diagnostic CSV is read for audit context only. Its supported rows are not used as selector-training labels, since some diagnostic targets are test-side outcomes.

{md_table(supported, ["dataset", "n_models", "width", "domain_shift", "target", "predictor", "claim_status", "support_scope"], 20)}

## Claim Boundary

{conclusion}

This report supports at most a validation-safe selector diagnostic. It does not show a general model-merging improvement, and it does not claim that diagnostics predict raw weight-average degradation.

## Output Files

- `reports/csv/{SELECTOR_CSV}`
- `reports/csv/{SUMMARY_CSV}`
- `reports/{REPORT_MD}`
- `reports/plots/{PLOT_PDF}`

## Environment

- Platform: {platform.platform()}
- Bootstrap samples: {args.bootstrap_samples}
- Selector folds: {args.folds}
- LCB z-value: {args.lcb_z}
"""
    report_path.write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    (args.reports_dir / "csv").mkdir(parents=True, exist_ok=True)
    (args.reports_dir / "plots").mkdir(parents=True, exist_ok=True)
    candidates, availability, _runs = load_candidates(args)
    predictor_stats = load_predictor_stats(args.predictor_stats_csv)
    results = evaluate_selectors(candidates, args)
    summary = build_summary(results, availability, predictor_stats, args.bootstrap_samples)

    selector_path = args.reports_dir / "csv" / SELECTOR_CSV
    summary_path = args.reports_dir / "csv" / SUMMARY_CSV
    plot_path = args.reports_dir / "plots" / PLOT_PDF
    results.to_csv(selector_path, index=False, lineterminator="\n")
    summary.to_csv(summary_path, index=False, lineterminator="\n")
    write_plot(results, summary, plot_path)
    write_report(args, results, summary, availability, predictor_stats)

    print(f"wrote {selector_path}")
    print(f"wrote {summary_path}")
    print(f"wrote {args.reports_dir / REPORT_MD}")
    print(f"wrote {plot_path}")


if __name__ == "__main__":
    main()
