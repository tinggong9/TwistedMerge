#!/usr/bin/env python
"""Diagnose greedy-soup acceptance using obstruction and validation features.

This is deliberately not a soup-beating experiment.  It treats greedy soup as
the empirical validation-descent baseline and asks whether obstruction
diagnostics help explain which locally trained candidates are accepted by the
saved greedy-soup trajectory.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model_merging_benchmark import format_markdown_table  # noqa: E402


SETTING_COLS = ["dataset", "architecture", "n_models", "width", "domain_shift", "matching"]
RUN_KEY_COLS = ["setting_id", "run_id", *SETTING_COLS, "seed"]
DIAGNOSTICS_CSV = "soup_acceptance_obstruction_diagnostics.csv"
SUMMARY_CSV = "soup_acceptance_obstruction_summary.csv"
REPORT_MD = "soup_acceptance_obstruction_diagnostics.md"

OBSTRUCTION_FEATURES = [
    "mean_cycle_score",
    "pairwise_alignment_residual_mean",
    "combined_obstruction_score",
    "sync_disagreement",
]
BARRIER_FEATURES = [
    "barrier_weight_average_val_max_loss_barrier",
    "barrier_git_rebasin_pairwise_ref0_val_max_loss_barrier",
    "barrier_c2m3_synchronized_val_max_loss_barrier",
    "barrier_greedy_soup_val_max_loss_barrier",
    "barrier_c2m3_val_max_delta_vs_weight_average",
    "barrier_c2m3_val_max_delta_vs_git_rebasin",
]
LOCAL_QUALITY_FEATURES = [
    "candidate_individual_val_accuracy",
    "candidate_individual_val_loss",
    "candidate_val_accuracy_gap_to_best_validation",
    "candidate_rank_fraction",
]
FEATURE_SETS = {
    "obstruction_only": OBSTRUCTION_FEATURES + BARRIER_FEATURES,
    "local_quality_only": LOCAL_QUALITY_FEATURES,
    "obstruction_plus_local_quality": OBSTRUCTION_FEATURES + BARRIER_FEATURES + LOCAL_QUALITY_FEATURES,
}
REGRESSION_TARGETS = [
    "validation_margin_after_adding_candidate_proxy",
    "absolute_validation_degradation_if_rejected_proxy",
]


def safe_float(value) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def finite_numeric(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)


def md_table(df: pd.DataFrame, columns: list[str], max_rows: int = 30) -> str:
    if df.empty:
        return "_No rows._"
    rows = df.head(max_rows).copy()
    for col in columns:
        if col not in rows.columns:
            rows[col] = ""
    return format_markdown_table(rows[columns].to_dict("records"), columns)


def auc_score(y_true: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=float)
    s = np.asarray(scores, dtype=float)
    mask = np.isfinite(y) & np.isfinite(s)
    y = y[mask]
    s = s[mask]
    n_pos = int((y > 0.5).sum())
    n_neg = int((y <= 0.5).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    sorted_scores = s[order]
    ranks = np.empty(len(s), dtype=float)
    start = 0
    while start < len(s):
        end = start + 1
        while end < len(s) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        rank = 0.5 * (start + 1 + end)
        ranks[order[start:end]] = rank
        start = end
    pos_rank_sum = float(ranks[y > 0.5].sum())
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def bootstrap_ci(values: list[float] | np.ndarray, samples: int, seed: int) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(max(samples, 1)):
        idx = rng.integers(0, len(arr), len(arr))
        draws.append(float(np.mean(arr[idx])))
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def quantile_ci(values: list[float] | np.ndarray) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float("nan"), float("nan")
    return float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))


def bootstrap_metric_ci(
    frame: pd.DataFrame,
    y_col: str,
    pred_col: str,
    metric_fn,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    clean = frame[[y_col, pred_col, "fixed_setting_id"]].dropna().copy()
    if clean.empty:
        return float("nan"), float("nan")
    groups = [group.copy() for _, group in clean.groupby("fixed_setting_id", dropna=False)]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(max(samples, 1)):
        picked = [groups[int(i)] for i in rng.integers(0, len(groups), len(groups))]
        sample = pd.concat(picked, ignore_index=True)
        val = metric_fn(sample[y_col].to_numpy(dtype=float), sample[pred_col].to_numpy(dtype=float))
        if math.isfinite(val):
            vals.append(val)
    if not vals:
        return float("nan"), float("nan")
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def r2_score(y_true: np.ndarray, pred: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(pred, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    y = y[mask]
    p = p[mask]
    if len(y) < 2:
        return float("nan")
    denom = float(np.sum((y - np.mean(y)) ** 2))
    if denom <= 1e-18:
        return float("nan")
    return 1.0 - float(np.sum((y - p) ** 2)) / denom


def mae_score(y_true: np.ndarray, pred: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(pred, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    if int(mask.sum()) == 0:
        return float("nan")
    return float(np.mean(np.abs(y[mask] - p[mask])))


def prepare_design(train: pd.DataFrame, eval_df: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    used = [feature for feature in features if feature in train.columns]
    if not used:
        used = ["__intercept_only__"]
        train = train.copy()
        eval_df = eval_df.copy()
        train[used[0]] = 0.0
        eval_df[used[0]] = 0.0
    x_train = train[used].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    x_eval = eval_df[used].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    means = np.nanmean(x_train, axis=0)
    means = np.where(np.isfinite(means), means, 0.0)
    x_train = np.where(np.isfinite(x_train), x_train, means)
    x_eval = np.where(np.isfinite(x_eval), x_eval, means)
    std = np.nanstd(x_train, axis=0)
    std = np.where(np.isfinite(std) & (std > 1e-12), std, 1.0)
    x_train = (x_train - means) / std
    x_eval = (x_eval - means) / std
    x_train = np.column_stack([np.ones(len(x_train)), x_train])
    x_eval = np.column_stack([np.ones(len(x_eval)), x_eval])
    return x_train, x_eval, used


def fit_logistic(x: np.ndarray, y: np.ndarray, l2: float = 1e-3, max_iter: int = 80) -> np.ndarray:
    beta = np.zeros(x.shape[1], dtype=float)
    penalty = np.eye(x.shape[1]) * l2
    penalty[0, 0] = 0.0
    for _ in range(max_iter):
        p = sigmoid(x @ beta)
        w = np.clip(p * (1.0 - p), 1e-8, None)
        grad = x.T @ (p - y) + penalty @ beta
        hess = (x.T * w) @ x + penalty
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(hess) @ grad
        beta -= step
        if float(np.max(np.abs(step))) < 1e-7:
            break
    return beta


def fit_linear(x: np.ndarray, y: np.ndarray, l2: float = 1e-6) -> np.ndarray:
    penalty = np.eye(x.shape[1]) * l2
    penalty[0, 0] = 0.0
    return np.linalg.pinv(x.T @ x + penalty) @ x.T @ y


def load_inputs(args) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    soup = pd.read_csv(args.soup_audit_csv)
    runs = pd.read_csv(args.fixed_runs_csv)
    barriers = pd.read_csv(args.barrier_targets_csv) if args.barrier_targets_csv.exists() else pd.DataFrame()
    return soup, runs, barriers


def run_level_obstruction_features(runs: pd.DataFrame) -> pd.DataFrame:
    observed = runs[
        (runs["alignment_source"].astype(str) == "observed")
        & (pd.to_numeric(runs["alignment_noise_fraction"], errors="coerce").fillna(0.0) == 0.0)
    ].copy()
    keep = [
        *RUN_KEY_COLS,
        "mean_cycle_score",
        "pairwise_alignment_residual_mean",
        "combined_obstruction_score",
        "sync_disagreement",
        "mean_individual_accuracy",
        "single_best_accuracy",
        "individual_accuracy_spread",
    ]
    keep = [col for col in keep if col in observed.columns]
    return observed.sort_values(["run_id", "method"], kind="stable").drop_duplicates("run_id")[keep].copy()


def barrier_setting_features(barriers: pd.DataFrame) -> pd.DataFrame:
    if barriers.empty:
        return pd.DataFrame(columns=[*SETTING_COLS, *BARRIER_FEATURES])
    observed = barriers[pd.to_numeric(barriers.get("max_eval_batches", 0), errors="coerce").fillna(0) >= 0].copy()
    rows = []
    for key, group in observed.groupby(SETTING_COLS, dropna=False):
        row = dict(zip(SETTING_COLS, key))
        for method in ["weight_average", "git_rebasin_pairwise_ref0", "c2m3_synchronized", "greedy_soup"]:
            part = group[group["method"].astype(str) == method]
            if part.empty:
                continue
            row[f"barrier_{method}_val_max_loss_barrier"] = safe_float(pd.to_numeric(part["val_max_loss_barrier"], errors="coerce").mean())
            row[f"barrier_{method}_val_midpoint_loss_barrier"] = safe_float(
                pd.to_numeric(part["val_midpoint_loss_barrier"], errors="coerce").mean()
            )
        if {
            "barrier_c2m3_synchronized_val_max_loss_barrier",
            "barrier_weight_average_val_max_loss_barrier",
        }.issubset(row):
            row["barrier_c2m3_val_max_delta_vs_weight_average"] = (
                row["barrier_c2m3_synchronized_val_max_loss_barrier"] - row["barrier_weight_average_val_max_loss_barrier"]
            )
        if {
            "barrier_c2m3_synchronized_val_max_loss_barrier",
            "barrier_git_rebasin_pairwise_ref0_val_max_loss_barrier",
        }.issubset(row):
            row["barrier_c2m3_val_max_delta_vs_git_rebasin"] = (
                row["barrier_c2m3_synchronized_val_max_loss_barrier"] - row["barrier_git_rebasin_pairwise_ref0_val_max_loss_barrier"]
            )
        rows.append(row)
    out = pd.DataFrame(rows)
    for col in BARRIER_FEATURES:
        if col not in out.columns:
            out[col] = np.nan
    return out[[*SETTING_COLS, *BARRIER_FEATURES]]


def build_candidate_table(soup: pd.DataFrame, runs: pd.DataFrame, barriers: pd.DataFrame) -> pd.DataFrame:
    run_features = run_level_obstruction_features(runs)
    barrier_features = barrier_setting_features(barriers)
    out = soup.merge(run_features, on=RUN_KEY_COLS, how="left", suffixes=("", "_run"))
    out = out.merge(barrier_features, on=SETTING_COLS, how="left")
    out["fixed_setting_id"] = out[SETTING_COLS].astype(str).agg("|".join, axis=1)
    out["candidate_accepted_by_greedy_soup"] = out["accepted"].astype(bool).astype(int)
    out["final_ingredient_count"] = (
        out["final_selection_indices"].astype(str).str.findall(r"-?\d+").map(len).astype(int)
    )
    out["candidate_rank_fraction"] = pd.to_numeric(out["candidate_rank"], errors="coerce") / pd.to_numeric(
        out["n_models"], errors="coerce"
    )
    out["candidate_val_accuracy_gap_to_best_validation"] = (
        pd.to_numeric(out["candidate_individual_val_accuracy"], errors="coerce")
        - pd.to_numeric(out["best_validation_model_val_accuracy"], errors="coerce")
    )
    out["validation_margin_after_adding_candidate"] = pd.to_numeric(
        out["validation_accuracy_margin_after_minus_before"], errors="coerce"
    )
    out["validation_margin_after_adding_candidate_logged"] = out["validation_margin_after_adding_candidate"].notna()
    out["validation_margin_after_adding_candidate_proxy"] = out["candidate_val_accuracy_gap_to_best_validation"]
    out["validation_margin_target_source"] = np.where(
        out["validation_margin_after_adding_candidate_logged"],
        "logged_candidate_soup_margin",
        "proxy_individual_validation_gap_to_best_validation_model",
    )
    rejected = out["candidate_accepted_by_greedy_soup"] == 0
    out["absolute_validation_degradation_if_rejected"] = np.nan
    out["absolute_validation_degradation_if_rejected_logged"] = False
    out["absolute_validation_degradation_if_rejected_proxy"] = np.where(
        rejected,
        np.maximum(0.0, -pd.to_numeric(out["validation_margin_after_adding_candidate_proxy"], errors="coerce")),
        0.0,
    )
    out["absolute_degradation_target_source"] = np.where(
        rejected,
        "proxy_best_validation_accuracy_minus_candidate_individual_validation_accuracy",
        "not_rejected_zero_by_definition",
    )
    out["uses_test_labels_for_training"] = False
    return out


def cross_validated_predictions(df: pd.DataFrame, bootstrap_samples: int) -> tuple[pd.DataFrame, list[dict]]:
    out = df.copy()
    summary_rows: list[dict] = []
    settings = sorted(out["fixed_setting_id"].dropna().unique())
    y_accept = out["candidate_accepted_by_greedy_soup"].to_numpy(dtype=float)
    for feature_set_name, features in FEATURE_SETS.items():
        accept_col = f"cv_acceptance_probability__{feature_set_name}"
        out[accept_col] = np.nan
        for target in REGRESSION_TARGETS:
            out[f"cv_prediction__{target}__{feature_set_name}"] = np.nan
        used_features = [feature for feature in features if feature in out.columns]
        for held_setting in settings:
            train = out[out["fixed_setting_id"] != held_setting].copy()
            valid = out[out["fixed_setting_id"] == held_setting].copy()
            if train.empty or valid.empty:
                continue
            x_train, x_valid, _used = prepare_design(train, valid, used_features)
            y_train = train["candidate_accepted_by_greedy_soup"].to_numpy(dtype=float)
            if len(np.unique(y_train)) >= 2:
                beta = fit_logistic(x_train, y_train)
                out.loc[valid.index, accept_col] = sigmoid(x_valid @ beta)
            for target in REGRESSION_TARGETS:
                y = pd.to_numeric(train[target], errors="coerce").to_numpy(dtype=float)
                mask = np.isfinite(y)
                if int(mask.sum()) < 3 or float(np.nanstd(y[mask])) <= 1e-12:
                    continue
                beta = fit_linear(x_train[mask], y[mask])
                out.loc[valid.index, f"cv_prediction__{target}__{feature_set_name}"] = x_valid @ beta

        auc = auc_score(y_accept, out[accept_col].to_numpy(dtype=float))
        auc_low, auc_high = bootstrap_metric_ci(
            out,
            "candidate_accepted_by_greedy_soup",
            accept_col,
            auc_score,
            bootstrap_samples,
            seed=72011 + len(summary_rows),
        )
        summary_rows.append(
            {
                "summary_type": "cross_validated_acceptance_auc",
                "target": "candidate_accepted_by_greedy_soup",
                "feature_set": feature_set_name,
                "predictor": "all_features",
                "n_rows": int(len(out)),
                "n_fixed_settings": int(len(settings)),
                "metric": "auc",
                "metric_value": auc,
                "bootstrap_ci_low": auc_low,
                "bootstrap_ci_high": auc_high,
                "claim_boundary": "explains_greedy_soup_acceptance_not_soup_beating",
            }
        )
        for target in REGRESSION_TARGETS:
            pred_col = f"cv_prediction__{target}__{feature_set_name}"
            y = pd.to_numeric(out[target], errors="coerce").to_numpy(dtype=float)
            pred = pd.to_numeric(out[pred_col], errors="coerce").to_numpy(dtype=float)
            r2 = r2_score(y, pred)
            mae = mae_score(y, pred)
            r2_low, r2_high = bootstrap_metric_ci(out, target, pred_col, r2_score, bootstrap_samples, seed=73017 + len(summary_rows))
            summary_rows.append(
                {
                    "summary_type": "cross_validated_regression",
                    "target": target,
                    "feature_set": feature_set_name,
                    "predictor": "all_features",
                    "n_rows": int(np.isfinite(y).sum()),
                    "n_fixed_settings": int(len(settings)),
                    "metric": "r2",
                    "metric_value": r2,
                    "bootstrap_ci_low": r2_low,
                    "bootstrap_ci_high": r2_high,
                    "mae": mae,
                    "claim_boundary": "proxy_validation_margin_not_logged_rejected_candidate_soup_margin",
                }
            )
    return out, summary_rows


def bootstrap_coefficients(
    frame: pd.DataFrame,
    used_features: list[str],
    target: str,
    model_type: str,
    samples: int,
    seed: int,
) -> np.ndarray:
    groups = [group.copy() for _, group in frame.groupby("fixed_setting_id", dropna=False)]
    if not groups:
        return np.empty((0, len(used_features) + 1))
    rng = np.random.default_rng(seed)
    betas = []
    for _ in range(max(samples, 1)):
        sample = pd.concat([groups[int(i)] for i in rng.integers(0, len(groups), len(groups))], ignore_index=True)
        xs, _xe, _u = prepare_design(sample, sample, used_features)
        y = pd.to_numeric(sample[target], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(y)
        if int(finite.sum()) < 3:
            continue
        if model_type == "logistic":
            yy = y[finite]
            if len(np.unique(yy)) < 2:
                continue
            betas.append(fit_logistic(xs[finite], yy))
        else:
            betas.append(fit_linear(xs[finite], y[finite]))
    if not betas:
        return np.empty((0, len(used_features) + 1))
    return np.vstack(betas)


def coefficient_summary(df: pd.DataFrame, bootstrap_samples: int, coefficient_bootstrap_samples: int) -> list[dict]:
    rows: list[dict] = []
    settings = sorted(df["fixed_setting_id"].dropna().unique())
    for feature_set_name, features in FEATURE_SETS.items():
        used_features = [feature for feature in features if feature in df.columns]
        train = df.copy()
        x_all, _x_eval, used = prepare_design(train, train, used_features)
        y_accept = train["candidate_accepted_by_greedy_soup"].to_numpy(dtype=float)
        if len(np.unique(y_accept)) >= 2:
            beta = fit_logistic(x_all, y_accept)
            boot = bootstrap_coefficients(
                train,
                used,
                "candidate_accepted_by_greedy_soup",
                "logistic",
                min(bootstrap_samples, coefficient_bootstrap_samples),
                seed=81031 + len(rows) * 7,
            )
            for idx, predictor in enumerate(["intercept", *used]):
                low, high = quantile_ci(boot[:, idx]) if len(boot) else (float("nan"), float("nan"))
                rows.append(
                    {
                        "summary_type": "logistic_coefficient",
                        "target": "candidate_accepted_by_greedy_soup",
                        "feature_set": feature_set_name,
                        "predictor": predictor,
                        "n_rows": int(len(train)),
                        "n_fixed_settings": int(len(settings)),
                        "metric": "standardized_logistic_beta",
                        "metric_value": float(beta[idx]),
                        "bootstrap_ci_low": low,
                        "bootstrap_ci_high": high,
                        "claim_boundary": "coefficient_for_explaining_greedy_soup_acceptance_only",
                    }
                )
        for target in REGRESSION_TARGETS:
            y = pd.to_numeric(train[target], errors="coerce").to_numpy(dtype=float)
            mask = np.isfinite(y)
            if int(mask.sum()) < 3:
                continue
            beta = fit_linear(x_all[mask], y[mask])
            boot = bootstrap_coefficients(
                train[mask].copy(),
                used,
                target,
                "linear",
                min(bootstrap_samples, coefficient_bootstrap_samples),
                seed=82037 + len(rows) * 11,
            )
            for idx, predictor in enumerate(["intercept", *used]):
                low, high = quantile_ci(boot[:, idx]) if len(boot) else (float("nan"), float("nan"))
                rows.append(
                    {
                        "summary_type": "regression_beta",
                        "target": target,
                        "feature_set": feature_set_name,
                        "predictor": predictor,
                        "n_rows": int(mask.sum()),
                        "n_fixed_settings": int(len(settings)),
                        "metric": "standardized_linear_beta",
                        "metric_value": float(beta[idx]),
                        "bootstrap_ci_low": low,
                        "bootstrap_ci_high": high,
                        "claim_boundary": "proxy_validation_target_not_test_selection",
                    }
                )
    return rows


def per_setting_stability(df: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    for setting_id, group in df.groupby("fixed_setting_id", dropna=False):
        for feature_set_name in FEATURE_SETS:
            pred_col = f"cv_acceptance_probability__{feature_set_name}"
            if pred_col not in group:
                continue
            auc = auc_score(group["candidate_accepted_by_greedy_soup"].to_numpy(dtype=float), group[pred_col].to_numpy(dtype=float))
            rows.append(
                {
                    "summary_type": "per_fixed_setting_acceptance_auc",
                    "fixed_setting_id": setting_id,
                    "target": "candidate_accepted_by_greedy_soup",
                    "feature_set": feature_set_name,
                    "predictor": "all_features",
                    "n_rows": int(len(group)),
                    "n_unique_seeds": int(group["seed"].nunique()),
                    "metric": "auc",
                    "metric_value": auc,
                    "claim_boundary": "stability_diagnostic_only",
                }
            )
            for target in REGRESSION_TARGETS:
                pcol = f"cv_prediction__{target}__{feature_set_name}"
                if pcol not in group:
                    continue
                rows.append(
                    {
                        "summary_type": "per_fixed_setting_regression_r2",
                        "fixed_setting_id": setting_id,
                        "target": target,
                        "feature_set": feature_set_name,
                        "predictor": "all_features",
                        "n_rows": int(len(group)),
                        "n_unique_seeds": int(group["seed"].nunique()),
                        "metric": "r2",
                        "metric_value": r2_score(group[target].to_numpy(dtype=float), group[pcol].to_numpy(dtype=float)),
                        "claim_boundary": "stability_diagnostic_only",
                    }
                )
    return rows


def target_availability_rows(df: pd.DataFrame) -> list[dict]:
    direct_margin_logged = int(df["validation_margin_after_adding_candidate_logged"].sum())
    rejected = df[df["candidate_accepted_by_greedy_soup"] == 0]
    return [
        {
            "summary_type": "target_availability",
            "target": "validation_margin_after_adding_candidate",
            "feature_set": "not_modelled_direct_target_unavailable",
            "predictor": "",
            "n_rows": int(len(df)),
            "metric": "logged_rows",
            "metric_value": direct_margin_logged,
            "claim_boundary": "true_candidate_soup_margins_not_logged_for_current_artifacts",
        },
        {
            "summary_type": "target_availability",
            "target": "absolute_validation_degradation_if_rejected",
            "feature_set": "not_modelled_direct_target_unavailable",
            "predictor": "",
            "n_rows": int(len(rejected)),
            "metric": "logged_rows",
            "metric_value": 0,
            "claim_boundary": "rejected_candidate_after_metrics_not_logged_proxy_only",
        },
        {
            "summary_type": "target_availability",
            "target": "final_ingredient_count",
            "feature_set": "constant_in_prompt22_audit",
            "predictor": "",
            "n_rows": int(df["run_id"].nunique()),
            "metric": "std",
            "metric_value": safe_float(df.drop_duplicates("run_id")["final_ingredient_count"].std(ddof=0)),
            "claim_boundary": "not_regressed_because_final_ingredient_count_is_constant_in_this_audit",
        },
    ]


def write_report(args, diagnostics: pd.DataFrame, summary: pd.DataFrame, path: Path) -> None:
    auc_rows = summary[summary["summary_type"] == "cross_validated_acceptance_auc"].copy()
    regression_rows = summary[summary["summary_type"] == "cross_validated_regression"].copy()
    beta_rows = summary[
        (summary["summary_type"].isin(["regression_beta", "logistic_coefficient"]))
        & (summary["predictor"].astype(str) != "intercept")
    ].copy()
    availability = summary[summary["summary_type"] == "target_availability"].copy()
    stability = summary[summary["summary_type"] == "per_fixed_setting_acceptance_auc"].copy()
    best_auc = auc_rows.sort_values("metric_value", ascending=False).head(1)
    obstruction_auc = auc_rows[auc_rows["feature_set"] == "obstruction_only"]
    local_auc = auc_rows[auc_rows["feature_set"] == "local_quality_only"]
    combined_auc = auc_rows[auc_rows["feature_set"] == "obstruction_plus_local_quality"]

    def metric_text(frame: pd.DataFrame) -> str:
        if frame.empty:
            return "not available"
        row = frame.iloc[0]
        return f"{safe_float(row['metric_value']):.4f} [{safe_float(row['bootstrap_ci_low']):.4f}, {safe_float(row['bootstrap_ci_high']):.4f}]"

    report = f"""# Soup Acceptance Obstruction Diagnostics

Generated by `experiments/soup_acceptance_obstruction_diagnostics.py`.

## Exact Command

```bash
{args.command_string}
```

## Scope

- Greedy soup is treated as the empirical validation-descent baseline to explain, not as a baseline to beat.
- Models are trained by leave-one-fixed-setting-out cross-validation. Test columns are not used as predictors or labels.
- benchmark series 22 did not log validation metrics for rejected averaged candidate soups. This report therefore models direct acceptance labels and validation-quality proxy margins, and it marks true rejected-candidate degradation as unlogged.
- Barrier predictors use validation-loss barrier summaries, not test barriers.

## Outputs

- `reports/csv/{DIAGNOSTICS_CSV}`
- `reports/csv/{SUMMARY_CSV}`
- `reports/{REPORT_MD}`

## Headline

The strongest cross-validated acceptance model is `{best_auc.iloc[0]['feature_set'] if not best_auc.empty else 'not_available'}` with AUC {metric_text(best_auc)}. Obstruction-only AUC is {metric_text(obstruction_auc)}; local-quality-only AUC is {metric_text(local_auc)}; combined AUC is {metric_text(combined_auc)}.

This is an explanation of greedy-soup selection behavior, not evidence for a soup-beating method.

## Target Availability

{md_table(availability, ["target", "feature_set", "n_rows", "metric", "metric_value", "claim_boundary"], 20)}

## Cross-Validated Acceptance

{md_table(auc_rows, ["feature_set", "target", "n_rows", "n_fixed_settings", "metric", "metric_value", "bootstrap_ci_low", "bootstrap_ci_high", "claim_boundary"], 20)}

## Cross-Validated Validation-Margin Proxies

{md_table(regression_rows, ["feature_set", "target", "n_rows", "n_fixed_settings", "metric", "metric_value", "bootstrap_ci_low", "bootstrap_ci_high", "mae", "claim_boundary"], 30)}

## Coefficients

{md_table(beta_rows, ["summary_type", "target", "feature_set", "predictor", "metric", "metric_value", "bootstrap_ci_low", "bootstrap_ci_high", "claim_boundary"], 50)}

## Per-Fixed-Setting Stability

{md_table(stability, ["fixed_setting_id", "feature_set", "n_rows", "n_unique_seeds", "metric", "metric_value", "claim_boundary"], 50)}

## Interpretation

The accepted candidate in these benchmark series 22 artifacts is the initial best-validation local model in every run, while later candidate-soup validation metrics for rejected averaged candidates are not logged. Consequently, local validation quality explains acceptance much more directly than global obstruction scores. Obstruction and barrier features are retained as diagnostics, but this report should be cited only as a soup-compatible explanation layer: it asks which diagnostics align with greedy soup's validation decisions, not whether any method beats greedy soup.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--soup-audit-csv", type=Path, default=ROOT / "reports" / "csv" / "greedy_soup_descent_audit.csv")
    parser.add_argument("--fixed-runs-csv", type=Path, default=ROOT / "reports" / "csv" / "fixed_setting_verification_runs.csv")
    parser.add_argument("--barrier-targets-csv", type=Path, default=ROOT / "reports" / "csv" / "alignment_barrier_targets.csv")
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "reports")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--coefficient-bootstrap-samples", type=int, default=500)
    args = parser.parse_args()
    args.command_string = " ".join([sys.executable, *sys.argv])

    soup, runs, barriers = load_inputs(args)
    diagnostics = build_candidate_table(soup, runs, barriers)
    diagnostics, summary_rows = cross_validated_predictions(diagnostics, args.bootstrap_samples)
    summary_rows.extend(coefficient_summary(diagnostics, args.bootstrap_samples, args.coefficient_bootstrap_samples))
    summary_rows.extend(per_setting_stability(diagnostics))
    summary_rows.extend(target_availability_rows(diagnostics))
    summary = pd.DataFrame(summary_rows)

    csv_dir = args.reports_dir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_path = csv_dir / DIAGNOSTICS_CSV
    summary_path = csv_dir / SUMMARY_CSV
    report_path = args.reports_dir / REPORT_MD
    diagnostics.to_csv(diagnostics_path, index=False, lineterminator="\n")
    summary.to_csv(summary_path, index=False, lineterminator="\n")
    write_report(args, diagnostics, summary, report_path)
    print(f"wrote {diagnostics_path}")
    print(f"wrote {summary_path}")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
