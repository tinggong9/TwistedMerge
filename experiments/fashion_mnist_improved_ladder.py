#!/usr/bin/env python
"""Fashion-MNIST validation of the improved ReLU-compatible ladder benchmark."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.improved_validated_ladder_merge_benchmark import (  # noqa: E402
    METHODS,
    bootstrap_mean_ci,
    git_commit,
    git_worktree_dirty,
    parse_csv,
    run_setting as _run_ladder_setting,
    safe_corr,
    sign_test_two_sided,
    standard_error,
)
from src.metrics import capture_environment  # noqa: E402
from src.model_merging_benchmark import load_dataset  # noqa: E402


METHOD_ORDER = [
    "weight_average",
    "c2m3_permutation",
    "monomial_scale",
    "shrinkage_monomial_scale",
    "global_monomial_scale",
    "optimized_monomial_scale",
    "validated_ladder_selector",
    "improved_validated_selector",
    "greedy_soup",
    "c2m3_greedy_soup",
    "monomial_scaled_greedy_soup",
    "shrinkage_monomial_greedy_soup",
    "global_monomial_greedy_soup",
    "optimized_monomial_greedy_soup",
    "union_candidate_soup",
    "ensemble_upper_bound",
]
BASELINES = ["c2m3_permutation", "greedy_soup", "weight_average"]
INT_COLUMNS = {
    "n_rows",
    "n_seeds",
    "n_pairs",
    "accuracy_wins",
    "accuracy_ties",
    "accuracy_losses",
    "fixed_settings_positive",
    "fixed_settings_total",
    "selector_chosen_test_better",
    "selector_chosen_test_tied",
    "selector_chosen_test_worse",
    "soup_mean_ingredient_count",
    "n_settings",
}


@dataclass(frozen=True)
class PairComparison:
    name: str
    method: str
    baseline: str
    claim_label: str


def default_seed_text(start: int, count: int) -> str:
    return ",".join(str(seed) for seed in range(start, start + count))


def setting_seed_plan(args) -> list[tuple[int, int, list[int], str]]:
    main_n, main_w = [int(item) for item in args.main_setting.split(":", maxsplit=1)]
    main_seeds = parse_csv(args.main_seeds, int)
    secondary_seeds = parse_csv(args.secondary_seeds, int)
    plan = []
    for n_models in parse_csv(args.model_counts, int):
        for width in parse_csv(args.widths, int):
            is_main = n_models == main_n and width == main_w
            plan.append((n_models, width, main_seeds if is_main else secondary_seeds, "main" if is_main else "secondary"))
    return plan


def command_string() -> str:
    env_prefix = [
        f"{name}={os.environ[name]}"
        for name in ("PYTHONPYCACHEPREFIX", "MPLCONFIGDIR")
        if os.environ.get(name)
    ]
    return " ".join([*env_prefix, sys.executable, *sys.argv])


def run_setting(args, spec, train_data, test_data, seed: int, n_models: int, width: int, setting_role: str) -> list[dict]:
    rows = _run_ladder_setting(args, spec, train_data, test_data, seed, n_models, width)
    setting_id = f"fashion_mnist_mlp_N{n_models}_W{width}_S{seed}"
    by_method = {row["method"]: row for row in rows}
    weight_acc = float(by_method["weight_average"]["accuracy"])
    weight_loss = float(by_method["weight_average"]["loss"])
    weight_val = float(by_method["weight_average"]["val_accuracy"])
    for row in rows:
        row["setting_id"] = setting_id
        row["dataset"] = "fashion_mnist"
        row["setting_role"] = setting_role
        row["accuracy_delta_vs_weight_average"] = float(row["accuracy"]) - weight_acc
        row["loss_delta_vs_weight_average"] = float(row["loss"]) - weight_loss
        row["validation_delta_vs_weight_average"] = float(row["val_accuracy"]) - weight_val
        row["noncentral_final_decision"] = row.get("ladder_final_decision") == "report_noncentral_holonomy"
        row["non_brauer_or_noncentral_candidate"] = not bool(row.get("supports_brauer_projective_interpretation", False))
    return rows


def comparison_grid(methods: list[str]) -> list[PairComparison]:
    rows = []
    for baseline in BASELINES:
        for method in methods:
            if method == baseline:
                continue
            rows.append(
                PairComparison(
                    name=f"{method}_vs_{baseline}",
                    method=method,
                    baseline=baseline,
                    claim_label=f"{method} over {baseline}",
                )
            )
    return rows


def summarize_methods(df: pd.DataFrame, n_bootstrap: int) -> list[dict]:
    rows = []
    for scope, keys in [("overall", ["method"]), ("fixed_setting", ["n_models", "width", "method"])]:
        for key_values, group in df.groupby(keys, dropna=False):
            if not isinstance(key_values, tuple):
                key_values = (key_values,)
            key_map = dict(zip(keys, key_values, strict=True))
            ci_low, ci_high = bootstrap_mean_ci(group["accuracy"], n_bootstrap=n_bootstrap, seed=5000 + len(rows))
            rows.append(
                {
                    "summary_type": "method_summary",
                    "scope": scope,
                    "n_models": key_map.get("n_models", "all"),
                    "width": key_map.get("width", "all"),
                    "method": key_map["method"],
                    "baseline": "",
                    "n_rows": int(len(group)),
                    "n_seeds": int(group["seed"].nunique()),
                    "mean_val_accuracy": float(pd.to_numeric(group["val_accuracy"], errors="coerce").mean()),
                    "mean_test_accuracy": float(pd.to_numeric(group["accuracy"], errors="coerce").mean()),
                    "test_accuracy_ci_low": ci_low,
                    "test_accuracy_ci_high": ci_high,
                    "test_accuracy_standard_error": standard_error(group["accuracy"]),
                    "mean_test_loss": float(pd.to_numeric(group["loss"], errors="coerce").mean()),
                    "mean_merge_degradation": float(pd.to_numeric(group["merge_degradation"], errors="coerce").mean()),
                    "mean_accuracy_delta_vs_c2m3": float(pd.to_numeric(group["accuracy_delta_vs_c2m3"], errors="coerce").mean()),
                    "mean_accuracy_delta_vs_greedy_soup": float(pd.to_numeric(group["accuracy_delta_vs_greedy_soup"], errors="coerce").mean()),
                    "mean_accuracy_delta_vs_weight_average": float(pd.to_numeric(group["accuracy_delta_vs_weight_average"], errors="coerce").mean()),
                    "mean_validation_delta_vs_c2m3": float(pd.to_numeric(group["validation_delta_vs_c2m3"], errors="coerce").mean()),
                    "mean_validation_delta_vs_greedy_soup": float(pd.to_numeric(group["validation_delta_vs_greedy_soup"], errors="coerce").mean()),
                    "mean_validation_delta_vs_weight_average": float(pd.to_numeric(group["validation_delta_vs_weight_average"], errors="coerce").mean()),
                    "validation_only_selection": bool(group["selector_no_test_leakage"].fillna(True).astype(bool).all()),
                    "symmetry_status": str(group["symmetry_status"].iloc[0]),
                    "is_single_model": bool(group["is_single_model"].iloc[0]),
                    "capacity_matched_to_weight_average": bool(group["capacity_matched_to_weight_average"].iloc[0]),
                }
            )
    return rows


def paired_rows(df: pd.DataFrame, n_bootstrap: int) -> list[dict]:
    rows = []
    fixed = df.pivot_table(
        index=["n_models", "width", "seed", "setting_id"],
        columns="method",
        values=["accuracy", "loss", "val_accuracy"],
        aggfunc="first",
    )
    fixed.columns = [f"{metric}__{method}" for metric, method in fixed.columns]
    fixed = fixed.reset_index()
    for comparison in comparison_grid([method for method in METHOD_ORDER if method in df["method"].unique()]):
        acc_method = f"accuracy__{comparison.method}"
        acc_base = f"accuracy__{comparison.baseline}"
        loss_method = f"loss__{comparison.method}"
        loss_base = f"loss__{comparison.baseline}"
        val_method = f"val_accuracy__{comparison.method}"
        val_base = f"val_accuracy__{comparison.baseline}"
        needed = [acc_method, acc_base, loss_method, loss_base, val_method, val_base, "n_models", "width"]
        if any(col not in fixed.columns for col in needed):
            continue
        clean = fixed[needed].dropna()
        accuracy_delta = clean[acc_method] - clean[acc_base]
        validation_delta = clean[val_method] - clean[val_base]
        loss_delta = clean[loss_method] - clean[loss_base]
        wins = int((accuracy_delta > 0).sum())
        ties = int((accuracy_delta == 0).sum())
        losses = int((accuracy_delta < 0).sum())
        ci_low, ci_high = bootstrap_mean_ci(accuracy_delta, n_bootstrap=n_bootstrap, seed=8110 + len(rows))
        fixed_positive = 0
        fixed_total = 0
        for (_n, _w), group in clean.groupby(["n_models", "width"], dropna=False):
            fixed_total += 1
            if float((group[acc_method] - group[acc_base]).mean()) > 0:
                fixed_positive += 1
        rows.append(
            {
                "summary_type": "paired_comparison",
                "scope": "overall_paired",
                "comparison": comparison.name,
                "claim_label": comparison.claim_label,
                "method": comparison.method,
                "baseline": comparison.baseline,
                "n_pairs": int(len(clean)),
                "paired_mean_accuracy_delta": float(accuracy_delta.mean()) if len(clean) else float("nan"),
                "paired_accuracy_delta_ci_low": ci_low,
                "paired_accuracy_delta_ci_high": ci_high,
                "paired_mean_validation_delta": float(validation_delta.mean()) if len(clean) else float("nan"),
                "paired_mean_loss_delta": float(loss_delta.mean()) if len(clean) else float("nan"),
                "accuracy_wins": wins,
                "accuracy_ties": ties,
                "accuracy_losses": losses,
                "sign_test_two_sided_p": sign_test_two_sided(wins, losses),
                "fixed_settings_positive": fixed_positive,
                "fixed_settings_total": fixed_total,
                "validation_only_selection": True,
            }
        )
    return rows


def selector_rows(df: pd.DataFrame) -> list[dict]:
    rows = []
    for method in ["validated_ladder_selector", "improved_validated_selector"]:
        selector = df[df["method"] == method].copy()
        if selector.empty:
            continue
        groups = [("overall", selector)]
        groups.extend((f"N{n}_W{w}", group) for (n, w), group in selector.groupby(["n_models", "width"]))
        for scope, group in groups:
            choices = group["selector_chose"].value_counts(dropna=False)
            better = group["selector_chosen_test_better"].fillna(False).astype(bool)
            tied = group["selector_chosen_test_tied"].fillna(False).astype(bool)
            worse = group["selector_chosen_test_worse"].fillna(False).astype(bool)
            rows.append(
                {
                    "summary_type": "selector_behavior",
                    "scope": scope,
                    "method": method,
                    "n_rows": int(len(group)),
                    "selector_choice_counts": json.dumps({str(k): int(v) for k, v in choices.items()}),
                    "selector_behavior_reference": str(group["selector_behavior_reference"].dropna().iloc[0]) if group["selector_behavior_reference"].notna().any() else "",
                    "selector_chosen_test_better": int(better.sum()),
                    "selector_chosen_test_tied": int(tied.sum()),
                    "selector_chosen_test_worse": int(worse.sum()),
                    "validation_only_selection": bool(group["selector_no_test_leakage"].fillna(True).astype(bool).all()),
                }
            )
    return rows


def alpha_tau_rows(df: pd.DataFrame) -> list[dict]:
    rows = []
    methods = ["shrinkage_monomial_scale", "global_monomial_scale", "optimized_monomial_scale"]
    subset = df[df["method"].isin(methods)].copy()
    for method, group in subset.groupby("method"):
        alpha = pd.to_numeric(group["selected_alpha"], errors="coerce")
        tau = pd.to_numeric(group["selected_tau"].replace(float("inf"), np.nan), errors="coerce")
        rows.append(
            {
                "summary_type": "alpha_tau_selection",
                "scope": "overall",
                "method": method,
                "n_rows": int(len(group)),
                "mean_selected_alpha": float(alpha.mean()),
                "fraction_alpha_zero": float((alpha == 0.0).mean()),
                "fraction_alpha_one_or_more": float((alpha >= 1.0).mean()),
                "mean_finite_selected_tau": float(tau.mean()),
                "fraction_tau_infinite": float(np.isinf(pd.to_numeric(group["selected_tau"], errors="coerce")).mean()),
                "validation_only_selection": True,
            }
        )
    return rows


def soup_rows(df: pd.DataFrame) -> list[dict]:
    rows = []
    soup_methods = [method for method in df["method"].unique() if "soup" in str(method)]
    for method in sorted(soup_methods):
        group = df[df["method"] == method].copy()
        rows.append(
            {
                "summary_type": "soup_ingredient_behavior",
                "scope": "overall",
                "method": method,
                "n_rows": int(len(group)),
                "soup_mean_ingredient_count": float(pd.to_numeric(group["soup_ingredient_count"], errors="coerce").mean()),
                "selected_type_examples": "; ".join(str(item) for item in group["soup_selected_types"].dropna().head(3).tolist()),
                "capacity_matched_to_weight_average": bool(group["capacity_matched_to_weight_average"].fillna(True).astype(bool).all()),
                "validation_only_selection": True,
            }
        )
    return rows


def residual_taxonomy_rows(df: pd.DataFrame) -> list[dict]:
    rows = []
    settings = df[df["method"] == "c2m3_permutation"].drop_duplicates("setting_id").copy()
    groups = [("overall", settings)]
    groups.extend((f"N{n}_W{w}", group) for (n, w), group in settings.groupby(["n_models", "width"]))
    for scope, group in groups:
        supports = group["supports_brauer_projective_interpretation"].fillna(False).astype(bool)
        finite = group["has_finite_index_candidate"].fillna(False).astype(bool)
        noncentral_final = group["noncentral_final_decision"].fillna(False).astype(bool)
        non_brauer = group["non_brauer_or_noncentral_candidate"].fillna(True).astype(bool)
        rows.append(
            {
                "summary_type": "residual_taxonomy",
                "scope": scope,
                "method": "",
                "n_models": "all" if scope == "overall" else int(group["n_models"].iloc[0]),
                "width": "all" if scope == "overall" else int(group["width"].iloc[0]),
                "n_settings": int(len(group)),
                "mean_permutation_cycle_score": float(pd.to_numeric(group["permutation_cycle_score"], errors="coerce").mean()),
                "mean_permutation_centrality": float(pd.to_numeric(group["permutation_centrality"], errors="coerce").mean()),
                "mean_signed_permutation_centrality": float(pd.to_numeric(group["signed_permutation_centrality"], errors="coerce").mean()),
                "mean_monomial_centrality": float(pd.to_numeric(group["monomial_phase_or_scale_centrality"], errors="coerce").mean()),
                "mean_monomial_centrality_improvement": float(pd.to_numeric(group["monomial_centrality_improvement_from_permutation"], errors="coerce").mean()),
                "noncentral_residual_fraction": float(noncentral_final.mean()) if len(group) else float("nan"),
                "non_brauer_or_noncentral_fraction": float(non_brauer.mean()) if len(group) else float("nan"),
                "finite_index_candidate_fraction": float(finite.mean()) if len(group) else float("nan"),
                "real_central_projective_candidate_fraction": float(supports.mean()) if len(group) else float("nan"),
                "ladder_final_decision_counts": json.dumps({str(k): int(v) for k, v in group["ladder_final_decision"].value_counts(dropna=False).items()}),
                "ladder_selected_level_counts": json.dumps({str(k): int(v) for k, v in group["ladder_selected_level"].value_counts(dropna=False).items()}),
            }
        )
    return rows


def residual_correlation_rows(df: pd.DataFrame) -> list[dict]:
    rows = []
    pivot = df.pivot_table(index=["n_models", "width", "seed", "setting_id"], columns="method", values="accuracy", aggfunc="first").reset_index()
    base = df[df["method"] == "monomial_scale"].drop_duplicates("setting_id")
    merged = base.merge(pivot, on=["n_models", "width", "seed", "setting_id"], suffixes=("", "__pivot"))
    predictors = [
        "permutation_cycle_score",
        "permutation_centrality",
        "signed_permutation_centrality",
        "monomial_phase_or_scale_centrality",
        "monomial_centrality_improvement_from_permutation",
        "global_scale_sync_rms_residual",
        "global_scale_sync_max_residual",
        "pairwise_alignment_residual",
        "sync_disagreement",
        "individual_accuracy_variance",
    ]
    target_defs = {
        "monomial_gain_vs_c2m3": merged["monomial_scale"] - merged["c2m3_permutation"],
        "shrinkage_gain_vs_c2m3": merged["shrinkage_monomial_scale"] - merged["c2m3_permutation"],
        "global_gain_vs_c2m3": merged["global_monomial_scale"] - merged["c2m3_permutation"],
        "optimized_gain_vs_c2m3": merged["optimized_monomial_scale"] - merged["c2m3_permutation"],
        "improved_selector_gain_vs_c2m3": merged["improved_validated_selector"] - merged["c2m3_permutation"],
        "improved_selector_gain_vs_greedy": merged["improved_validated_selector"] - merged["greedy_soup"],
    }
    for target_name, target in target_defs.items():
        for predictor in predictors:
            if predictor not in merged.columns:
                continue
            rows.append(
                {
                    "summary_type": "residual_diagnostic_correlation",
                    "scope": "overall",
                    "target": target_name,
                    "predictor": predictor,
                    "n_rows": int(len(merged)),
                    "pearson": safe_corr(merged[predictor], target, method="pearson"),
                    "spearman": safe_corr(merged[predictor], target, method="spearman"),
                }
            )
    return rows


def _comparison(summary_rows: list[dict], name: str) -> dict:
    for row in summary_rows:
        if row.get("summary_type") == "paired_comparison" and row.get("comparison") == name:
            return row
    return {}


def claim_decision_rows(summary_rows: list[dict]) -> list[dict]:
    rows = []
    taxonomy = next(
        (row for row in summary_rows if row.get("summary_type") == "residual_taxonomy" and row.get("scope") == "overall"),
        {},
    )
    methods = [
        row for row in summary_rows
        if row.get("summary_type") == "method_summary" and row.get("scope") == "overall"
    ]
    n_pairs = int(max((row.get("n_rows", 0) for row in methods), default=0))
    individual_target = max((float(row.get("mean_test_accuracy", 0.0)) for row in methods if row.get("method") == "ensemble_upper_bound"), default=0.0)

    def paired_support(name: str) -> tuple[str, str]:
        row = _comparison(summary_rows, name)
        if not row:
            return "Not evaluated", "paired comparison is missing"
        mean_delta = float(row["paired_mean_accuracy_delta"])
        ci_low = float(row["paired_accuracy_delta_ci_low"])
        fixed_positive = int(row["fixed_settings_positive"])
        fixed_total = int(row["fixed_settings_total"])
        if mean_delta > 0.0 and np.isfinite(ci_low) and ci_low > 0.0 and fixed_positive > fixed_total / 2 and n_pairs >= 30:
            return "Supported limited", "positive paired mean, positive bootstrap CI, and majority fixed-setting support"
        if mean_delta > 0.0:
            return "Supported descriptive", "positive paired mean but confidence or fixed-setting support is insufficient"
        return "Supported negative", "paired mean test accuracy delta is not positive"

    selector_vs_c2m3 = paired_support("improved_validated_selector_vs_c2m3_permutation")
    selector_vs_greedy = paired_support("improved_validated_selector_vs_greedy_soup")
    monomial_vs_c2m3 = paired_support("monomial_scale_vs_c2m3_permutation")
    shrinkage_vs_c2m3 = paired_support("shrinkage_monomial_scale_vs_c2m3_permutation")
    global_vs_c2m3 = paired_support("global_monomial_scale_vs_c2m3_permutation")
    optimized_vs_c2m3 = paired_support("optimized_monomial_scale_vs_c2m3_permutation")
    monomial_help_status = "Supported negative"
    monomial_help_reason = "no monomial-scale variant has positive supported paired evidence over C2M3"
    for status, reason in [monomial_vs_c2m3, shrinkage_vs_c2m3, global_vs_c2m3, optimized_vs_c2m3]:
        if status == "Supported limited":
            monomial_help_status = "Supported limited"
            monomial_help_reason = f"at least one monomial-scale variant is supported: {reason}"
            break
        if status == "Supported descriptive":
            monomial_help_status = "Supported descriptive"
            monomial_help_reason = f"at least one monomial-scale variant is descriptively positive: {reason}"

    non_brauer_fraction = float(taxonomy.get("non_brauer_or_noncentral_fraction", float("nan")))
    noncentral_fraction = float(taxonomy.get("noncentral_residual_fraction", float("nan")))
    finite_fraction = float(taxonomy.get("finite_index_candidate_fraction", float("nan")))
    central_fraction = float(taxonomy.get("real_central_projective_candidate_fraction", float("nan")))
    noncentral_status = "Supported limited" if noncentral_fraction >= 0.8 else "Supported negative result"
    noncentral_reason = (
        f"literal noncentral final-decision fraction={noncentral_fraction:.4f}; "
        "many remaining settings are GL-diagnostic-only rather than central/projective"
    )
    non_brauer_status = "Supported limited" if non_brauer_fraction >= 0.8 and central_fraction <= 0.05 else "Supported descriptive"
    non_brauer_reason = (
        f"non-Brauer/noncentral fraction={non_brauer_fraction:.4f}, "
        f"central/projective candidate fraction={central_fraction:.4f}, finite-index candidate fraction={finite_fraction:.4f}"
    )
    taxonomy_status = "Supported limited" if non_brauer_fraction >= 0.8 and finite_fraction <= 0.05 else "Supported descriptive"
    taxonomy_reason = (
        "taxonomy separates noncentral final decisions, GL-diagnostic-only reductions, "
        "and zero central/projective finite-index candidates"
    )
    greedy_boundary_status = "Supported negative result" if selector_vs_greedy[0] == "Supported negative" else selector_vs_greedy[0]
    greedy_boundary_reason = selector_vs_greedy[1]
    detector_status = "Supported limited" if finite_fraction <= 0.05 and central_fraction <= 0.05 else "Supported descriptive"
    detector_reason = "detector keeps finite-index and central/projective candidate fractions low on real Fashion-MNIST residuals"

    decisions = [
        ("fashion_selector_over_internal_c2m3", selector_vs_c2m3[0], selector_vs_c2m3[1]),
        ("fashion_monomial_scaling_helps", monomial_help_status, monomial_help_reason),
        ("fashion_improved_selector_over_greedy_soup", selector_vs_greedy[0], selector_vs_greedy[1]),
        ("fashion_greedy_soup_boundary_case", greedy_boundary_status, greedy_boundary_reason),
        ("fashion_residuals_mostly_noncentral", noncentral_status, noncentral_reason),
        ("fashion_residuals_non_brauer_no_central_projective", non_brauer_status, non_brauer_reason),
        ("fashion_residual_taxonomy_useful_beyond_mnist", taxonomy_status, taxonomy_reason),
        ("fashion_detector_conservative_no_finite_index_hallucination", detector_status, detector_reason),
        ("broad_model_merging_generalization", "Not supported", "Fashion-MNIST MLP is one additional dataset/architecture slice, not broad generalization"),
    ]
    if individual_target < 0.75:
        decisions.append(
            (
                "fashion_accuracy_target",
                "Caution",
                f"ensemble upper-bound mean accuracy proxy is only {individual_target:.4f}; inspect individual accuracies before making accuracy claims",
            )
        )
    for claim, decision, reason in decisions:
        rows.append(
            {
                "summary_type": "claim_decision",
                "scope": "overall",
                "claim": claim,
                "claim_decision": decision,
                "claim_reason": reason,
                "n_pairs": n_pairs,
            }
        )
    return rows


def summarize(df: pd.DataFrame, n_bootstrap: int) -> pd.DataFrame:
    rows: list[dict] = []
    rows.extend(summarize_methods(df, n_bootstrap))
    rows.extend(paired_rows(df, n_bootstrap))
    rows.extend(selector_rows(df))
    rows.extend(alpha_tau_rows(df))
    rows.extend(soup_rows(df))
    rows.extend(residual_taxonomy_rows(df))
    rows.extend(residual_correlation_rows(df))
    rows.extend(claim_decision_rows(rows))
    return pd.DataFrame(rows)


def format_value(value, column: str) -> str:
    if value == "":
        return ""
    if isinstance(value, bool):
        return str(value)
    try:
        if pd.isna(value):
            return "nan"
    except TypeError:
        pass
    if column in INT_COLUMNS:
        return str(int(round(float(value))))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4f}"
    return str(value)


def markdown_table(rows: list[dict], columns: list[str]) -> str:
    if not rows:
        return "_No rows._"
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(format_value(row.get(col, ""), col) for col in columns) + " |")
    return "\n".join(lines)


def write_delta_plot(df: pd.DataFrame, path: Path, baseline_column: str, title: str, ylabel: str) -> None:
    import matplotlib.pyplot as plt

    methods = [
        "monomial_scale",
        "shrinkage_monomial_scale",
        "global_monomial_scale",
        "optimized_monomial_scale",
        "improved_validated_selector",
        "union_candidate_soup",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    positions = np.arange(len(methods), dtype=float)
    for idx, method in enumerate(methods):
        group = df[df["method"] == method]
        if group.empty:
            continue
        jitter = np.linspace(-0.18, 0.18, len(group)) if len(group) > 1 else np.array([0.0])
        ax.scatter(np.full(len(group), positions[idx]) + jitter, group[baseline_column], alpha=0.58, s=18)
        ax.plot([positions[idx] - 0.22, positions[idx] + 0.22], [group[baseline_column].mean()] * 2, color="black", linewidth=1.2)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(positions)
    ax.set_xticklabels([method.replace("_", "\n") for method in methods], fontsize=7)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_residual_taxonomy_plot(summary: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    rows = summary[(summary["summary_type"] == "residual_taxonomy") & (summary["scope"] != "overall")].copy()
    if rows.empty:
        rows = summary[summary["summary_type"] == "residual_taxonomy"].copy()
    rows["label"] = rows.apply(lambda row: row["scope"], axis=1)
    metrics = [
        ("non_brauer_or_noncentral_fraction", "non-Brauer/noncentral"),
        ("noncentral_residual_fraction", "noncentral final"),
        ("finite_index_candidate_fraction", "finite-index"),
        ("real_central_projective_candidate_fraction", "central/projective"),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    x = np.arange(len(rows), dtype=float)
    width = 0.18
    for idx, (column, label) in enumerate(metrics):
        ax.bar(x + (idx - 1.5) * width, pd.to_numeric(rows[column], errors="coerce"), width=width, label=label)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Fraction of settings")
    ax.set_title("Fashion-MNIST Residual Taxonomy")
    ax.set_xticks(x)
    ax.set_xticklabels(rows["label"], rotation=35, ha="right", fontsize=7)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def latex_escape(value: object) -> str:
    return str(value).replace("_", "\\_")


def write_latex_table(summary: pd.DataFrame, path: Path) -> None:
    rows = summary[(summary["summary_type"] == "method_summary") & (summary["scope"] == "overall")].copy()
    rows["method_rank"] = rows["method"].map({method: idx for idx, method in enumerate(METHOD_ORDER)})
    rows = rows.sort_values("method_rank")
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Method & Val acc. & Test acc. & $\\Delta$ C2M3 & $\\Delta$ greedy \\\\",
        "\\midrule",
    ]
    for _idx, row in rows.iterrows():
        lines.append(
            f"{latex_escape(row['method'])} & "
            f"{float(row['mean_val_accuracy']):.4f} & "
            f"{float(row['mean_test_accuracy']):.4f} & "
            f"{float(row['mean_accuracy_delta_vs_c2m3']):+.4f} & "
            f"{float(row['mean_accuracy_delta_vs_greedy_soup']):+.4f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_report(
    args,
    df: pd.DataFrame,
    summary: pd.DataFrame,
    delta_c2m3_plot: Path,
    delta_greedy_plot: Path,
    residual_plot: Path,
    table_path: Path,
    path: Path,
) -> None:
    method_rows = [
        {
            "method": method,
            "symmetry_status": METHODS[method].symmetry_status,
            "is_single_model": METHODS[method].is_single_model,
            "capacity_matched": METHODS[method].capacity_matched_to_weight_average,
        }
        for method in METHOD_ORDER
    ]
    overall = summary[(summary["summary_type"] == "method_summary") & (summary["scope"] == "overall")].copy()
    overall["method_rank"] = overall["method"].map({method: idx for idx, method in enumerate(METHOD_ORDER)})
    overall = overall.sort_values("method_rank").drop(columns=["method_rank"]).to_dict("records")
    paired = summary[summary["summary_type"] == "paired_comparison"].copy()
    paired = paired[paired["method"].isin(["monomial_scale", "shrinkage_monomial_scale", "global_monomial_scale", "optimized_monomial_scale", "improved_validated_selector", "union_candidate_soup"])]
    selectors = summary[summary["summary_type"] == "selector_behavior"].to_dict("records")
    alpha_tau = summary[summary["summary_type"] == "alpha_tau_selection"].to_dict("records")
    soup = summary[summary["summary_type"] == "soup_ingredient_behavior"].to_dict("records")
    taxonomy = summary[summary["summary_type"] == "residual_taxonomy"].to_dict("records")
    correlations = summary[summary["summary_type"] == "residual_diagnostic_correlation"].head(30).to_dict("records")
    claims = summary[summary["summary_type"] == "claim_decision"].to_dict("records")
    claims_text = "\n".join(f"- `{row['claim']}`: {row['claim_decision']} ({row['claim_reason']})." for row in claims)
    seed_plan = "; ".join(f"N={n}, W={w}: {role}, {len(seeds)} seeds" for n, w, seeds, role in setting_seed_plan(args))
    individual_mean = float(df.drop_duplicates("setting_id")["individual_accuracy_mean"].mean())
    individual_max = float(df.drop_duplicates("setting_id")["individual_accuracy_max"].mean())
    target_note = "met" if individual_max >= 0.75 else "not met"
    report = f"""# Fashion-MNIST Improved Ladder Validation Report

This report is generated by `experiments/fashion_mnist_improved_ladder.py`.

## Exact Command

```bash
{args.command_string}
```

## Git State At Report Generation

- HEAD commit: `{git_commit()}`
- Worktree dirty: `{git_worktree_dirty()}`

## Dataset And Grid

- Dataset: Fashion-MNIST
- Architecture: one-hidden-layer ReLU MLP
- Optional CNN: not run in this benchmark, because the current monomial/optimized exact-gauge implementation is MLP hidden-unit specific.
- Model counts: `{args.model_counts}`
- Widths: `{args.widths}`
- Main setting: `{args.main_setting}` with `{args.main_seeds}`
- Secondary seeds: `{args.secondary_seeds}`
- Seed plan: {seed_plan}
- Epochs: `{args.epochs}`
- Train samples before validation split: `{args.max_train_samples}`
- Validation fraction: `{args.val_fraction}`
- Test samples: `{args.max_test_samples}` (`0` means full dataset)
- Mean individual test accuracy: `{individual_mean:.4f}`
- Mean best individual test accuracy: `{individual_max:.4f}`; 75% target status: `{target_note}`
- Matching: activation

All method choices are made using validation accuracy/loss only. Test metrics are computed after selection.

## Method Labels

{markdown_table(method_rows, ["method", "symmetry_status", "is_single_model", "capacity_matched"])}

## Main Performance Table

{markdown_table(overall, ["method", "n_rows", "n_seeds", "mean_val_accuracy", "mean_test_accuracy", "test_accuracy_ci_low", "test_accuracy_ci_high", "mean_accuracy_delta_vs_c2m3", "mean_accuracy_delta_vs_greedy_soup", "mean_accuracy_delta_vs_weight_average", "validation_only_selection"])}

## Paired Comparison Highlights

{markdown_table(paired.to_dict("records"), ["comparison", "n_pairs", "paired_mean_accuracy_delta", "paired_accuracy_delta_ci_low", "paired_accuracy_delta_ci_high", "paired_mean_validation_delta", "accuracy_wins", "accuracy_ties", "accuracy_losses", "fixed_settings_positive", "fixed_settings_total", "validation_only_selection"])}

Full paired comparisons for every method against C2M3, greedy soup, and weight average are in `reports/csv/fashion_mnist_improved_ladder_summary.csv`.

## Selector Behavior

{markdown_table(selectors, ["method", "scope", "n_rows", "selector_choice_counts", "selector_behavior_reference", "selector_chosen_test_better", "selector_chosen_test_tied", "selector_chosen_test_worse", "validation_only_selection"])}

## Monomial Scale Selection

{markdown_table(alpha_tau, ["method", "n_rows", "mean_selected_alpha", "fraction_alpha_zero", "fraction_alpha_one_or_more", "mean_finite_selected_tau", "fraction_tau_infinite", "validation_only_selection"])}

## Soup Ingredient Behavior

{markdown_table(soup, ["method", "n_rows", "soup_mean_ingredient_count", "selected_type_examples", "capacity_matched_to_weight_average", "validation_only_selection"])}

Union candidate soup is still one averaged MLP with the original architecture and parameter count; it is not an ensemble.

## Residual Taxonomy

{markdown_table(taxonomy, ["scope", "n_settings", "mean_permutation_cycle_score", "mean_permutation_centrality", "mean_signed_permutation_centrality", "mean_monomial_centrality", "mean_monomial_centrality_improvement", "noncentral_residual_fraction", "non_brauer_or_noncentral_fraction", "finite_index_candidate_fraction", "real_central_projective_candidate_fraction", "ladder_final_decision_counts"])}

## Residual Diagnostic Correlations

{markdown_table(correlations, ["target", "predictor", "n_rows", "pearson", "spearman"])}

## Figures And Tables

- `reports/plots/{delta_c2m3_plot.name}`
- `reports/plots/{delta_greedy_plot.name}`
- `reports/plots/{residual_plot.name}`
- `reports/tables/{table_path.name}`

## Claim Decision Table

{claims_text}

## Paper Framing Guidance

If the selector improves over internal C2M3 but not greedy soup, this should be framed as a Fashion-MNIST boundary case for practical merging and as additional evidence for the residual taxonomy. If the selector also fails over C2M3, the Fashion-MNIST result should be treated as a negative generalization result. In either case, this benchmark does not support claims about external C2M3 implementations, broad model-merging generalization, or Brauer/period-index classes in Fashion-MNIST residuals.

## Environment

```json
{json.dumps(capture_environment(), indent=2)}
```
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")


def write_config(args, path: Path) -> None:
    config = {
        "command": args.command_string,
        "git_commit": git_commit(),
        "dirty_worktree": git_worktree_dirty(),
        "dataset": "fashion_mnist",
        "architecture": "mlp_relu",
        "model_counts": parse_csv(args.model_counts, int),
        "widths": parse_csv(args.widths, int),
        "main_setting": args.main_setting,
        "main_seeds": parse_csv(args.main_seeds, int),
        "secondary_seeds": parse_csv(args.secondary_seeds, int),
        "seed_plan": [
            {"n_models": n, "width": w, "seeds": seeds, "role": role}
            for n, w, seeds, role in setting_seed_plan(args)
        ],
        "epochs": args.epochs,
        "max_train_samples": args.max_train_samples,
        "max_test_samples": args.max_test_samples,
        "val_fraction": args.val_fraction,
        "alpha_grid": parse_csv(args.alpha_grid, float),
        "tau_grid": parse_csv(args.tau_grid, str),
        "optimization_steps": args.optimization_steps,
        "optimization_lr": args.optimization_lr,
        "optimization_bound": args.optimization_bound,
        "optimization_l2": args.optimization_l2,
        "environment": capture_environment(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")


def update_claims_audit(summary: pd.DataFrame, path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    marker_prefix = "| Validation loss optimization over log-scales gives only a descriptive, statistically unsupported improvement over raw monomial scaling in the current MNIST MLP run. |"
    marker = (
        "| Validation loss optimization over log-scales gives only a descriptive, statistically unsupported improvement over raw monomial scaling in the current MNIST MLP run. "
        "| Supported descriptive | `reports/csv/improved_validated_ladder_merge_summary.csv` reports `optimized_monomial_scale_vs_monomial_scale` paired mean accuracy delta `0.0009`, CI `[-0.0030, 0.0053]`, and fixed-setting positives in only two of six settings. |"
    )
    managed_prefixes = [
        "| Fashion-MNIST MLP validation tests whether the improved MNIST selector result persists beyond MNIST. |",
        "| Fashion-MNIST provides an additional greedy-soup boundary check for the improved selector. |",
        "| Fashion-MNIST residuals remain mostly noncentral/non-Brauer under the tested structure groups. |",
        "| Fashion-MNIST residuals do not support a mostly noncentral final-decision claim in this run. |",
        "| Fashion-MNIST residual taxonomy remains useful beyond MNIST without claiming Brauer classes. |",
        "| Fashion-MNIST residuals remain non-Brauer under the tested structure groups. |",
        "| The Fashion-MNIST residual detector remains conservative about finite-index/Brauer structure. |",
    ]
    cleaned_lines = []
    for line in text.splitlines():
        if any(line.startswith(prefix) for prefix in managed_prefixes):
            continue
        cleaned_lines.append(marker if line.startswith(marker_prefix) else line)
    text = "\n".join(cleaned_lines)
    claims = summary[summary["summary_type"] == "claim_decision"]
    decisions = {row["claim"]: row for row in claims.to_dict("records")}
    selector = decisions.get("fashion_selector_over_internal_c2m3", {})
    greedy = decisions.get("fashion_greedy_soup_boundary_case", {})
    noncentral = decisions.get("fashion_residuals_mostly_noncentral", {})
    non_brauer = decisions.get("fashion_residuals_non_brauer_no_central_projective", {})
    taxonomy = decisions.get("fashion_residual_taxonomy_useful_beyond_mnist", {})
    detector = decisions.get("fashion_detector_conservative_no_finite_index_hallucination", {})
    rows = [
        f"| Fashion-MNIST MLP validation tests whether the improved MNIST selector result persists beyond MNIST. | {selector.get('claim_decision', 'Evaluated')} | `reports/fashion_mnist_improved_ladder_report.md` and `reports/csv/fashion_mnist_improved_ladder_summary.csv` record `{selector.get('claim_reason', 'paired comparison results')}`. |",
        f"| Fashion-MNIST provides an additional greedy-soup boundary check for the improved selector. | {greedy.get('claim_decision', 'Evaluated')} | `reports/csv/fashion_mnist_improved_ladder_summary.csv` records `{greedy.get('claim_reason', 'paired comparison against greedy soup')}`. |",
        f"| Fashion-MNIST residuals do not support a mostly noncentral final-decision claim in this run. | {noncentral.get('claim_decision', 'Evaluated')} | `reports/fashion_mnist_improved_ladder_report.md` records `{noncentral.get('claim_reason', 'literal noncentral final-decision fraction')}`. |",
        f"| Fashion-MNIST residuals remain non-Brauer under the tested structure groups. | {non_brauer.get('claim_decision', 'Evaluated')} | `reports/fashion_mnist_improved_ladder_report.md` records `{non_brauer.get('claim_reason', 'central/projective and finite-index candidate fractions')}`. |",
        f"| Fashion-MNIST residual taxonomy remains useful beyond MNIST without claiming Brauer classes. | {taxonomy.get('claim_decision', 'Evaluated')} | `reports/fashion_mnist_improved_ladder_report.md` records `{taxonomy.get('claim_reason', 'taxonomy decision split')}`. |",
        f"| The Fashion-MNIST residual detector remains conservative about finite-index/Brauer structure. | {detector.get('claim_decision', 'Evaluated')} | `reports/csv/fashion_mnist_improved_ladder_summary.csv` records `{detector.get('claim_reason', 'finite-index and central/projective candidate fractions')}`. |",
    ]
    replacement = "\n".join([marker, *rows])
    if marker in text:
        text = text.replace(marker, replacement, 1)
    else:
        text = text.replace("\n## Not Yet Supported", "\n" + "\n".join(rows) + "\n\n## Not Yet Supported", 1)

    artifact_marker = "| `reports/configs/improved_validated_ladder_merge_config.json` | Saved configuration and environment metadata for the improved validated ladder run. |"
    artifacts = [
        "| `experiments/fashion_mnist_improved_ladder.py` | Fashion-MNIST MLP benchmark for improved validated ladder selector, monomial scaling variants, soup baselines, and residual taxonomy. |",
        "| `reports/fashion_mnist_improved_ladder_report.md` | Fashion-MNIST validation report with paired method comparisons, selector behavior, residual taxonomy, plots, and claim decisions. |",
        "| `reports/csv/fashion_mnist_improved_ladder.csv` | Per-setting Fashion-MNIST improved ladder rows. |",
        "| `reports/csv/fashion_mnist_improved_ladder_summary.csv` | Fashion-MNIST method summaries, all paired comparisons, residual taxonomy, correlations, and claim decisions. |",
        "| `reports/plots/fashion_ladder_delta_vs_c2m3.pdf` | Fashion-MNIST test accuracy deltas versus internal C2M3-style synchronization. |",
        "| `reports/plots/fashion_ladder_delta_vs_greedy_soup.pdf` | Fashion-MNIST test accuracy deltas versus ordinary greedy soup. |",
        "| `reports/plots/fashion_residual_taxonomy.pdf` | Fashion-MNIST residual taxonomy fractions across fixed settings. |",
        "| `reports/tables/fashion_ladder_table.tex` | LaTeX summary table for the Fashion-MNIST improved ladder benchmark. |",
        "| `reports/configs/fashion_mnist_improved_ladder_config.json` | Saved configuration and environment metadata for the Fashion-MNIST improved ladder run. |",
    ]
    if "`experiments/fashion_mnist_improved_ladder.py`" not in text:
        if artifact_marker in text:
            text = text.replace(artifact_marker, "\n".join([artifact_marker, *artifacts]), 1)
        else:
            text = text.rstrip() + "\n" + "\n".join(artifacts) + "\n"
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-counts", default="3,4")
    parser.add_argument("--widths", default="32,64,128")
    parser.add_argument("--main-setting", default="4:64")
    parser.add_argument("--main-seeds", default=default_seed_text(5200, 10))
    parser.add_argument("--secondary-seeds", default=default_seed_text(5300, 5))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max-train-samples", type=int, default=12000)
    parser.add_argument("--max-test-samples", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dataset-seed", type=int, default=90210)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--max-order", type=int, default=12)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--alpha-grid", default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.25,1.5")
    parser.add_argument("--tau-grid", default="0.25,0.5,1.0,2.0,inf")
    parser.add_argument("--optimization-steps", type=int, default=30)
    parser.add_argument("--optimization-lr", type=float, default=0.05)
    parser.add_argument("--optimization-bound", type=float, default=1.0)
    parser.add_argument("--optimization-l2", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "reports")
    args = parser.parse_args()
    args.command_string = command_string()

    spec, train_data, test_data = load_dataset(
        "fashion_mnist",
        args.data_dir,
        args.max_train_samples,
        args.max_test_samples,
        args.dataset_seed,
    )
    rows = []
    for n_models, width, seeds, role in setting_seed_plan(args):
        for seed in seeds:
            print(f"running Fashion-MNIST seed={seed} n_models={n_models} width={width} role={role}", flush=True)
            rows.extend(run_setting(args, spec, train_data, test_data, seed, n_models, width, role))

    df = pd.DataFrame(rows)
    summary = summarize(df, args.bootstrap_samples)
    csv_dir = args.reports_dir / "csv"
    plot_dir = args.reports_dir / "plots"
    table_dir = args.reports_dir / "tables"
    config_dir = args.reports_dir / "configs"
    csv_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    results_path = csv_dir / "fashion_mnist_improved_ladder.csv"
    summary_path = csv_dir / "fashion_mnist_improved_ladder_summary.csv"
    delta_c2m3_plot = plot_dir / "fashion_ladder_delta_vs_c2m3.pdf"
    delta_greedy_plot = plot_dir / "fashion_ladder_delta_vs_greedy_soup.pdf"
    residual_plot = plot_dir / "fashion_residual_taxonomy.pdf"
    table_path = table_dir / "fashion_ladder_table.tex"
    report_path = args.reports_dir / "fashion_mnist_improved_ladder_report.md"
    config_path = config_dir / "fashion_mnist_improved_ladder_config.json"

    df.to_csv(results_path, index=False)
    summary.to_csv(summary_path, index=False)
    write_delta_plot(
        df,
        delta_c2m3_plot,
        "accuracy_delta_vs_c2m3",
        "Fashion-MNIST Delta vs Internal C2M3-Style Synchronization",
        "Test accuracy delta vs C2M3",
    )
    write_delta_plot(
        df,
        delta_greedy_plot,
        "accuracy_delta_vs_greedy_soup",
        "Fashion-MNIST Delta vs Greedy Soup",
        "Test accuracy delta vs greedy soup",
    )
    write_residual_taxonomy_plot(summary, residual_plot)
    write_latex_table(summary, table_path)
    write_report(args, df, summary, delta_c2m3_plot, delta_greedy_plot, residual_plot, table_path, report_path)
    write_config(args, config_path)
    update_claims_audit(summary, args.reports_dir / "claims_audit.md")

    for path in [results_path, summary_path, delta_c2m3_plot, delta_greedy_plot, residual_plot, table_path, report_path, config_path]:
        print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
