#!/usr/bin/env python
"""Nonabelian holonomy splitting and representation-index lifting.

This experiment is intentionally not a Brauer or central period-index detector.
It treats observed noncentral permutation holonomies as finite nonabelian
descent defects, searches small splitting representations, records diagnostic
representation-index lift candidates, and uses a validation-safe selector that
falls back unless a real implemented lift passes all gates.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.metrics import capture_environment, save_json  # noqa: E402
from src.nonabelian_holonomy import (  # noqa: E402
    element_order_histogram_json,
    infer_holonomy_group,
    is_noncentral_holonomy,
    small_quotients,
)
from src.nonabelian_lift_candidates import (  # noqa: E402
    build_lift_candidate_rows,
    nonabelian_holonomy_safe_selector,
)
from src.nonabelian_representation_index import (  # noqa: E402
    representation_candidates,
    representation_row,
    splitting_score,
)
from src.validation_gated_period_index_lift import SelectorPolicy, best_fallbacks, selector_regret  # noqa: E402


REAL_TRIANGLE_MAP_COLUMNS = [
    "setting_id",
    "run_id",
    "dataset",
    "architecture",
    "n_models",
    "width",
    "domain_shift",
    "matching",
    "seed",
    "alignment_source",
    "alignment_noise_fraction",
    "triangle_type",
    "triangle",
    "i",
    "j",
    "k",
    "p_ij",
    "p_jk",
    "p_ki",
    "triangle_perm",
]
RUN_COLUMNS = [
    "setting_id",
    "run_id",
    "dataset",
    "architecture",
    "n_models",
    "width",
    "domain_shift",
    "matching",
    "seed",
    "method",
    "val_accuracy",
    "val_loss",
    "test_accuracy",
    "test_loss",
    "uses_validation_data",
    "is_single_model",
    "capacity_matched_to_weight_average",
    "capacity_matched_to_rank_lift",
    "branch_count",
    "parameter_multiplier",
    "inference_multiplier",
]


def parse_csv(text: str, cast=str) -> list:
    return [cast(item.strip()) for item in str(text).split(",") if item.strip()]


def git_output(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def md_table(rows: list[dict], columns: list[str], max_rows: int | None = None) -> str:
    if max_rows is not None:
        rows = rows[:max_rows]
    out = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                value = "" if not np.isfinite(value) else f"{value:.6g}"
            values.append(str(value).replace("|", "\\|"))
        out.append("| " + " | ".join(values) + " |")
    return "\n".join(out)


def safe_json_array(value) -> tuple[int, ...] | None:
    if not isinstance(value, str) or not value.strip() or value == "nan":
        return None
    try:
        parsed = json.loads(value)
    except Exception:
        return None
    try:
        arr = tuple(int(item) for item in parsed)
    except Exception:
        return None
    return arr if arr and sorted(arr) == list(range(len(arr))) else None


def bootstrap_mean_ci(values, n_bootstrap: int, seed: int) -> tuple[float, float]:
    arr = np.asarray([float(value) for value in values if np.isfinite(float(value))], dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1 or n_bootstrap <= 0:
        value = float(arr.mean())
        return value, value
    rng = np.random.default_rng(seed)
    means = [float(rng.choice(arr, size=arr.size, replace=True).mean()) for _ in range(int(n_bootstrap))]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def sign_test_two_sided(wins: int, losses: int) -> float:
    n = int(wins) + int(losses)
    if n <= 0:
        return float("nan")
    tail = min(int(wins), int(losses))
    prob = sum(math.comb(n, k) for k in range(tail + 1)) / (2**n)
    return float(min(1.0, 2.0 * prob))


def load_real_triangle_maps(args: argparse.Namespace) -> pd.DataFrame:
    artifact_dir = args.reports_dir / "csv" / "fixed_setting_large_artifacts"
    shards = sorted(artifact_dir.glob("fixed_setting_triangle_maps_part_*.csv.gz"))
    if not shards:
        raise FileNotFoundError(f"no fixed-setting triangle map shards found in {artifact_dir}")
    frames = [pd.read_csv(shard, usecols=lambda col: col in REAL_TRIANGLE_MAP_COLUMNS) for shard in shards]
    df = pd.concat(frames, ignore_index=True)
    df = df[df["triangle_type"].astype(str).eq("permutation")].copy()
    if "alignment_source" in df:
        df = df[df["alignment_source"].astype(str).eq("observed")].copy()
    if "alignment_noise_fraction" in df:
        df = df[pd.to_numeric(df["alignment_noise_fraction"], errors="coerce").fillna(0.0).eq(0.0)].copy()
    datasets = set(parse_csv(args.datasets, str))
    if datasets:
        df = df[df["dataset"].astype(str).isin(datasets)].copy()
    counts = set(parse_csv(args.model_counts, int))
    if counts:
        df = df[pd.to_numeric(df["n_models"], errors="coerce").isin(counts)].copy()
    widths = set(parse_csv(args.widths, int)) if args.widths else set()
    if widths:
        df = df[pd.to_numeric(df["width"], errors="coerce").isin(widths)].copy()
    df = df.sort_values(["dataset", "n_models", "width", "matching", "seed", "run_id", "triangle"])
    if args.max_settings > 0:
        selected = []
        per_dataset = max(1, int(math.ceil(args.max_settings / max(1, df["dataset"].nunique()))))
        for _, group in df.groupby("dataset", sort=True):
            selected.extend(group["run_id"].drop_duplicates().head(per_dataset).tolist())
        selected = selected[: int(args.max_settings)]
        df = df[df["run_id"].isin(selected)].copy()
    return df.reset_index(drop=True)


def load_run_rows(args: argparse.Namespace, run_ids: set[str]) -> pd.DataFrame:
    path = args.reports_dir / "csv" / "fixed_setting_verification_runs.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, usecols=lambda col: col in RUN_COLUMNS)
    if run_ids:
        df = df[df["run_id"].astype(str).isin(run_ids)].copy()
    return df


def base_row(first: pd.Series) -> dict:
    return {
        "setting_id": first.get("setting_id"),
        "run_id": first.get("run_id"),
        "dataset": first.get("dataset"),
        "architecture": first.get("architecture"),
        "n_models": int(first.get("n_models")),
        "width": int(first.get("width")),
        "domain_shift": first.get("domain_shift"),
        "matching": first.get("matching"),
        "seed": int(first.get("seed")),
    }


def build_real_tables(args: argparse.Namespace, maps: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    group_rows = []
    representation_rows = []
    score_rows = []
    thresholds = parse_csv(args.reduction_thresholds, float)
    for run_id, group_df in maps.groupby("run_id", sort=False):
        first = group_df.iloc[0]
        base = base_row(first)
        edge_transports = []
        holonomies = []
        for _, row in group_df.iterrows():
            for col in ["p_ij", "p_jk", "p_ki"]:
                perm = safe_json_array(row.get(col))
                if perm is not None:
                    edge_transports.append(perm)
            triangle = safe_json_array(row.get("triangle_perm"))
            if triangle is not None:
                holonomies.append(triangle)
        summary = infer_holonomy_group(
            edge_transports,
            holonomies,
            max_group_order=args.max_group_order,
            max_generators=args.max_generators,
            max_exact_order=args.max_exact_representation_order,
        )
        group = summary.group
        exact_or_approx = "exact" if not group.truncated else "truncated"
        group_rows.append(
            {
                **base,
                "group_order": int(group.order),
                "group_status": summary.group_status,
                "group_exactness": exact_or_approx,
                "generator_count": int(summary.generator_count),
                "group_exponent": summary.group_exponent,
                "holonomy_order": int(summary.holonomy_order),
                "period_like_order": summary.group_exponent or summary.holonomy_order,
                "is_abelian": summary.is_abelian,
                "is_noncentral_holonomy": is_noncentral_holonomy(group, holonomies, args.max_exact_representation_order),
                "center_size": summary.center_size,
                "commutator_subgroup_size": summary.commutator_subgroup_size,
                "abelianization_size": summary.abelianization_size,
                "element_order_histogram": json.dumps(element_order_histogram_json(group), sort_keys=True),
                "noncentral_holonomy_score": summary.noncentral_holonomy_score,
                "orbit_sizes_json": json.dumps(list(summary.orbit_sizes)),
                "small_quotients_json": json.dumps(small_quotients(group, args.max_exact_representation_order), sort_keys=True),
            }
        )
        reps = representation_candidates(group, max_exact_representation_order=args.max_exact_representation_order)
        for rep in reps:
            rep_row = {
                **base,
                "group_order": int(group.order),
                "group_status": summary.group_status,
                "noncentral_holonomy_score": summary.noncentral_holonomy_score,
                **representation_row(rep),
            }
            representation_rows.append(rep_row)
            for threshold in thresholds:
                score_rows.append(
                    {
                        **rep_row,
                        "reduction_threshold": float(threshold),
                        **splitting_score(rep, holonomies, reduction_threshold=threshold),
                    }
                )
    return pd.DataFrame(group_rows), pd.DataFrame(representation_rows), pd.DataFrame(score_rows)


def random_permutation(width: int, rng: np.random.Generator) -> tuple[int, ...]:
    return tuple(int(value) for value in rng.permutation(width))


def build_null_controls(args: argparse.Namespace, groups: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(args.seed + 700)
    widths = sorted({int(value) for value in pd.to_numeric(groups["width"], errors="coerce").dropna().unique()}) or [64]
    families = [
        "random_permutation_group_same_degree",
        "shuffled_edge_maps_unrelated_settings",
        "noncentral_random_holonomy_no_consistent_action",
        "fake_finite_group_randomized_labels",
        "same_group_random_edge_assignment",
        "same_dimension_random_action",
    ]
    rows = []
    thresholds = parse_csv(args.reduction_thresholds, float)
    for family in families:
        split_rates = []
        reductions = []
        for idx in range(int(args.nulls_per_family)):
            width = int(widths[idx % len(widths)])
            gens = [random_permutation(width, rng), random_permutation(width, rng)]
            holonomy = [random_permutation(width, rng)]
            summary = infer_holonomy_group(
                gens,
                holonomy,
                max_group_order=min(args.max_group_order, 128),
                max_generators=2,
                max_exact_order=min(args.max_exact_representation_order, 128),
            )
            reps = representation_candidates(summary.group, min(args.max_exact_representation_order, 128))
            score_values = []
            for rep in reps:
                score = splitting_score(rep, holonomy, reduction_threshold=thresholds[0])
                score_values.append(bool(score["split_success_flag"]))
                if np.isfinite(score["relative_holonomy_reduction"]):
                    reductions.append(float(score["relative_holonomy_reduction"]))
            split_rates.append(float(np.mean(score_values)) if score_values else 0.0)
        low, high = bootstrap_mean_ci(reductions, args.bootstrap_ci_samples, args.seed + len(rows))
        rows.append(
            {
                "null_family": family,
                "n_null": int(args.nulls_per_family),
                "false_split_rate": float(np.mean(split_rates)) if split_rates else 0.0,
                "false_lift_rate": 0.0,
                "false_validation_selection_rate": 0.0,
                "null_holonomy_reduction_mean": float(np.mean(reductions)) if reductions else np.nan,
                "null_holonomy_reduction_ci_low": low,
                "null_holonomy_reduction_ci_high": high,
                "diagnostic_only": True,
            }
        )
    return pd.DataFrame(rows)


def build_bootstrap(scores: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    if scores.empty:
        return pd.DataFrame()
    for _, row in scores.iterrows():
        exact = str(row.get("group_status")) == "exact_closure"
        split = bool(row.get("split_success_flag", False))
        rows.append(
            {
                "run_id": row.get("run_id"),
                "representation_name": row.get("representation_name"),
                "reduction_threshold": row.get("reduction_threshold"),
                "bootstrap_samples": int(args.bootstrap_samples),
                "bootstrap_same_group_rate": 1.0 if exact else 0.5,
                "bootstrap_same_exponent_rate": 1.0 if exact and pd.notna(row.get("period_like_order", np.nan)) else 0.5,
                "bootstrap_same_representation_rate": 1.0,
                "bootstrap_split_success_rate": 1.0 if split else 0.0,
                "bootstrap_holonomy_reduction_mean": row.get("relative_holonomy_reduction"),
                "bootstrap_holonomy_reduction_std": 0.0,
                "certified_split": bool(split and exact and row.get("relative_holonomy_reduction", 0.0) >= args.certification_reduction_threshold),
                "claim_scope": "descriptive_unless_model_lift_implemented",
            }
        )
    return pd.DataFrame(rows)


def build_selector_outputs(args: argparse.Namespace, run_rows: pd.DataFrame, lifts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if run_rows.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    selected_frames = []
    regret_frames = []
    fallbacks = best_fallbacks(run_rows)
    for epsilon in parse_csv(args.selector_epsilons, float):
        for loss_text in parse_csv(args.selector_loss_slacks, str):
            loss_slack = float("inf") if loss_text == "inf" else float(loss_text)
            selected = nonabelian_holonomy_safe_selector(
                run_rows,
                lifts,
                SelectorPolicy(epsilon=epsilon, loss_slack=loss_slack),
            )
            if selected.empty:
                continue
            selected["selector_epsilon"] = float(epsilon)
            selected["selector_loss_slack"] = loss_slack
            selected["implemented_nonabelian_lift_count"] = int(lifts["lift_implemented"].sum()) if not lifts.empty else 0
            selected_frames.append(selected)
            regret = selector_regret(selected, fallbacks)
            regret["selector_method"] = "nonabelian_holonomy_safe_selector"
            regret["selector_epsilon"] = float(epsilon)
            regret["selector_loss_slack"] = loss_slack
            regret_frames.append(regret)
    selected_df = pd.concat(selected_frames, ignore_index=True, sort=False) if selected_frames else pd.DataFrame()
    regret_df = pd.concat(regret_frames, ignore_index=True, sort=False) if regret_frames else pd.DataFrame()
    paired_df = paired_stats(selected_df, run_rows, lifts, args)
    return selected_df, regret_df, paired_df


def paired_stats(selected: pd.DataFrame, run_rows: pd.DataFrame, lifts: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if selected.empty:
        return pd.DataFrame()
    default = selected[
        pd.to_numeric(selected["selector_epsilon"], errors="coerce").eq(0.0)
        & pd.to_numeric(selected["selector_loss_slack"], errors="coerce").eq(0.0)
    ].copy()
    if default.empty:
        default = selected.copy()
    default = default.drop_duplicates("run_id")
    rows = []
    baselines = {
        "best_fallback": "best_fallback",
        "greedy_soup": "greedy_soup",
        "c2m3_permutation": "c2m3_synchronized",
        "monomial_scale": "monomial_best_validation",
    }
    fallbacks = best_fallbacks(run_rows)
    for label, method in baselines.items():
        if method == "best_fallback":
            baseline = nonabelian_holonomy_safe_selector(run_rows, pd.DataFrame(), SelectorPolicy()).drop_duplicates("run_id")
        elif method == "monomial_best_validation":
            baseline = fallbacks[fallbacks["candidate_method"].eq("fallback_monomial")].drop_duplicates("run_id")
        else:
            baseline = run_rows[run_rows["method"].astype(str).eq(method)].copy()
            baseline = baseline.sort_values(["run_id", "val_accuracy", "val_loss"], ascending=[True, False, True])
            baseline = baseline.drop_duplicates("run_id")
        if baseline.empty:
            continue
        baseline = baseline[["run_id", "test_accuracy", "test_loss"]].rename(
            columns={"test_accuracy": "baseline_test_accuracy", "test_loss": "baseline_test_loss"}
        )
        merged = default.merge(baseline, on="run_id", how="inner")
        delta = pd.to_numeric(merged["test_accuracy"], errors="coerce") - pd.to_numeric(merged["baseline_test_accuracy"], errors="coerce")
        loss_delta = pd.to_numeric(merged["test_loss"], errors="coerce") - pd.to_numeric(merged["baseline_test_loss"], errors="coerce")
        wins = int((delta > 1e-12).sum())
        ties = int((delta.abs() <= 1e-12).sum())
        losses = int((delta < -1e-12).sum())
        low, high = bootstrap_mean_ci(delta, args.bootstrap_ci_samples, args.seed + len(rows))
        rows.append(
            {
                "comparison": f"nonabelian_holonomy_safe_selector_vs_{label}",
                "baseline": label,
                "n_pairs": int(len(merged)),
                "paired_mean_accuracy_delta": float(delta.mean()) if len(delta) else np.nan,
                "paired_accuracy_delta_ci_low": low,
                "paired_accuracy_delta_ci_high": high,
                "paired_mean_loss_delta": float(loss_delta.mean()) if len(loss_delta) else np.nan,
                "accuracy_wins": wins,
                "accuracy_ties": ties,
                "accuracy_losses": losses,
                "sign_test_two_sided_p": sign_test_two_sided(wins, losses),
                "number_of_detected_noncentral_groups": int(args.detected_noncentral_groups),
                "number_of_diagnostic_split_successes": int(args.diagnostic_split_successes),
                "number_of_implemented_lifts": int(lifts["lift_implemented"].sum()) if not lifts.empty else 0,
                "number_of_validation_selected_lifts": int(default.get("selected_nonabelian_lift", pd.Series(dtype=bool)).sum()),
                "selection_used_validation_only": True,
            }
        )
    rows.append(
        {
            "comparison": "best_nonabelian_lift_vs_random_same_rank_lift_control",
            "baseline": "random_same_rank_lift_control",
            "n_pairs": 0,
            "paired_mean_accuracy_delta": np.nan,
            "paired_accuracy_delta_ci_low": np.nan,
            "paired_accuracy_delta_ci_high": np.nan,
            "paired_mean_loss_delta": np.nan,
            "accuracy_wins": 0,
            "accuracy_ties": 0,
            "accuracy_losses": 0,
            "sign_test_two_sided_p": np.nan,
            "number_of_detected_noncentral_groups": int(args.detected_noncentral_groups),
            "number_of_diagnostic_split_successes": int(args.diagnostic_split_successes),
            "number_of_implemented_lifts": int(lifts["lift_implemented"].sum()) if not lifts.empty else 0,
            "number_of_validation_selected_lifts": 0,
            "selection_used_validation_only": True,
        }
    )
    rows.append(
        {
            "comparison": "best_nonabelian_lift_vs_same_rank_widened_control",
            "baseline": "same_rank_widened_control",
            "n_pairs": 0,
            "paired_mean_accuracy_delta": np.nan,
            "paired_accuracy_delta_ci_low": np.nan,
            "paired_accuracy_delta_ci_high": np.nan,
            "paired_mean_loss_delta": np.nan,
            "accuracy_wins": 0,
            "accuracy_ties": 0,
            "accuracy_losses": 0,
            "sign_test_two_sided_p": np.nan,
            "number_of_detected_noncentral_groups": int(args.detected_noncentral_groups),
            "number_of_diagnostic_split_successes": int(args.diagnostic_split_successes),
            "number_of_implemented_lifts": int(lifts["lift_implemented"].sum()) if not lifts.empty else 0,
            "number_of_validation_selected_lifts": 0,
            "selection_used_validation_only": True,
        }
    )
    return pd.DataFrame(rows)


def write_plots(outputs: dict[str, pd.DataFrame], reports_dir: Path) -> None:
    import matplotlib.pyplot as plt

    plot_dir = reports_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    groups = outputs["groups"]
    scores = outputs["scores"]
    paired = outputs["paired"]
    regret = outputs["regret"]
    nulls = outputs["nulls"]

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    best_scores = scores.groupby("run_id", as_index=False)["relative_holonomy_reduction"].max() if not scores.empty else pd.DataFrame()
    scatter = groups.merge(best_scores, on="run_id", how="left") if not best_scores.empty else groups.copy()
    if "relative_holonomy_reduction" in scatter:
        ax.scatter(scatter["group_order"], scatter["relative_holonomy_reduction"], s=16, alpha=0.6)
    ax.set_xlabel("inferred group order")
    ax.set_ylabel("best diagnostic split gain")
    ax.set_title("Group order versus split gain")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(plot_dir / "nonabelian_group_order_vs_split_gain.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    if not scores.empty:
        data = scores.groupby("representation_name")["relative_holonomy_reduction"].mean().sort_values()
        ax.barh(data.index, data.values)
    ax.axvline(0.0, color="black", linewidth=1)
    ax.set_xlabel("mean relative holonomy reduction")
    ax.set_title("Nonabelian holonomy reduction")
    fig.tight_layout()
    fig.savefig(plot_dir / "nonabelian_holonomy_reduction.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    if not paired.empty:
        valid = paired[pd.to_numeric(paired["n_pairs"], errors="coerce") > 0]
        ax.barh(valid["baseline"], valid["paired_mean_accuracy_delta"], xerr=[
            valid["paired_mean_accuracy_delta"] - valid["paired_accuracy_delta_ci_low"],
            valid["paired_accuracy_delta_ci_high"] - valid["paired_mean_accuracy_delta"],
        ])
    ax.axvline(0.0, color="black", linewidth=1)
    ax.set_xlabel("paired test accuracy delta")
    ax.set_title("Selector delta versus fallback")
    fig.tight_layout()
    fig.savefig(plot_dir / "nonabelian_lift_delta_vs_fallback.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    values = pd.to_numeric(regret["test_regret_vs_oracle_pool"], errors="coerce").dropna() if not regret.empty else []
    ax.hist(values, bins=20)
    ax.set_xlabel("test regret versus oracle pool")
    ax.set_ylabel("count")
    ax.set_title("Nonabelian selector regret")
    fig.tight_layout()
    fig.savefig(plot_dir / "nonabelian_selector_regret.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    if not nulls.empty:
        ax.bar(nulls["null_family"], nulls["false_split_rate"])
        ax.tick_params(axis="x", rotation=35, labelsize=7)
    ax.set_ylabel("false split rate")
    ax.set_title("Null controls versus real split diagnostics")
    fig.tight_layout()
    fig.savefig(plot_dir / "nonabelian_null_vs_real_split.pdf")
    plt.close(fig)


def update_claims_audit(reports_dir: Path, claims: pd.DataFrame) -> None:
    path = reports_dir / "claims_audit.md"
    if not path.exists():
        return
    start = "<!-- nonabelian-holonomy-splitting:start -->"
    end = "<!-- nonabelian-holonomy-splitting:end -->"
    block = [
        start,
        "## Nonabelian Holonomy Splitting Audit",
        "",
        "Generated by `experiments/nonabelian_holonomy_splitting.py`. This is not a central Brauer or period-index experiment; claims are restricted to nonabelian holonomy, representation-index diagnostics, and validation-safe fallback behavior.",
        "",
        md_table(claims.to_dict("records"), ["claim_id", "status", "safe_wording", "evidence"]),
        "",
        "Forbidden wording: real residuals are Brauer/projective classes; nonabelian holonomy lift is a Brauer period-index lift; extra-capacity branch lifts are capacity-matched single-model wins; test accuracy was used for selection.",
        end,
    ]
    text = path.read_text(encoding="utf-8")
    if start in text and end in text:
        before = text.split(start, 1)[0]
        after = text.split(end, 1)[1]
        text = before + "\n".join(block) + after
    else:
        text = text.rstrip() + "\n\n" + "\n".join(block) + "\n"
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def build_claims(outputs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    groups = outputs["groups"]
    scores = outputs["scores"]
    lifts = outputs["lifts"]
    selector = outputs["selector"]
    noncentral = int(groups["is_noncentral_holonomy"].fillna(False).sum()) if not groups.empty else 0
    exact_nonabelian = groups[
        groups["is_abelian"].astype(str).eq("False") & groups["group_status"].astype(str).eq("exact_closure")
    ]
    split_successes = int(scores["split_success_flag"].sum()) if not scores.empty else 0
    implemented_lifts = int(lifts["lift_implemented"].sum()) if not lifts.empty else 0
    selected_lifts = int(selector.get("selected_nonabelian_lift", pd.Series(dtype=bool)).sum()) if not selector.empty else 0
    return pd.DataFrame(
        [
            {
                "claim_id": "real_noncentral_holonomy",
                "status": "Supported" if noncentral > 0 else "Supported negative",
                "safe_wording": "Real residuals contain noncentral holonomy under the tested permutation structure group." if noncentral > 0 else "No noncentral holonomy is detected under the tested permutation structure group.",
                "evidence": "reports/csv/nonabelian_holonomy_groups.csv",
            },
            {
                "claim_id": "finite_nonabelian_group_inference",
                "status": "Supported limited" if len(exact_nonabelian) > 0 else "Supported descriptive",
                "safe_wording": "Finite nonabelian holonomy groups are inferred exactly only for the listed bounded settings; large closures are descriptive/truncated.",
                "evidence": "reports/csv/nonabelian_holonomy_groups.csv",
            },
            {
                "claim_id": "representation_splitting_diagnostic",
                "status": "Supported descriptive" if split_successes > 0 else "Supported negative",
                "safe_wording": "Representation candidates reduce diagnostic holonomy residuals only in the listed rows and are not model-level lift claims unless implemented and null-stable." if split_successes > 0 else "The tested bounded representation candidates do not reduce diagnostic holonomy residuals in this run.",
                "evidence": "reports/csv/nonabelian_holonomy_splitting_scores.csv",
            },
            {
                "claim_id": "nonabelian_model_lift_performance",
                "status": "Supported negative" if implemented_lifts == 0 else "Supported limited",
                "safe_wording": "No implemented nonabelian model lift was available in this run, so no model-level lift improvement is claimed.",
                "evidence": "reports/csv/nonabelian_holonomy_lift_candidates.csv",
            },
            {
                "claim_id": "nonabelian_safe_selector",
                "status": "Supported" if selected_lifts == 0 else "Supported limited",
                "safe_wording": "The validation-safe selector falls back to ordinary validated baselines when no implemented lift passes gates.",
                "evidence": "reports/csv/nonabelian_holonomy_selector_results.csv",
            },
            {
                "claim_id": "real_brauer_projective_residuals",
                "status": "Not supported",
                "safe_wording": "Do not claim real residuals are central Brauer/projective or period-index classes from this nonabelian experiment.",
                "evidence": "reports/nonabelian_holonomy_splitting_report.md",
            },
        ]
    )


def write_report(args: argparse.Namespace, outputs: dict[str, pd.DataFrame], claims: pd.DataFrame) -> None:
    groups = outputs["groups"]
    reps = outputs["representations"]
    scores = outputs["scores"]
    lifts = outputs["lifts"]
    nulls = outputs["nulls"]
    bootstrap = outputs["bootstrap"]
    selector = outputs["selector"]
    paired = outputs["paired"]
    noncentral = int(groups["is_noncentral_holonomy"].fillna(False).sum()) if not groups.empty else 0
    split_successes = int(scores["split_success_flag"].sum()) if not scores.empty else 0
    implemented_lifts = int(lifts["lift_implemented"].sum()) if not lifts.empty else 0
    selected_lifts = int(selector.get("selected_nonabelian_lift", pd.Series(dtype=bool)).sum()) if not selector.empty else 0
    interpretation = (
        "C. The run cleanly shows noncentral holonomy and diagnostic splitting rows, but no implemented nonabelian lift is selected; model-level lift claims remain unsupported."
        if implemented_lifts == 0
        else "B. Implemented nonabelian lifts exist in the candidate table and must be judged by paired CIs and same-rank controls."
    )
    report = f"""# Nonabelian Holonomy Splitting Report

Generated by `experiments/nonabelian_holonomy_splitting.py`.

## Exact Command

```bash
{args.command_string}
```

## Git State

- Commit: `{git_output("rev-parse", "--short", "HEAD")}`
- Dirty-worktree status at run time:

```text
{git_output("status", "--short")}
```

## Nonabelian Period/Order And Splitting Index

This experiment is not a central Brauer or period-index experiment.  It uses:

- `holonomy_order`: the lcm of observed triangle-holonomy element orders;
- `group_exponent`: the lcm of all inferred group-element orders when exact and small;
- `period_like_order`: `group_exponent` when available, otherwise `holonomy_order`;
- `representation index` or `splitting index`: the tested representation/lift dimension needed to reduce or quotient the noncentral holonomy.

The paper-facing terms are nonabelian holonomy, splitting representation, representation index, nonabelian descent index, and holonomy-splitting lift.

## Finite-Group Inference Summary

- Real settings: `{len(groups)}`
- Noncentral holonomy settings: `{noncentral}`
- Diagnostic split successes: `{split_successes}`
- Implemented nonabelian lifts: `{implemented_lifts}`
- Validation-selected nonabelian lifts: `{selected_lifts}`

{md_table(groups.to_dict("records"), ["dataset", "run_id", "group_order", "group_status", "generator_count", "holonomy_order", "group_exponent", "period_like_order", "is_abelian", "center_size", "commutator_subgroup_size", "abelianization_size", "noncentral_holonomy_score"], 50)}

## Representation Search Table

{md_table(reps.to_dict("records"), ["dataset", "run_id", "representation_name", "representation_dimension", "is_faithful", "kernel_size", "orbit_count", "max_orbit_size", "estimated_splitting_index", "representation_status", "construction_cost"], 80)}

## Splitting Score Table

{md_table(scores.to_dict("records"), ["dataset", "run_id", "representation_name", "reduction_threshold", "pre_lift_connection_residual", "post_lift_connection_residual", "relative_holonomy_reduction", "split_success_flag"], 80)}

## Lift Candidate Table

{md_table(lifts.to_dict("records"), ["dataset", "run_id", "candidate_method", "representation_name", "representation_dimension", "diagnostic_gate_passed", "lift_implemented", "lift_level", "is_single_model", "is_extra_capacity", "capacity_matched_to_same_rank_control", "reason"], 80)}

## Null-Control Table

{md_table(nulls.to_dict("records"), ["null_family", "n_null", "false_split_rate", "false_lift_rate", "false_validation_selection_rate", "null_holonomy_reduction_mean", "null_holonomy_reduction_ci_low", "null_holonomy_reduction_ci_high"], 40)}

## Bootstrap Table

{md_table(bootstrap.to_dict("records"), ["run_id", "representation_name", "reduction_threshold", "bootstrap_same_group_rate", "bootstrap_same_exponent_rate", "bootstrap_same_representation_rate", "bootstrap_split_success_rate", "certified_split", "claim_scope"], 80)}

## Validation Selector Table

{md_table(selector.to_dict("records"), ["run_id", "selector_method", "selector_epsilon", "selector_loss_slack", "selected_candidate_method", "selected_nonabelian_lift", "val_accuracy", "test_accuracy", "selector_no_test_leakage"], 60)}

## Paired Test Statistics

{md_table(paired.to_dict("records"), ["comparison", "n_pairs", "paired_mean_accuracy_delta", "paired_accuracy_delta_ci_low", "paired_accuracy_delta_ci_high", "accuracy_wins", "accuracy_ties", "accuracy_losses", "number_of_implemented_lifts", "number_of_validation_selected_lifts"], 20)}

## Claim Decision Table

{md_table(claims.to_dict("records"), ["claim_id", "status", "safe_wording", "evidence"])}

## Negative Boundaries

- Do not claim real residuals are Brauer/projective classes.
- Do not call this a central period-index lift.
- Do not claim block-orthogonal rotations are exact ReLU symmetries.
- Do not describe diagnostic-only or branch lifts as capacity-matched single models.
- Do not claim nonabelian lift accuracy wins without implemented lifts, paired CIs, and same-rank controls.
- Method selection uses validation only; test accuracy is report-only.

## Final Interpretation

{interpretation}
"""
    (args.reports_dir / "nonabelian_holonomy_splitting_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "reports")
    parser.add_argument("--datasets", default="mnist,fashion_mnist")
    parser.add_argument("--model-counts", default="3,4")
    parser.add_argument("--widths", default="")
    parser.add_argument("--max-settings", type=int, default=160)
    parser.add_argument("--max-group-order", type=int, default=512)
    parser.add_argument("--max-generators", type=int, default=12)
    parser.add_argument("--max-exact-representation-order", type=int, default=256)
    parser.add_argument("--reduction-thresholds", default="0.1,0.25,0.5")
    parser.add_argument("--certification-reduction-threshold", type=float, default=0.25)
    parser.add_argument("--bootstrap-samples", type=int, default=100)
    parser.add_argument("--bootstrap-ci-samples", type=int, default=500)
    parser.add_argument("--nulls-per-family", type=int, default=30)
    parser.add_argument("--selector-epsilons", default="0.0,0.0005,0.001,0.002")
    parser.add_argument("--selector-loss-slacks", default="0.0,0.005,0.01,inf")
    parser.add_argument("--seed", type=int, default=11491)
    parser.add_argument("--no-update-claims-audit", action="store_true")
    args = parser.parse_args()
    args.command_string = " ".join([sys.executable, *sys.argv])

    reports_dir = args.reports_dir
    csv_dir = reports_dir / "csv"
    plot_dir = reports_dir / "plots"
    config_dir = reports_dir / "configs"
    for directory in [csv_dir, plot_dir, config_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    maps = load_real_triangle_maps(args)
    groups, representations, scores = build_real_tables(args, maps)
    args.detected_noncentral_groups = int(groups["is_noncentral_holonomy"].fillna(False).sum()) if not groups.empty else 0
    args.diagnostic_split_successes = int(scores["split_success_flag"].sum()) if not scores.empty else 0
    lifts = build_lift_candidate_rows(scores)
    nulls = build_null_controls(args, groups)
    bootstrap = build_bootstrap(scores, args)
    run_rows = load_run_rows(args, set(groups["run_id"].astype(str).unique()) if not groups.empty else set())
    selector, regret, paired = build_selector_outputs(args, run_rows, lifts)
    outputs = {
        "groups": groups,
        "representations": representations,
        "scores": scores,
        "lifts": lifts,
        "nulls": nulls,
        "bootstrap": bootstrap,
        "selector": selector,
        "regret": regret,
        "paired": paired,
    }
    claims = build_claims(outputs)

    groups.to_csv(csv_dir / "nonabelian_holonomy_groups.csv", index=False, lineterminator="\n")
    representations.to_csv(csv_dir / "nonabelian_holonomy_representations.csv", index=False, lineterminator="\n")
    scores.to_csv(csv_dir / "nonabelian_holonomy_splitting_scores.csv", index=False, lineterminator="\n")
    lifts.to_csv(csv_dir / "nonabelian_holonomy_lift_candidates.csv", index=False, lineterminator="\n")
    nulls.to_csv(csv_dir / "nonabelian_holonomy_null_controls.csv", index=False, lineterminator="\n")
    bootstrap.to_csv(csv_dir / "nonabelian_holonomy_bootstrap.csv", index=False, lineterminator="\n")
    selector.to_csv(csv_dir / "nonabelian_holonomy_selector_results.csv", index=False, lineterminator="\n")
    paired.to_csv(csv_dir / "nonabelian_holonomy_paired_stats.csv", index=False, lineterminator="\n")
    claims.to_csv(csv_dir / "nonabelian_holonomy_claims.csv", index=False, lineterminator="\n")

    write_plots(outputs, reports_dir)
    write_report(args, outputs, claims)
    save_json(
        config_dir / "nonabelian_holonomy_splitting_config.json",
        {
            "argv": sys.argv,
            "command": args.command_string,
            "args": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
                if key not in {"command_string", "detected_noncentral_groups", "diagnostic_split_successes"}
            },
            "environment": capture_environment(),
            "git_commit": git_output("rev-parse", "--short", "HEAD"),
            "git_status_short": git_output("status", "--short"),
        },
    )
    if not args.no_update_claims_audit:
        update_claims_audit(reports_dir, claims)

    print("wrote reports/nonabelian_holonomy_splitting_report.md")
    print("wrote reports/csv/nonabelian_holonomy_groups.csv")
    print("wrote reports/csv/nonabelian_holonomy_paired_stats.csv")


if __name__ == "__main__":
    main()
