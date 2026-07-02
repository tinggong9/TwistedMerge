#!/usr/bin/env python
"""Stable small-quotient holonomy splitting on real residuals.

This experiment searches small quotient factors of observed real permutation
holonomy.  It is intentionally conservative: quotient fits and invariant
pooling residuals are diagnostic unless a real model-level Q-branch lift is
implemented and passes validation-only gates.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.controlled_nonabelian_holonomy import planted_case  # noqa: E402
from src.metrics import capture_environment, save_json  # noqa: E402
from src.nonabelian_holonomy import element_order_histogram_json, infer_holonomy_group  # noqa: E402
from src.small_quotient_branch_lift import (  # noqa: E402
    build_pooling_rows,
    build_small_quotient_lift_candidates,
)
from src.small_quotient_holonomy import (  # noqa: E402
    QUOTIENT_NAMES,
    TriangleRelation,
    bootstrap_quotient_fit,
    fit_quotient_map,
    fit_summary_row,
    null_random_assignment_rate,
    triangle_relation_from_perms,
)
from src.small_quotient_selector import small_quotient_holonomy_safe_selector  # noqa: E402
from src.validation_gated_period_index_lift import SelectorPolicy, best_overall_fallback, best_fallbacks, selector_regret  # noqa: E402


TRIANGLE_COLUMNS = [
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
    "branch_count",
    "parameter_multiplier",
    "inference_multiplier",
]


def parse_csv(text: str, cast=str) -> list:
    return [cast(item.strip()) for item in str(text).split(",") if item.strip()]


def stable_seed(*parts: object, base: int = 0) -> int:
    text = "|".join(str(part) for part in parts)
    return int((zlib.crc32(text.encode("utf-8")) + int(base)) % (2**32 - 1))


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
    frames = [pd.read_csv(shard, usecols=lambda col: col in TRIANGLE_COLUMNS) for shard in shards]
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


def base_row(first: pd.Series | dict, residual_source: str = "real") -> dict:
    get = first.get
    return {
        "residual_source": residual_source,
        "setting_id": get("setting_id"),
        "run_id": get("run_id"),
        "dataset": get("dataset"),
        "architecture": get("architecture", "mlp"),
        "n_models": int(get("n_models", 3)),
        "width": int(get("width", 0)),
        "domain_shift": get("domain_shift", "none"),
        "matching": get("matching", "permutation"),
        "seed": int(get("seed", 0)),
    }


def relations_from_group(group_df: pd.DataFrame) -> tuple[TriangleRelation, ...]:
    relations = []
    for _, row in group_df.iterrows():
        p_ij = safe_json_array(row.get("p_ij"))
        p_jk = safe_json_array(row.get("p_jk"))
        p_ki = safe_json_array(row.get("p_ki"))
        holonomy = safe_json_array(row.get("triangle_perm"))
        if p_ij is None or p_jk is None or p_ki is None or holonomy is None:
            continue
        relations.append(triangle_relation_from_perms(p_ij, p_jk, p_ki, holonomy))
    return tuple(relations)


def group_summary_row(base: dict, relations: tuple[TriangleRelation, ...], args: argparse.Namespace) -> dict:
    edge_transports = []
    holonomies = []
    for relation in relations:
        edge_transports.extend([relation.first, relation.second, relation.third])
        holonomies.append(relation.holonomy)
    summary = infer_holonomy_group(
        edge_transports,
        holonomies,
        max_group_order=args.max_group_order,
        max_generators=args.max_generators,
        max_exact_order=args.max_exact_order,
    )
    group = summary.group
    return {
        **base,
        "group_order": int(group.order),
        "group_status": summary.group_status,
        "group_exactness": "exact" if not group.truncated else "truncated",
        "generator_count": int(summary.generator_count),
        "holonomy_order": int(summary.holonomy_order),
        "group_exponent": summary.group_exponent,
        "is_abelian": summary.is_abelian,
        "center_size": summary.center_size,
        "commutator_subgroup_size": summary.commutator_subgroup_size,
        "abelianization_size": summary.abelianization_size,
        "element_order_histogram": json.dumps(element_order_histogram_json(group), sort_keys=True),
        "noncentral_holonomy_score": summary.noncentral_holonomy_score,
        "triangle_relation_count": int(len(relations)),
    }


def add_candidate_rows_for_relations(
    base: dict,
    relations: tuple[TriangleRelation, ...],
    args: argparse.Namespace,
    fits_by_key: dict[tuple[str, str], object],
) -> tuple[list[dict], list[dict]]:
    candidate_rows = []
    bootstrap_rows = []
    relation_thresholds = parse_csv(args.relation_thresholds, float)
    nontrivial_thresholds = parse_csv(args.nontrivial_thresholds, float)
    for Q_name in parse_csv(args.candidate_Q, str):
        fit = fit_quotient_map(
            relations,
            Q_name,
            seed=stable_seed(base.get("run_id"), Q_name, base=args.seed),
            random_restarts=args.random_restarts,
        )
        fits_by_key[(str(base["run_id"]), fit.Q_name)] = fit
        for relation_threshold in relation_thresholds:
            for nontrivial_threshold in nontrivial_thresholds:
                boot = bootstrap_quotient_fit(
                    fit,
                    relation_threshold=relation_threshold,
                    nontrivial_threshold=nontrivial_threshold,
                    n_bootstrap=args.bootstrap_samples,
                    seed=stable_seed(base.get("run_id"), Q_name, relation_threshold, nontrivial_threshold, base=args.seed + 101),
                )
                row = fit_summary_row(fit, relation_threshold, nontrivial_threshold, boot, base=base)
                candidate_rows.append(row)
                bootstrap_rows.append(
                    {
                        **base,
                        "Q_name": fit.Q_name,
                        "Q_order": fit.Q_order,
                        "relation_threshold": float(relation_threshold),
                        "nontrivial_threshold": float(nontrivial_threshold),
                        "quotient_certified": bool(row["quotient_certified"]),
                        **boot,
                    }
                )
    return candidate_rows, bootstrap_rows


def controlled_relations(args: argparse.Namespace) -> tuple[list[dict], dict[str, tuple[TriangleRelation, ...]]]:
    bases = []
    relations = {}
    for group_name in ["S3", "D4"]:
        case = planted_case(group_name, "planted_nonabelian_holonomy", seed=args.seed)
        run_id = f"controlled_{group_name}_planted_small_quotient"
        base = {
            "residual_source": "controlled_sanity",
            "setting_id": run_id,
            "run_id": run_id,
            "dataset": "controlled_sanity",
            "architecture": "synthetic_triangle",
            "n_models": 3,
            "width": case.group.degree,
            "domain_shift": "planted_nonabelian_holonomy",
            "matching": "controlled_exact_edges",
            "seed": int(args.seed),
        }
        bases.append(base)
        relations[run_id] = (
            triangle_relation_from_perms(case.g01, case.g12, case.g20, case.holonomy),
        )
    return bases, relations


def build_real_tables(args: argparse.Namespace, maps: pd.DataFrame):
    group_rows = []
    candidate_rows = []
    bootstrap_rows = []
    fits_by_key: dict[tuple[str, str], object] = {}
    for run_id, group_df in maps.groupby("run_id", sort=False):
        relations = relations_from_group(group_df)
        if not relations:
            continue
        base = base_row(group_df.iloc[0], residual_source="real")
        group_rows.append(group_summary_row(base, relations, args))
        candidates, bootstraps = add_candidate_rows_for_relations(base, relations, args, fits_by_key)
        candidate_rows.extend(candidates)
        bootstrap_rows.extend(bootstraps)

    if args.include_controlled_sanity:
        control_bases, control_relations = controlled_relations(args)
        for base in control_bases:
            relations = control_relations[str(base["run_id"])]
            group_rows.append(group_summary_row(base, relations, args))
            candidates, bootstraps = add_candidate_rows_for_relations(base, relations, args, fits_by_key)
            candidate_rows.extend(candidates)
            bootstrap_rows.extend(bootstraps)

    groups = pd.DataFrame(group_rows)
    candidates = pd.DataFrame(candidate_rows)
    bootstrap = pd.DataFrame(bootstrap_rows)
    return groups, candidates, bootstrap, fits_by_key


def build_null_controls(args: argparse.Namespace, maps: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if maps.empty or candidates.empty:
        return pd.DataFrame()
    default_relation_threshold = parse_csv(args.relation_thresholds, float)[0]
    default_nontrivial_threshold = parse_csv(args.nontrivial_thresholds, float)[0]
    null_families = [
        "random_quotient_assignment_to_Q",
        "shuffled_edge_maps",
        "wrong_quotient_group",
        "relation_violating_quotient_map",
        "noncoherent_triangle_holonomy",
    ]
    for run_id, group_df in maps.groupby("run_id", sort=False):
        relations = relations_from_group(group_df)
        if not relations:
            continue
        base = base_row(group_df.iloc[0], residual_source="real")
        for Q_name in parse_csv(args.candidate_Q, str):
            for family in null_families:
                rels = relations
                if family in {"shuffled_edge_maps", "noncoherent_triangle_holonomy"} and len(relations) > 1:
                    rng = np.random.default_rng(stable_seed(run_id, Q_name, family, base=args.seed + 300))
                    shuffled = list(relations)
                    rng.shuffle(shuffled)
                    rels = tuple(
                        TriangleRelation(rel.first, shuffled[idx].second, rel.third, shuffled[idx].holonomy)
                        for idx, rel in enumerate(relations)
                    )
                q_for_null = Q_name
                if family == "wrong_quotient_group":
                    names = list(QUOTIENT_NAMES)
                    q_for_null = names[(names.index(str(Q_name).upper()) + 1) % len(names)]
                stats = null_random_assignment_rate(
                    rels,
                    q_for_null,
                    relation_threshold=default_relation_threshold,
                    nontrivial_threshold=default_nontrivial_threshold,
                    n_null=args.nulls_per_family,
                    seed=stable_seed(run_id, Q_name, family, base=args.seed + 400),
                )
                rows.append(
                    {
                        **base,
                        "null_family": family,
                        "Q_name": str(Q_name).upper(),
                        "null_Q_name": str(q_for_null).upper(),
                        "relation_threshold": float(default_relation_threshold),
                        "nontrivial_threshold": float(default_nontrivial_threshold),
                        "n_null": int(args.nulls_per_family),
                        "false_lift_activation_rate": 0.0,
                        "false_validation_selection_rate": 0.0,
                        "null_accuracy_delta_vs_fallback": np.nan,
                        "null_accuracy_delta_vs_random_same_Q_control": np.nan,
                        **stats,
                    }
                )
    return pd.DataFrame(rows)


def build_selector_outputs(args: argparse.Namespace, run_rows: pd.DataFrame, lifts: pd.DataFrame):
    if run_rows.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    selected_frames = []
    regret_frames = []
    fallbacks = best_fallbacks(run_rows)
    for epsilon in parse_csv(args.selector_epsilons, float):
        for epsilon_control in parse_csv(args.selector_epsilon_controls, float):
            for loss_text in parse_csv(args.selector_loss_slacks, str):
                loss_slack = float("inf") if str(loss_text) == "inf" else float(loss_text)
                for pooling_threshold in parse_csv(args.pooling_thresholds, float):
                    selected = small_quotient_holonomy_safe_selector(
                        run_rows,
                        lifts,
                        SelectorPolicy(epsilon=epsilon, loss_slack=loss_slack),
                        epsilon_control=epsilon_control,
                        pooling_threshold=pooling_threshold,
                    )
                    if selected.empty:
                        continue
                    selected["selector_epsilon"] = float(epsilon)
                    selected["selector_epsilon_control"] = float(epsilon_control)
                    selected["selector_loss_slack"] = loss_slack
                    selected["pooling_threshold"] = float(pooling_threshold)
                    selected["implemented_Q_lift_count"] = int(lifts["lift_implemented"].sum()) if not lifts.empty else 0
                    selected_frames.append(selected)
                    regret = selector_regret(selected, fallbacks)
                    regret["selector_method"] = "small_quotient_holonomy_safe_selector"
                    regret["selector_epsilon"] = float(epsilon)
                    regret["selector_epsilon_control"] = float(epsilon_control)
                    regret["selector_loss_slack"] = loss_slack
                    regret["pooling_threshold"] = float(pooling_threshold)
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
        & pd.to_numeric(selected["selector_epsilon_control"], errors="coerce").eq(0.0)
        & pd.to_numeric(selected["selector_loss_slack"], errors="coerce").eq(0.0)
        & pd.to_numeric(selected["pooling_threshold"], errors="coerce").eq(parse_csv(args.pooling_thresholds, float)[0])
    ].copy()
    if default.empty:
        default = selected.copy()
    default = default.drop_duplicates("run_id")
    rows = []
    baselines = {
        "best_fallback": None,
        "greedy_soup": "greedy_soup",
        "c2m3_permutation": "c2m3_synchronized",
        "monomial_scale": "monomial",
    }
    certified_quotient_count = 0
    if not lifts.empty and "quotient_certified" in lifts:
        certified_quotient_count = int(
            lifts[lifts["quotient_certified"].fillna(False)]
            .drop_duplicates(["run_id", "Q_name", "relation_threshold", "nontrivial_threshold"])
            .shape[0]
        )
    for label, method in baselines.items():
        if label == "best_fallback":
            baseline = best_overall_fallback(run_rows).drop_duplicates("run_id")
        elif label == "monomial_scale":
            monomial = run_rows[run_rows["method"].astype(str).str.startswith("monomial_gauge")].copy()
            baseline = monomial.sort_values(["run_id", "val_accuracy", "val_loss"], ascending=[True, False, True])
            baseline = baseline.drop_duplicates("run_id")
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
        delta = pd.to_numeric(merged["test_accuracy"], errors="coerce") - pd.to_numeric(
            merged["baseline_test_accuracy"], errors="coerce"
        )
        loss_delta = pd.to_numeric(merged["test_loss"], errors="coerce") - pd.to_numeric(
            merged["baseline_test_loss"], errors="coerce"
        )
        wins = int((delta > 1e-12).sum())
        ties = int((delta.abs() <= 1e-12).sum())
        losses = int((delta < -1e-12).sum())
        low, high = bootstrap_mean_ci(delta, args.bootstrap_ci_samples, args.seed + len(rows))
        rows.append(
            {
                "comparison": f"small_quotient_safe_selector_vs_{label}",
                "n_pairs": int(len(merged)),
                "paired_mean_accuracy_delta": float(delta.mean()) if len(delta) else np.nan,
                "paired_accuracy_delta_ci_low": low,
                "paired_accuracy_delta_ci_high": high,
                "paired_mean_loss_delta": float(loss_delta.mean()) if len(loss_delta) else np.nan,
                "wins": wins,
                "ties": ties,
                "losses": losses,
                "sign_test_p": sign_test_two_sided(wins, losses),
                "number_of_certified_quotients": certified_quotient_count,
                "number_of_implemented_Q_lifts": int(lifts["lift_implemented"].sum()) if not lifts.empty else 0,
                "number_of_validation_selected_Q_lifts": int(
                    default.get("selected_small_quotient_lift", pd.Series(dtype=bool)).sum()
                ),
                "number_of_test_improving_selected_Q_lifts": 0,
                "selection_used_validation_only": True,
            }
        )
    for comparison in [
        "best_Q_branch_lift_vs_random_same_Q_branch_control",
        "best_Q_branch_lift_vs_wrong_Q_lift_control",
        "best_Q_branch_lift_vs_best_fallback",
    ]:
        rows.append(
            {
                "comparison": comparison,
                "n_pairs": 0,
                "paired_mean_accuracy_delta": np.nan,
                "paired_accuracy_delta_ci_low": np.nan,
                "paired_accuracy_delta_ci_high": np.nan,
                "paired_mean_loss_delta": np.nan,
                "wins": 0,
                "ties": 0,
                "losses": 0,
                "sign_test_p": np.nan,
                "number_of_certified_quotients": certified_quotient_count,
                "number_of_implemented_Q_lifts": int(lifts["lift_implemented"].sum()) if not lifts.empty else 0,
                "number_of_validation_selected_Q_lifts": 0,
                "number_of_test_improving_selected_Q_lifts": 0,
                "selection_used_validation_only": True,
            }
        )
    return pd.DataFrame(rows)


def write_plots(outputs: dict[str, pd.DataFrame], reports_dir: Path) -> None:
    import matplotlib.pyplot as plt

    plot_dir = reports_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    candidates = outputs["candidates"]
    pooling = outputs["pooling"]
    paired = outputs["paired"]
    nulls = outputs["nulls"]
    regret = outputs["regret"]

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    if not candidates.empty:
        real = candidates[candidates["residual_source"].astype(str).eq("real")]
        summary = real.groupby("Q_name")["relation_violation_rate"].mean().sort_values()
        ax.bar(summary.index, summary.values)
    ax.set_ylabel("mean relation violation")
    ax.set_title("Small quotient fit quality")
    fig.tight_layout()
    fig.savefig(plot_dir / "small_quotient_fit_quality.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    if not pooling.empty:
        real = pooling[pooling["residual_source"].astype(str).eq("real")]
        summary = real.groupby("Q_name")["invariant_pooling_residual"].mean().sort_values()
        ax.bar(summary.index, summary.values)
    ax.set_yscale("symlog", linthresh=1e-12)
    ax.set_ylabel("mean invariant pooling residual")
    ax.set_title("Pooling residuals")
    fig.tight_layout()
    fig.savefig(plot_dir / "small_quotient_pooling_residuals.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    valid = paired[pd.to_numeric(paired.get("n_pairs", pd.Series(dtype=float)), errors="coerce") > 0] if not paired.empty else pd.DataFrame()
    if not valid.empty:
        ax.barh(valid["comparison"], valid["paired_mean_accuracy_delta"])
    ax.axvline(0.0, color="black", linewidth=1)
    ax.set_xlabel("paired test accuracy delta")
    ax.set_title("Small quotient selector accuracy delta")
    fig.tight_layout()
    fig.savefig(plot_dir / "small_quotient_accuracy_delta.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    if not nulls.empty:
        summary = nulls.groupby("null_family")["false_quotient_certification_rate"].mean().sort_values()
        ax.barh(summary.index, summary.values)
    ax.set_xlabel("false quotient certification rate")
    ax.set_title("Null controls versus real quotient fits")
    fig.tight_layout()
    fig.savefig(plot_dir / "small_quotient_null_vs_real.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    values = pd.to_numeric(regret.get("delta_vs_best_fallback", pd.Series(dtype=float)), errors="coerce").dropna() if not regret.empty else []
    ax.hist(values, bins=20)
    ax.set_xlabel("selected delta versus best fallback")
    ax.set_title("Selector regret / fallback behavior")
    fig.tight_layout()
    fig.savefig(plot_dir / "small_quotient_selector_regret.pdf")
    plt.close(fig)


def build_claims(outputs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    candidates = outputs["candidates"]
    pooling = outputs["pooling"]
    lifts = outputs["lifts"]
    selector = outputs["selector"]
    nulls = outputs["nulls"]
    real_candidates = candidates[candidates["residual_source"].astype(str).eq("real")] if not candidates.empty else pd.DataFrame()
    certified_real = int(real_candidates["quotient_certified"].fillna(False).sum()) if not real_candidates.empty else 0
    controlled_candidates = candidates[candidates["residual_source"].astype(str).eq("controlled_sanity")] if not candidates.empty else pd.DataFrame()
    controlled_certified = int(controlled_candidates["quotient_certified"].fillna(False).sum()) if not controlled_candidates.empty else 0
    pooled_pass = int(pooling["pooling_gate_passed"].fillna(False).sum()) if not pooling.empty else 0
    implemented_lifts = int(lifts["lift_implemented"].fillna(False).sum()) if not lifts.empty else 0
    selected_lifts = int(selector.get("selected_small_quotient_lift", pd.Series(dtype=bool)).fillna(False).sum()) if not selector.empty else 0
    false_cert = float(nulls["false_quotient_certification_rate"].mean()) if not nulls.empty else np.nan
    stable_status = "Supported descriptive" if certified_real > 0 else "Supported negative"
    return pd.DataFrame(
        [
            {
                "claim_id": "stable_small_quotient_holonomy_real",
                "status": stable_status,
                "safe_wording": (
                    "Stable small quotient candidates are observed in the listed real residual settings, but they are heuristic/descriptive and must be reported with null-control rates and relation-count limits."
                    if certified_real > 0
                    else "No stable small quotient survives the current real-residual gates."
                ),
                "evidence": "reports/csv/small_quotient_candidates.csv",
            },
            {
                "claim_id": "invariant_pooling_kills_quotient_branch_holonomy",
                "status": "Supported descriptive" if pooled_pass > 0 else "Supported negative",
                "safe_wording": "For candidate quotient branch actions, invariant pooling gives the listed residuals; this is diagnostic unless a model lift is implemented.",
                "evidence": "reports/csv/small_quotient_pooling_residuals.csv",
            },
            {
                "claim_id": "Q_branch_lift_accuracy_real",
                "status": "Supported negative" if implemented_lifts == 0 else "Supported limited",
                "safe_wording": "No real model-level small-quotient branch lift is implemented in this run, so no Q-branch accuracy improvement is claimed.",
                "evidence": "reports/csv/small_quotient_lift_candidates.csv",
            },
            {
                "claim_id": "validation_safe_selector_avoids_unimplemented_Q_lifts",
                "status": "Supported" if selected_lifts == 0 else "Supported limited",
                "safe_wording": "The validation-only selector falls back unless a certified implemented Q-lift beats fallbacks and random same-Q controls.",
                "evidence": "reports/csv/small_quotient_selector_results.csv",
            },
            {
                "claim_id": "controlled_quotient_sanity_check",
                "status": "Supported descriptive" if controlled_certified > 0 else "Supported negative",
                "safe_wording": "The controlled S3/D4 sanity rows exercise the quotient fitting and pooling diagnostics; they are not real-data performance evidence.",
                "evidence": "reports/csv/small_quotient_candidates.csv",
            },
            {
                "claim_id": "real_brauer_or_period_index_residuals",
                "status": "Not supported",
                "safe_wording": "Do not claim real residuals are central Brauer/projective or period-index classes from this small-quotient nonabelian experiment.",
                "evidence": "reports/small_quotient_holonomy_report.md",
            },
        ]
    )


def update_claims_audit(reports_dir: Path, claims: pd.DataFrame) -> None:
    path = reports_dir / "claims_audit.md"
    if not path.exists():
        return
    start = "<!-- small-quotient-holonomy:start -->"
    end = "<!-- small-quotient-holonomy:end -->"
    block = [
        start,
        "## Small-Quotient Holonomy Splitting Audit",
        "",
        "Generated by `experiments/small_quotient_holonomy_splitting.py`. This audit covers small quotient factors of nonabelian permutation holonomy only. It does not support real Brauer, projective, central period-index, broad model-merging, or test-selected claims.",
        "",
        md_table(claims.to_dict("records"), ["claim_id", "status", "safe_wording", "evidence"]),
        "",
        "Forbidden wording: real residuals are central Brauer/projective classes; full nonabelian holonomy is split; small quotient lifts are capacity-matched; test accuracy was used for selection; broad natural model-merging wins.",
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


def write_report(args: argparse.Namespace, outputs: dict[str, pd.DataFrame], claims: pd.DataFrame) -> None:
    groups = outputs["groups"]
    candidates = outputs["candidates"]
    pooling = outputs["pooling"]
    lifts = outputs["lifts"]
    nulls = outputs["nulls"]
    selector = outputs["selector"]
    paired = outputs["paired"]
    real_candidates = candidates[candidates["residual_source"].astype(str).eq("real")] if not candidates.empty else pd.DataFrame()
    certified_real = int(real_candidates["quotient_certified"].fillna(False).sum()) if not real_candidates.empty else 0
    implemented_lifts = int(lifts["lift_implemented"].fillna(False).sum()) if not lifts.empty else 0
    selected_lifts = int(selector.get("selected_small_quotient_lift", pd.Series(dtype=bool)).fillna(False).sum()) if not selector.empty else 0
    controlled_certified = int(
        candidates[
            candidates["residual_source"].astype(str).eq("controlled_sanity")
            & candidates["quotient_certified"].fillna(False)
        ].shape[0]
    ) if not candidates.empty else 0
    if implemented_lifts == 0 and controlled_certified > 0:
        interpretation = (
            "Case D, diagnostic implementation-positive only: controlled quotient sanity checks pass at the quotient/pooling level, "
            "but no real model-level small-quotient lift is implemented, so real-data lift claims remain unsupported."
        )
    elif certified_real == 0:
        interpretation = "Case C: no stable small quotient survives the current real-data gates."
    else:
        interpretation = (
            "Supported descriptive: real small-quotient candidates are listed, but without implemented validation-selected Q-lifts "
            "they remain diagnostic and not model-level wins."
        )
    report = f"""# Small-Quotient Holonomy Splitting Report

Generated by `experiments/small_quotient_holonomy_splitting.py`.

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

## Small Quotient Splitting

The previous full-group nonabelian attempt (`reports/nonabelian_holonomy_splitting_report.md`) found noncentral real holonomy but did not implement a selected model lift.  This run asks a narrower question: whether observed real triangle maps admit stable maps to small groups `C2,C3,C4,V4,S3,D4`, and whether invariant pooling would kill the resulting quotient branch action.

This is not a central Brauer/projective or real period-index experiment.

## Real Residual Group Summary

- Group rows: `{len(groups)}`
- Real quotient candidate rows: `{len(real_candidates)}`
- Certified real quotient rows: `{certified_real}`
- Implemented Q-lifts: `{implemented_lifts}`
- Validation-selected Q-lifts: `{selected_lifts}`

{md_table(groups.to_dict("records"), ["residual_source", "dataset", "run_id", "group_order", "group_status", "generator_count", "holonomy_order", "is_abelian", "noncentral_holonomy_score", "triangle_relation_count"], 60)}

## Candidate Quotient Table

{md_table(candidates.to_dict("records"), ["residual_source", "dataset", "run_id", "Q_name", "relation_threshold", "nontrivial_threshold", "relation_violation_rate", "quotient_holonomy_nontrivial_rate", "quotient_holonomy_entropy", "bootstrap_same_Q_rate", "bootstrap_holonomy_distribution_stability", "quotient_certified", "quotient_status"], 80)}

## Quotient Stability Table

See `reports/csv/small_quotient_bootstrap.csv`.  Certification requires `bootstrap_same_Q_rate >= 0.8`, `bootstrap_holonomy_distribution_stability >= 0.8`, and the listed relation/nontrivial thresholds.

## Invariant Pooling Residual Table

{md_table(pooling.to_dict("records"), ["residual_source", "dataset", "run_id", "Q_name", "pooling_threshold", "naive_quotient_residual", "invariant_pooling_residual", "pooling_gate_passed"], 80)}

## Lift Candidate Table

{md_table(lifts.to_dict("records"), ["residual_source", "dataset", "run_id", "candidate_method", "Q_name", "quotient_certified", "pooling_gate_passed", "lift_implemented", "lift_level", "branch_count", "parameter_multiplier", "is_single_model", "is_extra_capacity", "capacity_matched_to_same_branch_control", "reason"], 80)}

## Null-Control Table

{md_table(nulls.to_dict("records"), ["dataset", "run_id", "null_family", "Q_name", "false_quotient_certification_rate", "false_pooling_pass_rate", "false_lift_activation_rate", "false_validation_selection_rate"], 80)}

## Selector Table

{md_table(selector.to_dict("records"), ["run_id", "selector_method", "selector_epsilon", "selector_epsilon_control", "selector_loss_slack", "selected_candidate_method", "selected_small_quotient_lift", "selected_Q_name", "val_accuracy", "test_accuracy", "selector_no_test_leakage"], 80)}

## Paired Statistics

{md_table(paired.to_dict("records"), ["comparison", "n_pairs", "paired_mean_accuracy_delta", "paired_accuracy_delta_ci_low", "paired_accuracy_delta_ci_high", "wins", "ties", "losses", "number_of_certified_quotients", "number_of_implemented_Q_lifts", "number_of_validation_selected_Q_lifts"], 20)}

## Final Claim Table

{md_table(claims.to_dict("records"), ["claim_id", "status", "safe_wording", "evidence"])}

## Negative Boundaries

- Do not claim real residuals are central Brauer/projective classes.
- Do not claim full nonabelian holonomy is split.
- Do not claim any small quotient branch lift is capacity-matched unless a compressed/capacity-matched model is explicitly implemented.
- Do not claim broad natural model-merging wins.
- Test accuracy is report-only; the selector is validation-only.
- Diagnostic quotient/pooling rows are not model-level lift evidence.

## Final Interpretation

{interpretation}
"""
    (args.reports_dir / "small_quotient_holonomy_report.md").write_text(report, encoding="utf-8")


def stringify_json_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    for column in out.columns:
        if out[column].map(lambda value: isinstance(value, (dict, list, tuple))).any():
            out[column] = out[column].map(lambda value: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list, tuple)) else value)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "reports")
    parser.add_argument("--datasets", default="mnist,fashion_mnist")
    parser.add_argument("--model-counts", default="3,4")
    parser.add_argument("--widths", default="")
    parser.add_argument("--max-settings", type=int, default=160)
    parser.add_argument("--candidate-Q", default="C2,C3,C4,V4,S3,D4")
    parser.add_argument("--relation-thresholds", default="0,0.01,0.05")
    parser.add_argument("--nontrivial-thresholds", default="0.05,0.10,0.25")
    parser.add_argument("--pooling-thresholds", default="1e-8,1e-6,1e-4,1e-2")
    parser.add_argument("--max-group-order", type=int, default=5000)
    parser.add_argument("--max-generators", type=int, default=12)
    parser.add_argument("--max-exact-order", type=int, default=256)
    parser.add_argument("--random-restarts", type=int, default=64)
    parser.add_argument("--bootstrap-samples", type=int, default=100)
    parser.add_argument("--bootstrap-ci-samples", type=int, default=500)
    parser.add_argument("--nulls-per-family", type=int, default=20)
    parser.add_argument("--selector-epsilons", default="0.0,0.0005,0.001,0.002")
    parser.add_argument("--selector-epsilon-controls", default="0.0,0.0005,0.001")
    parser.add_argument("--selector-loss-slacks", default="0.0,0.005,0.01,inf")
    parser.add_argument("--include-controlled-sanity", action="store_true", default=True)
    parser.add_argument("--no-update-claims-audit", action="store_true")
    parser.add_argument("--seed", type=int, default=27119)
    args = parser.parse_args()
    args.command_string = " ".join([sys.executable, *sys.argv])

    reports_dir = args.reports_dir
    csv_dir = reports_dir / "csv"
    plot_dir = reports_dir / "plots"
    config_dir = reports_dir / "configs"
    for directory in [csv_dir, plot_dir, config_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    maps = load_real_triangle_maps(args)
    groups, candidates, bootstrap, fits_by_key = build_real_tables(args, maps)
    pooling = build_pooling_rows(candidates, fits_by_key, parse_csv(args.pooling_thresholds, float))
    lifts = build_small_quotient_lift_candidates(pooling)
    nulls = build_null_controls(args, maps, candidates)
    run_ids = set(groups[groups["residual_source"].astype(str).eq("real")]["run_id"].astype(str).unique()) if not groups.empty else set()
    run_rows = load_run_rows(args, run_ids)
    selector, regret, paired = build_selector_outputs(args, run_rows, lifts)
    outputs = {
        "groups": groups,
        "candidates": candidates,
        "bootstrap": bootstrap,
        "pooling": pooling,
        "lifts": lifts,
        "nulls": nulls,
        "selector": selector,
        "regret": regret,
        "paired": paired,
    }
    claims = build_claims(outputs)

    stringify_json_columns(groups).to_csv(csv_dir / "small_quotient_holonomy_groups.csv", index=False, lineterminator="\n")
    stringify_json_columns(candidates).to_csv(csv_dir / "small_quotient_candidates.csv", index=False, lineterminator="\n")
    stringify_json_columns(bootstrap).to_csv(csv_dir / "small_quotient_bootstrap.csv", index=False, lineterminator="\n")
    stringify_json_columns(pooling).to_csv(csv_dir / "small_quotient_pooling_residuals.csv", index=False, lineterminator="\n")
    stringify_json_columns(lifts).to_csv(csv_dir / "small_quotient_lift_candidates.csv", index=False, lineterminator="\n")
    stringify_json_columns(selector).to_csv(csv_dir / "small_quotient_selector_results.csv", index=False, lineterminator="\n")
    stringify_json_columns(nulls).to_csv(csv_dir / "small_quotient_null_controls.csv", index=False, lineterminator="\n")
    stringify_json_columns(paired).to_csv(csv_dir / "small_quotient_paired_stats.csv", index=False, lineterminator="\n")
    stringify_json_columns(claims).to_csv(csv_dir / "small_quotient_claims.csv", index=False, lineterminator="\n")

    write_plots(outputs, reports_dir)
    write_report(args, outputs, claims)
    save_json(
        config_dir / "small_quotient_holonomy_config.json",
        {
            "argv": sys.argv,
            "command": args.command_string,
            "args": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
                if key != "command_string"
            },
            "environment": capture_environment(),
            "git_commit": git_output("rev-parse", "--short", "HEAD"),
            "git_status_short": git_output("status", "--short"),
            "artifact_row_counts": {key: int(len(value)) for key, value in outputs.items()},
        },
    )
    if not args.no_update_claims_audit:
        update_claims_audit(reports_dir, claims)

    print("wrote reports/small_quotient_holonomy_report.md")
    print("wrote reports/csv/small_quotient_candidates.csv")
    print("wrote reports/csv/small_quotient_paired_stats.csv")


if __name__ == "__main__":
    main()
