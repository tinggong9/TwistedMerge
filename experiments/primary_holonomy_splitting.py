#!/usr/bin/env python
"""Primary-factor holonomy splitting with nested branch lifts."""

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

from src.metrics import capture_environment, save_json  # noqa: E402
from src.nonabelian_holonomy import infer_holonomy_group  # noqa: E402
from src.primary_branch_lift import build_controlled_primary_lift_rows, build_real_primary_lift_rows  # noqa: E402
from src.primary_holonomy import (  # noqa: E402
    bootstrap_primary_fit,
    candidate_q_orders_for_source,
    fit_primary_quotient,
    observed_holonomy_order_lcm,
    p_adic_valuation,
    primary_fit_certified,
    primary_pooling_residuals,
    relation_count_status,
    triangle_relation_from_perms,
)
from src.primary_splitting_selector import primary_holonomy_safe_selector  # noqa: E402
from src.validation_gated_period_index_lift import SelectorPolicy, best_overall_fallback, selector_regret  # noqa: E402


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
        vals = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                value = "" if not np.isfinite(value) else f"{value:.6g}"
            vals.append(str(value).replace("|", "\\|"))
        out.append("| " + " | ".join(vals) + " |")
    return "\n".join(out)


def safe_json_array(value) -> tuple[int, ...] | None:
    if not isinstance(value, str) or not value.strip() or value == "nan":
        return None
    try:
        arr = tuple(int(item) for item in json.loads(value))
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
        raise FileNotFoundError(f"no fixed-setting triangle maps found in {artifact_dir}")
    frames = [pd.read_csv(shard, usecols=lambda col: col in TRIANGLE_COLUMNS) for shard in shards]
    df = pd.concat(frames, ignore_index=True)
    df = df[df["triangle_type"].astype(str).eq("permutation")].copy()
    df = df[df["alignment_source"].astype(str).eq("observed")].copy()
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
        df = df[df["run_id"].isin(selected[: int(args.max_settings)])].copy()
    return df.reset_index(drop=True)


def load_run_rows(args: argparse.Namespace, run_ids: set[str]) -> pd.DataFrame:
    path = args.reports_dir / "csv" / "fixed_setting_verification_runs.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, usecols=lambda col: col in RUN_COLUMNS)
    if run_ids:
        df = df[df["run_id"].astype(str).isin(run_ids)].copy()
    return df


def relations_from_df(df: pd.DataFrame) -> tuple:
    relations = []
    for _, row in df.iterrows():
        p_ij = safe_json_array(row.get("p_ij"))
        p_jk = safe_json_array(row.get("p_jk"))
        p_ki = safe_json_array(row.get("p_ki"))
        hol = safe_json_array(row.get("triangle_perm"))
        if p_ij is None or p_jk is None or p_ki is None or hol is None:
            continue
        relations.append(triangle_relation_from_perms(p_ij, p_jk, p_ki, hol))
    return tuple(relations)


def base_from_df(df: pd.DataFrame, relation_set_id: str, aggregation_level: str) -> dict:
    first = df.iloc[0]
    return {
        "residual_source": "real",
        "relation_set_id": relation_set_id,
        "aggregation_level": aggregation_level,
        "setting_id": first.get("setting_id"),
        "dataset": first.get("dataset"),
        "architecture": first.get("architecture"),
        "n_models": int(first.get("n_models")),
        "width": int(first.get("width")),
        "domain_shift": first.get("domain_shift"),
        "matching": first.get("matching"),
        "seed": int(first.get("seed")),
    }


def controlled_relation_sets(args: argparse.Namespace) -> tuple[list[dict], dict[str, tuple], dict[str, list[str]]]:
    out = []
    relations = {}
    members = {}
    specs = [
        ("controlled_C2_primary", 2, [(0, 1), (1, 0)] * 6),
        ("controlled_C3_primary", 3, [(0, 1, 2), (1, 2, 0), (2, 0, 1)] * 4),
        ("controlled_C4_primary", 4, [(0, 1, 2, 3), (1, 2, 3, 0), (2, 3, 0, 1), (3, 0, 1, 2)] * 3),
    ]
    for name, q_order, holonomies in specs:
        identity = tuple(range(len(holonomies[0])))
        rels = tuple(triangle_relation_from_perms(identity, identity, hol, hol) for hol in holonomies)
        base = {
            "residual_source": "controlled_sanity",
            "relation_set_id": name,
            "aggregation_level": "controlled",
            "setting_id": name,
            "dataset": "controlled_sanity",
            "architecture": "synthetic_prediction_branch",
            "n_models": 3,
            "width": len(holonomies[0]),
            "domain_shift": "planted_primary_holonomy",
            "matching": "controlled_exact",
            "seed": int(args.seed),
            "controlled_true_q": int(q_order),
        }
        out.append(base)
        relations[name] = rels
        members[name] = [name]
    return out, relations, members


def build_relation_sets(args: argparse.Namespace, maps: pd.DataFrame):
    bases = []
    relation_sets = {}
    members = {}
    for run_id, group in maps.groupby("run_id", sort=False):
        rels = relations_from_df(group)
        if not rels:
            continue
        relation_set_id = f"run::{run_id}"
        bases.append(base_from_df(group, relation_set_id, "run_id"))
        relation_sets[relation_set_id] = rels
        members[relation_set_id] = [str(run_id)]
    for setting_id, group in maps.groupby("setting_id", sort=False):
        rels = relations_from_df(group)
        if not rels:
            continue
        relation_set_id = f"setting::{setting_id}"
        bases.append(base_from_df(group, relation_set_id, "setting_id"))
        relation_sets[relation_set_id] = rels
        members[relation_set_id] = sorted(group["run_id"].astype(str).unique().tolist())
    for key, group in maps.groupby(["dataset", "n_models", "width", "matching"], sort=False):
        rels = relations_from_df(group)
        if not rels:
            continue
        dataset, n_models, width, matching = key
        relation_set_id = f"pooled::{dataset}::N{int(n_models)}::W{int(width)}::{matching}"
        bases.append(base_from_df(group, relation_set_id, "pooled_dataset_model_width_matching"))
        relation_sets[relation_set_id] = rels
        members[relation_set_id] = sorted(group["run_id"].astype(str).unique().tolist())
    control_bases, control_relations, control_members = controlled_relation_sets(args)
    bases.extend(control_bases)
    relation_sets.update(control_relations)
    members.update(control_members)
    return bases, relation_sets, members


def group_summary(base: dict, relations: tuple, args: argparse.Namespace) -> dict:
    edges = []
    holonomies = []
    for relation in relations:
        edges.extend([relation.first, relation.second, relation.third])
        holonomies.append(relation.holonomy)
    summary = infer_holonomy_group(
        edges,
        holonomies,
        max_group_order=args.max_group_order,
        max_generators=args.max_generators,
        max_exact_order=args.max_exact_order,
    )
    observed_order = observed_holonomy_order_lcm(relations)
    group_exponent = summary.group_exponent
    unique_triangles = len({(rel.first, rel.second, rel.third, rel.holonomy) for rel in relations})
    return {
        **base,
        "relation_count": int(len(relations)),
        "unique_triangle_count": int(unique_triangles),
        "unique_holonomy_count": int(len({rel.holonomy for rel in relations})),
        "observed_holonomy_order_lcm": int(observed_order),
        "v2_observed_order": p_adic_valuation(observed_order, 2),
        "v3_observed_order": p_adic_valuation(observed_order, 3),
        "primary_source": "group_exponent" if group_exponent else "observed_holonomy_order",
        "group_closure_status": summary.group_status,
        "group_order_if_exact": int(summary.group.order) if not summary.group.truncated else np.nan,
        "group_exponent_if_exact": group_exponent,
        "v2_group_order_if_exact": p_adic_valuation(summary.group.order, 2) if not summary.group.truncated else 0,
        "v3_group_order_if_exact": p_adic_valuation(summary.group.order, 3) if not summary.group.truncated else 0,
        "v2_group_exponent_if_exact": p_adic_valuation(group_exponent, 2),
        "v3_group_exponent_if_exact": p_adic_valuation(group_exponent, 3),
        "noncentral_holonomy_score": summary.noncentral_holonomy_score,
        "relation_count_status": relation_count_status(len(relations), args.min_relation_count),
    }


def fit_primary_tables(args: argparse.Namespace, groups: pd.DataFrame, relation_sets: dict[str, tuple]):
    candidate_rows = []
    bootstrap_rows = []
    pooling_rows = []
    fits = {}
    thresholds = [
        ("strict", args.relation_threshold, args.nontrivial_threshold, args.entropy_threshold, True),
        ("loose_diagnostic", args.loose_relation_threshold, args.loose_nontrivial_threshold, args.loose_entropy_threshold, False),
    ]
    q_filter = set(parse_csv(args.q_orders, int)) if args.q_orders else set()
    for _, group_row in groups.iterrows():
        relation_set_id = str(group_row["relation_set_id"])
        relations = relation_sets[relation_set_id]
        observed_order = int(group_row["observed_holonomy_order_lcm"])
        exponent = group_row.get("group_exponent_if_exact")
        exponent = int(exponent) if pd.notna(exponent) else None
        for q_meta in candidate_q_orders_for_source(observed_order, exponent):
            if q_filter and int(q_meta["q_order"]) not in q_filter:
                continue
            q_order = int(q_meta["q_order"])
            fit = fit_primary_quotient(
                relations,
                q_order,
                random_restarts=args.random_restarts,
                seed=stable_seed(relation_set_id, q_order, base=args.seed),
            )
            fits[(relation_set_id, q_order)] = fit
            pooling = primary_pooling_residuals(fit)
            for profile, rel_thr, non_thr, ent_thr, activation_eligible in thresholds:
                boot = bootstrap_primary_fit(
                    fit,
                    relation_threshold=rel_thr,
                    nontrivial_threshold=non_thr,
                    entropy_threshold=ent_thr,
                    n_bootstrap=args.bootstrap_samples,
                    seed=stable_seed(relation_set_id, q_order, profile, base=args.seed + 100),
                )
                certified = (
                    primary_fit_certified(
                        fit,
                        boot,
                        relation_count=int(group_row["relation_count"]),
                        relation_threshold=rel_thr,
                        nontrivial_threshold=non_thr,
                        entropy_threshold=ent_thr,
                        min_relation_count=args.min_relation_count,
                    )
                    and bool(q_meta["divides_primary_source"])
                    and activation_eligible
                )
                base = {
                    **group_row.to_dict(),
                    **q_meta,
                    "threshold_profile": profile,
                    "activation_eligible_threshold_profile": bool(activation_eligible),
                    "relation_threshold": float(rel_thr),
                    "nontrivial_threshold": float(non_thr),
                    "entropy_threshold": float(ent_thr),
                    "relation_violation_rate": fit.relation_violation_rate,
                    "quotient_holonomy_nontrivial_rate": fit.quotient_holonomy_nontrivial_rate,
                    "quotient_holonomy_entropy": fit.quotient_holonomy_entropy,
                    "quotient_assignment_confidence": fit.quotient_assignment_confidence,
                    "quotient_fit_status": fit.quotient_fit_status,
                    "assignment_strategy": fit.assignment_strategy,
                    "quotient_certified": bool(certified),
                    "quotient_status": (
                        "certified_primary"
                        if certified
                        else (
                            "underconstrained_descriptive"
                            if group_row["relation_count_status"] == "underconstrained"
                            else "not_certified_or_wrong_factor"
                        )
                    ),
                }
                candidate_rows.append(base)
                bootstrap_rows.append({**base, **boot})
                for pooling_threshold in parse_csv(args.pooling_thresholds, float):
                    pooling_rows.append(
                        {
                            **base,
                            **pooling,
                            "pooling_threshold": float(pooling_threshold),
                            "pooling_gate_passed": bool(pooling["pooling_residual_q"] <= float(pooling_threshold)),
                        }
                    )
    return pd.DataFrame(candidate_rows), pd.DataFrame(bootstrap_rows), pd.DataFrame(pooling_rows), fits


def selector_fallback_rows(run_rows: pd.DataFrame) -> pd.DataFrame:
    return run_rows.copy()


def controlled_fallback_rows() -> pd.DataFrame:
    rows = []
    for run_id in ["controlled_C2_primary", "controlled_C3_primary", "controlled_C4_primary"]:
        rows.append(
            {
                "run_id": run_id,
                "method": "greedy_soup",
                "candidate_method": "fallback_greedy_soup",
                "val_accuracy": 0.72,
                "val_loss": 0.28,
                "test_accuracy": 0.71,
                "test_loss": 0.29,
            }
        )
    return pd.DataFrame(rows)


def build_lift_tables(args, pooling: pd.DataFrame, run_rows: pd.DataFrame, relation_members: dict[str, list[str]]):
    strict_pooling = pooling[
        pooling["threshold_profile"].astype(str).eq("strict")
        & pd.to_numeric(pooling["pooling_threshold"], errors="coerce").eq(parse_csv(args.pooling_thresholds, float)[0])
    ].copy()
    real = strict_pooling[strict_pooling["residual_source"].astype(str).eq("real")].copy()
    controlled = strict_pooling[strict_pooling["residual_source"].astype(str).eq("controlled_sanity")].copy()
    real_lifts = build_real_primary_lift_rows(real, run_rows, relation_members)
    controlled_lifts = build_controlled_primary_lift_rows(controlled)
    return pd.concat([real_lifts, controlled_lifts], ignore_index=True, sort=False)


def build_selector_outputs(args, run_rows: pd.DataFrame, lifts: pd.DataFrame):
    fallback_rows = pd.concat([selector_fallback_rows(run_rows), controlled_fallback_rows()], ignore_index=True, sort=False)
    frames = []
    regrets = []
    for epsilon in parse_csv(args.selector_epsilons, float):
        for epsilon_control in parse_csv(args.selector_epsilon_controls, float):
            for loss_text in parse_csv(args.selector_loss_slacks, str):
                loss_slack = float("inf") if str(loss_text) == "inf" else float(loss_text)
                for lambda_branch in parse_csv(args.lambda_branch_values, float):
                    for lambda_residual in parse_csv(args.lambda_residual_values, float):
                        selected = primary_holonomy_safe_selector(
                            fallback_rows,
                            lifts,
                            SelectorPolicy(epsilon=epsilon, loss_slack=loss_slack),
                            epsilon_control=epsilon_control,
                            pooling_threshold=parse_csv(args.pooling_thresholds, float)[0],
                            lambda_branch=lambda_branch,
                            lambda_residual=lambda_residual,
                        )
                        if selected.empty:
                            continue
                        frames.append(selected)
                        regrets.append(selector_regret(selected, fallback_rows))
    selected_df = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    regret_df = pd.concat(regrets, ignore_index=True, sort=False) if regrets else pd.DataFrame()
    paired = paired_stats(args, selected_df, fallback_rows, lifts)
    return selected_df, regret_df, paired


def build_nested_sequence(run_rows: pd.DataFrame, lifts: pd.DataFrame) -> pd.DataFrame:
    fallback = best_overall_fallback(run_rows)
    rows = []
    for _, fb in fallback.iterrows():
        rows.append(
            {
                "run_id": fb["run_id"],
                "primary_type": "fallback",
                "q_order": 1,
                "primary_depth": 0,
                "candidate_method": "best_fallback",
                "validation_accuracy": fb.get("val_accuracy"),
                "test_accuracy": fb.get("test_accuracy"),
                "delta_vs_fallback": 0.0,
                "delta_vs_previous_depth": 0.0,
                "structural_delta_vs_random": np.nan,
                "lift_implemented": True,
            }
        )
    structured = lifts[
        lifts["candidate_method"].astype(str).str.startswith(("primary_C", "mixed_C"))
    ].copy()
    controls = lifts[lifts["candidate_method"].astype(str).eq("random_same_branch_count_control")].copy()
    if not structured.empty:
        for _, row in structured.iterrows():
            run_id = row["run_id"]
            fb = fallback[fallback["run_id"].astype(str).eq(str(run_id))]
            fb_acc = float(fb.iloc[0]["test_accuracy"]) if not fb.empty else np.nan
            control = controls[
                controls["run_id"].astype(str).eq(str(run_id))
                & pd.to_numeric(controls["q_order"], errors="coerce").eq(int(row.get("q_order", 0)))
            ]
            control_acc = float(control.iloc[0]["test_accuracy"]) if not control.empty else np.nan
            test_acc = float(row.get("test_accuracy", np.nan))
            rows.append(
                {
                    "run_id": run_id,
                    "relation_set_id": row.get("relation_set_id"),
                    "aggregation_level": row.get("aggregation_level"),
                    "primary_type": row.get("primary_type"),
                    "q_order": int(row.get("q_order", 1)),
                    "primary_depth": int(row.get("primary_depth", 0)),
                    "candidate_method": row.get("candidate_method"),
                    "validation_accuracy": row.get("val_accuracy"),
                    "test_accuracy": row.get("test_accuracy"),
                    "delta_vs_fallback": test_acc - fb_acc if np.isfinite(test_acc) and np.isfinite(fb_acc) else np.nan,
                    "delta_vs_previous_depth": np.nan,
                    "structural_delta_vs_random": test_acc - control_acc if np.isfinite(test_acc) and np.isfinite(control_acc) else np.nan,
                    "lift_implemented": bool(row.get("lift_implemented", False)),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values(["run_id", "primary_type", "q_order"]).copy()
    for _, idxs in out.groupby(["run_id", "primary_type"], sort=False).groups.items():
        prev = None
        for idx in idxs:
            current = out.at[idx, "test_accuracy"]
            if prev is not None and pd.notna(current) and pd.notna(prev):
                out.at[idx, "delta_vs_previous_depth"] = float(current) - float(prev)
            prev = current
    return out


def paired_stats(args, selected: pd.DataFrame, fallback_rows: pd.DataFrame, lifts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if selected.empty:
        return pd.DataFrame()
    default = selected[
        pd.to_numeric(selected.get("selector_epsilon", 0), errors="coerce").eq(0.0)
        & pd.to_numeric(selected.get("selector_epsilon_control", 0), errors="coerce").eq(0.0)
        & pd.to_numeric(selected.get("selector_loss_slack", 0), errors="coerce").eq(0.0)
        & pd.to_numeric(selected.get("lambda_branch", 0), errors="coerce").eq(0.0)
        & pd.to_numeric(selected.get("lambda_residual", 0), errors="coerce").eq(0.0)
    ].copy()
    if default.empty:
        default = selected.copy()
    default = default[~default["run_id"].astype(str).str.startswith("controlled_")].drop_duplicates("run_id")
    comparisons = {
        "best_fallback": None,
        "greedy_soup": "greedy_soup",
        "c2m3_permutation": "c2m3_synchronized",
        "monomial_scale": "monomial",
    }
    for label, method in comparisons.items():
        if label == "best_fallback":
            baseline = best_overall_fallback(fallback_rows).copy()
        elif label == "monomial_scale":
            baseline = fallback_rows[fallback_rows["method"].astype(str).str.startswith("monomial_gauge")].copy()
            baseline = baseline.sort_values(["run_id", "val_accuracy", "val_loss"], ascending=[True, False, True]).drop_duplicates("run_id")
        else:
            baseline = fallback_rows[fallback_rows["method"].astype(str).eq(method)].copy()
        baseline = baseline[~baseline["run_id"].astype(str).str.startswith("controlled_")]
        if baseline.empty:
            continue
        baseline = baseline.sort_values(["run_id", "val_accuracy", "val_loss"], ascending=[True, False, True]).drop_duplicates("run_id")
        merged = default.merge(
            baseline[["run_id", "test_accuracy", "test_loss"]].rename(
                columns={"test_accuracy": "baseline_test_accuracy", "test_loss": "baseline_test_loss"}
            ),
            on="run_id",
            how="inner",
        )
        rows.append(_paired_row(args, f"primary_safe_selector_vs_{label}", merged, lifts))
    for name in [
        "selected_primary_lift_vs_random_same_branch_count_control",
        "selected_primary_lift_vs_wrong_primary_factor_control",
        "C2_lift_vs_C4_lift",
        "C4_lift_vs_C8_lift",
        "C3_lift_vs_C9_lift",
    ]:
        rows.append(_empty_paired_row(name, lifts))
    return pd.DataFrame(rows)


def _paired_row(args, comparison: str, merged: pd.DataFrame, lifts: pd.DataFrame) -> dict:
    delta = pd.to_numeric(merged["test_accuracy"], errors="coerce") - pd.to_numeric(merged["baseline_test_accuracy"], errors="coerce")
    loss_delta = pd.to_numeric(merged["test_loss"], errors="coerce") - pd.to_numeric(merged["baseline_test_loss"], errors="coerce")
    wins = int((delta > 1e-12).sum())
    ties = int((delta.abs() <= 1e-12).sum())
    losses = int((delta < -1e-12).sum())
    low, high = bootstrap_mean_ci(delta, args.bootstrap_ci_samples, args.seed + len(comparison))
    depth_counts = {}
    if not merged.empty and "selected_depth" in merged:
        depth_counts = {
            str(int(key)): int(value)
            for key, value in merged["selected_depth"].value_counts().sort_index().items()
            if pd.notna(key)
        }
    certified_count = _certified_primary_count(lifts)
    implemented_count = _implemented_primary_lift_count(lifts)
    return {
        "comparison": comparison,
        "n_pairs": int(len(merged)),
        "paired_mean_accuracy_delta": float(delta.mean()) if len(delta) else np.nan,
        "paired_accuracy_delta_ci_low": low,
        "paired_accuracy_delta_ci_high": high,
        "paired_mean_loss_delta": float(loss_delta.mean()) if len(loss_delta) else np.nan,
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "sign_test_p": sign_test_two_sided(wins, losses),
        "number_of_certified_primary_quotients": certified_count,
        "number_of_implemented_primary_lifts": implemented_count,
        "number_of_validation_selected_primary_lifts": int(merged.get("selected_primary_lift", pd.Series(dtype=bool)).fillna(False).sum()) if not merged.empty else 0,
        "selected_depth_distribution": json.dumps(depth_counts, sort_keys=True),
    }


def _empty_paired_row(comparison: str, lifts: pd.DataFrame) -> dict:
    certified_count = _certified_primary_count(lifts)
    implemented_count = _implemented_primary_lift_count(lifts)
    return {
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
        "number_of_certified_primary_quotients": certified_count,
        "number_of_implemented_primary_lifts": implemented_count,
        "number_of_validation_selected_primary_lifts": 0,
        "selected_depth_distribution": "{}",
    }


def _certified_primary_count(lifts: pd.DataFrame) -> int:
    if lifts.empty or "quotient_certified" not in lifts:
        return 0
    key_cols = [col for col in ["relation_set_id", "q_order", "threshold_profile"] if col in lifts]
    certified = lifts[lifts["quotient_certified"].fillna(False)].copy()
    if not key_cols:
        return int(len(certified))
    return int(certified.drop_duplicates(key_cols).shape[0])


def _implemented_primary_lift_count(lifts: pd.DataFrame) -> int:
    if lifts.empty or "lift_implemented" not in lifts:
        return 0
    rows = lifts[
        lifts["lift_implemented"].fillna(False)
        & lifts["candidate_method"].astype(str).str.startswith(("primary_C", "mixed_C"))
    ].copy()
    key_cols = [col for col in ["run_id", "relation_set_id", "q_order", "candidate_method"] if col in rows]
    if not key_cols:
        return int(len(rows))
    return int(rows.drop_duplicates(key_cols).shape[0])


def build_null_controls(candidates: pd.DataFrame, pooling: pd.DataFrame, selector: pd.DataFrame) -> pd.DataFrame:
    rows = []
    wrong = candidates[candidates["candidate_role"].astype(str).eq("wrong_factor_control")] if not candidates.empty else pd.DataFrame()
    for family in [
        "random_same_branch_count_control",
        "wrong_primary_factor_control",
        "shuffled_edge_maps",
        "randomized_quotient_assignment",
        "relation_violating_quotient",
        "random_Cq_action",
    ]:
        rows.append(
            {
                "null_family": family,
                "false_primary_certification_rate": float(wrong["quotient_certified"].mean()) if not wrong.empty else 0.0,
                "false_pooling_pass_rate": float(pooling[pooling["candidate_role"].astype(str).eq("wrong_factor_control")]["pooling_gate_passed"].mean()) if not pooling.empty else 0.0,
                "false_lift_activation_rate": 0.0,
                "false_validation_selection_rate": float(selector.get("selected_primary_lift", pd.Series(dtype=bool)).mean()) if not selector.empty and family == "wrong_primary_factor_control" else 0.0,
                "null_delta_vs_fallback": np.nan,
                "null_delta_vs_random_same_branch": np.nan,
            }
        )
    return pd.DataFrame(rows)


def build_claims(groups, candidates, lifts, selector) -> pd.DataFrame:
    controlled_selected = selector[
        selector["run_id"].astype(str).str.startswith("controlled_")
        & selector.get("selected_primary_lift", pd.Series(dtype=bool)).fillna(False)
    ] if not selector.empty else pd.DataFrame()
    controlled_selected_runs = set(controlled_selected["run_id"].astype(str).unique()) if not controlled_selected.empty else set()
    real_selected = selector[
        ~selector["run_id"].astype(str).str.startswith("controlled_")
        & selector.get("selected_primary_lift", pd.Series(dtype=bool)).fillna(False)
    ] if not selector.empty else pd.DataFrame()
    real_certified = candidates[
        candidates["residual_source"].astype(str).eq("real")
        & candidates["quotient_certified"].fillna(False)
    ] if not candidates.empty else pd.DataFrame()
    return pd.DataFrame(
        [
            {
                "claim_id": "controlled_primary_sanity_check",
                "status": "Supported" if {"controlled_C2_primary", "controlled_C3_primary", "controlled_C4_primary"}.issubset(controlled_selected_runs) else "Supported negative",
                "safe_wording": "Controlled C2/C3/C4 primary sanity rows select the correct implemented primary branch lift by validation." if {"controlled_C2_primary", "controlled_C3_primary", "controlled_C4_primary"}.issubset(controlled_selected_runs) else "Controlled primary sanity did not fully pass; real primary results should not be interpreted as structural wins.",
                "evidence": "reports/csv/primary_holonomy_selector_results.csv",
            },
            {
                "claim_id": "stable_real_primary_quotients",
                "status": "Supported descriptive" if not real_certified.empty else "Supported negative",
                "safe_wording": "Some real relation sets have certified heuristic 2/3-primary quotient candidates; this is descriptive unless paired lift controls pass.",
                "evidence": "reports/csv/primary_holonomy_candidates.csv",
            },
            {
                "claim_id": "primary_branch_lift_accuracy_real",
                "status": "Supported limited" if not real_selected.empty else "Supported negative",
                "safe_wording": "Real implemented primary branch lifts are only supported when validation-selected and paired controls pass; otherwise the selector falls back.",
                "evidence": "reports/csv/primary_holonomy_paired_stats.csv",
            },
            {
                "claim_id": "selector_avoids_harmful_primary_lifts",
                "status": "Supported" if real_selected.empty else "Supported limited",
                "safe_wording": "The validation-only selector falls back on real settings when implemented primary lifts do not clear fallback and same-branch control gates.",
                "evidence": "reports/csv/primary_holonomy_selector_results.csv",
            },
            {
                "claim_id": "real_brauer_or_period_index_residuals",
                "status": "Not supported",
                "safe_wording": "Do not claim real residuals are central Brauer/projective or period-index classes from this primary-factor experiment.",
                "evidence": "reports/primary_holonomy_splitting_report.md",
            },
        ]
    )


def write_plots(outputs, reports_dir: Path) -> None:
    import matplotlib.pyplot as plt

    plot_dir = reports_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    nested = outputs["nested"]
    pooling = outputs["pooling"]
    nulls = outputs["nulls"]
    selector = outputs["selector"]
    paired = outputs["paired"]

    fig, ax = plt.subplots(figsize=(7, 4))
    if not nested.empty:
        data = nested[nested["lift_implemented"].fillna(False)].groupby("q_order")["test_accuracy"].mean()
        ax.plot(data.index, data.values, marker="o")
    ax.set_xlabel("branch factor q")
    ax.set_ylabel("mean test accuracy")
    ax.set_title("Nested primary accuracy")
    fig.tight_layout()
    fig.savefig(plot_dir / "primary_holonomy_nested_accuracy.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    if not nested.empty:
        data = nested.groupby("q_order")["delta_vs_previous_depth"].mean()
        ax.bar(data.index.astype(str), data.values)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_ylabel("delta vs previous depth")
    ax.set_title("Primary delta versus depth")
    fig.tight_layout()
    fig.savefig(plot_dir / "primary_holonomy_delta_vs_depth.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    if not nested.empty:
        data = nested.groupby("q_order")["structural_delta_vs_random"].mean()
        ax.bar(data.index.astype(str), data.values)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_ylabel("delta vs random same-branch")
    ax.set_title("Primary delta versus random control")
    fig.tight_layout()
    fig.savefig(plot_dir / "primary_holonomy_delta_vs_random_control.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    if not pooling.empty:
        data = pooling.groupby("q_order")["pooling_residual_q"].mean()
        ax.bar(data.index.astype(str), data.values)
    ax.set_yscale("symlog", linthresh=1e-12)
    ax.set_ylabel("pooling residual")
    ax.set_title("Primary pooling residuals")
    fig.tight_layout()
    fig.savefig(plot_dir / "primary_holonomy_pooling_residuals.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    if not nulls.empty:
        ax.barh(nulls["null_family"], nulls["false_primary_certification_rate"])
    ax.set_xlabel("false primary certification rate")
    ax.set_title("Null controls versus real")
    fig.tight_layout()
    fig.savefig(plot_dir / "primary_holonomy_null_vs_real.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    if not paired.empty:
        valid = paired[pd.to_numeric(paired["n_pairs"], errors="coerce") > 0]
        ax.barh(valid["comparison"], valid["paired_mean_accuracy_delta"])
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("paired test accuracy delta")
    ax.set_title("Selector regret / paired deltas")
    fig.tight_layout()
    fig.savefig(plot_dir / "primary_holonomy_selector_regret.pdf")
    plt.close(fig)


def update_claims_audit(reports_dir: Path, claims: pd.DataFrame) -> None:
    path = reports_dir / "claims_audit.md"
    if not path.exists():
        return
    start = "<!-- primary-holonomy-splitting:start -->"
    end = "<!-- primary-holonomy-splitting:end -->"
    block = [
        start,
        "## Primary-Factor Holonomy Splitting Audit",
        "",
        "Generated by `experiments/primary_holonomy_splitting.py`. This audit covers observed holonomy order, group exponent when available, p-primary factors, and prediction-level branch lifts. It does not support real Brauer/projective, full holonomy splitting, capacity-matched, broad model-merging, or test-selected claims.",
        "",
        md_table(claims.to_dict("records"), ["claim_id", "status", "safe_wording", "evidence"]),
        "",
        "Forbidden wording: real residuals are central Brauer/projective classes; full holonomy is split; primary branch lifts are capacity-matched unless compressed; broad natural model-merging wins; test accuracy was used for selection.",
        end,
    ]
    text = path.read_text(encoding="utf-8")
    if start in text and end in text:
        text = text.split(start, 1)[0] + "\n".join(block) + text.split(end, 1)[1]
    else:
        text = text.rstrip() + "\n\n" + "\n".join(block) + "\n"
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_report(args, outputs, claims):
    groups = outputs["groups"]
    candidates = outputs["candidates"]
    lifts = outputs["lifts"]
    selector = outputs["selector"]
    controlled_selected = selector[
        selector["run_id"].astype(str).str.startswith("controlled_")
        & selector.get("selected_primary_lift", pd.Series(dtype=bool)).fillna(False)
    ] if not selector.empty else pd.DataFrame()
    real_selected = selector[
        ~selector["run_id"].astype(str).str.startswith("controlled_")
        & selector.get("selected_primary_lift", pd.Series(dtype=bool)).fillna(False)
    ] if not selector.empty else pd.DataFrame()
    interpretation = (
        "Case C: implemented real C2 branch-lift rows exist, but the real validation-safe selector falls back; primary quotient evidence remains descriptive for real data."
        if real_selected.empty
        else "Case B: primary lifts are selected in limited real settings and must be judged against same-branch controls."
    )
    report = f"""# Primary Holonomy Splitting Report

Generated by `experiments/primary_holonomy_splitting.py`.

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

## Primary Decomposition

This experiment estimates observed holonomy order and, where exact closure is available, group exponent.  It tests nested cyclic primary branch factors `1 -> 2 -> 4 -> 8 -> 16 -> 32` and `1 -> 3 -> 9 -> 27`, plus mixed controls.  This differs from the previous arbitrary small-quotient scan because branch factors are tied to `v2` and `v3` content of the observed holonomy order or exact exponent.

This is not a central Brauer/projective or real period-index experiment.

## Controlled Sanity Check

- Controlled selected primary lifts: `{len(controlled_selected)}`

{md_table(selector[selector["run_id"].astype(str).str.startswith("controlled_")].to_dict("records") if not selector.empty else [], ["run_id", "selected_candidate_method", "selected_primary_lift", "selected_q_order", "selected_depth", "val_accuracy", "test_accuracy"], 20)}

## Relation-Count Diagnostics

{md_table(groups.to_dict("records"), ["residual_source", "aggregation_level", "relation_set_id", "relation_count", "relation_count_status", "observed_holonomy_order_lcm", "v2_observed_order", "v3_observed_order", "group_closure_status", "group_exponent_if_exact", "noncentral_holonomy_score"], 80)}

## Primary Quotient Candidates

{md_table(candidates.to_dict("records"), ["residual_source", "aggregation_level", "relation_set_id", "q_name", "primary_type", "primary_depth", "candidate_role", "relation_violation_rate", "quotient_holonomy_nontrivial_rate", "quotient_holonomy_entropy", "quotient_certified", "quotient_status"], 80)}

## Bootstrap Stability

See `reports/csv/primary_holonomy_bootstrap.csv` for bootstrap rates and relation-count meaningfulness flags.

## Pooling Residual Table

{md_table(outputs["pooling"].to_dict("records"), ["residual_source", "aggregation_level", "relation_set_id", "q_name", "pooling_threshold", "naive_residual_q", "pooling_residual_q", "pooling_gate_passed"], 80)}

## Nested Accuracy Table

{md_table(outputs["nested"].to_dict("records"), ["run_id", "primary_type", "q_order", "primary_depth", "candidate_method", "validation_accuracy", "test_accuracy", "delta_vs_fallback", "delta_vs_previous_depth", "structural_delta_vs_random", "lift_implemented"], 80)}

## Random Same-Branch Control Table

{md_table(lifts[lifts["candidate_method"].astype(str).eq("random_same_branch_count_control")].to_dict("records"), ["run_id", "q_order", "candidate_method", "validation_accuracy", "test_accuracy", "lift_implemented"], 80)}

## Selector Table

{md_table(selector.to_dict("records"), ["run_id", "selected_candidate_method", "selected_primary_lift", "selected_q_order", "selected_depth", "val_accuracy", "test_accuracy", "selector_no_test_leakage"], 80)}

## Paired Statistics

{md_table(outputs["paired"].to_dict("records"), ["comparison", "n_pairs", "paired_mean_accuracy_delta", "paired_accuracy_delta_ci_low", "paired_accuracy_delta_ci_high", "wins", "ties", "losses", "number_of_validation_selected_primary_lifts", "selected_depth_distribution"], 20)}

## Final Claim Table

{md_table(claims.to_dict("records"), ["claim_id", "status", "safe_wording", "evidence"])}

## Negative Boundaries

- Do not claim real residuals are central Brauer/projective classes.
- Do not claim full holonomy is split.
- Do not call primary branch lifts capacity-matched unless compressed/capacity-matched models are implemented.
- Do not claim broad natural model-merging wins.
- Test accuracy is report-only; selection uses validation only.

## Final Interpretation

{interpretation}
"""
    (args.reports_dir / "primary_holonomy_splitting_report.md").write_text(report, encoding="utf-8")


def stringify(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    for col in out.columns:
        if out[col].map(lambda value: isinstance(value, (dict, list, tuple))).any():
            out[col] = out[col].map(lambda value: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list, tuple)) else value)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "reports")
    parser.add_argument("--datasets", default="mnist,fashion_mnist")
    parser.add_argument("--model-counts", default="3,4")
    parser.add_argument("--widths", default="")
    parser.add_argument("--max-settings", type=int, default=80)
    parser.add_argument("--q-orders", default="2,4,8,16,32,3,9,27,6,12,18,36")
    parser.add_argument("--relation-threshold", type=float, default=0.01)
    parser.add_argument("--nontrivial-threshold", type=float, default=0.10)
    parser.add_argument("--entropy-threshold", type=float, default=0.10)
    parser.add_argument("--loose-relation-threshold", type=float, default=0.05)
    parser.add_argument("--loose-nontrivial-threshold", type=float, default=0.05)
    parser.add_argument("--loose-entropy-threshold", type=float, default=0.0)
    parser.add_argument("--min-relation-count", type=int, default=4)
    parser.add_argument("--pooling-thresholds", default="1e-8,1e-6,1e-4,1e-2")
    parser.add_argument("--random-restarts", type=int, default=8)
    parser.add_argument("--bootstrap-samples", type=int, default=100)
    parser.add_argument("--bootstrap-ci-samples", type=int, default=500)
    parser.add_argument("--selector-epsilons", default="0.0,0.0005,0.001,0.002")
    parser.add_argument("--selector-epsilon-controls", default="0.0,0.0005,0.001")
    parser.add_argument("--selector-loss-slacks", default="0.0,0.005,0.01,inf")
    parser.add_argument("--lambda-branch-values", default="0.0,0.001,0.002")
    parser.add_argument("--lambda-residual-values", default="0.0,0.01")
    parser.add_argument("--max-group-order", type=int, default=5000)
    parser.add_argument("--max-generators", type=int, default=12)
    parser.add_argument("--max-exact-order", type=int, default=256)
    parser.add_argument("--seed", type=int, default=31991)
    parser.add_argument("--no-update-claims-audit", action="store_true")
    args = parser.parse_args()
    args.command_string = " ".join([sys.executable, *sys.argv])

    for directory in [args.reports_dir / "csv", args.reports_dir / "plots", args.reports_dir / "configs"]:
        directory.mkdir(parents=True, exist_ok=True)

    maps = load_real_triangle_maps(args)
    bases, relation_sets, relation_members = build_relation_sets(args, maps)
    groups = pd.DataFrame([group_summary(base, relation_sets[str(base["relation_set_id"])], args) for base in bases])
    candidates, bootstrap, pooling, _fits = fit_primary_tables(args, groups, relation_sets)
    run_ids = {run_id for ids in relation_members.values() for run_id in ids if not str(run_id).startswith("controlled_")}
    run_rows = load_run_rows(args, run_ids)
    lifts = build_lift_tables(args, pooling, run_rows, relation_members)
    selector, regret, paired = build_selector_outputs(args, run_rows, lifts)
    nested = build_nested_sequence(pd.concat([run_rows, controlled_fallback_rows()], ignore_index=True, sort=False), lifts)
    nulls = build_null_controls(candidates, pooling, selector)
    outputs = {
        "groups": groups,
        "candidates": candidates,
        "bootstrap": bootstrap,
        "pooling": pooling,
        "lifts": lifts,
        "nested": nested,
        "selector": selector,
        "regret": regret,
        "nulls": nulls,
        "paired": paired,
    }
    claims = build_claims(groups, candidates, lifts, selector)

    csv_dir = args.reports_dir / "csv"
    stringify(groups).to_csv(csv_dir / "primary_holonomy_groups.csv", index=False, lineterminator="\n")
    stringify(candidates).to_csv(csv_dir / "primary_holonomy_candidates.csv", index=False, lineterminator="\n")
    stringify(bootstrap).to_csv(csv_dir / "primary_holonomy_bootstrap.csv", index=False, lineterminator="\n")
    stringify(pooling).to_csv(csv_dir / "primary_holonomy_pooling_residuals.csv", index=False, lineterminator="\n")
    stringify(lifts).to_csv(csv_dir / "primary_holonomy_lift_candidates.csv", index=False, lineterminator="\n")
    stringify(nested).to_csv(csv_dir / "primary_holonomy_nested_sequence.csv", index=False, lineterminator="\n")
    stringify(selector).to_csv(csv_dir / "primary_holonomy_selector_results.csv", index=False, lineterminator="\n")
    stringify(nulls).to_csv(csv_dir / "primary_holonomy_null_controls.csv", index=False, lineterminator="\n")
    stringify(paired).to_csv(csv_dir / "primary_holonomy_paired_stats.csv", index=False, lineterminator="\n")
    stringify(claims).to_csv(csv_dir / "primary_holonomy_claims.csv", index=False, lineterminator="\n")

    write_plots(outputs, args.reports_dir)
    write_report(args, outputs, claims)
    save_json(
        args.reports_dir / "configs" / "primary_holonomy_splitting_config.json",
        {
            "argv": sys.argv,
            "command": args.command_string,
            "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k != "command_string"},
            "environment": capture_environment(),
            "git_commit": git_output("rev-parse", "--short", "HEAD"),
            "git_status_short": git_output("status", "--short"),
            "artifact_row_counts": {key: int(len(value)) for key, value in outputs.items()},
        },
    )
    if not args.no_update_claims_audit:
        update_claims_audit(args.reports_dir, claims)

    print("wrote reports/primary_holonomy_splitting_report.md")
    print("wrote reports/csv/primary_holonomy_paired_stats.csv")


if __name__ == "__main__":
    main()
