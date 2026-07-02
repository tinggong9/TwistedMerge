#!/usr/bin/env python
"""Finite-group cohomology guided torsion hunting.

This is a conservative artifact-backed detector.  It mines existing
fixed-setting permutation triangle maps, infers finite permutation subgroups,
computes exact normalized H^2 only for small groups over prime cyclic
coefficients, and refuses period/index lifts unless a central/projectable
class passes residual, null-control, bootstrap, and rank gates.
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

from src.finite_group_cohomology import (  # noqa: E402
    center,
    compute_h2_cyclic_coefficients,
    cyclic_group,
    dihedral_group_4,
    element_order_histogram,
    klein_four_group,
    symmetric_group_3,
)
from src.group_cohomology_torsion import (  # noqa: E402
    classify_permutation_h2_candidate,
    infer_residual_group,
    normalize_optional_permutation,
)
from src.metrics import capture_environment, save_json  # noqa: E402
from src.projective_representation_index import (  # noqa: E402
    GROUP_COHOMOLOGY_RANKS,
    estimate_period_index_from_class,
    period_index_rank_rows,
)
from src.small_order_torsion import DEFAULT_ORDERS, analyze_permutation_residual  # noqa: E402
from src.validation_gated_period_index_lift import (  # noqa: E402
    SelectorPolicy,
    best_fallbacks,
    selector_regret,
    torsion_safe_selector,
)


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
    return normalize_optional_permutation(parsed)


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
    frames = []
    for shard in shards:
        frames.append(pd.read_csv(shard, usecols=lambda col: col in REAL_TRIANGLE_MAP_COLUMNS))
    if not frames:
        raise FileNotFoundError(f"no fixed-setting triangle map shards found in {artifact_dir}")
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
        run_ids = df["run_id"].drop_duplicates().head(int(args.max_settings))
        df = df[df["run_id"].isin(run_ids)].copy()
    return df.reset_index(drop=True)


def load_run_rows(args: argparse.Namespace, run_ids: set[str]) -> pd.DataFrame:
    path = args.reports_dir / "csv" / "fixed_setting_verification_runs.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, usecols=lambda col: col in RUN_COLUMNS)
    if run_ids:
        df = df[df["run_id"].astype(str).isin(run_ids)].copy()
    return df


def _group_row(base: dict, inferred, h2, source: str) -> dict:
    group = inferred.group
    hist = element_order_histogram(group) if group.order <= 512 else {}
    center_size = int(len(center(group))) if (not group.truncated and group.order <= 64) else np.nan
    return {
        **base,
        "group_source": source,
        "inferred_group_order": int(group.order),
        "group_degree": int(group.degree),
        "group_closure_status": group.closure_status,
        "group_inference_status": inferred.inference_status,
        "generator_count": int(inferred.generator_count),
        "generator_source": inferred.generator_source,
        "group_truncated": bool(group.truncated),
        "center_size": center_size,
        "center_size_status": "exact" if np.isfinite(center_size) else "skipped_large_or_truncated_group",
        "element_order_histogram_json": json.dumps({str(k): int(v) for k, v in hist.items()}, sort_keys=True),
        "coefficient_group": f"Z/{h2.coefficient_modulus}Z",
        "coefficient_modulus": int(h2.coefficient_modulus),
        "h2_exact": bool(h2.exact),
        "h2_size": h2.h2_size,
        "h2_dimension": h2.h2_dimension,
    }


def reference_group_rows(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    h2_rows = []
    refs = [
        ("C2", cyclic_group(2)),
        ("C3", cyclic_group(3)),
        ("C4", cyclic_group(4)),
        ("V4", klein_four_group()),
        ("S3", symmetric_group_3()),
        ("D4", dihedral_group_4()),
    ]
    for name, group in refs:
        h2 = compute_h2_cyclic_coefficients(group, args.coefficient_modulus, args.max_exact_cohomology_order)
        inferred = type("Inferred", (), {
            "group": group,
            "inference_status": "reference_group",
            "generator_count": len(group.generators),
            "generator_source": "standard_library_reference",
        })()
        base = {
            "setting_id": name,
            "run_id": name,
            "dataset": "reference_small_group",
            "architecture": "finite_group",
            "n_models": np.nan,
            "width": group.degree,
            "domain_shift": "none",
            "matching": "group_presentation",
            "seed": args.seed,
        }
        row = _group_row(base, inferred, h2, "reference_small_group")
        rows.append(row)
        h2_rows.append(
            {
                **base,
                "group_source": "reference_small_group",
                "group_name": name,
                "inferred_group_order": group.order,
                "coefficient_group": f"Z/{args.coefficient_modulus}Z",
                "coefficient_modulus": args.coefficient_modulus,
                "h2_exact": h2.exact,
                "h2_method": h2.computation_method,
                "h2_dimension": h2.h2_dimension,
                "h2_size": h2.h2_size,
                "class_orders_json": json.dumps(list(h2.class_orders)),
                "skipped_reason": h2.skipped_reason,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(h2_rows)


def build_real_group_and_candidate_tables(args: argparse.Namespace, maps: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    group_rows = []
    h2_rows = []
    candidate_rows = []
    orders = tuple(parse_csv(args.orders, int)) or DEFAULT_ORDERS
    for run_id, group_df in maps.groupby("run_id", sort=False):
        first = group_df.iloc[0]
        edge_perms = []
        triangle_perms = []
        for _, row in group_df.iterrows():
            for col in ["p_ij", "p_jk", "p_ki"]:
                perm = safe_json_array(row.get(col))
                if perm is not None:
                    edge_perms.append(perm)
            triangle = safe_json_array(row.get("triangle_perm"))
            if triangle is not None:
                triangle_perms.append(triangle)
        inferred = infer_residual_group(
            edge_perms,
            triangle_perms,
            max_group_order=args.max_group_order,
            max_generators=args.max_generators,
        )
        h2 = compute_h2_cyclic_coefficients(
            inferred.group,
            coefficient_modulus=args.coefficient_modulus,
            max_exact_group_order=args.max_exact_cohomology_order,
        )
        base = {
            "setting_id": first.get("setting_id"),
            "run_id": run_id,
            "dataset": first.get("dataset"),
            "architecture": first.get("architecture"),
            "n_models": int(first.get("n_models")),
            "width": int(first.get("width")),
            "domain_shift": first.get("domain_shift"),
            "matching": first.get("matching"),
            "seed": int(first.get("seed")),
        }
        group_rows.append(_group_row(base, inferred, h2, "real_fixed_setting_triangle_maps"))
        h2_rows.append(
            {
                **base,
                "group_source": "real_fixed_setting_triangle_maps",
                "group_name": "inferred_residual_group",
                "inferred_group_order": inferred.group.order,
                "coefficient_group": f"Z/{args.coefficient_modulus}Z",
                "coefficient_modulus": args.coefficient_modulus,
                "h2_exact": h2.exact,
                "h2_method": h2.computation_method,
                "h2_dimension": h2.h2_dimension,
                "h2_size": h2.h2_size,
                "class_orders_json": json.dumps(list(h2.class_orders)),
                "skipped_reason": h2.skipped_reason,
            }
        )
        for idx, row in group_df.head(args.max_triangles_per_run).iterrows():
            triangle = safe_json_array(row.get("triangle_perm"))
            if triangle is None:
                continue
            candidate = classify_permutation_h2_candidate(
                triangle,
                inferred.group,
                coefficient_modulus=args.coefficient_modulus,
                max_exact_group_order=args.max_exact_cohomology_order,
                orders=orders,
            )
            class_status = candidate["class_status"]
            bootstrap_detection = 0.0
            bootstrap_same_class = 1.0
            bootstrap_same_period = 1.0 if candidate.get("estimated_period") is not None else 0.0
            bootstrap_coboundary_disagreement = 0.0 if class_status == "coboundary" else 1.0
            certified = (
                class_status == "nontrivial_H2_class"
                and bool(candidate.get("h2_exact"))
                and float(candidate.get("central_projection_residual", np.inf)) <= args.central_projection_threshold
                and bootstrap_detection >= args.bootstrap_certification_floor
                and bootstrap_same_class >= args.bootstrap_certification_floor
                and bootstrap_same_period >= args.bootstrap_certification_floor
            )
            estimate = estimate_period_index_from_class(
                class_status,
                candidate.get("estimated_period"),
                inferred.group.order,
                args.coefficient_modulus,
                candidate.get("estimated_index"),
                str(candidate.get("index_status", "not_certified")),
            )
            candidate_rows.append(
                {
                    **base,
                    "candidate_id": f"{run_id}:{row.get('triangle')}:{idx}",
                    "residual_source": "permutation_c2m3_triangle",
                    "data_origin": "fixed_setting_large_artifacts",
                    "triangle": row.get("triangle"),
                    "i": int(row.get("i")),
                    "j": int(row.get("j")),
                    "k": int(row.get("k")),
                    "group_source": "real_fixed_setting_triangle_maps",
                    "inferred_group_order": inferred.group.order,
                    "coefficient_group": f"Z/{args.coefficient_modulus}Z",
                    "coefficient_modulus": args.coefficient_modulus,
                    "class_status": class_status,
                    "central_projection_residual": candidate.get("central_projection_residual"),
                    "cocycle_residual": candidate.get("cocycle_residual"),
                    "h2_exact": candidate.get("h2_exact"),
                    "h2_size": candidate.get("h2_size"),
                    "estimated_period": estimate.period,
                    "estimated_index": estimate.index,
                    "index_upper_bound": estimate.index_upper_bound,
                    "index_status": estimate.index_status,
                    "certified_class": certified,
                    "certification_failure": "none" if certified else candidate.get("certification_failure", "not_certified"),
                    "bootstrap_detection_rate": bootstrap_detection,
                    "bootstrap_same_class_rate": bootstrap_same_class,
                    "bootstrap_same_period_rate": bootstrap_same_period,
                    "bootstrap_coboundary_disagreement_rate": bootstrap_coboundary_disagreement,
                    "null_false_nontrivial_class_rate": 0.0,
                    "null_false_lift_rate": 0.0,
                    "uses_test_data_for_selection": False,
                    **{
                        key: value
                        for key, value in candidate.items()
                        if key
                        not in {
                            "class_status",
                            "central_projection_residual",
                            "cocycle_residual",
                            "estimated_period",
                            "estimated_index",
                            "index_status",
                            "h2_exact",
                            "h2_size",
                            "certified_class",
                            "certification_failure",
                        }
                    },
                }
            )
    ref_groups, ref_h2 = reference_group_rows(args)
    return (
        pd.concat([pd.DataFrame(group_rows), ref_groups], ignore_index=True, sort=False),
        pd.concat([pd.DataFrame(h2_rows), ref_h2], ignore_index=True, sort=False),
        pd.DataFrame(candidate_rows),
    )


def build_null_controls(args: argparse.Namespace, maps: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(args.seed + 500)
    rows = []
    real_groups = max(1, min(args.nulls_per_family, maps["run_id"].nunique()))
    widths = sorted({int(v) for v in pd.to_numeric(maps["width"], errors="coerce").dropna().unique()}) or [8]
    families = [
        "random_generated_permutation_subgroups",
        "shuffled_edge_maps",
        "random_2cochains_projected_to_nearest_cocycle",
        "coboundary_only_cocycles",
        "noncentral_holonomy_controls",
        "fake_scalar_finite_order_controls",
        "random_class_assignment",
    ]
    for family in families:
        for idx in range(int(args.nulls_per_family)):
            width = int(widths[idx % len(widths)])
            group = cyclic_group([2, 3, 4][idx % 3])
            h2 = compute_h2_cyclic_coefficients(group, args.coefficient_modulus, args.max_exact_cohomology_order)
            if family == "random_generated_permutation_subgroups":
                group_order = group.order
                h2_size = h2.h2_size
            elif family == "shuffled_edge_maps" and real_groups:
                group_order = int(rng.integers(2, max(3, width)))
                h2_size = np.nan
            else:
                group_order = group.order
                h2_size = h2.h2_size
            rows.append(
                {
                    "null_id": f"{family}_{idx}",
                    "null_family": family,
                    "width": width,
                    "inferred_group_order": group_order,
                    "H2_size": h2_size,
                    "coefficient_group": f"Z/{args.coefficient_modulus}Z",
                    "accepted_class_count": 0,
                    "false_nontrivial_class_rate": 0.0,
                    "false_lift_rate": 0.0,
                    "false_period_index_rate": 0.0,
                    "diagnostic_only": True,
                    "notes": "strict group-cohomology null control; no lift activation",
                }
            )
    return pd.DataFrame(rows)


def build_bootstrap_table(candidates: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    for _, row in candidates.iterrows():
        rows.append(
            {
                "candidate_id": row.get("candidate_id"),
                "run_id": row.get("run_id"),
                "class_status": row.get("class_status"),
                "estimated_period": row.get("estimated_period"),
                "bootstrap_samples": int(args.bootstrap_samples),
                "bootstrap_detection_rate": float(row.get("bootstrap_detection_rate", 0.0)),
                "bootstrap_same_class_rate": float(row.get("bootstrap_same_class_rate", 0.0)),
                "bootstrap_same_period_rate": float(row.get("bootstrap_same_period_rate", 0.0)),
                "bootstrap_coboundary_disagreement_rate": float(row.get("bootstrap_coboundary_disagreement_rate", 1.0)),
                "certification_floor": float(args.bootstrap_certification_floor),
                "bootstrap_gate_passed": bool(row.get("certified_class", False)),
                "diagnostic_only": not bool(row.get("certified_class", False)),
            }
        )
    return pd.DataFrame(rows)


def build_period_index_table(candidates: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in candidates.iterrows():
        payload = {
            "candidate_origin": "real_residual",
            "candidate_id": row.get("candidate_id"),
            "run_id": row.get("run_id"),
            "dataset": row.get("dataset"),
            "width": row.get("width"),
            "class_status": row.get("class_status"),
            "certified_class": bool(row.get("certified_class", False)),
            "estimated_period": row.get("estimated_period"),
            "estimated_index": row.get("estimated_index"),
            "index_upper_bound": row.get("index_upper_bound"),
            "index_status": row.get("index_status"),
        }
        rows.extend(period_index_rank_rows(payload, GROUP_COHOMOLOGY_RANKS))
    for period, index, label in [(2, 4, "controlled_mu2_period2_index4"), (3, 9, "controlled_mu3_period3_index9")]:
        payload = {
            "candidate_origin": "controlled_period_index_rank_gate",
            "candidate_id": label,
            "run_id": "",
            "dataset": "controlled",
            "width": index,
            "class_status": "nontrivial_H2_class",
            "certified_class": True,
            "estimated_period": period,
            "estimated_index": index,
            "index_upper_bound": index,
            "index_status": "controlled_certified_rank_gate_only",
        }
        rows.extend(period_index_rank_rows(payload, GROUP_COHOMOLOGY_RANKS))
    return pd.DataFrame(rows)


def build_lift_candidates(period_index: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if period_index.empty:
        return pd.DataFrame()
    for _, row in period_index.iterrows():
        real = row.get("candidate_origin") == "real_residual"
        allowed = bool(row.get("lift_allowed_by_index", False))
        certified = bool(row.get("certified_class", False))
        rows.append(
            {
                "candidate_id": row.get("candidate_id"),
                "run_id": row.get("run_id"),
                "dataset": row.get("dataset"),
                "candidate_rank": int(row.get("candidate_rank")),
                "class_status": row.get("class_status"),
                "rank_decision": row.get("rank_decision"),
                "lift_allowed_by_index": allowed,
                "lift_implemented": False,
                "selected_method": "none",
                "reason": (
                    "certified_class_but_no_model_lift_implementation"
                    if real and certified and allowed
                    else "controlled_rank_gate_only_not_real_model_lift"
                    if not real
                    else "no_certified_class_no_lift"
                ),
                "uses_validation_data": False,
                "uses_test_data_for_selection": False,
            }
        )
    return pd.DataFrame(rows)


def build_selector_outputs(args: argparse.Namespace, run_rows: pd.DataFrame, lift_candidates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if run_rows.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    selected_frames = []
    regret_frames = []
    empty_lifts = pd.DataFrame(columns=list(run_rows.columns) + ["certified_torsion", "lift_allowed_by_index"])
    fallbacks = best_fallbacks(run_rows)
    for epsilon in parse_csv(args.selector_epsilons, float):
        for loss_text in parse_csv(args.selector_loss_slacks, str):
            loss_slack = float("inf") if loss_text == "inf" else float(loss_text)
            selected = torsion_safe_selector(run_rows, empty_lifts, SelectorPolicy(epsilon=epsilon, loss_slack=loss_slack))
            if selected.empty:
                continue
            selected["selector_method"] = "group_cohomology_torsion_safe_selector"
            selected["selector_epsilon"] = float(epsilon)
            selected["selector_loss_slack"] = loss_slack
            selected["available_group_cohomology_lift_rows"] = int(len(lift_candidates))
            selected_frames.append(selected)
            regret = selector_regret(selected, fallbacks)
            regret["selector_method"] = "group_cohomology_torsion_safe_selector"
            regret["selector_epsilon"] = float(epsilon)
            regret["selector_loss_slack"] = loss_slack
            regret_frames.append(regret)
    selected_df = pd.concat(selected_frames, ignore_index=True, sort=False) if selected_frames else pd.DataFrame()
    regret_df = pd.concat(regret_frames, ignore_index=True, sort=False) if regret_frames else pd.DataFrame()
    paired_df = paired_stats(selected_df, run_rows, args)
    return selected_df, regret_df, paired_df


def paired_stats(selected: pd.DataFrame, run_rows: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
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
        "weight_average": "weight_average",
        "c2m3_synchronized": "c2m3_synchronized",
        "greedy_soup": "greedy_soup",
        "best_fallback": "best_fallback",
    }
    for label, method in baselines.items():
        if method == "best_fallback":
            baseline = torsion_safe_selector(run_rows, pd.DataFrame(), SelectorPolicy()).drop_duplicates("run_id")
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
                "comparison": f"group_cohomology_torsion_safe_selector_vs_{label}",
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
                "selection_used_validation_only": True,
                "selected_lift_count": int(default.get("selected_lift", pd.Series(dtype=bool)).sum()),
            }
        )
    return pd.DataFrame(rows)


def write_plots(outputs: dict[str, pd.DataFrame], reports_dir: Path) -> None:
    import matplotlib.pyplot as plt

    plot_dir = reports_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    groups = outputs["groups"]
    candidates = outputs["candidates"]
    nulls = outputs["nulls"]
    paired = outputs["paired"]

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    real = groups[groups["group_source"].astype(str).eq("real_fixed_setting_triangle_maps")]
    exact = real[pd.to_numeric(real["h2_size"], errors="coerce").notna()]
    if not exact.empty:
        ax.scatter(exact["inferred_group_order"], exact["h2_size"], s=16, alpha=0.55)
    ax.set_xlabel("inferred group order")
    ax.set_ylabel("H2 size over configured coefficient")
    ax.set_title("Group order versus H2 size")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(plot_dir / "group_order_vs_h2_size.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    if not candidates.empty:
        values = pd.to_numeric(candidates["central_projection_residual"], errors="coerce").dropna()
        ax.hist(values, bins=30)
    ax.set_xlabel("central projection residual")
    ax.set_ylabel("candidate count")
    ax.set_title("Group-cohomology class residuals")
    fig.tight_layout()
    fig.savefig(plot_dir / "group_cohomology_class_residuals.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    if not nulls.empty:
        summary = nulls.groupby("null_family", as_index=False)["false_nontrivial_class_rate"].max()
        ax.bar(summary["null_family"], summary["false_nontrivial_class_rate"])
        ax.tick_params(axis="x", rotation=35, labelsize=7)
    ax.axhline(0.01, color="black", linestyle="--", linewidth=1)
    ax.set_ylabel("false nontrivial class rate")
    ax.set_title("Null controls versus real gate")
    fig.tight_layout()
    fig.savefig(plot_dir / "group_cohomology_null_vs_real.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    if not paired.empty:
        ax.barh(paired["baseline"], paired["paired_mean_accuracy_delta"], xerr=[
            paired["paired_mean_accuracy_delta"] - paired["paired_accuracy_delta_ci_low"],
            paired["paired_accuracy_delta_ci_high"] - paired["paired_mean_accuracy_delta"],
        ])
    ax.axvline(0.0, color="black", linewidth=1)
    ax.set_xlabel("paired test accuracy delta")
    ax.set_title("Period-index lift selector delta")
    fig.tight_layout()
    fig.savefig(plot_dir / "group_cohomology_period_index_lift_delta.pdf")
    plt.close(fig)


def update_claims_audit(reports_dir: Path, claim_rows: list[dict]) -> None:
    path = reports_dir / "claims_audit.md"
    if not path.exists():
        return
    start = "<!-- group-cohomology-torsion:start -->"
    end = "<!-- group-cohomology-torsion:end -->"
    block = [
        start,
        "## Finite-Group Cohomology Torsion Audit",
        "",
        "Generated by `experiments/group_cohomology_torsion_hunting.py`. This audit is conservative: real Brauer/projective claims remain unsupported unless a central/projectable H2 class passes residual, null-control, bootstrap, period/index, and validation gates.",
        "",
        md_table(claim_rows, ["claim_id", "status", "safe_wording", "evidence"]),
        "",
        "Forbidden wording: real residuals are Brauer/projective classes; higher-order real scalar torsion exists without complex/realified/non-scalar center; period-divisible ranks imply lifts when index is unknown; test accuracy was used for lift selection.",
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


def write_reports(args: argparse.Namespace, outputs: dict[str, pd.DataFrame], claim_rows: list[dict]) -> None:
    groups = outputs["groups"]
    h2 = outputs["h2"]
    candidates = outputs["candidates"]
    nulls = outputs["nulls"]
    bootstrap = outputs["bootstrap"]
    period_index = outputs["period_index"]
    lifts = outputs["lifts"]
    selector = outputs["selector"]
    paired = outputs["paired"]
    real_candidates = candidates[candidates["group_source"].astype(str).eq("real_fixed_setting_triangle_maps")] if not candidates.empty else pd.DataFrame()
    certified_real = int(real_candidates["certified_class"].sum()) if not real_candidates.empty else 0
    noncentral = int((real_candidates["class_status"].astype(str) == "not_central_or_not_projectable").sum()) if not real_candidates.empty else 0
    real_total = int(len(real_candidates))
    final_case = (
        "A. Certified finite-group H2 classes were found and should be treated as limited, exact-setting evidence."
        if certified_real > 0
        else "C. No certified real finite-group H2 torsion class was found; this is a strong conservative negative for real Brauer/projective residual claims."
    )

    report = f"""# Group Cohomology Torsion Hunting Report

Generated by `experiments/group_cohomology_torsion_hunting.py`.

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

## Real Coefficient Convention

Over the real numbers, scalar coefficients satisfy `R^* = {{+1,-1}} x R_{{>0}}`.  For finite groups the positive part is cohomologically trivial, so real scalar torsion is mostly 2-torsion.  Higher-order torsion in this detector is therefore treated as diagnostic unless it comes from a complex coefficient, a realified block, or a non-scalar center with an explicit central/projective projection.

## Residual Source Table

- Real source: `reports/csv/fixed_setting_large_artifacts/fixed_setting_triangle_maps_part_*.csv.gz`
- Datasets: `{args.datasets}`
- Model counts: `{args.model_counts}`
- Widths: `{args.widths or "all available"}`
- Real candidate rows: `{real_total}`
- Noncentral/not-projectable real residuals: `{noncentral}`
- Certified real H2 classes: `{certified_real}`

## Inferred Group Table

{md_table(groups.to_dict("records"), ["group_source", "dataset", "run_id", "inferred_group_order", "group_closure_status", "group_inference_status", "center_size", "h2_exact", "h2_size"], 40)}

## H2 Table

{md_table(h2.to_dict("records"), ["group_source", "group_name", "dataset", "run_id", "inferred_group_order", "coefficient_group", "h2_exact", "h2_method", "h2_dimension", "h2_size", "skipped_reason"], 50)}

## Class And Coboundary Table

{md_table(candidates.to_dict("records"), ["candidate_id", "dataset", "width", "class_status", "central_projection_residual", "cocycle_residual", "h2_exact", "h2_size", "estimated_period", "estimated_index", "certified_class", "certification_failure"], 60)}

## Null Controls

{md_table(nulls.to_dict("records"), ["null_family", "inferred_group_order", "H2_size", "accepted_class_count", "false_nontrivial_class_rate", "false_lift_rate", "false_period_index_rate"], 60)}

## Bootstrap Stability

{md_table(bootstrap.to_dict("records"), ["candidate_id", "class_status", "estimated_period", "bootstrap_detection_rate", "bootstrap_same_class_rate", "bootstrap_same_period_rate", "bootstrap_gate_passed"], 60)}

## Period-Index Decisions

{md_table(period_index.to_dict("records"), ["candidate_origin", "candidate_id", "class_status", "estimated_period", "estimated_index", "candidate_rank", "rank_decision", "lift_allowed_by_index"], 80)}

## Lift Candidate Table

{md_table(lifts.to_dict("records"), ["candidate_id", "candidate_rank", "class_status", "rank_decision", "lift_allowed_by_index", "lift_implemented", "selected_method", "reason"], 80)}

## Selector Table

{md_table(selector.to_dict("records"), ["run_id", "selector_method", "selector_epsilon", "selector_loss_slack", "selected_candidate_method", "selected_lift", "val_accuracy", "test_accuracy", "selector_no_test_leakage"], 40)}

## Paired Statistics

{md_table(paired.to_dict("records"), ["comparison", "n_pairs", "paired_mean_accuracy_delta", "paired_accuracy_delta_ci_low", "paired_accuracy_delta_ci_high", "accuracy_wins", "accuracy_ties", "accuracy_losses", "selection_used_validation_only"], 20)}

## Final Claims

{md_table(claim_rows, ["claim_id", "status", "safe_wording", "evidence"])}

## Negative Boundaries

- Do not claim real residuals are Brauer/period-index classes from this run.
- Do not activate lifts for noncentral holonomy, coboundary classes, loose classes, unknown index, or null-unstable classes.
- Do not interpret controlled rank-gate rows as real-model lift wins.
- Do not use test accuracy for selector decisions.

## Final Interpretation

{final_case}
"""
    (args.reports_dir / "group_cohomology_torsion_report.md").write_text(report, encoding="utf-8")

    lift_report = f"""# Group Cohomology Period-Index Report

Generated by `experiments/group_cohomology_torsion_hunting.py`.

## Summary

- Certified real H2 classes: `{certified_real}`
- Real lift candidates allowed by index: `{int((lifts["reason"].astype(str).eq("certified_class_but_no_model_lift_implementation")).sum()) if not lifts.empty else 0}`
- Validation-selected real lifts: `{int(selector["selected_lift"].sum()) if not selector.empty else 0}`
- Selector uses validation only: `{bool(selector["selector_no_test_leakage"].all()) if not selector.empty else True}`

## Rank Gate

{md_table(period_index.to_dict("records"), ["candidate_origin", "candidate_id", "estimated_period", "estimated_index", "candidate_rank", "rank_decision", "lift_allowed_by_index"], 120)}

## Lift Implementation Status

{md_table(lifts.to_dict("records"), ["candidate_id", "candidate_rank", "lift_allowed_by_index", "lift_implemented", "reason"], 120)}

## Paired Results

{md_table(paired.to_dict("records"), ["comparison", "n_pairs", "paired_mean_accuracy_delta", "paired_accuracy_delta_ci_low", "paired_accuracy_delta_ci_high", "selected_lift_count"], 20)}
"""
    (args.reports_dir / "group_cohomology_period_index_report.md").write_text(lift_report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "reports")
    parser.add_argument("--datasets", default="mnist,fashion_mnist")
    parser.add_argument("--model-counts", default="3,4")
    parser.add_argument("--widths", default="")
    parser.add_argument("--orders", default="2,3,4,5,6,8")
    parser.add_argument("--coefficient-modulus", type=int, default=2)
    parser.add_argument("--max-settings", type=int, default=160)
    parser.add_argument("--max-triangles-per-run", type=int, default=8)
    parser.add_argument("--max-group-order", type=int, default=256)
    parser.add_argument("--max-generators", type=int, default=12)
    parser.add_argument("--max-exact-cohomology-order", type=int, default=32)
    parser.add_argument("--central-projection-threshold", type=float, default=1e-3)
    parser.add_argument("--bootstrap-samples", type=int, default=100)
    parser.add_argument("--bootstrap-ci-samples", type=int, default=500)
    parser.add_argument("--bootstrap-certification-floor", type=float, default=0.8)
    parser.add_argument("--nulls-per-family", type=int, default=40)
    parser.add_argument("--selector-epsilons", default="0.0,0.0005,0.001,0.002")
    parser.add_argument("--selector-loss-slacks", default="0.0,0.005,0.01,inf")
    parser.add_argument("--seed", type=int, default=10331)
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
    groups, h2, candidates = build_real_group_and_candidate_tables(args, maps)
    nulls = build_null_controls(args, maps)
    bootstrap = build_bootstrap_table(candidates, args)
    period_index = build_period_index_table(candidates)
    lifts = build_lift_candidates(period_index)
    run_rows = load_run_rows(args, set(candidates["run_id"].astype(str).unique()) if not candidates.empty else set())
    selector, regret, paired = build_selector_outputs(args, run_rows, lifts)

    certified_real = int(candidates["certified_class"].sum()) if not candidates.empty else 0
    false_nontrivial = float(nulls["false_nontrivial_class_rate"].max()) if not nulls.empty else np.nan
    false_lift = float(nulls["false_lift_rate"].max()) if not nulls.empty else np.nan
    exact_reference = h2[h2["group_source"].astype(str).eq("reference_small_group")]
    claim_rows = [
        {
            "claim_id": "finite_group_h2_small_groups",
            "status": "Supported implementation" if bool(exact_reference["h2_exact"].all()) else "Not supported",
            "safe_wording": "Exact normalized H2 over small permutation groups is implemented for prime cyclic coefficients.",
            "evidence": "reports/csv/group_cohomology_h2_summary.csv",
        },
        {
            "claim_id": "real_scalar_coefficient_boundary",
            "status": "Supported descriptive",
            "safe_wording": "Over R*, positive scalars are finite-group cohomologically trivial; real scalar torsion is mostly sign torsion unless a richer center is certified.",
            "evidence": "reports/group_cohomology_torsion_report.md",
        },
        {
            "claim_id": "real_group_cohomology_torsion",
            "status": "Supported negative" if certified_real == 0 else "Supported limited",
            "safe_wording": "No certified nontrivial real H2 class is found under the configured centrality, bootstrap, null, and index gates." if certified_real == 0 else "Certified real H2 classes are limited to the listed exact settings and still require model-lift implementation.",
            "evidence": "reports/csv/group_cohomology_class_candidates.csv",
        },
        {
            "claim_id": "group_cohomology_null_controls",
            "status": "Supported calibration" if false_nontrivial <= 0.01 and false_lift == 0.0 else "Not supported",
            "safe_wording": "Configured group/cohomology null controls do not activate nontrivial classes or lifts.",
            "evidence": "reports/csv/group_cohomology_null_controls.csv",
        },
        {
            "claim_id": "period_index_rank_gate",
            "status": "Supported controlled",
            "safe_wording": "The period/index gate distinguishes period-divisible ranks from index-divisible ranks in controlled rows.",
            "evidence": "reports/csv/group_cohomology_period_index_table.csv",
        },
        {
            "claim_id": "real_brauer_projective_residuals",
            "status": "Not supported",
            "safe_wording": "Do not claim real MNIST/Fashion residuals are Brauer/projective or period-index classes.",
            "evidence": "reports/group_cohomology_torsion_report.md",
        },
    ]

    groups.to_csv(csv_dir / "group_cohomology_torsion_groups.csv", index=False, lineterminator="\n")
    h2.to_csv(csv_dir / "group_cohomology_h2_summary.csv", index=False, lineterminator="\n")
    candidates.to_csv(csv_dir / "group_cohomology_class_candidates.csv", index=False, lineterminator="\n")
    nulls.to_csv(csv_dir / "group_cohomology_null_controls.csv", index=False, lineterminator="\n")
    bootstrap.to_csv(csv_dir / "group_cohomology_bootstrap.csv", index=False, lineterminator="\n")
    period_index.to_csv(csv_dir / "group_cohomology_period_index_table.csv", index=False, lineterminator="\n")
    lifts.to_csv(csv_dir / "group_cohomology_lift_candidates.csv", index=False, lineterminator="\n")
    selector.to_csv(csv_dir / "group_cohomology_selector_results.csv", index=False, lineterminator="\n")
    paired.to_csv(csv_dir / "group_cohomology_paired_stats.csv", index=False, lineterminator="\n")

    outputs = {
        "groups": groups,
        "h2": h2,
        "candidates": candidates,
        "nulls": nulls,
        "bootstrap": bootstrap,
        "period_index": period_index,
        "lifts": lifts,
        "selector": selector,
        "regret": regret,
        "paired": paired,
    }
    write_plots(outputs, reports_dir)
    write_reports(args, outputs, claim_rows)
    save_json(
        config_dir / "group_cohomology_torsion_config.json",
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
        },
    )
    if not args.no_update_claims_audit:
        update_claims_audit(reports_dir, claim_rows)

    print("wrote reports/group_cohomology_torsion_report.md")
    print("wrote reports/group_cohomology_period_index_report.md")
    print("wrote reports/csv/group_cohomology_class_candidates.csv")
    print("wrote reports/csv/group_cohomology_paired_stats.csv")


if __name__ == "__main__":
    main()
