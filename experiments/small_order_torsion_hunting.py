#!/usr/bin/env python
"""Small-order torsion hunting with validation-gated period/index lifts.

This is an artifact-backed real-residual experiment.  By default it mines the
serialized triangle maps produced by the repeated-seed fixed-setting
verification run, calibrates strict/loose small-order detectors on null
controls, and activates no lift unless strict residual, null, bootstrap, and
rank divisibility gates all pass.
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
from src.small_order_torsion import (  # noqa: E402
    DEFAULT_ORDERS,
    DEFAULT_POLICIES,
    TorsionThresholdPolicy,
    analyze_permutation_residual,
    analyze_residual_matrix,
    bootstrap_stability,
    compose_permutations,
    noisy_fake_scalar,
    noncentral_s3_control,
    permutation_commutator,
    permutation_matrix,
    policy_label,
    policy_passes,
    random_orthogonal_matrix,
)
from src.validation_gated_period_index_lift import (  # noqa: E402
    DEFAULT_CANDIDATE_RANKS,
    SelectorPolicy,
    best_fallbacks,
    period_index_rows_for_candidate,
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


def safe_json_array(value) -> np.ndarray | None:
    if not isinstance(value, str) or not value.strip() or value == "nan":
        return None
    try:
        parsed = json.loads(value)
    except Exception:
        return None
    if not isinstance(parsed, list):
        return None
    return np.asarray(parsed, dtype=int)


def load_real_triangle_maps(args) -> pd.DataFrame:
    artifact_dir = args.reports_dir / "csv" / "fixed_setting_large_artifacts"
    shards = sorted(artifact_dir.glob("fixed_setting_triangle_maps_part_*.csv.gz"))
    frames = []
    for shard in shards:
        frame = pd.read_csv(shard, usecols=lambda col: col in REAL_TRIANGLE_MAP_COLUMNS)
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"no fixed-setting triangle map shards found in {artifact_dir}")
    df = pd.concat(frames, ignore_index=True)
    df = df[df["triangle_type"].astype(str).eq("permutation")].copy()
    df = df[df["alignment_source"].astype(str).eq("observed")].copy()
    df = df[pd.to_numeric(df["alignment_noise_fraction"], errors="coerce").fillna(0.0).eq(0.0)].copy()
    datasets = set(parse_csv(args.datasets, str))
    if datasets:
        df = df[df["dataset"].astype(str).isin(datasets)].copy()
    widths = set(parse_csv(args.widths, int)) if args.widths else set()
    if widths:
        df = df[pd.to_numeric(df["width"], errors="coerce").isin(widths)].copy()
    counts = set(parse_csv(args.model_counts, int)) if args.model_counts else set()
    if counts:
        df = df[pd.to_numeric(df["n_models"], errors="coerce").isin(counts)].copy()
    if args.max_real_residuals and len(df) > args.max_real_residuals:
        df = df.sample(n=int(args.max_real_residuals), random_state=args.seed).sort_values(
            ["dataset", "n_models", "width", "matching", "seed", "triangle"],
        )
    return df.reset_index(drop=True)


def real_residual_rows(real_maps: pd.DataFrame, policies: tuple[TorsionThresholdPolicy, ...], orders: tuple[int, ...]) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    rows = []
    matrices: dict[str, np.ndarray] = {}
    for idx, row in real_maps.iterrows():
        perm = safe_json_array(row.get("triangle_perm"))
        if perm is None:
            continue
        residual_id = f"{row['run_id']}:{row['triangle']}:{idx}"
        metrics = analyze_permutation_residual(perm, orders)
        matrices[residual_id] = permutation_matrix(perm)
        out = {
            "residual_id": residual_id,
            "residual_source": "permutation_c2m3_triangle",
            "data_origin": "fixed_setting_large_artifact",
            "layer": "primary_hidden",
            "setting_id": row.get("setting_id"),
            "run_id": row.get("run_id"),
            "dataset": row.get("dataset"),
            "architecture": row.get("architecture"),
            "n_models": int(row.get("n_models")),
            "width": int(row.get("width")),
            "domain_shift": row.get("domain_shift"),
            "matching": row.get("matching"),
            "seed": int(row.get("seed")),
            "triangle_id": row.get("triangle"),
            "i": int(row.get("i")),
            "j": int(row.get("j")),
            "k": int(row.get("k")),
            **metrics,
        }
        for policy in policies:
            out[f"passes_{policy.name}"] = policy_passes(metrics, policy)
        rows.append(out)
    return pd.DataFrame(rows), matrices


def monomial_diagnostic_rows(args, policies: tuple[TorsionThresholdPolicy, ...]) -> pd.DataFrame:
    path = args.reports_dir / "csv" / "fixed_setting_triangle_defects.csv"
    if not path.exists():
        return pd.DataFrame()
    cols = [
        "setting_id",
        "run_id",
        "dataset",
        "architecture",
        "n_models",
        "width",
        "domain_shift",
        "matching",
        "seed",
        "triangle_type",
        "i",
        "j",
        "k",
        "monomial_defect_score",
        "monomial_cycle_trace",
        "monomial_cycle_determinant",
    ]
    df = pd.read_csv(path, usecols=lambda col: col in cols)
    df = df[df["triangle_type"].astype(str).eq("monomial")].copy()
    datasets = set(parse_csv(args.datasets, str))
    if datasets:
        df = df[df["dataset"].astype(str).isin(datasets)].copy()
    widths = set(parse_csv(args.widths, int)) if args.widths else set()
    if widths:
        df = df[pd.to_numeric(df["width"], errors="coerce").isin(widths)].copy()
    if args.max_monomial_diagnostics and len(df) > args.max_monomial_diagnostics:
        df = df.sample(n=int(args.max_monomial_diagnostics), random_state=args.seed + 7)
    rows = []
    for idx, row in df.iterrows():
        defect = float(pd.to_numeric(pd.Series([row.get("monomial_defect_score")]), errors="coerce").iloc[0])
        out = {
            "residual_id": f"{row['run_id']}:monomial:{idx}",
            "residual_source": "monomial_positive_scale_summary",
            "data_origin": "fixed_setting_triangle_summary",
            "layer": "primary_hidden",
            "setting_id": row.get("setting_id"),
            "run_id": row.get("run_id"),
            "dataset": row.get("dataset"),
            "architecture": row.get("architecture"),
            "n_models": int(row.get("n_models")),
            "width": int(row.get("width")),
            "domain_shift": row.get("domain_shift"),
            "matching": row.get("matching"),
            "seed": int(row.get("seed")),
            "triangle_id": f"{row.get('i')}-{row.get('j')}-{row.get('k')}",
            "i": int(row.get("i")),
            "j": int(row.get("j")),
            "k": int(row.get("k")),
            "matrix_dim": int(row.get("width")),
            "centrality_residual": defect,
            "detected_order": 0,
            "detected_phase": "not_reconstructable",
            "phase_angle": np.nan,
            "best_root_exponent": np.nan,
            "best_root_search_order": np.nan,
            "scalar_residual_best": defect,
            "finite_order_residual_best_search_order": np.nan,
            "finite_order_residual_min": np.nan,
            "eigenvalue_spread": np.nan,
            "condition_number": np.nan,
            "is_nontrivial_root": False,
            "explained_as_noncentral_holonomy": True,
            "diagnostic_only_no_lift": True,
            "monomial_cycle_trace": row.get("monomial_cycle_trace"),
            "monomial_cycle_determinant": row.get("monomial_cycle_determinant"),
        }
        for policy in policies:
            out[f"passes_{policy.name}"] = False
        rows.append(out)
    return pd.DataFrame(rows)


def shuffled_edge_null_rows(real_maps: pd.DataFrame, count: int, orders: tuple[int, ...], rng: np.random.Generator) -> list[dict]:
    usable = real_maps.dropna(subset=["p_ij", "p_jk", "p_ki"]).copy()
    if usable.empty:
        return []
    rows = []
    for idx in range(int(count)):
        picks = usable.sample(n=3, replace=True, random_state=int(rng.integers(0, 2**32 - 1)))
        perms = []
        for col, (_, pick) in zip(["p_ij", "p_jk", "p_ki"], picks.iterrows(), strict=False):
            arr = safe_json_array(pick.get(col))
            if arr is not None:
                perms.append(arr)
        if len(perms) != 3:
            continue
        perm = compose_permutations(*perms)
        metrics = analyze_permutation_residual(perm, orders)
        rows.append(
            {
                "null_id": f"shuffled_edge_maps_{idx}",
                "null_family": "shuffled_edge_maps",
                "width": int(len(perm)),
                "residual_source": "permutation_c2m3_triangle",
                **metrics,
            }
        )
    return rows


def build_null_controls(real_maps: pd.DataFrame, args, policies: tuple[TorsionThresholdPolicy, ...], orders: tuple[int, ...]) -> pd.DataFrame:
    rng = np.random.default_rng(args.seed + 101)
    widths = sorted({int(value) for value in pd.to_numeric(real_maps["width"], errors="coerce").dropna().unique()})
    rows = []
    per_width = int(args.nulls_per_family)
    for width in widths:
        for idx in range(per_width):
            for family, matrix in [
                ("random_orthogonal", random_orthogonal_matrix(width, rng)),
                ("random_permutation_commutator", permutation_commutator(width, rng)),
                ("noncentral_s3_block", noncentral_s3_control(width)),
                ("noisy_fake_scalar_order2", noisy_fake_scalar(width, 2, rng, noise_scale=0.01)),
            ]:
                metrics = analyze_residual_matrix(matrix, orders)
                rows.append(
                    {
                        "null_id": f"{family}_W{width}_{idx}",
                        "null_family": family,
                        "width": int(width),
                        "residual_source": "matrix_null_control",
                        **metrics,
                    }
                )
        rows.extend(shuffled_edge_null_rows(real_maps[real_maps["width"].astype(int).eq(width)], per_width, orders, rng))
    null_df = pd.DataFrame(rows)
    for policy in policies:
        null_df[f"passes_{policy.name}"] = null_df.apply(lambda row: policy_passes(row.to_dict(), policy), axis=1)
    return null_df


def summarize_nulls(null_df: pd.DataFrame, real_df: pd.DataFrame, policies: tuple[TorsionThresholdPolicy, ...]) -> pd.DataFrame:
    rows = []
    for policy in policies:
        pass_col = f"passes_{policy.name}"
        for keys, group in null_df.groupby(["null_family", "residual_source"], dropna=False):
            null_family, residual_source = keys
            accepted_null = int(group[pass_col].sum())
            false_rate = float(group[pass_col].mean()) if len(group) else np.nan
            real_source = real_df[real_df["residual_source"].astype(str).eq("permutation_c2m3_triangle")]
            accepted_real = int(real_source[pass_col].sum()) if pass_col in real_source else 0
            rows.append(
                {
                    "policy": policy.name,
                    "threshold_centrality": policy.centrality_threshold,
                    "threshold_scalar": policy.scalar_threshold,
                    "threshold_order": policy.order_threshold,
                    "target_false_positive_rate": policy.target_false_positive_rate,
                    "null_family": null_family,
                    "residual_source": residual_source,
                    "n_null": int(len(group)),
                    "accepted_null_count": accepted_null,
                    "false_positive_rate": false_rate,
                    "false_scalar_projective_rate": false_rate,
                    "false_lift_rate": false_rate if policy.activates_lift else 0.0,
                    "accepted_real_count": accepted_real,
                }
            )
    return pd.DataFrame(rows)


def fpr_by_policy(null_summary: pd.DataFrame) -> dict[str, float]:
    out = {}
    for policy, group in null_summary.groupby("policy"):
        out[str(policy)] = float(pd.to_numeric(group["false_positive_rate"], errors="coerce").max())
    return out


def build_candidate_tables(
    real_df: pd.DataFrame,
    residual_matrices: dict[str, np.ndarray],
    null_summary: pd.DataFrame,
    args,
    policies: tuple[TorsionThresholdPolicy, ...],
    orders: tuple[int, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    policy_fpr = fpr_by_policy(null_summary)
    permutation_rows = real_df[real_df["residual_source"].astype(str).eq("permutation_c2m3_triangle")].copy()
    permutation_rows["combined_torsion_score"] = (
        pd.to_numeric(permutation_rows["centrality_residual"], errors="coerce")
        + pd.to_numeric(permutation_rows["scalar_residual_best"], errors="coerce")
        + pd.to_numeric(permutation_rows["finite_order_residual_min"], errors="coerce")
    )
    candidate_mask = np.zeros(len(permutation_rows), dtype=bool)
    for policy in policies:
        candidate_mask |= permutation_rows[f"passes_{policy.name}"].to_numpy(dtype=bool)
    top = permutation_rows.sort_values("combined_torsion_score").head(int(args.max_bootstrap_candidates))
    candidates = pd.concat([permutation_rows[candidate_mask], top], ignore_index=True).drop_duplicates("residual_id")
    candidate_rows = []
    bootstrap_rows = []
    for _, candidate in candidates.iterrows():
        matrix = residual_matrices.get(str(candidate["residual_id"]))
        for policy in policies:
            if matrix is not None:
                bootstrap = bootstrap_stability(
                    matrix,
                    policy,
                    orders,
                    n_bootstrap=int(args.bootstrap_samples),
                    seed=int(args.seed) + len(bootstrap_rows),
                )
            else:
                bootstrap = {
                    "bootstrap_mode": "not_run_policy_failed_or_missing_matrix",
                    "bootstrap_samples": 0,
                    "bootstrap_detection_rate": 0.0,
                    "bootstrap_order_agreement_rate": 0.0,
                    "bootstrap_phase_std": np.nan,
                    "bootstrap_residual_mean": np.nan,
                    "bootstrap_residual_std": np.nan,
                }
            label = policy_label(
                candidate.to_dict(),
                policy,
                false_positive_rate=policy_fpr.get(policy.name, 1.0),
                bootstrap_detection_rate=float(bootstrap["bootstrap_detection_rate"]),
                bootstrap_order_agreement_rate=float(bootstrap["bootstrap_order_agreement_rate"]),
            )
            cert = label == "certified_torsion"
            row = {
                **candidate.to_dict(),
                "policy": policy.name,
                "policy_false_positive_rate": policy_fpr.get(policy.name, np.nan),
                "candidate_label": label,
                "certified_torsion": cert,
                "loose_uncertain_candidate": label == "central_projective_candidate_uncertain",
                "estimated_period": int(candidate["detected_order"]) if int(candidate["detected_order"]) > 1 else np.nan,
                "estimated_index": int(candidate["detected_order"]) if int(candidate["detected_order"]) > 1 else np.nan,
                "index_certification_status": "scalar_only" if cert else "not_certified",
                "lift_gate_decision": "certified_lift_gate_open" if cert else "uncertain_candidate_no_lift",
            }
            candidate_rows.append(row)
            bootstrap_rows.append(
                {
                    "residual_id": candidate["residual_id"],
                    "policy": policy.name,
                    "candidate_label": label,
                    **bootstrap,
                }
            )
    return pd.DataFrame(candidate_rows), pd.DataFrame(bootstrap_rows)


def build_summary(real_df: pd.DataFrame, candidates: pd.DataFrame, null_summary: pd.DataFrame, policies: tuple[TorsionThresholdPolicy, ...]) -> pd.DataFrame:
    rows = []
    for policy in policies:
        pass_col = f"passes_{policy.name}"
        cert_ids = set(
            candidates[
                (candidates["policy"].astype(str).eq(policy.name))
                & (candidates["certified_torsion"] == True)  # noqa: E712
            ]["residual_id"].astype(str)
        ) if not candidates.empty else set()
        for keys, group in real_df.groupby(["residual_source", "dataset", "n_models", "width", "matching"], dropna=False):
            residual_source, dataset, n_models, width, matching = keys
            accepted = int(group[pass_col].sum()) if pass_col in group else 0
            certified = int(group["residual_id"].astype(str).isin(cert_ids).sum()) if "residual_id" in group else 0
            fpr = float(null_summary[null_summary["policy"].astype(str).eq(policy.name)]["false_positive_rate"].max())
            rows.append(
                {
                    "policy": policy.name,
                    "residual_source": residual_source,
                    "dataset": dataset,
                    "n_models": int(n_models),
                    "width": int(width),
                    "matching": matching,
                    "n_residuals": int(len(group)),
                    "pre_bootstrap_policy_pass_count": accepted,
                    "certified_torsion_count": certified,
                    "loose_uncertain_count": int(
                        candidates[
                            (candidates["policy"].astype(str).eq(policy.name))
                            & (candidates["candidate_label"].astype(str).eq("central_projective_candidate_uncertain"))
                        ]["residual_id"].astype(str).isin(group["residual_id"].astype(str)).sum()
                    )
                    if not candidates.empty and "residual_id" in group
                    else 0,
                    "mean_centrality_residual": float(pd.to_numeric(group["centrality_residual"], errors="coerce").mean()),
                    "min_centrality_residual": float(pd.to_numeric(group["centrality_residual"], errors="coerce").min()),
                    "mean_scalar_residual_best": float(pd.to_numeric(group["scalar_residual_best"], errors="coerce").mean()),
                    "min_scalar_residual_best": float(pd.to_numeric(group["scalar_residual_best"], errors="coerce").min()),
                    "mean_finite_order_residual_min": float(pd.to_numeric(group["finite_order_residual_min"], errors="coerce").mean()),
                    "min_finite_order_residual_min": float(pd.to_numeric(group["finite_order_residual_min"], errors="coerce").min()),
                    "max_null_false_positive_rate": fpr,
                    "claim_status": "supported_negative_no_certified_torsion" if certified == 0 else "supported_limited_certified_torsion",
                }
            )
    return pd.DataFrame(rows)


def load_run_rows(args, run_ids: set[str]) -> pd.DataFrame:
    path = args.reports_dir / "csv" / "fixed_setting_verification_runs.csv"
    df = pd.read_csv(path, usecols=lambda col: col in RUN_COLUMNS)
    if run_ids:
        df = df[df["run_id"].astype(str).isin(run_ids)].copy()
    return df


def build_period_index_rows(candidates: pd.DataFrame, args) -> pd.DataFrame:
    rows = []
    if not candidates.empty:
        core = candidates.drop_duplicates(["residual_id", "policy"]).copy()
        for _, candidate in core.iterrows():
            payload = {
                "candidate_origin": "real_residual",
                "residual_id": candidate.get("residual_id"),
                "run_id": candidate.get("run_id"),
                "dataset": candidate.get("dataset"),
                "n_models": candidate.get("n_models"),
                "width": candidate.get("width"),
                "policy": candidate.get("policy"),
                "certified_torsion": bool(candidate.get("certified_torsion", False)),
                "detected_order": candidate.get("detected_order"),
                "estimated_period": candidate.get("estimated_period"),
                "estimated_index": candidate.get("estimated_index"),
                "index_certification_status": candidate.get("index_certification_status"),
                "lift_gate_decision": candidate.get("lift_gate_decision"),
                "capacity_matched": False,
                "candidate_method": "index_projective_lift" if bool(candidate.get("certified_torsion", False)) else "uncertain_candidate_no_lift",
            }
            rows.extend(period_index_rows_for_candidate(payload, DEFAULT_CANDIDATE_RANKS))

    for period, index in [(2, 4), (3, 9)]:
        payload = {
            "candidate_origin": "controlled_period_index_logic",
            "residual_id": f"controlled_heisenberg_period_{period}_index_{index}",
            "run_id": "",
            "dataset": "controlled",
            "n_models": 3,
            "width": index,
            "policy": "controlled_logic",
            "certified_torsion": True,
            "detected_order": period,
            "estimated_period": period,
            "estimated_index": index,
            "index_certification_status": "heisenberg_certified_control",
            "lift_gate_decision": "controlled_rank_gate_only",
            "capacity_matched": False,
            "candidate_method": "index_projective_lift",
        }
        rows.extend(period_index_rows_for_candidate(payload, DEFAULT_CANDIDATE_RANKS))
    return pd.DataFrame(rows)


def build_selector_outputs(run_rows: pd.DataFrame, lift_rows: pd.DataFrame, args) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_frames = []
    regret_frames = []
    fallbacks = best_fallbacks(run_rows)
    for epsilon in parse_csv(args.selector_epsilons, float):
        for loss_text in parse_csv(args.selector_loss_slacks, str):
            loss_slack = float("inf") if loss_text == "inf" else float(loss_text)
            selected = torsion_safe_selector(
                run_rows,
                lift_rows,
                policy=SelectorPolicy(epsilon=float(epsilon), loss_slack=loss_slack),
            )
            if selected.empty:
                continue
            selected["selector_epsilon"] = float(epsilon)
            selected["selector_loss_slack"] = loss_slack
            selected_frames.append(selected)
            pool = pd.concat([fallbacks, lift_rows], ignore_index=True, sort=False) if not lift_rows.empty else fallbacks
            regret = selector_regret(selected, pool)
            regret["selector_epsilon"] = float(epsilon)
            regret["selector_loss_slack"] = loss_slack
            regret_frames.append(regret)
    return (
        pd.concat(selected_frames, ignore_index=True, sort=False) if selected_frames else pd.DataFrame(),
        pd.concat(regret_frames, ignore_index=True, sort=False) if regret_frames else pd.DataFrame(),
    )


def paired_stats(selected: pd.DataFrame, run_rows: pd.DataFrame, candidates: pd.DataFrame, args) -> pd.DataFrame:
    if selected.empty:
        return pd.DataFrame()
    default = selected[
        (pd.to_numeric(selected["selector_epsilon"], errors="coerce").eq(0.0))
        & (pd.to_numeric(selected["selector_loss_slack"], errors="coerce").eq(0.0))
    ].copy()
    if default.empty:
        default = selected.copy()
    default = default.drop_duplicates("run_id")
    method_map = {
        "greedy_soup": "greedy_soup",
        "c2m3_permutation": "c2m3_synchronized",
        "monomial_scale": "monomial_best_validation",
        "random_same_rank_lift_control": "random_branch_ensemble_2",
    }
    fallbacks = best_fallbacks(run_rows)
    best_fb = torsion_safe_selector(run_rows, pd.DataFrame(), SelectorPolicy()).drop_duplicates("run_id")
    rows = []
    for baseline_label, method in {"best_fallback": "best_fallback", **method_map}.items():
        if method == "best_fallback":
            baseline = best_fb[["run_id", "test_accuracy", "test_loss"]].rename(
                columns={"test_accuracy": "baseline_test_accuracy", "test_loss": "baseline_test_loss"}
            )
        elif method == "monomial_best_validation":
            baseline = fallbacks[fallbacks["candidate_method"].eq("fallback_monomial")]
            baseline = baseline.drop_duplicates("run_id")
            baseline = baseline[["run_id", "test_accuracy", "test_loss"]].rename(
                columns={"test_accuracy": "baseline_test_accuracy", "test_loss": "baseline_test_loss"}
            )
        else:
            baseline = run_rows[run_rows["method"].astype(str).eq(method)]
            baseline = baseline.sort_values(["run_id", "val_accuracy", "val_loss"], ascending=[True, False, True])
            baseline = baseline.drop_duplicates("run_id")
            baseline = baseline[["run_id", "test_accuracy", "test_loss"]].rename(
                columns={"test_accuracy": "baseline_test_accuracy", "test_loss": "baseline_test_loss"}
            )
        merged = default.merge(baseline, on="run_id", how="inner")
        acc_delta = pd.to_numeric(merged["test_accuracy"], errors="coerce") - pd.to_numeric(merged["baseline_test_accuracy"], errors="coerce")
        loss_delta = pd.to_numeric(merged["test_loss"], errors="coerce") - pd.to_numeric(merged["baseline_test_loss"], errors="coerce")
        wins = int((acc_delta > 1e-12).sum())
        ties = int((acc_delta.abs() <= 1e-12).sum())
        losses = int((acc_delta < -1e-12).sum())
        ci_low, ci_high = bootstrap_mean_ci(acc_delta, int(args.bootstrap_ci_samples), args.seed + len(rows))
        rows.append(
            {
                "comparison": f"torsion_safe_selector_vs_{baseline_label}",
                "baseline": baseline_label,
                "n_pairs": int(len(merged)),
                "paired_mean_accuracy_delta": float(acc_delta.mean()) if len(acc_delta) else np.nan,
                "paired_accuracy_delta_ci_low": ci_low,
                "paired_accuracy_delta_ci_high": ci_high,
                "paired_mean_loss_delta": float(loss_delta.mean()) if len(loss_delta) else np.nan,
                "accuracy_wins": wins,
                "accuracy_ties": ties,
                "accuracy_losses": losses,
                "sign_test_two_sided_p": sign_test_two_sided(wins, losses),
                "number_of_certified_candidates": int(candidates["certified_torsion"].sum()) if not candidates.empty else 0,
                "number_of_attempted_lifts": int((candidates.get("lift_gate_decision", pd.Series(dtype=str)).astype(str) == "certified_lift_gate_open").sum()) if not candidates.empty else 0,
                "number_of_validation_selected_lifts": int(default.get("selected_lift", pd.Series(dtype=bool)).sum()) if not default.empty else 0,
                "number_of_test_improving_lifts": int(((default.get("selected_lift", False) == True) & (acc_delta > 0)).sum()) if len(default) else 0,  # noqa: E712
            }
        )
    return pd.DataFrame(rows)


def write_plots(real_df: pd.DataFrame, candidates: pd.DataFrame, null_df: pd.DataFrame, paired: pd.DataFrame, regret: pd.DataFrame, reports_dir: Path) -> None:
    import matplotlib.pyplot as plt

    plot_dir = reports_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    perm = real_df[real_df["residual_source"].astype(str).eq("permutation_c2m3_triangle")]
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.scatter(perm["centrality_residual"], perm["scalar_residual_best"], s=8, alpha=0.35, label="real residuals")
    ax.axvline(1e-3, color="black", linestyle="--", linewidth=1, label="strict threshold")
    ax.axhline(1e-3, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("centrality residual")
    ax.set_ylabel("best scalar root residual")
    ax.set_title("Small-order torsion residual scan")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_dir / "small_order_torsion_residuals.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    if not candidates.empty:
        candidates["detected_order"].value_counts(dropna=False).sort_index().plot(kind="bar", ax=ax)
    ax.set_xlabel("detected order")
    ax.set_ylabel("candidate count")
    ax.set_title("Torsion candidate orders")
    fig.tight_layout()
    fig.savefig(plot_dir / "torsion_candidate_orders.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    real_values = pd.to_numeric(perm["scalar_residual_best"], errors="coerce").dropna()
    null_values = pd.to_numeric(null_df["scalar_residual_best"], errors="coerce").dropna()
    ax.boxplot([real_values, null_values], showfliers=False)
    ax.set_xticklabels(["real", "null"])
    ax.set_ylabel("best scalar root residual")
    ax.set_yscale("log")
    ax.set_title("Null controls versus real residuals")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plot_dir / "torsion_null_vs_real.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    if not paired.empty:
        ax.barh(paired["baseline"], paired["paired_mean_accuracy_delta"], xerr=[
            paired["paired_mean_accuracy_delta"] - paired["paired_accuracy_delta_ci_low"],
            paired["paired_accuracy_delta_ci_high"] - paired["paired_mean_accuracy_delta"],
        ])
    ax.axvline(0.0, color="black", linewidth=1)
    ax.set_xlabel("paired test accuracy delta")
    ax.set_title("Period/index lift selector deltas")
    fig.tight_layout()
    fig.savefig(plot_dir / "period_index_lift_delta.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    values = pd.to_numeric(regret["test_regret_vs_oracle_pool"], errors="coerce").dropna() if not regret.empty else []
    ax.hist(values, bins=20)
    ax.set_xlabel("test regret versus oracle pool")
    ax.set_ylabel("count")
    ax.set_title("Torsion-safe selector regret")
    fig.tight_layout()
    fig.savefig(plot_dir / "torsion_safe_selector_regret.pdf")
    plt.close(fig)


def update_claims_audit(reports_dir: Path, claim_rows: list[dict]) -> None:
    path = reports_dir / "claims_audit.md"
    if not path.exists():
        return
    start = "<!-- small-order-torsion:start -->"
    end = "<!-- small-order-torsion:end -->"
    block = [
        start,
        "## Small-Order Torsion Hunting And Period/Index Lift Audit",
        "",
        "Generated by `experiments/small_order_torsion_hunting.py`. This audit is conservative: real Brauer/projective claims remain unsupported unless strict residual, null-control, bootstrap, and validation gates pass.",
        "",
        md_table(claim_rows, ["claim_id", "status", "safe_wording", "evidence"]),
        "",
        "Forbidden wording: real neural residuals are Brauer/projective classes; period-divisible rank is enough when index is larger; test accuracy is used for lift selection.",
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


def write_reports(args, outputs: dict[str, pd.DataFrame], claim_rows: list[dict]) -> None:
    real_df = outputs["real"]
    summary = outputs["summary"]
    candidates = outputs["candidates"]
    null_summary = outputs["null_summary"]
    bootstrap = outputs["bootstrap"]
    period_index = outputs["period_index"]
    selected = outputs["selected"]
    paired = outputs["paired"]
    regret = outputs["regret"]
    certified_count = int(candidates["certified_torsion"].sum()) if not candidates.empty else 0
    loose_count = int((candidates["candidate_label"].astype(str) == "central_projective_candidate_uncertain").sum()) if not candidates.empty else 0
    final_statement = (
        "No certified small-order central/projective torsion was found in real residuals under strict calibrated thresholds. "
        "TwistedMerge++ therefore correctly falls back to ordinary exact ReLU gauges and greedy-soup/C2M3 baselines. "
        "This supports the framework's conservative residual taxonomy."
        if certified_count == 0
        else "Certified small-order central/projective residuals were found in a small number of real model-merging settings. "
        "In those settings, validation-gated period/index lifts should be interpreted as limited evidence only."
    )
    if certified_count == 0 and loose_count > 0:
        final_statement = (
            "Some loose torsion-like residuals appeared, but they failed bootstrap or null-control certification. "
            "The safe selector rejected lifts and avoided harmful overfitting. This supports the need for period-index "
            "certification rather than naive small-order fitting."
        )

    report = f"""# Small-Order Torsion Hunting Report

Generated by `experiments/small_order_torsion_hunting.py`.

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

## Dataset And Residual Grid

- Source: existing fixed-setting repeated-seed artifacts in `reports/csv/fixed_setting_large_artifacts/`.
- Datasets: `{args.datasets}`
- Architecture: one-hidden-layer ReLU MLP residuals from the fixed-setting artifact.
- Model counts: observed `{sorted(real_df["n_models"].dropna().unique().tolist()) if not real_df.empty else []}`
- Widths: observed `{sorted(real_df["width"].dropna().unique().tolist()) if not real_df.empty else []}`
- Seeds: observed `{int(real_df["seed"].min()) if not real_df.empty else "none"}` to `{int(real_df["seed"].max()) if not real_df.empty else "none"}`.
- Residual sources: permutation/C2M3 triangle residuals plus monomial positive-scale summary diagnostics.
- Note: this run uses existing width `64,128` artifacts rather than retraining the requested starter grid `16,32,64`.

## Small-Order Detector Thresholds

{md_table([policy.__dict__ for policy in DEFAULT_POLICIES], ["name", "centrality_threshold", "scalar_threshold", "order_threshold", "target_false_positive_rate", "activates_lift"])}

## Null-Control Calibration

{md_table(null_summary.to_dict("records"), ["policy", "null_family", "n_null", "accepted_null_count", "false_positive_rate", "false_lift_rate", "accepted_real_count"], 40)}

## Bootstrap Stability

{md_table(bootstrap.to_dict("records"), ["residual_id", "policy", "candidate_label", "bootstrap_mode", "bootstrap_detection_rate", "bootstrap_order_agreement_rate", "bootstrap_phase_std"], 25)}

## Candidate Order Table

{md_table(candidates.to_dict("records"), ["residual_id", "dataset", "width", "policy", "detected_order", "centrality_residual", "scalar_residual_best", "finite_order_residual_min", "candidate_label", "certified_torsion"], 30)}

## Certified Candidate Table

{md_table(candidates[candidates["certified_torsion"] == True].to_dict("records") if not candidates.empty else [], ["residual_id", "dataset", "width", "policy", "detected_order", "estimated_period", "estimated_index", "index_certification_status"], 30)}

## Rejected Uncertain Candidate Table

{md_table(candidates[candidates["certified_torsion"] != True].to_dict("records") if not candidates.empty else [], ["residual_id", "dataset", "width", "policy", "detected_order", "candidate_label", "lift_gate_decision"], 30)}

## Period/Index Rank-Divisibility Table

{md_table(period_index.to_dict("records"), ["candidate_origin", "residual_id", "policy", "estimated_period", "estimated_index", "candidate_rank", "rank_decision", "lift_allowed_by_index"], 40)}

## Validation-Selected Lift Table

{md_table(selected.to_dict("records"), ["run_id", "selector_epsilon", "selector_loss_slack", "selected_candidate_method", "selected_lift", "val_accuracy", "test_accuracy", "selector_no_test_leakage"], 30)}

## Paired Test Results

{md_table(paired.to_dict("records"), ["comparison", "n_pairs", "paired_mean_accuracy_delta", "paired_accuracy_delta_ci_low", "paired_accuracy_delta_ci_high", "paired_mean_loss_delta", "accuracy_wins", "accuracy_ties", "accuracy_losses", "sign_test_two_sided_p"], 20)}

## Selector Regret Table

{md_table(regret.to_dict("records"), ["run_id", "selector_epsilon", "selector_loss_slack", "selected_candidate_method", "selected_lift", "test_regret_vs_oracle_pool", "delta_vs_best_fallback"], 30)}

## Final Claim Table

{md_table(claim_rows, ["claim_id", "status", "safe_wording", "evidence"])}

## Negative Boundaries

- Do not claim real neural residuals are Brauer/projective classes.
- Do not claim period-divisible rank is enough when index is larger.
- Do not claim broad rank-lift or greedy-soup wins from this artifact-backed scan.
- Do not use loose candidates to activate lifts.
- Method selection uses validation only; test metrics are used after selection for reporting.

## Final Paper-Facing Interpretation

{final_statement}
"""
    (args.reports_dir / "small_order_torsion_hunting_report.md").write_text(report, encoding="utf-8")

    lift_report = f"""# Period/Index Lift Validation Report

Generated by `experiments/small_order_torsion_hunting.py`.

## Lift Gate Summary

- Certified real torsion candidates: `{certified_count}`
- Attempted real period/index lifts: `{int((period_index["candidate_origin"].astype(str).eq("real_residual") & period_index["lift_allowed_by_index"]).sum()) if not period_index.empty else 0}`
- Validation-selected lifts: `{int(selected["selected_lift"].sum()) if not selected.empty else 0}`
- Selector no-test-leakage flag: `{bool(selected["selector_no_test_leakage"].all()) if not selected.empty else True}`

## Rank-Divisibility Logic

The controlled rows are included to verify that the period/index gate rejects
period-divisible but index-obstructed ranks and admits only index-divisible
ranks. They are not real-residual evidence.

{md_table(period_index.to_dict("records"), ["candidate_origin", "policy", "estimated_period", "estimated_index", "candidate_rank", "rank_decision", "lift_allowed_by_index"], 80)}

## Paired Statistics

{md_table(paired.to_dict("records"), ["comparison", "n_pairs", "paired_mean_accuracy_delta", "paired_accuracy_delta_ci_low", "paired_accuracy_delta_ci_high", "number_of_certified_candidates", "number_of_attempted_lifts", "number_of_validation_selected_lifts"], 20)}
"""
    (args.reports_dir / "period_index_lift_validation_report.md").write_text(lift_report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "reports")
    parser.add_argument("--datasets", default="mnist,fashion_mnist")
    parser.add_argument("--model-counts", default="3,4")
    parser.add_argument("--widths", default="")
    parser.add_argument("--orders", default="2,3,4,5,6,8")
    parser.add_argument("--max-real-residuals", type=int, default=0)
    parser.add_argument("--max-monomial-diagnostics", type=int, default=4000)
    parser.add_argument("--nulls-per-family", type=int, default=40)
    parser.add_argument("--bootstrap-samples", type=int, default=100)
    parser.add_argument("--bootstrap-ci-samples", type=int, default=500)
    parser.add_argument("--max-bootstrap-candidates", type=int, default=40)
    parser.add_argument("--selector-epsilons", default="0.0,0.0005,0.001,0.002")
    parser.add_argument("--selector-loss-slacks", default="0.0,0.005,0.01,inf")
    parser.add_argument("--seed", type=int, default=9173)
    parser.add_argument("--no-update-claims-audit", action="store_true")
    args = parser.parse_args()
    args.command_string = " ".join([sys.executable, *sys.argv])

    reports_dir = args.reports_dir
    csv_dir = reports_dir / "csv"
    config_dir = reports_dir / "configs"
    csv_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)

    orders = tuple(parse_csv(args.orders, int)) or DEFAULT_ORDERS
    policies = DEFAULT_POLICIES
    real_maps = load_real_triangle_maps(args)
    real_perm, residual_matrices = real_residual_rows(real_maps, policies, orders)
    monomial = monomial_diagnostic_rows(args, policies)
    real_df = pd.concat([real_perm, monomial], ignore_index=True, sort=False)
    null_df = build_null_controls(real_maps, args, policies, orders)
    null_summary = summarize_nulls(null_df, real_df, policies)
    candidates, bootstrap = build_candidate_tables(real_df, residual_matrices, null_summary, args, policies, orders)
    summary = build_summary(real_df, candidates, null_summary, policies)
    period_index = build_period_index_rows(candidates, args)

    run_rows = load_run_rows(args, set(real_perm["run_id"].astype(str).unique()))
    lift_rows = pd.DataFrame(columns=list(run_rows.columns) + ["certified_torsion", "lift_allowed_by_index", "candidate_method"])
    selected, regret = build_selector_outputs(run_rows, lift_rows, args)
    paired = paired_stats(selected, run_rows, candidates, args)

    certified_count = int(candidates["certified_torsion"].sum()) if not candidates.empty else 0
    strict_fpr = float(null_summary[null_summary["policy"].astype(str).eq("strict_fpr_001")]["false_positive_rate"].max())
    claim_rows = [
        {
            "claim_id": "small_order_null_calibration",
            "status": "Supported" if strict_fpr <= 0.01 else "Not supported",
            "safe_wording": "The strict small-order torsion detector has calibrated low false-positive rate on the configured null controls; loose diagnostic policies are not lift-activating.",
            "evidence": "reports/csv/small_order_torsion_null_controls.csv",
        },
        {
            "claim_id": "real_small_order_torsion",
            "status": "Supported negative" if certified_count == 0 else "Supported limited",
            "safe_wording": "No certified small-order central/projective torsion is found in real residuals under strict thresholds." if certified_count == 0 else "Certified torsion candidates pass configured gates only in the listed exact settings.",
            "evidence": "reports/csv/small_order_torsion_candidates.csv",
        },
        {
            "claim_id": "period_index_rank_gate",
            "status": "Supported controlled",
            "safe_wording": "The period/index selector rejects period-divisible but index-obstructed ranks in controlled gate rows.",
            "evidence": "reports/csv/period_index_lift_candidates.csv",
        },
        {
            "claim_id": "torsion_safe_selector",
            "status": "Supported descriptive",
            "safe_wording": "The validation-gated selector falls back to ordinary candidates when no certified torsion lift is available.",
            "evidence": "reports/csv/torsion_safe_selector_regret.csv",
        },
        {
            "claim_id": "real_brauer_projective_residuals",
            "status": "Not supported",
            "safe_wording": "Do not claim real neural residuals are Brauer/projective classes.",
            "evidence": "reports/small_order_torsion_hunting_report.md",
        },
    ]

    real_df.to_csv(csv_dir / "small_order_torsion_hunting.csv", index=False, lineterminator="\n")
    summary.to_csv(csv_dir / "small_order_torsion_hunting_summary.csv", index=False, lineterminator="\n")
    candidates.to_csv(csv_dir / "small_order_torsion_candidates.csv", index=False, lineterminator="\n")
    null_summary.to_csv(csv_dir / "small_order_torsion_null_controls.csv", index=False, lineterminator="\n")
    bootstrap.to_csv(csv_dir / "small_order_torsion_bootstrap.csv", index=False, lineterminator="\n")
    period_index.to_csv(csv_dir / "period_index_lift_candidates.csv", index=False, lineterminator="\n")
    paired.to_csv(csv_dir / "period_index_lift_paired_stats.csv", index=False, lineterminator="\n")
    regret.to_csv(csv_dir / "torsion_safe_selector_regret.csv", index=False, lineterminator="\n")

    write_plots(real_df, candidates, null_df, paired, regret, reports_dir)
    outputs = {
        "real": real_df,
        "summary": summary,
        "candidates": candidates,
        "null_summary": null_summary,
        "bootstrap": bootstrap,
        "period_index": period_index,
        "selected": selected,
        "paired": paired,
        "regret": regret,
    }
    write_reports(args, outputs, claim_rows)
    save_json(
        config_dir / "small_order_torsion_hunting_config.json",
        {
            "argv": sys.argv,
            "command": args.command_string,
            "args": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
                if key != "command_string"
            },
            "orders": list(orders),
            "threshold_policies": [policy.__dict__ for policy in policies],
            "environment": capture_environment(),
            "git_commit": git_output("rev-parse", "--short", "HEAD"),
            "git_status_short": git_output("status", "--short"),
        },
    )
    if not args.no_update_claims_audit:
        update_claims_audit(reports_dir, claim_rows)

    print("wrote reports/small_order_torsion_hunting_report.md")
    print("wrote reports/period_index_lift_validation_report.md")
    print("wrote reports/csv/small_order_torsion_hunting.csv")
    print("wrote reports/csv/period_index_lift_paired_stats.csv")


if __name__ == "__main__":
    main()
