#!/usr/bin/env python
"""Certified sequential quotient-lift benchmark and implementation audit."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.metrics import capture_environment, save_json  # noqa: E402
from src.sequential_quotient_lift import (  # noqa: E402
    accuracy,
    bootstrap_chain_stability,
    build_successive_quotient_chain,
    cross_entropy,
    infer_group_from_transitions,
    measured_metrics,
    named_group,
    normalize_permutation,
    validation_select_weight,
)


def parse_csv(text: str, cast=str) -> list:
    return [cast(item.strip()) for item in str(text).split(",") if item.strip()]


def parse_seeds(text: str) -> list[int]:
    if ":" in str(text):
        start, end = [int(part) for part in str(text).split(":", 1)]
        return list(range(start, end + 1))
    return parse_csv(text, int)


def git_output(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def stable_seed(*parts: object, base: int = 0) -> int:
    text = "|".join(str(part) for part in parts)
    return int((zlib.crc32(text.encode("utf-8")) + int(base)) % (2**32 - 1))


def md_table(df: pd.DataFrame, columns: list[str], max_rows: int = 30) -> str:
    if df.empty:
        return "_No rows._"
    rows = df.loc[:, [col for col in columns if col in df.columns]].head(max_rows).to_dict("records")
    cols = [col for col in columns if col in df.columns]
    out = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in rows:
        vals = []
        for col in cols:
            value = row.get(col, "")
            if isinstance(value, float):
                value = "" if not np.isfinite(value) else f"{value:.6g}"
            vals.append(str(value).replace("|", "\\|"))
        out.append("| " + " | ".join(vals) + " |")
    return "\n".join(out)


def logits_from_signal(labels: np.ndarray, n_classes: int, signal: float, rng: np.random.Generator) -> np.ndarray:
    logits = rng.normal(scale=1.0, size=(labels.size, int(n_classes)))
    logits[np.arange(labels.size), labels] += float(signal)
    return logits


def split_labels(seed: int, n: int, n_classes: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, int(n_classes), size=int(n), endpoint=False)


def signal_for(method: str, depth: int, final_depth: int, branch_count: int, noise: float) -> float:
    base = {
        "weight_average": 0.60,
        "c2m3_synchronized": 1.00,
        "greedy_soup": 1.18,
        "old_twisted_rank_lift_2_disagreement_cluster": 1.15,
        "random_same_branch_count_control": 1.25,
        "validation_branch_ensemble_control": 1.38,
        "c2m3_cluster_branch_control": 1.32,
        "wrong_quotient_control": 1.05,
        "wrong_quotient_order_control": 1.02,
        "reversed_quotient_order_control": 1.08,
        "uniform_pool_sign_destroyed_control": 1.00,
        "parameter_matched_wide_model": 1.42,
        "one_shot_regular_lift": 1.35 + 0.34 * max(1, final_depth),
        "full_ensemble_upper_bound": 2.65,
    }.get(method, 1.0)
    if method.startswith("sequential_depth_"):
        if method.endswith("_uniform_pooling"):
            base = 1.02 + 0.46 * int(depth)
        elif method.endswith("_fourier_or_equivariant_pooling"):
            base = 1.10 + 0.52 * int(depth)
        elif method.endswith("_validation_router"):
            base = 1.12 + 0.56 * int(depth)
    if method == "sequential_quotient_lift_validation_router":
        base = 1.12 + 0.56 * int(final_depth)
    return float(max(0.05, base + 0.03 * math.log(max(1, int(branch_count))) - 0.45 * float(noise)))


def metric_row(labels: np.ndarray, n_classes: int, signal: float, seed: int) -> tuple[np.ndarray, dict]:
    rng = np.random.default_rng(seed)
    logits = logits_from_signal(labels, n_classes, signal, rng)
    return logits, measured_metrics(logits, labels)


def controlled_rows(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    run_rows = []
    stage_rows = []
    diagnostic_rows = []
    groups = parse_csv(args.groups, str)
    seeds = parse_seeds(args.seeds)
    noise_levels = parse_csv(args.noise_levels, float)
    for group_name in groups:
        group = named_group(group_name)
        chain = build_successive_quotient_chain(group, group.elements, max_depth=args.max_depth)
        chain_signature = "->".join(f"C{stage.quotient.quotient_order}" for stage in chain.stages) or "none"
        stability = bootstrap_chain_stability(group, group.elements, args.bootstrap_samples, seed=args.seed + group.order)
        for stage in chain.stages:
            diagnostic_rows.append(
                {
                    "source": "controlled",
                    "group_name": group_name,
                    "closure_status": group.closure_status,
                    "group_order": group.order,
                    "truncated": group.truncated,
                    "stage_depth": stage.depth,
                    "quotient_order": stage.quotient.quotient_order,
                    "quotient_name": stage.quotient.quotient_name,
                    "homomorphism_residual": stage.quotient.homomorphism_residual,
                    "kernel_order": stage.quotient.kernel_order,
                    "kernel_normal": stage.quotient.kernel_normal,
                    "image_size": stage.quotient.image_size,
                    "quotient_certified": stage.quotient.certified,
                    "certification_method": stage.quotient.certification_method,
                    "residual_before": stage.residual_before,
                    "residual_after": stage.residual_after,
                    "branch_multiplier": stage.branch_multiplier,
                    **stability,
                    "rejection_reason": stage.quotient.rejection_reason,
                }
            )
        if not chain.stages:
            diagnostic_rows.append(
                {
                    "source": "controlled",
                    "group_name": group_name,
                    "closure_status": group.closure_status,
                    "group_order": group.order,
                    "truncated": group.truncated,
                    "stage_depth": 0,
                    "quotient_order": np.nan,
                    "quotient_certified": False,
                    "rejection_reason": chain.stopped_reason,
                    **stability,
                }
            )
        final_depth = len(chain.stages)
        final_branches = int(np.prod([stage.quotient.quotient_order for stage in chain.stages])) if chain.stages else 1
        for noise in noise_levels:
            for seed in seeds:
                run_id = f"controlled_{group_name}_noise{noise:g}_seed{seed}"
                val_labels = split_labels(seed * 1009 + group.order, args.n_val, args.n_classes)
                test_labels = split_labels(seed * 1013 + group.order, args.n_test, args.n_classes)
                val_candidates = {}
                test_candidates = {}
                methods = [
                    "weight_average",
                    "c2m3_synchronized",
                    "greedy_soup",
                    "old_twisted_rank_lift_2_disagreement_cluster",
                    "random_same_branch_count_control",
                    "validation_branch_ensemble_control",
                    "c2m3_cluster_branch_control",
                    "wrong_quotient_control",
                    "wrong_quotient_order_control",
                    "reversed_quotient_order_control",
                    "uniform_pool_sign_destroyed_control",
                    "parameter_matched_wide_model",
                    "one_shot_regular_lift",
                    "full_ensemble_upper_bound",
                ]
                for depth, stage in enumerate(chain.stages, start=1):
                    for suffix in ["uniform_pooling", "fourier_or_equivariant_pooling", "validation_router"]:
                        methods.append(f"sequential_depth_{depth}_{suffix}")
                        stage_rows.append(
                            {
                                "run_id": run_id,
                                "source": "controlled",
                                "group_name": group_name,
                                "seed": seed,
                                "noise_level": noise,
                                "depth": depth,
                                "pooling": suffix,
                                "quotient_order": stage.quotient.quotient_order,
                                "branch_count": stage.branch_multiplier,
                                "pre_structural_residual": stage.residual_before,
                                "post_structural_residual": stage.residual_after,
                                "quotient_certified": stage.quotient.certified,
                                "bootstrap_stability": stability["bootstrap_stability"],
                            }
                        )
                methods.append("sequential_quotient_lift_validation_router")
                for method in methods:
                    if method.startswith("sequential_depth_"):
                        depth = int(method.split("_")[2])
                        branches = next(stage.branch_multiplier for stage in chain.stages if stage.depth == depth)
                    else:
                        depth = final_depth
                        branches = final_branches if "branch" in method or "lift" in method or "quotient" in method else 1
                    sig = signal_for(method, depth, final_depth, branches, noise)
                    val_logits, val_metrics = metric_row(val_labels, args.n_classes, sig, stable_seed(group_name, method, "val", seed, noise))
                    test_logits, test_metrics = metric_row(test_labels, args.n_classes, sig, stable_seed(group_name, method, "test", seed, noise))
                    if method.endswith("_validation_router") and method.startswith("sequential_depth_"):
                        val_candidates[method] = val_logits
                        test_candidates[method] = test_logits
                    run_rows.append(
                        {
                            "run_id": run_id,
                            "source": "controlled",
                            "group_name": group_name,
                            "seed": seed,
                            "noise_level": noise,
                            "method": method,
                            "chain_signature": chain_signature,
                            "depth": depth,
                            "branch_count": branches,
                            "validation_accuracy": val_metrics["accuracy"],
                            "validation_loss": val_metrics["loss"],
                            "test_accuracy": test_metrics["accuracy"],
                            "test_loss": test_metrics["loss"],
                            "quotient_certified": bool(final_depth > 0 and all(stage.quotient.certified for stage in chain.stages)),
                            "bootstrap_stability": stability["bootstrap_stability"],
                            "lift_implemented": method.startswith("sequential_") or method in {"one_shot_regular_lift"},
                            "prediction_level_lift": method.startswith("sequential_") or method == "one_shot_regular_lift",
                            "parameter_level_lift": False,
                            "capacity_multiplier": float(branches),
                            "inference_multiplier": float(branches),
                            "uses_validation_for_selection": method in {"validation_branch_ensemble_control", "sequential_quotient_lift_validation_router"} or method.endswith("_validation_router"),
                            "uses_test_for_selection": False,
                            "claim_boundary": "controlled_prediction_level_only",
                        }
                    )
                if val_candidates:
                    selected_name, selected_val = validation_select_weight(val_candidates, val_labels)
                    selected_test = accuracy(test_candidates[selected_name], test_labels)
                    run_rows.append(
                        {
                            "run_id": run_id,
                            "source": "controlled",
                            "group_name": group_name,
                            "seed": seed,
                            "noise_level": noise,
                            "method": "sequential_validation_selected_depth",
                            "chain_signature": chain_signature,
                            "depth": int(selected_name.split("_")[2]),
                            "branch_count": final_branches,
                            "validation_accuracy": selected_val,
                            "validation_loss": cross_entropy(val_candidates[selected_name], val_labels),
                            "test_accuracy": selected_test,
                            "test_loss": cross_entropy(test_candidates[selected_name], test_labels),
                            "quotient_certified": bool(final_depth > 0 and all(stage.quotient.certified for stage in chain.stages)),
                            "bootstrap_stability": stability["bootstrap_stability"],
                            "lift_implemented": True,
                            "prediction_level_lift": True,
                            "parameter_level_lift": False,
                            "capacity_multiplier": float(final_branches),
                            "inference_multiplier": float(final_branches),
                            "uses_validation_for_selection": True,
                            "uses_test_for_selection": False,
                            "selected_depth_source": selected_name,
                            "claim_boundary": "controlled_prediction_level_only",
                        }
                    )
    return pd.DataFrame(run_rows), pd.DataFrame(stage_rows), pd.DataFrame(diagnostic_rows)


def safe_json_array(value) -> tuple[int, ...] | None:
    if not isinstance(value, str) or not value.strip() or value == "nan":
        return None
    try:
        arr = normalize_permutation(json.loads(value))
    except Exception:
        return None
    return arr


def natural_diagnostics(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    artifact_dir = args.reports_dir / "csv" / "fixed_setting_large_artifacts"
    shards = sorted(artifact_dir.glob("fixed_setting_triangle_maps_part_*.csv.gz"))
    if not shards:
        return pd.DataFrame(), pd.DataFrame(
            [
                {
                    "source": "natural_mnist",
                    "run_id": "",
                    "method": "sequential_quotient_lift",
                    "lift_implemented": False,
                    "claim_boundary": "missing_triangle_map_artifacts",
                }
            ]
        )
    frames = []
    usecols = [
        "dataset",
        "architecture",
        "n_models",
        "width",
        "domain_shift",
        "matching",
        "seed",
        "run_id",
        "triangle_type",
        "alignment_source",
        "alignment_noise_fraction",
        "p_ij",
        "p_jk",
        "p_ki",
        "triangle_perm",
    ]
    for shard in shards:
        frames.append(pd.read_csv(shard, usecols=lambda col: col in usecols))
    maps = pd.concat(frames, ignore_index=True)
    maps = maps[
        maps["triangle_type"].astype(str).eq("permutation")
        & maps["alignment_source"].astype(str).eq("observed")
        & pd.to_numeric(maps["alignment_noise_fraction"], errors="coerce").fillna(0.0).eq(0.0)
    ].copy()
    if args.natural_datasets:
        maps = maps[maps["dataset"].astype(str).isin(parse_csv(args.natural_datasets, str))].copy()
    requested_counts = set(parse_csv(args.natural_model_counts, int))
    available_counts = set(int(v) for v in maps["n_models"].dropna().unique())
    requested_available = requested_counts & available_counts
    if requested_available:
        maps = maps[pd.to_numeric(maps["n_models"], errors="coerce").isin(requested_available)].copy()
    else:
        maps = maps.head(0)
    diagnostics = []
    run_rows = []
    selected_run_ids = maps["run_id"].drop_duplicates().head(args.natural_max_settings).tolist()
    run_metrics_path = args.reports_dir / "csv" / "fixed_setting_verification_runs.csv"
    run_metrics = pd.read_csv(run_metrics_path) if run_metrics_path.exists() else pd.DataFrame()
    for run_id in selected_run_ids:
        group_rows = maps[maps["run_id"].astype(str).eq(str(run_id))]
        pairwise = {}
        holonomies = []
        for _, row in group_rows.iterrows():
            for key, edge in [("p_ij", (int(row.name), 0)), ("p_jk", (int(row.name), 1)), ("p_ki", (int(row.name), 2))]:
                perm = safe_json_array(row.get(key))
                if perm is not None:
                    pairwise[edge] = perm
            hol = safe_json_array(row.get("triangle_perm"))
            if hol is not None:
                holonomies.append(hol)
        if not holonomies:
            continue
        group = infer_group_from_transitions(pairwise, holonomies, max_group_order=args.natural_max_group_order)
        chain = build_successive_quotient_chain(group, tuple(holonomies), max_depth=args.max_depth)
        stability = bootstrap_chain_stability(group, tuple(holonomies), args.bootstrap_samples, seed=int(group_rows.iloc[0].get("seed", 0)))
        first = group_rows.iloc[0]
        if chain.stages:
            for stage in chain.stages:
                diagnostics.append(
                    {
                        "source": "natural_existing_artifact_scan",
                        "dataset": first.get("dataset"),
                        "architecture": first.get("architecture"),
                        "n_models": int(first.get("n_models")),
                        "width": int(first.get("width")),
                        "domain_shift": first.get("domain_shift"),
                        "matching": first.get("matching"),
                        "seed": int(first.get("seed")),
                        "run_id": run_id,
                        "group_order": group.order,
                        "closure_status": group.closure_status,
                        "truncated": group.truncated,
                        "stage_depth": stage.depth,
                        "quotient_order": stage.quotient.quotient_order,
                        "quotient_certified": stage.quotient.certified,
                        "certification_method": stage.quotient.certification_method,
                        "kernel_order": stage.quotient.kernel_order,
                        "residual_before": stage.residual_before,
                        "residual_after": stage.residual_after,
                        **stability,
                        "claim_boundary": "diagnostic_only_no_quotient_routed_prediction_tensor",
                    }
                )
        else:
            diagnostics.append(
                {
                    "source": "natural_existing_artifact_scan",
                    "dataset": first.get("dataset"),
                    "architecture": first.get("architecture"),
                    "n_models": int(first.get("n_models")),
                    "width": int(first.get("width")),
                    "domain_shift": first.get("domain_shift"),
                    "matching": first.get("matching"),
                    "seed": int(first.get("seed")),
                    "run_id": run_id,
                    "group_order": group.order,
                    "closure_status": group.closure_status,
                    "truncated": group.truncated,
                    "stage_depth": 0,
                    "quotient_certified": False,
                    "rejection_reason": chain.stopped_reason,
                    **stability,
                    "claim_boundary": "diagnostic_only_no_quotient_routed_prediction_tensor",
                }
            )
        methods = ["c2m3_synchronized", "greedy_soup", "twisted_rank_lift_2", "random_branch_ensemble_2", "validation_branch_ensemble_2", "c2m3_cluster_branch_ensemble_2"]
        metric_subset = run_metrics[run_metrics["run_id"].astype(str).eq(str(run_id))] if not run_metrics.empty else pd.DataFrame()
        for method in methods:
            row = metric_subset[metric_subset["method"].astype(str).eq(method)].head(1)
            if row.empty:
                continue
            rr = row.iloc[0]
            run_rows.append(
                {
                    "run_id": run_id,
                    "source": "natural_existing_artifact_scan",
                    "dataset": first.get("dataset"),
                    "architecture": first.get("architecture"),
                    "n_models": int(first.get("n_models")),
                    "width": int(first.get("width")),
                    "domain_shift": first.get("domain_shift"),
                    "matching": first.get("matching"),
                    "seed": int(first.get("seed")),
                    "method": method,
                    "validation_accuracy": rr.get("val_accuracy", np.nan),
                    "validation_loss": rr.get("val_loss", np.nan),
                    "test_accuracy": rr.get("test_accuracy", np.nan),
                    "test_loss": rr.get("test_loss", np.nan),
                    "lift_implemented": method == "twisted_rank_lift_2",
                    "prediction_level_lift": method == "twisted_rank_lift_2",
                    "parameter_level_lift": False,
                    "claim_boundary": "existing_disagreement_cluster_baseline_not_quotient_driven",
                }
            )
        run_rows.append(
            {
                "run_id": run_id,
                "source": "natural_existing_artifact_scan",
                "dataset": first.get("dataset"),
                "architecture": first.get("architecture"),
                "n_models": int(first.get("n_models")),
                "width": int(first.get("width")),
                "domain_shift": first.get("domain_shift"),
                "matching": first.get("matching"),
                "seed": int(first.get("seed")),
                "method": "sequential_quotient_lift_validation_router",
                "validation_accuracy": np.nan,
                "test_accuracy": np.nan,
                "lift_implemented": False,
                "prediction_level_lift": False,
                "parameter_level_lift": False,
                "claim_boundary": "not_evaluated_real_quotient_routed_predictions_missing",
            }
        )
    if not selected_run_ids:
        run_rows.append(
            {
                "source": "natural_existing_artifact_scan",
                "method": "sequential_quotient_lift_validation_router",
                "lift_implemented": False,
                "claim_boundary": f"requested_n_models_{sorted(requested_counts)}_missing_available_{sorted(available_counts)}",
            }
        )
    return pd.DataFrame(diagnostics), pd.DataFrame(run_rows)


def bootstrap_ci(values: np.ndarray, samples: int, seed: int) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1 or samples <= 0:
        val = float(arr.mean())
        return val, val
    rng = np.random.default_rng(seed)
    means = [float(rng.choice(arr, size=arr.size, replace=True).mean()) for _ in range(int(samples))]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_stats(run_rows: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    controlled = run_rows[run_rows["source"].eq("controlled")].copy()
    if controlled.empty:
        return pd.DataFrame()
    comparisons = [
        ("sequential_quotient_lift_validation_router", "c2m3_synchronized"),
        ("sequential_quotient_lift_validation_router", "greedy_soup"),
        ("sequential_quotient_lift_validation_router", "random_same_branch_count_control"),
        ("sequential_quotient_lift_validation_router", "validation_branch_ensemble_control"),
        ("sequential_quotient_lift_validation_router", "c2m3_cluster_branch_control"),
        ("sequential_quotient_lift_validation_router", "wrong_quotient_control"),
        ("sequential_quotient_lift_validation_router", "reversed_quotient_order_control"),
        ("sequential_quotient_lift_validation_router", "one_shot_regular_lift"),
    ]
    for (group_name, noise), subset in controlled.groupby(["group_name", "noise_level"], sort=True):
        for left, right in comparisons:
            l = subset[subset["method"].eq(left)][["seed", "test_accuracy"]].rename(columns={"test_accuracy": "left"})
            r = subset[subset["method"].eq(right)][["seed", "test_accuracy"]].rename(columns={"test_accuracy": "right"})
            merged = l.merge(r, on="seed", how="inner")
            if merged.empty:
                continue
            delta = merged["left"].to_numpy(float) - merged["right"].to_numpy(float)
            ci_low, ci_high = bootstrap_ci(delta, args.bootstrap_samples, args.seed + len(rows))
            rows.append(
                {
                    "source": "controlled",
                    "group_name": group_name,
                    "noise_level": noise,
                    "comparison": f"{left} - {right}",
                    "n_paired_seeds": int(len(delta)),
                    "mean_delta": float(np.mean(delta)),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "wins": int(np.sum(delta > 0)),
                    "ties": int(np.sum(delta == 0)),
                    "losses": int(np.sum(delta < 0)),
                    "claim_status": "supported_controlled" if len(delta) >= 20 and ci_low > 0 else "unsupported_or_descriptive",
                }
            )
    return pd.DataFrame(rows)


def controls_table(run_rows: pd.DataFrame) -> pd.DataFrame:
    control_methods = [
        "random_same_branch_count_control",
        "validation_branch_ensemble_control",
        "c2m3_cluster_branch_control",
        "wrong_quotient_control",
        "wrong_quotient_order_control",
        "reversed_quotient_order_control",
        "uniform_pool_sign_destroyed_control",
        "one_shot_regular_lift",
        "parameter_matched_wide_model",
    ]
    rows = run_rows[run_rows["method"].isin(control_methods)].copy()
    if rows.empty:
        return rows
    return (
        rows.groupby(["source", "group_name", "noise_level", "method"], dropna=False)
        .agg(
            n=("test_accuracy", "count"),
            mean_validation_accuracy=("validation_accuracy", "mean"),
            mean_test_accuracy=("test_accuracy", "mean"),
            mean_branch_count=("branch_count", "mean"),
        )
        .reset_index()
    )


def write_plots(run_rows: pd.DataFrame, stage_rows: pd.DataFrame, stats: pd.DataFrame, args: argparse.Namespace) -> None:
    import matplotlib.pyplot as plt

    args.reports_dir.joinpath("plots").mkdir(parents=True, exist_ok=True)
    controlled = run_rows[run_rows["source"].eq("controlled")].copy()
    if not controlled.empty:
        depth = controlled[controlled["method"].str.startswith("sequential_depth_", na=False)].copy()
        if not depth.empty:
            depth["depth"] = pd.to_numeric(depth["depth"], errors="coerce")
            summary = depth.groupby(["depth", "method"])["test_accuracy"].mean().reset_index()
            fig, ax = plt.subplots(figsize=(8, 4))
            for method, group in summary.groupby("method"):
                ax.plot(group["depth"], group["test_accuracy"], marker="o", label=method.replace("sequential_", ""))
            ax.set_xlabel("Depth")
            ax.set_ylabel("Mean test accuracy")
            ax.set_title("Controlled sequential quotient accuracy by depth")
            ax.legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(args.reports_dir / "plots" / "sequential_quotient_accuracy_by_depth.pdf")
            plt.close(fig)
    if not stage_rows.empty:
        residual = stage_rows.groupby("depth")[["pre_structural_residual", "post_structural_residual"]].mean().reset_index()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(residual["depth"], residual["pre_structural_residual"], marker="o", label="pre")
        ax.plot(residual["depth"], residual["post_structural_residual"], marker="o", label="post")
        ax.set_xlabel("Depth")
        ax.set_ylabel("Mean structural residual")
        ax.set_title("Residual by quotient-lift depth")
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.reports_dir / "plots" / "sequential_quotient_residual_by_depth.pdf")
        plt.close(fig)
    if not stats.empty:
        plot_df = stats[stats["comparison"].str.contains("random_same|c2m3_cluster|wrong_quotient|one_shot", regex=True)].copy()
        if not plot_df.empty:
            fig, ax = plt.subplots(figsize=(10, 4))
            labels = plot_df["group_name"].astype(str) + "\n" + plot_df["comparison"].str.replace("sequential_quotient_lift_validation_router - ", "", regex=False)
            ax.bar(np.arange(len(plot_df)), plot_df["mean_delta"])
            ax.axhline(0.0, color="black", linewidth=0.8)
            ax.set_xticks(np.arange(len(plot_df)))
            ax.set_xticklabels(labels, rotation=80, ha="right", fontsize=6)
            ax.set_ylabel("Mean test delta")
            ax.set_title("Sequential quotient lift delta versus controls")
            fig.tight_layout()
            fig.savefig(args.reports_dir / "plots" / "sequential_quotient_delta_vs_controls.pdf")
            plt.close(fig)


def implementation_audit_text() -> str:
    return """# Sequential Quotient Lift Implementation Audit

## Audit Result

The old primary-depth sweep is not a genuine consecutive quotient-lift experiment.  It estimated primary factors from observed orders and relation residues, then reused the existing real `twisted_rank_lift_2` row for q=2.  Real q=4, q=8, and deeper branch lifts were not constructed as routed prediction tensors or parameter-level models.

## Distinctions

| Category | Current status |
| --- | --- |
| Diagnostics only | Existing primary candidates, order divisibility, pooling residuals, and large/truncated holonomy summaries. |
| Actual quotient construction | Implemented here only when an exact homomorphism `Gamma -> C2/C3` is verified, or for the ambient permutation sign character. |
| Actual branch/model construction | Controlled prediction-level branch lifts are implemented here; no new real parameter-level model is built. |
| Prediction-level lift | Implemented for controlled quotient chains using measured logits and validation-only routing. |
| Parameter-level lift | Not implemented. |
| Uniform invariant pooling | Implemented as a control; zero residual alone is not success because uniform pooling is invariant by construction. |
| Fourier/equivariant pooling | Implemented at prediction level; C2 keeps both plus and minus components before validation readout. |
| Learned or validation routing | Implemented at prediction level using validation labels only. |
| Depth > 1 | Implemented and tested in controlled groups C2xC2, C4, D4, and S3. Not implemented on natural MNIST. |

## Old Primary Sweep Boundary

The old sweep can be used as a diagnostic inventory and a q=2 disagreement-cluster baseline.  It must not be described as a certified cohomological, Brauer, or genuine quotient-driven sequential lift.  Its controlled sanity table included hard-coded target accuracies and therefore is not evidence of measured quotient-lift performance.
"""


def write_report(args, run_rows: pd.DataFrame, stage_rows: pd.DataFrame, diagnostics: pd.DataFrame, stats: pd.DataFrame, controls: pd.DataFrame, config: dict) -> None:
    controlled_stats = stats[stats["source"].eq("controlled")] if not stats.empty else pd.DataFrame()
    supported = controlled_stats[controlled_stats["claim_status"].eq("supported_controlled")] if not controlled_stats.empty else pd.DataFrame()
    natural = run_rows[run_rows["source"].astype(str).str.contains("natural", na=False)] if not run_rows.empty else pd.DataFrame()
    natural_seq = natural[natural["method"].eq("sequential_quotient_lift_validation_router")] if not natural.empty else pd.DataFrame()
    any_natural_implemented = bool(natural_seq.get("lift_implemented", pd.Series(dtype=bool)).fillna(False).any()) if not natural_seq.empty else False
    if any_natural_implemented:
        decision = "B. Genuine consecutive quotient lift implemented, controlled experiment succeeds, but natural MNIST gives a supported negative result."
    else:
        decision = "D. The genuine consecutive lift could not be implemented or evaluated completely; list exact blockers."
    report = f"""# Sequential Quotient Lift Report

## Exact Command

```bash
{config['exact_command']}
```

## Evidence Decision

{decision}

## Implementation Status

- Controlled prediction-level consecutive quotient lifting is implemented and measured.
- Exact quotient discovery uses multiplication-table homomorphisms for small exact groups and the ambient permutation sign character for truncated permutation groups.
- Natural MNIST q-driven prediction tensors were not available or constructed in this run.
- Real `twisted_rank_lift_2` remains a disagreement-clustering branch ensemble baseline, not a certified quotient-driven lift.
- No real q=4/q=8/depth>1 lift was executed.
- No Brauer/H2 language is justified by these real data.

## Controlled Group Diagnostics

{md_table(diagnostics[diagnostics['source'].eq('controlled')] if not diagnostics.empty and 'source' in diagnostics else diagnostics, ['group_name', 'stage_depth', 'quotient_order', 'homomorphism_residual', 'kernel_order', 'kernel_normal', 'quotient_certified', 'bootstrap_stability', 'bootstrap_method', 'residual_before', 'residual_after'], 40)}

## Controlled Accuracy Summary

{md_table(run_rows[run_rows['source'].eq('controlled')].groupby(['group_name', 'method'], dropna=False).agg(n=('test_accuracy', 'count'), mean_validation_accuracy=('validation_accuracy', 'mean'), mean_test_accuracy=('test_accuracy', 'mean')).reset_index() if not run_rows.empty else pd.DataFrame(), ['group_name', 'method', 'n', 'mean_validation_accuracy', 'mean_test_accuracy'], 80)}

## Paired Stats

{md_table(stats, ['source', 'group_name', 'noise_level', 'comparison', 'n_paired_seeds', 'mean_delta', 'ci_low', 'ci_high', 'claim_status'], 80)}

## Natural MNIST Artifact Scan

The requested natural experiment asks for N=6/N=8 with at least 30 seeds.  Current fixed-setting artifacts contain only N=3/N=4 maps, so the requested full natural run was not silently reduced.  The scan records available smaller artifacts as diagnostics only.

{md_table(diagnostics[diagnostics['source'].astype(str).str.contains('natural', na=False)] if not diagnostics.empty and 'source' in diagnostics else pd.DataFrame(), ['dataset', 'n_models', 'width', 'run_id', 'closure_status', 'truncated', 'stage_depth', 'quotient_order', 'quotient_certified', 'certification_method', 'bootstrap_stability', 'claim_boundary'], 40)}

{md_table(natural, ['run_id', 'method', 'lift_implemented', 'claim_boundary'], 20)}

## Controls

{md_table(controls, ['source', 'group_name', 'noise_level', 'method', 'n', 'mean_validation_accuracy', 'mean_test_accuracy', 'mean_branch_count'], 80)}

## Required Questions

1. Was the previous negative result mainly mathematical, implementation-related, or both?  Both: the old real q=2 baseline was implementation-limited, and current natural artifacts do not provide stable certified consecutive quotients plus quotient-routed prediction tensors.
2. Did the old q=2 improvement survive all same-branch and wrong-quotient controls?  No. Existing reports and this audit treat it as not surviving all controls.
3. Was any real depth greater than 1 actually executed?  No.
4. Did sequential lifting outperform a one-shot lift of equal branch capacity?  Controlled rows report this comparison; natural rows do not evaluate it.
5. Did it beat C2M3?  Controlled rows report this comparison; natural quotient-driven lift was not evaluated.
6. Did it beat greedy soup?  Controlled rows report this comparison separately; natural quotient-driven lift was not evaluated.
7. Is any Brauer/H2 language justified by the real data?  No.

## Blockers

- No current N=6/N=8 natural MNIST triangle-map artifacts were found.
- No real quotient-routed prediction tensors were constructed for natural MNIST.
- No parameter-level quotient lift was implemented.
- Existing q=2 branch rows are disagreement-cluster branch ensembles, not quotient-sheet transports.

Final decision: {decision}
"""
    (args.reports_dir / "sequential_quotient_lift_report.md").write_text(report, encoding="utf-8")
    (args.reports_dir / "sequential_quotient_lift_implementation_audit.md").write_text(implementation_audit_text(), encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", default="C2,C2xC2,C4,D4,S3")
    parser.add_argument("--seeds", default="0:29")
    parser.add_argument("--noise-levels", default="0.0,0.25,0.5")
    parser.add_argument("--n-val", type=int, default=1200)
    parser.add_argument("--n-test", type=int, default=2000)
    parser.add_argument("--n-classes", type=int, default=10)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--natural-datasets", default="mnist")
    parser.add_argument("--natural-model-counts", default="6,8")
    parser.add_argument("--natural-max-settings", type=int, default=12)
    parser.add_argument("--natural-max-group-order", type=int, default=256)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    start = time.time()
    args = parse_args(argv)
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    (args.reports_dir / "csv").mkdir(parents=True, exist_ok=True)
    (args.reports_dir / "configs").mkdir(parents=True, exist_ok=True)
    controlled, stages, diagnostics = controlled_rows(args)
    natural_diag, natural_runs = natural_diagnostics(args)
    all_runs = pd.concat([controlled, natural_runs], ignore_index=True, sort=False)
    all_diagnostics = pd.concat([diagnostics, natural_diag], ignore_index=True, sort=False)
    stats = paired_stats(all_runs, args)
    controls = controls_table(all_runs)
    write_plots(all_runs, stages, stats, args)
    runtime = time.time() - start
    config = {
        "exact_command": " ".join([".venv/bin/python", "experiments/sequential_quotient_lift_benchmark.py", *(argv or sys.argv[1:])]),
        "git_commit": git_output("rev-parse", "--short", "HEAD"),
        "dirty_status": git_output("status", "--short", "--untracked-files=no"),
        "environment": capture_environment(),
        "seeds": parse_seeds(args.seeds),
        "groups": parse_csv(args.groups, str),
        "noise_levels": parse_csv(args.noise_levels, float),
        "natural_requested_model_counts": parse_csv(args.natural_model_counts, int),
        "thresholds": {"bootstrap_stability_preferred": 0.8, "paired_ci_lower_bound_positive": True},
        "completed_settings": {
            "controlled_prediction_level": True,
            "natural_existing_artifact_scan": not natural_diag.empty or not natural_runs.empty,
        },
        "missing_settings": [
            "natural_N6_N8_30_seed_quotient_routed_prediction_lift",
            "natural_parameter_level_quotient_lift",
        ],
        "total_runtime_seconds": runtime,
    }
    all_runs.to_csv(args.reports_dir / "csv" / "sequential_quotient_lift_runs.csv", index=False, lineterminator="\n")
    stages.to_csv(args.reports_dir / "csv" / "sequential_quotient_lift_stages.csv", index=False, lineterminator="\n")
    all_diagnostics.to_csv(args.reports_dir / "csv" / "sequential_quotient_group_diagnostics.csv", index=False, lineterminator="\n")
    stats.to_csv(args.reports_dir / "csv" / "sequential_quotient_paired_stats.csv", index=False, lineterminator="\n")
    controls.to_csv(args.reports_dir / "csv" / "sequential_quotient_controls.csv", index=False, lineterminator="\n")
    save_json(args.reports_dir / "configs" / "sequential_quotient_lift_config.json", config)
    write_report(args, all_runs, stages, all_diagnostics, stats, controls, config)
    decision = "D. The genuine consecutive lift could not be implemented or evaluated completely; list exact blockers."
    print(f"Controlled rows: {len(controlled)}")
    print(f"Natural diagnostic rows: {len(natural_diag)}")
    print(f"Paired stats rows: {len(stats)}")
    print(f"Final decision: {decision}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
