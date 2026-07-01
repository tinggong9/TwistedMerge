#!/usr/bin/env python
"""Calibrate noise-stable obstruction bounds for synthetic and real artifacts."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt  # noqa: E402

from src.metrics import capture_environment, save_json  # noqa: E402
from src.model_merging_benchmark import format_markdown_table  # noqa: E402


DEFAULT_NOISE_LEVELS = (0.0, 0.02, 0.05, 0.10, 0.20)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def parse_float_list(text: str) -> list[float]:
    return [float(item.strip()) for item in str(text).split(",") if item.strip()]


def wrap_angle(values: np.ndarray) -> np.ndarray:
    return (values + np.pi) % (2.0 * np.pi) - np.pi


def finite_mean(values: pd.Series | np.ndarray) -> float:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def finite_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def bootstrap_mean_interval(values: np.ndarray, samples: int, rng: np.random.Generator) -> tuple[float, float, float]:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        return float("nan"), float("nan"), float("nan")
    if clean.size == 1 or samples <= 0:
        mean = float(clean.mean())
        return 0.0, mean, mean
    draws = rng.choice(clean, size=(samples, clean.size), replace=True).mean(axis=1)
    return float(np.std(draws, ddof=1)), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def base_row(
    *,
    source: str,
    family: str,
    case_id: str,
    seed: int | str,
    n_triangles: int,
    def_true: float,
    def_observed: float,
    cochain_distance: float,
    actual_synchronization_residual: float,
    noise_model: str,
    noise_level: float,
    noise_floor: float,
    noise_floor_ci_low: float,
    noise_floor_ci_high: float,
    k_threshold: float,
    has_true_cochain: bool,
    barrier_target: float = float("nan"),
    barrier_target_name: str = "",
    finite_central_gate_passed: bool = False,
    boundary_note: str = "",
    dataset: str = "",
    architecture: str = "",
    n_models: float = float("nan"),
    width: float = float("nan"),
    domain_shift: str = "",
    matching: str = "",
    setting_id: str = "",
    run_id: str = "",
    alignment_source: str = "",
    alignment_noise_fraction: float = float("nan"),
    evidence_role: str = "",
) -> dict:
    lower_bound = max(0.0, def_true - cochain_distance) if has_true_cochain else float("nan")
    bound_gap = actual_synchronization_residual - lower_bound if has_true_cochain else float("nan")
    direct_lipschitz_lhs = abs(def_observed - def_true) if has_true_cochain else float("nan")
    direct_lipschitz_holds = (
        bool(direct_lipschitz_lhs <= cochain_distance + 1e-12) if has_true_cochain else ""
    )
    stable_obstruction = (
        bool(def_observed > k_threshold * noise_floor)
        if np.isfinite(def_observed) and np.isfinite(noise_floor)
        else ""
    )
    return {
        "source": source,
        "family": family,
        "case_id": case_id,
        "setting_id": setting_id,
        "run_id": run_id,
        "dataset": dataset,
        "architecture": architecture,
        "n_models": n_models,
        "width": width,
        "domain_shift": domain_shift,
        "matching": matching,
        "seed": seed,
        "alignment_source": alignment_source,
        "alignment_noise_fraction": alignment_noise_fraction,
        "evidence_role": evidence_role,
        "noise_model": noise_model,
        "noise_level": noise_level,
        "has_true_cochain": has_true_cochain,
        "n_triangles": int(n_triangles),
        "def_true": def_true,
        "def_observed": def_observed,
        "cochain_distance": cochain_distance,
        "direct_lipschitz_lhs": direct_lipschitz_lhs,
        "direct_lipschitz_holds": direct_lipschitz_holds,
        "predicted_lower_bound": lower_bound,
        "actual_synchronization_residual": actual_synchronization_residual,
        "actual_minus_predicted_lower_bound": bound_gap,
        "barrier_target": barrier_target,
        "barrier_target_name": barrier_target_name,
        "noise_floor": noise_floor,
        "noise_floor_ci_low": noise_floor_ci_low,
        "noise_floor_ci_high": noise_floor_ci_high,
        "k_threshold": k_threshold,
        "stable_obstruction": stable_obstruction,
        "finite_central_gate_passed": finite_central_gate_passed,
        "real_brauer_claim_allowed": False,
        "boundary_note": boundary_note,
    }


def synthetic_mu2_rows(args: argparse.Namespace, rng: np.random.Generator) -> list[dict]:
    rows: list[dict] = []
    n_triangles = args.synthetic_triangles
    patterns = {
        "trivial": np.zeros(n_triangles, dtype=int),
        "sparse_nontrivial": np.array([1 if idx % 7 == 0 else 0 for idx in range(n_triangles)], dtype=int),
        "half_nontrivial": np.array([idx % 2 for idx in range(n_triangles)], dtype=int),
    }
    for family, true_bits in patterns.items():
        for noise_level, seed in product(args.noise_levels, range(args.synthetic_seeds)):
            local_rng = np.random.default_rng(args.seed_offset + 1000 * seed + int(noise_level * 10000))
            flips = local_rng.random(n_triangles) < noise_level
            observed = np.logical_xor(true_bits.astype(bool), flips).astype(float)
            true = true_bits.astype(float)
            cochain_distance = float(np.mean(observed != true))
            def_true = float(np.mean(true))
            def_observed = float(np.mean(observed))
            rows.append(
                base_row(
                    source="synthetic_mu2",
                    family=family,
                    case_id=f"mu2_{family}_noise{noise_level:g}_seed{seed}",
                    seed=seed,
                    n_triangles=n_triangles,
                    def_true=def_true,
                    def_observed=def_observed,
                    cochain_distance=cochain_distance,
                    actual_synchronization_residual=def_observed,
                    noise_model="mu2_triangle_bit_flip",
                    noise_level=float(noise_level),
                    noise_floor=cochain_distance,
                    noise_floor_ci_low=cochain_distance,
                    noise_floor_ci_high=cochain_distance,
                    k_threshold=args.k_threshold,
                    has_true_cochain=True,
                    finite_central_gate_passed=True,
                    boundary_note="Controlled mu2 cochain-level calibration; not a real model-merging claim.",
                )
            )
    return rows


def synthetic_u1_rows(args: argparse.Namespace) -> list[dict]:
    rows: list[dict] = []
    n_triangles = args.synthetic_triangles
    t = np.linspace(0.0, 1.0, n_triangles)
    patterns = {
        "trivial": np.zeros(n_triangles, dtype=float),
        "smooth_low_phase": np.pi * 0.18 * np.sin(2.0 * np.pi * t),
        "smooth_high_phase": np.pi * 0.55 * np.sin(2.0 * np.pi * t + 0.25),
    }
    for family, true_angles in patterns.items():
        for noise_level, seed in product(args.noise_levels, range(args.synthetic_seeds)):
            local_rng = np.random.default_rng(args.seed_offset + 50000 + 1000 * seed + int(noise_level * 10000))
            perturb = local_rng.normal(0.0, noise_level * np.pi, size=n_triangles)
            observed_angles = wrap_angle(true_angles + perturb)
            distance = float(np.mean(np.abs(wrap_angle(observed_angles - true_angles)) / np.pi))
            def_true = float(np.mean(np.abs(wrap_angle(true_angles)) / np.pi))
            def_observed = float(np.mean(np.abs(observed_angles) / np.pi))
            rows.append(
                base_row(
                    source="synthetic_u1",
                    family=family,
                    case_id=f"u1_{family}_noise{noise_level:g}_seed{seed}",
                    seed=seed,
                    n_triangles=n_triangles,
                    def_true=def_true,
                    def_observed=def_observed,
                    cochain_distance=distance,
                    actual_synchronization_residual=def_observed,
                    noise_model="u1_wrapped_gaussian_phase",
                    noise_level=float(noise_level),
                    noise_floor=distance,
                    noise_floor_ci_low=distance,
                    noise_floor_ci_high=distance,
                    k_threshold=args.k_threshold,
                    has_true_cochain=True,
                    finite_central_gate_passed=False,
                    boundary_note="Controlled U(1) cochain-level calibration; finite-central gates are not invoked.",
                )
            )
    return rows


def controlled_barrier_targets(method_path: Path) -> dict[tuple, tuple[float, str]]:
    if not method_path.exists():
        return {}
    cols = [
        "family",
        "seed",
        "width",
        "n_models",
        "method",
        "test_accuracy",
        "delta_vs_weight_average",
    ]
    df = pd.read_csv(method_path, usecols=lambda col: col in cols)
    targets: dict[tuple, tuple[float, str]] = {}
    for key, group in df.groupby(["family", "seed", "width", "n_models"], dropna=False):
        pivot = group.pivot_table(index=[], columns="method", values="test_accuracy", aggfunc="mean")
        if not pivot.empty and "twisted_q2_branch" in pivot.columns and "ordinary_weight_average" in pivot.columns:
            gap = float(pivot["twisted_q2_branch"].iloc[0] - pivot["ordinary_weight_average"].iloc[0])
            targets[key] = (gap, "twisted_q2_accuracy_minus_weight_average")
            continue
        q2 = group[group["method"] == "twisted_q2_branch"]
        if not q2.empty and "delta_vs_weight_average" in q2.columns:
            targets[key] = (finite_mean(q2["delta_vs_weight_average"]), "twisted_q2_delta_vs_weight_average")
    return targets


def controlled_twisted_overlap_rows(args: argparse.Namespace) -> list[dict]:
    triangle_path = args.reports_dir / "csv" / "controlled_twisted_overlap_triangles.csv"
    if not triangle_path.exists():
        return []
    df = pd.read_csv(triangle_path)
    target_map = controlled_barrier_targets(args.reports_dir / "csv" / "controlled_twisted_overlap.csv")
    rows: list[dict] = []
    group_cols = ["family", "seed", "width", "n_models"]
    for key, group in df.groupby(group_cols, dropna=False):
        family, seed, width, n_models = key
        true_sign = pd.to_numeric(group["true_alpha_sign"], errors="coerce").to_numpy(dtype=float)
        observed_sign = pd.to_numeric(group["observed_triangle_sign"], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(true_sign) & np.isfinite(observed_sign) & (true_sign != 0.0)
        if not valid.any():
            continue
        true_bits = (true_sign[valid] < 0).astype(float)
        observed_bits = (observed_sign[valid] < 0).astype(float)
        # On the tetrahedral mu2 face complex used by the controlled overlap
        # artifact, a 2-cochain is a coboundary exactly when the product of its
        # face signs is +1. The normalized distance to coboundaries is therefore
        # zero for even parity and one face flip divided by the number of faces
        # for odd parity. This intentionally differs from raw distance to the
        # identity cochain, since coboundary twists are gauge-trivial.
        def_true = 0.0 if int(np.prod(true_sign[valid])) > 0 else 1.0 / float(valid.sum())
        def_observed = 0.0 if int(np.prod(observed_sign[valid])) > 0 else 1.0 / float(valid.sum())
        cochain_distance = float(np.mean(true_bits != observed_bits))
        coboundary_residual = finite_mean(group.get("coboundary_residual", pd.Series(dtype=float)))
        sync_residual = coboundary_residual if np.isfinite(coboundary_residual) else def_observed
        finite_gate = bool(group.get("central_twist_claim_allowed", pd.Series([False])).map(finite_bool).any())
        barrier_target, barrier_name = target_map.get(key, (float("nan"), ""))
        rows.append(
            base_row(
                source="controlled_twisted_overlap",
                family=str(family),
                case_id=f"controlled_{family}_width{int(width)}_seed{int(seed)}",
                seed=int(seed),
                n_triangles=int(valid.sum()),
                def_true=def_true,
                def_observed=def_observed,
                cochain_distance=cochain_distance,
                actual_synchronization_residual=sync_residual,
                noise_model="observed_vs_known_mu2_triangle_sign",
                noise_level=cochain_distance,
                noise_floor=cochain_distance,
                noise_floor_ci_low=cochain_distance,
                noise_floor_ci_high=cochain_distance,
                k_threshold=args.k_threshold,
                has_true_cochain=True,
                barrier_target=barrier_target,
                barrier_target_name=barrier_name,
                finite_central_gate_passed=finite_gate,
                width=float(width),
                n_models=float(n_models),
                boundary_note=(
                    "Controlled neural-overlap artifact with known finite-central triangle signs; "
                    "Def is normalized distance to mu2 coboundaries on the tetrahedral face set; "
                    "nonzero-H2 status remains separate from ordinary untwisted trivialization."
                ),
            )
        )
    return rows


def fixed_setting_barrier_targets(path: Path) -> pd.DataFrame:
    key_cols = ["setting_id", "run_id", "alignment_source", "alignment_noise_fraction"]
    if not path.exists():
        return pd.DataFrame(columns=key_cols + ["barrier_target", "barrier_target_name"])
    usecols = key_cols + ["method", "linear_mode_connectivity_barrier"]
    df = pd.read_csv(path, usecols=lambda col: col in usecols)
    if "alignment_source" not in df.columns:
        df["alignment_source"] = "observed"
    if "alignment_noise_fraction" not in df.columns:
        df["alignment_noise_fraction"] = 0.0
    if "method" in df.columns:
        preferred = df[df["method"].isin(["c2m3_synchronized", "git_rebasin_pairwise_ref0"])].copy()
        if preferred.empty:
            preferred = df.copy()
    else:
        preferred = df.copy()
    if preferred.empty or "linear_mode_connectivity_barrier" not in preferred.columns:
        return pd.DataFrame(columns=key_cols + ["barrier_target", "barrier_target_name"])
    out = (
        preferred.groupby(key_cols, dropna=False)["linear_mode_connectivity_barrier"]
        .mean()
        .reset_index()
        .rename(columns={"linear_mode_connectivity_barrier": "barrier_target"})
    )
    out["barrier_target_name"] = "mean_linear_mode_connectivity_barrier"
    return out


def fixed_setting_run_metadata(path: Path) -> pd.DataFrame:
    key_cols = ["setting_id", "run_id", "alignment_source", "alignment_noise_fraction"]
    if not path.exists():
        return pd.DataFrame(columns=key_cols)
    cols = [
        *key_cols,
        "method",
        "dataset",
        "architecture",
        "n_models",
        "width",
        "domain_shift",
        "matching",
        "seed",
        "evidence_role",
        "sync_disagreement",
        "pairwise_alignment_residual_mean",
        "weight_merge_degradation",
    ]
    df = pd.read_csv(path, usecols=lambda col: col in cols)
    if "method" in df.columns:
        weight = df[df["method"] == "weight_average"].copy()
        if weight.empty:
            weight = df.drop_duplicates(key_cols).copy()
    else:
        weight = df.drop_duplicates(key_cols).copy()
    return weight.drop_duplicates(key_cols)


def real_fixed_setting_rows(args: argparse.Namespace) -> list[dict]:
    triangle_path = args.reports_dir / "csv" / "fixed_setting_triangle_defects.csv"
    if not triangle_path.exists():
        return []
    key_cols = [
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
    ]
    usecols = [*key_cols, "cycle_defect", "cycle_score", "triangle_defect_rate"]
    triangles = pd.read_csv(triangle_path, usecols=lambda col: col in usecols)
    defect_col = "cycle_defect" if "cycle_defect" in triangles.columns else "cycle_score"
    triangles[defect_col] = pd.to_numeric(triangles[defect_col], errors="coerce")

    run_cols = [
        "setting_id",
        "run_id",
        "alignment_source",
        "alignment_noise_fraction",
        "cycle_defect",
    ]
    run_noise = []
    rng = np.random.default_rng(args.seed_offset + 900000)
    for key, group in triangles.groupby(
        ["setting_id", "run_id", "alignment_source", "alignment_noise_fraction"],
        dropna=False,
    ):
        values = pd.to_numeric(group[defect_col], errors="coerce").to_numpy(dtype=float)
        floor, low, high = bootstrap_mean_interval(values, args.bootstrap_samples, rng)
        run_noise.append(
            {
                "setting_id": key[0],
                "run_id": key[1],
                "alignment_source": key[2],
                "alignment_noise_fraction": key[3],
                "run_def_observed": float(np.nanmean(values)),
                "run_n_triangles": int(np.isfinite(values).sum()),
                "run_noise_floor": floor,
                "run_noise_floor_ci_low": low,
                "run_noise_floor_ci_high": high,
            }
        )
    run_df = pd.DataFrame(run_noise)

    setting_noise = []
    for key, group in triangles.groupby(
        ["setting_id", "alignment_source", "alignment_noise_fraction"],
        dropna=False,
    ):
        values = pd.to_numeric(group[defect_col], errors="coerce").to_numpy(dtype=float)
        floor, low, high = bootstrap_mean_interval(values, args.bootstrap_samples, rng)
        setting_noise.append(
            {
                "setting_id": key[0],
                "alignment_source": key[1],
                "alignment_noise_fraction": key[2],
                "setting_noise_floor": floor,
                "setting_noise_floor_ci_low": low,
                "setting_noise_floor_ci_high": high,
            }
        )
    setting_df = pd.DataFrame(setting_noise)
    meta = fixed_setting_run_metadata(args.reports_dir / "csv" / "fixed_setting_verification_runs.csv")
    barriers = fixed_setting_barrier_targets(args.reports_dir / "csv" / "alignment_barrier_targets.csv")

    merged = run_df.merge(
        setting_df,
        on=["setting_id", "alignment_source", "alignment_noise_fraction"],
        how="left",
    ).merge(
        meta,
        on=["setting_id", "run_id", "alignment_source", "alignment_noise_fraction"],
        how="left",
        suffixes=("", "_meta"),
    ).merge(
        barriers,
        on=["setting_id", "run_id", "alignment_source", "alignment_noise_fraction"],
        how="left",
    )

    rows: list[dict] = []
    for _, row in merged.iterrows():
        run_floor = float(row.get("run_noise_floor", float("nan")))
        setting_floor = float(row.get("setting_noise_floor", float("nan")))
        finite_floors = [value for value in [run_floor, setting_floor] if np.isfinite(value)]
        noise_floor = max(finite_floors) if finite_floors else float("nan")
        low = float(row.get("setting_noise_floor_ci_low", float("nan")))
        high = float(row.get("setting_noise_floor_ci_high", float("nan")))
        sync = float(row.get("sync_disagreement", float("nan")))
        if not np.isfinite(sync):
            sync = float(row.get("pairwise_alignment_residual_mean", float("nan")))
        barrier_target = float(row.get("barrier_target", float("nan")))
        if not np.isfinite(barrier_target):
            barrier_target = float(row.get("weight_merge_degradation", float("nan")))
            barrier_name = "weight_merge_degradation"
        else:
            barrier_name = str(row.get("barrier_target_name", "mean_linear_mode_connectivity_barrier"))
        rows.append(
            base_row(
                source="real_fixed_setting_triangle_proxy",
                family="real_alignment_unknown_true_cochain",
                case_id=str(row.get("run_id", "")),
                setting_id=str(row.get("setting_id", "")),
                run_id=str(row.get("run_id", "")),
                dataset=str(row.get("dataset", "")),
                architecture=str(row.get("architecture", "")),
                n_models=float(row.get("n_models", float("nan"))),
                width=float(row.get("width", float("nan"))),
                domain_shift=str(row.get("domain_shift", "")),
                matching=str(row.get("matching", "")),
                seed=int(row.get("seed")) if pd.notna(row.get("seed")) else "",
                alignment_source=str(row.get("alignment_source", "")),
                alignment_noise_fraction=float(row.get("alignment_noise_fraction", float("nan"))),
                evidence_role=str(row.get("evidence_role", "")),
                n_triangles=int(row.get("run_n_triangles", 0)),
                def_true=float("nan"),
                def_observed=float(row.get("run_def_observed", float("nan"))),
                cochain_distance=float("nan"),
                actual_synchronization_residual=sync,
                noise_model="triangle_defect_bootstrap_proxy_no_saved_activations",
                noise_level=float(row.get("alignment_noise_fraction", float("nan"))),
                noise_floor=noise_floor,
                noise_floor_ci_low=low,
                noise_floor_ci_high=high,
                k_threshold=args.k_threshold,
                has_true_cochain=False,
                barrier_target=barrier_target,
                barrier_target_name=barrier_name,
                finite_central_gate_passed=False,
                boundary_note=(
                    "Real fixed-setting truth cochain is unknown; noise floor bootstraps saved triangle defects, "
                    "not overlap activations. This row cannot certify Brauer/projective classes."
                ),
            )
        )
    return rows


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["source", "family", "noise_model", "alignment_source", "alignment_noise_fraction"]
    summaries = []
    for key, group in rows.groupby(group_cols, dropna=False):
        meta = dict(zip(group_cols, key))
        direct = group["direct_lipschitz_holds"]
        direct_valid = direct[direct != ""].astype(bool)
        stable = group["stable_obstruction"]
        stable_valid = stable[stable != ""].astype(bool)
        summaries.append(
            {
                **meta,
                "n_rows": int(len(group)),
                "n_cases": int(group["case_id"].nunique()),
                "n_unique_seeds": int(pd.Series(group["seed"]).nunique()),
                "mean_def_true": finite_mean(group["def_true"]),
                "mean_def_observed": finite_mean(group["def_observed"]),
                "mean_cochain_distance": finite_mean(group["cochain_distance"]),
                "mean_noise_floor": finite_mean(group["noise_floor"]),
                "direct_lipschitz_pass_rate": float(direct_valid.mean()) if len(direct_valid) else float("nan"),
                "stable_obstruction_fraction": float(stable_valid.mean()) if len(stable_valid) else float("nan"),
                "mean_predicted_lower_bound": finite_mean(group["predicted_lower_bound"]),
                "mean_actual_synchronization_residual": finite_mean(group["actual_synchronization_residual"]),
                "mean_actual_minus_predicted_lower_bound": finite_mean(group["actual_minus_predicted_lower_bound"]),
                "mean_barrier_target": finite_mean(group["barrier_target"]),
                "finite_central_gate_pass_rate": float(group["finite_central_gate_passed"].astype(bool).mean()),
                "real_brauer_claim_allowed": False,
            }
        )
    return pd.DataFrame(summaries).sort_values(group_cols).reset_index(drop=True)


def plot_observed_vs_noise_floor(df: pd.DataFrame, path: Path, k_threshold: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plot_df = df[np.isfinite(df["def_observed"]) & np.isfinite(df["noise_floor"])].copy()
    if plot_df.empty:
        return
    families = list(dict.fromkeys(plot_df["source"].astype(str)))
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(8, 5))
    for idx, source in enumerate(families):
        part = plot_df[plot_df["source"].astype(str) == source]
        ax.scatter(
            part["noise_floor"],
            part["def_observed"],
            s=24,
            alpha=0.65,
            label=source,
            color=cmap(idx % 10),
            edgecolor="none",
        )
    xmax = max(float(plot_df["noise_floor"].max()), 1e-6)
    xline = np.linspace(0.0, xmax, 200)
    ax.plot(xline, k_threshold * xline, color="black", linewidth=1.0, linestyle="--", label=f"{k_threshold:g}x noise floor")
    ax.set_xlabel("estimated noise floor")
    ax.set_ylabel("observed defect Def(c_hat)")
    ax.set_title("Observed obstruction defect versus noise floor")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def report_text(
    args: argparse.Namespace,
    df: pd.DataFrame,
    summary: pd.DataFrame,
    csv_path: Path,
    summary_path: Path,
    plot_path: Path,
    config_path: Path,
) -> str:
    env = capture_environment()
    command_parts = []
    for name in ["PYTHONPYCACHEPREFIX", "MPLCONFIGDIR"]:
        value = os.environ.get(name)
        if value:
            command_parts.append(f"{name}={value}")
    command_parts.extend([sys.executable, *sys.argv])
    exact_command = " ".join(command_parts)

    controlled = df[df["has_true_cochain"].astype(bool)]
    direct_valid = controlled["direct_lipschitz_holds"]
    direct_valid = direct_valid[direct_valid != ""].astype(bool)
    direct_pass_rate = float(direct_valid.mean()) if len(direct_valid) else float("nan")
    lower_bound_gap = pd.to_numeric(controlled["actual_minus_predicted_lower_bound"], errors="coerce")
    lower_bound_gap_min = float(lower_bound_gap.min()) if lower_bound_gap.notna().any() else float("nan")
    lower_bound_violation_count = int((lower_bound_gap < -1e-12).sum())

    real = df[df["source"] == "real_fixed_setting_triangle_proxy"]
    real_summary = (
        real.groupby(["dataset", "architecture", "n_models", "domain_shift", "matching", "alignment_source"], dropna=False)
        .agg(
            n_rows=("case_id", "count"),
            mean_def_observed=("def_observed", "mean"),
            mean_noise_floor=("noise_floor", "mean"),
            stable_fraction=("stable_obstruction", lambda x: pd.Series(x[x != ""]).astype(bool).mean() if len(x[x != ""]) else np.nan),
            mean_sync_residual=("actual_synchronization_residual", "mean"),
            mean_barrier_target=("barrier_target", "mean"),
        )
        .reset_index()
    )
    real_table_rows = real_summary.head(12).to_dict("records") if not real_summary.empty else []

    overview_cols = [
        "source",
        "family",
        "n_rows",
        "mean_def_observed",
        "mean_noise_floor",
        "direct_lipschitz_pass_rate",
        "stable_obstruction_fraction",
    ]
    overview = summary[overview_cols].head(14).to_dict("records")

    real_cols = [
        "dataset",
        "architecture",
        "n_models",
        "domain_shift",
        "matching",
        "alignment_source",
        "n_rows",
        "mean_def_observed",
        "mean_noise_floor",
        "stable_fraction",
        "mean_sync_residual",
    ]

    return "\n".join(
        [
            "# Obstruction/noise bound calibration",
            "",
            "## Exact command",
            "",
            f"`{exact_command}`",
            "",
            "## Environment",
            "",
            f"- Commit: `{git_commit()}`",
            f"- Python: `{env['python']}`",
            f"- Platform: `{env['platform']}`",
            f"- Packages: `{env['packages']}`",
            "",
            "## Outputs",
            "",
            f"- Calibration rows: `{csv_path}`",
            f"- Summary rows: `{summary_path}`",
            f"- Plot: `{plot_path}`",
            f"- Config: `{config_path}`",
            "",
            "## Calibration definitions",
            "",
            "- `Def(c)` is source-specific and explicitly recorded in the CSV: synthetic cochain toys use mean normalized distance from identity; controlled tetrahedral mu2 rows use normalized distance to mu2 coboundaries; real rows use the saved fixed-setting cycle defect because the true cochain is unknown.",
            "- `||c_hat - c||` is the same mean normalized distance between the noisy and true triangle cochains when a true cochain is available.",
            f"- The practical threshold is `stable_obstruction = Def(c_hat) > {args.k_threshold:g} * noise_floor`.",
            "- For real fixed-setting rows the true cochain is unknown, so the script estimates a proxy noise floor from saved triangle-defect bootstrap samples. It does not recompute activations because overlap activations are not saved in the current artifacts.",
            "",
            "## Bound checks",
            "",
            f"- Controlled rows with known true cochains: {len(controlled)}",
            f"- Direct check `|Def(c_hat)-Def(c)| <= ||c_hat-c||` pass rate: {direct_pass_rate:.4f}",
            f"- Minimum `actual_synchronization_residual - predicted_lower_bound` on controlled rows: {lower_bound_gap_min:.4f}",
            f"- Lower-bound violation count on controlled rows: {lower_bound_violation_count}",
            "- Real rows set `real_brauer_claim_allowed = False`; finite-central gates are only marked on controlled finite-central rows that already carry that gate.",
            "",
            "## Summary table",
            "",
            format_markdown_table(overview, overview_cols) if overview else "_No rows generated._",
            "",
            "## Real fixed-setting proxy table",
            "",
            format_markdown_table(real_table_rows, real_cols) if real_table_rows else "_No real fixed-setting rows available._",
            "",
            "## Claim boundary",
            "",
            "- Supported by this report: the controlled mu2 and U(1) definitions satisfy the direct Lipschitz-style defect stability check under injected cochain noise.",
            "- Supported as a diagnostic artifact: fixed-setting real rows can be assigned a saved-triangle bootstrap noise floor and threshold flag.",
            "- Not supported here: real Brauer/projective class detection, activation-level bootstrap stability, or any claim that obstruction scores prove real model-merging failure by themselves.",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "reports")
    parser.add_argument("--synthetic-seeds", type=int, default=24)
    parser.add_argument("--synthetic-triangles", type=int, default=32)
    parser.add_argument("--noise-levels", type=parse_float_list, default=list(DEFAULT_NOISE_LEVELS))
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--k-threshold", type=float, default=3.0)
    parser.add_argument("--seed-offset", type=int, default=23023)
    args = parser.parse_args()

    args.reports_dir.mkdir(parents=True, exist_ok=True)
    (args.reports_dir / "csv").mkdir(parents=True, exist_ok=True)
    (args.reports_dir / "plots").mkdir(parents=True, exist_ok=True)
    (args.reports_dir / "configs").mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed_offset)
    rows = []
    rows.extend(synthetic_mu2_rows(args, rng))
    rows.extend(synthetic_u1_rows(args))
    rows.extend(controlled_twisted_overlap_rows(args))
    rows.extend(real_fixed_setting_rows(args))
    df = pd.DataFrame(rows)
    summary = summarize(df)

    csv_path = args.reports_dir / "csv" / "obstruction_noise_bound_calibration.csv"
    summary_path = args.reports_dir / "csv" / "obstruction_noise_bound_summary.csv"
    plot_path = args.reports_dir / "plots" / "def_observed_vs_noise_floor.pdf"
    report_path = args.reports_dir / "obstruction_noise_bound_calibration.md"
    config_path = args.reports_dir / "configs" / "obstruction_noise_bound_calibration_config.json"

    df.to_csv(csv_path, index=False, lineterminator="\n")
    summary.to_csv(summary_path, index=False, lineterminator="\n")
    plot_observed_vs_noise_floor(df, plot_path, args.k_threshold)
    save_json(
        config_path,
        {
            "argv": sys.argv,
            "args": {
                **vars(args),
                "reports_dir": str(args.reports_dir),
            },
            "environment": capture_environment(),
            "outputs": {
                "csv": str(csv_path),
                "summary_csv": str(summary_path),
                "plot": str(plot_path),
                "report": str(report_path),
            },
        },
    )
    report_path.write_text(
        report_text(args, df, summary, csv_path, summary_path, plot_path, config_path),
        encoding="utf-8",
    )
    print(f"wrote {csv_path}")
    print(f"wrote {summary_path}")
    print(f"wrote {plot_path}")
    print(f"wrote {report_path}")
    print(f"wrote {config_path}")


if __name__ == "__main__":
    main()
