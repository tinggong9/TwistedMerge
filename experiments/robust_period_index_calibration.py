#!/usr/bin/env python
"""Multi-seed calibration for robust central period-index detection."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt  # noqa: E402

from src.metrics import capture_environment, save_json  # noqa: E402
from src.model_merging_benchmark import format_markdown_table  # noqa: E402
from src.period_index_central import clock_matrix, heisenberg_generators, shift_matrix  # noqa: E402
from src.period_index_detector import RobustPeriodIndexDetection, robust_detect_commutator_matrix_period_index  # noqa: E402
from src.period_index_mining import (  # noqa: E402
    add_entrywise_noise,
    add_unitary_noise,
    generate_noncentral_controls,
    project_to_nearest_unitary,
)


DEFAULT_NOISE_LEVELS = (0.0, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 5e-2)
DEFAULT_NOISE_TYPES = ("unitary_near_identity", "entrywise_projected_unitary")
DEFAULT_SEEDS = 20
LIFT_METHOD = "period_index_projective_morita_lift"


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    source: str
    d: int | None
    k: int | None
    true_period: int | None
    true_index: int | None
    width: int
    candidate_ranks: tuple[int, ...]
    generator_factory: object


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def heisenberg_generator_dict(d: int, k: int) -> dict[str, np.ndarray]:
    system = heisenberg_generators(d, k)
    generators: dict[str, np.ndarray] = {}
    for idx in range(k):
        generators[f"U{idx + 1}"] = system.U[idx]
        generators[f"V{idx + 1}"] = system.V[idx]
    return generators


def rank_deficient_d3_one_pair() -> dict[str, np.ndarray]:
    return {
        "A": clock_matrix(3),
        "B": shift_matrix(3),
        "C": np.eye(3, dtype=complex),
        "D": np.eye(3, dtype=complex),
    }


def abelian_trivial_control(width: int = 4) -> dict[str, np.ndarray]:
    angles = np.linspace(0.0, 0.4, width)
    return {
        "D1": np.diag(np.exp(1j * angles)).astype(complex),
        "D2": np.diag(np.exp(-0.7j * angles)).astype(complex),
        "I": np.eye(width, dtype=complex),
    }


def random_unitary_noncentral(width: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    out = {}
    for idx in range(2):
        raw = rng.normal(size=(width, width)) + 1j * rng.normal(size=(width, width))
        out[f"Q{idx + 1}"] = project_to_nearest_unitary(raw)
    return out


def nearly_scalar_but_noncentral_control(width: int, seed: int, scale: float = 0.35) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    diagonal = np.diag(np.linspace(-1.0, 1.0, width)).astype(complex)
    shift = np.zeros((width, width), dtype=complex)
    for idx in range(width - 1):
        shift[idx, idx + 1] = 1.0
        shift[idx + 1, idx] = -1.0
    raw = rng.normal(size=(width, width)) + 1j * rng.normal(size=(width, width))
    skew = raw - raw.conj().T
    return {
        "near_A": project_to_nearest_unitary(np.eye(width, dtype=complex) + scale * diagonal + 0.03 * skew),
        "near_B": project_to_nearest_unitary(np.eye(width, dtype=complex) + scale * shift),
    }


def apply_noise(
    generators: dict[str, np.ndarray],
    noise_level: float,
    noise_type: str,
    seed: int,
) -> dict[str, np.ndarray]:
    if noise_level <= 0:
        return {name: np.asarray(matrix, dtype=complex) for name, matrix in generators.items()}
    rng = np.random.default_rng(seed)
    noisy: dict[str, np.ndarray] = {}
    for name, matrix in generators.items():
        if noise_type == "unitary_near_identity":
            noisy[name] = add_unitary_noise(matrix, noise_level, rng=rng)
        elif noise_type == "entrywise_projected_unitary":
            noisy[name] = add_entrywise_noise(matrix, noise_level, project_unitary=True, rng=rng)
        elif noise_type == "entrywise_unprojected":
            noisy[name] = add_entrywise_noise(matrix, noise_level, project_unitary=False, rng=rng)
        else:
            raise ValueError(f"unknown noise_type: {noise_type}")
    return noisy


def central_specs() -> tuple[CaseSpec, ...]:
    return (
        CaseSpec(
            case_id="heisenberg_d2_k2",
            source="central_positive",
            d=2,
            k=2,
            true_period=2,
            true_index=4,
            width=4,
            candidate_ranks=(2, 4, 8),
            generator_factory=lambda seed: heisenberg_generator_dict(2, 2),
        ),
        CaseSpec(
            case_id="heisenberg_d3_k2",
            source="central_positive",
            d=3,
            k=2,
            true_period=3,
            true_index=9,
            width=9,
            candidate_ranks=(3, 6, 9, 18),
            generator_factory=lambda seed: heisenberg_generator_dict(3, 2),
        ),
        CaseSpec(
            case_id="heisenberg_d2_k3",
            source="central_positive",
            d=2,
            k=3,
            true_period=2,
            true_index=8,
            width=8,
            candidate_ranks=(2, 4, 8, 16),
            generator_factory=lambda seed: heisenberg_generator_dict(2, 3),
        ),
        CaseSpec(
            case_id="rank_deficient_d3_one_pair",
            source="central_positive",
            d=3,
            k=1,
            true_period=3,
            true_index=3,
            width=3,
            candidate_ranks=(3, 6),
            generator_factory=lambda seed: rank_deficient_d3_one_pair(),
        ),
        CaseSpec(
            case_id="composite_d4_k1",
            source="central_positive",
            d=4,
            k=1,
            true_period=4,
            true_index=4,
            width=4,
            candidate_ranks=(4, 8),
            generator_factory=lambda seed: heisenberg_generator_dict(4, 1),
        ),
    )


def negative_specs() -> tuple[CaseSpec, ...]:
    return (
        CaseSpec(
            case_id="s3_permutation_noncentral",
            source="noncentral_negative",
            d=None,
            k=None,
            true_period=None,
            true_index=None,
            width=3,
            candidate_ranks=(3,),
            generator_factory=lambda seed: generate_noncentral_controls(3, 0.0, seed=seed, control_type="permutation"),
        ),
        CaseSpec(
            case_id="random_gl_noncentral",
            source="noncentral_negative",
            d=None,
            k=None,
            true_period=None,
            true_index=None,
            width=4,
            candidate_ranks=(4,),
            generator_factory=lambda seed: generate_noncentral_controls(4, 0.0, seed=seed, control_type="random_gl"),
        ),
        CaseSpec(
            case_id="random_unitary_noncentral",
            source="noncentral_negative",
            d=None,
            k=None,
            true_period=None,
            true_index=None,
            width=4,
            candidate_ranks=(4,),
            generator_factory=lambda seed: random_unitary_noncentral(4, seed),
        ),
        CaseSpec(
            case_id="nearly_scalar_but_noncentral_control",
            source="noncentral_negative",
            d=None,
            k=None,
            true_period=None,
            true_index=None,
            width=4,
            candidate_ranks=(4,),
            generator_factory=lambda seed: nearly_scalar_but_noncentral_control(4, seed),
        ),
        CaseSpec(
            case_id="abelian_trivial_control",
            source="trivial_abelian_negative",
            d=None,
            k=None,
            true_period=None,
            true_index=None,
            width=4,
            candidate_ranks=(4,),
            generator_factory=lambda seed: abelian_trivial_control(4),
        ),
    )


def expected_rank_decision(true_period: int | None, true_index: int | None, rank: int, status: str) -> str:
    if true_period is None or true_index is None:
        return "not_central_projective"
    if status == "candidate_uncertain":
        return "central_projective_candidate_uncertain"
    if status == "rejected_noncentral":
        return "not_central_projective"
    if status == "unknown_index":
        return "central_projective_index_unknown"
    if rank % true_index == 0:
        return "period_index_lift_success"
    if rank % true_period == 0:
        return "period_divisible_index_obstructed"
    return "rank_obstructed"


def selected_method_for(detection: RobustPeriodIndexDetection) -> str:
    if detection.status == "certified" and detection.decision == "period_index_lift_success":
        return LIFT_METHOD
    return "none"


def row_passes(
    *,
    source: str,
    true_period: int | None,
    true_index: int | None,
    rank: int,
    detection: RobustPeriodIndexDetection,
    selected_method: str,
    expected_decision: str,
) -> bool:
    if selected_method == LIFT_METHOD and expected_decision != "period_index_lift_success":
        return False
    if source in {"noncentral_negative", "trivial_abelian_negative"}:
        return selected_method != LIFT_METHOD and detection.decision == "not_central_projective"
    if detection.status == "certified":
        return (
            detection.period == true_period
            and detection.index == true_index
            and detection.decision == expected_rank_decision(true_period, true_index, rank, "certified")
        )
    return selected_method != LIFT_METHOD


def pack_row(
    spec: CaseSpec,
    *,
    candidate_rank: int,
    noise_type: str,
    noise_level: float,
    seed: int,
    detection: RobustPeriodIndexDetection,
) -> dict[str, object]:
    selected_method = selected_method_for(detection)
    expected_decision = expected_rank_decision(spec.true_period, spec.true_index, candidate_rank, detection.status)
    period_correct = spec.true_period is not None and detection.period == spec.true_period
    index_correct = spec.true_index is not None and detection.index == spec.true_index
    source = spec.source
    if spec.true_period is not None and spec.true_index is not None and candidate_rank % spec.true_index != 0:
        source = "rank_divisibility"
    pass_fail = row_passes(
        source=spec.source,
        true_period=spec.true_period,
        true_index=spec.true_index,
        rank=candidate_rank,
        detection=detection,
        selected_method=selected_method,
        expected_decision=expected_decision,
    )
    return {
        "case_id": spec.case_id,
        "source": source,
        "d": spec.d,
        "k": spec.k,
        "true_period": spec.true_period,
        "true_index": spec.true_index,
        "candidate_rank": candidate_rank,
        "noise_type": noise_type,
        "noise_level": noise_level,
        "seed": seed,
        "detector_status": detection.status,
        "detected_period": detection.period,
        "detected_index": detection.index,
        "period_correct": period_correct,
        "index_correct": index_correct,
        "period_divides_rank": detection.period_divides_rank,
        "index_divides_rank": detection.index_divides_rank,
        "decision": detection.decision,
        "selected_method": selected_method,
        "expected_decision": expected_decision,
        "pass_fail": "pass" if pass_fail else "fail",
        "max_centrality_score": detection.max_centrality_score,
        "max_phase_residual": detection.max_phase_residual,
        "min_root_margin": detection.min_root_margin,
        "min_root_confidence": detection.min_root_confidence,
        "alternating_rank": detection.alternating_rank,
        "radical_size": detection.radical_size,
        "quotient_size": detection.quotient_size,
        "threshold_level": detection.threshold_level,
        "notes": " ".join(detection.notes),
    }


def calibration_rows(
    *,
    seeds: int,
    noise_levels: Iterable[float],
    noise_types: Iterable[str],
    centrality_tol_grid: tuple[float, ...] = (1e-6, 1e-4, 1e-3, 1e-2),
    phase_tol_grid: tuple[float, ...] = (1e-6, 1e-4, 1e-3, 1e-2),
    confidence_margin: float = 0.25,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    all_specs = (*central_specs(), *negative_specs())
    for spec in all_specs:
        for noise_type in noise_types:
            for noise_level in noise_levels:
                for seed in range(seeds):
                    base = spec.generator_factory(seed)
                    noisy = apply_noise(base, float(noise_level), noise_type, seed=100000 + 997 * seed + spec.width)
                    for candidate_rank in spec.candidate_ranks:
                        detection = robust_detect_commutator_matrix_period_index(
                            noisy,
                            candidate_rank=candidate_rank,
                            centrality_tol_grid=centrality_tol_grid,
                            phase_tol_grid=phase_tol_grid,
                            confidence_margin=confidence_margin,
                        )
                        rows.append(
                            pack_row(
                                spec,
                                candidate_rank=candidate_rank,
                                noise_type=noise_type,
                                noise_level=float(noise_level),
                                seed=seed,
                                detection=detection,
                            )
                        )
    return rows


def summary_rows(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    frame["is_certified"] = frame["detector_status"] == "certified"
    frame["is_uncertain"] = frame["detector_status"] == "candidate_uncertain"
    frame["is_rejected"] = frame["detector_status"] == "rejected_noncentral"
    frame["is_negative"] = frame["source"].isin(["noncentral_negative", "trivial_abelian_negative"])
    frame["is_positive"] = frame["source"].isin(["central_positive", "rank_divisibility"])
    frame["false_lift"] = (frame["selected_method"] == LIFT_METHOD) & (frame["expected_decision"] != "period_index_lift_success")
    frame["false_positive_central"] = frame["is_negative"] & frame["is_certified"]
    frame["false_positive_lift"] = frame["is_negative"] & (frame["selected_method"] == LIFT_METHOD)
    frame["false_negative_rejection"] = frame["is_positive"] & frame["is_rejected"]
    frame["obstructed_success"] = (
        (frame["expected_decision"] == "period_divisible_index_obstructed")
        & (frame["selected_method"] == "none")
    )

    grouped = (
        frame.groupby(["case_id", "source", "noise_type", "noise_level", "candidate_rank"], dropna=False)
        .agg(
            n=("seed", "count"),
            certification_rate=("is_certified", "mean"),
            uncertain_rate=("is_uncertain", "mean"),
            rejection_rate=("is_rejected", "mean"),
            correct_period_rate=("period_correct", "mean"),
            correct_index_rate=("index_correct", "mean"),
            false_lift_rate=("false_lift", "mean"),
            false_positive_central_rate=("false_positive_central", "mean"),
            false_positive_lift_rate=("false_positive_lift", "mean"),
            false_negative_rejection_rate=("false_negative_rejection", "mean"),
            period_divisible_index_obstructed_rejection_rate=("obstructed_success", "mean"),
            mean_centrality_score=("max_centrality_score", "mean"),
            mean_phase_residual=("max_phase_residual", "mean"),
            mean_root_margin=("min_root_margin", "mean"),
        )
        .reset_index()
    )
    return grouped


def policy_stats(df: pd.DataFrame) -> pd.DataFrame:
    policies = []
    small_noise = df["noise_level"] <= 1e-5
    medium_noise = (df["noise_level"] >= 3e-5) & (df["noise_level"] <= 1e-3)
    large_noise = df["noise_level"] >= 1e-2
    policy_grid = [
        (1e-5, 1e-5, 0.25),
        (3e-5, 3e-5, 0.25),
        (1e-4, 1e-4, 0.25),
        (3e-4, 3e-4, 0.25),
        (1e-4, 1e-4, 0.5),
        (3e-4, 3e-4, 0.5),
        (1e-3, 1e-3, 0.5),
    ]
    for centrality_tol, phase_tol, margin in policy_grid:
        finite_index = df["detected_index"].notna()
        confident = df["min_root_confidence"].fillna(0.0) >= margin
        would_certify = (
            (df["max_centrality_score"] <= centrality_tol)
            & (df["max_phase_residual"] <= phase_tol)
            & confident
            & finite_index
            & (df["decision"] != "not_central_projective")
        )
        expected_lift = df["expected_decision"] == "period_index_lift_success"
        would_lift = would_certify & expected_lift
        negatives = df["source"].isin(["noncentral_negative", "trivial_abelian_negative"])
        positives = df["source"].isin(["central_positive", "rank_divisibility"])
        policies.append(
            {
                "centrality_tol": centrality_tol,
                "phase_tol": phase_tol,
                "confidence_margin": margin,
                "small_noise_certification_rate": float((would_certify & positives & small_noise).sum() / max(int((positives & small_noise).sum()), 1)),
                "false_positive_central_rate": float((would_certify & negatives).sum() / max(int(negatives.sum()), 1)),
                "false_positive_lift_rate": float((would_lift & negatives).sum() / max(int(negatives.sum()), 1)),
                "medium_noise_uncertain_rate": float(((~would_certify) & positives & medium_noise & (df["detector_status"] == "candidate_uncertain")).sum() / max(int((positives & medium_noise).sum()), 1)),
                "large_noise_rejection_rate": float(((~would_certify) & positives & large_noise & (df["detector_status"] == "rejected_noncentral")).sum() / max(int((positives & large_noise).sum()), 1)),
            }
        )
    policy_df = pd.DataFrame(policies)
    feasible = policy_df[(policy_df["false_positive_lift_rate"] == 0) & (policy_df["false_positive_central_rate"] == 0)]
    if feasible.empty:
        policy_df["recommended"] = False
        best_idx = policy_df.sort_values(
            ["false_positive_lift_rate", "false_positive_central_rate", "small_noise_certification_rate"],
            ascending=[True, True, False],
        ).index[0]
    else:
        policy_df["recommended"] = False
        best_idx = feasible.sort_values("small_noise_certification_rate", ascending=False).index[0]
    policy_df.loc[best_idx, "recommended"] = True
    return policy_df


def _display_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for col in [name for name in ["noise_level", "first_uncertain_noise", "first_rejection_noise"] if name in out.columns]:
        out[col] = out[col].map(
            lambda value: "nan"
            if pd.isna(value)
            else f"{float(value):.0e}"
            if 0 < abs(float(value)) < 1e-3
            else f"{float(value):g}"
        )
    return out


def write_plots(summary_df: pd.DataFrame, plots_dir: Path) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    positives = summary_df[summary_df["source"].isin(["central_positive", "rank_divisibility"])]
    negatives = summary_df[summary_df["source"].isin(["noncentral_negative", "trivial_abelian_negative"])]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for (case_id, noise_type), group in positives.groupby(["case_id", "noise_type"]):
        minimal_rank = group["candidate_rank"].max()
        selected = group[group["candidate_rank"] == minimal_rank].sort_values("noise_level")
        ax.plot(selected["noise_level"], selected["certification_rate"], marker="o", label=f"{case_id}/{noise_type}")
    ax.set_xscale("symlog", linthresh=1e-8)
    ax.set_xlabel("noise level")
    ax.set_ylabel("certification rate")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=6, ncol=2)
    ax.set_title("Robust period-index certification rate")
    fig.tight_layout()
    fig.savefig(plots_dir / "robust_period_index_certification_rate.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for noise_type, group in negatives.groupby("noise_type"):
        selected = group.groupby("noise_level", as_index=False)[["false_positive_central_rate", "false_positive_lift_rate"]].mean()
        ax.plot(selected["noise_level"], selected["false_positive_central_rate"], marker="o", label=f"{noise_type} central")
        ax.plot(selected["noise_level"], selected["false_positive_lift_rate"], marker="x", linestyle="--", label=f"{noise_type} lift")
    ax.set_xscale("symlog", linthresh=1e-8)
    ax.set_xlabel("noise level")
    ax.set_ylabel("false positive rate")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=7)
    ax.set_title("Negative-control false positive rates")
    fig.tight_layout()
    fig.savefig(plots_dir / "robust_period_index_false_positive_rate.pdf")
    plt.close(fig)

    phase = positives.groupby(["noise_level"], as_index=False)[["certification_rate", "uncertain_rate", "rejection_rate"]].mean()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.stackplot(
        phase["noise_level"],
        phase["certification_rate"],
        phase["uncertain_rate"],
        phase["rejection_rate"],
        labels=["certified", "uncertain", "rejected"],
        alpha=0.85,
    )
    ax.set_xscale("symlog", linthresh=1e-8)
    ax.set_xlabel("noise level")
    ax.set_ylabel("mean rate across positive cases")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right")
    ax.set_title("Noise phase diagram")
    fig.tight_layout()
    fig.savefig(plots_dir / "robust_period_index_noise_phase_diagram.pdf")
    plt.close(fig)


def write_report(
    args: argparse.Namespace,
    df: pd.DataFrame,
    summary_df: pd.DataFrame,
    policy_df: pd.DataFrame,
    path: Path,
) -> None:
    key_columns = [
        "case_id",
        "source",
        "noise_type",
        "noise_level",
        "candidate_rank",
        "n",
        "certification_rate",
        "uncertain_rate",
        "rejection_rate",
        "correct_period_rate",
        "correct_index_rate",
        "false_lift_rate",
        "false_positive_central_rate",
        "false_positive_lift_rate",
        "false_negative_rejection_rate",
    ]
    transition_columns = [
        "case_id",
        "noise_type",
        "minimal_lift_rank",
        "small_noise_certification_rate",
        "first_uncertain_noise",
        "first_rejection_noise",
        "max_false_lift_rate",
    ]
    negative_columns = [
        "case_id",
        "source",
        "noise_type",
        "n",
        "max_false_positive_central_rate",
        "max_false_positive_lift_rate",
        "mean_rejection_rate",
    ]
    policy_columns = [
        "centrality_tol",
        "phase_tol",
        "confidence_margin",
        "small_noise_certification_rate",
        "false_positive_central_rate",
        "false_positive_lift_rate",
        "medium_noise_uncertain_rate",
        "large_noise_rejection_rate",
        "recommended",
    ]
    selected_noise_levels = {0.0, 1e-6, 1e-5, 3e-5, 1e-4, 1e-3, 1e-2}
    positive = summary_df[summary_df["source"] == "central_positive"]
    minimal_ranks = positive.groupby("case_id", as_index=False)["candidate_rank"].min()
    positive_minimal = positive.merge(minimal_ranks, on=["case_id", "candidate_rank"], how="inner")
    positive_curves = positive_minimal[positive_minimal["noise_level"].isin(selected_noise_levels)]
    transition_rows = []
    for (case_id, noise_type), group in positive_minimal.groupby(["case_id", "noise_type"]):
        ordered = group.sort_values("noise_level")
        small = ordered[ordered["noise_level"] <= 1e-5]
        uncertain = ordered[ordered["uncertain_rate"] > 0]
        rejected = ordered[ordered["rejection_rate"] > 0]
        transition_rows.append(
            {
                "case_id": case_id,
                "noise_type": noise_type,
                "minimal_lift_rank": int(ordered["candidate_rank"].iloc[0]),
                "small_noise_certification_rate": float(small["certification_rate"].mean()) if not small.empty else float("nan"),
                "first_uncertain_noise": float(uncertain["noise_level"].min()) if not uncertain.empty else None,
                "first_rejection_noise": float(rejected["noise_level"].min()) if not rejected.empty else None,
                "max_false_lift_rate": float(ordered["false_lift_rate"].max()),
            }
        )
    transition_df = pd.DataFrame(transition_rows)
    obstructed = summary_df[
        (summary_df["source"] == "rank_divisibility")
        & (summary_df["noise_level"].isin(selected_noise_levels))
    ].head(60)
    negatives_raw = summary_df[summary_df["source"].isin(["noncentral_negative", "trivial_abelian_negative"])]
    negatives = (
        negatives_raw.groupby(["case_id", "source", "noise_type"], as_index=False)
        .agg(
            n=("n", "sum"),
            max_false_positive_central_rate=("false_positive_central_rate", "max"),
            max_false_positive_lift_rate=("false_positive_lift_rate", "max"),
            mean_rejection_rate=("rejection_rate", "mean"),
        )
    )
    false_rows = summary_df[
        (summary_df["false_positive_lift_rate"] > 0)
        | (summary_df["false_positive_central_rate"] > 0)
        | (summary_df["false_lift_rate"] > 0)
    ]
    recommendation = policy_df[policy_df["recommended"]].iloc[0].to_dict()
    policy_display = policy_df.copy()
    for col in ["centrality_tol", "phase_tol"]:
        policy_display[col] = policy_display[col].map(lambda value: f"{float(value):.0e}")
    report = f"""# Robust Period-Index Calibration Report

This report is generated by `experiments/robust_period_index_calibration.py`.

## Exact Command

```bash
{args.command_string}
```

## Commit Hash

`{git_commit()}`

## Calibration Design

The calibration uses controlled central period-index systems, period-divisible
but index-obstructed ranks, noncentral controls, and a trivial abelian control.
The generated artifact uses `{args.seeds}` seeds per case/noise/noise-type
combination.  The default is `{DEFAULT_SEEDS}` seeds rather than 50 to keep
local runtime tractable; rerun with `--seeds 50` for a heavier calibration.

Noise levels: `{list(args.noise_levels)}`.
Noise types: `{list(args.noise_types)}`.

## Positive Central Robustness Curves

{format_markdown_table(_display_frame(positive_curves).to_dict("records"), key_columns)}

## Noise Transition Summary

{format_markdown_table(_display_frame(transition_df).to_dict("records"), transition_columns)}

## Period-Divisible But Index-Obstructed Rank Results

{format_markdown_table(_display_frame(obstructed).to_dict("records"), key_columns)}

## Noncentral And Trivial Negative Controls

{format_markdown_table(_display_frame(negatives).to_dict("records"), negative_columns)}

## False Positive And False Lift Rates

Rows with nonzero false-positive or false-lift rates:

{format_markdown_table(_display_frame(false_rows).to_dict("records"), key_columns) if not false_rows.empty else "No nonzero false-positive or false-lift grouped rows were observed."}

## Recommended Threshold Policy

{format_markdown_table(policy_display.to_dict("records"), policy_columns)}

Recommended policy: centrality tolerance `{recommendation["centrality_tol"]}`,
phase tolerance `{recommendation["phase_tol"]}`, confidence margin
`{recommendation["confidence_margin"]}`.  This recommendation is empirical on
the controlled calibration rows and preserves the certified-only lift policy.

## Algorithmic Conclusion

The detector recovers period/index at exact and small noise when the robust
commutator metrics remain within the certified threshold.  As noise grows, rows
move into `central_projective_candidate_uncertain` and then into
`rejected_noncentral`; uncertain rows do not lift.  Period-divisible but
index-obstructed ranks keep `selected_method = none` whenever certified.
Negative controls in this calibration did not select a period-index lift under
the recommended policy.

## Negative Boundaries

- This is controlled synthetic calibration, not evidence that natural neural
  residuals are Brauer classes.
- Uncertain candidates are diagnostics, not valid lifts.
- Period divisibility alone is not a lift certificate.
- Noncentral controls are not described as central/Brauer/projective classes.
- No C2M3 or natural model-merging performance improvement is claimed.

## Plots

- `reports/plots/robust_period_index_certification_rate.pdf`
- `reports/plots/robust_period_index_false_positive_rate.pdf`
- `reports/plots/robust_period_index_noise_phase_diagram.pdf`

## Environment

```json
{json.dumps(capture_environment(), indent=2)}
```
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "reports")
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--noise-levels", type=float, nargs="+", default=list(DEFAULT_NOISE_LEVELS))
    parser.add_argument("--noise-types", nargs="+", default=list(DEFAULT_NOISE_TYPES))
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()
    args.command_string = " ".join([sys.executable, *sys.argv])

    rows = calibration_rows(
        seeds=args.seeds,
        noise_levels=args.noise_levels,
        noise_types=args.noise_types,
    )
    df = pd.DataFrame(rows)
    summary_df = summary_rows(df)
    policy_df = policy_stats(df)

    csv_path = args.reports_dir / "csv" / "robust_period_index_calibration.csv"
    summary_path = args.reports_dir / "csv" / "robust_period_index_calibration_summary.csv"
    policy_path = args.reports_dir / "csv" / "robust_period_index_calibration_threshold_policies.csv"
    report_path = args.reports_dir / "robust_period_index_calibration_report.md"
    config_path = args.reports_dir / "configs" / "robust_period_index_calibration_config.json"
    plots_dir = args.reports_dir / "plots"

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    policy_df.to_csv(policy_path, index=False)
    if not args.skip_plots:
        write_plots(summary_df, plots_dir)
    save_json(
        config_path,
        {
            "argv": sys.argv,
            "environment": capture_environment(),
            "commit": git_commit(),
            "seeds": args.seeds,
            "noise_levels": args.noise_levels,
            "noise_types": args.noise_types,
            "default_seeds_note": "20 seeds used by default for local runtime; rerun with --seeds 50 for heavier calibration.",
        },
    )
    write_report(args, df, summary_df, policy_df, report_path)
    print(f"wrote {csv_path}")
    print(f"wrote {summary_path}")
    print(f"wrote {policy_path}")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
