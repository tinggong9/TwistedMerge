#!/usr/bin/env python
"""Nearest finite-Heisenberg projection benchmark for noisy learned chart maps."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt  # noqa: E402

from src.metrics import capture_environment, save_json  # noqa: E402
from src.model_merging_benchmark import format_markdown_table  # noqa: E402
from src.nearest_heisenberg_projection import (  # noqa: E402
    HEISENBERG_PROJECTION_METHOD,
    PROJECTION_RESIDUAL_THRESHOLDS,
    HeisenbergProjectionResult,
    canonical_heisenberg_generators,
    project_to_nearest_finite_heisenberg,
)
from src.period_index_detector import RobustPeriodIndexDetection  # noqa: E402
from src.time_frequency_benchmark import generate_paired_time_frequency_chart_dataset  # noqa: E402
from src.time_frequency_chart_denoising import (  # noqa: E402
    DenoisedChartRecovery,
    fit_nearest_unitary_projection,
    fit_raw_least_squares_denoising,
    fit_unitary_global_chart_synchronization,
)
from src.time_frequency_learned_charts import (  # noqa: E402
    CALIBRATED_CONFIDENCE_MARGIN,
    CALIBRATED_TOLERANCE,
    LIFT_METHOD,
    detect_recovered_chart_generators,
    random_noncentral_chart_generators,
    relative_residual,
    selected_method_for,
)


BASELINE_METHODS = (
    "raw_least_squares",
    "nearest_unitary_projection",
    "unitary_global_chart_synchronization",
)
PROJECTION_METHODS = (
    "nearest_heisenberg_projection",
    "unitary_then_heisenberg_projection",
    "global_sync_then_heisenberg_projection",
)
ALL_METHODS = BASELINE_METHODS + PROJECTION_METHODS


@dataclass(frozen=True)
class CaseSpec:
    d: int
    k: int
    candidate_ranks: tuple[int, ...]
    n_classes: int = 3

    @property
    def case_id(self) -> str:
        return f"time_frequency_d{self.d}_k{self.k}"

    @property
    def expected_period(self) -> int:
        return self.d

    @property
    def expected_index(self) -> int:
        return self.d**self.k


CSV_COLUMNS = [
    "case_id",
    "d",
    "k",
    "seed",
    "noise_level",
    "method",
    "candidate_rank",
    "expected_period",
    "expected_index",
    "projection_residual",
    "projection_residual_threshold",
    "projection_accepted",
    "learned_operator_error_raw",
    "learned_operator_error_denoised",
    "commutator_residual_before",
    "commutator_residual_after",
    "exponent_matrix_residual",
    "detector_status_before_projection",
    "detector_status_after_projection",
    "detected_period",
    "detected_index",
    "correct_period",
    "correct_index",
    "period_divides_rank",
    "index_divides_rank",
    "decision",
    "selected_method",
    "expected_decision",
    "pass_fail",
    "false_lift",
    "max_centrality_score",
    "max_phase_residual",
    "min_root_margin",
    "notes",
]


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def default_cases(*, include_d2k3: bool) -> tuple[CaseSpec, ...]:
    cases = [CaseSpec(2, 2, (2, 4)), CaseSpec(3, 2, (3, 6, 9))]
    if include_d2k3:
        cases.append(CaseSpec(2, 3, (2, 4, 8)))
    return tuple(cases)


def expected_rank_decision(case: CaseSpec, rank: int, detector_status: str) -> str:
    if detector_status == "candidate_uncertain":
        return "central_projective_candidate_uncertain"
    if detector_status == "rejected_noncentral":
        return "not_central_projective"
    if detector_status == "unknown_index":
        return "central_projective_index_unknown"
    if rank > 0 and rank % case.expected_index == 0:
        return "period_index_lift_success"
    if rank > 0 and rank % case.expected_period == 0:
        return "period_divisible_index_obstructed"
    return "rank_obstructed"


def expected_projection_decision(
    case: CaseSpec,
    rank: int,
    *,
    projection_accepted: bool,
    control: bool = False,
) -> str:
    if control:
        return "not_central_projective"
    if not projection_accepted:
        return "heisenberg_projection_rejected"
    if rank > 0 and rank % case.expected_index == 0:
        return "heisenberg_projection_lift_success"
    return "heisenberg_projection_index_obstructed"


def _operator_error(
    case: CaseSpec,
    generators: Mapping[str, np.ndarray],
) -> float:
    try:
        canonical = canonical_heisenberg_generators(case.d, case.k, generator_names=tuple(generators))
    except Exception:
        return float("nan")
    errors = []
    for name in canonical:
        learned = np.asarray(generators[name])
        if learned.shape != canonical[name].shape:
            return float("nan")
        errors.append(relative_residual(learned, canonical[name]))
    return float(np.mean(errors)) if errors else float("nan")


def baseline_passes(case: CaseSpec, rank: int, detection: RobustPeriodIndexDetection, selected_method: str) -> bool:
    expected = expected_rank_decision(case, rank, detection.status)
    if selected_method == LIFT_METHOD and expected != "period_index_lift_success":
        return False
    if detection.status != "certified":
        return selected_method == "none"
    return (
        detection.period == case.expected_period
        and detection.index == case.expected_index
        and detection.decision == expected
        and ((selected_method == LIFT_METHOD) == (expected == "period_index_lift_success"))
    )


def projection_passes(
    case: CaseSpec,
    rank: int,
    projection: HeisenbergProjectionResult,
    *,
    control: bool,
) -> bool:
    if projection.selected_method != "none":
        return (
            not control
            and projection.projection_accepted
            and projection.decision == "heisenberg_projection_lift_success"
            and rank % case.expected_index == 0
            and projection.detector_after_projection.status == "certified"
            and projection.detector_after_projection.period == case.expected_period
            and projection.detector_after_projection.index == case.expected_index
        )
    if control:
        return not projection.projection_accepted
    if not projection.projection_accepted:
        return True
    detection = projection.detector_after_projection
    if detection.status != "certified":
        return True
    return detection.period == case.expected_period and detection.index == case.expected_index


def pack_baseline_row(
    *,
    case: CaseSpec,
    seed: int,
    noise_level: float,
    method: str,
    threshold: float,
    raw_error: float,
    recovery: DenoisedChartRecovery,
    candidate_rank: int,
) -> dict[str, object]:
    detection = detect_recovered_chart_generators(recovery.candidate_generators, candidate_rank)
    selected_method = selected_method_for(detection)
    expected = expected_rank_decision(case, candidate_rank, detection.status)
    false_lift = selected_method != "none" and expected != "period_index_lift_success"
    return {
        "case_id": f"{case.case_id}_{method}_rank{candidate_rank}_noise{noise_level:g}_seed{seed}_thr{threshold:g}",
        "d": case.d,
        "k": case.k,
        "seed": seed,
        "noise_level": float(noise_level),
        "method": method,
        "candidate_rank": candidate_rank,
        "expected_period": case.expected_period,
        "expected_index": case.expected_index,
        "projection_residual": float("nan"),
        "projection_residual_threshold": float(threshold),
        "projection_accepted": False,
        "learned_operator_error_raw": raw_error,
        "learned_operator_error_denoised": _operator_error(case, recovery.candidate_generators),
        "commutator_residual_before": detection.max_phase_residual,
        "commutator_residual_after": detection.max_phase_residual,
        "exponent_matrix_residual": 0.0 if detection.status == "certified" else float("nan"),
        "detector_status_before_projection": detection.status,
        "detector_status_after_projection": detection.status,
        "detected_period": detection.period,
        "detected_index": detection.index,
        "correct_period": detection.period == case.expected_period,
        "correct_index": detection.index == case.expected_index,
        "period_divides_rank": detection.period_divides_rank,
        "index_divides_rank": detection.index_divides_rank,
        "decision": detection.decision,
        "selected_method": selected_method,
        "expected_decision": expected,
        "pass_fail": "pass" if baseline_passes(case, candidate_rank, detection, selected_method) else "fail",
        "false_lift": bool(false_lift),
        "max_centrality_score": detection.max_centrality_score,
        "max_phase_residual": detection.max_phase_residual,
        "min_root_margin": detection.min_root_margin,
        "notes": "; ".join(recovery.notes + tuple(detection.notes)),
    }


def pack_projection_row(
    *,
    case: CaseSpec,
    seed: int,
    noise_level: float,
    method: str,
    threshold: float,
    raw_error: float,
    learned_generators: Mapping[str, np.ndarray],
    candidate_rank: int,
    control: bool = False,
    notes_prefix: str = "",
) -> dict[str, object]:
    projection = project_to_nearest_finite_heisenberg(
        learned_generators,
        expected_d=case.d,
        expected_k=case.k,
        candidate_rank=candidate_rank,
        projection_residual_threshold=threshold,
        generator_names=tuple(canonical_heisenberg_generators(case.d, case.k)),
    )
    detection = projection.detector_after_projection
    expected = expected_projection_decision(
        case,
        candidate_rank,
        projection_accepted=projection.projection_accepted,
        control=control,
    )
    false_lift = projection.selected_method != "none" and expected != "heisenberg_projection_lift_success"
    notes = [notes_prefix] if notes_prefix else []
    notes.extend(projection.notes)
    return {
        "case_id": f"{case.case_id}_{method}_rank{candidate_rank}_noise{noise_level:g}_seed{seed}_thr{threshold:g}",
        "d": case.d,
        "k": case.k,
        "seed": seed,
        "noise_level": float(noise_level),
        "method": method,
        "candidate_rank": candidate_rank,
        "expected_period": case.expected_period,
        "expected_index": case.expected_index,
        "projection_residual": projection.projection_residual,
        "projection_residual_threshold": float(threshold),
        "projection_accepted": projection.projection_accepted,
        "learned_operator_error_raw": raw_error,
        "learned_operator_error_denoised": _operator_error(case, projection.projected_generators),
        "commutator_residual_before": projection.commutator_residual_before,
        "commutator_residual_after": projection.commutator_residual_after,
        "exponent_matrix_residual": projection.exponent_matrix_residual,
        "detector_status_before_projection": projection.detector_before_projection.status,
        "detector_status_after_projection": detection.status,
        "detected_period": detection.period,
        "detected_index": detection.index,
        "correct_period": detection.period == case.expected_period,
        "correct_index": detection.index == case.expected_index,
        "period_divides_rank": detection.period_divides_rank,
        "index_divides_rank": detection.index_divides_rank,
        "decision": projection.decision,
        "selected_method": projection.selected_method,
        "expected_decision": expected,
        "pass_fail": "pass" if projection_passes(case, candidate_rank, projection, control=control) else "fail",
        "false_lift": bool(false_lift),
        "max_centrality_score": detection.max_centrality_score,
        "max_phase_residual": detection.max_phase_residual,
        "min_root_margin": detection.min_root_margin,
        "notes": "; ".join(notes),
    }


def _named_random_noncentral(case: CaseSpec, seed: int) -> dict[str, np.ndarray]:
    names = tuple(canonical_heisenberg_generators(case.d, case.k))
    random_maps = random_noncentral_chart_generators(case.expected_index, count=len(names), seed=seed)
    return {name: np.asarray(matrix, dtype=complex) for name, matrix in zip(names, random_maps.values(), strict=True)}


def _named_nearly_scalar_noncentral(case: CaseSpec, seed: int) -> dict[str, np.ndarray]:
    names = tuple(canonical_heisenberg_generators(case.d, case.k))
    width = case.expected_index
    rng = np.random.default_rng(seed)
    maps = {}
    for idx, name in enumerate(names):
        raw = rng.normal(size=(width, width)) + 1j * rng.normal(size=(width, width))
        perturbation = raw / max(float(np.linalg.norm(raw, ord="fro")), 1e-12)
        maps[name] = np.eye(width, dtype=complex) + 0.03 * (idx + 1) * perturbation
    return maps


def _named_trivial_abelian(case: CaseSpec) -> dict[str, np.ndarray]:
    width = case.expected_index
    return {name: np.eye(width, dtype=complex) for name in canonical_heisenberg_generators(case.d, case.k)}


def negative_control_rows(
    *,
    cases: Iterable[CaseSpec],
    seeds: int,
    seed_offset: int,
    thresholds: Iterable[float],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for case in cases:
        rank = case.expected_index
        for seed_delta in range(seeds):
            seed = int(seed_offset + seed_delta)
            controls = [
                ("control_random_noncentral", _named_random_noncentral(case, 1000 + seed), "random noncentral chart maps"),
                ("control_nearly_scalar_noncentral", _named_nearly_scalar_noncentral(case, 2000 + seed), "nearly scalar but noncentral maps"),
                ("control_trivial_abelian", _named_trivial_abelian(case), "trivial abelian identity maps"),
            ]
            wrong_d = 3 if case.d == 2 else 2
            controls.append(
                (
                    "control_wrong_period",
                    canonical_heisenberg_generators(wrong_d, case.k),
                    f"wrong-period canonical maps actual_d={wrong_d}",
                )
            )
            for threshold in thresholds:
                for control_name, generators, note in controls:
                    for method, learned in [
                        ("nearest_heisenberg_projection", generators),
                        ("unitary_then_heisenberg_projection", generators),
                    ]:
                        row = pack_projection_row(
                            case=case,
                            seed=seed,
                            noise_level=-1.0,
                            method=method,
                            threshold=float(threshold),
                            raw_error=_operator_error(case, learned),
                            learned_generators=learned,
                            candidate_rank=rank,
                            control=True,
                            notes_prefix=note,
                        )
                        row["case_id"] = f"{control_name}_{row['case_id']}"
                        rows.append(row)
    return rows


def scenario_rows(
    *,
    cases: Iterable[CaseSpec],
    seeds: int,
    seed_offset: int,
    noise_levels: Iterable[float],
    thresholds: Iterable[float],
    train_samples: int,
    validation_samples: int,
    test_samples: int,
    include_controls: bool,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    case_tuple = tuple(cases)
    threshold_tuple = tuple(float(value) for value in thresholds)
    for case in case_tuple:
        for noise_level in noise_levels:
            for seed_delta in range(seeds):
                seed = int(seed_offset + seed_delta)
                dataset = generate_paired_time_frequency_chart_dataset(
                    case.d,
                    case.k,
                    n_classes=case.n_classes,
                    train_samples=train_samples,
                    validation_samples=validation_samples,
                    test_samples=test_samples,
                    noise_level=float(noise_level),
                    seed=seed,
                )
                raw = fit_raw_least_squares_denoising(dataset)
                nearest_unitary = fit_nearest_unitary_projection(dataset)
                unitary_global = fit_unitary_global_chart_synchronization(dataset)
                recoveries = {
                    "raw_least_squares": raw,
                    "nearest_unitary_projection": nearest_unitary,
                    "unitary_global_chart_synchronization": unitary_global,
                }
                raw_error = raw.learned_operator_error_mean_raw
                projection_sources = {
                    "nearest_heisenberg_projection": raw.candidate_generators,
                    "unitary_then_heisenberg_projection": nearest_unitary.candidate_generators,
                    "global_sync_then_heisenberg_projection": unitary_global.candidate_generators,
                }
                for threshold in threshold_tuple:
                    for rank in case.candidate_ranks:
                        for method, recovery in recoveries.items():
                            rows.append(
                                pack_baseline_row(
                                    case=case,
                                    seed=seed,
                                    noise_level=float(noise_level),
                                    method=method,
                                    threshold=threshold,
                                    raw_error=raw_error,
                                    recovery=recovery,
                                    candidate_rank=rank,
                                )
                            )
                        for method, learned_generators in projection_sources.items():
                            rows.append(
                                pack_projection_row(
                                    case=case,
                                    seed=seed,
                                    noise_level=float(noise_level),
                                    method=method,
                                    threshold=threshold,
                                    raw_error=raw_error,
                                    learned_generators=learned_generators,
                                    candidate_rank=rank,
                                )
                            )
    if include_controls:
        rows.extend(
            negative_control_rows(
                cases=case_tuple,
                seeds=seeds,
                seed_offset=seed_offset,
                thresholds=threshold_tuple,
            )
        )
    return rows


def summary_rows(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    frame["projection_accepted_bool"] = frame["projection_accepted"].astype(bool)
    frame["is_projection_method"] = frame["method"].isin(PROJECTION_METHODS)
    frame["is_control"] = frame["case_id"].str.startswith("control_")
    frame["certified_before"] = frame["detector_status_before_projection"] == "certified"
    frame["certified_after_raw"] = frame["detector_status_after_projection"] == "certified"
    frame["certified_after_effective"] = np.where(
        frame["is_projection_method"],
        frame["projection_accepted_bool"] & frame["certified_after_raw"],
        frame["certified_after_raw"],
    )
    frame["correct_period_effective"] = frame["correct_period"] & frame["certified_after_effective"]
    frame["correct_index_effective"] = frame["correct_index"] & frame["certified_after_effective"]
    frame["lift_selected"] = frame["selected_method"] != "none"
    frame["false_lift_bool"] = frame["false_lift"].astype(bool)
    frame["false_positive_central"] = (
        frame["is_control"] & frame["projection_accepted_bool"] & frame["certified_after_raw"]
    )
    frame["obstructed_rejected"] = (
        frame["expected_decision"].isin(["period_divisible_index_obstructed", "heisenberg_projection_index_obstructed"])
        & (frame["selected_method"] == "none")
    )
    grouped = (
        frame.groupby(["method", "d", "k", "noise_level", "candidate_rank", "projection_residual_threshold"], dropna=False)
        .agg(
            n=("case_id", "count"),
            projection_acceptance_rate=("projection_accepted_bool", "mean"),
            certification_rate_before_projection=("certified_before", "mean"),
            certification_rate_after_projection=("certified_after_effective", "mean"),
            correct_period_rate=("correct_period_effective", "mean"),
            correct_index_rate=("correct_index_effective", "mean"),
            lift_rate=("lift_selected", "mean"),
            false_lift_rate=("false_lift_bool", "mean"),
            false_positive_central_rate=("false_positive_central", "mean"),
            period_divisible_index_obstructed_rejection_rate=("obstructed_rejected", "mean"),
            mean_projection_residual=("projection_residual", "mean"),
            mean_commutator_residual_before=("commutator_residual_before", "mean"),
            mean_commutator_residual_after=("commutator_residual_after", "mean"),
            mean_operator_error_raw=("learned_operator_error_raw", "mean"),
            mean_operator_error_denoised=("learned_operator_error_denoised", "mean"),
        )
        .reset_index()
    )
    baseline = grouped[grouped["method"].isin(BASELINE_METHODS)][
        ["d", "k", "noise_level", "candidate_rank", "projection_residual_threshold", "certification_rate_after_projection"]
    ].groupby(["d", "k", "noise_level", "candidate_rank", "projection_residual_threshold"], as_index=False).max()
    baseline = baseline.rename(columns={"certification_rate_after_projection": "best_previous_certification_rate"})
    grouped = grouped.merge(
        baseline,
        on=["d", "k", "noise_level", "candidate_rank", "projection_residual_threshold"],
        how="left",
    )
    grouped["certification_gain_over_best_previous"] = (
        grouped["certification_rate_after_projection"] - grouped["best_previous_certification_rate"].fillna(0.0)
    )
    return grouped.drop(columns=["best_previous_certification_rate"])


def _display_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda value: "nan" if pd.isna(value) else f"{float(value):.4g}")
    return out


def write_plots(df: pd.DataFrame, summary_df: pd.DataFrame, plots_dir: Path) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    lift_rank = summary_df["candidate_rank"] == (summary_df["d"] ** summary_df["k"])
    non_control = summary_df["noise_level"] >= 0
    selected = summary_df[lift_rank & non_control]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for method, group in selected.groupby("method"):
        thresholded = group[group["projection_residual_threshold"] == group["projection_residual_threshold"].max()]
        rate = thresholded.groupby("noise_level", as_index=False)["certification_rate_after_projection"].mean()
        ax.plot(rate["noise_level"], rate["certification_rate_after_projection"], marker="o", label=method)
    ax.set_xlabel("chart observation noise")
    ax.set_ylabel("effective certification rate")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xscale("symlog", linthresh=1e-4)
    ax.set_title("Nearest Heisenberg projection certification")
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(plots_dir / "time_frequency_heisenberg_projection_certification_rate.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    projection = summary_df[summary_df["method"].isin(PROJECTION_METHODS) & non_control]
    for method, group in projection.groupby("method"):
        thresholded = group[group["projection_residual_threshold"] == group["projection_residual_threshold"].max()]
        residual = thresholded.groupby("noise_level", as_index=False)["mean_projection_residual"].mean()
        ax.plot(residual["noise_level"], residual["mean_projection_residual"], marker="o", label=method)
    ax.set_xlabel("chart observation noise")
    ax.set_ylabel("mean projection residual")
    ax.set_xscale("symlog", linthresh=1e-4)
    ax.set_yscale("symlog", linthresh=1e-10)
    ax.set_title("Nearest Heisenberg projection residual")
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(plots_dir / "time_frequency_heisenberg_projection_residual.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    false_lift = summary_df.groupby(["method", "noise_level"], as_index=False)["false_lift_rate"].mean()
    for method, group in false_lift.groupby("method"):
        ax.plot(group["noise_level"], group["false_lift_rate"], marker="o", label=method)
    ax.set_xlabel("chart observation noise")
    ax.set_ylabel("false lift rate")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xscale("symlog", linthresh=1e-4)
    ax.set_title("Nearest Heisenberg projection false lifts")
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(plots_dir / "time_frequency_heisenberg_projection_false_lift.pdf")
    plt.close(fig)


def write_report(args: argparse.Namespace, df: pd.DataFrame, summary_df: pd.DataFrame, path: Path) -> None:
    summary_columns = [
        "method",
        "d",
        "k",
        "noise_level",
        "candidate_rank",
        "projection_residual_threshold",
        "n",
        "projection_acceptance_rate",
        "certification_rate_before_projection",
        "certification_rate_after_projection",
        "certification_gain_over_best_previous",
        "lift_rate",
        "false_lift_rate",
        "false_positive_central_rate",
    ]
    residual_columns = [
        "method",
        "d",
        "k",
        "noise_level",
        "candidate_rank",
        "projection_residual_threshold",
        "projection_acceptance_rate",
        "mean_projection_residual",
        "mean_commutator_residual_before",
        "mean_commutator_residual_after",
        "mean_operator_error_raw",
        "mean_operator_error_denoised",
    ]
    rank_columns = [
        "method",
        "d",
        "k",
        "candidate_rank",
        "projection_residual_threshold",
        "projection_accepted",
        "detector_status_after_projection",
        "decision",
        "selected_method",
        "expected_decision",
        "pass_fail",
    ]
    lift_ranks = summary_df[
        (summary_df["candidate_rank"] == (summary_df["d"] ** summary_df["k"]))
        & (summary_df["noise_level"] >= 0)
        & (summary_df["projection_residual_threshold"] == max(args.projection_thresholds))
    ]
    residual_table = summary_df[
        (summary_df["method"].isin(PROJECTION_METHODS))
        & (summary_df["noise_level"].isin([0.0, 1e-3, 1e-2, 5e-2]))
    ].head(60)
    rank_table = df[
        (df["noise_level"] == 0.0)
        & (df["seed"] == args.seed_offset)
        & (df["projection_residual_threshold"] == max(args.projection_thresholds))
    ].head(60)
    controls = df[df["case_id"].str.startswith("control_")]
    false_or_control = summary_df[
        (summary_df["false_lift_rate"] > 0)
        | (summary_df["false_positive_central_rate"] > 0)
        | (summary_df["noise_level"] < 0)
    ].head(60)
    gains = summary_df[
        (summary_df["method"].isin(PROJECTION_METHODS))
        & (summary_df["certification_gain_over_best_previous"] > 0)
    ].sort_values(["certification_gain_over_best_previous", "noise_level"], ascending=[False, True]).head(20)
    failed = summary_df[
        (summary_df["method"].isin(PROJECTION_METHODS))
        & (summary_df["noise_level"] >= 0)
        & (summary_df["projection_acceptance_rate"] < 1.0)
    ].head(30)

    if false_or_control.empty:
        false_or_control = pd.DataFrame(
            [{"method": "all_tested_methods", "false_lift_rate": 0.0, "false_positive_central_rate": 0.0, "n": len(df)}]
        )

    report = f"""# Time-Frequency Heisenberg Projection Report

This report is generated by `experiments/time_frequency_heisenberg_projection_benchmark.py`.

## Exact Command

```bash
{args.command_string}
```

## Commit Hash

`{git_commit()}`

## Purpose

Previous denoising improved small-noise learned chart recovery, but moderate
noise often still failed the calibrated central period-index detector.  This
benchmark tests nearest finite-Heisenberg projection with a residual gate:
canonical replacement is not accepted unless the learned commutator form
matches the expected finite-Heisenberg form and the learned-to-projected
operator residual is below the configured threshold.

## Projection Methods

- `raw_least_squares`: baseline learned input chart maps.
- `nearest_unitary_projection`: baseline nearest-orthogonal/unitary denoising.
- `unitary_global_chart_synchronization`: baseline synchronized unitary gauges.
- `nearest_heisenberg_projection`: commutator-form projection from raw maps.
- `unitary_then_heisenberg_projection`: unitary-denoised maps followed by
  finite-Heisenberg projection.
- `global_sync_then_heisenberg_projection`: synchronized gauges followed by
  finite-Heisenberg projection.

## Main Certification Table

{format_markdown_table(_display_frame(lift_ranks).to_dict("records"), summary_columns)}

## Projection Residual Table

{format_markdown_table(_display_frame(residual_table).to_dict("records"), residual_columns)}

## Rank-Threshold Table

{format_markdown_table(_display_frame(rank_table).to_dict("records"), rank_columns)}

## False-Lift And Negative-Control Table

{format_markdown_table(_display_frame(false_or_control).to_dict("records"), [col for col in summary_columns if col in false_or_control.columns])}

Negative controls generated {len(controls)} rows covering random noncentral,
nearly scalar noncentral, trivial abelian, and wrong-period controls.

## Best Method Discussion

The projection methods are evaluated by effective certification rate after the
projection residual gate.  A projected detector certificate is not enough: rows
with large projection residuals keep `selected_method = none`.

## What Improved

{format_markdown_table(_display_frame(gains).to_dict("records"), summary_columns)}

## What Failed

{format_markdown_table(_display_frame(failed).to_dict("records"), summary_columns)}

Large-noise rows often fail by residual threshold even though the canonical
projected generators themselves are certified.  Those rows are intentionally
reported as rejected rather than lifted.

## Algorithmic Conclusion

Nearest finite-Heisenberg projection is a controlled diagnostic/projection
step.  It supports a lift only when the commutator-form check, projection
residual threshold, robust period-index detector, and index-divisibility check
all pass.

## Negative Boundaries

- No MNIST/CIFAR residual is claimed to be a Brauer or period-index class.
- No lift is selected from period divisibility alone.
- No lift is selected from projection unless the residual is accepted and the
  robust detector certifies period and index.
- Noncentral maps are not called central/Brauer classes.
- Canonical finite-Heisenberg replacement without small residual is not counted
  as recovery.

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
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--train-samples", type=int, default=2000)
    parser.add_argument("--validation-samples", type=int, default=500)
    parser.add_argument("--test-samples", type=int, default=1000)
    parser.add_argument(
        "--noise-levels",
        type=float,
        nargs="+",
        default=[0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 5e-2],
    )
    parser.add_argument(
        "--projection-thresholds",
        type=float,
        nargs="+",
        default=list(PROJECTION_RESIDUAL_THRESHOLDS),
    )
    parser.add_argument("--include-d2k3", action="store_true")
    parser.add_argument("--skip-controls", action="store_true")
    args = parser.parse_args()
    args.command_string = " ".join([sys.executable, *sys.argv])

    cases = default_cases(include_d2k3=args.include_d2k3)
    rows = scenario_rows(
        cases=cases,
        seeds=args.seeds,
        seed_offset=args.seed_offset,
        noise_levels=args.noise_levels,
        thresholds=args.projection_thresholds,
        train_samples=args.train_samples,
        validation_samples=args.validation_samples,
        test_samples=args.test_samples,
        include_controls=not args.skip_controls,
    )
    df = pd.DataFrame(rows, columns=CSV_COLUMNS)
    summary_df = summary_rows(df)

    csv_path = args.reports_dir / "csv" / "time_frequency_heisenberg_projection_benchmark.csv"
    summary_path = args.reports_dir / "csv" / "time_frequency_heisenberg_projection_summary.csv"
    report_path = args.reports_dir / "time_frequency_heisenberg_projection_report.md"
    config_path = args.reports_dir / "configs" / "time_frequency_heisenberg_projection_config.json"
    plots_dir = args.reports_dir / "plots"

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    write_plots(df, summary_df, plots_dir)
    save_json(
        config_path,
        {
            "argv": sys.argv,
            "command": args.command_string,
            "commit": git_commit(),
            "environment": capture_environment(),
            "cases": [
                {
                    "d": case.d,
                    "k": case.k,
                    "candidate_ranks": list(case.candidate_ranks),
                    "expected_period": case.expected_period,
                    "expected_index": case.expected_index,
                }
                for case in cases
            ],
            "noise_levels": args.noise_levels,
            "seeds": args.seeds,
            "seed_offset": args.seed_offset,
            "train_samples": args.train_samples,
            "validation_samples": args.validation_samples,
            "test_samples": args.test_samples,
            "projection_residual_thresholds": args.projection_thresholds,
            "methods": list(ALL_METHODS),
            "negative_controls": not args.skip_controls,
            "calibrated_centrality_tolerance": CALIBRATED_TOLERANCE,
            "calibrated_phase_tolerance": CALIBRATED_TOLERANCE,
            "calibrated_confidence_margin": CALIBRATED_CONFIDENCE_MARGIN,
            "scope": {
                "mnist_cifar_claim": "not_claimed",
                "canonical_replacement_without_small_residual": "rejected",
                "uncertain_lift_policy": "no_lift",
                "period_divisibility_policy": "index_divisibility_required_for_lift",
                "projection_method": HEISENBERG_PROJECTION_METHOD,
            },
        },
    )
    write_report(args, df, summary_df, report_path)

    print(f"wrote {csv_path}")
    print(f"wrote {summary_path}")
    print(f"wrote {report_path}")
    print(f"wrote {plots_dir / 'time_frequency_heisenberg_projection_certification_rate.pdf'}")
    print(f"wrote {plots_dir / 'time_frequency_heisenberg_projection_residual.pdf'}")
    print(f"wrote {plots_dir / 'time_frequency_heisenberg_projection_false_lift.pdf'}")


if __name__ == "__main__":
    main()
