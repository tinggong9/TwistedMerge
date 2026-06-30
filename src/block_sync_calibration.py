"""Calibration helpers for accepting global block synchronization."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable

import numpy as np

from .global_block_synchronization import GlobalBlockSyncResult


@dataclass(frozen=True)
class BlockSyncCalibration:
    threshold: float
    target_false_positive_rate: float
    observed_false_positive_rate: float
    observed_true_positive_rate: float
    n_positive: int
    n_negative: int
    accepted_positive_count: int
    accepted_negative_count: int
    notes: str = ""


@dataclass(frozen=True)
class BlockSyncPolicy:
    name: str
    threshold: float
    target_false_positive_rate: float
    observed_false_positive_rate: float
    observed_true_positive_rate: float
    uncertain_band: float = 0.0
    uncertain_rate: float = 0.0
    false_scalar_projective_lift_rate: float = 0.0
    raw_calibrated_threshold: float | None = None
    effective_threshold: float | None = None
    numerical_floor: float = 0.0
    notes: str = ""


def _finite_residuals(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        raise ValueError("at least one finite residual is required")
    return arr


def calibrate_connection_residual_threshold(
    positive_residuals: Iterable[float],
    negative_residuals: Iterable[float],
    *,
    target_false_positive_rate: float = 0.05,
    safety_margin: float = 0.0,
) -> BlockSyncCalibration:
    """Choose an acceptance threshold from positive and negative controls.

    Positive controls are globally generated gauges that should be accepted.
    Negative controls are fake projected cycles or noncentral holonomies that
    should be rejected.  The selected threshold maximizes positive-control
    acceptance subject to the empirical false-positive-rate target.
    """

    positives = _finite_residuals(positive_residuals)
    negatives = _finite_residuals(negative_residuals)
    if not 0.0 <= target_false_positive_rate <= 1.0:
        raise ValueError("target_false_positive_rate must lie in [0, 1]")
    if safety_margin < 0.0:
        raise ValueError("safety_margin must be nonnegative")

    candidates = np.unique(np.concatenate([positives, negatives]))
    best_threshold = float(np.min(candidates) - safety_margin)
    best_true_positive_rate = -1.0
    best_false_positive_rate = float("inf")
    for raw_threshold in candidates:
        threshold = float(raw_threshold - safety_margin)
        false_positive_rate = float(np.mean(negatives <= threshold))
        if false_positive_rate > target_false_positive_rate:
            continue
        true_positive_rate = float(np.mean(positives <= threshold))
        if (
            true_positive_rate > best_true_positive_rate
            or (
                np.isclose(true_positive_rate, best_true_positive_rate)
                and false_positive_rate < best_false_positive_rate
            )
        ):
            best_threshold = threshold
            best_true_positive_rate = true_positive_rate
            best_false_positive_rate = false_positive_rate

    if best_true_positive_rate < 0.0:
        best_threshold = float(np.min(negatives) - safety_margin)
        best_false_positive_rate = float(np.mean(negatives <= best_threshold))
        best_true_positive_rate = float(np.mean(positives <= best_threshold))

    accepted_positive = int(np.sum(positives <= best_threshold))
    accepted_negative = int(np.sum(negatives <= best_threshold))
    return BlockSyncCalibration(
        threshold=float(best_threshold),
        target_false_positive_rate=float(target_false_positive_rate),
        observed_false_positive_rate=float(accepted_negative / len(negatives)),
        observed_true_positive_rate=float(accepted_positive / len(positives)),
        n_positive=int(len(positives)),
        n_negative=int(len(negatives)),
        accepted_positive_count=accepted_positive,
        accepted_negative_count=accepted_negative,
        notes="threshold selected on controlled positive/negative residuals",
    )


def calibrate_block_sync_policies(
    positive_residuals: Iterable[float],
    negative_residuals: Iterable[float],
    *,
    loose_band_fraction: float = 0.15,
    numerical_floor: float = 0.0,
) -> list[BlockSyncPolicy]:
    """Return strict, balanced, and loose diagnostic residual policies.

    The strict and balanced policies are acceptance policies.  The loose policy
    keeps the same balanced threshold but adds an uncertainty band around it;
    callers should report uncertain rows instead of converting them into
    accepted block-gauge evidence.
    """

    if numerical_floor < 0.0:
        raise ValueError("numerical_floor must be nonnegative")

    positives = _finite_residuals(positive_residuals)
    negatives = _finite_residuals(negative_residuals)
    strict = calibrate_connection_residual_threshold(
        positives,
        negatives,
        target_false_positive_rate=0.0,
    )
    balanced = calibrate_connection_residual_threshold(
        positives,
        negatives,
        target_false_positive_rate=0.01,
    )
    strict_threshold = max(float(strict.threshold), float(numerical_floor))
    balanced_threshold = max(float(balanced.threshold), float(numerical_floor))
    strict_accepted_positive = int(np.sum(positives <= strict_threshold))
    strict_accepted_negative = int(np.sum(negatives <= strict_threshold))
    balanced_accepted_positive = int(np.sum(positives <= balanced_threshold))
    balanced_accepted_negative = int(np.sum(negatives <= balanced_threshold))

    rows = [
        BlockSyncPolicy(
            name="strict",
            threshold=strict_threshold,
            target_false_positive_rate=strict.target_false_positive_rate,
            observed_false_positive_rate=float(strict_accepted_negative / len(negatives)),
            observed_true_positive_rate=float(strict_accepted_positive / len(positives)),
            raw_calibrated_threshold=float(strict.threshold),
            effective_threshold=strict_threshold,
            numerical_floor=float(numerical_floor),
            notes="zero empirical false positives on supplied negative controls after numerical floor",
        ),
        BlockSyncPolicy(
            name="balanced",
            threshold=balanced_threshold,
            target_false_positive_rate=balanced.target_false_positive_rate,
            observed_false_positive_rate=float(balanced_accepted_negative / len(negatives)),
            observed_true_positive_rate=float(balanced_accepted_positive / len(positives)),
            raw_calibrated_threshold=float(balanced.threshold),
            effective_threshold=balanced_threshold,
            numerical_floor=float(numerical_floor),
            notes="maximizes positive acceptance subject to empirical FPR <= 0.01 after numerical floor",
        ),
    ]
    band = max(abs(balanced_threshold) * float(loose_band_fraction), 1e-12)
    all_residuals = np.concatenate([positives, negatives])
    uncertain_rate = float(np.mean(np.abs(all_residuals - balanced_threshold) <= band))
    rows.append(
        BlockSyncPolicy(
            name="loose_diagnostic",
            threshold=balanced_threshold,
            target_false_positive_rate=balanced.target_false_positive_rate,
            observed_false_positive_rate=float(balanced_accepted_negative / len(negatives)),
            observed_true_positive_rate=float(balanced_accepted_positive / len(positives)),
            uncertain_band=band,
            uncertain_rate=uncertain_rate,
            raw_calibrated_threshold=float(balanced.threshold),
            effective_threshold=balanced_threshold,
            numerical_floor=float(numerical_floor),
            notes="near-threshold rows are uncertain rather than accepted/rejected",
        )
    )
    return rows


def apply_calibration_floor(calibration: BlockSyncCalibration, numerical_floor: float) -> BlockSyncCalibration:
    """Return an acceptance calibration with an explicit numerical floor."""

    if numerical_floor < 0.0:
        raise ValueError("numerical_floor must be nonnegative")
    effective_threshold = max(float(calibration.threshold), float(numerical_floor))
    if effective_threshold == calibration.threshold:
        return calibration
    return replace(
        calibration,
        threshold=effective_threshold,
        notes=(
            f"{calibration.notes}; raw_calibrated_threshold={calibration.threshold:.6g}; "
            f"effective_threshold={effective_threshold:.6g}; numerical_floor={numerical_floor:.6g}"
        ),
    )


def apply_block_sync_policy(residual: float, policy: BlockSyncPolicy) -> str:
    value = float(residual)
    if policy.uncertain_band > 0.0 and abs(value - policy.threshold) <= policy.uncertain_band:
        return "uncertain"
    return "accept" if value <= policy.threshold else "reject"


def connection_residual_value(result_or_residual: GlobalBlockSyncResult | float) -> float:
    if isinstance(result_or_residual, GlobalBlockSyncResult):
        return float(result_or_residual.connection_residual)
    return float(result_or_residual)


def accepted_sync_from_calibration(
    result_or_residual: GlobalBlockSyncResult | float,
    calibration: BlockSyncCalibration,
) -> bool:
    """Return whether a residual is accepted by a calibrated threshold."""

    return bool(connection_residual_value(result_or_residual) <= calibration.threshold)


def classify_sync_evidence(
    *,
    observed_scalar_projective_candidate: bool,
    observed_centrality_score: float,
    projected_cycle_score: float,
    connection_residual: float,
    calibration: BlockSyncCalibration,
    centrality_tolerance: float = 1e-6,
) -> str:
    """Label block-sync evidence while guarding against projection traps."""

    accepted = accepted_sync_from_calibration(connection_residual, calibration)
    if observed_scalar_projective_candidate:
        return "observed_scalar_projective_candidate"
    if projected_cycle_score <= centrality_tolerance and accepted:
        return "global_gauge_consistent"
    if projected_cycle_score <= centrality_tolerance and not accepted:
        return "projected_cycle_only_connection_large"
    if observed_centrality_score > centrality_tolerance:
        return "noncentral_block_holonomy"
    return "diagnostic_only_no_projective_claim"
