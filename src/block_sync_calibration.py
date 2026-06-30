"""Calibration helpers for accepting global block synchronization."""

from __future__ import annotations

from dataclasses import dataclass
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
