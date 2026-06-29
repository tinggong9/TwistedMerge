"""Descended and rank-lifted model merging routines."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .alignment import (
    align_mu2_weights,
    align_u1_weights,
    optimal_phase,
    project_mu2_weight,
    project_u1_weight,
    rotate_weight,
    wrap_angle,
)
from .metrics import binary_accuracy
from .twisted_merge_algorithm import TwistedMerge, TwistedMergeConfig, TwistedMergeResult


@dataclass(frozen=True)
class MergeResult:
    node_weights: np.ndarray
    global_weight: np.ndarray | None
    branch_weights: np.ndarray
    assignments: np.ndarray


def descended_mu2_merge(local_weights: np.ndarray, gauges_hat: np.ndarray) -> MergeResult:
    aligned = align_mu2_weights(local_weights, gauges_hat)
    global_weight = aligned.mean(axis=0)
    node_weights = np.stack([project_mu2_weight(global_weight, g) for g in gauges_hat], axis=0)
    return MergeResult(
        node_weights=node_weights,
        global_weight=global_weight,
        branch_weights=global_weight.reshape(1, -1),
        assignments=np.zeros(local_weights.shape[0], dtype=int),
    )


def _principal_direction(weights: np.ndarray) -> np.ndarray:
    centered = weights.copy()
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    direction = vh[0]
    if np.sum(weights[0] * direction) < 0:
        direction *= -1
    return direction


def rank_lift_mu2_merge(
    local_weights: np.ndarray,
    gauges_hat: np.ndarray,
    x_val: list[np.ndarray],
    y_val: list[np.ndarray],
) -> MergeResult:
    aligned = align_mu2_weights(local_weights, gauges_hat)
    direction = _principal_direction(aligned)
    signs = np.where(aligned @ direction >= 0.0, 1.0, -1.0)
    branches = []
    for sign in [1.0, -1.0]:
        cluster = aligned[signs == sign]
        if len(cluster) == 0:
            branches.append(sign * direction)
        else:
            branches.append(cluster.mean(axis=0))
    branch_weights = np.stack(branches, axis=0)
    node_weights = []
    assignments = []
    for i, gauge in enumerate(gauges_hat):
        candidates = np.stack([project_mu2_weight(branch, gauge) for branch in branch_weights], axis=0)
        scores = [binary_accuracy(x_val[i], y_val[i], candidate) for candidate in candidates]
        assignment = int(np.argmax(scores))
        assignments.append(assignment)
        node_weights.append(candidates[assignment])
    return MergeResult(
        node_weights=np.stack(node_weights, axis=0),
        global_weight=None,
        branch_weights=branch_weights,
        assignments=np.asarray(assignments, dtype=int),
    )


def descended_u1_merge(local_weights: np.ndarray, phases_hat: np.ndarray) -> MergeResult:
    aligned = align_u1_weights(local_weights, phases_hat)
    global_weight = aligned.mean(axis=0)
    node_weights = np.stack([project_u1_weight(global_weight, phase) for phase in phases_hat], axis=0)
    return MergeResult(
        node_weights=node_weights,
        global_weight=global_weight,
        branch_weights=global_weight.reshape(1, -1),
        assignments=np.zeros(local_weights.shape[0], dtype=int),
    )


def _phase_branch_centers(phases: np.ndarray, n_branches: int) -> np.ndarray:
    if n_branches <= 1:
        return np.array([0.0])
    grid = np.linspace(-np.pi, np.pi, n_branches, endpoint=False)
    centers = []
    for center in grid:
        dist = np.abs(wrap_angle(phases - center))
        if np.any(dist <= np.pi / n_branches):
            angles = phases[dist <= np.pi / n_branches]
            centers.append(float(np.angle(np.mean(np.exp(1j * angles)))))
        else:
            centers.append(float(center))
    return np.asarray(centers, dtype=float)


def rank_lift_u1_merge(
    local_weights: np.ndarray,
    phases_hat: np.ndarray,
    x_val: list[np.ndarray],
    y_val: list[np.ndarray],
    n_branches: int = 4,
) -> MergeResult:
    aligned = align_u1_weights(local_weights, phases_hat)
    reference = _principal_direction(aligned)
    residual_phases = np.asarray([optimal_phase(reference, w) for w in aligned])
    centers = _phase_branch_centers(residual_phases, n_branches)
    branch_weights = []
    for center in centers:
        distances = np.abs(wrap_angle(residual_phases - center))
        cluster = aligned[distances == distances.min()] if n_branches >= len(aligned) else aligned[distances <= np.pi / max(n_branches, 1)]
        if len(cluster) == 0:
            branch_weights.append(rotate_weight(reference, center))
        else:
            branch_weights.append(cluster.mean(axis=0))
    branch_weights = np.stack(branch_weights, axis=0)
    node_weights = []
    assignments = []
    for i, phase in enumerate(phases_hat):
        candidates = np.stack([project_u1_weight(branch, phase) for branch in branch_weights], axis=0)
        scores = [binary_accuracy(x_val[i], y_val[i], candidate) for candidate in candidates]
        assignment = int(np.argmax(scores))
        assignments.append(assignment)
        node_weights.append(candidates[assignment])
    return MergeResult(
        node_weights=np.stack(node_weights, axis=0),
        global_weight=None,
        branch_weights=branch_weights,
        assignments=np.asarray(assignments, dtype=int),
    )
