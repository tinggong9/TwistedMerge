"""Conservative transition, residual, Hodge, and lift-dispatch primitives.

The module deliberately separates structural diagnostics from prediction.  In
particular, a persistent numerical residual is not labelled an H^2 obstruction
unless callers independently establish closure and coefficient assumptions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class TransitionEstimate:
    matrix: np.ndarray
    family: str
    calibration_error: float
    heldout_error: float
    rank: int
    condition_number: float


@dataclass(frozen=True)
class CycleDiagnostics:
    residual: np.ndarray
    distance_to_identity: float
    distance_to_real_center: float
    eigenvalues: np.ndarray
    singular_values_h_minus_i: np.ndarray
    effective_residual_rank: int


@dataclass(frozen=True)
class HodgeDecomposition:
    gradient: np.ndarray
    harmonic: np.ndarray
    coexact: np.ndarray
    reconstruction_error: float
    gradient_harmonic_inner: float
    gradient_coexact_inner: float
    harmonic_coexact_inner: float


@dataclass(frozen=True)
class ConfidenceGateResult:
    activate: bool
    mean_gain: float
    standard_error: float
    lower_bound: float
    threshold: float


@dataclass(frozen=True)
class CorrectionDecision:
    mode: str
    activate_lift: bool
    reason: str
    branch_count: int


def _relative_error(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.linalg.norm(prediction - target) / max(np.linalg.norm(target), 1e-12))


def _permutation_transition(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source_centered = source - source.mean(axis=0, keepdims=True)
    target_centered = target - target.mean(axis=0, keepdims=True)
    source_scale = np.linalg.norm(source_centered, axis=0, keepdims=True).clip(1e-12)
    target_scale = np.linalg.norm(target_centered, axis=0, keepdims=True).clip(1e-12)
    correlation = (source_centered / source_scale).T @ (target_centered / target_scale)
    source_idx, target_idx = linear_sum_assignment(-np.abs(correlation))
    matrix = np.zeros((target.shape[1], source.shape[1]), dtype=float)
    matrix[target_idx, source_idx] = 1.0
    return matrix


def estimate_transition(
    source: np.ndarray,
    target: np.ndarray,
    *,
    family: str = "orthogonal",
    regularization: float = 1e-6,
    rank: int | None = None,
    heldout: tuple[np.ndarray, np.ndarray] | None = None,
    blocks: Sequence[Sequence[int]] | None = None,
) -> TransitionEstimate:
    """Fit ``g`` so row activations satisfy ``source @ g.T ~= target``."""

    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    if source.ndim != 2 or target.ndim != 2 or source.shape != target.shape:
        raise ValueError("source and target must be equally shaped 2D activation arrays")
    d = source.shape[1]
    if family == "permutation":
        matrix = _permutation_transition(source, target)
    elif family == "positive_monomial":
        permutation = _permutation_transition(source, target)
        permuted = source @ permutation.T
        scale = np.sum(permuted * target, axis=0) / (np.sum(permuted * permuted, axis=0) + regularization)
        matrix = np.diag(np.maximum(scale, regularization)) @ permutation
    elif family == "orthogonal":
        u, _, vt = np.linalg.svd(source.T @ target, full_matrices=False)
        matrix = (u @ vt).T
    elif family in {"whitened_linear", "cca"}:
        gram = source.T @ source + regularization * np.eye(d)
        right_map = np.linalg.solve(gram, source.T @ target)
        matrix = right_map.T
    elif family in {"low_rank", "lora_basis"}:
        gram = source.T @ source + regularization * np.eye(d)
        full = np.linalg.solve(gram, source.T @ target).T
        u, singular, vt = np.linalg.svd(full, full_matrices=False)
        chosen = rank if rank is not None else max(1, min(d, np.linalg.matrix_rank(full)))
        matrix = (u[:, :chosen] * singular[:chosen]) @ vt[:chosen]
    elif family == "block_orthogonal":
        if not blocks:
            raise ValueError("block_orthogonal requires blocks")
        matrix = np.zeros((d, d), dtype=float)
        for block in blocks:
            idx = np.asarray(block, dtype=int)
            u, _, vt = np.linalg.svd(source[:, idx].T @ target[:, idx], full_matrices=False)
            matrix[np.ix_(idx, idx)] = (u @ vt).T
    else:
        raise ValueError(f"unsupported transition family: {family}")
    calibration_error = _relative_error(source @ matrix.T, target)
    heldout_error = float("nan")
    if heldout is not None:
        heldout_error = _relative_error(np.asarray(heldout[0]) @ matrix.T, np.asarray(heldout[1]))
    singular = np.linalg.svd(matrix, compute_uv=False)
    nonzero = singular[singular > 1e-12]
    condition = float(nonzero.max() / nonzero.min()) if nonzero.size else float("inf")
    return TransitionEstimate(
        matrix=matrix,
        family=family,
        calibration_error=calibration_error,
        heldout_error=heldout_error,
        rank=int(np.linalg.matrix_rank(matrix)),
        condition_number=condition,
    )


def inverse_consistency(first: np.ndarray, reverse: np.ndarray) -> float:
    identity = np.eye(first.shape[1])
    return float(np.linalg.norm(reverse @ first - identity, ord="fro") / np.sqrt(identity.size))


def cycle_residual(*transitions: np.ndarray, tolerance: float = 1e-8) -> CycleDiagnostics:
    if len(transitions) < 2:
        raise ValueError("a cycle requires at least two transitions")
    residual = np.eye(transitions[0].shape[0])
    for transition in transitions:
        residual = residual @ np.asarray(transition, dtype=float)
    identity = np.eye(residual.shape[0])
    scalar = np.trace(residual) / residual.shape[0]
    delta = residual - identity
    singular = np.linalg.svd(delta, compute_uv=False)
    cutoff = tolerance * max(float(singular[0]) if singular.size else 0.0, 1.0)
    return CycleDiagnostics(
        residual=residual,
        distance_to_identity=float(np.linalg.norm(delta, ord="fro")),
        distance_to_real_center=float(np.linalg.norm(residual - scalar * identity, ord="fro")),
        eigenvalues=np.linalg.eigvals(residual),
        singular_values_h_minus_i=singular,
        effective_residual_rank=int(np.sum(singular > cutoff)),
    )


def _weighted_projection(basis: np.ndarray, values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    if basis.shape[1] == 0:
        return np.zeros_like(values)
    gram = basis.T @ weights @ basis
    coefficient = np.linalg.pinv(gram, rcond=1e-12) @ basis.T @ weights @ values
    return basis @ coefficient


def weighted_hodge_decomposition(
    vertex_edge_incidence: np.ndarray,
    edge_face_incidence: np.ndarray,
    cochain: np.ndarray,
    *,
    edge_weights: Iterable[float] | None = None,
    face_weights: Iterable[float] | None = None,
) -> HodgeDecomposition:
    """Decompose edge cochains on a weighted 2-complex.

    ``vertex_edge_incidence @ edge_face_incidence`` must vanish.  The coexact
    basis uses the weighted adjoint, preserving weighted orthogonality.
    """

    b1 = np.asarray(vertex_edge_incidence, dtype=float)
    b2 = np.asarray(edge_face_incidence, dtype=float)
    values = np.asarray(cochain, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    if b1.shape[1] != b2.shape[0] or b2.shape[0] != values.shape[0]:
        raise ValueError("incidence and cochain dimensions do not agree")
    if np.linalg.norm(b1 @ b2) > 1e-9:
        raise ValueError("incidence matrices violate boundary-of-boundary equals zero")
    ew = np.ones(b1.shape[1]) if edge_weights is None else np.asarray(list(edge_weights), dtype=float)
    fw = np.ones(b2.shape[1]) if face_weights is None else np.asarray(list(face_weights), dtype=float)
    if np.any(ew <= 0) or np.any(fw <= 0):
        raise ValueError("weights must be positive")
    weight = np.diag(ew)
    gradient_basis = b1.T
    coexact_basis = np.diag(1.0 / ew) @ b2 @ np.diag(fw)
    gradient = _weighted_projection(gradient_basis, values, weight)
    coexact = _weighted_projection(coexact_basis, values, weight)
    harmonic = values - gradient - coexact

    def inner(left: np.ndarray, right: np.ndarray) -> float:
        return float(np.sum(left * (weight @ right)))

    reconstructed = gradient + harmonic + coexact
    return HodgeDecomposition(
        gradient=gradient.squeeze(),
        harmonic=harmonic.squeeze(),
        coexact=coexact.squeeze(),
        reconstruction_error=float(np.linalg.norm(reconstructed - values)),
        gradient_harmonic_inner=inner(gradient, harmonic),
        gradient_coexact_inner=inner(gradient, coexact),
        harmonic_coexact_inner=inner(harmonic, coexact),
    )


def conservative_confidence_gate(
    paired_gains: Iterable[float], *, z_alpha: float = 1.96, threshold: float = 0.0
) -> ConfidenceGateResult:
    gains = np.asarray(list(paired_gains), dtype=float)
    if gains.size == 0:
        raise ValueError("at least one paired gain is required")
    mean = float(gains.mean())
    se = float(gains.std(ddof=1) / np.sqrt(gains.size)) if gains.size > 1 else float("inf")
    lower = mean - z_alpha * se
    return ConfidenceGateResult(lower > threshold, mean, se, lower, threshold)


def dispatch_correction(
    *,
    residual_norm: float,
    harmonic_norm: float,
    certified_central_order: int | None = None,
    certified_noncentral_generators: int = 0,
    requested_rank: int | None = None,
    proved_rank_threshold: int | None = None,
    gate: ConfidenceGateResult | None = None,
    tolerance: float = 1e-6,
) -> CorrectionDecision:
    if residual_norm <= tolerance or harmonic_norm <= tolerance:
        return CorrectionDecision("strict_synchronization", False, "residual is removable", 1)
    if gate is None or not gate.activate:
        return CorrectionDecision("ordinary_validated_family", False, "lift confidence gate did not pass", 1)
    if certified_central_order is not None:
        threshold = proved_rank_threshold or certified_central_order
        if requested_rank is None or requested_rank < threshold:
            return CorrectionDecision("ordinary_validated_family", False, "requested rank is below certified threshold", 1)
        return CorrectionDecision("central_character_lift", True, "central certificate and gain gate passed", certified_central_order)
    if certified_noncentral_generators > 0:
        return CorrectionDecision(
            "noncentral_representation_lift", True, "generator and gain certificates passed", certified_noncentral_generators + 1
        )
    return CorrectionDecision("ordinary_validated_family", False, "persistent residual is uncertified", 1)
