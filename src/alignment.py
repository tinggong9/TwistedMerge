"""Alignment helpers for sign and phase-valued model transformations."""

from __future__ import annotations

import numpy as np


def wrap_angle(angle: np.ndarray | float) -> np.ndarray | float:
    """Wrap angles to the interval [-pi, pi)."""
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


def normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, eps)


def apply_sign(weights: np.ndarray, signs: np.ndarray) -> np.ndarray:
    return weights * signs.reshape(-1, 1)


def rotate_weight(weight: np.ndarray, angle: float) -> np.ndarray:
    """Rotate a flat real vector in independent 2D blocks."""
    if weight.shape[-1] % 2 != 0:
        raise ValueError("U(1) rotation requires an even feature dimension.")
    pairs = weight.reshape(-1, 2)
    c = np.cos(angle)
    s = np.sin(angle)
    rot = np.array([[c, -s], [s, c]], dtype=weight.dtype)
    return (pairs @ rot.T).reshape(weight.shape)


def rotate_weights(weights: np.ndarray, angles: np.ndarray) -> np.ndarray:
    return np.stack([rotate_weight(w, float(a)) for w, a in zip(weights, angles)], axis=0)


def align_mu2_weights(weights: np.ndarray, gauges: np.ndarray) -> np.ndarray:
    """Move local sign-gauged weights into a common gauge estimate."""
    return apply_sign(weights, gauges)


def project_mu2_weight(global_weight: np.ndarray, gauge: float) -> np.ndarray:
    return float(gauge) * global_weight


def align_u1_weights(weights: np.ndarray, phases: np.ndarray) -> np.ndarray:
    """Move local phase-gauged weights into a common gauge estimate."""
    return rotate_weights(weights, -phases)


def project_u1_weight(global_weight: np.ndarray, phase: float) -> np.ndarray:
    return rotate_weight(global_weight, float(phase))


def optimal_phase(reference: np.ndarray, target: np.ndarray) -> float:
    """Least-squares phase rotating reference toward target over 2D blocks."""
    if reference.shape != target.shape or reference.shape[-1] % 2 != 0:
        raise ValueError("reference and target must have equal even length.")
    ref = reference.reshape(-1, 2)
    tgt = target.reshape(-1, 2)
    # Maximize sum dot(R(a) ref_k, tgt_k).
    a = np.sum(ref[:, 0] * tgt[:, 0] + ref[:, 1] * tgt[:, 1])
    b = np.sum(ref[:, 0] * tgt[:, 1] - ref[:, 1] * tgt[:, 0])
    return float(np.arctan2(b, a))


def sign_from_scores(scores: np.ndarray) -> np.ndarray:
    signs = np.where(scores >= 0.0, 1.0, -1.0)
    if signs[0] < 0:
        signs *= -1.0
    return signs
