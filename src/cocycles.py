"""Cocycle generation, obstruction scores, and synchronization methods."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from .alignment import sign_from_scores, wrap_angle
from .simplicial_mu2 import (
    compute_triangle_defects,
    is_coboundary_mu2,
    obstruction_score,
    try_global_gauge_synchronization,
    twisted_sheaf_prediction,
)


Edge = tuple[int, int]


@dataclass(frozen=True)
class Mu2Cocycle:
    n_nodes: int
    edges: list[Edge]
    signs: dict[Edge, int]
    true_gauges: np.ndarray
    flipped_edges: list[Edge]


@dataclass(frozen=True)
class U1Cocycle:
    n_nodes: int
    edges: list[Edge]
    phases: dict[Edge, float]
    true_phases: np.ndarray
    noise_std: float


def complete_graph_edges(n_nodes: int) -> list[Edge]:
    return [(i, j) for i in range(n_nodes) for j in range(i + 1, n_nodes)]


def canonical_edge(i: int, j: int) -> tuple[Edge, int]:
    if i < j:
        return (i, j), 1
    return (j, i), -1


def oriented_mu2(signs: dict[Edge, int], i: int, j: int) -> int:
    edge, _ = canonical_edge(i, j)
    return int(signs[edge])


def oriented_u1(phases: dict[Edge, float], i: int, j: int) -> float:
    edge, orientation = canonical_edge(i, j)
    return float(orientation * phases[edge])


def sample_mu2_cocycle(
    n_nodes: int,
    flip_prob: float,
    rng: np.random.Generator,
    edges: list[Edge] | None = None,
) -> Mu2Cocycle:
    """Sample sign edge observations from hidden node gauges plus edge flips."""
    edges = complete_graph_edges(n_nodes) if edges is None else edges
    true_gauges = rng.choice(np.array([-1, 1], dtype=int), size=n_nodes)
    true_gauges[0] = 1
    signs: dict[Edge, int] = {}
    flipped_edges: list[Edge] = []
    for i, j in edges:
        sign = int(true_gauges[i] * true_gauges[j])
        if rng.random() < flip_prob:
            sign *= -1
            flipped_edges.append((i, j))
        signs[(i, j)] = sign
    return Mu2Cocycle(n_nodes, edges, signs, true_gauges.astype(float), flipped_edges)


def sample_u1_cocycle(
    n_nodes: int,
    noise_std: float,
    rng: np.random.Generator,
    edges: list[Edge] | None = None,
) -> U1Cocycle:
    """Sample phase edge observations from hidden node phases plus angular noise."""
    edges = complete_graph_edges(n_nodes) if edges is None else edges
    true_phases = rng.uniform(-np.pi, np.pi, size=n_nodes)
    true_phases[0] = 0.0
    phases: dict[Edge, float] = {}
    for i, j in edges:
        clean = true_phases[j] - true_phases[i]
        phases[(i, j)] = float(wrap_angle(clean + rng.normal(0.0, noise_std)))
    return U1Cocycle(n_nodes, edges, phases, true_phases, noise_std)


def mu2_triangle_obstruction(n_nodes: int, signs: dict[Edge, int]) -> dict[str, float]:
    values = []
    for i, j, k in combinations(range(n_nodes), 3):
        holonomy = oriented_mu2(signs, i, j) * oriented_mu2(signs, j, k) * oriented_mu2(signs, k, i)
        values.append(0.0 if holonomy == 1 else 1.0)
    arr = np.asarray(values, dtype=float)
    return {
        "obstruction_score": float(arr.mean()) if arr.size else 0.0,
        "max_obstruction": float(arr.max()) if arr.size else 0.0,
        "n_triangles": float(arr.size),
    }


def u1_triangle_obstruction(n_nodes: int, phases: dict[Edge, float]) -> dict[str, float]:
    values = []
    for i, j, k in combinations(range(n_nodes), 3):
        holonomy = oriented_u1(phases, i, j) + oriented_u1(phases, j, k) + oriented_u1(phases, k, i)
        values.append(abs(float(wrap_angle(holonomy))) / np.pi)
    arr = np.asarray(values, dtype=float)
    return {
        "obstruction_score": float(arr.mean()) if arr.size else 0.0,
        "max_obstruction": float(arr.max()) if arr.size else 0.0,
        "n_triangles": float(arr.size),
    }


def estimate_mu2_gauges_spectral(n_nodes: int, signs: dict[Edge, int]) -> np.ndarray:
    """Estimate node sign gauges by the leading eigenvector of the sign matrix."""
    matrix = np.eye(n_nodes)
    for (i, j), sign in signs.items():
        matrix[i, j] = sign
        matrix[j, i] = sign
    vals, vecs = np.linalg.eigh(matrix)
    gauges = sign_from_scores(vecs[:, int(np.argmax(vals))])
    return gauges.astype(float)


def estimate_u1_phases_spectral(n_nodes: int, phases: dict[Edge, float]) -> np.ndarray:
    """Estimate node phases by angular synchronization."""
    matrix = np.eye(n_nodes, dtype=np.complex128)
    for (i, j), phase in phases.items():
        # Observations use phase_ij = theta_j - theta_i, so the Hermitian
        # synchronization matrix should contain exp(i * (theta_i - theta_j)).
        value = np.exp(-1j * phase)
        matrix[i, j] = value
        matrix[j, i] = np.conjugate(value)
    vals, vecs = np.linalg.eigh(matrix)
    leading = vecs[:, int(np.argmax(vals))]
    phases_hat = np.angle(leading)
    phases_hat = wrap_angle(phases_hat - phases_hat[0])
    return np.asarray(phases_hat, dtype=float)


def mu2_edge_agreement(gauges: np.ndarray, signs: dict[Edge, int]) -> float:
    hits = [float(gauges[i] * gauges[j] == sign) for (i, j), sign in signs.items()]
    return float(np.mean(hits)) if hits else 1.0


def u1_edge_residual(phases_hat: np.ndarray, phases: dict[Edge, float]) -> float:
    residuals = []
    for (i, j), phase in phases.items():
        residuals.append(abs(float(wrap_angle(phases_hat[j] - phases_hat[i] - phase))) / np.pi)
    return float(np.mean(residuals)) if residuals else 0.0
