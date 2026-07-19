"""Conservative central-projective diagnostics for Application B."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from math import gcd

import numpy as np
import torch

from src.holonomy_application_transitions import loop_product


@dataclass(frozen=True)
class CentralityDiagnostics:
    scalar_real: float
    scalar_imag: float
    centrality_residual: float
    normalized_centrality_residual: float
    eigenvalue_dispersion: float


@dataclass(frozen=True)
class RootDiagnostics:
    order: int
    exponent: int
    residual: float
    margin: float
    confidence: float


def scalar_centrality(matrix: torch.Tensor) -> CentralityDiagnostics:
    values = matrix.detach().cpu().double().numpy().astype(np.complex128)
    dimension = values.shape[0]
    scalar = complex(np.trace(values) / max(dimension, 1))
    identity = np.eye(dimension, dtype=np.complex128)
    numerator = float(np.linalg.norm(values - scalar * identity, ord="fro"))
    matrix_norm = max(float(np.linalg.norm(values, ord="fro")), 1e-12)
    scalar_norm = max(float(np.linalg.norm(scalar * identity, ord="fro")), 1e-12)
    eigenvalues = np.linalg.eigvals(values)
    dispersion = float(np.mean(np.abs(eigenvalues - scalar)))
    return CentralityDiagnostics(
        scalar_real=float(scalar.real),
        scalar_imag=float(scalar.imag),
        centrality_residual=numerator / matrix_norm,
        normalized_centrality_residual=numerator / scalar_norm,
        eigenvalue_dispersion=dispersion,
    )


def reduce_root(order: int, exponent: int) -> tuple[int, int]:
    exponent %= order
    if exponent == 0:
        return 1, 0
    divisor = gcd(order, exponent)
    return order // divisor, (exponent // divisor) % (order // divisor)


def nearest_root(matrix: torch.Tensor, max_order: int = 6) -> RootDiagnostics:
    values = matrix.detach().cpu().double().numpy().astype(np.complex128)
    dimension = values.shape[0]
    denominator = max(float(np.linalg.norm(values, ord="fro")), 1e-12)
    candidates: dict[tuple[int, int], float] = {}
    for order in range(1, max_order + 1):
        for exponent in range(order):
            reduced = reduce_root(order, exponent)
            root = np.exp(2j * np.pi * reduced[1] / reduced[0])
            residual = float(
                np.linalg.norm(values - root * np.eye(dimension), ord="fro") / denominator
            )
            candidates[reduced] = min(candidates.get(reduced, float("inf")), residual)
    ordered = sorted(candidates.items(), key=lambda item: (item[1], item[0]))
    (order, exponent), residual = ordered[0]
    margin = float(ordered[1][1] - residual) if len(ordered) > 1 else float("inf")
    confidence = float(margin / max(margin + residual, 1e-12)) if np.isfinite(margin) else 1.0
    return RootDiagnostics(order, exponent, float(residual), margin, confidence)


def triangle_defect(
    transitions: Mapping[tuple[int, int], torch.Tensor], triangle: Sequence[int]
) -> torch.Tensor:
    if len(triangle) != 3:
        raise ValueError("triangle must contain three distinct vertices")
    first, second, third = map(int, triangle)
    return loop_product(transitions, (first, second, third, first))


def normalized_commutator_residual(left: torch.Tensor, right: torch.Tensor) -> float:
    numerator = torch.linalg.norm(left @ right - right @ left)
    denominator = (torch.linalg.norm(left) * torch.linalg.norm(right)).clamp_min(1e-12)
    return float(numerator / denominator)


def wrap_phase(value: float) -> float:
    return float((value + np.pi) % (2 * np.pi) - np.pi)


def scalar_phase(diagnostics: CentralityDiagnostics) -> float:
    return float(np.angle(complex(diagnostics.scalar_real, diagnostics.scalar_imag)))


def tetrahedral_cocycle_rows(
    triangle_phases: Mapping[tuple[int, int, int], float], vertices: int
) -> list[dict[str, float | str]]:
    rows = []
    for i, j, k, l in combinations(range(vertices), 4):
        residual = wrap_phase(
            triangle_phases[(j, k, l)]
            - triangle_phases[(i, k, l)]
            + triangle_phases[(i, j, l)]
            - triangle_phases[(i, j, k)]
        )
        rows.append(
            {
                "tetrahedron": f"{i}-{j}-{k}-{l}",
                "cocycle_phase_residual": residual,
                "normalized_cocycle_residual": abs(residual) / np.pi,
            }
        )
    return rows


def coboundary_fit(
    triangle_phases: Mapping[tuple[int, int, int], float], vertices: int
) -> tuple[float, dict[str, float]]:
    edges = list(combinations(range(vertices), 2))
    edge_index = {edge: index for index, edge in enumerate(edges)}
    triangles = list(combinations(range(vertices), 3))
    design = np.zeros((len(triangles), len(edges)), dtype=np.float64)
    target = np.zeros(len(triangles), dtype=np.float64)

    def assign(row: int, source: int, target_vertex: int, coefficient: float) -> None:
        edge = (min(source, target_vertex), max(source, target_vertex))
        sign = 1.0 if source < target_vertex else -1.0
        design[row, edge_index[edge]] += coefficient * sign

    for row, (i, j, k) in enumerate(triangles):
        assign(row, i, j, 1.0)
        assign(row, j, k, 1.0)
        assign(row, k, i, 1.0)
        target[row] = triangle_phases[(i, j, k)]
    solution, *_ = np.linalg.lstsq(design, target, rcond=None)
    residual = np.asarray([wrap_phase(value) for value in design @ solution - target])
    rephasings = {f"{left}-{right}": float(solution[index]) for index, (left, right) in enumerate(edges)}
    return float(np.sqrt(np.mean(residual**2)) / np.pi), rephasings


def gauge_transform_connection(
    transitions: Mapping[tuple[int, int], torch.Tensor], gauges: Sequence[torch.Tensor]
) -> dict[tuple[int, int], torch.Tensor]:
    return {
        (source, target): gauges[target] @ transition @ torch.linalg.pinv(gauges[source])
        for (source, target), transition in transitions.items()
    }
