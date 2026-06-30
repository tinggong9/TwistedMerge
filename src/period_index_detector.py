"""Conservative detector for central period-index generator data."""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class PeriodIndexDetection:
    detected: bool
    period: int | None
    index: int | None
    independent_pair_count: int | None
    exponent_matrix: list[list[int]] | None
    centrality_score: float
    phase_residual: float
    candidate_rank: int
    period_divides_rank: bool | None
    index_divides_rank: bool | None
    decision: str
    notes: list[str]


def _relative_residual(left: np.ndarray, right: np.ndarray) -> float:
    denom = max(float(np.linalg.norm(right, ord="fro")), 1e-12)
    return float(np.linalg.norm(left - right, ord="fro") / denom)


def _root(order: int, exponent: int) -> complex:
    return complex(np.exp(2j * np.pi * (exponent % order) / order))


def _commutator(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return A @ B @ np.linalg.inv(A) @ np.linalg.inv(B)


def _scalar_centrality(matrix: np.ndarray) -> tuple[float, complex | None]:
    width = matrix.shape[0]
    scalar = complex(np.trace(matrix) / max(width, 1))
    if abs(scalar) <= 1e-12:
        denom = max(float(np.linalg.norm(np.eye(width), ord="fro")), 1e-12)
        return float(np.linalg.norm(matrix, ord="fro") / denom), None
    target = scalar * np.eye(width, dtype=complex)
    denom = max(float(np.linalg.norm(target, ord="fro")), 1e-12)
    return float(np.linalg.norm(matrix - target, ord="fro") / denom), scalar / abs(scalar)


def _fit_exponents(
    commutators: list[np.ndarray],
    max_root_order: int,
) -> tuple[int | None, list[int] | None, float]:
    best: tuple[float, int, list[int]] | None = None
    for period in range(2, max_root_order + 1):
        exponents: list[int] = []
        residuals: list[float] = []
        for matrix in commutators:
            width = matrix.shape[0]
            candidates = [
                (
                    _relative_residual(matrix, _root(period, exponent) * np.eye(width, dtype=complex)),
                    exponent,
                )
                for exponent in range(period)
            ]
            residual, exponent = min(candidates, key=lambda item: (item[0], item[1]))
            residuals.append(residual)
            exponents.append(int(exponent))
        if not any(exponent % period != 0 for exponent in exponents):
            continue
        score = float(max(residuals, default=0.0))
        if best is None or score < best[0] - 1e-12 or (abs(score - best[0]) <= 1e-12 and period < best[1]):
            best = (score, period, exponents)
    if best is None:
        return None, None, float("inf")
    return best[1], best[2], best[0]


def _build_exponent_matrix(size: int, pair_exponents: Mapping[tuple[int, int], int], period: int) -> list[list[int]]:
    matrix = [[0 for _ in range(size)] for _ in range(size)]
    for (i, j), exponent in pair_exponents.items():
        value = int(exponent % period)
        matrix[i][j] = value
        matrix[j][i] = int((-value) % period)
    return matrix


def _independent_pair_count(exponent_matrix: list[list[int]], period: int) -> int | None:
    size = len(exponent_matrix)
    if size == 0 or size % 2 != 0:
        return None
    matched: set[int] = set()
    pair_count = 0
    for row_idx, row in enumerate(exponent_matrix):
        nonzero = [col_idx for col_idx, value in enumerate(row) if col_idx != row_idx and value % period != 0]
        if len(nonzero) != 1:
            return None
        partner = nonzero[0]
        exponent = row[partner] % period
        if gcd(exponent, period) != 1:
            return None
        if row_idx in matched:
            continue
        if partner in matched:
            return None
        if exponent_matrix[partner][row_idx] % period != (-exponent) % period:
            return None
        matched.add(row_idx)
        matched.add(partner)
        pair_count += 1
    return pair_count if len(matched) == size and pair_count > 0 else None


def _empty_result(candidate_rank: int, decision: str, notes: list[str]) -> PeriodIndexDetection:
    return PeriodIndexDetection(
        detected=False,
        period=None,
        index=None,
        independent_pair_count=None,
        exponent_matrix=None,
        centrality_score=float("inf"),
        phase_residual=float("inf"),
        candidate_rank=int(candidate_rank),
        period_divides_rank=None,
        index_divides_rank=None,
        decision=decision,
        notes=notes,
    )


def detect_period_index_structure(
    generators: Mapping[str, np.ndarray],
    candidate_rank: int,
    max_root_order: int = 12,
    centrality_tolerance: float = 1e-8,
    phase_tolerance: float = 1e-8,
) -> PeriodIndexDetection:
    """Detect a controlled central/projective period-index pattern.

    The detector recognizes the independent-pair finite Heisenberg form and is
    deliberately conservative otherwise: scalar commutators with an
    unrecognized invariant return ``central_projective_index_unknown``.
    """

    names = list(generators)
    matrices = [np.asarray(generators[name], dtype=complex) for name in names]
    notes: list[str] = []
    rank = int(candidate_rank)
    if len(matrices) < 2:
        return _empty_result(rank, "not_central_projective", ["at least two generators are required"])
    first_shape = matrices[0].shape
    if len(first_shape) != 2 or first_shape[0] != first_shape[1]:
        return _empty_result(rank, "not_central_projective", ["generators must be square matrices"])
    if any(matrix.shape != first_shape for matrix in matrices):
        return _empty_result(rank, "not_central_projective", ["all generators must have the same square shape"])

    pair_keys: list[tuple[int, int]] = []
    commutators: list[np.ndarray] = []
    try:
        for i in range(len(matrices)):
            for j in range(i + 1, len(matrices)):
                pair_keys.append((i, j))
                commutators.append(_commutator(matrices[i], matrices[j]))
    except np.linalg.LinAlgError:
        return _empty_result(rank, "not_central_projective", ["a generator was singular"])

    centralities = [_scalar_centrality(matrix)[0] for matrix in commutators]
    centrality_score = float(max(centralities, default=0.0))
    if centrality_score > centrality_tolerance:
        return PeriodIndexDetection(
            detected=False,
            period=None,
            index=None,
            independent_pair_count=None,
            exponent_matrix=None,
            centrality_score=centrality_score,
            phase_residual=float("inf"),
            candidate_rank=rank,
            period_divides_rank=None,
            index_divides_rank=None,
            decision="not_central_projective",
            notes=["at least one generator commutator is noncentral"],
        )

    period, exponents, phase_residual = _fit_exponents(commutators, max_root_order)
    if period is None or exponents is None or phase_residual > phase_tolerance:
        return PeriodIndexDetection(
            detected=False,
            period=None,
            index=None,
            independent_pair_count=None,
            exponent_matrix=None,
            centrality_score=centrality_score,
            phase_residual=phase_residual,
            candidate_rank=rank,
            period_divides_rank=None,
            index_divides_rank=None,
            decision="not_central_projective",
            notes=["central commutators did not fit a nontrivial finite root-of-unity period"],
        )

    pair_exponents = {pair: exponent for pair, exponent in zip(pair_keys, exponents, strict=True)}
    exponent_matrix = _build_exponent_matrix(len(matrices), pair_exponents, period)
    pair_count = _independent_pair_count(exponent_matrix, period)
    period_divides = rank > 0 and rank % period == 0
    index: int | None = None
    index_divides: bool | None = None
    decision: str
    if pair_count is None:
        decision = "central_projective_index_unknown"
        notes.append("scalar finite commutators were detected, but the independent-pair invariant was not recognized")
    else:
        index = period**pair_count
        index_divides = rank > 0 and rank % index == 0
        notes.append(
            f"recognized {pair_count} independent Heisenberg pair(s); period={period}, index={index}"
        )
        if index_divides:
            decision = "period_index_lift_success"
        elif period_divides:
            decision = "period_divisible_index_obstructed"
            notes.append("candidate rank is divisible by the period but not by the index")
        else:
            decision = "rank_obstructed"

    return PeriodIndexDetection(
        detected=True,
        period=period,
        index=index,
        independent_pair_count=pair_count,
        exponent_matrix=exponent_matrix,
        centrality_score=centrality_score,
        phase_residual=phase_residual,
        candidate_rank=rank,
        period_divides_rank=period_divides,
        index_divides_rank=index_divides,
        decision=decision,
        notes=notes,
    )
