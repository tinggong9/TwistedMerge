"""Conservative detectors for central period-index generator data."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import gcd, isqrt, lcm
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
    detector_mode: str = "unknown"
    generator_names: tuple[str, ...] = ()
    alternating_rank: int | None = None
    radical_size: int | None = None
    quotient_size: int | None = None

    @property
    def max_centrality_score(self) -> float:
        return self.centrality_score

    @property
    def max_phase_residual(self) -> float:
        return self.phase_residual


@dataclass(frozen=True)
class CommutatorRootObservation:
    pair: tuple[str, str]
    centrality_score: float
    scalar_phase: complex | None
    root_order: int
    root_exponent: int
    phase_residual: float
    root_margin: float


@dataclass(frozen=True)
class RobustPeriodIndexDetection:
    status: str
    certified: bool
    detector_mode: str
    generator_names: tuple[str, ...]
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
    threshold_level: str | None = None
    centrality_tolerance: float | None = None
    phase_tolerance: float | None = None
    min_root_margin: float | None = None
    min_root_confidence: float | None = None
    alternating_rank: int | None = None
    radical_size: int | None = None
    quotient_size: int | None = None
    pair_observations: tuple[CommutatorRootObservation, ...] = ()
    exact_detection: PeriodIndexDetection | None = None

    @property
    def max_centrality_score(self) -> float:
        return self.centrality_score

    @property
    def max_phase_residual(self) -> float:
        return self.phase_residual


@dataclass(frozen=True)
class _RootFit:
    order: int
    exponent: int
    residual: float


@dataclass(frozen=True)
class _RobustRootFit:
    order: int
    exponent: int
    residual: float
    margin: float


def _relative_residual(left: np.ndarray, right: np.ndarray) -> float:
    denom = max(float(np.linalg.norm(right, ord="fro")), 1e-12)
    return float(np.linalg.norm(left - right, ord="fro") / denom)


def _root(order: int, exponent: int) -> complex:
    return complex(np.exp(2j * np.pi * (exponent % order) / order))


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    for candidate in range(2, isqrt(value) + 1):
        if value % candidate == 0:
            return False
    return True


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


def _phase_order(order: int, exponent: int) -> tuple[int, int]:
    value = int(exponent % order)
    if value == 0:
        return 1, 0
    divisor = gcd(order, value)
    return order // divisor, (value // divisor) % (order // divisor)


def _nearest_root(matrix: np.ndarray, max_root_order: int) -> _RootFit:
    width = matrix.shape[0]
    best: tuple[float, int, int] | None = None
    for order in range(1, max_root_order + 1):
        for exponent in range(order):
            residual = _relative_residual(matrix, _root(order, exponent) * np.eye(width, dtype=complex))
            if best is None or residual < best[0] - 1e-12 or (
                abs(residual - best[0]) <= 1e-12 and order < best[1]
            ):
                best = (residual, order, exponent)
    if best is None:
        return _RootFit(order=1, exponent=0, residual=float("inf"))
    residual, raw_order, raw_exponent = best
    order, exponent = _phase_order(raw_order, raw_exponent)
    return _RootFit(order=order, exponent=exponent, residual=float(residual))


def _nearest_root_with_margin(matrix: np.ndarray, max_root_order: int) -> _RobustRootFit:
    width = matrix.shape[0]
    by_root: dict[tuple[int, int], float] = {}
    for order in range(1, max_root_order + 1):
        for exponent in range(order):
            reduced_order, reduced_exponent = _phase_order(order, exponent)
            residual = _relative_residual(matrix, _root(reduced_order, reduced_exponent) * np.eye(width, dtype=complex))
            key = (reduced_order, reduced_exponent)
            if key not in by_root or residual < by_root[key]:
                by_root[key] = residual
    if not by_root:
        return _RobustRootFit(order=1, exponent=0, residual=float("inf"), margin=0.0)
    ordered = sorted(by_root.items(), key=lambda item: (item[1], item[0][0], item[0][1]))
    (order, exponent), best_residual = ordered[0]
    second_residual = ordered[1][1] if len(ordered) > 1 else float("inf")
    margin = float(second_residual - best_residual) if np.isfinite(second_residual) else float("inf")
    return _RobustRootFit(
        order=int(order),
        exponent=int(exponent),
        residual=float(best_residual),
        margin=margin,
    )


def _robust_result_from_detection(
    *,
    status: str,
    certified: bool,
    detection: PeriodIndexDetection | None,
    candidate_rank: int,
    generator_names: tuple[str, ...],
    observations: tuple[CommutatorRootObservation, ...],
    threshold_level: str | None,
    centrality_tolerance: float | None,
    phase_tolerance: float | None,
    min_root_margin: float | None,
    notes: list[str],
    min_root_confidence: float | None = None,
    decision: str | None = None,
) -> RobustPeriodIndexDetection:
    if detection is None:
        centrality_score = float(max((obs.centrality_score for obs in observations), default=float("inf")))
        phase_residual = float(max((obs.phase_residual for obs in observations), default=float("inf")))
        return RobustPeriodIndexDetection(
            status=status,
            certified=False,
            detector_mode="robust_commutator_matrix",
            generator_names=generator_names,
            period=None,
            index=None,
            independent_pair_count=None,
            exponent_matrix=None,
            centrality_score=centrality_score,
            phase_residual=phase_residual,
            candidate_rank=int(candidate_rank),
            period_divides_rank=None,
            index_divides_rank=None,
            decision=decision or "not_central_projective",
            notes=notes,
            threshold_level=threshold_level,
            centrality_tolerance=centrality_tolerance,
            phase_tolerance=phase_tolerance,
            min_root_margin=min_root_margin,
            min_root_confidence=min_root_confidence,
            pair_observations=observations,
        )

    effective_decision = decision or detection.decision
    return RobustPeriodIndexDetection(
        status=status,
        certified=certified,
        detector_mode="robust_commutator_matrix",
        generator_names=detection.generator_names,
        period=detection.period,
        index=detection.index,
        independent_pair_count=detection.independent_pair_count,
        exponent_matrix=detection.exponent_matrix,
        centrality_score=detection.centrality_score,
        phase_residual=detection.phase_residual,
        candidate_rank=detection.candidate_rank,
        period_divides_rank=detection.period_divides_rank,
        index_divides_rank=detection.index_divides_rank if certified else None,
        decision=effective_decision,
        notes=notes,
        threshold_level=threshold_level,
        centrality_tolerance=centrality_tolerance,
        phase_tolerance=phase_tolerance,
        min_root_margin=min_root_margin,
        min_root_confidence=min_root_confidence,
        alternating_rank=detection.alternating_rank,
        radical_size=detection.radical_size,
        quotient_size=detection.quotient_size,
        pair_observations=observations,
        exact_detection=detection,
    )


def _build_exponent_matrix(size: int, pair_exponents: Mapping[tuple[int, int], int], period: int) -> list[list[int]]:
    matrix = [[0 for _ in range(size)] for _ in range(size)]
    for (i, j), exponent in pair_exponents.items():
        value = int(exponent % period)
        matrix[i][j] = value
        matrix[j][i] = int((-value) % period)
    return matrix


def _rank_mod_prime(matrix: list[list[int]], prime: int) -> int:
    data = [[value % prime for value in row] for row in matrix]
    n_rows = len(data)
    n_cols = len(data[0]) if data else 0
    rank = 0
    for col in range(n_cols):
        pivot = None
        for row in range(rank, n_rows):
            if data[row][col] % prime != 0:
                pivot = row
                break
        if pivot is None:
            continue
        data[rank], data[pivot] = data[pivot], data[rank]
        inv = pow(data[rank][col], -1, prime)
        data[rank] = [(value * inv) % prime for value in data[rank]]
        for row in range(n_rows):
            if row == rank or data[row][col] % prime == 0:
                continue
            factor = data[row][col] % prime
            data[row] = [(data[row][idx] - factor * data[rank][idx]) % prime for idx in range(n_cols)]
        rank += 1
    return rank


def _radical_size_bruteforce(
    matrix: list[list[int]],
    period: int,
    max_bruteforce_states: int,
) -> tuple[int | None, int | None, str | None]:
    size = len(matrix)
    state_count = period**size
    if state_count > max_bruteforce_states:
        return None, state_count, f"brute-force radical skipped: {state_count} states exceeds limit {max_bruteforce_states}"
    radical_size = 0
    for vector in product(range(period), repeat=size):
        if all(sum(matrix[row][col] * vector[col] for col in range(size)) % period == 0 for row in range(size)):
            radical_size += 1
    return radical_size, state_count, None


def _integer_log_base(value: int, base: int) -> int | None:
    if value <= 0 or base <= 1:
        return None
    exponent = 0
    current = 1
    while current < value:
        current *= base
        exponent += 1
    return exponent if current == value else None


def _rank_from_radical(period: int, generator_count: int, radical_size: int) -> int | None:
    radical_exponent = _integer_log_base(radical_size, period)
    if radical_exponent is None:
        return None
    rank = generator_count - radical_exponent
    return rank if rank >= 0 else None


def _decision(
    *,
    period: int | None,
    index: int | None,
    candidate_rank: int,
    unknown: bool = False,
) -> tuple[str, bool | None, bool | None, list[str]]:
    notes: list[str] = []
    period_divides = candidate_rank > 0 and period is not None and candidate_rank % period == 0
    if unknown or period is None or index is None:
        return "central_projective_index_unknown", period_divides if period is not None else None, None, notes
    index_divides = candidate_rank > 0 and candidate_rank % index == 0
    if index_divides:
        return "period_index_lift_success", period_divides, True, notes
    if period_divides:
        notes.append("candidate rank is divisible by the period but not by the index")
        return "period_divisible_index_obstructed", period_divides, False, notes
    return "rank_obstructed", period_divides, False, notes


def _not_central_result(
    *,
    candidate_rank: int,
    generator_names: tuple[str, ...],
    centrality_score: float = float("inf"),
    phase_residual: float = float("inf"),
    notes: list[str],
) -> PeriodIndexDetection:
    return PeriodIndexDetection(
        detected=False,
        detector_mode="commutator_matrix",
        generator_names=generator_names,
        period=None,
        index=None,
        independent_pair_count=None,
        alternating_rank=None,
        radical_size=None,
        quotient_size=None,
        exponent_matrix=None,
        centrality_score=centrality_score,
        phase_residual=phase_residual,
        candidate_rank=int(candidate_rank),
        period_divides_rank=None,
        index_divides_rank=None,
        decision="not_central_projective",
        notes=notes,
    )


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


def detect_commutator_matrix_period_index(
    generators: Mapping[str, np.ndarray],
    candidate_rank: int,
    max_root_order: int = 12,
    centrality_tol: float = 1e-6,
    phase_tol: float = 1e-6,
    max_bruteforce_states: int = 200000,
) -> PeriodIndexDetection:
    """Detect period-index data from the central commutator matrix.

    The certified index is computed from the alternating exponent form.  Prime
    periods use rank over the finite field.  Composite periods use a
    conservative brute-force radical computation for small state spaces.
    """

    generator_names = tuple(generators)
    matrices = [np.asarray(generators[name], dtype=complex) for name in generator_names]
    rank = int(candidate_rank)
    if len(matrices) < 2:
        return _not_central_result(candidate_rank=rank, generator_names=generator_names, notes=["at least two generators are required"])
    first_shape = matrices[0].shape
    if len(first_shape) != 2 or first_shape[0] != first_shape[1]:
        return _not_central_result(candidate_rank=rank, generator_names=generator_names, notes=["generators must be square matrices"])
    if any(matrix.shape != first_shape for matrix in matrices):
        return _not_central_result(
            candidate_rank=rank,
            generator_names=generator_names,
            notes=["all generators must have the same square shape"],
        )

    pair_keys: list[tuple[int, int]] = []
    commutators: list[np.ndarray] = []
    try:
        for i in range(len(matrices)):
            for j in range(i + 1, len(matrices)):
                pair_keys.append((i, j))
                commutators.append(_commutator(matrices[i], matrices[j]))
    except np.linalg.LinAlgError:
        return _not_central_result(candidate_rank=rank, generator_names=generator_names, notes=["a generator was singular"])

    centralities = [_scalar_centrality(matrix)[0] for matrix in commutators]
    centrality_score = float(max(centralities, default=0.0))
    if centrality_score > centrality_tol:
        return _not_central_result(
            candidate_rank=rank,
            generator_names=generator_names,
            centrality_score=centrality_score,
            notes=["at least one generator commutator is noncentral"],
        )

    fits = [_nearest_root(matrix, max_root_order) for matrix in commutators]
    phase_residual = float(max((fit.residual for fit in fits), default=0.0))
    if phase_residual > phase_tol:
        return _not_central_result(
            candidate_rank=rank,
            generator_names=generator_names,
            centrality_score=centrality_score,
            phase_residual=phase_residual,
            notes=["central commutators did not fit finite roots of unity within tolerance"],
        )

    nontrivial_orders = [fit.order for fit in fits if fit.order > 1]
    if not nontrivial_orders:
        return _not_central_result(
            candidate_rank=rank,
            generator_names=generator_names,
            centrality_score=centrality_score,
            phase_residual=phase_residual,
            notes=["all scalar commutators are trivial; no nontrivial central projective class was detected"],
        )
    period = int(lcm(*nontrivial_orders))
    if period > max_root_order:
        period_divides = rank > 0 and rank % period == 0
        return PeriodIndexDetection(
            detected=True,
            detector_mode="commutator_matrix",
            generator_names=generator_names,
            period=period,
            index=None,
            independent_pair_count=None,
            alternating_rank=None,
            radical_size=None,
            quotient_size=None,
            exponent_matrix=None,
            centrality_score=centrality_score,
            phase_residual=phase_residual,
            candidate_rank=rank,
            period_divides_rank=period_divides,
            index_divides_rank=None,
            decision="central_projective_index_unknown",
            notes=[f"common period {period} exceeds max_root_order {max_root_order}; index not certified"],
        )

    pair_exponents = {
        pair: int((fit.exponent * (period // fit.order)) % period)
        for pair, fit in zip(pair_keys, fits, strict=True)
    }
    exponent_matrix = _build_exponent_matrix(len(matrices), pair_exponents, period)
    notes: list[str] = []
    index: int | None = None
    alternating_rank: int | None = None
    radical_size: int | None = None
    quotient_size: int | None = None

    if _is_prime(period):
        alternating_rank = _rank_mod_prime(exponent_matrix, period)
        if alternating_rank % 2 != 0:
            notes.append("alternating rank over the prime field was odd; index not certified")
        else:
            index = period ** (alternating_rank // 2)
            quotient_size = index * index
            radical_size = period ** (len(matrices) - alternating_rank)
            notes.append(f"prime-period rank computation certified rank {alternating_rank} and index {index}")
    else:
        radical_size, group_size, skip_reason = _radical_size_bruteforce(
            exponent_matrix,
            period,
            max_bruteforce_states,
        )
        if skip_reason is not None:
            notes.append(skip_reason)
        elif radical_size is not None and radical_size > 0:
            if group_size % radical_size != 0:
                notes.append("radical size did not divide the ambient group size; index not certified")
            else:
                quotient_size = group_size // radical_size
                candidate_index = isqrt(quotient_size)
                if candidate_index * candidate_index == quotient_size:
                    index = candidate_index
                    alternating_rank = _rank_from_radical(period, len(matrices), radical_size)
                    notes.append(
                        f"composite-period brute-force radical certified radical size {radical_size} and index {index}"
                    )
                else:
                    notes.append("nondegenerate quotient size was not a square; index not certified")

    if index is not None and index <= 1:
        return _not_central_result(
            candidate_rank=rank,
            generator_names=generator_names,
            centrality_score=centrality_score,
            phase_residual=phase_residual,
            notes=["alternating form has trivial projective index"],
        )

    pair_count = _integer_log_base(index, period) if index is not None else None
    decision, period_divides, index_divides, decision_notes = _decision(
        period=period,
        index=index,
        candidate_rank=rank,
        unknown=index is None,
    )
    notes.extend(decision_notes)

    return PeriodIndexDetection(
        detected=True,
        detector_mode="commutator_matrix",
        generator_names=generator_names,
        period=period,
        index=index,
        independent_pair_count=pair_count,
        alternating_rank=alternating_rank,
        radical_size=radical_size,
        quotient_size=quotient_size,
        exponent_matrix=exponent_matrix,
        centrality_score=centrality_score,
        phase_residual=phase_residual,
        candidate_rank=rank,
        period_divides_rank=period_divides,
        index_divides_rank=index_divides,
        decision=decision,
        notes=notes,
    )


def robust_detect_commutator_matrix_period_index(
    generators: Mapping[str, np.ndarray],
    candidate_rank: int,
    max_root_order: int = 12,
    centrality_tol_grid: tuple[float, ...] = (1e-6, 1e-4, 1e-3, 1e-2),
    phase_tol_grid: tuple[float, ...] = (1e-6, 1e-4, 1e-3, 1e-2),
    confidence_margin: float = 0.25,
    max_bruteforce_states: int = 200000,
) -> RobustPeriodIndexDetection:
    """Robust central commutator-matrix detector for noisy generators.

    The result is certified only when commutators pass the strict or medium
    threshold levels and nearest-root choices have a margin.  Loose detections
    are reported as diagnostics and never as lift certificates.
    """

    generator_names = tuple(generators)
    matrices = [np.asarray(generators[name], dtype=complex) for name in generator_names]
    rank = int(candidate_rank)
    empty_observations: tuple[CommutatorRootObservation, ...] = ()
    centrality_grid = tuple(sorted(float(value) for value in centrality_tol_grid))
    phase_grid = tuple(sorted(float(value) for value in phase_tol_grid))
    if not centrality_grid or not phase_grid:
        raise ValueError("centrality_tol_grid and phase_tol_grid must be nonempty")
    loose_centrality = centrality_grid[-1]
    loose_phase = phase_grid[-1]

    def invalid(notes: list[str], *, decision: str = "not_central_projective") -> RobustPeriodIndexDetection:
        return _robust_result_from_detection(
            status="rejected_noncentral",
            certified=False,
            detection=None,
            candidate_rank=rank,
            generator_names=generator_names,
            observations=empty_observations,
            threshold_level=None,
            centrality_tolerance=None,
            phase_tolerance=None,
            min_root_margin=None,
            decision=decision,
            notes=notes,
        )

    if len(matrices) < 2:
        return invalid(["at least two generators are required"])
    first_shape = matrices[0].shape
    if len(first_shape) != 2 or first_shape[0] != first_shape[1]:
        return invalid(["generators must be square matrices"])
    if any(matrix.shape != first_shape for matrix in matrices):
        return invalid(["all generators must have the same square shape"])

    pair_keys: list[tuple[int, int]] = []
    commutators: list[np.ndarray] = []
    try:
        for i in range(len(matrices)):
            for j in range(i + 1, len(matrices)):
                pair_keys.append((i, j))
                commutators.append(_commutator(matrices[i], matrices[j]))
    except np.linalg.LinAlgError:
        return invalid(["a generator was singular"])

    observations: list[CommutatorRootObservation] = []
    for (i, j), commutator in zip(pair_keys, commutators, strict=True):
        centrality, scalar_phase = _scalar_centrality(commutator)
        root_fit = _nearest_root_with_margin(commutator, max_root_order)
        observations.append(
            CommutatorRootObservation(
                pair=(generator_names[i], generator_names[j]),
                centrality_score=centrality,
                scalar_phase=scalar_phase,
                root_order=root_fit.order,
                root_exponent=root_fit.exponent,
                phase_residual=root_fit.residual,
                root_margin=root_fit.margin,
            )
        )
    observation_tuple = tuple(observations)
    max_centrality = float(max((obs.centrality_score for obs in observations), default=0.0))
    max_phase_residual = float(max((obs.phase_residual for obs in observations), default=0.0))
    min_root_margin = float(min((obs.root_margin for obs in observations), default=float("inf")))
    min_root_confidence = float(
        min(
            (
                obs.root_margin / max(obs.root_margin + obs.phase_residual, 1e-12)
                for obs in observations
            ),
            default=1.0,
        )
    )
    if max_centrality > loose_centrality:
        return _robust_result_from_detection(
            status="rejected_noncentral",
            certified=False,
            detection=None,
            candidate_rank=rank,
            generator_names=generator_names,
            observations=observation_tuple,
            threshold_level=None,
            centrality_tolerance=loose_centrality,
            phase_tolerance=loose_phase,
            min_root_margin=min_root_margin,
            min_root_confidence=min_root_confidence,
            decision="not_central_projective",
            notes=[
                f"commutator centrality {max_centrality:.6g} exceeds the loose threshold {loose_centrality:.6g}"
            ],
        )

    threshold_count = max(len(centrality_grid), len(phase_grid))
    threshold_rows: list[tuple[str, float, float, int]] = []
    for idx in range(threshold_count):
        centrality_tol = centrality_grid[min(idx, len(centrality_grid) - 1)]
        phase_tol = phase_grid[min(idx, len(phase_grid) - 1)]
        if idx == 0:
            level = "strict"
        elif idx == 1:
            level = "medium"
        else:
            level = "loose"
        threshold_rows.append((level, centrality_tol, phase_tol, idx))

    passed = [
        (level, centrality_tol, phase_tol, idx)
        for level, centrality_tol, phase_tol, idx in threshold_rows
        if max_centrality <= centrality_tol and max_phase_residual <= phase_tol
    ]
    if not passed:
        return _robust_result_from_detection(
            status="unknown_index",
            certified=False,
            detection=None,
            candidate_rank=rank,
            generator_names=generator_names,
            observations=observation_tuple,
            threshold_level=None,
            centrality_tolerance=loose_centrality,
            phase_tolerance=loose_phase,
            min_root_margin=min_root_margin,
            min_root_confidence=min_root_confidence,
            decision="central_projective_index_unknown",
            notes=[
                "commutators were close enough to scalar to avoid noncentral rejection, "
                "but did not fit finite roots of unity on the configured tolerance grid"
            ],
        )

    level, centrality_tol, phase_tol, idx = passed[0]
    detection = detect_commutator_matrix_period_index(
        generators,
        candidate_rank=rank,
        max_root_order=max_root_order,
        centrality_tol=centrality_tol,
        phase_tol=phase_tol,
        max_bruteforce_states=max_bruteforce_states,
    )
    notes = list(detection.notes)
    notes.append(
        f"robust threshold level {level} accepted centrality {max_centrality:.6g} "
        f"and phase residual {max_phase_residual:.6g}"
    )

    if detection.decision == "not_central_projective":
        return _robust_result_from_detection(
            status="rejected_noncentral",
            certified=False,
            detection=detection,
            candidate_rank=rank,
            generator_names=generator_names,
            observations=observation_tuple,
            threshold_level=level,
            centrality_tolerance=centrality_tol,
            phase_tolerance=phase_tol,
            min_root_margin=min_root_margin,
            decision=detection.decision,
            notes=notes,
        )

    if min_root_confidence < confidence_margin:
        notes.append(
            f"nearest-root relative confidence {min_root_confidence:.6g} is below confidence margin {confidence_margin:.6g}"
        )
        return _robust_result_from_detection(
            status="candidate_uncertain",
            certified=False,
            detection=detection,
            candidate_rank=rank,
            generator_names=generator_names,
            observations=observation_tuple,
            threshold_level=level,
            centrality_tolerance=centrality_tol,
            phase_tolerance=phase_tol,
            min_root_margin=min_root_margin,
            min_root_confidence=min_root_confidence,
            decision="central_projective_candidate_uncertain",
            notes=notes,
        )

    if idx >= 2:
        notes.append("only loose thresholds passed; this is a diagnostic candidate, not a lift certificate")
        return _robust_result_from_detection(
            status="candidate_uncertain",
            certified=False,
            detection=detection,
            candidate_rank=rank,
            generator_names=generator_names,
            observations=observation_tuple,
            threshold_level=level,
            centrality_tolerance=centrality_tol,
            phase_tolerance=phase_tol,
            min_root_margin=min_root_margin,
            min_root_confidence=min_root_confidence,
            decision="central_projective_candidate_uncertain",
            notes=notes,
        )

    if detection.index is None:
        notes.append("scalar central commutators were visible, but the index was not certified")
        return _robust_result_from_detection(
            status="unknown_index",
            certified=False,
            detection=detection,
            candidate_rank=rank,
            generator_names=generator_names,
            observations=observation_tuple,
            threshold_level=level,
            centrality_tolerance=centrality_tol,
            phase_tolerance=phase_tol,
            min_root_margin=min_root_margin,
            min_root_confidence=min_root_confidence,
            decision="central_projective_index_unknown",
            notes=notes,
        )

    return _robust_result_from_detection(
        status="certified",
        certified=True,
        detection=detection,
        candidate_rank=rank,
        generator_names=generator_names,
        observations=observation_tuple,
        threshold_level=level,
        centrality_tolerance=centrality_tol,
        phase_tolerance=phase_tol,
        min_root_margin=min_root_margin,
        min_root_confidence=min_root_confidence,
        notes=notes,
    )


def _detect_independent_pair_structure(
    generators: Mapping[str, np.ndarray],
    candidate_rank: int,
    max_root_order: int = 12,
    centrality_tolerance: float = 1e-8,
    phase_tolerance: float = 1e-8,
) -> PeriodIndexDetection:
    general = detect_commutator_matrix_period_index(
        generators,
        candidate_rank,
        max_root_order=max_root_order,
        centrality_tol=centrality_tolerance,
        phase_tol=phase_tolerance,
        max_bruteforce_states=0,
    )
    if not general.detected or general.period is None or general.exponent_matrix is None:
        return general
    pair_count = _independent_pair_count(general.exponent_matrix, general.period)
    if pair_count is None:
        return general
    index = general.period**pair_count
    decision, period_divides, index_divides, decision_notes = _decision(
        period=general.period,
        index=index,
        candidate_rank=candidate_rank,
    )
    notes = [f"recognized {pair_count} independent Heisenberg pair(s); period={general.period}, index={index}"]
    notes.extend(decision_notes)
    return PeriodIndexDetection(
        detected=True,
        detector_mode="independent_pair",
        generator_names=general.generator_names,
        period=general.period,
        index=index,
        independent_pair_count=pair_count,
        alternating_rank=2 * pair_count,
        radical_size=general.period ** (len(general.generator_names) - 2 * pair_count),
        quotient_size=index * index,
        exponent_matrix=general.exponent_matrix,
        centrality_score=general.centrality_score,
        phase_residual=general.phase_residual,
        candidate_rank=int(candidate_rank),
        period_divides_rank=period_divides,
        index_divides_rank=index_divides,
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
    """Backward-compatible period-index detector entry point.

    The commutator-matrix detector is tried first.  If it sees scalar finite
    commutators but cannot certify an index, the previous independent-pair
    recognizer is allowed to certify the controlled Heisenberg pattern.
    """

    general = detect_commutator_matrix_period_index(
        generators,
        candidate_rank,
        max_root_order=max_root_order,
        centrality_tol=centrality_tolerance,
        phase_tol=phase_tolerance,
    )
    if general.decision != "central_projective_index_unknown":
        return general

    independent = _detect_independent_pair_structure(
        generators,
        candidate_rank,
        max_root_order=max_root_order,
        centrality_tolerance=centrality_tolerance,
        phase_tolerance=phase_tolerance,
    )
    if independent.index is not None:
        return independent
    return general
