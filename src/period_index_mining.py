"""Noise models and synthetic mining helpers for period-index diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Mapping, Sequence

import numpy as np

from .period_index_central import heisenberg_generators
from .period_index_detector import (
    RobustPeriodIndexDetection,
    robust_detect_commutator_matrix_period_index,
)


IndexPair = tuple[int, int]
Loop = tuple[int, ...]


@dataclass(frozen=True)
class LoopGeneratorCandidate:
    name: str
    loop: Loop
    matrix: np.ndarray
    unitary_error: float
    nontriviality: float
    scalar_commutator_score: float
    score: float


@dataclass(frozen=True)
class PeriodIndexMiningResult:
    status: str
    generators: dict[str, np.ndarray]
    selected_loops: tuple[Loop, ...]
    candidates: tuple[LoopGeneratorCandidate, ...]
    explanation: tuple[str, ...]


@dataclass(frozen=True)
class MinedPeriodIndexDetection:
    mining: PeriodIndexMiningResult
    detection: RobustPeriodIndexDetection | None


def _rng(seed: int | None = None, rng: np.random.Generator | None = None) -> np.random.Generator:
    return rng if rng is not None else np.random.default_rng(seed)


def project_to_nearest_unitary(matrix: np.ndarray) -> np.ndarray:
    """Project a square matrix to the nearest unitary in Frobenius norm."""

    arr = np.asarray(matrix, dtype=complex)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("matrix must be square")
    left, _singular_values, right = np.linalg.svd(arr)
    return left @ right


def add_unitary_noise(
    matrix: np.ndarray,
    epsilon: float,
    *,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Left-multiply a matrix by a small random near-identity unitary."""

    arr = np.asarray(matrix, dtype=complex)
    generator = _rng(seed, rng)
    real = generator.normal(size=arr.shape)
    imag = generator.normal(size=arr.shape)
    raw = real + 1j * imag
    skew = raw - raw.conj().T
    near_identity = np.eye(arr.shape[0], dtype=complex) + float(epsilon) * skew
    return project_to_nearest_unitary(near_identity) @ arr


def add_entrywise_noise(
    matrix: np.ndarray,
    epsilon: float,
    *,
    project_unitary: bool = True,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Add complex Gaussian entrywise noise and optionally project to unitary."""

    arr = np.asarray(matrix, dtype=complex)
    generator = _rng(seed, rng)
    noise = (generator.normal(size=arr.shape) + 1j * generator.normal(size=arr.shape)) / np.sqrt(2.0)
    perturbed = arr + float(epsilon) * noise
    return project_to_nearest_unitary(perturbed) if project_unitary and arr.ndim == 2 and arr.shape[0] == arr.shape[1] else perturbed


def generate_noisy_heisenberg_generators(
    d: int,
    k: int,
    noise_level: float,
    noise_type: str,
    *,
    seed: int | None = 0,
) -> dict[str, np.ndarray]:
    """Return noisy finite-Heisenberg generators using deterministic names."""

    system = heisenberg_generators(d, k)
    clean: dict[str, np.ndarray] = {}
    for idx in range(k):
        clean[f"U{idx + 1}"] = system.U[idx]
        clean[f"V{idx + 1}"] = system.V[idx]
    if noise_level <= 0 or noise_type in {"none", "exact"}:
        return clean
    generator = np.random.default_rng(seed)
    noisy: dict[str, np.ndarray] = {}
    for name, matrix in clean.items():
        if noise_type == "unitary_near_identity":
            noisy[name] = add_unitary_noise(matrix, noise_level, rng=generator)
        elif noise_type == "entrywise_projected_unitary":
            noisy[name] = add_entrywise_noise(matrix, noise_level, project_unitary=True, rng=generator)
        else:
            raise ValueError(f"unknown noise_type: {noise_type}")
    return noisy


def generate_noncentral_controls(
    width: int,
    noise_level: float = 0.0,
    *,
    seed: int | None = 0,
    control_type: str = "permutation",
) -> dict[str, np.ndarray]:
    """Generate noncentral matrix controls for robust rejection tests."""

    if width < 3:
        raise ValueError("width must be at least 3 for the noncentral controls")
    generator = np.random.default_rng(seed)
    if control_type == "permutation":
        p = np.eye(width, dtype=complex)
        q = np.eye(width, dtype=complex)
        p[[0, 1], :] = p[[1, 0], :]
        q[[1, 2], :] = q[[2, 1], :]
        controls = {"s12": p, "s23": q}
    elif control_type == "random_gl":
        controls = {}
        for idx in range(2):
            raw = generator.normal(size=(width, width)) + 1j * generator.normal(size=(width, width))
            controls[f"G{idx + 1}"] = raw + width * np.eye(width, dtype=complex)
    else:
        raise ValueError(f"unknown control_type: {control_type}")
    if noise_level <= 0:
        return controls
    return {
        name: add_entrywise_noise(matrix, noise_level, project_unitary=(control_type == "permutation"), rng=generator)
        for name, matrix in controls.items()
    }


def _relative_residual(left: np.ndarray, right: np.ndarray) -> float:
    denom = max(float(np.linalg.norm(right, ord="fro")), 1e-12)
    return float(np.linalg.norm(left - right, ord="fro") / denom)


def _unitary_error(matrix: np.ndarray) -> float:
    arr = np.asarray(matrix, dtype=complex)
    identity = np.eye(arr.shape[0], dtype=complex)
    return _relative_residual(arr.conj().T @ arr, identity)


def _nontriviality(matrix: np.ndarray) -> float:
    arr = np.asarray(matrix, dtype=complex)
    return _relative_residual(arr, np.eye(arr.shape[0], dtype=complex))


def _scalar_centrality(matrix: np.ndarray) -> float:
    arr = np.asarray(matrix, dtype=complex)
    scalar = complex(np.trace(arr) / max(arr.shape[0], 1))
    target = scalar * np.eye(arr.shape[0], dtype=complex)
    denom = max(float(np.linalg.norm(target, ord="fro")), 1e-12)
    return float(np.linalg.norm(arr - target, ord="fro") / denom)


def _commutator(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return A @ B @ np.linalg.inv(A) @ np.linalg.inv(B)


def _candidate_scalar_score(matrix: np.ndarray, all_matrices: Sequence[np.ndarray]) -> float:
    scores = []
    for other in all_matrices:
        if other is matrix:
            continue
        try:
            scores.append(_scalar_centrality(_commutator(matrix, other)))
        except np.linalg.LinAlgError:
            scores.append(float("inf"))
    return float(max(scores, default=0.0))


def _infer_nodes(transition_maps: Mapping[IndexPair, np.ndarray]) -> tuple[int, ...]:
    nodes = set()
    for left, right in transition_maps:
        nodes.add(int(left))
        nodes.add(int(right))
    return tuple(sorted(nodes))


def _default_loops(transition_maps: Mapping[IndexPair, np.ndarray]) -> tuple[Loop, ...]:
    nodes = _infer_nodes(transition_maps)
    loops: list[Loop] = []
    for triple in combinations(nodes, 3):
        loop = (triple[0], triple[1], triple[2], triple[0])
        if _loop_available(transition_maps, loop):
            loops.append(loop)
        reverse = (triple[0], triple[2], triple[1], triple[0])
        if _loop_available(transition_maps, reverse):
            loops.append(reverse)
    for quad in combinations(nodes, 4):
        loop = (quad[0], quad[1], quad[2], quad[3], quad[0])
        if _loop_available(transition_maps, loop):
            loops.append(loop)
    return tuple(loops)


def _closed_loop(loop: Sequence[int]) -> Loop:
    if len(loop) < 2:
        raise ValueError("loop must contain at least two nodes")
    value = tuple(int(node) for node in loop)
    return value if value[0] == value[-1] else (*value, value[0])


def _loop_available(transition_maps: Mapping[IndexPair, np.ndarray], loop: Sequence[int]) -> bool:
    closed = _closed_loop(loop)
    return all((closed[idx], closed[idx + 1]) in transition_maps for idx in range(len(closed) - 1))


def _loop_holonomy(transition_maps: Mapping[IndexPair, np.ndarray], loop: Sequence[int]) -> np.ndarray:
    closed = _closed_loop(loop)
    if not _loop_available(transition_maps, closed):
        raise KeyError(f"missing transition map for loop {closed}")
    first = np.asarray(transition_maps[(closed[0], closed[1])], dtype=complex)
    holonomy = np.eye(first.shape[0], dtype=complex)
    for idx in range(len(closed) - 1):
        holonomy = holonomy @ np.asarray(transition_maps[(closed[idx], closed[idx + 1])], dtype=complex)
    return holonomy


def mine_period_index_generators(
    transition_maps: Mapping[IndexPair, np.ndarray] | None,
    loops: Sequence[Sequence[int]] | None = None,
    max_generators: int = 6,
    nontriviality_tol: float = 1e-8,
) -> PeriodIndexMiningResult:
    """Mine a small generator set from short loop holonomies."""

    if not transition_maps:
        return PeriodIndexMiningResult(
            status="not_evaluated",
            generators={},
            selected_loops=(),
            candidates=(),
            explanation=("no transition maps were supplied",),
        )
    candidate_loops = tuple(_closed_loop(loop) for loop in loops) if loops is not None else _default_loops(transition_maps)
    available_loops = tuple(loop for loop in candidate_loops if _loop_available(transition_maps, loop))
    if not available_loops:
        return PeriodIndexMiningResult(
            status="not_evaluated",
            generators={},
            selected_loops=(),
            candidates=(),
            explanation=("no supplied or inferred short loops were available in the transition map graph",),
        )

    matrices = [_loop_holonomy(transition_maps, loop) for loop in available_loops]
    candidates: list[LoopGeneratorCandidate] = []
    for idx, (loop, matrix) in enumerate(zip(available_loops, matrices, strict=True)):
        unitary_error = _unitary_error(matrix)
        nontriviality = _nontriviality(matrix)
        scalar_score = _candidate_scalar_score(matrix, matrices)
        score = unitary_error + scalar_score - 0.05 * nontriviality
        candidates.append(
            LoopGeneratorCandidate(
                name=f"L{idx + 1}",
                loop=loop,
                matrix=matrix,
                unitary_error=unitary_error,
                nontriviality=nontriviality,
                scalar_commutator_score=scalar_score,
                score=score,
            )
        )

    ranked = sorted(
        (candidate for candidate in candidates if candidate.nontriviality > nontriviality_tol),
        key=lambda candidate: (candidate.score, candidate.unitary_error, -candidate.nontriviality),
    )
    selected = ranked[: max(0, int(max_generators))]
    if len(selected) < 2:
        return PeriodIndexMiningResult(
            status="not_evaluated",
            generators={},
            selected_loops=(),
            candidates=tuple(candidates),
            explanation=("fewer than two nontrivial loop holonomies were available",),
        )

    generators = {f"M{idx + 1}": candidate.matrix for idx, candidate in enumerate(selected)}
    return PeriodIndexMiningResult(
        status="mined_candidate",
        generators=generators,
        selected_loops=tuple(candidate.loop for candidate in selected),
        candidates=tuple(candidates),
        explanation=(
            f"selected {len(selected)} loop holonomies with scalar pairwise commutators",
            "mined candidates are diagnostics unless the robust detector certifies their index",
        ),
    )


def detect_mined_period_index(
    transition_maps: Mapping[IndexPair, np.ndarray] | None,
    candidate_rank: int,
    *,
    loops: Sequence[Sequence[int]] | None = None,
    max_generators: int = 6,
    max_root_order: int = 12,
    centrality_tol_grid: tuple[float, ...] = (1e-6, 1e-4, 1e-3, 1e-2),
    phase_tol_grid: tuple[float, ...] = (1e-6, 1e-4, 1e-3, 1e-2),
    confidence_margin: float = 0.25,
    max_bruteforce_states: int = 200000,
) -> MinedPeriodIndexDetection:
    mining = mine_period_index_generators(
        transition_maps,
        loops=loops,
        max_generators=max_generators,
    )
    if mining.status != "mined_candidate":
        return MinedPeriodIndexDetection(mining=mining, detection=None)
    detection = robust_detect_commutator_matrix_period_index(
        mining.generators,
        candidate_rank=candidate_rank,
        max_root_order=max_root_order,
        centrality_tol_grid=centrality_tol_grid,
        phase_tol_grid=phase_tol_grid,
        confidence_margin=confidence_margin,
        max_bruteforce_states=max_bruteforce_states,
    )
    detection = RobustPeriodIndexDetection(
        status=detection.status,
        certified=detection.certified,
        detector_mode="robust_commutator_matrix_mined_candidate",
        generator_names=detection.generator_names,
        period=detection.period,
        index=detection.index,
        independent_pair_count=detection.independent_pair_count,
        exponent_matrix=detection.exponent_matrix,
        centrality_score=detection.centrality_score,
        phase_residual=detection.phase_residual,
        candidate_rank=detection.candidate_rank,
        period_divides_rank=detection.period_divides_rank,
        index_divides_rank=detection.index_divides_rank,
        decision=detection.decision,
        notes=[*detection.notes, *mining.explanation],
        threshold_level=detection.threshold_level,
        centrality_tolerance=detection.centrality_tolerance,
        phase_tolerance=detection.phase_tolerance,
        min_root_margin=detection.min_root_margin,
        min_root_confidence=detection.min_root_confidence,
        alternating_rank=detection.alternating_rank,
        radical_size=detection.radical_size,
        quotient_size=detection.quotient_size,
        pair_observations=detection.pair_observations,
        exact_detection=detection.exact_detection,
    )
    return MinedPeriodIndexDetection(mining=mining, detection=detection)
