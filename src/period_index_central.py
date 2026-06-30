"""Central period-index benchmark utilities.

This module builds the controlled k-pair finite Heisenberg example

    U_i V_i = zeta V_i U_i

with all other generator pairs commuting.  The representation is realized on
``(C^d)^{tensor k}``, so the period is ``d`` while the benchmark index is
``d^k``.  Candidate lift ranks are accepted exactly when they are multiples of
``d^k``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HeisenbergGenerators:
    d: int
    k: int
    zeta: complex
    U: tuple[np.ndarray, ...]
    V: tuple[np.ndarray, ...]
    dimension: int

    def __iter__(self):
        yield self.U
        yield self.V
        yield self.dimension


@dataclass(frozen=True)
class RelationCheck:
    d: int
    k: int
    dimension: int
    max_relation_residual: float
    relation_residuals: dict[str, float]
    all_relations_hold: bool


@dataclass(frozen=True)
class CentralPeriodIndexMetadata:
    d: int
    k: int
    period: int
    index: int
    dimension: int
    zeta: complex
    lift_kind: str = "finite_rank_projective_or_morita_lift"
    ordinary_untwisted_descent_on_original_rank: bool = False
    original_class_vanishes_on_same_cover: bool = False
    model_note: str = "controlled central/projective period-index benchmark"

    def candidate_rank_allowed(self, rank: int) -> bool:
        return rank > 0 and rank % self.index == 0

    def interpretation(self, rank: int) -> str:
        if self.candidate_rank_allowed(rank):
            return (
                f"rank {rank} is a projective/Morita lift rank for the k={self.k}, "
                f"d={self.d} central period-index system; it absorbs the class in "
                "the lifted representation."
            )
        if rank > 0 and rank % self.period == 0:
            return (
                f"rank {rank} is divisible by the period {self.period}, but not by "
                f"the index {self.index}; period divisibility alone is not enough."
            )
        return (
            f"rank {rank} is obstructed for the k={self.k}, d={self.d} central "
            f"period-index system; no ordinary same-cover trivialization is claimed."
        )


@dataclass(frozen=True)
class PeriodIndexLiftResult:
    d: int
    k: int
    period: int
    index: int
    candidate_rank: int
    period_divides_rank: bool
    index_divides_rank: bool
    obstruction_prediction: str
    constructed_lift_success: bool
    max_relation_residual: float
    is_minimal_success: bool
    lift_kind: str
    ordinary_untwisted_descent_on_original_rank: bool
    original_class_vanishes_on_same_cover: bool
    interpretation: str


def primitive_root(d: int) -> complex:
    if d <= 0:
        raise ValueError("d must be positive")
    return complex(np.exp(2j * np.pi / d))


def clock_matrix(d: int, zeta: complex | None = None) -> np.ndarray:
    if d <= 0:
        raise ValueError("d must be positive")
    root = primitive_root(d) if zeta is None else complex(zeta)
    return np.diag([root**j for j in range(d)]).astype(complex)


def shift_matrix(d: int) -> np.ndarray:
    if d <= 0:
        raise ValueError("d must be positive")
    matrix = np.zeros((d, d), dtype=complex)
    for col in range(d):
        matrix[(col + 1) % d, col] = 1.0
    return matrix


def tensor_on_factor(matrix: np.ndarray, factor: int, k: int, d: int) -> np.ndarray:
    arr = np.asarray(matrix, dtype=complex)
    if arr.shape != (d, d):
        raise ValueError("matrix must have shape (d, d)")
    if k <= 0:
        raise ValueError("k must be positive")
    if factor < 0 or factor >= k:
        raise ValueError("factor must satisfy 0 <= factor < k")

    out = np.array([[1.0 + 0.0j]])
    identity = np.eye(d, dtype=complex)
    for idx in range(k):
        out = np.kron(out, arr if idx == factor else identity)
    return out


def heisenberg_generators(d: int, k: int) -> HeisenbergGenerators:
    if d <= 0:
        raise ValueError("d must be positive")
    if k <= 0:
        raise ValueError("k must be positive")
    zeta = primitive_root(d)
    U0 = clock_matrix(d, zeta)
    V0 = shift_matrix(d)
    U = tuple(tensor_on_factor(U0, factor, k, d) for factor in range(k))
    V = tuple(tensor_on_factor(V0, factor, k, d) for factor in range(k))
    return HeisenbergGenerators(d=d, k=k, zeta=zeta, U=U, V=V, dimension=d**k)


def period_index_metadata(d: int, k: int) -> CentralPeriodIndexMetadata:
    if d <= 0:
        raise ValueError("d must be positive")
    if k <= 0:
        raise ValueError("k must be positive")
    index = d**k
    return CentralPeriodIndexMetadata(
        d=d,
        k=k,
        period=d,
        index=index,
        dimension=index,
        zeta=primitive_root(d),
    )


def _relative_residual(left: np.ndarray, right: np.ndarray) -> float:
    denom = max(float(np.linalg.norm(right, ord="fro")), 1e-12)
    return float(np.linalg.norm(left - right, ord="fro") / denom)


def _inverse(matrix: np.ndarray) -> np.ndarray:
    return np.linalg.inv(np.asarray(matrix, dtype=complex))


def central_commutator_defect_score(A: np.ndarray, B: np.ndarray, zeta: complex) -> float:
    if A.shape != B.shape or A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A and B must be square matrices of the same shape")
    commutator = A @ B @ _inverse(A) @ _inverse(B)
    target = complex(zeta) * np.eye(A.shape[0], dtype=complex)
    return _relative_residual(commutator, target)


def check_heisenberg_relations(
    generators: HeisenbergGenerators,
    tolerance: float = 1e-10,
) -> RelationCheck:
    U = generators.U
    V = generators.V
    zeta = generators.zeta
    residuals: dict[str, float] = {}
    for i in range(generators.k):
        residuals[f"U{i + 1}V{i + 1}=zetaV{i + 1}U{i + 1}"] = _relative_residual(
            U[i] @ V[i],
            zeta * (V[i] @ U[i]),
        )
        residuals[f"commutator_U{i + 1}_V{i + 1}"] = central_commutator_defect_score(U[i], V[i], zeta)

    for i in range(generators.k):
        for j in range(generators.k):
            if i != j:
                residuals[f"U{i + 1}V{j + 1}_commutes"] = _relative_residual(U[i] @ V[j], V[j] @ U[i])

    for i in range(generators.k):
        for j in range(i + 1, generators.k):
            residuals[f"U{i + 1}U{j + 1}_commutes"] = _relative_residual(U[i] @ U[j], U[j] @ U[i])
            residuals[f"V{i + 1}V{j + 1}_commutes"] = _relative_residual(V[i] @ V[j], V[j] @ V[i])

    max_residual = float(max(residuals.values(), default=0.0))
    return RelationCheck(
        d=generators.d,
        k=generators.k,
        dimension=generators.dimension,
        max_relation_residual=max_residual,
        relation_residuals=residuals,
        all_relations_hold=max_residual <= tolerance,
    )


def _block_diag(matrices: list[np.ndarray]) -> np.ndarray:
    if not matrices:
        return np.zeros((0, 0), dtype=complex)
    size = sum(matrix.shape[0] for matrix in matrices)
    out = np.zeros((size, size), dtype=complex)
    cursor = 0
    for matrix in matrices:
        n = matrix.shape[0]
        out[cursor : cursor + n, cursor : cursor + n] = matrix
        cursor += n
    return out


def direct_sum_lift(d: int, k: int, rank: int) -> HeisenbergGenerators | None:
    metadata = period_index_metadata(d, k)
    if not metadata.candidate_rank_allowed(rank):
        return None
    base = heisenberg_generators(d, k)
    copies = rank // metadata.index
    U = tuple(_block_diag([base.U[i].copy() for _ in range(copies)]) for i in range(k))
    V = tuple(_block_diag([base.V[i].copy() for _ in range(copies)]) for i in range(k))
    return HeisenbergGenerators(d=d, k=k, zeta=metadata.zeta, U=U, V=V, dimension=rank)


def attempted_diagnostic_lift(d: int, k: int, rank: int) -> HeisenbergGenerators:
    if rank <= 0:
        raise ValueError("rank must be positive")
    metadata = period_index_metadata(d, k)
    exact = direct_sum_lift(d, k, rank)
    if exact is not None:
        return exact

    base = heisenberg_generators(d, k)
    copies = rank // metadata.index
    leftover = rank % metadata.index
    U_blocks: list[list[np.ndarray]] = [[] for _ in range(k)]
    V_blocks: list[list[np.ndarray]] = [[] for _ in range(k)]
    for i in range(k):
        U_blocks[i].extend(base.U[i].copy() for _ in range(copies))
        V_blocks[i].extend(base.V[i].copy() for _ in range(copies))
        if leftover:
            U_blocks[i].append(np.eye(leftover, dtype=complex))
            V_blocks[i].append(np.eye(leftover, dtype=complex))
        if not U_blocks[i]:
            U_blocks[i].append(np.eye(rank, dtype=complex))
            V_blocks[i].append(np.eye(rank, dtype=complex))
    return HeisenbergGenerators(
        d=d,
        k=k,
        zeta=metadata.zeta,
        U=tuple(_block_diag(blocks) for blocks in U_blocks),
        V=tuple(_block_diag(blocks) for blocks in V_blocks),
        dimension=rank,
    )


def check_period_index_obstruction(
    d: int,
    k: int,
    rank: int,
    tolerance: float = 1e-10,
) -> PeriodIndexLiftResult:
    metadata = period_index_metadata(d, k)
    if rank <= 0:
        relation_residual = float("nan")
        constructed = False
    else:
        candidate = direct_sum_lift(d, k, rank) or attempted_diagnostic_lift(d, k, rank)
        relation_residual = check_heisenberg_relations(candidate, tolerance=tolerance).max_relation_residual
        constructed = metadata.candidate_rank_allowed(rank) and relation_residual <= tolerance

    index_divides_rank = metadata.candidate_rank_allowed(rank)
    period_divides_rank = rank > 0 and rank % metadata.period == 0
    return PeriodIndexLiftResult(
        d=d,
        k=k,
        period=metadata.period,
        index=metadata.index,
        candidate_rank=rank,
        period_divides_rank=period_divides_rank,
        index_divides_rank=index_divides_rank,
        obstruction_prediction="lift_success" if index_divides_rank else "obstructed",
        constructed_lift_success=constructed,
        max_relation_residual=relation_residual,
        is_minimal_success=constructed and rank == metadata.index,
        lift_kind=metadata.lift_kind,
        ordinary_untwisted_descent_on_original_rank=metadata.ordinary_untwisted_descent_on_original_rank,
        original_class_vanishes_on_same_cover=metadata.original_class_vanishes_on_same_cover,
        interpretation=metadata.interpretation(rank),
    )


def toy_prediction_losses(d: int, k: int, rank: int) -> dict[str, float | str]:
    result = check_period_index_obstruction(d, k, rank)
    lifted_loss = 0.0 if result.constructed_lift_success else 1.0
    return {
        "ordinary_rank_r_loss": lifted_loss,
        "lifted_rank_r_loss": lifted_loss,
        "branch_projective_loss": 0.0,
        "extra_capacity_label": "projective_branch_extra_capacity",
        "prediction_note": "algebraic period-index proxy only; no MNIST/CIFAR claim",
    }
