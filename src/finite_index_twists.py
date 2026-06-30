"""Finite-index projective torsion twist utilities.

The core toy theorem is the determinant obstruction for a projective pair

    A B = zeta B A.

If zeta has order d and A, B are invertible r x r complex matrices, then d
divides r.  Clock and shift matrices realize the converse in rank d, and
direct sums realize every multiple of d.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd

import numpy as np


@dataclass(frozen=True)
class FiniteTorsionClass:
    q: int
    exponent: int
    order: int
    zeta: complex
    period: int
    expected_index: int
    lift_kind: str = "finite_rank_projective_or_morita_lift"
    ordinary_untwisted_descent_on_original_rank: bool = False

    def lift_success(self, rank: int) -> bool:
        return determinant_obstruction_allows(self.order, rank)

    def interpretation(self, rank: int) -> str:
        if self.lift_success(rank):
            return (
                f"rank {rank} is a finite-rank projective/Morita lift for order {self.order}; "
                "this absorbs the torsion defect but does not make the original class vanish."
            )
        return (
            f"rank {rank} is excluded by the determinant obstruction for order {self.order}; "
            "no ordinary untwisted descent is claimed."
        )


@dataclass(frozen=True)
class RankAbsorptionResult:
    q: int
    exponent: int
    order: int
    rank: int
    determinant_allows: bool
    constructed_lift_success: bool
    commutator_residual: float
    is_minimal_success: bool
    ordinary_untwisted_descent_on_original_rank: bool
    lift_kind: str
    interpretation: str


def primitive_root_of_unity(order: int) -> complex:
    if order <= 0:
        raise ValueError("order must be positive")
    return complex(np.exp(2j * np.pi / order))


def root_of_unity(q: int, exponent: int = 1) -> complex:
    if q <= 0:
        raise ValueError("q must be positive")
    return complex(np.exp(2j * np.pi * (exponent % q) / q))


def torsion_order(q: int, exponent: int = 1) -> int:
    if q <= 0:
        raise ValueError("q must be positive")
    a = exponent % q
    if a == 0:
        return 1
    return q // gcd(q, a)


def finite_torsion_class(q: int, exponent: int = 1) -> FiniteTorsionClass:
    order = torsion_order(q, exponent)
    return FiniteTorsionClass(
        q=q,
        exponent=exponent,
        order=order,
        zeta=root_of_unity(q, exponent),
        period=order,
        expected_index=order,
    )


def clock_matrix(order: int, zeta: complex | None = None) -> np.ndarray:
    if order <= 0:
        raise ValueError("order must be positive")
    root = primitive_root_of_unity(order) if zeta is None else complex(zeta)
    return np.diag([root**k for k in range(order)]).astype(complex)


def shift_matrix(order: int) -> np.ndarray:
    if order <= 0:
        raise ValueError("order must be positive")
    matrix = np.zeros((order, order), dtype=complex)
    for col in range(order):
        matrix[(col + 1) % order, col] = 1.0
    return matrix


def block_diag(matrices: list[np.ndarray]) -> np.ndarray:
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


def commutator(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return A @ B @ np.linalg.inv(A) @ np.linalg.inv(B)


def commutator_defect_score(A: np.ndarray, B: np.ndarray, zeta: complex) -> float:
    if A.shape != B.shape or A.shape[0] != A.shape[1]:
        raise ValueError("A and B must be square matrices of the same shape")
    target = complex(zeta) * np.eye(A.shape[0], dtype=complex)
    denom = max(float(np.linalg.norm(target, ord="fro")), 1e-12)
    return float(np.linalg.norm(commutator(A, B) - target, ord="fro") / denom)


def determinant_obstruction_allows(order: int, rank: int) -> bool:
    if order <= 0:
        raise ValueError("order must be positive")
    if rank <= 0:
        return False
    return rank % order == 0


def determinant_obstruction_allows_class(q: int, exponent: int, rank: int) -> bool:
    return determinant_obstruction_allows(torsion_order(q, exponent), rank)


def direct_sum_lift(q: int, exponent: int, rank: int) -> tuple[np.ndarray, np.ndarray] | None:
    cls = finite_torsion_class(q, exponent)
    if not determinant_obstruction_allows(cls.order, rank):
        return None
    if cls.order == 1:
        return np.eye(rank, dtype=complex), np.eye(rank, dtype=complex)
    copies = rank // cls.order
    U = clock_matrix(cls.order, cls.zeta)
    V = shift_matrix(cls.order)
    return block_diag([U.copy() for _ in range(copies)]), block_diag([V.copy() for _ in range(copies)])


def attempted_lift(q: int, exponent: int, rank: int) -> tuple[np.ndarray, np.ndarray]:
    """Return an invertible rank-r attempt.

    For ranks not divisible by the order, this pads exact clock-shift blocks
    with identity blocks.  The residual is intentionally nonzero on the padded
    part; this is a diagnostic failed attempt, not a hidden construction.
    """

    cls = finite_torsion_class(q, exponent)
    if rank <= 0:
        raise ValueError("rank must be positive")
    if cls.order == 1:
        return np.eye(rank, dtype=complex), np.eye(rank, dtype=complex)
    copies = rank // cls.order
    leftover = rank % cls.order
    blocks_u = [clock_matrix(cls.order, cls.zeta) for _ in range(copies)]
    blocks_v = [shift_matrix(cls.order) for _ in range(copies)]
    if leftover:
        blocks_u.append(np.eye(leftover, dtype=complex))
        blocks_v.append(np.eye(leftover, dtype=complex))
    if not blocks_u:
        blocks_u = [np.eye(rank, dtype=complex)]
        blocks_v = [np.eye(rank, dtype=complex)]
    return block_diag(blocks_u), block_diag(blocks_v)


def evaluate_rank_absorption(
    q: int,
    exponent: int,
    rank: int,
    tolerance: float = 1e-10,
) -> RankAbsorptionResult:
    cls = finite_torsion_class(q, exponent)
    A, B = attempted_lift(q, exponent, rank)
    residual = commutator_defect_score(A, B, cls.zeta)
    allowed = cls.lift_success(rank)
    constructed = allowed and residual <= tolerance
    return RankAbsorptionResult(
        q=q,
        exponent=exponent,
        order=cls.order,
        rank=rank,
        determinant_allows=allowed,
        constructed_lift_success=constructed,
        commutator_residual=residual,
        is_minimal_success=constructed and rank == cls.expected_index,
        ordinary_untwisted_descent_on_original_rank=cls.ordinary_untwisted_descent_on_original_rank,
        lift_kind=cls.lift_kind,
        interpretation=cls.interpretation(rank),
    )


def toy_prediction_losses(q: int, exponent: int, rank: int) -> dict[str, float | bool | str]:
    """Tiny algebraic prediction proxy for the threshold.

    The loss is zero exactly when the algebraic relation is realized.  The
    branch score is zero because it stores sectors separately, so it is labeled
    extra capacity and is not a capacity-matched single-model result.
    """

    result = evaluate_rank_absorption(q, exponent, rank)
    lifted_loss = 0.0 if result.constructed_lift_success else 1.0
    return {
        "ordinary_rank_r_loss": lifted_loss,
        "lifted_rank_r_loss": lifted_loss,
        "branch_projective_loss": 0.0,
        "branch_extra_capacity": True,
        "prediction_note": "algebraic proxy only; no MNIST/CIFAR claim",
    }
