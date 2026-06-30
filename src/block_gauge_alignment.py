"""Block-orthogonal feature-space alignment diagnostics.

The maps in this module use the same row-vector convention as the other
alignment utilities: ``features_i @ G_ij`` approximates ``features_j``.
For ReLU MLPs these block rotations are feature-space diagnostics, not exact
parameter symmetries of the network.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Mapping

import numpy as np


IndexPair = tuple[int, int]


@dataclass(frozen=True)
class BlockAlignmentStats:
    pair: IndexPair
    block_size: int
    n_blocks: int
    used_remainder_block: bool
    mean_block_residual: float
    max_block_residual: float


@dataclass(frozen=True)
class BlockPartition:
    method: str
    block_size: int
    blocks: tuple[tuple[int, ...], ...]
    seed: int | None = None
    notes: str = ""

    @property
    def width(self) -> int:
        return sum(len(block) for block in self.blocks)

    def as_arrays(self) -> list[np.ndarray]:
        return [np.asarray(block, dtype=int) for block in self.blocks]

    def assignment_string(self) -> str:
        assignment = np.empty(self.width, dtype=int)
        for block_idx, block in enumerate(self.blocks):
            assignment[np.asarray(block, dtype=int)] = block_idx
        return ",".join(str(int(item)) for item in assignment)


def contiguous_blocks(width: int, block_size: int, allow_remainder: bool = True) -> list[np.ndarray]:
    """Return contiguous hidden-unit blocks as integer index arrays."""

    width = int(width)
    block_size = int(block_size)
    if width <= 0:
        raise ValueError("width must be positive")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    blocks: list[np.ndarray] = []
    start = 0
    while start < width:
        stop = min(start + block_size, width)
        if stop - start < block_size and not allow_remainder:
            break
        blocks.append(np.arange(start, stop, dtype=int))
        start = stop
    if not blocks:
        raise ValueError("no blocks were produced")
    return blocks


def make_contiguous_partition(width: int, block_size: int, allow_remainder: bool = True) -> BlockPartition:
    return BlockPartition(
        method="contiguous",
        block_size=int(block_size),
        blocks=tuple(tuple(int(item) for item in block) for block in contiguous_blocks(width, block_size, allow_remainder)),
        notes="contiguous hidden-unit blocks",
    )


def orthogonal_procrustes(
    source: np.ndarray,
    target: np.ndarray,
    center: bool = True,
) -> tuple[np.ndarray, float]:
    """Solve ``min_Q ||source Q - target||_F`` over orthogonal/unitary ``Q``."""

    src = np.asarray(source)
    tgt = np.asarray(target)
    if src.shape != tgt.shape or src.ndim != 2:
        raise ValueError("source and target must be rank-2 arrays with the same shape")
    src = src.astype(complex if np.iscomplexobj(src) or np.iscomplexobj(tgt) else float)
    tgt = tgt.astype(src.dtype, copy=False)
    if center:
        src = src - src.mean(axis=0, keepdims=True)
        tgt = tgt - tgt.mean(axis=0, keepdims=True)
    cross = src.conj().T @ tgt
    left, _singular, right_h = np.linalg.svd(cross, full_matrices=False)
    transform = left @ right_h
    denom = max(float(np.linalg.norm(src, ord="fro")), float(np.linalg.norm(tgt, ord="fro")), 1e-12)
    residual = float(np.linalg.norm(src @ transform - tgt, ord="fro") / denom)
    if not np.iscomplexobj(source) and not np.iscomplexobj(target):
        transform = transform.real
    return transform, residual


def estimate_block_orthogonal_alignments(
    pairwise_permutations: Mapping[IndexPair, np.ndarray],
    activations: Mapping[int, np.ndarray],
    n_models: int,
    width: int,
    block_size: int,
    *,
    allow_remainder: bool = True,
    center: bool = True,
) -> tuple[dict[IndexPair, np.ndarray], dict[IndexPair, BlockAlignmentStats]]:
    """Estimate block-diagonal orthogonal maps after neuron permutation matching.

    For each source block ``B`` and pair ``(i, j)``, the target columns are
    ``pairwise_permutations[(i, j)][B]``.  The resulting matrix has the
    Procrustes block in rows ``B`` and those target columns, so multiplying
    row-vector activations from model ``i`` by the matrix approximates model
    ``j`` activations in model ``j`` coordinates.
    """

    blocks = contiguous_blocks(width, block_size, allow_remainder=allow_remainder)
    used_remainder = any(len(block) != int(block_size) for block in blocks)
    matrices: dict[IndexPair, np.ndarray] = {}
    stats: dict[IndexPair, BlockAlignmentStats] = {}

    for i, j in product(range(n_models), repeat=2):
        pair = (i, j)
        if i == j:
            matrices[pair] = np.eye(width, dtype=complex)
            stats[pair] = BlockAlignmentStats(
                pair=pair,
                block_size=int(block_size),
                n_blocks=len(blocks),
                used_remainder_block=used_remainder,
                mean_block_residual=0.0,
                max_block_residual=0.0,
            )
            continue

        perm = np.asarray(pairwise_permutations[pair], dtype=int)
        if perm.shape != (width,):
            raise ValueError(f"permutation for pair {pair} has shape {perm.shape}, expected {(width,)}")
        source_features = np.asarray(activations[i])
        target_features = np.asarray(activations[j])
        matrix = np.zeros((width, width), dtype=complex)
        residuals = []
        for block in blocks:
            target_cols = perm[block]
            transform, residual = orthogonal_procrustes(
                source_features[:, block],
                target_features[:, target_cols],
                center=center,
            )
            matrix[np.ix_(block, target_cols)] = transform
            residuals.append(residual)
        matrices[pair] = matrix
        stats[pair] = BlockAlignmentStats(
            pair=pair,
            block_size=int(block_size),
            n_blocks=len(blocks),
            used_remainder_block=used_remainder,
            mean_block_residual=float(np.mean(residuals)) if residuals else float("nan"),
            max_block_residual=float(np.max(residuals)) if residuals else float("nan"),
        )

    return matrices, stats


def induce_model_blocks(
    reference_blocks: list[np.ndarray],
    reference_to_model_permutations: Mapping[int, np.ndarray],
) -> dict[int, list[np.ndarray]]:
    """Map reference-coordinate blocks into every model's local coordinates."""

    model_blocks: dict[int, list[np.ndarray]] = {}
    for model_idx, perm in reference_to_model_permutations.items():
        value = np.asarray(perm, dtype=int)
        model_blocks[int(model_idx)] = [value[np.asarray(block, dtype=int)] for block in reference_blocks]
    return model_blocks


def estimate_block_orthogonal_alignments_for_model_blocks(
    model_blocks: Mapping[int, list[np.ndarray]],
    activations: Mapping[int, np.ndarray],
    n_models: int,
    width: int,
    block_size: int,
    *,
    center: bool = True,
) -> tuple[dict[IndexPair, np.ndarray], dict[IndexPair, BlockAlignmentStats]]:
    """Estimate block maps when each model has an explicit block assignment."""

    matrices: dict[IndexPair, np.ndarray] = {}
    stats: dict[IndexPair, BlockAlignmentStats] = {}
    used_remainder = any(
        len(block) != int(block_size)
        for blocks in model_blocks.values()
        for block in blocks
    )
    n_blocks = len(next(iter(model_blocks.values())))
    for blocks in model_blocks.values():
        if len(blocks) != n_blocks:
            raise ValueError("all models must have the same number of blocks")

    for i, j in product(range(n_models), repeat=2):
        pair = (i, j)
        if i == j:
            matrices[pair] = np.eye(width, dtype=complex)
            stats[pair] = BlockAlignmentStats(
                pair=pair,
                block_size=int(block_size),
                n_blocks=n_blocks,
                used_remainder_block=used_remainder,
                mean_block_residual=0.0,
                max_block_residual=0.0,
            )
            continue
        source_features = np.asarray(activations[i])
        target_features = np.asarray(activations[j])
        matrix = np.zeros((width, width), dtype=complex)
        residuals = []
        for source_cols, target_cols in zip(model_blocks[i], model_blocks[j], strict=True):
            transform, residual = orthogonal_procrustes(
                source_features[:, source_cols],
                target_features[:, target_cols],
                center=center,
            )
            matrix[np.ix_(source_cols, target_cols)] = transform
            residuals.append(residual)
        matrices[pair] = matrix
        stats[pair] = BlockAlignmentStats(
            pair=pair,
            block_size=int(block_size),
            n_blocks=n_blocks,
            used_remainder_block=used_remainder,
            mean_block_residual=float(np.mean(residuals)) if residuals else float("nan"),
            max_block_residual=float(np.max(residuals)) if residuals else float("nan"),
        )
    return matrices, stats


def summarize_block_alignment_stats(stats: Mapping[IndexPair, BlockAlignmentStats]) -> dict[str, float | bool]:
    values = [item.mean_block_residual for item in stats.values() if item.pair[0] != item.pair[1]]
    maxima = [item.max_block_residual for item in stats.values() if item.pair[0] != item.pair[1]]
    return {
        "mean_pairwise_block_residual": float(np.mean(values)) if values else 0.0,
        "max_pairwise_block_residual": float(np.max(maxima)) if maxima else 0.0,
        "used_remainder_block": any(item.used_remainder_block for item in stats.values()),
    }
