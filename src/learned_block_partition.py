"""Deterministic learned hidden-unit block partitions."""

from __future__ import annotations

import numpy as np

from .block_gauge_alignment import BlockPartition, make_contiguous_partition


def _normalize_columns(matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix, dtype=float)
    arr = arr - arr.mean(axis=0, keepdims=True)
    return arr / np.maximum(np.linalg.norm(arr, axis=0, keepdims=True), 1e-12)


def _greedy_similarity_partition(
    similarity: np.ndarray,
    block_size: int,
    *,
    method: str,
    seed: int,
    allow_remainder: bool = True,
) -> BlockPartition:
    sim = np.asarray(similarity, dtype=float)
    if sim.ndim != 2 or sim.shape[0] != sim.shape[1]:
        raise ValueError("similarity must be a square matrix")
    width = sim.shape[0]
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    rng = np.random.default_rng(seed)
    jitter = rng.uniform(0.0, 1e-9, size=width)
    remaining: set[int] = set(range(width))
    blocks: list[tuple[int, ...]] = []
    while remaining:
        if len(remaining) < block_size and not allow_remainder:
            break
        candidates = np.asarray(sorted(remaining), dtype=int)
        row_scores = sim[candidates][:, candidates].sum(axis=1) + jitter[candidates]
        seed_unit = int(candidates[int(np.argmax(row_scores))])
        block = [seed_unit]
        remaining.remove(seed_unit)
        while remaining and len(block) < block_size:
            candidates = np.asarray(sorted(remaining), dtype=int)
            scores = sim[np.ix_(candidates, block)].mean(axis=1) + jitter[candidates]
            chosen = int(candidates[int(np.argmax(scores))])
            block.append(chosen)
            remaining.remove(chosen)
        blocks.append(tuple(block))
    if not blocks:
        raise ValueError("no blocks were produced")
    return BlockPartition(
        method=method,
        block_size=int(block_size),
        blocks=tuple(blocks),
        seed=int(seed),
        notes="greedy deterministic similarity clustering",
    )


def activation_correlation_partition(
    activations: np.ndarray,
    block_size: int,
    *,
    seed: int = 0,
    allow_remainder: bool = True,
) -> BlockPartition:
    """Cluster hidden units by absolute activation correlation."""

    normalized = _normalize_columns(np.asarray(activations, dtype=float))
    similarity = np.abs(normalized.T @ normalized)
    np.fill_diagonal(similarity, 1.0)
    return _greedy_similarity_partition(
        similarity,
        block_size,
        method="activation_correlation",
        seed=seed,
        allow_remainder=allow_remainder,
    )


def output_weight_similarity_partition(
    output_weights: np.ndarray,
    block_size: int,
    *,
    seed: int = 0,
    allow_remainder: bool = True,
) -> BlockPartition:
    """Cluster hidden units by cosine similarity of outgoing classifier weights."""

    weights = np.asarray(output_weights, dtype=float)
    if weights.ndim != 2:
        raise ValueError("output_weights must be a rank-2 array")
    normalized = weights / np.maximum(np.linalg.norm(weights, axis=0, keepdims=True), 1e-12)
    similarity = np.abs(normalized.T @ normalized)
    np.fill_diagonal(similarity, 1.0)
    return _greedy_similarity_partition(
        similarity,
        block_size,
        method="output_weight_similarity",
        seed=seed,
        allow_remainder=allow_remainder,
    )


def make_block_partition(
    method: str,
    width: int,
    block_size: int,
    *,
    activations: np.ndarray | None = None,
    output_weights: np.ndarray | None = None,
    seed: int = 0,
    allow_remainder: bool = True,
) -> BlockPartition:
    if method == "contiguous":
        return make_contiguous_partition(width, block_size, allow_remainder=allow_remainder)
    if method == "activation_correlation":
        if activations is None:
            raise ValueError("activation_correlation partition requires activations")
        return activation_correlation_partition(activations, block_size, seed=seed, allow_remainder=allow_remainder)
    if method == "output_weight_similarity":
        if output_weights is None:
            raise ValueError("output_weight_similarity partition requires output_weights")
        return output_weight_similarity_partition(output_weights, block_size, seed=seed, allow_remainder=allow_remainder)
    raise ValueError(f"unknown block partition method: {method}")
