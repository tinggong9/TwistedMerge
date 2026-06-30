"""Deterministic learned hidden-unit block partitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .block_gauge_alignment import BlockPartition, make_contiguous_partition


@dataclass(frozen=True)
class ValidationBlockSelection:
    partition: BlockPartition
    selected_name: str
    selected_score: float
    candidate_scores: dict[str, float]
    metric_source: str = "validation"
    used_test_metrics: bool = False


def _normalize_columns(matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix, dtype=float)
    arr = arr - arr.mean(axis=0, keepdims=True)
    return arr / np.maximum(np.linalg.norm(arr, axis=0, keepdims=True), 1e-12)


def _ordered_arrays(values: Mapping[int, np.ndarray] | Sequence[np.ndarray]) -> list[np.ndarray]:
    if isinstance(values, Mapping):
        return [np.asarray(values[key]) for key in sorted(values)]
    return [np.asarray(item) for item in values]


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


def global_activation_correlation(
    activations_by_model: Mapping[int, np.ndarray] | Sequence[np.ndarray],
) -> np.ndarray:
    """Average absolute hidden-unit activation correlations across models.

    The inputs must already be in a common hidden-unit coordinate system, for
    example after permutation synchronization.  This function returns only the
    validation/training-set similarity matrix; downstream selection decides how
    to use it without looking at test metrics.
    """

    arrays = _ordered_arrays(activations_by_model)
    if not arrays:
        raise ValueError("at least one activation matrix is required")
    width = arrays[0].shape[1]
    similarity = np.zeros((width, width), dtype=float)
    for activations in arrays:
        normalized = _normalize_columns(activations)
        if normalized.ndim != 2 or normalized.shape[1] != width:
            raise ValueError("all activation matrices must be rank-2 with the same width")
        similarity += np.abs(normalized.T @ normalized)
    similarity /= float(len(arrays))
    np.fill_diagonal(similarity, 1.0)
    return similarity


def global_output_weight_similarity(
    output_weights_by_model: Mapping[int, np.ndarray] | Sequence[np.ndarray],
) -> np.ndarray:
    """Average absolute cosine similarity of outgoing classifier columns."""

    arrays = _ordered_arrays(output_weights_by_model)
    if not arrays:
        raise ValueError("at least one output-weight matrix is required")
    width = arrays[0].shape[1]
    similarity = np.zeros((width, width), dtype=float)
    for weights in arrays:
        arr = np.asarray(weights, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != width:
            raise ValueError("all output-weight matrices must be rank-2 with the same hidden width")
        normalized = arr / np.maximum(np.linalg.norm(arr, axis=0, keepdims=True), 1e-12)
        similarity += np.abs(normalized.T @ normalized)
    similarity /= float(len(arrays))
    np.fill_diagonal(similarity, 1.0)
    return similarity


def residual_greedy_blocks(
    residual_or_similarity: np.ndarray,
    block_size: int,
    *,
    seed: int = 0,
    larger_is_better: bool = False,
    allow_remainder: bool = True,
    method: str = "residual_greedy",
) -> BlockPartition:
    """Cluster units greedily from a residual or similarity matrix.

    By default low residuals mean compatible units, so the matrix is negated
    before greedy clustering.  Set ``larger_is_better=True`` when the input is
    already a similarity matrix such as ``global_activation_correlation``.
    """

    arr = np.asarray(residual_or_similarity, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("residual_or_similarity must be a square matrix")
    similarity = arr.copy() if larger_is_better else -arr
    np.fill_diagonal(similarity, float(np.max(similarity)) + 1.0 if similarity.size else 1.0)
    return _greedy_similarity_partition(
        similarity,
        block_size,
        method=method,
        seed=seed,
        allow_remainder=allow_remainder,
    )


def validation_selected_blocks(
    candidates: Mapping[str, BlockPartition] | Sequence[BlockPartition],
    validation_scores: Mapping[str, float] | Sequence[float],
    *,
    metric_source: str = "validation",
    prefer: str = "min",
) -> ValidationBlockSelection:
    """Select a block partition using validation-only diagnostics.

    ``prefer="min"`` is for residual/loss-like scores; ``prefer="max"`` is for
    accuracy-like scores.  The function rejects metric-source labels mentioning
    test data so reports can assert the selector did not leak test metrics.
    """

    if "test" in metric_source.lower():
        raise ValueError("validation_selected_blocks must not use test metrics")
    if isinstance(candidates, Mapping):
        candidate_map = dict(candidates)
    else:
        candidate_map = {f"candidate_{idx}": item for idx, item in enumerate(candidates)}
    if isinstance(validation_scores, Mapping):
        score_map = {str(key): float(value) for key, value in validation_scores.items()}
    else:
        score_map = {f"candidate_{idx}": float(value) for idx, value in enumerate(validation_scores)}
    if not candidate_map:
        raise ValueError("at least one candidate partition is required")
    missing = set(candidate_map) - set(score_map)
    if missing:
        raise ValueError(f"missing validation scores for candidates: {sorted(missing)}")
    if prefer not in {"min", "max"}:
        raise ValueError("prefer must be 'min' or 'max'")
    key_fn = min if prefer == "min" else max
    selected_name = key_fn(candidate_map, key=lambda name: score_map[name])
    return ValidationBlockSelection(
        partition=candidate_map[selected_name],
        selected_name=str(selected_name),
        selected_score=float(score_map[selected_name]),
        candidate_scores={str(name): float(score_map[name]) for name in candidate_map},
        metric_source=metric_source,
        used_test_metrics=False,
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
