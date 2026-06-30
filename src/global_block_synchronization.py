"""Global synchronization for block-orthogonal feature-space gauges."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
from typing import Mapping

import numpy as np


IndexPair = tuple[int, int]
Triple = tuple[int, int, int]


@dataclass(frozen=True)
class GlobalBlockSyncResult:
    synchronized_maps: dict[IndexPair, np.ndarray]
    gauges: dict[tuple[int, int], np.ndarray]
    connection_residual: float
    max_connection_residual: float
    feature_alignment_residual: float | None
    method: str = "spectral_connection_laplacian"


def project_to_orthogonal(matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix)
    left, _singular, right_h = np.linalg.svd(arr, full_matrices=False)
    projected = left @ right_h
    if not np.iscomplexobj(matrix):
        projected = projected.real
    return projected


def default_triples(n_models: int) -> list[Triple]:
    return list(combinations(range(n_models), 3))


def matrix_centrality_score(matrix: np.ndarray) -> tuple[float, complex]:
    arr = np.asarray(matrix, dtype=complex)
    scalar = complex(np.trace(arr) / max(arr.shape[0], 1))
    target = scalar * np.eye(arr.shape[0], dtype=complex)
    denom = max(float(np.linalg.norm(np.eye(arr.shape[0]), ord="fro")), 1e-12)
    return float(np.linalg.norm(arr - target, ord="fro") / denom), scalar


def triangle_defects(maps: Mapping[IndexPair, np.ndarray], triples: list[Triple]) -> dict[Triple, np.ndarray]:
    return {
        triple: maps[(triple[0], triple[1])] @ maps[(triple[1], triple[2])] @ maps[(triple[2], triple[0])]
        for triple in triples
    }


def cycle_score(defects: Mapping[Triple, np.ndarray]) -> float:
    scores = []
    for matrix in defects.values():
        eye = np.eye(matrix.shape[0], dtype=complex)
        scores.append(float(np.linalg.norm(matrix - eye, ord="fro") / max(float(np.linalg.norm(eye, ord="fro")), 1e-12)))
    return float(np.mean(scores)) if scores else 0.0


def mean_centrality(defects: Mapping[Triple, np.ndarray]) -> float:
    values = [matrix_centrality_score(matrix)[0] for matrix in defects.values()]
    return float(np.mean(values)) if values else 0.0


def feature_alignment_residual_for_maps(
    maps: Mapping[IndexPair, np.ndarray],
    model_blocks: Mapping[int, list[np.ndarray]],
    activations: Mapping[int, np.ndarray],
    n_models: int,
) -> float:
    residuals = []
    for i, j in product(range(n_models), repeat=2):
        if i == j:
            continue
        source = np.asarray(activations[i])
        target = np.asarray(activations[j])
        matrix = np.asarray(maps[(i, j)])
        for source_cols, target_cols in zip(model_blocks[i], model_blocks[j], strict=True):
            predicted = source[:, source_cols] @ matrix[np.ix_(source_cols, target_cols)]
            observed = target[:, target_cols]
            observed = observed - observed.mean(axis=0, keepdims=True)
            predicted = predicted - predicted.mean(axis=0, keepdims=True)
            denom = max(float(np.linalg.norm(predicted, ord="fro")), float(np.linalg.norm(observed, ord="fro")), 1e-12)
            residuals.append(float(np.linalg.norm(predicted - observed, ord="fro") / denom))
    return float(np.mean(residuals)) if residuals else 0.0


def global_block_spectral_synchronization(
    pairwise_maps: Mapping[IndexPair, np.ndarray],
    model_blocks: Mapping[int, list[np.ndarray]],
    n_models: int,
    width: int,
    *,
    activations: Mapping[int, np.ndarray] | None = None,
) -> GlobalBlockSyncResult:
    """Synchronize pairwise block maps into globally cycle-consistent maps.

    The observed row-vector maps should satisfy ``G_ij ~= Q_i Q_j^T`` on each
    block when a global block gauge exists.  The returned synchronized maps are
    built from the recovered per-model gauges and therefore have trivial
    triangle cycles by construction.  The connection residual records whether
    those maps actually explain the observed pairwise data.
    """

    n_blocks = len(next(iter(model_blocks.values())))
    for blocks in model_blocks.values():
        if len(blocks) != n_blocks:
            raise ValueError("all models must have the same number of blocks")

    gauges: dict[tuple[int, int], np.ndarray] = {}
    for block_idx in range(n_blocks):
        block_dim = len(model_blocks[0][block_idx])
        if any(len(model_blocks[i][block_idx]) != block_dim for i in range(n_models)):
            raise ValueError("spectral block synchronization requires matching block dimensions")
        connection = np.zeros((n_models * block_dim, n_models * block_dim), dtype=complex)
        for i, j in product(range(n_models), repeat=2):
            rows = model_blocks[i][block_idx]
            cols = model_blocks[j][block_idx]
            block = np.asarray(pairwise_maps[(i, j)])[np.ix_(rows, cols)]
            connection[i * block_dim : (i + 1) * block_dim, j * block_dim : (j + 1) * block_dim] = block
        connection = 0.5 * (connection + connection.conj().T)
        eigenvalues, eigenvectors = np.linalg.eigh(connection)
        top = eigenvectors[:, np.argsort(eigenvalues)[-block_dim:]]
        for model_idx in range(n_models):
            raw = top[model_idx * block_dim : (model_idx + 1) * block_dim, :]
            gauges[(model_idx, block_idx)] = project_to_orthogonal(raw)

    synchronized_maps: dict[IndexPair, np.ndarray] = {}
    residuals = []
    for i, j in product(range(n_models), repeat=2):
        matrix = np.zeros((width, width), dtype=complex)
        for block_idx in range(n_blocks):
            rows = model_blocks[i][block_idx]
            cols = model_blocks[j][block_idx]
            qi = gauges[(i, block_idx)]
            qj = gauges[(j, block_idx)]
            block = qi @ qj.conj().T
            matrix[np.ix_(rows, cols)] = block
            observed = np.asarray(pairwise_maps[(i, j)])[np.ix_(rows, cols)]
            residuals.append(float(np.linalg.norm(observed - block, ord="fro") / max(np.sqrt(block.shape[0]), 1e-12)))
        synchronized_maps[(i, j)] = matrix

    feature_residual = (
        feature_alignment_residual_for_maps(synchronized_maps, model_blocks, activations, n_models)
        if activations is not None
        else None
    )
    return GlobalBlockSyncResult(
        synchronized_maps=synchronized_maps,
        gauges=gauges,
        connection_residual=float(np.mean(residuals)) if residuals else 0.0,
        max_connection_residual=float(np.max(residuals)) if residuals else 0.0,
        feature_alignment_residual=feature_residual,
    )


def global_sync_accepted(result: GlobalBlockSyncResult, tolerance: float = 0.15) -> bool:
    return bool(result.connection_residual <= tolerance)
