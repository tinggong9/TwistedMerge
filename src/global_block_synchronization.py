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
    initial_connection_residual: float | None = None
    initial_feature_alignment_residual: float | None = None
    objective_value: float | None = None
    initial_objective_value: float | None = None
    n_iterations: int = 0
    converged: bool = True
    objective_history: tuple[float, ...] = ()


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


def build_maps_from_block_gauges(
    gauges: Mapping[tuple[int, int], np.ndarray],
    model_blocks: Mapping[int, list[np.ndarray]],
    n_models: int,
    width: int,
) -> dict[IndexPair, np.ndarray]:
    n_blocks = len(next(iter(model_blocks.values())))
    maps: dict[IndexPair, np.ndarray] = {}
    for i, j in product(range(n_models), repeat=2):
        matrix = np.zeros((width, width), dtype=complex)
        for block_idx in range(n_blocks):
            rows = model_blocks[i][block_idx]
            cols = model_blocks[j][block_idx]
            qi = np.asarray(gauges[(i, block_idx)])
            qj = np.asarray(gauges[(j, block_idx)])
            matrix[np.ix_(rows, cols)] = qi @ qj.conj().T
        maps[(i, j)] = matrix
    return maps


def connection_residual_for_maps(
    observed_maps: Mapping[IndexPair, np.ndarray],
    synchronized_maps: Mapping[IndexPair, np.ndarray],
    model_blocks: Mapping[int, list[np.ndarray]],
    n_models: int,
) -> tuple[float, float]:
    residuals = []
    n_blocks = len(next(iter(model_blocks.values())))
    for i, j in product(range(n_models), repeat=2):
        for block_idx in range(n_blocks):
            rows = model_blocks[i][block_idx]
            cols = model_blocks[j][block_idx]
            observed = np.asarray(observed_maps[(i, j)])[np.ix_(rows, cols)]
            predicted = np.asarray(synchronized_maps[(i, j)])[np.ix_(rows, cols)]
            residuals.append(
                float(np.linalg.norm(observed - predicted, ord="fro") / max(np.sqrt(predicted.shape[0]), 1e-12))
            )
    return (
        float(np.mean(residuals)) if residuals else 0.0,
        float(np.max(residuals)) if residuals else 0.0,
    )


def block_sync_objective(
    observed_maps: Mapping[IndexPair, np.ndarray],
    synchronized_maps: Mapping[IndexPair, np.ndarray],
    model_blocks: Mapping[int, list[np.ndarray]],
    n_models: int,
    *,
    activations: Mapping[int, np.ndarray] | None = None,
    lambda_feature: float = 0.0,
    lambda_reg: float = 0.0,
    gauges: Mapping[tuple[int, int], np.ndarray] | None = None,
) -> float:
    connection_mean, _connection_max = connection_residual_for_maps(
        observed_maps,
        synchronized_maps,
        model_blocks,
        n_models,
    )
    objective = connection_mean**2
    if activations is not None and lambda_feature > 0.0:
        feature = feature_alignment_residual_for_maps(synchronized_maps, model_blocks, activations, n_models)
        objective += float(lambda_feature) * feature**2
    if gauges is not None and lambda_reg > 0.0:
        reg_values = []
        for matrix in gauges.values():
            dim = matrix.shape[0]
            reg_values.append(float(np.linalg.norm(matrix.conj().T @ matrix - np.eye(dim), ord="fro") ** 2))
        if reg_values:
            objective += float(lambda_reg) * float(np.mean(reg_values))
    return float(objective)


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

    synchronized_maps = build_maps_from_block_gauges(gauges, model_blocks, n_models, width)
    connection_residual, max_connection_residual = connection_residual_for_maps(
        pairwise_maps,
        synchronized_maps,
        model_blocks,
        n_models,
    )

    feature_residual = (
        feature_alignment_residual_for_maps(synchronized_maps, model_blocks, activations, n_models)
        if activations is not None
        else None
    )
    return GlobalBlockSyncResult(
        synchronized_maps=synchronized_maps,
        gauges=gauges,
        connection_residual=connection_residual,
        max_connection_residual=max_connection_residual,
        feature_alignment_residual=feature_residual,
    )


def spectral_initialization(
    pairwise_maps: Mapping[IndexPair, np.ndarray],
    model_blocks: Mapping[int, list[np.ndarray]],
    n_models: int,
    width: int,
    *,
    activations: Mapping[int, np.ndarray] | None = None,
) -> GlobalBlockSyncResult:
    return global_block_spectral_synchronization(
        pairwise_maps,
        model_blocks,
        n_models,
        width,
        activations=activations,
    )


def _copy_gauges(gauges: Mapping[tuple[int, int], np.ndarray]) -> dict[tuple[int, int], np.ndarray]:
    return {key: np.asarray(value).copy() for key, value in gauges.items()}


def _random_orthogonal(dim: int, rng: np.random.Generator) -> np.ndarray:
    matrix = rng.normal(size=(dim, dim))
    q, r = np.linalg.qr(matrix)
    signs = np.sign(np.diag(r))
    signs[signs == 0.0] = 1.0
    return q * signs


def random_block_gauges(
    model_blocks: Mapping[int, list[np.ndarray]],
    n_models: int,
    *,
    seed: int,
) -> dict[tuple[int, int], np.ndarray]:
    rng = np.random.default_rng(seed)
    n_blocks = len(next(iter(model_blocks.values())))
    gauges: dict[tuple[int, int], np.ndarray] = {}
    for model_idx in range(n_models):
        for block_idx in range(n_blocks):
            gauges[(model_idx, block_idx)] = _random_orthogonal(len(model_blocks[model_idx][block_idx]), rng)
    return gauges


def alternating_block_procrustes_refinement(
    pairwise_maps: Mapping[IndexPair, np.ndarray],
    model_blocks: Mapping[int, list[np.ndarray]],
    n_models: int,
    width: int,
    *,
    activations: Mapping[int, np.ndarray] | None = None,
    initial_gauges: Mapping[tuple[int, int], np.ndarray] | None = None,
    lambda_feature: float = 0.25,
    lambda_reg: float = 0.0,
    max_iters: int = 25,
    tolerance: float = 1e-5,
) -> GlobalBlockSyncResult:
    """Refine block gauges by coordinate Procrustes updates.

    The update optimizes the connection residual and, when activations are
    supplied, adds a feature-alignment term.  The returned gauges are the best
    objective value encountered, so refinement cannot silently degrade the
    initialization.
    """

    spectral = spectral_initialization(pairwise_maps, model_blocks, n_models, width, activations=activations)
    gauges = _copy_gauges(initial_gauges if initial_gauges is not None else spectral.gauges)
    maps = build_maps_from_block_gauges(gauges, model_blocks, n_models, width)
    objective = block_sync_objective(
        pairwise_maps,
        maps,
        model_blocks,
        n_models,
        activations=activations,
        lambda_feature=lambda_feature,
        lambda_reg=lambda_reg,
        gauges=gauges,
    )
    best_objective = objective
    best_gauges = _copy_gauges(gauges)
    history = [objective]
    converged = False
    n_blocks = len(next(iter(model_blocks.values())))

    for iteration in range(1, int(max_iters) + 1):
        previous_objective = objective
        for model_idx in range(n_models):
            for block_idx in range(n_blocks):
                rows_i = model_blocks[model_idx][block_idx]
                update = np.zeros_like(gauges[(model_idx, block_idx)], dtype=complex)
                xi = np.asarray(activations[model_idx])[:, rows_i] if activations is not None else None
                for other_idx in range(n_models):
                    if other_idx == model_idx:
                        continue
                    rows_j = model_blocks[other_idx][block_idx]
                    qj = gauges[(other_idx, block_idx)]
                    gij = np.asarray(pairwise_maps[(model_idx, other_idx)])[np.ix_(rows_i, rows_j)]
                    gji = np.asarray(pairwise_maps[(other_idx, model_idx)])[np.ix_(rows_j, rows_i)]
                    update += (gij + gji.conj().T) @ qj
                    if activations is not None and lambda_feature > 0.0:
                        xj = np.asarray(activations[other_idx])[:, rows_j]
                        xi_center = xi - xi.mean(axis=0, keepdims=True)
                        xj_center = xj - xj.mean(axis=0, keepdims=True)
                        denom = max(float(np.linalg.norm(xi_center, ord="fro") * np.linalg.norm(xj_center, ord="fro")), 1e-12)
                        update += float(lambda_feature) * ((xi_center.conj().T @ xj_center) / denom) @ qj
                gauges[(model_idx, block_idx)] = project_to_orthogonal(update)

        maps = build_maps_from_block_gauges(gauges, model_blocks, n_models, width)
        objective = block_sync_objective(
            pairwise_maps,
            maps,
            model_blocks,
            n_models,
            activations=activations,
            lambda_feature=lambda_feature,
            lambda_reg=lambda_reg,
            gauges=gauges,
        )
        history.append(objective)
        if objective < best_objective:
            best_objective = objective
            best_gauges = _copy_gauges(gauges)
        improvement = previous_objective - objective
        if abs(improvement) <= float(tolerance) * max(abs(previous_objective), 1e-12):
            converged = True
            break

    best_maps = build_maps_from_block_gauges(best_gauges, model_blocks, n_models, width)
    connection_residual, max_connection_residual = connection_residual_for_maps(
        pairwise_maps,
        best_maps,
        model_blocks,
        n_models,
    )
    feature_residual = (
        feature_alignment_residual_for_maps(best_maps, model_blocks, activations, n_models)
        if activations is not None
        else None
    )
    return GlobalBlockSyncResult(
        synchronized_maps=best_maps,
        gauges=best_gauges,
        connection_residual=connection_residual,
        max_connection_residual=max_connection_residual,
        feature_alignment_residual=feature_residual,
        method="alternating_block_procrustes_refinement",
        initial_connection_residual=spectral.connection_residual,
        initial_feature_alignment_residual=spectral.feature_alignment_residual,
        objective_value=best_objective,
        initial_objective_value=history[0],
        n_iterations=len(history) - 1,
        converged=converged,
        objective_history=tuple(float(item) for item in history),
    )


def residual_optimized_global_block_sync(
    pairwise_maps: Mapping[IndexPair, np.ndarray],
    model_blocks: Mapping[int, list[np.ndarray]],
    n_models: int,
    width: int,
    *,
    activations: Mapping[int, np.ndarray] | None = None,
    lambda_feature: float = 0.25,
    lambda_reg: float = 0.0,
    max_iters: int = 25,
    tolerance: float = 1e-5,
    n_restarts: int = 4,
    seed: int = 0,
) -> GlobalBlockSyncResult:
    spectral = spectral_initialization(pairwise_maps, model_blocks, n_models, width, activations=activations)
    candidates = [
        alternating_block_procrustes_refinement(
            pairwise_maps,
            model_blocks,
            n_models,
            width,
            activations=activations,
            initial_gauges=spectral.gauges,
            lambda_feature=lambda_feature,
            lambda_reg=lambda_reg,
            max_iters=max_iters,
            tolerance=tolerance,
        )
    ]
    for restart_idx in range(max(int(n_restarts), 0)):
        candidates.append(
            alternating_block_procrustes_refinement(
                pairwise_maps,
                model_blocks,
                n_models,
                width,
                activations=activations,
                initial_gauges=random_block_gauges(
                    model_blocks,
                    n_models,
                    seed=int(seed) + 1009 * (restart_idx + 1),
                ),
                lambda_feature=lambda_feature,
                lambda_reg=lambda_reg,
                max_iters=max_iters,
                tolerance=tolerance,
            )
        )

    best = min(
        candidates,
        key=lambda result: (
            float(result.objective_value if result.objective_value is not None else result.connection_residual),
            float(result.connection_residual),
        ),
    )
    return GlobalBlockSyncResult(
        synchronized_maps=best.synchronized_maps,
        gauges=best.gauges,
        connection_residual=best.connection_residual,
        max_connection_residual=best.max_connection_residual,
        feature_alignment_residual=best.feature_alignment_residual,
        method="residual_optimized_global_block_sync",
        initial_connection_residual=spectral.connection_residual,
        initial_feature_alignment_residual=spectral.feature_alignment_residual,
        objective_value=best.objective_value,
        initial_objective_value=block_sync_objective(
            pairwise_maps,
            spectral.synchronized_maps,
            model_blocks,
            n_models,
            activations=activations,
            lambda_feature=lambda_feature,
            lambda_reg=lambda_reg,
            gauges=spectral.gauges,
        ),
        n_iterations=best.n_iterations,
        converged=best.converged,
        objective_history=best.objective_history,
    )


def global_sync_accepted(result: GlobalBlockSyncResult, tolerance: float = 0.15) -> bool:
    return bool(result.connection_residual <= tolerance)
