"""Denoising and synchronization helpers for learned time-frequency charts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .nearest_heisenberg_projection import (
    HeisenbergProjectionResult,
    project_to_nearest_finite_heisenberg,
)
from .period_index_mining import project_to_nearest_unitary
from .time_frequency_benchmark import (
    PairedChartSplit,
    PairedTimeFrequencyChartDataset,
    complex_to_real_block_matrix,
    time_frequency_generator_chart_names,
)
from .time_frequency_learned_charts import (
    complex_linearity_residual,
    real_block_matrix_to_complex,
    relative_residual,
    ridge_linear_map,
)


RIDGE_GRID = (1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2)
DENOISING_METHODS = (
    "raw_least_squares",
    "ridge_least_squares",
    "nearest_unitary_projection",
    "complex_unitary_projection",
    "global_chart_synchronization",
    "unitary_global_chart_synchronization",
    "nearest_heisenberg_projection",
    "unitary_then_heisenberg_projection",
    "global_sync_then_heisenberg_projection",
)


@dataclass(frozen=True)
class DenoisedChartRecovery:
    denoising_method: str
    d: int
    k: int
    real_dimension: int
    chart_count: int
    transition_maps_raw: dict[str, np.ndarray]
    transition_maps_denoised: dict[str, np.ndarray]
    candidate_generators: dict[str, np.ndarray]
    learned_operator_error_mean_raw: float
    learned_operator_error_mean_denoised: float
    learned_operator_error_max_raw: float
    learned_operator_error_max_denoised: float
    pair_reconstruction_residual_train_raw: float
    pair_reconstruction_residual_test_raw: float
    pair_reconstruction_residual_train_denoised: float
    pair_reconstruction_residual_test_denoised: float
    global_sync_residual: float
    unitary_projection_residual: float
    complex_structure_residual: float
    selected_ridge: float | None = None
    heisenberg_projection: HeisenbergProjectionResult | None = None
    projection_residual: float = float("nan")
    projection_residual_threshold: float | None = None
    projection_accepted: bool | None = None
    commutator_residual_before_projection: float = float("nan")
    commutator_residual_after_projection: float = float("nan")
    exponent_matrix_residual: float = float("nan")
    heisenberg_projection_decision: str | None = None
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class PairwiseSynchronization:
    chart_names: tuple[str, ...]
    gauges: dict[str, np.ndarray]
    synchronized_pairwise: dict[tuple[str, str], np.ndarray]
    sync_residual: float
    unitary_projection_residual: float
    complex_structure_residual: float
    cycle_residual_before: float
    cycle_residual_after: float


def _ordered_rows(split: PairedChartSplit, chart_name: str) -> np.ndarray:
    rows = split.chart_rows(chart_name)
    order = np.argsort(split.chart_sample_ids(chart_name), kind="stable")
    return rows[order]


def _candidate_names(dataset: PairedTimeFrequencyChartDataset, names: tuple[str, ...] | None) -> tuple[str, ...]:
    requested = names or time_frequency_generator_chart_names(dataset.k)
    return tuple(name for name in requested if name in dataset.chart_names)


def _identity_transitions(
    dataset: PairedTimeFrequencyChartDataset,
    *,
    ridge: float,
    generator_names: tuple[str, ...] | None = None,
) -> dict[str, np.ndarray]:
    names = _candidate_names(dataset, generator_names)
    source = _ordered_rows(dataset.train, "I")
    return {
        name: ridge_linear_map(source, _ordered_rows(dataset.train, name), ridge=ridge)
        for name in names
    }


def _pairwise_transitions(
    dataset: PairedTimeFrequencyChartDataset,
    *,
    ridge: float,
) -> dict[tuple[str, str], np.ndarray]:
    transitions: dict[tuple[str, str], np.ndarray] = {}
    for target_name in dataset.chart_names:
        target = _ordered_rows(dataset.train, target_name)
        for source_name in dataset.chart_names:
            source = _ordered_rows(dataset.train, source_name)
            transitions[(target_name, source_name)] = ridge_linear_map(source, target, ridge=ridge)
    return transitions


def _mean_identity_residual(
    split: PairedChartSplit,
    transitions: Mapping[str, np.ndarray],
) -> float:
    source = _ordered_rows(split, "I")
    residuals = []
    for name, matrix in transitions.items():
        target = _ordered_rows(split, name)
        predicted = source @ np.asarray(matrix, dtype=float).T
        residuals.append(relative_residual(predicted, target))
    return float(np.mean(residuals)) if residuals else float("nan")


def _operator_errors(
    dataset: PairedTimeFrequencyChartDataset,
    transitions: Mapping[str, np.ndarray],
) -> tuple[float, float]:
    errors = []
    for name, matrix in transitions.items():
        known = complex_to_real_block_matrix(dataset.chart_operators[name])
        errors.append(relative_residual(np.asarray(matrix, dtype=float), known))
    if not errors:
        return float("nan"), float("nan")
    return float(np.mean(errors)), float(np.max(errors))


def _real_nearest_orthogonal(matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("matrix must be square")
    left, _singular_values, right = np.linalg.svd(arr)
    return (left @ right).astype(float)


def nearest_orthogonal_projection(matrix: np.ndarray) -> np.ndarray:
    """Project a real square matrix to the nearest orthogonal matrix."""

    return _real_nearest_orthogonal(matrix)


def _complex_unitary_projection_from_real(matrix: np.ndarray) -> np.ndarray:
    return project_to_nearest_unitary(real_block_matrix_to_complex(matrix))


def _mean_projection_residual(raw: Mapping[str, np.ndarray], denoised: Mapping[str, np.ndarray]) -> float:
    residuals = [relative_residual(np.asarray(denoised[name]), np.asarray(raw[name])) for name in raw]
    return float(np.mean(residuals)) if residuals else float("nan")


def _mean_complex_structure_residual(transitions: Mapping[str, np.ndarray]) -> float:
    residuals = [complex_linearity_residual(matrix) for matrix in transitions.values()]
    return float(np.mean(residuals)) if residuals else float("nan")


def _candidate_generators_from_real(transitions: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {name: real_block_matrix_to_complex(matrix) for name, matrix in transitions.items()}


def _pack_recovery(
    *,
    method: str,
    dataset: PairedTimeFrequencyChartDataset,
    raw: dict[str, np.ndarray],
    denoised: dict[str, np.ndarray],
    candidates: dict[str, np.ndarray],
    global_sync_residual: float = float("nan"),
    unitary_projection_residual: float = float("nan"),
    complex_structure_residual: float | None = None,
    selected_ridge: float | None = None,
    heisenberg_projection: HeisenbergProjectionResult | None = None,
    notes: tuple[str, ...] = (),
) -> DenoisedChartRecovery:
    raw_mean, raw_max = _operator_errors(dataset, raw)
    denoised_mean, denoised_max = _operator_errors(dataset, denoised)
    return DenoisedChartRecovery(
        denoising_method=method,
        d=dataset.d,
        k=dataset.k,
        real_dimension=dataset.dimension_real,
        chart_count=dataset.chart_count,
        transition_maps_raw=raw,
        transition_maps_denoised=denoised,
        candidate_generators=candidates,
        learned_operator_error_mean_raw=raw_mean,
        learned_operator_error_mean_denoised=denoised_mean,
        learned_operator_error_max_raw=raw_max,
        learned_operator_error_max_denoised=denoised_max,
        pair_reconstruction_residual_train_raw=_mean_identity_residual(dataset.train, raw),
        pair_reconstruction_residual_test_raw=_mean_identity_residual(dataset.test, raw),
        pair_reconstruction_residual_train_denoised=_mean_identity_residual(dataset.train, denoised),
        pair_reconstruction_residual_test_denoised=_mean_identity_residual(dataset.test, denoised),
        global_sync_residual=float(global_sync_residual),
        unitary_projection_residual=float(unitary_projection_residual),
        complex_structure_residual=(
            _mean_complex_structure_residual(denoised)
            if complex_structure_residual is None
            else float(complex_structure_residual)
        ),
        selected_ridge=selected_ridge,
        heisenberg_projection=heisenberg_projection,
        projection_residual=(
            float("nan") if heisenberg_projection is None else heisenberg_projection.projection_residual
        ),
        projection_residual_threshold=(
            None if heisenberg_projection is None else heisenberg_projection.projection_residual_threshold
        ),
        projection_accepted=None if heisenberg_projection is None else heisenberg_projection.projection_accepted,
        commutator_residual_before_projection=(
            float("nan") if heisenberg_projection is None else heisenberg_projection.commutator_residual_before
        ),
        commutator_residual_after_projection=(
            float("nan") if heisenberg_projection is None else heisenberg_projection.commutator_residual_after
        ),
        exponent_matrix_residual=(
            float("nan") if heisenberg_projection is None else heisenberg_projection.exponent_matrix_residual
        ),
        heisenberg_projection_decision=None if heisenberg_projection is None else heisenberg_projection.decision,
        notes=notes,
    )


def fit_raw_least_squares_denoising(
    dataset: PairedTimeFrequencyChartDataset,
    *,
    ridge: float = 1e-8,
    generator_names: tuple[str, ...] | None = None,
) -> DenoisedChartRecovery:
    raw = _identity_transitions(dataset, ridge=ridge, generator_names=generator_names)
    return _pack_recovery(
        method="raw_least_squares",
        dataset=dataset,
        raw=raw,
        denoised=raw,
        candidates=_candidate_generators_from_real(raw),
        selected_ridge=float(ridge),
        notes=("baseline ridge least-squares maps from identity chart to generator charts",),
    )


def select_ridge_by_validation_residual(
    dataset: PairedTimeFrequencyChartDataset,
    *,
    ridge_grid: tuple[float, ...] = RIDGE_GRID,
    generator_names: tuple[str, ...] | None = None,
) -> tuple[float, dict[str, np.ndarray], float]:
    if not ridge_grid:
        raise ValueError("ridge_grid must be nonempty")
    best: tuple[float, float, dict[str, np.ndarray]] | None = None
    for ridge in ridge_grid:
        transitions = _identity_transitions(dataset, ridge=float(ridge), generator_names=generator_names)
        validation_residual = _mean_identity_residual(dataset.validation, transitions)
        key = (float(validation_residual), float(ridge))
        if best is None or key < (best[0], best[1]):
            best = (float(validation_residual), float(ridge), transitions)
    assert best is not None
    return best[1], best[2], best[0]


def fit_ridge_least_squares_denoising(
    dataset: PairedTimeFrequencyChartDataset,
    *,
    ridge_grid: tuple[float, ...] = RIDGE_GRID,
    baseline_ridge: float = 1e-8,
    generator_names: tuple[str, ...] | None = None,
) -> DenoisedChartRecovery:
    raw = _identity_transitions(dataset, ridge=baseline_ridge, generator_names=generator_names)
    best_ridge, denoised, validation_residual = select_ridge_by_validation_residual(
        dataset,
        ridge_grid=ridge_grid,
        generator_names=generator_names,
    )
    return _pack_recovery(
        method="ridge_least_squares",
        dataset=dataset,
        raw=raw,
        denoised=denoised,
        candidates=_candidate_generators_from_real(denoised),
        selected_ridge=best_ridge,
        notes=(f"ridge selected by validation pair residual {validation_residual:.6g}",),
    )


def fit_nearest_unitary_projection(
    dataset: PairedTimeFrequencyChartDataset,
    *,
    ridge: float = 1e-8,
    generator_names: tuple[str, ...] | None = None,
) -> DenoisedChartRecovery:
    raw = _identity_transitions(dataset, ridge=ridge, generator_names=generator_names)
    denoised = {name: _real_nearest_orthogonal(matrix) for name, matrix in raw.items()}
    return _pack_recovery(
        method="nearest_unitary_projection",
        dataset=dataset,
        raw=raw,
        denoised=denoised,
        candidates=_candidate_generators_from_real(denoised),
        unitary_projection_residual=_mean_projection_residual(raw, denoised),
        selected_ridge=float(ridge),
        notes=("real polar projection to nearest orthogonal map before detector extraction",),
    )


def fit_complex_unitary_projection(
    dataset: PairedTimeFrequencyChartDataset,
    *,
    ridge: float = 1e-8,
    generator_names: tuple[str, ...] | None = None,
) -> DenoisedChartRecovery:
    raw = _identity_transitions(dataset, ridge=ridge, generator_names=generator_names)
    projected_complex = {
        name: _complex_unitary_projection_from_real(matrix)
        for name, matrix in raw.items()
    }
    denoised = {name: complex_to_real_block_matrix(matrix) for name, matrix in projected_complex.items()}
    projection_residuals = [
        relative_residual(projected_complex[name], real_block_matrix_to_complex(raw[name]))
        for name in raw
    ]
    return _pack_recovery(
        method="complex_unitary_projection",
        dataset=dataset,
        raw=raw,
        denoised=denoised,
        candidates=projected_complex,
        unitary_projection_residual=float(np.mean(projection_residuals)) if projection_residuals else float("nan"),
        complex_structure_residual=0.0,
        selected_ridge=float(ridge),
        notes=("complex polar projection after converting real block maps back to complex form",),
    )


def _complex_pairwise_maps(
    pairwise_real: Mapping[tuple[str, str], np.ndarray],
    *,
    project_unitary: bool,
) -> tuple[dict[tuple[str, str], np.ndarray], float, float]:
    pairwise_complex: dict[tuple[str, str], np.ndarray] = {}
    projection_residuals: list[float] = []
    complex_residuals: list[float] = []
    for key, matrix in pairwise_real.items():
        complex_residuals.append(complex_linearity_residual(matrix))
        complex_matrix = real_block_matrix_to_complex(matrix)
        if project_unitary:
            projected = project_to_nearest_unitary(complex_matrix)
            projection_residuals.append(relative_residual(projected, complex_matrix))
            pairwise_complex[key] = projected
        else:
            pairwise_complex[key] = complex_matrix
    projection_residual = float(np.mean(projection_residuals)) if projection_residuals else float("nan")
    complex_residual = float(np.mean(complex_residuals)) if complex_residuals else float("nan")
    return pairwise_complex, projection_residual, complex_residual


def cycle_consistency_residual(
    pairwise_maps: Mapping[tuple[str, str], np.ndarray],
    chart_names: tuple[str, ...],
) -> float:
    """Mean residual of ``L[a,b] L[b,c] = L[a,c]`` over chart triples."""

    residuals = []
    for target in chart_names:
        for middle in chart_names:
            for source in chart_names:
                if target == middle or middle == source or target == source:
                    continue
                left = pairwise_maps[(target, middle)] @ pairwise_maps[(middle, source)]
                right = pairwise_maps[(target, source)]
                residuals.append(relative_residual(left, right))
    return float(np.mean(residuals)) if residuals else float("nan")


def synchronize_pairwise_complex_maps(
    pairwise_maps: Mapping[tuple[str, str], np.ndarray],
    chart_names: tuple[str, ...],
    *,
    project_gauges_unitary: bool,
) -> PairwiseSynchronization:
    """Recover global chart gauges from pairwise maps by block spectral synchronization."""

    if not chart_names:
        raise ValueError("chart_names must be nonempty")
    first = np.asarray(pairwise_maps[(chart_names[0], chart_names[0])], dtype=complex)
    if first.ndim != 2 or first.shape[0] != first.shape[1]:
        raise ValueError("pairwise maps must be square")
    dim = first.shape[0]
    count = len(chart_names)
    block = np.zeros((count * dim, count * dim), dtype=complex)
    for row, target in enumerate(chart_names):
        for col, source in enumerate(chart_names):
            matrix = np.asarray(pairwise_maps[(target, source)], dtype=complex)
            block[row * dim : (row + 1) * dim, col * dim : (col + 1) * dim] = matrix
    block = 0.5 * (block + block.conj().T)
    values, vectors = np.linalg.eigh(block)
    order = np.argsort(values)[-dim:][::-1]
    basis = vectors[:, order] @ np.diag(np.sqrt(np.maximum(values[order], 0.0)))
    identity_block = basis[:dim, :]
    identity_inverse = np.linalg.pinv(identity_block)

    gauges: dict[str, np.ndarray] = {}
    for idx, name in enumerate(chart_names):
        gauge = basis[idx * dim : (idx + 1) * dim, :] @ identity_inverse
        gauges[name] = project_to_nearest_unitary(gauge) if project_gauges_unitary else gauge
    gauges[chart_names[0]] = np.eye(dim, dtype=complex)

    synchronized: dict[tuple[str, str], np.ndarray] = {}
    residuals = []
    for target in chart_names:
        for source in chart_names:
            map_estimate = gauges[target] @ np.linalg.pinv(gauges[source])
            synchronized[(target, source)] = map_estimate
            residuals.append(relative_residual(map_estimate, pairwise_maps[(target, source)]))

    return PairwiseSynchronization(
        chart_names=chart_names,
        gauges=gauges,
        synchronized_pairwise=synchronized,
        sync_residual=float(np.mean(residuals)) if residuals else float("nan"),
        unitary_projection_residual=float("nan"),
        complex_structure_residual=float("nan"),
        cycle_residual_before=cycle_consistency_residual(pairwise_maps, chart_names),
        cycle_residual_after=cycle_consistency_residual(synchronized, chart_names),
    )


def fit_global_chart_synchronization(
    dataset: PairedTimeFrequencyChartDataset,
    *,
    ridge: float = 1e-8,
    generator_names: tuple[str, ...] | None = None,
) -> DenoisedChartRecovery:
    raw = _identity_transitions(dataset, ridge=ridge, generator_names=generator_names)
    pairwise_real = _pairwise_transitions(dataset, ridge=ridge)
    pairwise_complex, projection_residual, complex_residual = _complex_pairwise_maps(
        pairwise_real,
        project_unitary=False,
    )
    sync = synchronize_pairwise_complex_maps(
        pairwise_complex,
        dataset.chart_names,
        project_gauges_unitary=False,
    )
    names = _candidate_names(dataset, generator_names)
    candidates = {name: sync.gauges[name] for name in names}
    denoised = {name: complex_to_real_block_matrix(candidates[name]) for name in names}
    return _pack_recovery(
        method="global_chart_synchronization",
        dataset=dataset,
        raw=raw,
        denoised=denoised,
        candidates=candidates,
        global_sync_residual=sync.sync_residual,
        unitary_projection_residual=projection_residual,
        complex_structure_residual=complex_residual,
        selected_ridge=float(ridge),
        notes=(
            "block spectral synchronization from all pairwise chart maps",
            f"cycle residual {sync.cycle_residual_before:.6g} to {sync.cycle_residual_after:.6g}",
        ),
    )


def fit_unitary_global_chart_synchronization(
    dataset: PairedTimeFrequencyChartDataset,
    *,
    ridge: float = 1e-8,
    generator_names: tuple[str, ...] | None = None,
) -> DenoisedChartRecovery:
    raw = _identity_transitions(dataset, ridge=ridge, generator_names=generator_names)
    pairwise_real = _pairwise_transitions(dataset, ridge=ridge)
    pairwise_complex, projection_residual, complex_residual = _complex_pairwise_maps(
        pairwise_real,
        project_unitary=True,
    )
    sync = synchronize_pairwise_complex_maps(
        pairwise_complex,
        dataset.chart_names,
        project_gauges_unitary=True,
    )
    names = _candidate_names(dataset, generator_names)
    candidates = {name: sync.gauges[name] for name in names}
    denoised = {name: complex_to_real_block_matrix(candidates[name]) for name in names}
    return _pack_recovery(
        method="unitary_global_chart_synchronization",
        dataset=dataset,
        raw=raw,
        denoised=denoised,
        candidates=candidates,
        global_sync_residual=sync.sync_residual,
        unitary_projection_residual=projection_residual,
        complex_structure_residual=complex_residual,
        selected_ridge=float(ridge),
        notes=(
            "unitary-projected pairwise maps followed by block spectral synchronization",
            f"cycle residual {sync.cycle_residual_before:.6g} to {sync.cycle_residual_after:.6g}",
        ),
    )


def fit_nearest_heisenberg_projection(
    dataset: PairedTimeFrequencyChartDataset,
    *,
    projection_residual_threshold: float = 1e-2,
    ridge: float = 1e-8,
    generator_names: tuple[str, ...] | None = None,
    candidate_rank: int | None = None,
) -> DenoisedChartRecovery:
    names = _candidate_names(dataset, generator_names)
    rank = dataset.d**dataset.k if candidate_rank is None else int(candidate_rank)
    raw = _identity_transitions(dataset, ridge=ridge, generator_names=names)
    learned = _candidate_generators_from_real(raw)
    projection = project_to_nearest_finite_heisenberg(
        learned,
        expected_d=dataset.d,
        expected_k=dataset.k,
        candidate_rank=rank,
        projection_residual_threshold=projection_residual_threshold,
        generator_names=names,
    )
    denoised = {name: complex_to_real_block_matrix(projection.projected_generators[name]) for name in names}
    return _pack_recovery(
        method="nearest_heisenberg_projection",
        dataset=dataset,
        raw=raw,
        denoised=denoised,
        candidates=projection.projected_generators,
        heisenberg_projection=projection,
        selected_ridge=float(ridge),
        notes=(
            "commutator-form nearest finite-Heisenberg projection from raw learned maps",
            f"projection_accepted={projection.projection_accepted}",
            f"projection_decision={projection.decision}",
        ),
    )


def fit_unitary_then_heisenberg_projection(
    dataset: PairedTimeFrequencyChartDataset,
    *,
    projection_residual_threshold: float = 1e-2,
    ridge: float = 1e-8,
    generator_names: tuple[str, ...] | None = None,
    candidate_rank: int | None = None,
) -> DenoisedChartRecovery:
    names = _candidate_names(dataset, generator_names)
    rank = dataset.d**dataset.k if candidate_rank is None else int(candidate_rank)
    raw = _identity_transitions(dataset, ridge=ridge, generator_names=names)
    unitary = fit_complex_unitary_projection(dataset, ridge=ridge, generator_names=names)
    projection = project_to_nearest_finite_heisenberg(
        unitary.candidate_generators,
        expected_d=dataset.d,
        expected_k=dataset.k,
        candidate_rank=rank,
        projection_residual_threshold=projection_residual_threshold,
        generator_names=names,
    )
    denoised = {name: complex_to_real_block_matrix(projection.projected_generators[name]) for name in names}
    return _pack_recovery(
        method="unitary_then_heisenberg_projection",
        dataset=dataset,
        raw=raw,
        denoised=denoised,
        candidates=projection.projected_generators,
        unitary_projection_residual=unitary.unitary_projection_residual,
        heisenberg_projection=projection,
        selected_ridge=float(ridge),
        notes=(
            "complex-unitary projection followed by commutator-form finite-Heisenberg projection",
            f"projection_accepted={projection.projection_accepted}",
            f"projection_decision={projection.decision}",
        ),
    )


def fit_global_sync_then_heisenberg_projection(
    dataset: PairedTimeFrequencyChartDataset,
    *,
    projection_residual_threshold: float = 1e-2,
    ridge: float = 1e-8,
    generator_names: tuple[str, ...] | None = None,
    candidate_rank: int | None = None,
) -> DenoisedChartRecovery:
    names = _candidate_names(dataset, generator_names)
    rank = dataset.d**dataset.k if candidate_rank is None else int(candidate_rank)
    sync = fit_unitary_global_chart_synchronization(dataset, ridge=ridge, generator_names=names)
    projection = project_to_nearest_finite_heisenberg(
        sync.candidate_generators,
        expected_d=dataset.d,
        expected_k=dataset.k,
        candidate_rank=rank,
        projection_residual_threshold=projection_residual_threshold,
        generator_names=names,
    )
    denoised = {name: complex_to_real_block_matrix(projection.projected_generators[name]) for name in names}
    return _pack_recovery(
        method="global_sync_then_heisenberg_projection",
        dataset=dataset,
        raw=sync.transition_maps_raw,
        denoised=denoised,
        candidates=projection.projected_generators,
        global_sync_residual=sync.global_sync_residual,
        unitary_projection_residual=sync.unitary_projection_residual,
        complex_structure_residual=sync.complex_structure_residual,
        heisenberg_projection=projection,
        selected_ridge=float(ridge),
        notes=(
            "unitary global synchronization followed by commutator-form finite-Heisenberg projection",
            f"projection_accepted={projection.projection_accepted}",
            f"projection_decision={projection.decision}",
        ),
    )


def fit_all_denoised_chart_recoveries(
    dataset: PairedTimeFrequencyChartDataset,
    *,
    ridge: float = 1e-8,
    ridge_grid: tuple[float, ...] = RIDGE_GRID,
    generator_names: tuple[str, ...] | None = None,
) -> list[DenoisedChartRecovery]:
    return [
        fit_raw_least_squares_denoising(dataset, ridge=ridge, generator_names=generator_names),
        fit_ridge_least_squares_denoising(
            dataset,
            ridge_grid=ridge_grid,
            baseline_ridge=ridge,
            generator_names=generator_names,
        ),
        fit_nearest_unitary_projection(dataset, ridge=ridge, generator_names=generator_names),
        fit_complex_unitary_projection(dataset, ridge=ridge, generator_names=generator_names),
        fit_global_chart_synchronization(dataset, ridge=ridge, generator_names=generator_names),
        fit_unitary_global_chart_synchronization(dataset, ridge=ridge, generator_names=generator_names),
    ]
