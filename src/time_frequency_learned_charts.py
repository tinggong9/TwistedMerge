"""Learned chart recovery helpers for finite time-frequency benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .period_index_detector import RobustPeriodIndexDetection, robust_detect_commutator_matrix_period_index
from .time_frequency_benchmark import (
    PairedChartSplit,
    PairedTimeFrequencyChartDataset,
    complex_to_real_block_matrix,
    time_frequency_generator_chart_names,
)


LIFT_METHOD = "period_index_projective_morita_lift"
CALIBRATED_TOLERANCE = 3e-4
CALIBRATED_CONFIDENCE_MARGIN = 0.25
DENOISED_CHART_SCOPE_NOTE = (
    "Denoised learned-chart recoveries may improve noisy paired map estimates, "
    "but lift selection still requires robust period-index certification and "
    "candidate-rank divisibility by the certified index."
)


@dataclass(frozen=True)
class LearnedChartRecovery:
    level: str
    d: int
    k: int
    real_dimension: int
    latent_dimension: int
    chart_count: int
    transition_maps: dict[str, np.ndarray]
    candidate_generators: dict[str, np.ndarray]
    learned_operator_error_mean: float
    learned_operator_error_max: float
    pair_reconstruction_residual_train: float
    pair_reconstruction_residual_validation: float
    pair_reconstruction_residual_test: float
    train_accuracy: float
    validation_accuracy: float
    test_accuracy: float
    reconstruction_residual_train: float | None = None
    reconstruction_residual_test: float | None = None
    generator_mining_used: bool = False
    notes: tuple[str, ...] = ()


def relative_residual(left: np.ndarray, right: np.ndarray) -> float:
    denom = max(float(np.linalg.norm(right)), 1e-12)
    return float(np.linalg.norm(left - right) / denom)


def ridge_linear_map(source: np.ndarray, target: np.ndarray, ridge: float = 1e-8) -> np.ndarray:
    """Learn ``target ~= source @ L.T`` and return the column-action matrix ``L``."""

    x = np.asarray(source, dtype=float)
    y = np.asarray(target, dtype=float)
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError("source and target must be row matrices with the same row count")
    gram = x.T @ x + float(ridge) * np.eye(x.shape[1])
    coef = np.linalg.solve(gram, x.T @ y)
    return coef.T


def real_block_matrix_to_complex(matrix: np.ndarray) -> np.ndarray:
    """Approximate the complex matrix represented by a real block map."""

    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1] or arr.shape[0] % 2 != 0:
        raise ValueError("matrix must be square with even dimension")
    half = arr.shape[0] // 2
    top_left = arr[:half, :half]
    top_right = arr[:half, half:]
    bottom_left = arr[half:, :half]
    bottom_right = arr[half:, half:]
    real = 0.5 * (top_left + bottom_right)
    imag = 0.5 * (bottom_left - top_right)
    return real + 1j * imag


def complex_linearity_residual(matrix: np.ndarray) -> float:
    recovered = real_block_matrix_to_complex(matrix)
    projected = complex_to_real_block_matrix(recovered)
    return relative_residual(np.asarray(matrix, dtype=float), projected)


def selected_method_for(detection: RobustPeriodIndexDetection | None) -> str:
    if detection is not None and detection.status == "certified" and detection.decision == "period_index_lift_success":
        return LIFT_METHOD
    return "none"


def detect_recovered_chart_generators(
    generators: Mapping[str, np.ndarray],
    candidate_rank: int,
) -> RobustPeriodIndexDetection:
    return robust_detect_commutator_matrix_period_index(
        generators,
        candidate_rank=candidate_rank,
        centrality_tol_grid=(CALIBRATED_TOLERANCE,),
        phase_tol_grid=(CALIBRATED_TOLERANCE,),
        confidence_margin=CALIBRATED_CONFIDENCE_MARGIN,
    )


def _split_rows(split: PairedChartSplit, chart_name: str) -> np.ndarray:
    rows = split.chart_rows(chart_name)
    order = np.argsort(split.chart_sample_ids(chart_name), kind="stable")
    return rows[order]


def _split_labels(split: PairedChartSplit, chart_name: str = "I") -> np.ndarray:
    ids = split.chart_sample_ids(chart_name)
    order = np.argsort(ids, kind="stable")
    return split.labels[ids[order]]


def _candidate_names(dataset: PairedTimeFrequencyChartDataset, generator_names: tuple[str, ...] | None) -> tuple[str, ...]:
    names = generator_names or time_frequency_generator_chart_names(dataset.k)
    return tuple(name for name in names if name in dataset.chart_names)


def _mean_transition_residual(
    split: PairedChartSplit,
    transitions: Mapping[str, np.ndarray],
) -> float:
    source = _split_rows(split, "I")
    residuals = []
    for name, matrix in transitions.items():
        target = _split_rows(split, name)
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


def _ridge_classifier(train_x: np.ndarray, train_y: np.ndarray, n_classes: int, ridge: float = 1e-6) -> np.ndarray:
    one_hot = np.eye(n_classes)[np.asarray(train_y, dtype=int)]
    return ridge_linear_map(train_x, one_hot, ridge=ridge)


def _classifier_accuracy(weight: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
    logits = np.asarray(x, dtype=float) @ np.asarray(weight, dtype=float).T
    return float(np.mean(np.argmax(logits, axis=1) == np.asarray(y, dtype=int)))


def _identity_chart_accuracy(dataset: PairedTimeFrequencyChartDataset) -> tuple[float, float, float]:
    train_x = _split_rows(dataset.train, "I")
    train_y = _split_labels(dataset.train)
    weight = _ridge_classifier(train_x, train_y, dataset.n_classes)
    return (
        _classifier_accuracy(weight, train_x, train_y),
        _classifier_accuracy(weight, _split_rows(dataset.validation, "I"), _split_labels(dataset.validation)),
        _classifier_accuracy(weight, _split_rows(dataset.test, "I"), _split_labels(dataset.test)),
    )


def identity_chart_ridge_accuracies(dataset: PairedTimeFrequencyChartDataset) -> tuple[float, float, float]:
    """Ridge-classifier sanity accuracies on the identity chart only."""

    return _identity_chart_accuracy(dataset)


def fit_input_least_squares_chart(
    dataset: PairedTimeFrequencyChartDataset,
    *,
    ridge: float = 1e-8,
    generator_names: tuple[str, ...] | None = None,
) -> LearnedChartRecovery:
    names = _candidate_names(dataset, generator_names)
    source = _split_rows(dataset.train, "I")
    transitions = {
        name: ridge_linear_map(source, _split_rows(dataset.train, name), ridge=ridge)
        for name in names
    }
    candidate_generators = {name: real_block_matrix_to_complex(matrix) for name, matrix in transitions.items()}
    error_mean, error_max = _operator_errors(dataset, transitions)
    train_acc, val_acc, test_acc = _identity_chart_accuracy(dataset)
    return LearnedChartRecovery(
        level="input_least_squares_chart",
        d=dataset.d,
        k=dataset.k,
        real_dimension=dataset.dimension_real,
        latent_dimension=dataset.dimension_real,
        chart_count=dataset.chart_count,
        transition_maps=transitions,
        candidate_generators=candidate_generators,
        learned_operator_error_mean=error_mean,
        learned_operator_error_max=error_max,
        pair_reconstruction_residual_train=_mean_transition_residual(dataset.train, transitions),
        pair_reconstruction_residual_validation=_mean_transition_residual(dataset.validation, transitions),
        pair_reconstruction_residual_test=_mean_transition_residual(dataset.test, transitions),
        train_accuracy=train_acc,
        validation_accuracy=val_acc,
        test_accuracy=test_acc,
        notes=("ridge map learned from paired input chart observations",),
    )


def _autoencoder_basis(rows: np.ndarray, latent_dimension: int) -> np.ndarray:
    dim = rows.shape[1]
    if latent_dimension >= dim:
        return np.eye(dim)
    _u, _s, vt = np.linalg.svd(rows, full_matrices=False)
    return vt[:latent_dimension].T


def _encode(rows: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return np.asarray(rows, dtype=float) @ np.asarray(basis, dtype=float)


def _reconstruction_residual(split: PairedChartSplit, bases: Mapping[str, np.ndarray]) -> float:
    residuals = []
    for name, basis in bases.items():
        rows = _split_rows(split, name)
        reconstructed = _encode(rows, basis) @ basis.T
        residuals.append(relative_residual(reconstructed, rows))
    return float(np.mean(residuals)) if residuals else float("nan")


def fit_linear_autoencoder_chart(
    dataset: PairedTimeFrequencyChartDataset,
    *,
    latent_dimension: int | None = None,
    ridge: float = 1e-8,
    generator_names: tuple[str, ...] | None = None,
) -> LearnedChartRecovery:
    names = _candidate_names(dataset, generator_names)
    latent_dim = dataset.dimension_real if latent_dimension is None else int(latent_dimension)
    if latent_dim <= 0:
        raise ValueError("latent_dimension must be positive")

    bases = {
        chart_name: _autoencoder_basis(_split_rows(dataset.train, chart_name), min(latent_dim, dataset.dimension_real))
        for chart_name in dataset.chart_names
    }
    identity_basis = bases["I"]
    source_latent = _encode(_split_rows(dataset.train, "I"), identity_basis)
    transitions: dict[str, np.ndarray] = {}
    candidate_generators: dict[str, np.ndarray] = {}
    operator_errors: list[float] = []
    for name in names:
        target_latent = _encode(_split_rows(dataset.train, name), bases[name])
        latent_transition = ridge_linear_map(source_latent, target_latent, ridge=ridge)
        transitions[name] = latent_transition
        identity_latent_generator = identity_basis.T @ bases[name] @ latent_transition
        if identity_latent_generator.shape[0] == dataset.dimension_real:
            candidate_generators[name] = real_block_matrix_to_complex(identity_latent_generator)
            operator_errors.append(
                relative_residual(
                    identity_latent_generator,
                    complex_to_real_block_matrix(dataset.chart_operators[name]),
                )
            )
        else:
            candidate_generators[name] = identity_latent_generator.astype(complex)
            operator_errors.append(float("nan"))

    train_acc, val_acc, test_acc = _identity_chart_accuracy(dataset)
    train_pair = _latent_transition_residual(dataset.train, bases, transitions)
    validation_pair = _latent_transition_residual(dataset.validation, bases, transitions)
    test_pair = _latent_transition_residual(dataset.test, bases, transitions)
    finite_errors = [value for value in operator_errors if np.isfinite(value)]
    return LearnedChartRecovery(
        level="linear_autoencoder_chart",
        d=dataset.d,
        k=dataset.k,
        real_dimension=dataset.dimension_real,
        latent_dimension=min(latent_dim, dataset.dimension_real),
        chart_count=dataset.chart_count,
        transition_maps=transitions,
        candidate_generators=candidate_generators,
        learned_operator_error_mean=float(np.mean(finite_errors)) if finite_errors else float("nan"),
        learned_operator_error_max=float(np.max(finite_errors)) if finite_errors else float("nan"),
        pair_reconstruction_residual_train=train_pair,
        pair_reconstruction_residual_validation=validation_pair,
        pair_reconstruction_residual_test=test_pair,
        train_accuracy=train_acc,
        validation_accuracy=val_acc,
        test_accuracy=test_acc,
        reconstruction_residual_train=_reconstruction_residual(dataset.train, bases),
        reconstruction_residual_test=_reconstruction_residual(dataset.test, bases),
        notes=("linear autoencoder uses per-chart linear reconstruction bases; full dimension preserves input coordinates",),
    )


def _latent_transition_residual(
    split: PairedChartSplit,
    bases: Mapping[str, np.ndarray],
    transitions: Mapping[str, np.ndarray],
) -> float:
    source = _encode(_split_rows(split, "I"), bases["I"])
    residuals = []
    for name, matrix in transitions.items():
        target = _encode(_split_rows(split, name), bases[name])
        predicted = source @ np.asarray(matrix, dtype=float).T
        residuals.append(relative_residual(predicted, target))
    return float(np.mean(residuals)) if residuals else float("nan")


def fit_supervised_encoder_chart(
    dataset: PairedTimeFrequencyChartDataset,
    *,
    ridge: float = 1e-6,
    generator_names: tuple[str, ...] | None = None,
) -> LearnedChartRecovery:
    names = _candidate_names(dataset, generator_names)
    encoders: dict[str, np.ndarray] = {}
    for chart_name in dataset.chart_names:
        encoders[chart_name] = _ridge_classifier(
            _split_rows(dataset.train, chart_name),
            _split_labels(dataset.train, chart_name),
            dataset.n_classes,
            ridge=ridge,
        )
    source_features = _split_rows(dataset.train, "I") @ encoders["I"].T
    transitions = {}
    for name in names:
        target_features = _split_rows(dataset.train, name) @ encoders[name].T
        transitions[name] = ridge_linear_map(source_features, target_features, ridge=ridge)
    candidate_generators = {name: matrix.astype(complex) for name, matrix in transitions.items()}

    train_x = _split_rows(dataset.train, "I")
    train_y = _split_labels(dataset.train)
    return LearnedChartRecovery(
        level="supervised_encoder_chart",
        d=dataset.d,
        k=dataset.k,
        real_dimension=dataset.dimension_real,
        latent_dimension=dataset.n_classes,
        chart_count=dataset.chart_count,
        transition_maps=transitions,
        candidate_generators=candidate_generators,
        learned_operator_error_mean=float("nan"),
        learned_operator_error_max=float("nan"),
        pair_reconstruction_residual_train=_feature_transition_residual(dataset.train, encoders, transitions),
        pair_reconstruction_residual_validation=_feature_transition_residual(dataset.validation, encoders, transitions),
        pair_reconstruction_residual_test=_feature_transition_residual(dataset.test, encoders, transitions),
        train_accuracy=_classifier_accuracy(encoders["I"], train_x, train_y),
        validation_accuracy=_classifier_accuracy(encoders["I"], _split_rows(dataset.validation, "I"), _split_labels(dataset.validation)),
        test_accuracy=_classifier_accuracy(encoders["I"], _split_rows(dataset.test, "I"), _split_labels(dataset.test)),
        notes=(
            "supervised label features are exploratory and may discard phase/projective information",
            "no lift is selected unless the robust detector certifies period and index",
        ),
    )


def _feature_transition_residual(
    split: PairedChartSplit,
    encoders: Mapping[str, np.ndarray],
    transitions: Mapping[str, np.ndarray],
) -> float:
    source = _split_rows(split, "I") @ encoders["I"].T
    residuals = []
    for name, matrix in transitions.items():
        target = _split_rows(split, name) @ encoders[name].T
        predicted = source @ np.asarray(matrix, dtype=float).T
        residuals.append(relative_residual(predicted, target))
    return float(np.mean(residuals)) if residuals else float("nan")


def random_noncentral_chart_generators(width: int, count: int = 4, seed: int = 0) -> dict[str, np.ndarray]:
    if width < 2:
        raise ValueError("width must be at least 2")
    rng = np.random.default_rng(seed)
    generators = {}
    for idx in range(count):
        raw = rng.normal(size=(width, width)) + 1j * rng.normal(size=(width, width))
        generators[f"R{idx + 1}"] = raw + width * np.eye(width, dtype=complex)
    return generators
