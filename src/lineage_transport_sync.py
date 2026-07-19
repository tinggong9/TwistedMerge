"""Unlabeled representation transport and cycle synchronization utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


Array = np.ndarray
Edge = tuple[str, str]
TRANSITION_METHODS = (
    "orthogonal_procrustes",
    "ridge_linear",
    "low_rank_residual",
    "whitened_cca",
)


@dataclass(frozen=True)
class TransitionFit:
    method: str
    matrix: Array
    fit_residual: float
    condition_number: float
    effective_rank: int
    singular_value_spread: float


@dataclass(frozen=True)
class SynchronizationResult:
    frames: dict[str, Array]
    maps_to_common: dict[str, Array]
    synchronized_transitions: dict[Edge, Array]
    connection_residual: float
    max_connection_residual: float
    edge_count: int


def _as_float_matrix(value: Array) -> Array:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError("representations must be finite two-dimensional arrays")
    return array


def _center(value: Array) -> Array:
    array = _as_float_matrix(value)
    return array - array.mean(axis=0, keepdims=True)


def project_to_orthogonal(matrix: Array) -> Array:
    left, _singular, right = np.linalg.svd(np.asarray(matrix, dtype=np.float64), full_matrices=False)
    return left @ right


def orthogonal_procrustes(source: Array, target: Array) -> Array:
    source_c = _center(source)
    target_c = _center(target)
    if source_c.shape != target_c.shape:
        raise ValueError("orthogonal transport requires matching representation shapes")
    return project_to_orthogonal(target_c.T @ source_c)


def ridge_linear(source: Array, target: Array, ridge: float = 1e-3) -> Array:
    source_c = _center(source)
    target_c = _center(target)
    if len(source_c) != len(target_c) or source_c.shape[1] != target_c.shape[1]:
        raise ValueError("ridge transport requires aligned equal-dimensional samples")
    gram = source_c.T @ source_c + float(ridge) * np.eye(source_c.shape[1])
    return np.linalg.solve(gram, source_c.T @ target_c).T


def low_rank_residual(source: Array, target: Array, rank: int = 8, ridge: float = 1e-3) -> Array:
    full = ridge_linear(source, target, ridge=ridge)
    identity = np.eye(full.shape[0])
    delta = full - identity
    left, singular, right = np.linalg.svd(delta, full_matrices=False)
    kept = min(int(rank), len(singular))
    return identity + (left[:, :kept] * singular[:kept]) @ right[:kept]


def _matrix_power_psd(matrix: Array, exponent: float, ridge: float) -> Array:
    values, vectors = np.linalg.eigh(0.5 * (matrix + matrix.T))
    powered = np.maximum(values, float(ridge)) ** float(exponent)
    return (vectors * powered) @ vectors.T


def whitened_cca(source: Array, target: Array, ridge: float = 1e-4) -> Array:
    source_c = _center(source)
    target_c = _center(target)
    if source_c.shape != target_c.shape:
        raise ValueError("whitened transport requires matching representation shapes")
    denominator = max(len(source_c) - 1, 1)
    source_cov = source_c.T @ source_c / denominator
    target_cov = target_c.T @ target_c / denominator
    source_inverse_root = _matrix_power_psd(source_cov, -0.5, ridge)
    target_inverse_root = _matrix_power_psd(target_cov, -0.5, ridge)
    source_white = source_c @ source_inverse_root
    target_white = target_c @ target_inverse_root
    rotation = orthogonal_procrustes(source_white, target_white)
    target_root = _matrix_power_psd(target_cov, 0.5, ridge)
    return target_root @ rotation @ source_inverse_root


def fit_matrix(method: str, source: Array, target: Array) -> Array:
    if method == "orthogonal_procrustes":
        return orthogonal_procrustes(source, target)
    if method == "ridge_linear":
        return ridge_linear(source, target, ridge=1e-3)
    if method == "low_rank_residual":
        return low_rank_residual(source, target, rank=8, ridge=1e-3)
    if method == "whitened_cca":
        return whitened_cca(source, target, ridge=1e-4)
    raise ValueError(f"unknown transition method: {method}")


def normalized_residual(source: Array, target: Array, matrix: Array) -> float:
    source_c = _center(source)
    target_c = _center(target)
    prediction = source_c @ np.asarray(matrix, dtype=np.float64).T
    return float(np.linalg.norm(prediction - target_c) / max(np.linalg.norm(target_c), 1e-12))


def matrix_diagnostics(matrix: Array) -> dict[str, float | int]:
    singular = np.linalg.svd(np.asarray(matrix, dtype=np.float64), compute_uv=False)
    maximum = float(singular.max()) if len(singular) else 0.0
    minimum = float(singular.min()) if len(singular) else 0.0
    threshold = max(maximum * 1e-6, 1e-12)
    return {
        "condition_number": float(maximum / max(minimum, 1e-12)),
        "effective_rank": int(np.sum(singular > threshold)),
        "singular_value_spread": float(maximum - minimum),
    }


def fit_transition(method: str, source: Array, target: Array) -> TransitionFit:
    matrix = fit_matrix(method, source, target)
    diagnostics = matrix_diagnostics(matrix)
    return TransitionFit(
        method=method,
        matrix=matrix,
        fit_residual=normalized_residual(source, target, matrix),
        condition_number=float(diagnostics["condition_number"]),
        effective_rank=int(diagnostics["effective_rank"]),
        singular_value_spread=float(diagnostics["singular_value_spread"]),
    )


def select_transition(
    fit_source: Array,
    fit_target: Array,
    validation_source: Array,
    validation_target: Array,
    methods: Sequence[str] = TRANSITION_METHODS,
) -> tuple[TransitionFit, dict[str, TransitionFit], dict[str, float]]:
    fits = {method: fit_transition(method, fit_source, fit_target) for method in methods}
    validation = {
        method: normalized_residual(validation_source, validation_target, fit.matrix)
        for method, fit in fits.items()
    }
    selected_method = min(methods, key=lambda method: (validation[method], method))
    return fits[selected_method], fits, validation


def inverse_consistency(left_to_right: Array, right_to_left: Array) -> float:
    dimension = left_to_right.shape[0]
    identity = np.eye(dimension)
    return float(
        np.linalg.norm(np.asarray(right_to_left) @ np.asarray(left_to_right) - identity)
        / max(np.linalg.norm(identity), 1e-12)
    )


def loop_product(transitions: Mapping[Edge, Array], loop: Sequence[str]) -> Array:
    if len(loop) < 2 or loop[0] != loop[-1]:
        raise ValueError("loop must repeat its starting vertex")
    first = np.asarray(transitions[(loop[0], loop[1])], dtype=np.float64)
    product = np.eye(first.shape[0])
    for source, target in zip(loop[:-1], loop[1:], strict=True):
        product = np.asarray(transitions[(source, target)], dtype=np.float64) @ product
    return product


def identity_distance(matrix: Array) -> float:
    identity = np.eye(matrix.shape[0])
    return float(np.linalg.norm(np.asarray(matrix) - identity) / max(np.linalg.norm(identity), 1e-12))


def nearest_orthogonal_distance(matrix: Array) -> float:
    matrix = np.asarray(matrix, dtype=np.float64)
    denominator = max(np.linalg.norm(matrix), 1e-12)
    return float(np.linalg.norm(matrix - project_to_orthogonal(matrix)) / denominator)


def commutator(left: Array, right: Array) -> Array:
    return np.asarray(left) @ np.asarray(right) @ np.linalg.pinv(left) @ np.linalg.pinv(right)


def commutator_distance(left: Array, right: Array) -> float:
    return identity_distance(commutator(left, right))


def loop_statistics(matrix: Array) -> dict[str, float]:
    matrix = np.asarray(matrix, dtype=np.float64)
    eigenvalues = np.linalg.eigvals(matrix)
    singular = np.linalg.svd(matrix, compute_uv=False)
    sign, logabsdet = np.linalg.slogdet(matrix)
    modulus = np.abs(eigenvalues)
    phases = np.abs(np.angle(eigenvalues))
    return {
        "identity_distance": identity_distance(matrix),
        "spectral_radius": float(modulus.max()),
        "singular_value_spread": float(singular.max() - singular.min()),
        "trace_over_dimension_real": float(np.trace(matrix).real / max(matrix.shape[0], 1)),
        "determinant_sign": float(sign),
        "log_absolute_determinant": float(logabsdet),
        "nearest_orthogonal_distance": nearest_orthogonal_distance(matrix),
        "eigen_modulus_mean": float(modulus.mean()),
        "eigen_modulus_std": float(modulus.std()),
        "absolute_eigenphase_mean": float(phases.mean()),
        "absolute_eigenphase_max": float(phases.max()),
    }


def synchronize_frames(
    transitions: Mapping[Edge, Array], nodes: Sequence[str]
) -> SynchronizationResult:
    if len(nodes) < 2:
        raise ValueError("synchronization needs at least two nodes")
    node_list = list(dict.fromkeys(nodes))
    node_index = {node: index for index, node in enumerate(node_list)}
    if not transitions:
        raise ValueError("synchronization needs at least one directed edge")
    dimension = next(iter(transitions.values())).shape[0]
    connection = np.zeros((len(node_list) * dimension, len(node_list) * dimension), dtype=np.float64)
    counts = np.zeros((len(node_list), len(node_list)), dtype=np.float64)
    for (source, target), value in transitions.items():
        if source == target or source not in node_index or target not in node_index:
            continue
        matrix = project_to_orthogonal(value)
        source_index = node_index[source]
        target_index = node_index[target]
        target_slice = slice(target_index * dimension, (target_index + 1) * dimension)
        source_slice = slice(source_index * dimension, (source_index + 1) * dimension)
        connection[target_slice, source_slice] += matrix
        counts[target_index, source_index] += 1.0
    for target in range(len(node_list)):
        for source in range(len(node_list)):
            if counts[target, source] > 0:
                target_slice = slice(target * dimension, (target + 1) * dimension)
                source_slice = slice(source * dimension, (source + 1) * dimension)
                connection[target_slice, source_slice] /= counts[target, source]
    connection = 0.5 * (connection + connection.T)
    degrees = np.zeros(len(node_list), dtype=np.float64)
    for (source, target) in transitions:
        if source != target and source in node_index and target in node_index:
            degrees[node_index[source]] += 1.0
            degrees[node_index[target]] += 1.0
    normalizer = np.repeat(np.maximum(degrees, 1.0) ** -0.5, dimension)
    normalized = normalizer[:, None] * connection * normalizer[None, :]
    eigenvalues, eigenvectors = np.linalg.eigh(normalized)
    basis = eigenvectors[:, np.argsort(eigenvalues)[-dimension:]]
    frames: dict[str, Array] = {}
    for node, index in node_index.items():
        block = basis[index * dimension : (index + 1) * dimension]
        frames[node] = project_to_orthogonal(block)
    synchronized = {
        (source, target): frames[target] @ frames[source].T
        for source, target in transitions
        if source in frames and target in frames and source != target
    }
    residuals = []
    for edge, predicted in synchronized.items():
        observed = project_to_orthogonal(transitions[edge])
        residuals.append(float(np.linalg.norm(observed - predicted) / max(np.linalg.norm(observed), 1e-12)))
    maps_to_common = {node: frame for node, frame in frames.items()}
    return SynchronizationResult(
        frames=frames,
        maps_to_common=maps_to_common,
        synchronized_transitions=synchronized,
        connection_residual=float(np.mean(residuals)) if residuals else 0.0,
        max_connection_residual=float(np.max(residuals)) if residuals else 0.0,
        edge_count=len(residuals),
    )


def bootstrap_transition_instability(
    source: Array,
    target: Array,
    method: str,
    full_matrix: Array,
    *,
    samples: int = 100,
    seed: int = 0,
) -> tuple[float, float, float]:
    source = _as_float_matrix(source)
    target = _as_float_matrix(target)
    rng = np.random.default_rng(seed)
    denominator = max(np.linalg.norm(full_matrix), 1e-12)
    values = []
    for _ in range(int(samples)):
        indices = rng.integers(0, len(source), size=len(source))
        candidate = fit_matrix(method, source[indices], target[indices])
        values.append(float(np.linalg.norm(candidate - full_matrix) / denominator))
    array = np.asarray(values)
    return float(array.mean()), float(np.quantile(array, 0.025)), float(np.quantile(array, 0.975))


def bootstrap_loop_identity_distance(
    representations: Mapping[str, Array],
    loop: Sequence[str],
    methods: Mapping[Edge, str],
    *,
    samples: int = 100,
    seed: int = 0,
) -> tuple[float, float, float, Array]:
    products = bootstrap_loop_products(
        representations, loop, methods, samples=samples, seed=seed
    )
    array = np.asarray([identity_distance(product) for product in products], dtype=np.float64)
    return (
        float(array.mean()),
        float(np.quantile(array, 0.025)),
        float(np.quantile(array, 0.975)),
        array,
    )


def bootstrap_loop_products(
    representations: Mapping[str, Array],
    loop: Sequence[str],
    methods: Mapping[Edge, str],
    *,
    samples: int = 100,
    seed: int = 0,
) -> Array:
    lengths = {len(representations[node]) for node in set(loop)}
    if len(lengths) != 1:
        raise ValueError("loop representations must share one anchor ordering")
    count = lengths.pop()
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(int(samples)):
        indices = rng.integers(0, count, size=count)
        transitions = {
            edge: fit_matrix(
                methods[edge], representations[edge[0]][indices], representations[edge[1]][indices]
            )
            for edge in zip(loop[:-1], loop[1:], strict=True)
        }
        values.append(loop_product(transitions, loop))
    return np.stack(values)
