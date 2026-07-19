"""Transition estimation, synchronization, and loop diagnostics for Application A."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import torch


def orthogonal_polar(matrix: torch.Tensor) -> torch.Tensor:
    u, _singular, vh = torch.linalg.svd(matrix, full_matrices=False)
    return u @ vh


def activation_procrustes(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Fit Q for target ~= source @ Q.T with an orthogonal Procrustes map."""

    cross = target.T @ source
    return orthogonal_polar(cross)


def ridge_transition(
    source: torch.Tensor, target: torch.Tensor, ridge: float = 1e-4
) -> torch.Tensor:
    dimension = source.shape[1]
    gram = source.T @ source + ridge * torch.eye(dimension, dtype=source.dtype)
    coefficients = torch.linalg.solve(gram, source.T @ target)
    return coefficients.T


def low_rank_subspace_transition(
    source: torch.Tensor, target: torch.Tensor, rank: int, ridge: float = 1e-4
) -> torch.Tensor:
    full = ridge_transition(source, target, ridge=ridge)
    delta = full - torch.eye(full.shape[0], dtype=full.dtype)
    u, singular, vh = torch.linalg.svd(delta, full_matrices=False)
    truncated = (u[:, :rank] * singular[:rank]) @ vh[:rank]
    return torch.eye(full.shape[0], dtype=full.dtype) + truncated


def weight_transition(source_adapter: torch.Tensor, target_adapter: torch.Tensor) -> torch.Tensor:
    return target_adapter @ torch.linalg.pinv(source_adapter)


def joint_transition(
    source: torch.Tensor,
    target: torch.Tensor,
    source_adapter: torch.Tensor,
    target_adapter: torch.Tensor,
) -> torch.Tensor:
    activation = activation_procrustes(source, target)
    weight = orthogonal_polar(weight_transition(source_adapter, target_adapter))
    return orthogonal_polar(activation + weight)


def fit_transition(
    method: str,
    source: torch.Tensor,
    target: torch.Tensor,
    source_adapter: torch.Tensor,
    target_adapter: torch.Tensor,
    low_rank: int = 8,
) -> torch.Tensor:
    if method == "weight_based":
        return weight_transition(source_adapter, target_adapter)
    if method == "activation_procrustes":
        return activation_procrustes(source, target)
    if method == "low_rank_subspace":
        return low_rank_subspace_transition(source, target, rank=low_rank)
    if method == "joint_weight_activation":
        return joint_transition(source, target, source_adapter, target_adapter)
    raise ValueError(f"unknown transition method: {method}")


def normalized_fit_residual(source: torch.Tensor, target: torch.Tensor, transition: torch.Tensor) -> float:
    prediction = source @ transition.T
    return float(torch.linalg.norm(prediction - target) / torch.linalg.norm(target).clamp_min(1e-12))


def inverse_consistency(left_to_right: torch.Tensor, right_to_left: torch.Tensor) -> float:
    identity = torch.eye(left_to_right.shape[0], dtype=left_to_right.dtype)
    return float(
        torch.linalg.norm(right_to_left @ left_to_right - identity)
        / torch.linalg.norm(identity)
    )


def bootstrap_transition_stability(
    method: str,
    source: torch.Tensor,
    target: torch.Tensor,
    source_adapter: torch.Tensor,
    target_adapter: torch.Tensor,
    full_transition: torch.Tensor,
    samples: int,
    seed: int,
    low_rank: int = 8,
) -> tuple[float, float, float]:
    generator = torch.Generator().manual_seed(seed)
    residuals = []
    denominator = torch.linalg.norm(full_transition).clamp_min(1e-12)
    for _ in range(samples):
        indices = torch.randint(0, len(source), (len(source),), generator=generator)
        candidate = fit_transition(
            method,
            source[indices],
            target[indices],
            source_adapter,
            target_adapter,
            low_rank=low_rank,
        )
        residuals.append(float(torch.linalg.norm(candidate - full_transition) / denominator))
    values = np.asarray(residuals, dtype=np.float64)
    return float(values.mean()), float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def loop_product(
    transitions: Mapping[tuple[int, int], torch.Tensor], loop: Sequence[int]
) -> torch.Tensor:
    if len(loop) < 2 or loop[0] != loop[-1]:
        raise ValueError("loop must contain at least one edge and repeat its starting vertex")
    dimension = transitions[(loop[0], loop[1])].shape[0]
    product = torch.eye(dimension, dtype=transitions[(loop[0], loop[1])].dtype)
    for source, target in zip(loop[:-1], loop[1:], strict=True):
        product = transitions[(source, target)] @ product
    return product


def identity_distance(matrix: torch.Tensor) -> float:
    identity = torch.eye(matrix.shape[0], dtype=matrix.dtype)
    return float(torch.linalg.norm(matrix - identity) / torch.linalg.norm(identity))


def commutator(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return left @ right @ torch.linalg.pinv(left) @ torch.linalg.pinv(right)


def commutator_distance(left: torch.Tensor, right: torch.Tensor) -> float:
    return identity_distance(commutator(left, right))


def connection_synchronization(
    transitions: Mapping[tuple[int, int], torch.Tensor], nodes: int
) -> tuple[list[torch.Tensor], float]:
    """Spectrally synchronize orthogonalized directed transitions."""

    dimension = next(iter(transitions.values())).shape[0]
    connection = torch.zeros((nodes * dimension, nodes * dimension), dtype=torch.float64)
    degrees = torch.zeros(nodes, dtype=torch.float64)
    for source in range(nodes):
        for target in range(nodes):
            if source == target or (target, source) not in transitions:
                continue
            block = orthogonal_polar(transitions[(target, source)].double())
            row = slice(source * dimension, (source + 1) * dimension)
            column = slice(target * dimension, (target + 1) * dimension)
            connection[row, column] = block
            degrees[source] += 1
    normalizer = torch.repeat_interleave(degrees.clamp_min(1).rsqrt(), dimension)
    normalized = normalizer[:, None] * connection * normalizer[None, :]
    eigenvalues, eigenvectors = torch.linalg.eigh(normalized)
    basis = eigenvectors[:, torch.argsort(eigenvalues, descending=True)[:dimension]]
    frames = []
    for node in range(nodes):
        block = basis[node * dimension : (node + 1) * dimension]
        frames.append(orthogonal_polar(block).float())
    gauges = [frame.T for frame in frames]
    residuals = []
    for (source, target), transition in transitions.items():
        if source == target:
            continue
        predicted = frames[target] @ frames[source].T
        denominator = torch.linalg.norm(transition).clamp_min(1e-12)
        residuals.append(float(torch.linalg.norm(orthogonal_polar(transition) - predicted) / denominator))
    return gauges, float(np.mean(residuals))


def spectral_summary(matrix: torch.Tensor) -> dict[str, float]:
    eigenvalues = torch.linalg.eigvals(matrix.double()).cpu().numpy()
    return {
        "spectral_radius": float(np.max(np.abs(eigenvalues))),
        "mean_eigenvalue_modulus": float(np.mean(np.abs(eigenvalues))),
        "mean_absolute_eigenphase": float(np.mean(np.abs(np.angle(eigenvalues)))),
        "trace_over_dimension_real": float(np.real(eigenvalues.mean())),
        "trace_over_dimension_imag": float(np.imag(eigenvalues.mean())),
    }
