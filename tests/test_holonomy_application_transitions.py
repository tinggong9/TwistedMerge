from __future__ import annotations

import torch

from src.holonomy_application_transitions import (
    activation_procrustes,
    commutator_distance,
    connection_synchronization,
    identity_distance,
    inverse_consistency,
    loop_product,
    normalized_fit_residual,
)


def random_orthogonal(dimension: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    q, _r = torch.linalg.qr(torch.randn(dimension, dimension, generator=generator))
    return q


def test_procrustes_recovers_exact_orthogonal_transition() -> None:
    generator = torch.Generator().manual_seed(8)
    source = torch.randn(200, 7, generator=generator)
    expected = random_orthogonal(7, 9)
    target = source @ expected.T
    observed = activation_procrustes(source, target)
    assert torch.allclose(observed, expected, atol=1e-5, rtol=1e-5)
    assert normalized_fit_residual(source, target, observed) < 1e-5
    assert inverse_consistency(observed, observed.T) < 1e-5


def test_consistent_connection_has_trivial_loops_and_synchronizes() -> None:
    frames = [random_orthogonal(5, seed) for seed in range(4)]
    transitions = {
        (source, target): frames[target] @ frames[source].T
        for source in range(4)
        for target in range(4)
        if source != target
    }
    triangle = loop_product(transitions, (0, 1, 2, 0))
    square = loop_product(transitions, (0, 1, 2, 3, 0))
    assert identity_distance(triangle) < 1e-5
    assert identity_distance(square) < 1e-5
    assert commutator_distance(triangle, square) < 1e-5
    gauges, residual = connection_synchronization(transitions, nodes=4)
    assert len(gauges) == 4
    assert residual < 1e-5
