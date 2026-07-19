from __future__ import annotations

import torch

from src.holonomy_period_index_capacity import (
    candidate_generators,
    carrier_vectors,
    chart_operators,
    coherent_fusion,
    relation_residual,
    unitarity_residual,
)


def test_full_index_representations_satisfy_projective_relations() -> None:
    for name, index in (
        ("period2_index2", 2),
        ("period2_index4", 4),
        ("period3_index3", 3),
    ):
        case, generators = candidate_generators(name, index)
        _case, operators = chart_operators(name, index)
        assert relation_residual(case, generators) < 1e-6
        assert unitarity_residual(operators) < 1e-6


def test_index_insufficient_capacity_fails_unitarity() -> None:
    for name, capacity in (
        ("period2_index2", 1),
        ("period2_index4", 3),
        ("period3_index3", 2),
    ):
        _case, operators = chart_operators(name, capacity)
        assert unitarity_residual(operators) > 0.1


def test_coherent_full_index_layer_recovers_actual_logits() -> None:
    generator = torch.Generator().manual_seed(51)
    local_logits = torch.randn(8, 17, 10, generator=generator)
    _case, operators = chart_operators("period3_index3", 3)
    carriers = carrier_vectors(3, 10, 52)
    observed = coherent_fusion(local_logits, operators, carriers)
    expected = local_logits.mean(0)
    assert torch.allclose(observed, expected, atol=2e-5, rtol=2e-5)
