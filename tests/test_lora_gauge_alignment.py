from __future__ import annotations

import numpy as np

from src.lora_gauge_alignment import (
    canonical_svd_factors,
    effective_delta,
    estimate_transition,
    gauge_transform,
    global_align,
    mean_effective_delta,
    merged_factor_delta,
    reference_align,
    sample_gauge,
)


def test_all_gauge_families_preserve_effective_delta() -> None:
    rng = np.random.default_rng(20260719)
    b = rng.normal(size=(9, 3))
    a = rng.normal(size=(3, 7))
    original = effective_delta((b, a))
    for family in ("orthogonal", "positive_diagonal", "dense", "ill_conditioned"):
        transformed = gauge_transform(b, a, sample_gauge(rng, 3, family))
        tolerance = 2e-6 if family == "ill_conditioned" else 2e-12
        assert np.allclose(effective_delta(transformed), original, rtol=tolerance, atol=tolerance)


def test_directed_transition_orientation_is_explicit() -> None:
    rng = np.random.default_rng(11)
    b = rng.normal(size=(8, 3))
    a = rng.normal(size=(3, 6))
    q_source = sample_gauge(rng, 3, "dense", 12.0)
    q_target = sample_gauge(rng, 3, "positive_diagonal", 5.0)
    source = gauge_transform(b, a, q_source)
    target = gauge_transform(b, a, q_target)
    expected = np.linalg.solve(q_source, q_target)
    for mode in ("b", "a", "joint"):
        estimate = estimate_transition(source, target, mode=mode)
        assert np.allclose(estimate.matrix, expected, atol=2e-11, rtol=2e-11)
        assert estimate.b_relative_residual < 1e-11
        assert estimate.a_relative_residual < 1e-11


def _shared_subspace_adapters(seed: int = 7):
    rng = np.random.default_rng(seed)
    shared_b = rng.normal(size=(10, 3))
    factors = [(shared_b.copy(), rng.normal(size=(3, 8))) for _ in range(4)]
    gauges = [sample_gauge(rng, 3, "dense", 20.0) for _ in factors]
    scrambled = [gauge_transform(*factor, gauge) for factor, gauge in zip(factors, gauges)]
    return factors, scrambled


def test_reference_and_global_alignment_recover_shared_basis_merge() -> None:
    factors, scrambled = _shared_subspace_adapters()
    expected = mean_effective_delta(factors)
    reference_factors, _, _ = reference_align(scrambled, mode="b")
    global_factors, _, _ = global_align(scrambled, mode="b")
    assert np.allclose(merged_factor_delta(reference_factors), expected, atol=2e-11, rtol=2e-11)
    assert np.allclose(merged_factor_delta(global_factors), expected, atol=2e-11, rtol=2e-11)


def test_canonical_svd_factors_are_deterministic_and_rank_bounded() -> None:
    rng = np.random.default_rng(17)
    matrix = rng.normal(size=(11, 7))
    first = canonical_svd_factors(matrix, rank=3)
    second = canonical_svd_factors(matrix.copy(), rank=3)
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
    assert np.linalg.matrix_rank(effective_delta(first)) <= 3
