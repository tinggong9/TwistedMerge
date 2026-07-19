from __future__ import annotations

import numpy as np

from src.lora_cycle_diagnostics import cycle_aware_merge, triangle_cycle_metrics
from src.lora_gauge_alignment import (
    canonical_svd_factors,
    effective_delta,
    estimate_pairwise_transitions,
    gauge_transform,
    global_align,
    mean_effective_delta,
    merged_factor_delta,
    sample_gauge,
    truncated_svd,
)


def _fixture(seed: int = 23):
    rng = np.random.default_rng(seed)
    shared_b = rng.normal(size=(7, 3))
    factors = [(shared_b.copy(), rng.normal(size=(3, 9))) for _ in range(4)]
    return rng, factors


def test_many_equivalent_scrambles_leave_synchronized_merge_fixed() -> None:
    rng, factors = _fixture()
    expected = mean_effective_delta(factors)
    svd_expected = truncated_svd(expected, 3)
    synchronized_outputs = []
    canonical_outputs = []
    naive_outputs = []
    for _ in range(20):
        scrambled = [
            gauge_transform(*factor, sample_gauge(rng, 3, "dense", 25.0))
            for factor in factors
        ]
        aligned, _, _ = global_align(scrambled, mode="b")
        synchronized_outputs.append(merged_factor_delta(aligned))
        canonical = [canonical_svd_factors(effective_delta(factor), 3) for factor in scrambled]
        canonical_b = np.mean([factor[0] for factor in canonical], axis=0)
        canonical_a = np.mean([factor[1] for factor in canonical], axis=0)
        canonical_outputs.append(canonical_b @ canonical_a)
        naive_b = np.mean([factor[0] for factor in scrambled], axis=0)
        naive_a = np.mean([factor[1] for factor in scrambled], axis=0)
        naive_outputs.append(naive_b @ naive_a)

    for output in synchronized_outputs:
        assert np.allclose(output, expected, atol=3e-11, rtol=3e-11)
    assert all(
        np.allclose(output, canonical_outputs[0], atol=3e-11, rtol=3e-11)
        for output in canonical_outputs[1:]
    )
    assert np.allclose(truncated_svd(mean_effective_delta(scrambled), 3), svd_expected, atol=3e-11, rtol=3e-11)
    naive_variation = max(np.linalg.norm(output - naive_outputs[0]) for output in naive_outputs[1:])
    assert naive_variation > 1e-2


def test_cycle_metrics_close_for_exact_transition_cocycle() -> None:
    rng, factors = _fixture()
    scrambled = [gauge_transform(*factor, sample_gauge(rng, 3, "dense", 10.0)) for factor in factors]
    transitions = estimate_pairwise_transitions(scrambled, mode="b")
    metrics = triangle_cycle_metrics(transitions, len(scrambled))
    assert len(metrics) == 4
    assert max(metric.normalized_frobenius_defect for metric in metrics) < 1e-11
    assert max(metric.spectral_defect for metric in metrics) < 1e-11


def test_cycle_aware_method_falls_back_on_injected_inconsistency() -> None:
    _, factors = _fixture()
    transitions = estimate_pairwise_transitions(factors, mode="b")
    corrupted = {key: value.matrix.copy() for key, value in transitions.items()}
    corrupted[(0, 1)] = corrupted[(0, 1)] @ np.diag([1.3, 1.0, 1.0])
    result = cycle_aware_merge(factors, transitions=corrupted, rank=3, cycle_tolerance=1e-8)
    expected = truncated_svd(mean_effective_delta(factors), 3)
    assert result.decision == "fallback_full_delta_svd"
    assert "cycle_defect" in result.reason
    assert np.allclose(result.delta, expected, atol=1e-12, rtol=1e-12)
