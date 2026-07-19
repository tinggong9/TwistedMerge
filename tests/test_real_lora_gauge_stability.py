from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiments.real_lora_gauge_stability import (
    PRIMARY_FAMILIES,
    phase_a_gates,
    preservation_metrics,
)
from src.lora_gauge_alignment import effective_delta, gauge_transform, sample_gauge
from src.lora_gauge_practical import (
    cycle_defects,
    merge_trained_factors,
    orthogonal_transitions,
    whiten_factor,
)


def distinct_factors(seed: int = 4):
    rng = np.random.default_rng(seed)
    return [(rng.normal(size=(12, 4)), rng.normal(size=(4, 12))) for _ in range(5)]


def scrambled(factors, family: str, seed: int = 9):
    rng = np.random.default_rng(seed)
    gauges = [sample_gauge(rng, 4, family, {"orthogonal": 1, "positive_diagonal": 8, "dense": 30}[family]) for _ in factors]
    return [gauge_transform(*factor, gauge) for factor, gauge in zip(factors, gauges)], gauges


@pytest.mark.parametrize("family", PRIMARY_FAMILIES)
@pytest.mark.parametrize("method", ["pairwise_reference_alignment", "global_synchronization"])
def test_whitened_alignment_is_invariant_for_distinct_trained_like_factors(family, method):
    factors = distinct_factors()
    changed, _ = scrambled(factors, family)
    reference = effective_delta(merge_trained_factors(factors, method).factors)
    observed = effective_delta(merge_trained_factors(changed, method).factors)
    assert np.linalg.norm(reference - observed) / np.linalg.norm(reference) < 1e-10


def test_naive_factor_average_is_representation_dependent():
    factors = distinct_factors()
    changed, _ = scrambled(factors, "dense")
    reference = effective_delta(merge_trained_factors(factors, "naive_factor_average").factors)
    observed = effective_delta(merge_trained_factors(changed, "naive_factor_average").factors)
    assert np.linalg.norm(reference - observed) / np.linalg.norm(reference) > 1e-2


def test_oracle_inverts_planted_scramble():
    factors = distinct_factors()
    changed, gauges = scrambled(factors, "positive_diagonal")
    reference = effective_delta(merge_trained_factors(factors, "naive_factor_average").factors)
    observed = effective_delta(merge_trained_factors(changed, "oracle_alignment", planted_gauges=gauges).factors)
    assert np.allclose(reference, observed, atol=1e-11, rtol=1e-11)


def test_factor_methods_report_zero_dense_allocations():
    factors = distinct_factors()
    for method in ("naive_factor_average", "pairwise_reference_alignment", "global_synchronization"):
        result = merge_trained_factors(factors, method)
        assert result.dense_allocation_count == 0
        assert result.temporary_dense_bytes == 0
    assert merge_trained_factors(factors, "full_delta_svd").dense_allocation_count > 0


def test_cycle_defects_are_invariant_after_whitening():
    factors = distinct_factors()
    changed, _ = scrambled(factors, "dense")
    _, before = orthogonal_transitions(factors)
    _, after = orthogonal_transitions(changed)
    assert np.allclose(cycle_defects(before, len(factors)), cycle_defects(after, len(factors)), atol=1e-10)


def test_whitening_produces_orthonormal_b_columns():
    for factor in distinct_factors():
        b, _ = whiten_factor(factor)
        assert np.allclose(b.T @ b, np.eye(4), atol=1e-10)


def test_preservation_metrics_accept_exact_gauge():
    factors = distinct_factors()[:2]
    changed, _ = scrambled(factors, "dense")
    rng = np.random.default_rng(30)
    heads = [(rng.normal(size=(3, 12)), rng.normal(size=3)) for _ in factors]
    features = rng.normal(size=(2, 20, 12))
    labels = rng.integers(0, 3, size=20)
    result = preservation_metrics(factors, changed, heads, features, labels)
    assert result["accepted_scramble"]
    assert result["maximum_individual_prediction_disagreement"] == 0.0


def test_phase_gate_uses_independent_groups_not_scramble_count():
    run_rows = []
    preservation_rows = []
    for group in (0, 1, 2):
        for family in PRIMARY_FAMILIES:
            for scramble_index in range(2):
                preservation_rows.append({"group_seed": group, "family": family, "accepted_scramble": True})
                run_rows.extend(
                    [
                        {
                            "group_seed": group,
                            "family": family,
                            "method": "naive_factor_average",
                            "relative_delta_change_from_unscrambled": 0.2,
                            "validation_accuracy_change": -0.1,
                            "output_rank_cap": 4,
                        },
                        {
                            "group_seed": group,
                            "family": family,
                            "method": "global_synchronization",
                            "relative_delta_change_from_unscrambled": 1e-12,
                            "validation_accuracy_change": 0.0,
                            "output_rank_cap": 4,
                        },
                    ]
                )
    gates = phase_a_gates(pd.DataFrame(run_rows), pd.DataFrame(preservation_rows), (0, 1, 2))
    assert all(gates.values())
