from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiments.lora_factor_space_scalability import (
    FACTOR_SPACE_METHODS,
    METHODS,
    TWISTED_FACTOR_METHODS,
    DenseAllocationTracker,
    analytical_factor_bytes,
    benchmark_gates,
    canonical_low_rank_factors,
    correctness_probes,
    dense_mean,
    generate_factors,
    low_rank_distance,
    merge_case,
)
from src.lora_gauge_alignment import gauge_transform, sample_gauge


def fixture(dimension=48, rank=4, adapters=4):
    return generate_factors(dimension, rank, adapters, 71, 0.08, 0.06)


def materialize(factor):
    return factor[0] @ factor[1]


def test_dense_sentinel_rejects_factor_method_allocation():
    tracker = DenseAllocationTracker(allow_dense=False)
    with pytest.raises(RuntimeError, match="dense effective-update"):
        tracker.record((32, 32), np.dtype(np.float32))


def test_factor_methods_report_zero_dense_allocations():
    factors = fixture()
    for method in FACTOR_SPACE_METHODS:
        output = merge_case(factors, method, seed=8)
        assert output.tracker.dense_allocation_count == 0
        assert output.tracker.dense_allocation_bytes == 0


def test_dense_methods_explicitly_record_materialization():
    factors = fixture()
    for method in ("dense_deterministic_truncated_svd", "dense_randomized_svd"):
        output = merge_case(factors, method, seed=8)
        assert output.tracker.dense_allocation_count == 1
        assert output.tracker.dense_allocation_bytes == 48 * 48 * 4


def test_dense_mean_matches_explicit_average():
    factors = fixture()
    tracker = DenseAllocationTracker(allow_dense=True)
    observed = dense_mean(factors, tracker)
    expected = np.mean([materialize(factor) for factor in factors], axis=0)
    assert np.allclose(observed, expected, atol=2e-6, rtol=2e-6)


def test_canonical_low_rank_factors_preserve_update():
    for factor in fixture():
        canonical = canonical_low_rank_factors(factor)
        assert np.allclose(materialize(canonical), materialize(factor), atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize("method", TWISTED_FACTOR_METHODS)
def test_twisted_factor_methods_are_gauge_invariant(method):
    factors = fixture()
    rng = np.random.default_rng(90)
    gauges = [sample_gauge(rng, 4, "dense", 30).astype(np.float32) for _ in factors]
    changed = [gauge_transform(*factor, gauge) for factor, gauge in zip(factors, gauges)]
    before = merge_case(factors, method, seed=9).factors
    after = merge_case(changed, method, seed=9).factors
    probes = rng.normal(size=(12, 48)).astype(np.float32)
    before_values = (probes @ before[1].T) @ before[0].T
    after_values = (probes @ after[1].T) @ after[0].T
    assert np.linalg.norm(before_values - after_values) / np.linalg.norm(before_values) < 2e-4


def test_all_methods_return_requested_rank():
    factors = fixture(rank=4)
    for method in METHODS:
        output = merge_case(factors, method, seed=9).factors
        assert output[0].shape[1] == 4
        assert output[0].dtype == np.float32
        assert output[1].dtype == np.float32


def test_low_rank_distance_matches_dense_distance():
    first, second = fixture(adapters=2)
    assert np.isclose(low_rank_distance(first, second), np.linalg.norm(materialize(first) - materialize(second)), rtol=2e-5)


def test_correctness_probe_never_materializes_dense_reference():
    factors = fixture()
    output = merge_case(factors, "global_synchronization", seed=9)
    result = correctness_probes(factors, output, "global_synchronization", 9)
    assert not result["dense_reference_materialized_for_correctness"]
    assert result["reference_mode"] == "exact_low_rank_algebraic_identity"
    assert result["gauge_invariance_probe_error"] < 2e-4


def test_analytical_memory_separates_dense_and_factor_space():
    dense = analytical_factor_bytes(4096, 8, 8, "dense_deterministic_truncated_svd")
    factor = analytical_factor_bytes(4096, 8, 8, "global_synchronization")
    assert factor < dense / 2


def test_benchmark_gate_uses_dense_counts_and_grouped_memory():
    run_rows = []
    correctness_rows = []
    memory_rows = []
    timing_rows = []
    for method in METHODS:
        dense_count = 0 if method in FACTOR_SPACE_METHODS else 1
        run_rows.append({"method": method, "dense_effective_update_allocations": dense_count})
        correctness_rows.append({"method": method, "gauge_invariance_probe_error": 1e-6, "output_rank": 4, "rank": 4})
        for rank in (4, 8, 16, 32):
            for adapter_count in (4, 8, 16):
                memory_rows.append(
                    {
                        "dimension_m": 4096,
                        "rank": rank,
                        "adapter_count": adapter_count,
                        "method": method,
                        "temporary_allocation_bytes_analytical": 10 if method == "global_synchronization" else 100,
                        "incremental_peak_rss_bytes": 10 if method == "global_synchronization" else 100,
                    }
                )
                timing_rows.append(
                    {
                        "dimension_m": 4096,
                        "rank": rank,
                        "adapter_count": adapter_count,
                        "method": method,
                        "median_wall_seconds": 0.1 if method == "global_synchronization" else 1.0,
                    }
                )
    gates = benchmark_gates(
        pd.DataFrame(run_rows),
        pd.DataFrame(timing_rows),
        pd.DataFrame(memory_rows),
        pd.DataFrame(correctness_rows),
    )
    assert gates["positive_scalability_gate"]
    assert gates["crossover_dimension"] == 4096
