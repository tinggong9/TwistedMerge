import numpy as np

from experiments.full_model_hidden_geometry import (
    GAUGES,
    cycle_statistics,
    fit_map,
    hodge_components,
    maps_for,
    null_draws,
)


def test_hidden_geometry_maps_hodge_resamples_and_nulls_are_executed():
    rng = np.random.default_rng(3)
    vertex = [rng.normal(size=(64, 8)) for _ in range(4)]
    for gauge in GAUGES:
        matrix = fit_map(vertex[0], vertex[1], gauge)
        assert matrix.shape == (8, 8)
    maps = maps_for(vertex, "orthogonal_procrustes")
    statistics = cycle_statistics(maps)
    hodge = hodge_components(maps)
    nulls = null_draws(maps, observed_fit=0.1, seed=4, draws=2)
    assert statistics["cycle_residual"] >= 0
    assert set(hodge) == {"hodge_exact", "hodge_coexact", "hodge_harmonic", "distance_to_coboundaries"}
    assert len(nulls) == 2 * 5
    assert {row["null_family"] for row in nulls} == {
        "edge_shuffle", "matched_norm_coboundary", "matched_fit_random_gauge", "graph_topology_shuffle", "calibration_label_independent_bootstrap"
    }
