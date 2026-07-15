import numpy as np

from experiments.full_model_transition_geometry import orthogonal_map, transition_diagnostics


def test_orthogonal_transition_recovers_exact_map():
    rng = np.random.default_rng(4)
    source = rng.normal(size=(600, 8)); target_map = np.linalg.qr(rng.normal(size=(8, 8)))[0]
    fitted = orthogonal_map(source, source @ target_map)
    assert np.linalg.norm(fitted - target_map) < 1e-8


def test_transition_diagnostic_runs_five_resamples_and_null_gate():
    rng = np.random.default_rng(5); base = rng.normal(size=(600, 8))
    representations = [base @ np.linalg.qr(rng.normal(size=(8, 8)))[0] for _ in range(4)]
    rows, diagnostic, maps = transition_diagnostics(representations, 0, "test")
    assert len(rows) == 5
    assert diagnostic["selected_family"] in {row["alignment_family"] for row in rows}
    assert len(maps) == 16
