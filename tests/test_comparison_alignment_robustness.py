import numpy as np

from experiments.comparison_alignment_robustness import hodge, incidence, run_comparison_complexes


def test_complex_hodge_and_false_positive_protocol_execute():
    b1, b2 = incidence(3, [(0, 1), (1, 2), (0, 2)], [(0, 1, 2)])
    components = hodge(b1.T @ np.asarray([0.0, 1.0, 2.0]), b1, b2)
    assert components["distance_to_coboundaries"] < 1e-10
    rows, summary, scaling = run_comparison_complexes(trials=2)
    assert len(summary) == 7
    assert len(rows) == 7 * 2 * 2
    assert len(scaling) == len(rows)
