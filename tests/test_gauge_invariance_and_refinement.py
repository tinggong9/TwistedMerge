import numpy as np

from experiments.gauge_invariance_and_refinement import gauge_transition, transition_metrics, transitions_from_vertices


def test_vertex_gauge_law_and_cycle_invariance():
    rng = np.random.default_rng(1)
    vertices = [np.linalg.qr(rng.normal(size=(4, 4)))[0] for _ in range(3)]
    applied = [np.linalg.qr(rng.normal(size=(4, 4)))[0] for _ in range(3)]
    transitions = transitions_from_vertices(vertices)
    transformed = {(i, j): gauge_transition(transitions[i, j], applied[i], applied[j]) for i, j in transitions}
    expected = transitions_from_vertices([applied[i] @ vertices[i] for i in range(3)])
    assert max(np.linalg.norm(transformed[key] - expected[key]) for key in expected) < 1e-10
    assert transition_metrics(transformed)["face_residual"] < 1e-10
