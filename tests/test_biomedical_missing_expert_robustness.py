import numpy as np

from experiments.biomedical_missing_expert_robustness import synchronize_scalar_edges


def test_scalar_graph_synchronization_recovers_exact_offsets():
    potential = np.asarray([0.0, 1.2, -0.4, 2.0])
    edges = [(left, right, potential[right] - potential[left]) for left in range(4) for right in range(left + 1, 4)]
    recovered, residual = synchronize_scalar_edges(edges)
    assert np.allclose(recovered, potential)
    assert np.max(np.abs(residual)) < 1e-10


def test_corrupted_edge_has_nonzero_hodge_residual():
    edges = [(0, 1, 1.0), (1, 2, 1.0), (0, 2, 5.0)]
    _, residual = synchronize_scalar_edges(edges, nodes=3)
    assert np.linalg.norm(residual) > 0.1
