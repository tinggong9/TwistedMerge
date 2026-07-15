from __future__ import annotations

import numpy as np

from experiments.compact_federated_frame import observed_edges, quarter_turn_frames, synchronize_frames


def test_exact_frame_edges_are_synchronizable() -> None:
    frames = quarter_turn_frames()
    edges = observed_edges(frames, "exact", np.random.default_rng(0))
    estimates = synchronize_frames(edges)
    for (i, j), edge in edges.items():
        assert np.linalg.norm(edge - estimates[i].T @ estimates[j]) < 1e-8
