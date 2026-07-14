from __future__ import annotations

import numpy as np

from src.twist_subspace import bootstrap_rank_stability, extract_twist_subspace, project_residual


def test_extracts_rank_one_persistent_subspace() -> None:
    left = np.linspace(1, 2, 20)[:, None]
    right = np.array([[1.0, -2.0, 0.5]])
    data = left @ right
    result = extract_twist_subspace(data, epsilon=1e-10)
    assert result.chosen_rank == 1
    assert result.explained_energy > 0.999999
    assert np.allclose(project_residual(data, result), data)
    assert bootstrap_rank_stability(data, epsilon=1e-10, samples=20) == {1: 1.0}
