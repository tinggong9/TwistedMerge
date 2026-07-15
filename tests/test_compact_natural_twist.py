from __future__ import annotations

import numpy as np
import torch

from experiments.compact_natural_twist import CompactMLP, SmallCNN, orthogonal_map, permutation_map


def test_compact_models_have_common_output() -> None:
    inputs = torch.zeros(3, 1, 28, 28)
    assert CompactMLP()(inputs).shape == (3, 10)
    assert SmallCNN()(inputs).shape == (3, 10)


def test_transition_maps_have_expected_shape() -> None:
    rng = np.random.default_rng(2)
    source = rng.normal(size=(50, 10))
    target = source[:, rng.permutation(10)]
    assert orthogonal_map(source, target).shape == (10, 10)
    permutation = permutation_map(source, target)
    assert np.allclose(permutation.sum(0), 1)
    assert np.allclose(permutation.sum(1), 1)
