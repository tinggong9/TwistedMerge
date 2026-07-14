from __future__ import annotations

import numpy as np

from experiments.federated_sensor_frame_merge import canonicalize_weight, rotation_permutation


def test_canonicalized_rotated_frame_preserves_logits() -> None:
    rng = np.random.default_rng(4)
    x = rng.normal(size=(20, 28 * 28))
    weight = rng.normal(size=(10, 28 * 28))
    permutation = rotation_permutation(1)
    canonical = canonicalize_weight(weight, permutation)
    assert np.allclose(x[:, permutation] @ weight.T, x @ canonical.T)
