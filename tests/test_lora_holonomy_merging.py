from __future__ import annotations

import numpy as np

from experiments.lora_holonomy_merging import factor_delta, gauge_transform, random_invertible


def test_lora_gauge_transform_preserves_delta() -> None:
    rng = np.random.default_rng(3)
    b = rng.normal(size=(7, 3))
    a = rng.normal(size=(3, 5))
    transformed = gauge_transform(b, a, random_invertible(rng, 3))
    assert np.allclose(factor_delta(b, a), factor_delta(*transformed))
