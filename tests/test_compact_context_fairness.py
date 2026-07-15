from __future__ import annotations

import numpy as np

from experiments.compact_context_fairness import corrupted_context, make_setting


def test_compact_context_setting_is_deterministic() -> None:
    first = make_setting("S3", 32, 2)
    second = make_setting("S3", 32, 2)
    assert np.array_equal(first["labels_test"], second["labels_test"])
    assert first["x_train"].shape == (1024, 32)


def test_context_corruption_endpoints() -> None:
    indices = np.arange(6)
    exact, observed = corrupted_context(indices, 6, 0.0, np.random.default_rng(1))
    assert np.array_equal(observed, indices)
    assert np.array_equal(exact.argmax(1), indices)
