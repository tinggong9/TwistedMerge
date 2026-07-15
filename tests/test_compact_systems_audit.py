from __future__ import annotations

import numpy as np

from experiments.compact_systems_audit import kl_divergence


def test_kl_is_zero_for_identical_logits() -> None:
    logits = np.array([[1.0, -1.0], [0.2, 0.1]])
    assert abs(kl_divergence(logits, logits)) < 1e-12
