from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.natural_twist_discovery import leave_one_setting_out_mse, permutation_test


def test_permutation_test_detects_strong_fixed_relation() -> None:
    x = np.linspace(-1, 1, 100)
    corr, p_value = permutation_test(x, 2 * x + 0.01 * np.sin(x), samples=200, seed=2)
    assert corr > 0.99
    assert p_value < 0.02


def test_loso_predictor_is_out_of_sample() -> None:
    frame = pd.DataFrame({"a": np.arange(20, dtype=float), "b": np.arange(20, dtype=float) ** 2})
    assert leave_one_setting_out_mse(frame, ["a"], "b") > 0
