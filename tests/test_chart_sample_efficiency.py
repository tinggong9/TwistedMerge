import pytest

from experiments.chart_sample_efficiency import BUDGETS, first_crossing


def test_sample_efficiency_budgets_and_interpolation():
    assert BUDGETS == (32, 64, 128, 256, 512, 1000)
    assert first_crossing([(32, 0.6), (64, 0.8)], 0.7) == pytest.approx(48.0)
    assert first_crossing([(32, 0.6), (64, 0.65)], 0.7) == "not_reached"
