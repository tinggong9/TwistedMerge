import numpy as np

from experiments.complete_natural_stability_gate import residual_statistics


def test_fixed_stability_protocol_uses_five_resamples_and_200_nulls():
    rng = np.random.default_rng(7); base = rng.normal(size=(500, 5))
    calibration = [base @ np.linalg.qr(rng.normal(size=(5, 5)))[0] for _ in range(4)]
    resamples, nulls, summary = residual_statistics(calibration, "orthogonal", 0)
    assert len(resamples) == 5
    assert len(nulls) == 4 * 200
    assert summary["selected_family"] == "orthogonal"
