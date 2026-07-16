import numpy as np

from experiments.segmentation_transition_geometry import cycle_residual, normalized_fit, orthogonal_map


def test_orthogonal_procrustes_recovers_rotation():
    rng = np.random.default_rng(7)
    source = rng.normal(size=(64, 5))
    rotation = np.linalg.qr(rng.normal(size=(5, 5)))[0]
    target = source @ rotation
    fitted = orthogonal_map(source, target)
    assert normalized_fit(source, target, fitted) < 1e-10


def test_cycle_residual_zero_for_coboundary_maps():
    rng = np.random.default_rng(11)
    gauges = [np.linalg.qr(rng.normal(size=(4, 4)))[0] for _ in range(3)]
    assert cycle_residual(gauges[0].T @ gauges[1], gauges[1].T @ gauges[2], gauges[0].T @ gauges[2]) < 1e-10
