import numpy as np

from experiments.realistic_multiview_twist import fixed_views, normalize_points, view_feature


def test_multiview_features_are_executed_from_points():
    rng = np.random.default_rng(8); points = rng.normal(size=(100, 3))
    normalized = normalize_points(points)
    assert normalized.shape == (256, 3)
    features = [view_feature(normalized @ rotation.T) for rotation in fixed_views()]
    assert all(feature.shape == features[0].shape for feature in features)
    assert any(not np.allclose(features[0], feature) for feature in features[1:])
