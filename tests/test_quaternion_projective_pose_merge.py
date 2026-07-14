from __future__ import annotations

import numpy as np

from experiments.quaternion_projective_pose_merge import (
    markley_mean,
    normalize_quaternion,
    pose_metrics,
    quaternion_to_rotation,
)


def test_quaternion_sign_is_invisible_in_so3() -> None:
    rng = np.random.default_rng(1)
    quaternion = normalize_quaternion(rng.normal(size=(20, 4)))
    assert np.allclose(quaternion_to_rotation(quaternion), quaternion_to_rotation(-quaternion))


def test_quadratic_mean_handles_opposite_lifts() -> None:
    target = normalize_quaternion(np.array([[1.0, 0.2, -0.1, 0.3]]))
    observations = np.stack([target, -target, target, -target], axis=1)
    prediction = markley_mean(observations)
    error, accuracy = pose_metrics(prediction, target)
    assert error < 1e-6
    assert accuracy == 1.0
