import numpy as np

from experiments.quaternion_pose_near_term import cycle_diagnostics, parse_off, pose_branches, random_rotation
from experiments.quaternion_projective_pose_merge import pose_metrics, rotation_to_quaternion


def test_parse_off_normalizes_real_mesh_shape():
    mesh = parse_off(b"OFF\n4 4 0\n0 0 0\n1 0 0\n0 2 0\n0 0 3\n3 0 1 2\n3 0 1 3\n3 0 2 3\n3 1 2 3\n")
    assert mesh.shape == (4, 3)
    assert np.allclose(mesh.mean(axis=0), 0)
    assert np.isclose(np.sqrt(np.mean(np.sum(mesh * mesh, axis=1))), 1)


def test_mesh_moment_branches_include_generated_rotation():
    rng = np.random.default_rng(22)
    mesh = rng.normal(size=(200, 3)) * np.asarray([3.0, 2.0, 1.0])
    rotation = random_rotation(rng)
    branches, scores = pose_branches(mesh, mesh @ rotation.T)
    target = rotation_to_quaternion(rotation[None])
    best = min(pose_metrics(rotation_to_quaternion(branch[None]), target)[0] for branch in branches)
    selected = pose_metrics(rotation_to_quaternion(branches[np.argmin(scores)][None]), target)[0]
    assert best < 1e-5
    assert selected < 1e-5


def test_quaternion_chart_signs_have_null_cycle_residual():
    quaternion = np.asarray([0.5, 0.5, 0.5, 0.5])
    negative_rate, so3_residual = cycle_diagnostics(quaternion)
    assert negative_rate == 0
    assert so3_residual < 1e-12
