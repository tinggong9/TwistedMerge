import numpy as np
import torch

from experiments.spatial_output_common import (
    compose_d4,
    d4_matrix,
    inverse_chart,
    transform_points,
    transform_vector_field,
)


def test_d4_vector_matrices_obey_composition_and_inverse():
    for left in range(8):
        assert np.array_equal(d4_matrix(left) @ d4_matrix(inverse_chart(left)), np.eye(2, dtype=np.int64))
        for right in range(8):
            assert np.array_equal(d4_matrix(left) @ d4_matrix(right), d4_matrix(compose_d4(left, right)))


def test_points_and_vector_components_recover_under_inverse():
    points = np.asarray([[3.0, 7.0], [19.0, 12.0]])
    field = torch.randn(2, 2, 21, 21)
    for chart in range(8):
        inverse = inverse_chart(chart)
        assert np.allclose(transform_points(transform_points(points, chart, 21), inverse, 21), points)
        assert torch.allclose(transform_vector_field(transform_vector_field(field, chart), inverse), field)
