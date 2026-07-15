import numpy as np

from experiments.input_inferred_chart_recovery import inverse, split_indices, transform


def test_d4_transform_inverse_roundtrip():
    image = np.arange(64).reshape(8, 8)
    for chart in range(8):
        assert np.array_equal(inverse(transform(image, chart), chart), image)


def test_data_roles_are_disjoint():
    split = split_indices(10_000, 3)
    sets = [set(values.tolist()) for values in split.values()]
    assert sum(len(values) for values in sets) == len(set().union(*sets))
