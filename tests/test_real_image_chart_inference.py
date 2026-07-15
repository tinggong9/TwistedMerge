import numpy as np

from experiments.real_image_chart_inference import inverse, transform


def test_d4_image_transforms_are_exactly_invertible():
    image = np.arange(25).reshape(5, 5)
    for chart in range(8):
        assert np.array_equal(inverse(transform(image, chart), chart), image)
