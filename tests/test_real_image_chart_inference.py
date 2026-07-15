import numpy as np

from experiments.real_image_chart_inference import inverse, transform


def test_d4_image_transforms_are_exactly_invertible():
    image = np.arange(25).reshape(5, 5)
    for chart in range(8):
        assert np.array_equal(inverse(transform(image, chart), chart), image)


def test_d4_image_transforms_preserve_rgb_channels():
    image = np.arange(3 * 5 * 5).reshape(3, 5, 5)
    for chart in range(8):
        transformed = transform(image, chart)
        assert transformed.shape == image.shape
        assert np.array_equal(inverse(transformed, chart), image)
