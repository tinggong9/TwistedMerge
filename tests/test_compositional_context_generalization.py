import numpy as np

from experiments.compositional_context_generalization import encode


def test_sequence_encoding_preserves_order():
    encoded = encode([("r", "s"), ("s", "r")])
    assert encoded.shape == (2, 16)
    assert not np.array_equal(encoded[0], encoded[1])
