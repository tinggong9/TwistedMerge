import unittest

import numpy as np

from src.model_merging_benchmark import (
    DatasetSpec,
    make_model,
    permute_model_to_reference,
    require_torch,
    set_seed,
)


def invert_perm(perm: np.ndarray) -> np.ndarray:
    inv = np.empty_like(perm)
    inv[perm] = np.arange(len(perm))
    return inv


class StrongerModelMergingArchitecturesTests(unittest.TestCase):
    def test_mlp2_layerwise_permutation_is_function_preserving_and_invertible(self):
        torch, _, _ = require_torch()
        set_seed(123)
        spec = DatasetSpec("toy", (1, 4, 4), 3)
        base = make_model("mlp2", spec, width=5)
        p1 = np.array([2, 0, 4, 1, 3])
        p2 = np.array([1, 3, 0, 4, 2])
        permuted = permute_model_to_reference(base, "mlp2", spec, 5, {"hidden1": p1, "hidden2": p2})
        x = torch.randn(7, 1, 4, 4)
        with torch.no_grad():
            self.assertTrue(torch.allclose(base(x), permuted(x), atol=1e-6))
        recovered = permute_model_to_reference(
            permuted,
            "mlp2",
            spec,
            5,
            {"hidden1": invert_perm(p1), "hidden2": invert_perm(p2)},
        )
        with torch.no_grad():
            self.assertTrue(torch.allclose(base(x), recovered(x), atol=1e-6))

    def test_small_cnn_layerwise_permutation_is_function_preserving_and_invertible(self):
        torch, _, _ = require_torch()
        set_seed(456)
        spec = DatasetSpec("toy", (1, 8, 8), 4)
        base = make_model("small_cnn", spec, width=4)
        p1 = np.array([2, 0, 3, 1])
        p2 = np.array([1, 3, 0, 2])
        permuted = permute_model_to_reference(base, "small_cnn", spec, 4, {"conv1": p1, "conv2": p2})
        x = torch.randn(5, 1, 8, 8)
        with torch.no_grad():
            self.assertTrue(torch.allclose(base(x), permuted(x), atol=1e-6))
        recovered = permute_model_to_reference(
            permuted,
            "small_cnn",
            spec,
            4,
            {"conv1": invert_perm(p1), "conv2": invert_perm(p2)},
        )
        with torch.no_grad():
            self.assertTrue(torch.allclose(base(x), recovered(x), atol=1e-6))


if __name__ == "__main__":
    unittest.main()
