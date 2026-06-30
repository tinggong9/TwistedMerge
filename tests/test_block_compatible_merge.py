import unittest

import numpy as np

from src.block_compatible_merge import (
    average_linear_hidden_models,
    make_linear_hidden_mlp,
    max_logit_difference,
    transform_linear_hidden_block_gauge,
)
from src.block_gauge_alignment import BlockPartition
from src.model_merging_benchmark import require_torch


def rotation(theta: float) -> np.ndarray:
    return np.array(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ]
    )


class BlockCompatibleMergeTests(unittest.TestCase):
    def test_linear_hidden_block_rotation_preserves_logits_and_parameter_count(self):
        torch, _, _ = require_torch()
        torch.manual_seed(123)
        model = make_linear_hidden_mlp(input_dim=5, width=4, num_classes=3)
        partition = BlockPartition("contiguous", 2, ((0, 1), (2, 3)))
        gauges = {0: rotation(0.4), 1: rotation(-0.7)}
        inputs = torch.randn(11, 5)

        transformed, metadata = transform_linear_hidden_block_gauge(model, partition, gauges)

        self.assertLess(max_logit_difference(model, transformed, inputs), 1e-6)
        self.assertTrue(metadata.same_parameter_count)
        self.assertTrue(metadata.exact_same_architecture_symmetry)
        self.assertFalse(metadata.adapter_extra_parameters)
        self.assertEqual(metadata.activation, "identity")

    def test_aligned_block_compatible_average_recovers_base_function(self):
        torch, _, _ = require_torch()
        torch.manual_seed(321)
        model = make_linear_hidden_mlp(input_dim=6, width=4, num_classes=2)
        partition = BlockPartition("contiguous", 2, ((0, 1), (2, 3)))
        gauges = {0: rotation(0.8), 1: rotation(-0.3)}
        inverse_gauges = {0: gauges[0].T, 1: gauges[1].T}
        transformed, _metadata = transform_linear_hidden_block_gauge(model, partition, gauges)
        aligned_back, _metadata_back = transform_linear_hidden_block_gauge(transformed, partition, inverse_gauges)
        merged = average_linear_hidden_models([model, aligned_back])
        inputs = torch.randn(9, 6)

        self.assertLess(max_logit_difference(model, aligned_back, inputs), 1e-6)
        self.assertLess(max_logit_difference(model, merged, inputs), 1e-6)


if __name__ == "__main__":
    unittest.main()
