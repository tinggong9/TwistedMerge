import unittest
from pathlib import Path

import numpy as np

from experiments.block_gauge_phase_diagram import relu_diagnostic_rows
from src.block_compatible_merge import (
    average_linear_hidden_models,
    make_linear_hidden_mlp,
    max_logit_difference,
    parameter_count,
    transform_linear_hidden_block_gauge,
)
from src.block_gauge_alignment import BlockPartition
from src.model_merging_benchmark import require_torch


ROOT = Path(__file__).resolve().parents[1]


def rotation(theta: float) -> np.ndarray:
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


class Args:
    reports_dir = ROOT / "reports"


class BlockCompatibleLearningTaskTests(unittest.TestCase):
    def test_block_compatible_linear_hidden_transform_and_average(self):
        torch, _, _ = require_torch()
        torch.manual_seed(5)
        model = make_linear_hidden_mlp(input_dim=6, width=4, num_classes=3)
        partition = BlockPartition("contiguous", 2, ((0, 1), (2, 3)))
        gauges = {0: rotation(0.5), 1: rotation(-0.2)}
        inverse = {idx: matrix.T for idx, matrix in gauges.items()}
        transformed, metadata = transform_linear_hidden_block_gauge(model, partition, gauges)
        aligned, _metadata = transform_linear_hidden_block_gauge(transformed, partition, inverse)
        merged = average_linear_hidden_models([model, aligned])
        inputs = torch.randn(12, 6)

        self.assertLess(max_logit_difference(model, transformed, inputs), 1e-6)
        self.assertLess(max_logit_difference(model, merged, inputs), 1e-6)
        self.assertEqual(parameter_count(model), parameter_count(merged))
        self.assertTrue(metadata.exact_same_architecture_symmetry)

    def test_relu_block_rows_are_diagnostic_only(self):
        rows = relu_diagnostic_rows(Args())
        if rows.empty:
            self.skipTest("prior ReLU diagnostic CSV is not available")
        self.assertFalse(rows["exact_same_architecture_symmetry"].any())
        self.assertTrue(rows["diagnostic_only"].all())
        self.assertFalse(rows["block_merge_accuracy_reported"].any())


if __name__ == "__main__":
    unittest.main()
