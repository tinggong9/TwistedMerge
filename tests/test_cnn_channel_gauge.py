import unittest

import numpy as np

from src.cnn_channel_gauge import (
    apply_channel_gauge,
    count_parameters,
    inference_cost_units,
    make_small_fashion_cnn,
)
from src.model_merging_benchmark import require_torch, set_seed


class CnnChannelGaugeTests(unittest.TestCase):
    def setUp(self):
        torch, _, _ = require_torch()
        set_seed(44)
        self.model = make_small_fashion_cnn()
        self.x = torch.randn(5, 1, 28, 28)

    def assert_logits_preserved(self, transformed, atol=1e-5):
        torch, _, _ = require_torch()
        with torch.no_grad():
            original = self.model(self.x)
            changed = transformed(self.x)
        self.assertTrue(torch.allclose(original, changed, atol=atol, rtol=atol))
        self.assertEqual(count_parameters(self.model), count_parameters(transformed))
        self.assertEqual(inference_cost_units(), inference_cost_units(transformed.gauge_spec))

    def test_channel_permutation_gauge_preserves_logits(self):
        transformed = apply_channel_gauge(
            self.model,
            conv1_perm=np.random.default_rng(1).permutation(16),
            conv2_perm=np.random.default_rng(2).permutation(32),
            hidden_perm=np.random.default_rng(3).permutation(128),
        )

        self.assert_logits_preserved(transformed)

    def test_positive_channel_scaling_gauge_preserves_logits(self):
        rng = np.random.default_rng(4)
        transformed = apply_channel_gauge(
            self.model,
            conv1_scales=np.exp(rng.normal(0.0, 0.25, size=16)),
            conv2_scales=np.exp(rng.normal(0.0, 0.25, size=32)),
            hidden_scales=np.exp(rng.normal(0.0, 0.25, size=128)),
        )

        self.assert_logits_preserved(transformed)

    def test_combined_permutation_and_scaling_gauge_preserves_logits(self):
        rng = np.random.default_rng(5)
        transformed = apply_channel_gauge(
            self.model,
            conv1_perm=rng.permutation(16),
            conv2_perm=rng.permutation(32),
            hidden_perm=rng.permutation(128),
            conv1_scales=np.exp(rng.normal(0.0, 0.18, size=16)),
            conv2_scales=np.exp(rng.normal(0.0, 0.18, size=32)),
            hidden_scales=np.exp(rng.normal(0.0, 0.18, size=128)),
        )

        self.assert_logits_preserved(transformed)

    def test_conv_to_conv_channel_transform_formula(self):
        torch, _, _ = require_torch()
        conv1_perm = np.array([3, 1, 2, 0, *range(4, 16)])
        conv2_perm = np.array([2, 0, 1, 3, *range(4, 32)])
        conv1_scales = np.linspace(0.75, 1.50, 16)
        conv2_scales = np.linspace(0.60, 1.40, 32)

        transformed = apply_channel_gauge(
            self.model,
            conv1_perm=conv1_perm,
            conv2_perm=conv2_perm,
            conv1_scales=conv1_scales,
            conv2_scales=conv2_scales,
        )

        with torch.no_grad():
            expected = (
                self.model.conv2.weight.detach()[conv2_perm][:, conv1_perm, :, :]
                * torch.tensor(conv2_scales, dtype=self.model.conv2.weight.dtype).view(-1, 1, 1, 1)
                / torch.tensor(conv1_scales, dtype=self.model.conv2.weight.dtype).view(1, -1, 1, 1)
            )
        self.assertTrue(torch.allclose(transformed.conv2.weight, expected, atol=1e-6))

    def test_conv_to_linear_flattened_blocks_transform_formula(self):
        torch, _, _ = require_torch()
        conv2_perm = np.array([5, 2, 7, 1, 0, 3, 4, 6, *range(8, 32)])
        conv2_scales = np.linspace(0.70, 1.30, 32)

        transformed = apply_channel_gauge(
            self.model,
            conv2_perm=conv2_perm,
            conv2_scales=conv2_scales,
        )

        block = self.model.gauge_spec.conv2_block_size
        with torch.no_grad():
            for new_channel, old_channel in enumerate(conv2_perm):
                new_slice = slice(new_channel * block, (new_channel + 1) * block)
                old_slice = slice(int(old_channel) * block, int(old_channel + 1) * block)
                expected = self.model.fc1.weight.detach()[:, old_slice] / float(conv2_scales[new_channel])
                self.assertTrue(torch.allclose(transformed.fc1.weight[:, new_slice], expected, atol=1e-6))

    def test_linear_hidden_scaling_formula(self):
        torch, _, _ = require_torch()
        hidden_perm = np.array([4, 1, 3, 2, 0, *range(5, 128)])
        hidden_scales = np.linspace(0.65, 1.45, 128)

        transformed = apply_channel_gauge(
            self.model,
            hidden_perm=hidden_perm,
            hidden_scales=hidden_scales,
        )

        scale_tensor = torch.tensor(hidden_scales, dtype=self.model.fc1.weight.dtype)
        with torch.no_grad():
            expected_fc1 = self.model.fc1.weight.detach()[hidden_perm] * scale_tensor.view(-1, 1)
            expected_classifier = self.model.classifier.weight.detach()[:, hidden_perm] / scale_tensor.view(1, -1)
        self.assertTrue(torch.allclose(transformed.fc1.weight, expected_fc1, atol=1e-6))
        self.assertTrue(torch.allclose(transformed.classifier.weight, expected_classifier, atol=1e-6))

    def test_invalid_scale_rejected(self):
        with self.assertRaises(ValueError):
            apply_channel_gauge(self.model, conv1_scales=np.zeros(16))


if __name__ == "__main__":
    unittest.main()
