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

    def test_invalid_scale_rejected(self):
        with self.assertRaises(ValueError):
            apply_channel_gauge(self.model, conv1_scales=np.zeros(16))


if __name__ == "__main__":
    unittest.main()
