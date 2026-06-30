import unittest

import numpy as np
import torch

from src.improved_monomial_merge import build_scaled_average_model, shrink_log_scales
from src.ladder_merge_methods import transform_mlp_positive_scale
from src.model_merging_benchmark import DatasetSpec, average_models, make_model, permute_model_to_reference, set_seed


def assert_same_state(testcase, left, right, atol=1e-8):
    for key in left.state_dict():
        testcase.assertTrue(
            torch.allclose(left.state_dict()[key], right.state_dict()[key], atol=atol),
            msg=f"state mismatch at {key}",
        )


class MonomialShrinkageTests(unittest.TestCase):
    def setUp(self):
        self.spec = DatasetSpec(name="toy", input_shape=(1, 2, 2), num_classes=3)
        self.width = 4
        set_seed(123)
        self.models = [make_model("mlp", self.spec, self.width) for _ in range(2)]
        self.perms = {0: np.arange(self.width), 1: np.array([2, 0, 3, 1])}
        self.raw_logs = np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                np.log([1.5, 0.8, 2.0, 0.6]),
            ]
        )

    def test_alpha_zero_recovers_c2m3_aligned_average(self):
        logs = shrink_log_scales(self.raw_logs, alpha=0.0, tau=float("inf"))
        scaled_average = build_scaled_average_model(self.models, self.spec, self.width, self.perms, logs)
        aligned = [
            permute_model_to_reference(model, "mlp", self.spec, self.width, self.perms[idx])
            for idx, model in enumerate(self.models)
        ]
        c2m3_average = average_models(aligned, "mlp", self.spec, self.width)

        assert_same_state(self, scaled_average, c2m3_average)

    def test_alpha_one_recovers_previous_monomial_scaling(self):
        logs = shrink_log_scales(self.raw_logs, alpha=1.0, tau=float("inf"))
        scaled_average = build_scaled_average_model(self.models, self.spec, self.width, self.perms, logs)
        previous_scaled = [
            transform_mlp_positive_scale(
                model,
                self.spec,
                self.width,
                self.perms[idx],
                np.exp(self.raw_logs[idx]),
            )
            for idx, model in enumerate(self.models)
        ]
        previous_average = average_models(previous_scaled, "mlp", self.spec, self.width)

        assert_same_state(self, scaled_average, previous_average)

    def test_positive_scale_transform_preserves_outputs(self):
        x = torch.randn(7, *self.spec.input_shape)
        scales = np.exp(np.array([0.2, -0.4, 0.1, 0.3]))
        identity = np.arange(self.width)
        transformed = transform_mlp_positive_scale(self.models[0], self.spec, self.width, identity, scales)

        with torch.no_grad():
            before = self.models[0](x)
            after = transformed(x)

        self.assertTrue(torch.allclose(before, after, atol=1e-6))

    def test_log_scale_clipping_and_shrinkage(self):
        logs = np.array([[-2.0, -0.2, 0.5, 3.0]])
        shrunk = shrink_log_scales(logs, alpha=0.5, tau=1.0)

        np.testing.assert_allclose(shrunk, np.array([[-0.5, -0.1, 0.25, 0.5]]))


if __name__ == "__main__":
    unittest.main()
