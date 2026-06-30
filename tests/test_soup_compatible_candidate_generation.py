import unittest

import numpy as np

from src.improved_monomial_merge import (
    assert_capacity_matched_mlp,
    build_scaled_average_model,
    greedy_soup_with_metadata,
    shrink_log_scales,
)
from src.model_merging_benchmark import (
    DatasetSpec,
    average_models,
    make_loader,
    make_model,
    permute_model_to_reference,
    require_torch,
    set_seed,
)


def assert_same_state(testcase, left, right, atol=1e-8):
    for key in left.state_dict():
        testcase.assertTrue(
            require_torch()[0].allclose(left.state_dict()[key], right.state_dict()[key], atol=atol),
            msg=f"state mismatch at {key}",
        )


class SoupCompatibleCandidateGenerationTests(unittest.TestCase):
    def setUp(self):
        torch, _, _ = require_torch()
        self.spec = DatasetSpec(name="toy", input_shape=(1, 2, 2), num_classes=3)
        self.width = 4
        set_seed(44)
        self.models = [make_model("mlp", self.spec, self.width) for _ in range(2)]
        self.perms = {0: np.arange(self.width), 1: np.array([2, 0, 3, 1])}
        self.raw_logs = np.array([[0.0, 0.0, 0.0, 0.0], np.log([1.4, 0.7, 1.9, 0.8])])
        x = torch.randn(16, *self.spec.input_shape)
        y = torch.randint(0, self.spec.num_classes, (16,))
        self.loader = make_loader(torch.utils.data.TensorDataset(x, y), batch_size=8, shuffle=False, seed=9)

    def test_alpha_zero_recovers_c2m3_aligned_averaging(self):
        logs = shrink_log_scales(self.raw_logs, alpha=0.0, tau=float("inf"))
        scaled_average = build_scaled_average_model(self.models, self.spec, self.width, self.perms, logs)
        aligned = [
            permute_model_to_reference(model, "mlp", self.spec, self.width, self.perms[idx])
            for idx, model in enumerate(self.models)
        ]
        c2m3_average = average_models(aligned, "mlp", self.spec, self.width)

        assert_same_state(self, scaled_average, c2m3_average)

    def test_alpha_one_tau_inf_recovers_raw_monomial_scaling(self):
        logs = shrink_log_scales(self.raw_logs, alpha=1.0, tau=float("inf"))

        np.testing.assert_allclose(logs, self.raw_logs)

    def test_final_soup_output_is_one_capacity_matched_mlp(self):
        soup = greedy_soup_with_metadata(
            self.models,
            ["a", "b"],
            self.loader,
            self.loader,
            require_torch()[0].device("cpu"),
            "mlp",
            self.spec,
            self.width,
        )

        assert_capacity_matched_mlp(soup.model, self.spec, self.width)
        self.assertGreaterEqual(len(soup.selected_indices), 1)


if __name__ == "__main__":
    unittest.main()
