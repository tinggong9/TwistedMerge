import unittest

import numpy as np

from src.ladder_merge_methods import METHOD_METADATA, transform_mlp_positive_scale
from src.model_merging_benchmark import DatasetSpec, make_model, require_torch, set_seed


class LadderMergeMethodTests(unittest.TestCase):
    def test_positive_scale_preserves_relu_mlp_outputs(self):
        torch, _, _ = require_torch()
        set_seed(123)
        spec = DatasetSpec(name="tiny", input_shape=(1, 2, 2), num_classes=3)
        width = 5
        model = make_model("mlp", spec, width)
        x = torch.randn(7, 1, 2, 2)
        scales = np.array([0.5, 1.25, 2.0, 0.75, 1.5])
        transformed = transform_mlp_positive_scale(
            model,
            spec,
            width,
            np.arange(width),
            scales,
        )

        with torch.no_grad():
            original_logits = model(x)
            transformed_logits = transformed(x)

        self.assertTrue(torch.allclose(original_logits, transformed_logits, atol=1e-6))

    def test_signed_and_gl_metadata_are_not_exact_relu_claims(self):
        self.assertEqual(METHOD_METADATA["signed_permutation"].symmetry_status, "heuristic_relu_not_exact")
        self.assertEqual(METHOD_METADATA["low_rank_GL_diagnostic"].symmetry_status, "diagnostic_not_single_model_for_relu")
        self.assertFalse(METHOD_METADATA["low_rank_GL_diagnostic"].is_single_model)


if __name__ == "__main__":
    unittest.main()
