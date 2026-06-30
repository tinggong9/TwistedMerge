import unittest

import numpy as np
import torch

from src.model_merging_benchmark import DatasetSpec, make_loader, make_model, set_seed
from src.rank_lift_baselines import (
    CAPACITY_METADATA_FIELDS,
    c2m3_cluster_branch_ensemble,
    count_parameters,
    method_capacity_metadata,
    random_branch_ensemble,
    validation_branch_ensemble,
)


class RankLiftBaselineTests(unittest.TestCase):
    def setUp(self):
        self.spec = DatasetSpec(name="toy", input_shape=(1, 2, 2), num_classes=3)
        self.width = 4
        self.models = []
        for idx in range(4):
            set_seed(900 + idx)
            self.models.append(make_model("mlp", self.spec, self.width))
        x = torch.randn(20, *self.spec.input_shape)
        y = torch.randint(0, self.spec.num_classes, (20,))
        dataset = torch.utils.data.TensorDataset(x, y)
        self.val_loader = make_loader(dataset, batch_size=5, shuffle=False, seed=11)
        self.test_loader = make_loader(dataset, batch_size=5, shuffle=False, seed=12)
        self.device = torch.device("cpu")

    def pairwise(self):
        identity = np.arange(self.width)
        swap = np.array([1, 0, 2, 3])
        reverse = np.array([3, 2, 1, 0])
        perms = {}
        for i in range(len(self.models)):
            for j in range(len(self.models)):
                if i == j:
                    perms[(i, j)] = identity.copy()
                elif {i, j} == {0, 1}:
                    perms[(i, j)] = swap.copy()
                elif {i, j} == {2, 3}:
                    perms[(i, j)] = identity.copy()
                else:
                    perms[(i, j)] = reverse.copy()
        return perms

    def assert_metadata_schema(self, metadata):
        self.assertTrue(set(CAPACITY_METADATA_FIELDS).issubset(metadata.keys()))
        self.assertGreater(metadata["parameter_count"], 0)
        self.assertGreater(metadata["parameter_multiplier"], 0.0)
        self.assertGreater(metadata["inference_multiplier"], 0.0)

    def test_random_branch_count_and_capacity_metadata(self):
        branches = random_branch_ensemble(self.models, 2, "mlp", self.spec, self.width, seed=123)

        self.assertEqual(len(branches), 2)
        metadata = method_capacity_metadata("random_branch_ensemble_2", branches, self.models[0])
        self.assert_metadata_schema(metadata)
        self.assertEqual(metadata["branch_count"], 2)
        self.assertEqual(metadata["parameter_count"], 2 * count_parameters(self.models[0]))
        self.assertAlmostEqual(metadata["parameter_multiplier"], 2.0)
        self.assertAlmostEqual(metadata["inference_multiplier"], 2.0)
        self.assertFalse(metadata["uses_obstruction_residual"])
        self.assertFalse(metadata["uses_validation_data"])
        self.assertTrue(metadata["capacity_matched_to_rank_lift"])
        self.assertFalse(metadata["capacity_matched_to_weight_average"])

    def test_validation_branch_uses_validation_not_obstruction(self):
        branches = validation_branch_ensemble(
            self.models,
            self.val_loader,
            self.test_loader,
            2,
            "mlp",
            self.spec,
            self.width,
            self.device,
        )

        self.assertEqual(len(branches), 2)
        metadata = method_capacity_metadata("validation_branch_ensemble_2", branches, self.models[0])
        self.assert_metadata_schema(metadata)
        self.assertTrue(metadata["uses_validation_data"])
        self.assertFalse(metadata["uses_obstruction_residual"])
        self.assertTrue(metadata["capacity_matched_to_rank_lift"])

    def test_c2m3_cluster_branch_count_and_metadata(self):
        branches = c2m3_cluster_branch_ensemble(self.models, self.pairwise(), 2, "mlp", self.spec, self.width)

        self.assertEqual(len(branches), 2)
        metadata = method_capacity_metadata("c2m3_cluster_branch_ensemble_2", branches, self.models[0])
        self.assert_metadata_schema(metadata)
        self.assertEqual(metadata["branch_count"], 2)
        self.assertFalse(metadata["uses_obstruction_residual"])
        self.assertFalse(metadata["uses_validation_data"])

    def test_capacity_metadata_schema_for_all_experiment_rows(self):
        method_objects = {
            "twisted_rank_lift_2": self.models[:2],
            "random_branch_ensemble_2": self.models[:2],
            "validation_branch_ensemble_2": self.models[:2],
            "c2m3_cluster_branch_ensemble_2": self.models[:2],
            "weight_average": self.models[0],
        }

        for method, obj in method_objects.items():
            with self.subTest(method=method):
                metadata = method_capacity_metadata(method, obj, self.models[0])
                self.assert_metadata_schema(metadata)
                for field in CAPACITY_METADATA_FIELDS:
                    self.assertIn(field, metadata)


if __name__ == "__main__":
    unittest.main()
