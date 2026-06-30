import unittest

import torch

from src.improved_monomial_merge import assert_capacity_matched_mlp, count_parameters, greedy_soup_with_metadata
from src.model_merging_benchmark import DatasetSpec, make_loader, make_model, set_seed


class UnionCandidateSoupTests(unittest.TestCase):
    def test_union_candidate_soup_is_single_capacity_matched_model(self):
        spec = DatasetSpec(name="toy", input_shape=(1, 2, 2), num_classes=2)
        width = 3
        set_seed(202)
        models = [make_model("mlp", spec, width) for _ in range(4)]
        labels = ["original", "c2m3", "shrinkage", "global"]
        x = torch.randn(18, *spec.input_shape)
        y = torch.randint(0, spec.num_classes, (18,))
        dataset = torch.utils.data.TensorDataset(x, y)
        val_loader = make_loader(dataset, batch_size=6, shuffle=False, seed=1)
        test_loader = make_loader(dataset, batch_size=6, shuffle=False, seed=2)

        result = greedy_soup_with_metadata(models, labels, val_loader, test_loader, torch.device("cpu"), "mlp", spec, width)

        self.assertGreaterEqual(len(result.selected_indices), 1)
        self.assertEqual(len(result.selected_indices), len(result.selected_labels))
        self.assertIn(result.selected_labels[0], labels)
        assert_capacity_matched_mlp(result.model, spec, width)
        self.assertEqual(count_parameters(result.model), count_parameters(models[0]))
        self.assertIsNotNone(result.test_metrics)


if __name__ == "__main__":
    unittest.main()
