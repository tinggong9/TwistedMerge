import unittest

from src.greedy_aware_monomial import nested_validation_split
from src.model_merging_benchmark import require_torch


class NestedValidationNoLeakageTests(unittest.TestCase):
    def test_nested_validation_splits_are_disjoint(self):
        torch, _, _ = require_torch()
        dataset = torch.utils.data.TensorDataset(torch.arange(100).float().view(100, 1), torch.arange(100) % 2)

        train_inner, val_model, val_selector = nested_validation_split(
            dataset,
            val_model_fraction=0.2,
            val_selector_fraction=0.2,
            seed=12,
        )

        train_idx = set(train_inner.indices)
        val_model_idx = set(val_model.indices)
        val_selector_idx = set(val_selector.indices)
        self.assertFalse(train_idx & val_model_idx)
        self.assertFalse(train_idx & val_selector_idx)
        self.assertFalse(val_model_idx & val_selector_idx)
        self.assertEqual(len(train_idx | val_model_idx | val_selector_idx), len(dataset))


if __name__ == "__main__":
    unittest.main()
