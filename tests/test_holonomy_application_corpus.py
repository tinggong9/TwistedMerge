from __future__ import annotations

import numpy as np
import pytest
import torch

from experiments.chart_followup_common import apply_d4, compose_d4, inverse_chart
from src.holonomy_application_corpus import (
    CorpusSplitSizes,
    LowRankChartAdapter,
    deterministic_split_indices,
    tensor_mapping_sha256,
    validate_split_indices,
)


def test_d4_action_matches_composition_table() -> None:
    image = torch.arange(2 * 5 * 5, dtype=torch.float32).reshape(1, 2, 5, 5)
    for left in range(8):
        for right in range(8):
            sequential = apply_d4(apply_d4(image, right), left)
            composed = apply_d4(image, compose_d4(left, right))
            assert torch.equal(sequential, composed)
    for chart in range(8):
        assert torch.equal(apply_d4(apply_d4(image, chart), inverse_chart(chart)), image)


def test_split_manifest_is_deterministic_disjoint_and_bounded() -> None:
    sizes = CorpusSplitSizes(20, 7, 8, 9, 11)
    left = deterministic_split_indices(100, 40, sizes, seed=17)
    right = deterministic_split_indices(100, 40, sizes, seed=17)
    assert all(np.array_equal(left[name], right[name]) for name in left)
    validate_split_indices(left, 100, 40)
    all_train = np.concatenate([left[name] for name in left if name != "test"])
    assert len(all_train) == len(np.unique(all_train)) == 44


def test_split_validation_rejects_train_overlap() -> None:
    sizes = CorpusSplitSizes(20, 7, 8, 9, 11)
    splits = deterministic_split_indices(100, 40, sizes, seed=17)
    splits["validation"][0] = splits["adapter_train"][0]
    with pytest.raises(ValueError, match="overlap"):
        validate_split_indices(splits, 100, 40)


def test_effective_adapter_matches_forward_activations() -> None:
    torch.manual_seed(4)
    model = LowRankChartAdapter(feature_dim=7, rank=3, classes=5)
    with torch.no_grad():
        model.up.weight.normal_(std=0.1)
    features = torch.randn(11, 7)
    expected = features @ model.effective_adapter().T
    assert torch.allclose(model.forward_activations(features), expected, atol=1e-7, rtol=1e-7)
    assert model(features).shape == (11, 5)


def test_tensor_hash_is_label_independent_and_content_sensitive() -> None:
    logits = torch.arange(30, dtype=torch.float32).reshape(3, 10)
    labels = torch.tensor([0, 1, 2])
    before = tensor_mapping_sha256({"logits": logits})
    _ = (logits.argmax(1) == labels[torch.tensor([2, 0, 1])]).float().mean()
    after = tensor_mapping_sha256({"logits": logits})
    changed = logits.clone()
    changed[0, 0] += 1
    assert before == after
    assert before != tensor_mapping_sha256({"logits": changed})
