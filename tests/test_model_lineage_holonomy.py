from __future__ import annotations

import numpy as np
import torch

from src.holonomy_application_corpus import LowRankChartAdapter
from src.model_lineage_holonomy import (
    AdaptationResult,
    LineageSplitSizes,
    adapt_on_task,
    adjacent_swap_pairs,
    apply_task_corruption,
    deterministic_split_indices,
    lineage_edges,
    lineage_nodes,
    order_comparison_pairs,
    state_dict_sha256,
    two_task_squares,
)


def test_frozen_lineage_graph_has_exact_nodes_edges_and_loops() -> None:
    nodes = lineage_nodes()
    assert len(nodes) == 16
    assert len({node.name for node in nodes}) == 16
    assert len(lineage_edges()) == 15
    assert {node.name for node in nodes if node.depth == 3} == {
        "M_ABC",
        "M_ACB",
        "M_BAC",
        "M_BCA",
        "M_CAB",
        "M_CBA",
    }
    assert len(adjacent_swap_pairs()) == 6
    assert len(order_comparison_pairs()) == 9
    assert two_task_squares()["AB_square"] == ("M0", "M_A", "M_AB", "M_B", "M0")


def test_lineage_splits_are_deterministic_and_disjoint() -> None:
    sizes = LineageSplitSizes(20, 5, 6, 7, 8, 9)
    left = deterministic_split_indices(100, 30, sizes, seed=11)
    right = deterministic_split_indices(100, 30, sizes, seed=11)
    assert all(np.array_equal(left[name], right[name]) for name in left)
    train = np.concatenate([value for name, value in left.items() if name != "application_test"])
    assert len(train) == len(np.unique(train)) == 46


def test_corruptions_are_deterministic_and_distinct() -> None:
    images = torch.linspace(0.0, 1.0, 4 * 3 * 8 * 8).reshape(4, 3, 8, 8)
    indices = np.asarray([9, 2, 7, 4])
    noise_left = apply_task_corruption(images, "A", indices)
    noise_right = apply_task_corruption(images, "A", indices)
    blur = apply_task_corruption(images, "B", indices)
    color = apply_task_corruption(images, "C", indices)
    assert torch.equal(noise_left, noise_right)
    assert not torch.equal(noise_left, blur)
    assert not torch.equal(blur, color)
    assert all(float(value.min()) >= 0.0 and float(value.max()) <= 1.0 for value in (noise_left, blur, color))


def test_adaptation_uses_fixed_budget_and_reproducible_task_schedule() -> None:
    torch.manual_seed(3)
    parent = LowRankChartAdapter(feature_dim=6, rank=2, classes=3)
    features = torch.randn(17, 6)
    labels = torch.arange(17) % 3
    left = adapt_on_task(parent, features, labels, independent_seed=2, task="B", epochs=2, batch_size=5)
    right = adapt_on_task(parent, features, labels, independent_seed=2, task="B", epochs=2, batch_size=5)
    assert isinstance(left, AdaptationResult)
    assert left.epochs == 2
    assert left.optimizer_steps == 8
    assert state_dict_sha256(left.model) == state_dict_sha256(right.model)
    assert state_dict_sha256(parent) != state_dict_sha256(left.model)
