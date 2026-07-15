import numpy as np
import torch

from experiments.chart_followup_common import (
    TMP,
    ImageCNN,
    apply_d4,
    compose_d4,
    d4_table,
    inverse_chart,
    save_logits_before_evaluation,
    split_indices,
)


def test_image_cnn_preserves_spatial_features_for_both_dataset_sizes():
    assert ImageCNN(10, 1, 4)(torch.rand(2, 1, 28, 28)).shape == (2, 10)
    assert ImageCNN(10, 3, 4)(torch.rand(2, 3, 32, 32)).shape == (2, 10)


def test_d4_table_is_a_group_action():
    table = d4_table()
    assert np.array_equal(table[0], np.arange(8))
    assert np.array_equal(table[:, 0], np.arange(8))
    for left in range(8):
        assert compose_d4(left, inverse_chart(left)) == 0
        for middle in range(8):
            for right in range(8):
                assert compose_d4(compose_d4(left, middle), right) == compose_d4(left, compose_d4(middle, right))


def test_composition_matches_tensor_action():
    image = torch.arange(49, dtype=torch.float32).reshape(1, 1, 7, 7)
    for left in range(8):
        for right in range(8):
            sequential = apply_d4(apply_d4(image, right), left)
            combined = apply_d4(image, compose_d4(left, right))
            assert torch.equal(sequential, combined)


def test_split_roles_are_disjoint():
    split = split_indices(20, 60_000)
    values = [set(indices.tolist()) for indices in split.values()]
    for index, left in enumerate(values):
        for right in values[index + 1 :]:
            assert left.isdisjoint(right)


def test_saved_logits_do_not_change_under_label_permutation():
    audit = save_logits_before_evaluation(
        "unit_label_permutation",
        {"candidate": torch.tensor([[1.0, 2.0], [3.0, 4.0]])},
        torch.tensor([0, 1]),
        123,
    )
    assert audit["candidate_hashes_unchanged"]
    assert audit["file_hash_unchanged"]
    (TMP / "logits" / "unit_label_permutation.npz").unlink(missing_ok=True)
