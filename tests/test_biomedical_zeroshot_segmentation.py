import torch

from experiments.biomedical_zeroshot_segmentation import SEEN, UNSEEN, seen_chart_augmentation
from experiments.spatial_output_common import apply_d4


def test_seen_and_unseen_chart_roles_are_disjoint_and_complete():
    assert set(SEEN).isdisjoint(UNSEEN)
    assert set(SEEN) | set(UNSEEN) == set(range(8))


def test_heldout_charts_are_not_reflections_of_training_role_names():
    assert 2 in UNSEEN and 3 in UNSEEN and 5 in UNSEEN


def test_zeroshot_augmentation_never_uses_heldout_chart_elements():
    images = torch.arange(24 * 3 * 9 * 9, dtype=torch.float32).reshape(24, 3, 9, 9)
    masks = torch.arange(24 * 9 * 9, dtype=torch.float32).reshape(24, 1, 9, 9)
    augmented_images, augmented_masks = seen_chart_augmentation(images, masks, 123)
    for index in range(len(images)):
        assert any(torch.equal(augmented_images[index], apply_d4(images[index:index+1], chart)[0]) for chart in SEEN)
        assert any(torch.equal(augmented_masks[index], apply_d4(masks[index:index+1], chart)[0]) for chart in SEEN)
