import torch

from experiments.exact_mask_retransport import exact_predictor, generated_masks
from experiments.spatial_output_common import apply_d4, inverse_d4


def test_spatial_mask_needs_output_action_but_class_label_does_not():
    label = torch.tensor([4])
    mask = generated_masks()["asymmetric_polygon"]
    image = torch.cat([mask, torch.zeros_like(mask), torch.zeros_like(mask)], dim=1)
    transformed = apply_d4(image, 3)
    canonical_prediction = exact_predictor(inverse_d4(transformed, 3))
    assert torch.equal(label, label)
    assert not torch.equal(canonical_prediction, apply_d4(mask, 3))
    assert torch.equal(apply_d4(canonical_prediction, 3), apply_d4(mask, 3))
