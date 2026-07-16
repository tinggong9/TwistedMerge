import torch

from experiments.exact_mask_retransport import exact_predictor, generated_masks
from experiments.spatial_output_common import apply_d4, inverse_d4


def test_exact_mask_retransport_for_every_d4_element():
    for mask in generated_masks().values():
        image = torch.cat([mask, torch.zeros_like(mask), torch.ones_like(mask) * 0.25], dim=1)
        for chart in range(8):
            transformed = apply_d4(image, chart)
            prediction = apply_d4(exact_predictor(inverse_d4(transformed, chart)), chart)
            assert torch.equal(prediction, apply_d4(mask, chart))


def test_omitting_retransport_fails_on_asymmetric_arrow():
    mask = generated_masks()["arrow"]
    image = torch.cat([mask, torch.zeros_like(mask), torch.zeros_like(mask)], dim=1)
    transformed = apply_d4(image, 1)
    prediction = exact_predictor(inverse_d4(transformed, 1))
    assert not torch.equal(prediction, apply_d4(mask, 1))
