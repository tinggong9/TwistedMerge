import torch

from experiments.multidomain_biomedical_experts import synthetic_domain


def test_synthetic_domains_are_color_shifts_not_spatial_actions():
    image = torch.rand(2, 3, 16, 16)
    for domain in range(5):
        value = synthetic_domain(image, domain)
        assert value.shape == image.shape
        assert torch.all((0 <= value) & (value <= 1))


def test_synthetic_domains_do_not_change_masks_or_coordinates():
    image = torch.zeros(1, 3, 8, 8)
    image[..., 2, 5] = 1
    for domain in range(5):
        assert synthetic_domain(image, domain)[0, :, 2, 5].sum() > 0
