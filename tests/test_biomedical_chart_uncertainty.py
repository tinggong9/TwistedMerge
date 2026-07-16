import torch

from experiments.biomedical_chart_uncertainty import PERTURBATIONS, _perturb


def test_all_uncertainty_perturbations_preserve_tensor_shape():
    images = torch.rand(2, 3, 32, 32)
    for name in PERTURBATIONS:
        assert _perturb(images, name, 1).shape == images.shape


def test_symmetric_perturbation_is_d4_invariant():
    from experiments.spatial_output_common import apply_d4

    image = _perturb(torch.rand(1, 3, 32, 32), "approximately_symmetric", 1)
    assert torch.allclose(image, apply_d4(image, 5), atol=1e-6)
