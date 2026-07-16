import torch

from experiments.spatial_output_common import D4SymmetrizedUNet, TinyUNet, apply_d4, inverse_chart


def test_symmetrized_unet_is_exactly_equivariant():
    torch.manual_seed(3)
    model = D4SymmetrizedUNet(TinyUNet(width=2)).eval()
    image = torch.rand(1, 3, 32, 32)
    with torch.no_grad():
        base = model(image)
        for chart in range(8):
            assert torch.allclose(model(apply_d4(image, chart)), apply_d4(base, chart), atol=2e-6)


def test_all_required_discovery_methods_are_present():
    from experiments.biomedical_segmentation_discovery import METHODS

    assert len(METHODS) == 20
    assert "inferred_chart_canonicalize_pool_retransport" in METHODS
    assert "wrong_output_action_control" in METHODS
