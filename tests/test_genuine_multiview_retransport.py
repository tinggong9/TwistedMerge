import numpy as np
import torch

from experiments.genuine_multiview_retransport import (
    DEVICE,
    ViewCNN,
    aligned_logits,
    null_rows,
    render_points,
    rotate_points,
    transition_maps,
    transition_statistics,
)


def test_multiview_path_uses_explicit_3d_maps_and_trained_cnns():
    rng = np.random.default_rng(2)
    points = rng.normal(size=(256, 3)).astype(np.float32)
    points /= np.linalg.norm(points, axis=1).max()
    observed = rotate_points(points, np.pi / 2)
    recovered = rotate_points(observed, -np.pi / 2)
    assert np.max(np.abs(recovered - points)) < 1e-6
    image = render_points(observed)
    assert image.shape == (2, 32, 32)
    model = ViewCNN(10)
    assert any(isinstance(module, torch.nn.Conv2d) for module in model.modules())
    features = [rng.normal(size=(20, 8)) for _ in range(4)]
    maps = transition_maps(features)
    assert transition_statistics(maps)["cycle_residual"] >= 0
    assert len(null_rows(maps, seed=3, draws=2)) == 8
    experts = [ViewCNN(10).to(DEVICE).eval() for _ in range(4)]
    logits = aligned_logits(experts, torch.tensor(image).unsqueeze(0), [np.eye(24) for _ in range(4)])
    assert logits.shape == (1, 10)
    assert not logits.requires_grad
