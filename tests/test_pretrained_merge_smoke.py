import torch

from experiments.pretrained_merge_smoke import dare_merge, slerp, ties_merge


def test_vector_merge_controls_preserve_shape_and_finiteness():
    base = torch.zeros(10)
    tasks = [torch.linspace(0, 1, 10), torch.linspace(1, 0, 10), torch.ones(10), -torch.ones(10)]
    for merged in (ties_merge(base, tasks), dare_merge(base, tasks), slerp(tasks[0], tasks[1])):
        assert merged.shape == base.shape
        assert torch.isfinite(merged).all()
