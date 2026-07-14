from __future__ import annotations

import torch

from experiments.pretrained_merge_smoke import dare_merge, slerp, ties_merge


def test_internal_merge_baselines_return_finite_matched_vectors() -> None:
    base = torch.zeros(20)
    tasks = [torch.linspace(0, 1, 20), torch.linspace(1, 0, 20)]
    for merged in (ties_merge(base, tasks), dare_merge(base, tasks, seed=2), slerp(tasks[0], tasks[1])):
        assert merged.shape == base.shape
        assert torch.isfinite(merged).all()
