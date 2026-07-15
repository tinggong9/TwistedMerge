from __future__ import annotations

import torch

from experiments.compact_pretrained_vision import delta_merge


def test_delta_merge_average_on_tiny_state() -> None:
    base = {"layer4.weight": torch.zeros(2), "fc.weight": torch.zeros(2), "counter": torch.tensor(1)}
    first = {"layer4.weight": torch.ones(2), "fc.weight": torch.ones(2), "counter": torch.tensor(1)}
    second = {"layer4.weight": torch.full((2,), 3.0), "fc.weight": torch.full((2,), 3.0), "counter": torch.tensor(1)}
    merged = delta_merge(base, [first, second], "average")
    assert torch.equal(merged["layer4.weight"], torch.full((2,), 2.0))
