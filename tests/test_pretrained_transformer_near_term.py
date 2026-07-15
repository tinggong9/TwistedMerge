import torch

from experiments.pretrained_transformer_near_term import merge


def test_transformer_delta_merges_preserve_nonfloating_state():
    base = {"weight": torch.zeros(4), "counter": torch.tensor(2, dtype=torch.long)}
    states = [{"weight": torch.ones(4), "counter": torch.tensor(2)}, {"weight": torch.full((4,), 3.0), "counter": torch.tensor(2)}]
    result = merge(base, states, "mean")
    assert torch.allclose(result["weight"], torch.full((4,), 2.0))
    assert result["counter"].item() == 2
