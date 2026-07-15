import torch

from experiments.real_lora_adapter_near_term import average_states, delta_factor_state


def states():
    return [
        {"x.lora_A.default.weight": torch.tensor([[1.0, 0.0], [0.0, 1.0]]), "x.lora_B.default.weight": torch.tensor([[1.0, 0.0], [0.0, 1.0]])},
        {"x.lora_A.default.weight": torch.tensor([[2.0, 0.0], [0.0, 2.0]]), "x.lora_B.default.weight": torch.tensor([[0.5, 0.0], [0.0, 0.5]])},
    ]


def test_factor_and_delta_merges_are_executed_tensor_states():
    raw = average_states(states())
    delta = delta_factor_state(states(), "mean")
    assert raw.keys() == delta.keys()
    assert torch.allclose(delta["x.lora_B.default.weight"] @ delta["x.lora_A.default.weight"], torch.eye(2), atol=1e-5)
