import torch

from experiments.broader_language_extended import SECOND_MODEL_ID, SECOND_MODEL_REVISION, adapter_residual


def test_second_language_base_is_pinned():
    assert SECOND_MODEL_ID == "prajjwal1/bert-tiny"
    assert len(SECOND_MODEL_REVISION) == 40


def test_adapter_residual_is_deterministic():
    states = []
    for seed in range(4):
        generator = torch.Generator().manual_seed(seed)
        states.append({"layer.lora_B.weight": torch.randn((8, 4), generator=generator)})
    first, flag1 = adapter_residual(states, 3)
    second, flag2 = adapter_residual(states, 3)
    assert first == second
    assert flag1 == flag2
