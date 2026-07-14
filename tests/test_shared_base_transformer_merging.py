from __future__ import annotations

import torch

from experiments.shared_base_transformer_merging import TinyTransformer, flatten_state, state_from_vector


def test_tiny_transformer_round_trip_state_and_forward() -> None:
    model = TinyTransformer()
    vector, metadata = flatten_state(model.state_dict())
    restored = TinyTransformer()
    restored.load_state_dict(state_from_vector(vector, metadata))
    tokens = torch.randint(0, 40, (5, 8))
    assert torch.allclose(model(tokens), restored(tokens))
