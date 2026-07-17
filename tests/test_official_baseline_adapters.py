from __future__ import annotations

import numpy as np
import torch

import src.official_baseline_adapters as adapters
from src.official_baseline_adapters import (
    average_state_dicts,
    c2m3_mlp_permutation_spec,
    flatten_float_state,
    git_rebasin_arrays_to_torch_state,
    torch_state_to_git_rebasin_arrays,
    vector_to_float_state,
)


def example_state(width: int = 4):
    return {
        "hidden.weight": torch.arange(width * 6, dtype=torch.float32).reshape(width, 6),
        "hidden.bias": torch.arange(width, dtype=torch.float32),
        "classifier.weight": torch.arange(3 * width, dtype=torch.float32).reshape(3, width),
        "classifier.bias": torch.arange(3, dtype=torch.float32),
    }


def test_git_rebasin_axis_conversion_round_trips_exactly():
    state = example_state()
    arrays = torch_state_to_git_rebasin_arrays(state)
    assert arrays["dense0_kernel"].shape == (6, 4)
    assert arrays["dense1_kernel"].shape == (4, 3)
    restored = git_rebasin_arrays_to_torch_state(arrays, state)
    for key in state:
        assert torch.equal(restored[key], state[key])


def test_c2m3_adapter_spec_covers_each_hidden_axis_once():
    spec = c2m3_mlp_permutation_spec()
    assert spec.perm_to_layers_and_axes == {
        "P_0": [
            ("hidden.weight", 0),
            ("hidden.bias", 0),
            ("classifier.weight", 1),
        ]
    }
    assert spec.layer_and_axes_to_perm["classifier.bias"] == (None,)


def test_average_state_dicts_is_capacity_preserving_and_numeric():
    first = example_state()
    second = {key: value + 2 for key, value in first.items()}
    averaged = average_state_dicts([first, second])
    assert set(averaged) == set(first)
    for key in first:
        assert np.allclose(averaged[key].numpy(), (first[key] + 1).numpy())


def test_flat_state_conversion_uses_stable_keys_and_round_trips():
    state = example_state()
    vector, meta = flatten_float_state(state)
    restored = vector_to_float_state(vector, meta, state)
    assert [row[0] for row in meta] == sorted(state)
    for key in state:
        assert torch.equal(restored[key], state[key])


def test_official_ties_density_is_passed_as_retained_top_fraction(monkeypatch):
    captured = {}

    class FakeOfficialTies:
        @staticmethod
        def merge_methods(**kwargs):
            captured.update(kwargs)
            return kwargs["flat_task_checks"].mean(dim=0)

    monkeypatch.setattr(adapters, "load_official_ties_core", lambda _source: FakeOfficialTies)
    base = example_state()
    tasks = [
        {key: value + offset for key, value in base.items()}
        for offset in (1.0, 2.0, 3.0)
    ]
    merged = adapters.official_ties_state(base, tasks, adapters.Path("unused"), density=0.2, scale=1.0)

    assert captured["reset_type"] == "topk"
    assert captured["reset_thresh"] == 0.2
    for key in base:
        assert torch.allclose(merged[key], base[key] + 2.0)

    adapters.official_ties_state(base, tasks, adapters.Path("unused"), density=1.0, scale=1.0)
    assert 0.999 < captured["reset_thresh"] < 1.0
