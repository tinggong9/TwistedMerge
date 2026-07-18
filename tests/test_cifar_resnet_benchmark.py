from __future__ import annotations

import numpy as np
import pytest
import torch

from experiments.post_iclr_resnet18_cifar10 import gate_status
from src.cifar_resnet_benchmark import (
    TrainingRecipe,
    calibration_metrics,
    deterministic_split_indices,
    make_cifar_resnet18,
    parameter_count,
)


def test_cifar_resnet18_uses_standard_cifar_stem() -> None:
    model = make_cifar_resnet18()
    assert model.conv1.kernel_size == (3, 3)
    assert model.conv1.stride == (1, 1)
    assert isinstance(model.maxpool, torch.nn.Identity)
    assert model(torch.randn(2, 3, 32, 32)).shape == (2, 10)
    assert parameter_count(model) == 11_173_962


def test_deterministic_split_is_disjoint_and_complete() -> None:
    training, validation = deterministic_split_indices(100, 20, 17)
    training_again, validation_again = deterministic_split_indices(100, 20, 17)
    assert training == training_again
    assert validation == validation_again
    assert len(training) == 80
    assert len(validation) == 20
    assert not set(training).intersection(validation)
    assert set(training).union(validation) == set(range(100))


@pytest.mark.parametrize("validation_size", [0, 10])
def test_invalid_split_is_rejected(validation_size: int) -> None:
    with pytest.raises(ValueError):
        deterministic_split_indices(10, validation_size, 0)


def test_calibration_metrics_are_zero_for_perfect_confident_predictions() -> None:
    logits = torch.tensor([[100.0, -100.0], [-100.0, 100.0]])
    targets = torch.tensor([0, 1])
    result = calibration_metrics(logits, targets)
    assert result["ece"] == pytest.approx(0.0, abs=1e-7)
    assert result["brier"] == pytest.approx(0.0, abs=1e-7)


def test_pilot_gate_requires_all_preregistered_conditions() -> None:
    import pandas as pd

    passing = pd.DataFrame({"validation_accuracy": [0.922, 0.928, 0.925]})
    result = gate_status(passing, expected_seeds=3, stage="pilot")
    assert result["status"] == "base_quality_gate_passed"
    assert result["test_evaluations"] == 0

    noisy = pd.DataFrame({"validation_accuracy": [0.90, 0.95, 0.92]})
    assert gate_status(noisy, expected_seeds=3, stage="pilot")["status"] == "base_quality_gate_failed"


def test_recipe_serialization_is_stable() -> None:
    recipe = TrainingRecipe()
    assert recipe.to_dict()["epochs"] == 150
    assert recipe.to_dict()["validation_size"] == 5000
