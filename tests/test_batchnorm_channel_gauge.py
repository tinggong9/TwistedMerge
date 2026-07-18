from __future__ import annotations

from dataclasses import replace

import pytest

from src.batchnorm_channel_gauge import (
    parameter_count,
    permute_resnet_channels,
    random_resnet18_permutations,
    scale_conv_batchnorm,
    scale_relu_conv_pair,
)


torch = pytest.importorskip("torch")
torchvision = pytest.importorskip("torchvision")


class NoBatchNormPair(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 5, 3, padding=1)

    def forward(self, inputs):
        return self.conv2(torch.relu(self.conv1(inputs)))


def resnet18():
    model = torchvision.models.resnet18(weights=None, num_classes=10)
    model.eval()
    return model


@pytest.mark.parametrize("training", [False, True])
def test_resnet18_channel_permutation_preserves_logits(training: bool) -> None:
    torch.manual_seed(7)
    model = resnet18()
    permutations = random_resnet18_permutations(model, seed=11)
    transformed = permute_resnet_channels(model, permutations)
    model.train(training)
    transformed.train(training)
    inputs = torch.randn(4, 3, 32, 32)
    with torch.no_grad():
        first = model(inputs)
        second = transformed(inputs)
    assert parameter_count(model) == parameter_count(transformed)
    # Train-mode BatchNorm amplifies float32 reduction-order roundoff after an
    # otherwise exact channel permutation; eval mode is tighter.
    tolerance = 1e-4 if training else 2e-5
    assert torch.max(torch.abs(first - second)).item() < tolerance
    assert torch.equal(first.argmax(dim=1), second.argmax(dim=1))


def test_identity_shortcut_rejects_inconsistent_basis() -> None:
    model = resnet18()
    permutations = random_resnet18_permutations(model, seed=3)
    bad_layer1 = tuple(reversed(permutations.stages["layer1"]))
    bad = replace(permutations, stages={**permutations.stages, "layer1": bad_layer1})
    with pytest.raises(ValueError, match="identity shortcut"):
        permute_resnet_channels(model, bad)


@pytest.mark.parametrize("strategy", ["affine", "running_affine"])
def test_eval_exact_batchnorm_scaling_strategies(strategy: str) -> None:
    torch.manual_seed(13)
    model = resnet18()
    model.bn1.running_mean.normal_()
    model.bn1.running_var.uniform_(0.2, 2.0)
    scales = torch.linspace(0.35, 2.4, model.conv1.out_channels)
    transformed = scale_conv_batchnorm(model, "conv1", "bn1", scales, strategy=strategy)
    model.eval()
    transformed.eval()
    inputs = torch.randn(3, 3, 32, 32)
    with torch.no_grad():
        first = model(inputs)
        second = transformed(inputs)
    assert torch.max(torch.abs(first - second)).item() < 2e-5
    assert torch.equal(first.argmax(dim=1), second.argmax(dim=1))


def test_running_statistics_only_is_not_exact_with_large_epsilon() -> None:
    torch.manual_seed(19)
    model = resnet18()
    model.bn1.eps = 0.1
    model.bn1.running_mean.normal_()
    model.bn1.running_var.uniform_(0.1, 1.0)
    scales = torch.linspace(0.25, 3.0, model.conv1.out_channels)
    transformed = scale_conv_batchnorm(model, "conv1", "bn1", scales, strategy="running")
    inputs = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        error = torch.max(torch.abs(model(inputs) - transformed(inputs))).item()
    assert error > 1e-5


def test_no_batchnorm_positive_relu_gauge_is_exact() -> None:
    torch.manual_seed(23)
    model = NoBatchNormPair().eval()
    scales = torch.linspace(0.2, 2.5, 8)
    transformed = scale_relu_conv_pair(model, "conv1", "conv2", scales).eval()
    inputs = torch.randn(5, 3, 12, 12)
    with torch.no_grad():
        first = model(inputs)
        second = transformed(inputs)
    assert torch.max(torch.abs(first - second)).item() < 2e-6
    assert parameter_count(model) == parameter_count(transformed)
