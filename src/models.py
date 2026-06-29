"""PyTorch models for future MNIST/CIFAR model-merging experiments."""

from __future__ import annotations


def _require_torch():
    try:
        import torch
        import torch.nn as nn
    except Exception as exc:  # pragma: no cover - depends on optional install
        raise RuntimeError(
            "PyTorch is required for image model-merging experiments. "
            "Install with `python -m pip install -r requirements.txt`."
        ) from exc
    return torch, nn


def make_mlp(input_dim: int = 784, hidden_dim: int = 256, n_classes: int = 10):
    _, nn = _require_torch()
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, n_classes),
    )


def make_small_cnn(in_channels: int = 1, n_classes: int = 10):
    _, nn = _require_torch()
    return nn.Sequential(
        nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((4, 4)),
        nn.Flatten(),
        nn.Linear(64 * 4 * 4, 128),
        nn.ReLU(),
        nn.Linear(128, n_classes),
    )
