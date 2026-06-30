"""Exact channel permutation and positive-scale gauges for a small ReLU CNN."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .model_merging_benchmark import require_torch


@dataclass(frozen=True)
class CnnGaugeSpec:
    in_channels: int = 1
    conv1_channels: int = 16
    conv2_channels: int = 32
    hidden_units: int = 128
    spatial_after_pool: int = 7
    num_classes: int = 10

    @property
    def conv2_block_size(self) -> int:
        return self.spatial_after_pool * self.spatial_after_pool


class SmallFashionCNN(require_torch()[1].Module):
    """No-BatchNorm Fashion-MNIST CNN with exact ReLU channel gauges."""

    def __init__(self, spec: CnnGaugeSpec | None = None):
        torch, nn, _ = require_torch()
        super().__init__()
        self.gauge_spec = spec or CnnGaugeSpec()
        s = self.gauge_spec
        self.conv1 = nn.Conv2d(s.in_channels, s.conv1_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(s.conv1_channels, s.conv2_channels, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(s.conv2_channels * s.conv2_block_size, s.hidden_units)
        self.classifier = nn.Linear(s.hidden_units, s.num_classes)

    def forward(self, x, return_features: bool = False):
        torch, _, F = require_torch()
        h1 = F.relu(self.conv1(x))
        p1 = F.max_pool2d(h1, kernel_size=2)
        h2 = F.relu(self.conv2(p1))
        p2 = F.max_pool2d(h2, kernel_size=2)
        flat = p2.flatten(1)
        h3 = F.relu(self.fc1(flat))
        logits = self.classifier(h3)
        if return_features:
            return logits, {"conv1": h1, "conv2": h2, "fc1": h3}
        return logits


def make_small_fashion_cnn() -> SmallFashionCNN:
    return SmallFashionCNN()


def count_parameters(model) -> int:
    return int(sum(param.numel() for param in model.parameters()))


def inference_cost_units(spec: CnnGaugeSpec | None = None) -> int:
    """Static multiply-add proxy; invariant under channel gauges."""

    s = spec or CnnGaugeSpec()
    input_spatial = s.spatial_after_pool * 4
    conv2_spatial = input_spatial // 2
    conv1 = input_spatial * input_spatial * s.conv1_channels * s.in_channels * 3 * 3
    conv2 = conv2_spatial * conv2_spatial * s.conv2_channels * s.conv1_channels * 3 * 3
    fc1 = s.conv2_channels * s.conv2_block_size * s.hidden_units
    fc2 = s.hidden_units * s.num_classes
    return int(conv1 + conv2 + fc1 + fc2)


def _identity_perm(n: int) -> np.ndarray:
    return np.arange(n, dtype=int)


def _ones(n: int) -> np.ndarray:
    return np.ones(n, dtype=float)


def _validate_perm(perm: Sequence[int], n: int, name: str) -> np.ndarray:
    arr = np.asarray(perm, dtype=int)
    if arr.shape != (n,) or set(arr.tolist()) != set(range(n)):
        raise ValueError(f"{name} must be a permutation of length {n}")
    return arr


def _validate_scales(scales: Sequence[float], n: int, name: str) -> np.ndarray:
    arr = np.asarray(scales, dtype=float)
    if arr.shape != (n,):
        raise ValueError(f"{name} must have length {n}")
    if not np.all(np.isfinite(arr)) or np.any(arr <= 0.0):
        raise ValueError(f"{name} must contain positive finite values")
    return arr


def clone_cnn(model: SmallFashionCNN) -> SmallFashionCNN:
    cloned = SmallFashionCNN(model.gauge_spec)
    cloned.load_state_dict({key: value.detach().cpu().clone() for key, value in model.state_dict().items()})
    return cloned


def apply_channel_gauge(
    model: SmallFashionCNN,
    *,
    conv1_perm: Sequence[int] | None = None,
    conv2_perm: Sequence[int] | None = None,
    hidden_perm: Sequence[int] | None = None,
    conv1_scales: Sequence[float] | None = None,
    conv2_scales: Sequence[float] | None = None,
    hidden_scales: Sequence[float] | None = None,
) -> SmallFashionCNN:
    """Return an exactly function-preserving channel-gauged CNN.

    Convention: transformed activation channel ``r`` equals
    ``scale[r] * old_activation[perm[r]]``.  The next layer's corresponding
    input weights are divided by that same positive scale.
    """

    torch, _, _ = require_torch()
    s = model.gauge_spec
    p1 = _validate_perm(conv1_perm if conv1_perm is not None else _identity_perm(s.conv1_channels), s.conv1_channels, "conv1_perm")
    p2 = _validate_perm(conv2_perm if conv2_perm is not None else _identity_perm(s.conv2_channels), s.conv2_channels, "conv2_perm")
    ph = _validate_perm(hidden_perm if hidden_perm is not None else _identity_perm(s.hidden_units), s.hidden_units, "hidden_perm")
    a1 = _validate_scales(conv1_scales if conv1_scales is not None else _ones(s.conv1_channels), s.conv1_channels, "conv1_scales")
    a2 = _validate_scales(conv2_scales if conv2_scales is not None else _ones(s.conv2_channels), s.conv2_channels, "conv2_scales")
    ah = _validate_scales(hidden_scales if hidden_scales is not None else _ones(s.hidden_units), s.hidden_units, "hidden_scales")

    out = SmallFashionCNN(s)
    dtype = model.conv1.weight.detach().cpu().dtype
    t1 = torch.tensor(a1, dtype=dtype)
    t2 = torch.tensor(a2, dtype=dtype)
    th = torch.tensor(ah, dtype=dtype)
    with torch.no_grad():
        out.conv1.weight.copy_(model.conv1.weight.detach().cpu()[p1] * t1.view(-1, 1, 1, 1))
        out.conv1.bias.copy_(model.conv1.bias.detach().cpu()[p1] * t1)

        conv2_weight = model.conv2.weight.detach().cpu()[p2][:, p1, :, :]
        conv2_weight = conv2_weight * t2.view(-1, 1, 1, 1) / t1.view(1, -1, 1, 1)
        out.conv2.weight.copy_(conv2_weight)
        out.conv2.bias.copy_(model.conv2.bias.detach().cpu()[p2] * t2)

        old_fc1 = model.fc1.weight.detach().cpu()[ph].clone()
        new_fc1 = torch.empty_like(old_fc1)
        block = s.conv2_block_size
        for new_channel, old_channel in enumerate(p2):
            old_slice = slice(int(old_channel) * block, int(old_channel + 1) * block)
            new_slice = slice(new_channel * block, (new_channel + 1) * block)
            new_fc1[:, new_slice] = old_fc1[:, old_slice] / float(a2[new_channel])
        new_fc1 = new_fc1 * th.view(-1, 1)
        out.fc1.weight.copy_(new_fc1)
        out.fc1.bias.copy_(model.fc1.bias.detach().cpu()[ph] * th)

        out.classifier.weight.copy_(model.classifier.weight.detach().cpu()[:, ph] / th.view(1, -1))
        out.classifier.bias.copy_(model.classifier.bias.detach().cpu())
    return out


def average_cnn_models(models: Sequence[SmallFashionCNN]) -> SmallFashionCNN:
    if not models:
        raise ValueError("at least one model is required")
    torch, _, _ = require_torch()
    out = SmallFashionCNN(models[0].gauge_spec)
    states = [model.state_dict() for model in models]
    state = out.state_dict()
    with torch.no_grad():
        for key in state:
            state[key].copy_(torch.stack([item[key].detach().cpu() for item in states], dim=0).mean(dim=0))
    out.load_state_dict(state)
    return out


def align_cnn_to_reference(
    model: SmallFashionCNN,
    *,
    conv1_perm: Sequence[int],
    conv2_perm: Sequence[int],
    hidden_perm: Sequence[int],
) -> SmallFashionCNN:
    return apply_channel_gauge(
        model,
        conv1_perm=conv1_perm,
        conv2_perm=conv2_perm,
        hidden_perm=hidden_perm,
    )


def apply_inverse_positive_alignment(
    model: SmallFashionCNN,
    *,
    conv1_perm: Sequence[int],
    conv2_perm: Sequence[int],
    hidden_perm: Sequence[int],
    conv1_reference_to_model_scales: Sequence[float],
    conv2_reference_to_model_scales: Sequence[float],
    hidden_reference_to_model_scales: Sequence[float],
) -> SmallFashionCNN:
    """Align model activations to a reference distribution.

    If target activations are approximately ``scale * reference`` after
    permutation, the exact positive gauge needed to match the reference is the
    inverse scale.
    """

    return apply_channel_gauge(
        model,
        conv1_perm=conv1_perm,
        conv2_perm=conv2_perm,
        hidden_perm=hidden_perm,
        conv1_scales=1.0 / np.asarray(conv1_reference_to_model_scales, dtype=float),
        conv2_scales=1.0 / np.asarray(conv2_reference_to_model_scales, dtype=float),
        hidden_scales=1.0 / np.asarray(hidden_reference_to_model_scales, dtype=float),
    )
