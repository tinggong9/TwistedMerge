"""Exact block-gauge transforms for a block-compatible linear-hidden model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np

from .block_gauge_alignment import BlockPartition
from .model_merging_benchmark import require_torch


@dataclass(frozen=True)
class BlockCompatibleMetadata:
    parameter_count: int
    transformed_parameter_count: int
    same_parameter_count: bool
    is_single_model: bool
    exact_same_architecture_symmetry: bool
    adapter_extra_parameters: bool
    activation: str
    notes: str


def block_diag_from_blocks(
    blocks: Iterable[Iterable[int]],
    gauges: Mapping[int, np.ndarray],
    width: int,
) -> np.ndarray:
    matrix = np.eye(width, dtype=float)
    for block_idx, block in enumerate(blocks):
        indices = np.asarray(tuple(block), dtype=int)
        gauge = np.asarray(gauges[block_idx], dtype=float)
        if gauge.shape != (len(indices), len(indices)):
            raise ValueError(f"gauge {block_idx} has incompatible shape {gauge.shape}")
        matrix[np.ix_(indices, indices)] = gauge
    return matrix


class LinearHiddenMLP(require_torch()[1].Module):
    """One-hidden-layer MLP with identity hidden activation.

    General orthogonal block rotations are exact reparameterizations here.  This
    is intentionally separate from the repo's ReLU MLP, where block rotations
    are diagnostics unless restricted to exact ReLU symmetries.
    """

    def __init__(self, input_dim: int, width: int, num_classes: int):
        _, nn, _ = require_torch()
        super().__init__()
        self.flatten = nn.Flatten()
        self.hidden = nn.Linear(input_dim, width)
        self.classifier = nn.Linear(width, num_classes)

    def forward(self, x, return_features: bool = False):
        h = self.hidden(self.flatten(x))
        logits = self.classifier(h)
        if return_features:
            return logits, h
        return logits


def make_linear_hidden_mlp(input_dim: int, width: int, num_classes: int = 10) -> LinearHiddenMLP:
    return LinearHiddenMLP(input_dim=input_dim, width=width, num_classes=num_classes)


def parameter_count(model) -> int:
    return int(sum(int(param.numel()) for param in model.parameters()))


def clone_linear_hidden_mlp(model: LinearHiddenMLP) -> LinearHiddenMLP:
    torch, _, _ = require_torch()
    cloned = LinearHiddenMLP(
        input_dim=int(model.hidden.in_features),
        width=int(model.hidden.out_features),
        num_classes=int(model.classifier.out_features),
    )
    cloned.load_state_dict({key: value.detach().cpu().clone() for key, value in model.state_dict().items()})
    with torch.no_grad():
        for param in cloned.parameters():
            param.copy_(param.detach().cpu())
    return cloned


def transform_linear_hidden_block_gauge(
    model: LinearHiddenMLP,
    partition: BlockPartition,
    gauges: Mapping[int, np.ndarray],
) -> tuple[LinearHiddenMLP, BlockCompatibleMetadata]:
    """Apply an exact hidden-space block gauge to a linear-hidden model.

    For row-vector hidden features ``h = x W_in^T + b``, the transformed model
    uses ``h' = h Q`` and ``W_out' = W_out Q``.  Orthogonal block ``Q`` leaves
    logits unchanged with the same architecture and parameter count.
    """

    torch, _, _ = require_torch()
    width = int(model.hidden.out_features)
    q = block_diag_from_blocks(partition.blocks, gauges, width)
    transformed = clone_linear_hidden_mlp(model)
    q_tensor = torch.tensor(q, dtype=transformed.hidden.weight.dtype)
    with torch.no_grad():
        hidden_weight = model.hidden.weight.detach().cpu()
        hidden_bias = model.hidden.bias.detach().cpu()
        classifier_weight = model.classifier.weight.detach().cpu()
        classifier_bias = model.classifier.bias.detach().cpu()
        transformed.hidden.weight.copy_(q_tensor.T @ hidden_weight)
        transformed.hidden.bias.copy_(hidden_bias @ q_tensor)
        transformed.classifier.weight.copy_(classifier_weight @ q_tensor)
        transformed.classifier.bias.copy_(classifier_bias)

    before = parameter_count(model)
    after = parameter_count(transformed)
    metadata = BlockCompatibleMetadata(
        parameter_count=before,
        transformed_parameter_count=after,
        same_parameter_count=before == after,
        is_single_model=True,
        exact_same_architecture_symmetry=True,
        adapter_extra_parameters=False,
        activation="identity",
        notes="Exact block-orthogonal reparameterization for linear hidden activations.",
    )
    return transformed, metadata


def average_linear_hidden_models(models: list[LinearHiddenMLP]) -> LinearHiddenMLP:
    if not models:
        raise ValueError("at least one model is required")
    torch, _, _ = require_torch()
    first = models[0]
    merged = LinearHiddenMLP(
        input_dim=int(first.hidden.in_features),
        width=int(first.hidden.out_features),
        num_classes=int(first.classifier.out_features),
    )
    source_states = [model.state_dict() for model in models]
    state = merged.state_dict()
    with torch.no_grad():
        for key in state:
            state[key].copy_(torch.stack([src[key].detach().cpu() for src in source_states], dim=0).mean(dim=0))
    merged.load_state_dict(state)
    return merged


def max_logit_difference(model_a, model_b, inputs) -> float:
    torch, _, _ = require_torch()
    model_a.eval()
    model_b.eval()
    with torch.no_grad():
        logits_a = model_a(inputs)
        logits_b = model_b(inputs)
    return float(torch.max(torch.abs(logits_a - logits_b)).detach().cpu())
