"""Core models and integrity helpers for the holonomy application corpus."""

from __future__ import annotations

import hashlib
import random
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class CorpusSplitSizes:
    adapter_train: int
    overlap_fit: int
    overlap_validation: int
    validation: int
    test: int


def deterministic_split_indices(
    train_population: int,
    test_population: int,
    sizes: CorpusSplitSizes,
    seed: int,
) -> dict[str, np.ndarray]:
    """Create disjoint train-side splits and a separately sampled test split."""

    train_total = sizes.adapter_train + sizes.overlap_fit + sizes.overlap_validation + sizes.validation
    if train_total > train_population:
        raise ValueError("requested train-side splits exceed the training population")
    if sizes.test > test_population:
        raise ValueError("requested test split exceeds the test population")
    rng = np.random.default_rng(seed)
    train_order = rng.permutation(train_population)[:train_total]
    test_order = rng.permutation(test_population)[: sizes.test]
    offsets = np.cumsum(
        [0, sizes.adapter_train, sizes.overlap_fit, sizes.overlap_validation, sizes.validation]
    )
    names = ("adapter_train", "overlap_fit", "overlap_validation", "validation")
    result = {
        name: train_order[offsets[index] : offsets[index + 1]].astype(np.int64)
        for index, name in enumerate(names)
    }
    result["test"] = test_order.astype(np.int64)
    return result


def validate_split_indices(
    splits: Mapping[str, np.ndarray], train_population: int, test_population: int
) -> None:
    required = {"adapter_train", "overlap_fit", "overlap_validation", "validation", "test"}
    if set(splits) != required:
        raise ValueError(f"split names differ from required schema: {sorted(splits)}")
    train_names = required - {"test"}
    seen: set[int] = set()
    for name in train_names:
        values = np.asarray(splits[name], dtype=np.int64)
        if len(values) != len(np.unique(values)):
            raise ValueError(f"duplicate indices within {name}")
        if len(values) and (int(values.min()) < 0 or int(values.max()) >= train_population):
            raise ValueError(f"out-of-range train index in {name}")
        overlap = seen.intersection(map(int, values))
        if overlap:
            raise ValueError(f"train-side split overlap detected in {name}")
        seen.update(map(int, values))
    test = np.asarray(splits["test"], dtype=np.int64)
    if len(test) != len(np.unique(test)):
        raise ValueError("duplicate indices within test")
    if len(test) and (int(test.min()) < 0 or int(test.max()) >= test_population):
        raise ValueError("out-of-range test index")


class LowRankChartAdapter(nn.Module):
    """A rank-r residual feature adapter followed by a classification head."""

    def __init__(self, feature_dim: int, rank: int, classes: int = 10) -> None:
        super().__init__()
        if rank < 1 or rank > feature_dim:
            raise ValueError("rank must lie between 1 and feature_dim")
        self.feature_dim = int(feature_dim)
        self.rank = int(rank)
        self.classes = int(classes)
        self.down = nn.Linear(feature_dim, rank, bias=False)
        self.up = nn.Linear(rank, feature_dim, bias=False)
        self.head = nn.Linear(feature_dim, classes)
        nn.init.normal_(self.down.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.up.weight)

    def effective_adapter(self) -> torch.Tensor:
        identity = torch.eye(
            self.feature_dim, dtype=self.down.weight.dtype, device=self.down.weight.device
        )
        return identity + self.up.weight @ self.down.weight

    def forward_activations(self, features: torch.Tensor) -> torch.Tensor:
        return features @ self.effective_adapter().T

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_activations(features))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def classification_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    probabilities = logits.softmax(dim=1)
    predictions = logits.argmax(dim=1)
    accuracy = float((predictions == labels).float().mean())
    nll = float(nn.functional.cross_entropy(logits, labels))
    one_hot = nn.functional.one_hot(labels, logits.shape[1]).to(probabilities.dtype)
    brier = float(((probabilities - one_hot) ** 2).sum(dim=1).mean())
    confidence, predicted = probabilities.max(dim=1)
    correct = (predicted == labels).to(probabilities.dtype)
    ece = 0.0
    for lower in torch.linspace(0.0, 0.9, 10, device=logits.device):
        mask = (confidence >= lower) & (confidence < lower + 0.1)
        if bool(mask.any()):
            ece += float(mask.float().mean() * (confidence[mask].mean() - correct[mask].mean()).abs())
    return {"accuracy": accuracy, "nll": nll, "brier": brier, "ece": ece}


def train_chart_adapter(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    validation_features: torch.Tensor,
    validation_labels: torch.Tensor,
    feature_dim: int,
    rank: int,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
) -> tuple[LowRankChartAdapter, dict[str, float | int]]:
    seed_everything(seed)
    model = LowRankChartAdapter(feature_dim=feature_dim, rank=rank)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    generator = torch.Generator().manual_seed(seed + 1)
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    completed = 0
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(train_features), generator=generator)
        for indices in order.split(batch_size):
            optimizer.zero_grad(set_to_none=True)
            logits = model(train_features[indices])
            loss = nn.functional.cross_entropy(logits, train_labels[indices])
            loss.backward()
            optimizer.step()
        completed = epoch + 1
        model.eval()
        with torch.no_grad():
            validation_loss = float(
                nn.functional.cross_entropy(model(validation_features), validation_labels)
            )
        if validation_loss < best_loss - 1e-6:
            best_loss = validation_loss
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError("adapter training produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        metrics = classification_metrics(model(validation_features), validation_labels)
    metrics["epochs_completed"] = completed
    return model, metrics


def tensor_mapping_sha256(values: Mapping[str, np.ndarray | torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(values):
        array = values[name]
        if isinstance(array, torch.Tensor):
            array = array.detach().cpu().numpy()
        contiguous = np.ascontiguousarray(array)
        digest.update(name.encode("utf-8"))
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def parameter_count(model: nn.Module) -> int:
    return sum(int(parameter.numel()) for parameter in model.parameters())


def state_bytes(model: nn.Module) -> int:
    return sum(int(value.numel() * value.element_size()) for value in model.state_dict().values())
