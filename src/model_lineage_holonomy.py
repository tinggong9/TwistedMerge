"""Natural task-lineage construction for the model-lineage holonomy audit.

The module deliberately contains no D4 chart action or gauge scrambling.  Its
vertices are produced by sequential optimization from a shared checkpoint.
"""

from __future__ import annotations

import copy
import hashlib
import random
from dataclasses import dataclass
from itertools import permutations
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from src.holonomy_application_corpus import LowRankChartAdapter, classification_metrics


TASKS = ("A", "B", "C")
THREE_TASK_ORDERS = tuple("".join(order) for order in permutations(TASKS))
TWO_TASK_ORDERS = tuple(left + right for left in TASKS for right in TASKS if left != right)


@dataclass(frozen=True)
class LineageNode:
    name: str
    order: str
    parent: str | None
    appended_task: str | None
    depth: int


@dataclass(frozen=True)
class LineageSplitSizes:
    adaptation_train: int = 2500
    transport_fit: int = 384
    transport_validation: int = 384
    transport_test: int = 384
    model_validation: int = 512
    application_test: int = 1000


@dataclass(frozen=True)
class AdaptationResult:
    model: LowRankChartAdapter
    epochs: int
    optimizer_steps: int
    examples_per_epoch: int
    final_training_loss: float
    wall_seconds: float


def lineage_nodes() -> tuple[LineageNode, ...]:
    nodes = [LineageNode("M0", "", None, None, 0)]
    for task in TASKS:
        nodes.append(LineageNode(f"M_{task}", task, "M0", task, 1))
    for order in TWO_TASK_ORDERS:
        nodes.append(LineageNode(f"M_{order}", order, f"M_{order[:-1]}", order[-1], 2))
    for order in THREE_TASK_ORDERS:
        nodes.append(LineageNode(f"M_{order}", order, f"M_{order[:-1]}", order[-1], 3))
    return tuple(nodes)


def lineage_node_map() -> dict[str, LineageNode]:
    return {node.name: node for node in lineage_nodes()}


def lineage_edges(*, directed_both_ways: bool = False) -> tuple[tuple[str, str], ...]:
    edges = [(node.parent, node.name) for node in lineage_nodes() if node.parent is not None]
    if directed_both_ways:
        edges = [edge for pair in edges for edge in (pair, (pair[1], pair[0]))]
    return tuple(edges)


def two_task_square(task_left: str, task_right: str) -> tuple[str, ...]:
    if task_left not in TASKS or task_right not in TASKS or task_left == task_right:
        raise ValueError("a square requires two distinct registered tasks")
    return ("M0", f"M_{task_left}", f"M_{task_left}{task_right}", f"M_{task_right}", "M0")


def two_task_squares() -> dict[str, tuple[str, ...]]:
    return {
        "AB_square": two_task_square("A", "B"),
        "AC_square": two_task_square("A", "C"),
        "BC_square": two_task_square("B", "C"),
    }


def adjacent_swap_pairs() -> tuple[tuple[str, str], ...]:
    return (
        ("ABC", "BAC"),
        ("ABC", "ACB"),
        ("ACB", "CAB"),
        ("BAC", "BCA"),
        ("BCA", "CBA"),
        ("CAB", "CBA"),
    )


def order_comparison_pairs() -> tuple[tuple[str, str], ...]:
    return (("AB", "BA"), ("AC", "CA"), ("BC", "CB"), *adjacent_swap_pairs())


def branch_pairs() -> tuple[tuple[str, str], ...]:
    return (("A", "B"), ("A", "C"), ("B", "C"))


def deterministic_split_indices(
    train_population: int,
    test_population: int,
    sizes: LineageSplitSizes = LineageSplitSizes(),
    seed: int = 7192026,
) -> dict[str, np.ndarray]:
    train_counts = (
        sizes.adaptation_train,
        sizes.transport_fit,
        sizes.transport_validation,
        sizes.transport_test,
        sizes.model_validation,
    )
    if sum(train_counts) > train_population or sizes.application_test > test_population:
        raise ValueError("requested lineage splits exceed the available population")
    train_order = np.random.default_rng(seed).permutation(train_population)[: sum(train_counts)]
    offsets = np.cumsum((0, *train_counts))
    names = (
        "adaptation_train",
        "transport_fit",
        "transport_validation",
        "transport_test",
        "model_validation",
    )
    result = {
        name: train_order[offsets[index] : offsets[index + 1]].astype(np.int64)
        for index, name in enumerate(names)
    }
    result["application_test"] = np.random.default_rng(seed + 1).permutation(test_population)[
        : sizes.application_test
    ].astype(np.int64)
    validate_lineage_splits(result, train_population, test_population)
    return result


def validate_lineage_splits(
    splits: Mapping[str, np.ndarray], train_population: int, test_population: int
) -> None:
    required = {
        "adaptation_train",
        "transport_fit",
        "transport_validation",
        "transport_test",
        "model_validation",
        "application_test",
    }
    if set(splits) != required:
        raise ValueError("lineage split names differ from the frozen schema")
    seen: set[int] = set()
    for name in sorted(required - {"application_test"}):
        values = np.asarray(splits[name], dtype=np.int64)
        if len(values) != len(np.unique(values)):
            raise ValueError(f"duplicate indices within {name}")
        if len(values) and (values.min() < 0 or values.max() >= train_population):
            raise ValueError(f"out-of-range train index in {name}")
        if seen.intersection(int(value) for value in values):
            raise ValueError(f"train-side split overlap detected in {name}")
        seen.update(int(value) for value in values)
    test = np.asarray(splits["application_test"], dtype=np.int64)
    if len(test) != len(np.unique(test)):
        raise ValueError("duplicate application-test indices")
    if len(test) and (test.min() < 0 or test.max() >= test_population):
        raise ValueError("out-of-range application-test index")


def _gaussian_kernel(size: int = 5, sigma: float = 1.0) -> torch.Tensor:
    coordinate = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
    kernel = torch.exp(-(coordinate**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return torch.outer(kernel, kernel)


def apply_task_corruption(
    images: torch.Tensor,
    task: str,
    identity_indices: Sequence[int] | np.ndarray | torch.Tensor,
) -> torch.Tensor:
    """Apply one frozen corruption without consulting labels.

    ``images`` must be NCHW float data in ``[0,1]``.  The Gaussian noise is
    seeded per dataset identity, so it is independent of batching and order.
    """

    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("images must have shape N x 3 x H x W")
    indices = [int(value) for value in identity_indices]
    if len(indices) != len(images):
        raise ValueError("identity index count must match image count")
    x = images.float()
    if task == "A":
        noise = []
        for index in indices:
            generator = torch.Generator(device="cpu").manual_seed(42101 + index)
            noise.append(torch.randn(x.shape[1:], generator=generator, dtype=x.dtype))
        return (x + 0.15 * torch.stack(noise).to(x.device)).clamp(0.0, 1.0)
    if task == "B":
        kernel = _gaussian_kernel(5, 1.0).to(device=x.device, dtype=x.dtype)
        weight = kernel.view(1, 1, 5, 5).repeat(3, 1, 1, 1)
        padded = nn.functional.pad(x, (2, 2, 2, 2), mode="reflect")
        return nn.functional.conv2d(padded, weight, groups=3)
    if task == "C":
        scales = torch.tensor((1.10, 0.90, 1.00), device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        offsets = torch.tensor((0.04, -0.02, 0.00), device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        return (((x - 0.5) * 1.35 + 0.5) * scales + offsets).clamp(0.0, 1.0)
    raise ValueError(f"unknown task: {task}")


def clone_adapter(model: LowRankChartAdapter) -> LowRankChartAdapter:
    clone = LowRankChartAdapter(model.feature_dim, model.rank, model.classes)
    clone.load_state_dict(copy.deepcopy(model.state_dict()))
    clone.eval()
    return clone


def state_dict_sha256(model_or_state: nn.Module | Mapping[str, torch.Tensor]) -> str:
    state = model_or_state.state_dict() if isinstance(model_or_state, nn.Module) else model_or_state
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def stable_training_seed(independent_seed: int, task: str) -> int:
    if task not in TASKS:
        raise ValueError(f"unknown task: {task}")
    return 820000 + int(independent_seed) * 1000 + TASKS.index(task) * 100


def adapt_on_task(
    parent: LowRankChartAdapter,
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    independent_seed: int,
    task: str,
    epochs: int = 12,
    batch_size: int = 256,
    learning_rate: float = 0.003,
    weight_decay: float = 0.0001,
) -> AdaptationResult:
    """Train one lineage edge with a fixed per-task shuffle schedule."""

    import time

    if len(features) != len(labels) or len(features) == 0:
        raise ValueError("features and labels must be nonempty and aligned")
    model = clone_adapter(parent)
    seed = stable_training_seed(independent_seed, task)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    generator = torch.Generator().manual_seed(seed + 1)
    steps = 0
    final_loss = float("nan")
    started = time.perf_counter()
    model.train()
    for _epoch in range(int(epochs)):
        order = torch.randperm(len(features), generator=generator)
        for indices in order.split(int(batch_size)):
            optimizer.zero_grad(set_to_none=True)
            logits = model(features[indices])
            loss = nn.functional.cross_entropy(logits, labels[indices])
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach())
            steps += 1
    model.eval()
    return AdaptationResult(
        model=model,
        epochs=int(epochs),
        optimizer_steps=steps,
        examples_per_epoch=len(features),
        final_training_loss=final_loss,
        wall_seconds=float(time.perf_counter() - started),
    )


def representations(model: LowRankChartAdapter, late_features: torch.Tensor) -> dict[str, torch.Tensor]:
    with torch.no_grad():
        delta = late_features @ (model.up.weight @ model.down.weight).T
        penultimate = late_features + delta
    return {"adapter": delta, "penultimate": penultimate}


def evaluate_domains(
    model: LowRankChartAdapter,
    domain_features: Mapping[str, torch.Tensor],
    labels: torch.Tensor,
) -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    model.eval()
    with torch.no_grad():
        for task in TASKS:
            rows[task] = classification_metrics(model(domain_features[task]), labels)
    return rows


def parameter_distance(left: LowRankChartAdapter, right: LowRankChartAdapter) -> float:
    left_values = torch.cat([value.detach().reshape(-1).double() for value in left.state_dict().values()])
    right_values = torch.cat([value.detach().reshape(-1).double() for value in right.state_dict().values()])
    denominator = torch.linalg.norm(left_values).clamp_min(1e-12)
    return float(torch.linalg.norm(left_values - right_values) / denominator)


def prediction_disagreement(left_logits: torch.Tensor, right_logits: torch.Tensor) -> float:
    if left_logits.shape != right_logits.shape:
        raise ValueError("logit tensors must have equal shape")
    return float((left_logits.argmax(1) != right_logits.argmax(1)).float().mean())


def feature_discrepancy(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.shape != right.shape:
        raise ValueError("feature tensors must have equal shape")
    denominator = torch.linalg.norm(right).clamp_min(1e-12)
    return float(torch.linalg.norm(left - right) / denominator)


def order_sensitivity_score(
    *,
    mean_accuracy_delta: float,
    worst_accuracy_delta: float,
    forgetting_delta: float,
    disagreement: float,
    feature_difference: float,
    ece_delta: float,
    checkpoint_distance: float,
) -> float:
    return float(
        0.30 * abs(mean_accuracy_delta)
        + 0.20 * abs(worst_accuracy_delta)
        + 0.20 * abs(forgetting_delta)
        + 0.10 * disagreement
        + 0.10 * feature_difference / (1.0 + feature_difference)
        + 0.05 * abs(ece_delta)
        + 0.05 * checkpoint_distance / (1.0 + checkpoint_distance)
    )


def state_parameter_count(model: nn.Module) -> tuple[int, int]:
    trainable = sum(int(parameter.numel()) for parameter in model.parameters() if parameter.requires_grad)
    total = sum(int(parameter.numel()) for parameter in model.parameters())
    return trainable, total


def state_bytes(model: nn.Module) -> int:
    return sum(int(value.numel() * value.element_size()) for value in model.state_dict().values())


def stack_domains(values: Mapping[str, torch.Tensor], tasks: Iterable[str] = TASKS) -> torch.Tensor:
    return torch.cat([values[task] for task in tasks], dim=0)
