#!/usr/bin/env python3
"""Shared trained-model utilities for the focused chart follow-up program."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torchvision.datasets import CIFAR10, FashionMNIST

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.next_program_common import (
    git_head,
    mps_peak_mb,
    paired_bootstrap,
    process_peak_mb,
    seed_everything,
    sha256_file,
    synchronize,
    torch_device,
    write_csv,
    write_json,
)

OUT = ROOT / "reports" / "chart_followup"
TMP = Path(os.environ.get("TWISTEDMERGE_CHART_TMP_ROOT", ROOT / "reports" / "tmp" / "chart_followup")).expanduser().resolve()
DATA = Path(os.environ.get("TWISTEDMERGE_DATA_ROOT", ROOT / "data")).expanduser().resolve()
DEVICE = torch_device()

STAGE_DIRS = ("ablation", "zeroshot", "cifar", "cost", "compression", "sample_efficiency")


def ensure_dirs() -> None:
    for path in (OUT, TMP, TMP / "checkpoints", TMP / "logits"):
        path.mkdir(parents=True, exist_ok=True)
    for name in STAGE_DIRS:
        for child in ("", "tables", "plots"):
            (OUT / name / child).mkdir(parents=True, exist_ok=True)


def source_sha(path: Path) -> str:
    return sha256_file(path)


def chart_parts(chart: int) -> tuple[int, int]:
    return chart % 4, int(chart >= 4)


def compose_d4(left: int, right: int) -> int:
    """Return left * right for actions R^k F^b used by apply_d4."""

    left_rotation, left_reflection = chart_parts(left)
    right_rotation, right_reflection = chart_parts(right)
    rotation = (left_rotation + (-right_rotation if left_reflection else right_rotation)) % 4
    reflection = (left_reflection + right_reflection) % 2
    return rotation + 4 * reflection


def d4_table() -> np.ndarray:
    return np.asarray([[compose_d4(left, right) for right in range(8)] for left in range(8)], dtype=np.int64)


def inverse_chart(chart: int) -> int:
    for candidate in range(8):
        if compose_d4(chart, candidate) == 0 and compose_d4(candidate, chart) == 0:
            return candidate
    raise AssertionError(f"D4 chart {chart} has no inverse")


def apply_d4(images: torch.Tensor, charts: torch.Tensor | int) -> torch.Tensor:
    if isinstance(charts, int):
        chart_values = torch.full((len(images),), charts, dtype=torch.long, device=images.device)
    else:
        chart_values = charts.to(device=images.device, dtype=torch.long)
    result = torch.empty_like(images)
    for chart in range(8):
        mask = chart_values == chart
        if not bool(mask.any()):
            continue
        values = images[mask]
        if chart >= 4:
            values = torch.flip(values, dims=(-1,))
        result[mask] = torch.rot90(values, chart % 4, dims=(-2, -1))
    return result


def inverse_d4(images: torch.Tensor, chart: int) -> torch.Tensor:
    values = torch.rot90(images, -(chart % 4), dims=(-2, -1))
    return torch.flip(values, dims=(-1,)) if chart >= 4 else values


def wrong_inverse_d4(images: torch.Tensor, chart: int) -> torch.Tensor:
    """Deliberately use the incorrect reflection/rotation order."""

    values = torch.flip(images, dims=(-1,)) if chart >= 4 else images
    return torch.rot90(values, -(chart % 4), dims=(-2, -1))


class ImageCNN(nn.Module):
    def __init__(self, outputs: int, channels: int, width: int = 12):
        super().__init__()
        self.channels = channels
        self.width = width
        self.features = nn.Sequential(
            nn.Conv2d(channels, width, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(width, 2 * width, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.feature_size = 2 * width * 7 * 7
        self.head = nn.Linear(self.feature_size, outputs)

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(images)
        # Fashion-MNIST produces 7x7 maps and CIFAR-10 produces 8x8 maps.
        # Symmetric interpolation preserves spatial capacity while avoiding
        # the MPS adaptive-pooling divisibility limitation on 28x28 inputs.
        if features.shape[-2:] != (7, 7):
            features = nn.functional.interpolate(features, size=(7, 7), mode="bilinear", align_corners=False)
        return features.flatten(1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(images))


class D4EquivariantChartCNN(nn.Module):
    """Eight regular-representation scores from one shared learned scorer."""

    def __init__(self, channels: int, width: int = 10):
        super().__init__()
        self.channels = channels
        self.width = width
        self.score = ImageCNN(1, channels=channels, width=width)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.score(inverse_d4(images, chart)) for chart in range(8)], dim=1)


class OrbitTaskCNN(nn.Module):
    """Direct D4-invariant task CNN with a learned shared orbit scorer."""

    def __init__(self, channels: int, width: int = 10, classes: int = 10):
        super().__init__()
        self.channels = channels
        self.width = width
        self.base = ImageCNN(classes, channels=channels, width=width)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.base(inverse_d4(images, chart)) for chart in range(8)]).mean(0)


class LowRankContextAdapter(nn.Module):
    def __init__(self, channels: int, width: int = 10, rank: int = 4, classes: int = 10):
        super().__init__()
        self.backbone = ImageCNN(classes, channels=channels, width=width)
        self.chart_down = nn.Linear(8, rank, bias=False)
        self.feature_down = nn.Linear(self.backbone.feature_size, rank, bias=False)
        self.up = nn.Linear(rank, classes, bias=False)

    def forward(self, images: torch.Tensor, chart_probabilities: torch.Tensor) -> torch.Tensor:
        features = self.backbone.forward_features(images)
        logits = self.backbone.head(features)
        return logits + self.up(self.feature_down(features) * self.chart_down(chart_probabilities))


class LearnedMultiplicationChartCNN(nn.Module):
    """Equivariant image scorer with a learned, auditable multiplication tensor."""

    def __init__(self, channels: int, width: int = 10):
        super().__init__()
        self.image = D4EquivariantChartCNN(channels=channels, width=width)
        self.table_logits = nn.Parameter(torch.zeros(8, 8, 8))
        self.table_scale = nn.Parameter(torch.tensor(0.0))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        marginal = self.table_logits.softmax(-1).mean((0, 1))
        return self.image(images) + self.table_scale.tanh() * marginal


def parameter_count(model: nn.Module) -> int:
    return sum(int(value.numel()) for value in model.parameters())


def model_bytes(model: nn.Module) -> int:
    return sum(int(value.numel() * value.element_size()) for value in model.state_dict().values())


def batches(size: int, batch_size: int, seed: int) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return list(torch.randperm(size, generator=generator).split(batch_size))


def dataset_tensors(dataset: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    if dataset == "FashionMNIST":
        training = FashionMNIST(DATA, train=True, download=False)
        testing = FashionMNIST(DATA, train=False, download=False)
        return (
            training.data.float().unsqueeze(1) / 255.0,
            training.targets.long(),
            testing.data.float().unsqueeze(1) / 255.0,
            testing.targets.long(),
            1,
        )
    if dataset == "CIFAR10":
        training = CIFAR10(DATA, train=True, download=False)
        testing = CIFAR10(DATA, train=False, download=False)
        train_images = torch.from_numpy(training.data).permute(0, 3, 1, 2).float() / 255.0
        test_images = torch.from_numpy(testing.data).permute(0, 3, 1, 2).float() / 255.0
        return train_images, torch.tensor(training.targets), test_images, torch.tensor(testing.targets), 3
    raise ValueError(f"unsupported dataset: {dataset}")


def split_indices(seed: int, population: int, local_train: int = 6000) -> dict[str, np.ndarray]:
    order = np.random.default_rng(210_000_000 + seed).permutation(population)
    sizes = {
        "local_train": local_train,
        "chart_train": 1000,
        "validation": 500,
        "calibration": 500,
        "threshold": 500,
    }
    result: dict[str, np.ndarray] = {}
    start = 0
    for name, size in sizes.items():
        result[name] = order[start : start + size]
        start += size
    return result


def make_chart_examples(
    images: torch.Tensor,
    seed: int,
    allowed_charts: Sequence[int],
    condition: str,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    rng = np.random.default_rng(seed)
    charts = rng.choice(np.asarray(allowed_charts, dtype=np.int64), size=len(images))
    transformed = apply_d4(images, torch.tensor(charts, dtype=torch.long))
    return transformed, torch.tensor(charts, dtype=torch.long), np.asarray([condition] * len(images))


CONDITIONS = (
    "seen_rotations",
    "heldout_rotations",
    "heldout_reflections",
    "heldout_compositions",
    "gaussian_corruption",
    "blur",
    "color_jitter",
    "chart_distribution_shift",
    "ambiguous_symmetric_inputs",
)


def conditioned_test_examples(
    images: torch.Tensor,
    seed: int,
    include_color: bool,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    rng = np.random.default_rng(seed)
    active = list(CONDITIONS if include_color else tuple(value for value in CONDITIONS if value != "color_jitter"))
    per = len(images) // len(active)
    charts: list[np.ndarray] = []
    conditions: list[str] = []
    choices = {
        "seen_rotations": [0, 1, 4],
        "heldout_rotations": [2, 3],
        "heldout_reflections": [5],
        "heldout_compositions": [6, 7],
        "gaussian_corruption": list(range(8)),
        "blur": list(range(8)),
        "color_jitter": list(range(8)),
        "chart_distribution_shift": [6, 7, 7, 7],
        "ambiguous_symmetric_inputs": list(range(8)),
    }
    for index, condition in enumerate(active):
        count = per if index < len(active) - 1 else len(images) - per * (len(active) - 1)
        charts.append(rng.choice(choices[condition], size=count))
        conditions.extend([condition] * count)
    chart_array = np.concatenate(charts).astype(np.int64)
    transformed = apply_d4(images, torch.tensor(chart_array))
    condition_array = np.asarray(conditions)
    generator = torch.Generator().manual_seed(seed + 97)
    gaussian = torch.tensor(condition_array == "gaussian_corruption")
    if bool(gaussian.any()):
        noise = torch.randn(transformed[gaussian].shape, generator=generator)
        transformed[gaussian] = (transformed[gaussian] + 0.20 * noise).clamp(0, 1)
    blur = torch.tensor(condition_array == "blur")
    if bool(blur.any()):
        transformed[blur] = nn.functional.avg_pool2d(transformed[blur], 3, stride=1, padding=1)
    color = torch.tensor(condition_array == "color_jitter")
    if bool(color.any()):
        scales = torch.linspace(0.65, 1.35, int(color.sum())).view(-1, 1, 1, 1)
        transformed[color] = (transformed[color] * scales).clamp(0, 1)
    ambiguous = torch.tensor(condition_array == "ambiguous_symmetric_inputs")
    if bool(ambiguous.any()):
        transformed[ambiguous] = 0.5 * (transformed[ambiguous] + torch.rot90(transformed[ambiguous], 2, (-2, -1)))
    return transformed, torch.tensor(chart_array), condition_array


def train_classifier(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    validation_images: torch.Tensor | None,
    validation_labels: torch.Tensor | None,
    seed: int,
    epochs: int,
    batch_size: int = 128,
    lr: float = 0.002,
    augment: Callable[[torch.Tensor, int], torch.Tensor] | None = None,
) -> tuple[nn.Module, float, int]:
    seed_everything(seed)
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_loss = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    started = time.perf_counter()
    completed_epochs = 0
    for epoch in range(epochs):
        model.train()
        for indices in batches(len(images), batch_size, seed + epoch):
            batch = images[indices]
            if augment is not None:
                batch = augment(batch, seed + epoch * 1009 + int(indices[0]))
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch.to(DEVICE))
            loss = nn.functional.cross_entropy(logits, labels[indices].to(DEVICE))
            loss.backward()
            optimizer.step()
        completed_epochs = epoch + 1
        if validation_images is None or validation_labels is None:
            continue
        model.eval()
        with torch.no_grad():
            values = torch.cat([model(part.to(DEVICE)).cpu() for part in validation_images.split(128)])
            validation_loss = float(nn.functional.cross_entropy(values, validation_labels))
        if validation_loss < best_loss - 1e-5:
            best_loss = validation_loss
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= 2:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model.eval(), time.perf_counter() - started, completed_epochs


def ordinary_chart_augmentation(images: torch.Tensor, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    noise = 0.04 * torch.randn(images.shape, generator=generator)
    shifted = torch.roll(images, shifts=(seed % 3 - 1, (seed // 3) % 3 - 1), dims=(-2, -1))
    return (shifted + noise).clamp(0, 1)


def model_logits(model: nn.Module, images: torch.Tensor, batch_size: int = 128) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        return torch.cat([model(part.to(DEVICE)).cpu() for part in images.split(batch_size)])


def model_features(model: ImageCNN, images: torch.Tensor, batch_size: int = 128) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        return torch.cat([model.forward_features(part.to(DEVICE)).cpu() for part in images.split(batch_size)])


def train_adapter(
    model: LowRankContextAdapter,
    images: torch.Tensor,
    labels: torch.Tensor,
    probabilities: torch.Tensor,
    validation_images: torch.Tensor,
    validation_labels: torch.Tensor,
    validation_probabilities: torch.Tensor,
    seed: int,
    epochs: int,
) -> tuple[LowRankContextAdapter, float]:
    seed_everything(seed)
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    best = math.inf
    state = None
    started = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        for indices in batches(len(images), 64, seed + epoch):
            optimizer.zero_grad(set_to_none=True)
            logits = model(images[indices].to(DEVICE), probabilities[indices].to(DEVICE))
            loss = nn.functional.cross_entropy(logits, labels[indices].to(DEVICE))
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            logits = model(validation_images.to(DEVICE), validation_probabilities.to(DEVICE))
            loss = float(nn.functional.cross_entropy(logits, validation_labels.to(DEVICE)))
        if loss < best:
            best = loss
            state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    if state is not None:
        model.load_state_dict(state)
    return model.eval(), time.perf_counter() - started


def calibrate_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    candidates = np.geomspace(0.35, 4.0, 41)
    losses = [float(nn.functional.cross_entropy(logits / float(value), labels)) for value in candidates]
    return float(candidates[int(np.argmin(losses))])


def chart_probabilities(model: nn.Module, images: torch.Tensor, temperature: float) -> torch.Tensor:
    return (model_logits(model, images) / temperature).softmax(1)


def task_branches(images: torch.Tensor, experts: Sequence[ImageCNN], batch_size: int = 128) -> torch.Tensor:
    branches = []
    for chart in range(8):
        canonical = inverse_d4(images, chart)
        expert_logits = [model_logits(expert, canonical, batch_size) for expert in experts]
        branches.append(torch.stack(expert_logits).mean(0))
    return torch.stack(branches, dim=1)


def one_expert_branches(images: torch.Tensor, expert: ImageCNN, batch_size: int = 128) -> torch.Tensor:
    return torch.stack([model_logits(expert, inverse_d4(images, chart), batch_size) for chart in range(8)], dim=1)


def task_feature_branches(images: torch.Tensor, experts: Sequence[ImageCNN], batch_size: int = 128) -> torch.Tensor:
    branches = []
    for chart in range(8):
        canonical = inverse_d4(images, chart)
        branches.append(torch.stack([model_features(expert, canonical, batch_size) for expert in experts]).mean(0))
    return torch.stack(branches, dim=1)


def retransport_logits(
    images: torch.Tensor,
    experts: Sequence[ImageCNN],
    probabilities: torch.Tensor,
    batch_size: int = 128,
) -> torch.Tensor:
    features = task_feature_branches(images, experts, batch_size)
    pooled = torch.einsum("nb,nbf->nf", probabilities, features)
    weight = torch.stack([expert.head.weight.detach().cpu() for expert in experts]).mean(0)
    bias = torch.stack([expert.head.bias.detach().cpu() for expert in experts]).mean(0)
    return pooled @ weight.T + bias


def wrong_order_branches(images: torch.Tensor, experts: Sequence[ImageCNN], batch_size: int = 128) -> torch.Tensor:
    branches = []
    for chart in range(8):
        wrong = wrong_inverse_d4(images, chart)
        branches.append(torch.stack([model_logits(expert, wrong, batch_size) for expert in experts]).mean(0))
    return torch.stack(branches, dim=1)


def specialized_expert_logits(images: torch.Tensor, experts: Sequence[ImageCNN], batch_size: int = 128) -> torch.Tensor:
    values = [model_logits(expert, images, batch_size) for expert in experts]
    # Each expert owns the rotation and reflection with the same rotation index.
    return torch.stack([values[chart % 4] for chart in range(8)], dim=1)


def d4_tta_logits(images: torch.Tensor, model: ImageCNN, batch_size: int = 128) -> torch.Tensor:
    return torch.stack([model_logits(model, inverse_d4(images, chart), batch_size) for chart in range(8)]).mean(0)


def choose_abstention_threshold(
    confidence: torch.Tensor,
    structured_logits: torch.Tensor,
    fallback_logits: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    candidates = torch.quantile(confidence, torch.linspace(0, 1, 21)).unique()
    scored = []
    for threshold in candidates:
        logits = torch.where((confidence >= threshold).unsqueeze(1), structured_logits, fallback_logits)
        scored.append((float((logits.argmax(1) == labels).float().mean()), -float(threshold), float(threshold)))
    return max(scored)[2]


def classwise_ece(probabilities: np.ndarray, labels: np.ndarray, bins: int = 10) -> list[float]:
    result = []
    for class_index in range(probabilities.shape[1]):
        confidence = probabilities[:, class_index]
        truth = labels == class_index
        value = 0.0
        for lower in np.linspace(0.0, 1.0 - 1.0 / bins, bins):
            mask = (confidence >= lower) & (confidence < lower + 1.0 / bins)
            if mask.any():
                value += float(mask.mean()) * abs(float(truth[mask].mean()) - float(confidence[mask].mean()))
        result.append(value)
    return result


def extended_metrics(logits: torch.Tensor | np.ndarray, labels: torch.Tensor | np.ndarray) -> dict[str, object]:
    values = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.int64)
    shifted = values - values.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True).clip(min=1e-300)
    predictions = probabilities.argmax(1)
    confidence = probabilities.max(1)
    correct = predictions == targets
    ece = 0.0
    for lower in np.linspace(0.0, 0.9, 10):
        mask = (confidence >= lower) & (confidence < lower + 0.1)
        if mask.any():
            ece += float(mask.mean()) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return {
        "task_accuracy": float(correct.mean()),
        "negative_log_likelihood": float(-np.log(probabilities[np.arange(len(targets)), targets].clip(min=1e-300)).mean()),
        "ece": float(ece),
        "classwise_ece": json.dumps(classwise_ece(probabilities, targets)),
    }


def save_logits_before_evaluation(
    name: str,
    candidates: Mapping[str, torch.Tensor | np.ndarray],
    test_labels: torch.Tensor | np.ndarray,
    seed: int,
) -> dict[str, object]:
    path = TMP / "logits" / f"{name}.npz"
    arrays = {key: np.ascontiguousarray(np.asarray(value), dtype=np.float32) for key, value in candidates.items()}
    np.savez_compressed(path, **arrays)
    hashes_before = {key: hashlib.sha256(value.tobytes()).hexdigest() for key, value in arrays.items()}
    file_before = sha256_file(path)
    permuted = np.asarray(test_labels).copy()
    np.random.default_rng(seed).shuffle(permuted)
    hashes_after = {key: hashlib.sha256(value.tobytes()).hexdigest() for key, value in arrays.items()}
    file_after = sha256_file(path)
    return {
        "logits_path": str(path.relative_to(ROOT)),
        "logits_sha256": file_before,
        "candidate_hashes_unchanged": hashes_before == hashes_after,
        "file_hash_unchanged": file_before == file_after,
        "permuted_labels_differ": not np.array_equal(np.asarray(test_labels), permuted),
    }


def paired_interval_rows(
    runs: Sequence[Mapping[str, object]],
    comparisons: Sequence[tuple[str, str, str]],
    metric: str,
    seed: int,
) -> list[dict[str, object]]:
    rows = []
    seeds = sorted({int(row["seed"]) for row in runs})
    for label, left, right in comparisons:
        deltas = []
        for collection_seed in seeds:
            left_value = next(float(row[metric]) for row in runs if int(row["seed"]) == collection_seed and row["method"] == left)
            right_value = next(float(row[metric]) for row in runs if int(row["seed"]) == collection_seed and row["method"] == right)
            deltas.append(left_value - right_value)
        mean, low, high = paired_bootstrap(deltas, seed + len(rows))
        rows.append({"comparison": label, "left_method": left, "right_method": right, "metric": metric, "collections": len(deltas), "mean_delta": mean, "ci_low": low, "ci_high": high})
    return rows


def measure_actual(fn: Callable[[], Any], warmups: int, repeats: int) -> dict[str, float | int | None]:
    started = time.perf_counter()
    result = fn()
    synchronize(DEVICE)
    cold = (time.perf_counter() - started) * 1000.0
    for _ in range(warmups):
        result = fn()
    synchronize(DEVICE)
    timings = []
    for _ in range(repeats):
        synchronize(DEVICE)
        started = time.perf_counter()
        result = fn()
        synchronize(DEVICE)
        timings.append((time.perf_counter() - started) * 1000.0)
    if hasattr(result, "shape"):
        _ = result.shape
    return {
        "cold_start_latency_ms": cold,
        "warm_start_latency_ms": float(np.median(timings)),
        "latency_q1_ms": float(np.quantile(timings, 0.25)),
        "latency_q3_ms": float(np.quantile(timings, 0.75)),
        "peak_process_memory_mb": process_peak_mb(),
        "peak_accelerator_memory_mb": mps_peak_mb(),
        "warmups": warmups,
        "timed_repetitions": repeats,
    }


def checkpoint_payload(models: Mapping[str, nn.Module], **metadata: object) -> dict[str, object]:
    return {
        **metadata,
        "models": {
            name: {key: value.detach().cpu() for key, value in model.state_dict().items()}
            for name, model in models.items()
        },
    }


def factual_report(path: Path, title: str, paragraphs: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n\n".join([f"# {title}", *paragraphs]) + "\n", encoding="utf-8")


def provenance(script: Path, command: str, seed: int | str) -> dict[str, object]:
    return {
        "execution_commit": git_head(),
        "source_sha256": source_sha(script),
        "command": command,
        "seed": seed,
    }


ensure_dirs()
