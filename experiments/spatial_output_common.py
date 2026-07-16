#!/usr/bin/env python3
"""Shared exact actions, trained models, metrics, and evidence helpers.

The spatial-output program intentionally keeps output actions explicit.  A
prediction made in a canonical frame is never compared with a mask in another
frame: it is first transported with the same D4 element that acted on the
input.  Classification outputs are handled separately because their D4 action
is trivial.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "spatial_output_program"
TMP = Path(
    os.environ.get(
        "TWISTEDMERGE_SPATIAL_TMP_ROOT",
        ROOT / "reports" / "tmp" / "spatial_output_program",
    )
).expanduser().resolve()
DATA = Path(
    os.environ.get(
        "TWISTEDMERGE_SPATIAL_DATA_ROOT", ROOT / "data" / "Kvasir-SEG-subset"
    )
).expanduser().resolve()

from experiments.chart_followup_common import (  # noqa: E402
    ImageCNN,
    apply_d4,
    chart_parts,
    compose_d4,
    d4_table,
    inverse_chart,
    inverse_d4,
    wrong_inverse_d4,
)
from experiments.next_program_common import (  # noqa: E402
    append_csv,
    git_head,
    mps_peak_mb,
    paired_bootstrap,
    process_peak_mb,
    seed_everything,
    sha256_bytes,
    sha256_file,
    synchronize,
    torch_device,
    write_csv,
    write_json,
)

DEVICE = torch_device()
IMAGE_SIZE = int(os.environ.get("TWISTEDMERGE_SPATIAL_IMAGE_SIZE", "128"))
COMMAND_FIELDS = (
    "exact_command",
    "execution_commit",
    "source_sha256",
    "seed_scope",
    "dataset_revision_or_checksum",
    "start_time",
    "runtime_seconds",
    "exit_code",
    "state",
    "factual_summary",
)


def ensure_dirs() -> None:
    directories = (
        OUT,
        OUT / "sanity" / "plots",
        OUT / "data",
        OUT / "baselines",
        OUT / "biomedical" / "discovery" / "tables",
        OUT / "biomedical" / "discovery" / "plots",
        OUT / "biomedical" / "discovery" / "predictions",
        OUT / "biomedical" / "zeroshot" / "predictions",
        OUT / "biomedical" / "uncertainty" / "plots",
        OUT / "biomedical" / "cost",
        OUT / "multidomain" / "predictions",
        OUT / "robustness",
        OUT / "transitions",
        OUT / "confirmation",
        OUT / "landmarks",
        OUT / "extended_3d",
        OUT / "microscopy",
        OUT / "checkpoints",
        TMP,
        TMP / "checkpoints",
        TMP / "predictions",
    )
    for path in directories:
        path.mkdir(parents=True, exist_ok=True)


def utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def source_sha(path: Path) -> str:
    return sha256_file(path)


def execution_identity(path: Path) -> dict[str, str]:
    return {"execution_commit": git_head(), "source_sha256": source_sha(path)}


def record_command(
    *,
    command: str,
    source: Path,
    seed_scope: str,
    dataset_revision: str,
    started_at: str,
    runtime: float,
    exit_code: int,
    state: str,
    summary: str,
) -> None:
    ensure_dirs()
    append_csv(
        OUT / "commands.csv",
        {
            "exact_command": command,
            "execution_commit": git_head(),
            "source_sha256": source_sha(source),
            "seed_scope": seed_scope,
            "dataset_revision_or_checksum": dataset_revision,
            "start_time": started_at,
            "runtime_seconds": f"{runtime:.6f}",
            "exit_code": exit_code,
            "state": state,
            "factual_summary": summary,
        },
        COMMAND_FIELDS,
    )


def update_status(stage: str, state: str, summary: str) -> None:
    ensure_dirs()
    path = OUT / "status.json"
    payload: dict[str, Any] = {"updated_at": utc_now(), "stages": {}}
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
    payload["updated_at"] = utc_now()
    payload.setdefault("stages", {})[stage] = {
        "state": state,
        "summary": summary,
        "updated_at": payload["updated_at"],
    }
    write_json(path, payload)
    lines = ["# Spatial-output program status", ""]
    for name, row in sorted(payload["stages"].items()):
        lines.append(f"- `{name}`: {row['state']}; {row['summary']}")
    (OUT / "status.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def stage_complete(path: Path, payload: Mapping[str, Any]) -> None:
    body = dict(payload)
    body.update(execution_identity(path))
    body["completed_at"] = utc_now()
    write_json(path.parent / ".complete.json", body)
    stage = str(body.get("stage", "stage")).replace("/", "_")
    write_json(path.parent / f".{stage}.complete.json", body)


def d4_matrix(chart: int) -> np.ndarray:
    """Matrix on (x right, y down) vector components for apply_d4."""

    rotation, reflection = chart_parts(chart)
    rotate = (
        np.asarray([[1, 0], [0, 1]], dtype=np.int64),
        np.asarray([[0, 1], [-1, 0]], dtype=np.int64),
        np.asarray([[-1, 0], [0, -1]], dtype=np.int64),
        np.asarray([[0, -1], [1, 0]], dtype=np.int64),
    )[rotation]
    reflect = np.asarray([[-1, 0], [0, 1]], dtype=np.int64)
    return rotate @ (reflect if reflection else np.eye(2, dtype=np.int64))


def transform_points(points: np.ndarray, chart: int, size: int) -> np.ndarray:
    """Apply a D4 action to zero-based (x,y) pixel coordinates."""

    values = np.asarray(points, dtype=np.float64)
    center = (size - 1) / 2.0
    return (values - center) @ d4_matrix(chart).T + center


def transform_vector_field(field: torch.Tensor, chart: int) -> torch.Tensor:
    """Transform field coordinates and both vector components."""

    if field.ndim != 4 or field.shape[1] != 2:
        raise ValueError("vector field must have shape N,2,H,W")
    spatial = apply_d4(field, chart)
    matrix = torch.as_tensor(d4_matrix(chart), dtype=field.dtype, device=field.device)
    return torch.einsum("ij,njhw->nihw", matrix, spatial)


def wrong_vector_field_coordinates_only(field: torch.Tensor, chart: int) -> torch.Tensor:
    return apply_d4(field, chart)


def binary_boundary(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    values = np.asarray(mask, dtype=bool)
    eroded = ndimage.binary_erosion(values, iterations=radius, border_value=0)
    return values ^ eroded


def _binary_counts(probability: np.ndarray, target: np.ndarray) -> tuple[float, float, float]:
    prediction = np.asarray(probability) >= 0.5
    truth = np.asarray(target) >= 0.5
    intersection = float(np.logical_and(prediction, truth).sum())
    return intersection, float(prediction.sum()), float(truth.sum())


def dice_score(probability: np.ndarray, target: np.ndarray) -> float:
    intersection, predicted, truth = _binary_counts(probability, target)
    return (2.0 * intersection + 1e-8) / (predicted + truth + 1e-8)


def iou_score(probability: np.ndarray, target: np.ndarray) -> float:
    intersection, predicted, truth = _binary_counts(probability, target)
    return (intersection + 1e-8) / (predicted + truth - intersection + 1e-8)


def boundary_dice(probability: np.ndarray, target: np.ndarray, tolerance: int = 2) -> float:
    predicted = binary_boundary(np.asarray(probability) >= 0.5)
    truth = binary_boundary(np.asarray(target) >= 0.5)
    if not predicted.any() and not truth.any():
        return 1.0
    truth_near = ndimage.binary_dilation(truth, iterations=tolerance)
    predicted_near = ndimage.binary_dilation(predicted, iterations=tolerance)
    matched = float(np.logical_and(predicted, truth_near).sum()) + float(
        np.logical_and(truth, predicted_near).sum()
    )
    return matched / max(float(predicted.sum() + truth.sum()), 1.0)


def surface_distances(probability: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    predicted = binary_boundary(np.asarray(probability) >= 0.5)
    truth = binary_boundary(np.asarray(target) >= 0.5)
    if not predicted.any() or not truth.any():
        return math.nan, math.nan
    to_truth = ndimage.distance_transform_edt(~truth)[predicted]
    to_predicted = ndimage.distance_transform_edt(~predicted)[truth]
    joined = np.concatenate([to_truth, to_predicted])
    return float(np.quantile(joined, 0.95)), float(joined.mean())


def pixel_ece(probability: np.ndarray, target: np.ndarray, bins: int = 10) -> float:
    values = np.asarray(probability, dtype=np.float64).ravel()
    labels = np.asarray(target, dtype=np.float64).ravel()
    result = 0.0
    for lower in np.linspace(0.0, 1.0, bins, endpoint=False):
        chosen = (values >= lower) & (values < lower + 1.0 / bins)
        if chosen.any():
            result += float(chosen.mean()) * abs(float(values[chosen].mean() - labels[chosen].mean()))
    return result


def segmentation_metrics(probability: np.ndarray, target: np.ndarray) -> dict[str, float]:
    values = np.asarray(probability, dtype=np.float64)
    labels = np.asarray(target, dtype=np.float64)
    dice_values, iou_values, boundary_values, hd_values, assd_values = [], [], [], [], []
    for prediction, truth in zip(values, labels, strict=True):
        prediction_2d = np.squeeze(prediction)
        truth_2d = np.squeeze(truth)
        dice_values.append(dice_score(prediction_2d, truth_2d))
        iou_values.append(iou_score(prediction_2d, truth_2d))
        boundary_values.append(boundary_dice(prediction_2d, truth_2d))
        hd95, assd = surface_distances(prediction_2d, truth_2d)
        if math.isfinite(hd95):
            hd_values.append(hd95)
            assd_values.append(assd)
    binary = values >= 0.5
    truth_binary = labels >= 0.5
    tp = float(np.logical_and(binary, truth_binary).sum())
    fp = float(np.logical_and(binary, ~truth_binary).sum())
    fn = float(np.logical_and(~binary, truth_binary).sum())
    clipped = np.clip(values, 1e-6, 1 - 1e-6)
    nll = -float((labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped)).mean())
    return {
        "dice": float(np.mean(dice_values)),
        "iou": float(np.mean(iou_values)),
        "boundary_dice": float(np.mean(boundary_values)),
        "hausdorff95": float(np.mean(hd_values)) if hd_values else math.nan,
        "assd": float(np.mean(assd_values)) if assd_values else math.nan,
        "pixel_nll": nll,
        "pixel_ece": pixel_ece(values, labels),
        "foreground_recall": tp / max(tp + fn, 1.0),
        "foreground_precision": tp / max(tp + fp, 1.0),
    }


def equivariance_metrics(model: nn.Module, images: torch.Tensor, charts: torch.Tensor) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        base = torch.sigmoid(model(images.to(DEVICE))).cpu()
        transformed_input = apply_d4(images, charts)
        transformed_prediction = torch.sigmoid(model(transformed_input.to(DEVICE))).cpu()
        expected = apply_d4(base, charts)
    delta = transformed_prediction - expected
    consistency = []
    boundary_consistency = []
    for predicted, target in zip(transformed_prediction.numpy(), expected.numpy(), strict=True):
        consistency.append(dice_score(predicted[0], target[0]))
        boundary_consistency.append(boundary_dice(predicted[0], target[0]))
    return {
        "pixel_l1": float(delta.abs().mean()),
        "pixel_l2": float(torch.sqrt((delta.square()).mean())),
        "dice_consistency": float(np.mean(consistency)),
        "boundary_consistency": float(np.mean(boundary_consistency)),
    }


class ConvBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(output_channels, output_channels, 3, padding=1),
            nn.ReLU(),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.block(values)


class TinyUNet(nn.Module):
    """A small trained U-Net suitable for an 8 GB M1."""

    def __init__(self, channels: int = 3, width: int = 6):
        super().__init__()
        self.channels = channels
        self.width = width
        self.early = ConvBlock(channels, width)
        self.middle = ConvBlock(width, 2 * width)
        self.bottleneck = ConvBlock(2 * width, 4 * width)
        self.late = ConvBlock(4 * width + 2 * width, 2 * width)
        self.final = ConvBlock(2 * width + width, width)
        self.head = nn.Conv2d(width, 1, 1)

    def forward_features(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        early = self.early(images)
        middle = self.middle(F.max_pool2d(early, 2))
        bottleneck = self.bottleneck(F.max_pool2d(middle, 2))
        late = self.late(torch.cat([F.interpolate(bottleneck, size=middle.shape[-2:], mode="bilinear", align_corners=False), middle], dim=1))
        final = self.final(torch.cat([F.interpolate(late, size=early.shape[-2:], mode="bilinear", align_corners=False), early], dim=1))
        logits = self.head(final)
        return {
            "early_encoder": early,
            "middle_encoder": middle,
            "bottleneck": bottleneck,
            "late_decoder": late,
            "pre_mask_logits": logits,
        }

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.forward_features(images)["pre_mask_logits"]


class D4SymmetrizedUNet(nn.Module):
    """Exactly D4-equivariant orbit symmetrization of a trained U-Net."""

    def __init__(self, base: TinyUNet):
        super().__init__()
        self.base = base

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        outputs = []
        for chart in range(8):
            canonical = inverse_d4(images, chart)
            outputs.append(apply_d4(self.base(canonical), chart))
        return torch.stack(outputs).mean(0)


def parameter_count(model: nn.Module) -> int:
    return sum(int(value.numel()) for value in model.parameters())


def model_bytes(model: nn.Module) -> int:
    return sum(int(value.numel() * value.element_size()) for value in model.state_dict().values())


def _batch_indices(size: int, batch_size: int, seed: int) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return list(torch.randperm(size, generator=generator).split(batch_size))


def _dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probability = torch.sigmoid(logits)
    intersection = (probability * target).sum((1, 2, 3))
    return 1 - ((2 * intersection + 1) / (probability.sum((1, 2, 3)) + target.sum((1, 2, 3)) + 1)).mean()


def train_segmenter(
    model: TinyUNet,
    images: torch.Tensor,
    masks: torch.Tensor,
    validation_images: torch.Tensor,
    validation_masks: torch.Tensor,
    seed: int,
    epochs: int,
    augmentation: Callable[[torch.Tensor, torch.Tensor, int], tuple[torch.Tensor, torch.Tensor]] | None = None,
    batch_size: int = 8,
) -> tuple[TinyUNet, float, list[dict[str, float]]]:
    seed_everything(seed)
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    best_state = copy.deepcopy(model.state_dict())
    best_dice = -math.inf
    rows: list[dict[str, float]] = []
    started = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        losses = []
        for step, indices in enumerate(_batch_indices(len(images), batch_size, seed + epoch)):
            batch_images = images[indices]
            batch_masks = masks[indices]
            if augmentation is not None:
                batch_images, batch_masks = augmentation(batch_images, batch_masks, seed + epoch * 10_000 + step)
            batch_images = batch_images.to(DEVICE)
            batch_masks = batch_masks.to(DEVICE)
            logits = model(batch_images)
            loss = F.binary_cross_entropy_with_logits(logits, batch_masks) + _dice_loss(logits, batch_masks)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        validation_probability = predict_probability(model, validation_images, batch_size=batch_size)
        validation_dice = dice_score(validation_probability, validation_masks.numpy())
        rows.append({"epoch": float(epoch), "loss": float(np.mean(losses)), "validation_dice": validation_dice})
        if validation_dice > best_dice:
            best_dice = validation_dice
            best_state = copy.deepcopy(model.state_dict())
    synchronize(DEVICE)
    elapsed = time.perf_counter() - started
    model.load_state_dict(best_state)
    return model, elapsed, rows


def chart_augmentation(images: torch.Tensor, masks: torch.Tensor, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    charts = torch.randint(0, 8, (len(images),), generator=generator)
    return apply_d4(images, charts), apply_d4(masks, charts)


def train_chart_model(
    model: nn.Module,
    images: torch.Tensor,
    charts: torch.Tensor,
    validation_images: torch.Tensor,
    validation_charts: torch.Tensor,
    seed: int,
    epochs: int,
    batch_size: int = 16,
) -> tuple[nn.Module, float, float]:
    seed_everything(seed)
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    best_state = copy.deepcopy(model.state_dict())
    best_accuracy = -math.inf
    started = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        for indices in _batch_indices(len(images), batch_size, seed + epoch):
            logits = model(images[indices].to(DEVICE))
            loss = F.cross_entropy(logits, charts[indices].to(DEVICE))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        validation_logits = predict_logits(model, validation_images, batch_size)
        accuracy = float((validation_logits.argmax(1) == validation_charts).float().mean())
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_state = copy.deepcopy(model.state_dict())
    synchronize(DEVICE)
    elapsed = time.perf_counter() - started
    model.load_state_dict(best_state)
    return model, elapsed, best_accuracy


def predict_logits(model: nn.Module, images: torch.Tensor, batch_size: int = 16) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        return torch.cat([model(batch.to(DEVICE)).cpu() for batch in images.split(batch_size)])


def predict_probability(model: nn.Module, images: torch.Tensor, batch_size: int = 8) -> np.ndarray:
    return torch.sigmoid(predict_logits(model, images, batch_size)).numpy()


def chart_probabilities(model: nn.Module, images: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    return (predict_logits(model, images, 16) / temperature).softmax(1)


def calibrate_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    temperatures = torch.logspace(-1, 1, 81)
    losses = [float(F.cross_entropy(logits / value, labels)) for value in temperatures]
    return float(temperatures[int(np.argmin(losses))])


def hard_canonical_retransport(
    images: torch.Tensor,
    model: nn.Module,
    charts: torch.Tensor,
    *,
    output_action: str = "correct",
) -> torch.Tensor:
    result = torch.empty((len(images), 1, images.shape[-2], images.shape[-1]), dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        for chart in range(8):
            chosen = charts == chart
            if not bool(chosen.any()):
                continue
            canonical = inverse_d4(images[chosen], chart)
            logits = model(canonical.to(DEVICE)).cpu()
            if output_action == "correct":
                logits = apply_d4(logits, chart)
            elif output_action == "inverse":
                logits = apply_d4(logits, inverse_chart(chart))
            elif output_action == "none":
                pass
            elif output_action == "wrong_order":
                logits = wrong_inverse_d4(logits, chart)
            else:
                raise ValueError(output_action)
            result[chosen] = logits
    return result


def soft_canonical_retransport(
    images: torch.Tensor,
    models: Sequence[nn.Module],
    probabilities: torch.Tensor,
    *,
    output_action: str = "correct",
) -> torch.Tensor:
    """Output-space probability mixture sum_g p(g) g F(g^-1 x)."""

    result = torch.zeros((len(images), 1, images.shape[-2], images.shape[-1]), dtype=torch.float32)
    with torch.no_grad():
        for chart in range(8):
            canonical = inverse_d4(images, chart)
            expert_logits = torch.stack([predict_logits(model, canonical, 8) for model in models]).mean(0)
            if output_action == "correct":
                transported = apply_d4(expert_logits, chart)
            elif output_action == "none":
                transported = expert_logits
            elif output_action == "inverse":
                transported = apply_d4(expert_logits, inverse_chart(chart))
            elif output_action == "wrong_order":
                transported = wrong_inverse_d4(expert_logits, chart)
            else:
                raise ValueError(output_action)
            result += probabilities[:, chart, None, None, None] * transported
    return result


def expert_original_frame_logits(images: torch.Tensor, models: Sequence[nn.Module]) -> torch.Tensor:
    return torch.stack([predict_logits(model, images, 8) for model in models], dim=1)


def average_state_dict(models: Sequence[TinyUNet], weights: Sequence[float] | None = None) -> TinyUNet:
    if not models:
        raise ValueError("at least one model is required")
    if weights is None:
        weights = [1.0 / len(models)] * len(models)
    total = float(sum(weights))
    normalized = [float(value) / total for value in weights]
    result = TinyUNet(models[0].channels, models[0].width)
    state: dict[str, torch.Tensor] = {}
    for key in models[0].state_dict():
        state[key] = sum(weight * model.state_dict()[key].detach().cpu() for weight, model in zip(normalized, models, strict=True))
    result.load_state_dict(state)
    return result.to(DEVICE)


def save_checkpoint(path: Path, model: nn.Module, metadata: Mapping[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "metadata": dict(metadata),
    }
    torch.save(payload, path)
    return {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def load_checkpoint(path: Path, model: nn.Module) -> tuple[nn.Module, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["state_dict"])
    return model.to(DEVICE), dict(payload.get("metadata", {}))


def save_predictions_before_metrics(
    path: Path,
    predictions: Mapping[str, np.ndarray],
    labels: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    """Persist predictions, then prove label permutation cannot change them."""

    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {name: np.asarray(value, dtype=np.float16) for name, value in predictions.items()}
    hashes_before = {name: sha256_bytes(value.tobytes()) for name, value in arrays.items()}
    np.savez_compressed(path, **arrays)
    file_before = sha256_file(path)
    permuted_labels = np.asarray(labels).copy()
    np.random.default_rng(seed).shuffle(permuted_labels)
    hashes_after = {name: sha256_bytes(value.tobytes()) for name, value in arrays.items()}
    file_after = sha256_file(path)
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": file_before,
        "bytes": path.stat().st_size,
        "candidate_hashes_unchanged": hashes_before == hashes_after,
        "file_hash_unchanged": file_before == file_after,
        "permuted_labels_differ": not np.array_equal(labels, permuted_labels),
    }


def _load_image(path: Path, mask: bool, size: int) -> torch.Tensor:
    mode = "L" if mask else "RGB"
    interpolation = Image.Resampling.NEAREST if mask else Image.Resampling.BILINEAR
    with Image.open(path) as handle:
        image = handle.convert(mode).resize((size, size), interpolation)
        values = np.asarray(image, dtype=np.float32)
    if mask:
        return torch.from_numpy((values >= 127).astype(np.float32)).unsqueeze(0)
    return torch.from_numpy(values.transpose(2, 0, 1) / 255.0)


def dataset_paths(split: str) -> list[tuple[Path, Path]]:
    images = DATA / split / "images"
    masks = DATA / split / "masks"
    if not images.exists() or not masks.exists():
        return []
    result = []
    for image_path in sorted(images.glob("*.png")):
        mask_path = masks / image_path.name
        if mask_path.exists():
            result.append((image_path, mask_path))
    return result


def load_dataset_split(split: str, size: int = IMAGE_SIZE) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    pairs = dataset_paths(split)
    if not pairs:
        raise FileNotFoundError(f"no paired Kvasir-SEG files under {DATA / split}")
    images = torch.stack([_load_image(image, False, size) for image, _ in pairs])
    masks = torch.stack([_load_image(mask, True, size) for _, mask in pairs])
    return images, masks, [image.name for image, _ in pairs]


def role_split(seed: int) -> dict[str, Any]:
    train_images, train_masks, train_names = load_dataset_split("train")
    validation_images, validation_masks, validation_names = load_dataset_split("validation")
    test_images, test_masks, test_names = load_dataset_split("test")
    rng = np.random.default_rng(310_000_000 + seed)
    train_order = rng.permutation(len(train_images))
    validation_order = rng.permutation(len(validation_images))
    expert_count = max(1, int(round(0.75 * len(train_order))))
    early_count = max(1, len(validation_order) // 3)
    calibration_count = max(1, len(validation_order) // 3)
    expert_indices = train_order[:expert_count]
    chart_indices = train_order[expert_count:]
    if not len(chart_indices):
        chart_indices = train_order[-max(1, len(train_order) // 4):]
        expert_indices = train_order[: -len(chart_indices)]
    early_indices = validation_order[:early_count]
    calibration_indices = validation_order[early_count : early_count + calibration_count]
    threshold_indices = validation_order[early_count + calibration_count :]
    if not len(threshold_indices):
        threshold_indices = validation_order[-1:]
    test_limit = min(len(test_images), int(os.environ.get("TWISTEDMERGE_SPATIAL_TEST_LIMIT", "16")))
    return {
        "expert_images": train_images[expert_indices],
        "expert_masks": train_masks[expert_indices],
        "expert_names": [train_names[int(index)] for index in expert_indices],
        "chart_images": train_images[chart_indices],
        "chart_masks": train_masks[chart_indices],
        "chart_names": [train_names[int(index)] for index in chart_indices],
        "early_images": validation_images[early_indices],
        "early_masks": validation_masks[early_indices],
        "early_names": [validation_names[int(index)] for index in early_indices],
        "calibration_images": validation_images[calibration_indices],
        "calibration_masks": validation_masks[calibration_indices],
        "calibration_names": [validation_names[int(index)] for index in calibration_indices],
        "threshold_images": validation_images[threshold_indices],
        "threshold_masks": validation_masks[threshold_indices],
        "threshold_names": [validation_names[int(index)] for index in threshold_indices],
        "test_images": test_images[:test_limit],
        "test_masks": test_masks[:test_limit],
        "test_names": test_names[:test_limit],
    }


def make_chart_dataset(
    images: torch.Tensor,
    seed: int,
    allowed_charts: Sequence[int] = tuple(range(8)),
) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    charts = torch.tensor(rng.choice(np.asarray(allowed_charts), size=len(images)), dtype=torch.long)
    return apply_d4(images, charts), charts


def transformed_test(payload: Mapping[str, Any], seed: int, allowed_charts: Sequence[int] = tuple(range(8))) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    charts = torch.tensor(rng.choice(np.asarray(allowed_charts), size=len(payload["test_images"])), dtype=torch.long)
    return apply_d4(payload["test_images"], charts), apply_d4(payload["test_masks"], charts), charts


def dataset_checksum() -> str:
    digest = hashlib.sha256()
    for split in ("train", "validation", "test"):
        for image, mask in dataset_paths(split):
            for path in (image, mask):
                digest.update(path.relative_to(DATA).as_posix().encode())
                digest.update(sha256_file(path).encode())
    return digest.hexdigest()


def dataset_ready() -> bool:
    return all(dataset_paths(split) for split in ("train", "validation", "test"))


def dataset_counts() -> dict[str, int]:
    return {split: len(dataset_paths(split)) for split in ("train", "validation", "test")}


def run_checked(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=ROOT, check=False, capture_output=True, text=True)


def factual_report(path: Path, title: str, facts: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# " + title + "\n\n" + "\n".join(f"- {fact}" for fact in facts) + "\n", encoding="utf-8")


def latex_table(path: Path, columns: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    def escape(value: Any) -> str:
        return str(value).replace("_", "\\_").replace("%", "\\%")

    lines = ["\\begin{tabular}{" + "l" * len(columns) + "}", "\\toprule", " & ".join(map(escape, columns)) + " \\\\", "\\midrule"]
    for row in rows:
        lines.append(" & ".join(escape(row.get(column, "")) for column in columns) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def paired_rows(
    summaries: Sequence[Mapping[str, Any]],
    comparisons: Sequence[tuple[str, str, str]],
    metric: str,
    seed: int,
) -> list[dict[str, Any]]:
    lookup = {(int(row["seed"]), str(row["method"])): float(row[metric]) for row in summaries}
    result = []
    seeds = sorted({int(row["seed"]) for row in summaries})
    for name, left, right in comparisons:
        deltas = [lookup[(value, left)] - lookup[(value, right)] for value in seeds if (value, left) in lookup and (value, right) in lookup]
        mean, lower, upper = paired_bootstrap(deltas, seed + len(result), samples=10_000)
        result.append({"comparison": name, "left": left, "right": right, "metric": metric, "mean_delta": mean, "ci_lower": lower, "ci_upper": upper, "seeds": len(deltas)})
    return result


def measure_complete_path(
    fn: Callable[[], torch.Tensor], warmups: int, repeats: int
) -> dict[str, Any]:
    result = fn()
    synchronize(DEVICE)
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
    _ = result.shape
    return {
        "latency_median_ms": float(np.median(timings)),
        "latency_q1_ms": float(np.quantile(timings, 0.25)),
        "latency_q3_ms": float(np.quantile(timings, 0.75)),
        "peak_process_memory_mb": process_peak_mb(),
        "peak_accelerator_memory_mb": mps_peak_mb(),
        "warmups": warmups,
        "timed_repetitions": repeats,
    }


__all__ = [
    "COMMAND_FIELDS",
    "DATA",
    "DEVICE",
    "D4SymmetrizedUNet",
    "IMAGE_SIZE",
    "ImageCNN",
    "OUT",
    "ROOT",
    "TMP",
    "TinyUNet",
    "apply_d4",
    "average_state_dict",
    "binary_boundary",
    "boundary_dice",
    "calibrate_temperature",
    "chart_augmentation",
    "chart_probabilities",
    "compose_d4",
    "d4_matrix",
    "d4_table",
    "dataset_checksum",
    "dataset_counts",
    "dataset_paths",
    "dataset_ready",
    "dice_score",
    "ensure_dirs",
    "equivariance_metrics",
    "execution_identity",
    "expert_original_frame_logits",
    "factual_report",
    "git_head",
    "hard_canonical_retransport",
    "inverse_chart",
    "inverse_d4",
    "iou_score",
    "latex_table",
    "load_checkpoint",
    "load_dataset_split",
    "make_chart_dataset",
    "measure_complete_path",
    "model_bytes",
    "paired_bootstrap",
    "paired_rows",
    "parameter_count",
    "pixel_ece",
    "predict_logits",
    "predict_probability",
    "record_command",
    "role_split",
    "save_checkpoint",
    "save_predictions_before_metrics",
    "segmentation_metrics",
    "sha256_bytes",
    "sha256_file",
    "soft_canonical_retransport",
    "source_sha",
    "stage_complete",
    "surface_distances",
    "train_chart_model",
    "train_segmenter",
    "transform_points",
    "transform_vector_field",
    "transformed_test",
    "update_status",
    "utc_now",
    "write_csv",
    "write_json",
    "wrong_inverse_d4",
    "wrong_vector_field_coordinates_only",
]
