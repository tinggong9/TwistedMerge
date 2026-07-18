"""Credible CIFAR ResNet-18 training utilities for the post-ICLR benchmark.

The helpers in this module deliberately keep the pilot validation-only.  Test
data are loaded only by a later confirmatory evaluator after the recipe has
been frozen.
"""

from __future__ import annotations

import hashlib
import math
import os
import random
import resource
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


@dataclass(frozen=True)
class TrainingRecipe:
    epochs: int = 150
    batch_size: int = 128
    learning_rate: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    warmup_epochs: int = 5
    label_smoothing: float = 0.0
    validation_size: int = 5000
    split_seed: int = 24680
    num_workers: int = 2
    # PyTorch 2.12 MPS backward currently rejects channels-last ResNet tensors.
    channels_last: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str):
    import torch

    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def synchronize(device) -> None:
    import torch

    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def peak_rss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value / (1024.0 * 1024.0) if sys.platform == "darwin" else value / 1024.0


def make_cifar_resnet18(num_classes: int = 10):
    """Return torchvision ResNet-18 with the standard CIFAR stem."""

    import torch
    import torchvision

    model = torchvision.models.resnet18(weights=None, num_classes=num_classes)
    model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = torch.nn.Identity()
    return model


def parameter_count(model) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def deterministic_split_indices(length: int, validation_size: int, seed: int) -> tuple[list[int], list[int]]:
    if validation_size <= 0 or validation_size >= length:
        raise ValueError("validation_size must lie strictly between zero and dataset length")
    generator = np.random.default_rng(seed)
    order = generator.permutation(length)
    validation = order[:validation_size].astype(int).tolist()
    training = order[validation_size:].astype(int).tolist()
    return training, validation


def _limit_indices(indices: list[int], limit: int, seed: int) -> list[int]:
    if limit <= 0 or limit >= len(indices):
        return indices
    generator = np.random.default_rng(seed)
    chosen = generator.choice(np.asarray(indices), size=limit, replace=False)
    return chosen.astype(int).tolist()


def _seed_worker(worker_id: int) -> None:
    import torch

    seed = int(torch.initial_seed() % (2**32))
    np.random.seed(seed)
    random.seed(seed)


def cifar10_train_val_loaders(
    data_root: Path,
    recipe: TrainingRecipe,
    *,
    model_seed: int,
    train_limit: int = 0,
    validation_limit: int = 0,
    download: bool = True,
):
    """Create disjoint augmented-train and deterministic-validation loaders."""

    import torch
    import torchvision
    import torchvision.transforms as transforms

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    evaluation_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)]
    )
    augmented = torchvision.datasets.CIFAR10(data_root, train=True, download=download, transform=train_transform)
    evaluation = torchvision.datasets.CIFAR10(data_root, train=True, download=False, transform=evaluation_transform)
    training_indices, validation_indices = deterministic_split_indices(
        len(augmented), recipe.validation_size, recipe.split_seed
    )
    training_indices = _limit_indices(training_indices, train_limit, recipe.split_seed + 11)
    validation_indices = _limit_indices(validation_indices, validation_limit, recipe.split_seed + 13)
    train_dataset = torch.utils.data.Subset(augmented, training_indices)
    validation_dataset = torch.utils.data.Subset(evaluation, validation_indices)
    generator = torch.Generator().manual_seed(model_seed + 101)
    loader_options = {
        "batch_size": recipe.batch_size,
        "num_workers": recipe.num_workers,
        "pin_memory": False,
        "worker_init_fn": _seed_worker,
        "persistent_workers": recipe.num_workers > 0,
    }
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        generator=generator,
        **loader_options,
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        shuffle=False,
        **loader_options,
    )
    metadata = {
        "dataset": "CIFAR-10 train partition",
        "training_examples": len(train_dataset),
        "validation_examples": len(validation_dataset),
        "training_validation_overlap": len(set(training_indices).intersection(validation_indices)),
        "split_seed": recipe.split_seed,
        "test_partition_loaded": False,
        "normalization_mean": list(CIFAR10_MEAN),
        "normalization_std": list(CIFAR10_STD),
        "augmentation": "RandomCrop(32,padding=4) + RandomHorizontalFlip",
    }
    return train_loader, validation_loader, metadata


def dataset_archive_metadata(data_root: Path) -> dict:
    archive = data_root / "cifar-10-python.tar.gz"
    if not archive.exists():
        return {"archive": str(archive), "exists": False}
    digest = hashlib.sha256()
    with archive.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "archive": str(archive),
        "exists": True,
        "bytes": archive.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def calibration_metrics(logits, targets, bins: int = 15) -> dict[str, float]:
    import torch
    import torch.nn.functional as functional

    probabilities = logits.softmax(dim=1)
    predictions = probabilities.argmax(dim=1)
    confidence = probabilities.max(dim=1).values
    correct = predictions.eq(targets)
    ece = torch.zeros((), dtype=probabilities.dtype)
    boundaries = torch.linspace(0.0, 1.0, bins + 1)
    for lower, upper in zip(boundaries[:-1], boundaries[1:]):
        selected = confidence.gt(lower) & confidence.le(upper)
        if selected.any():
            ece += selected.float().mean() * (confidence[selected].mean() - correct[selected].float().mean()).abs()
    one_hot = functional.one_hot(targets, num_classes=probabilities.shape[1]).to(probabilities.dtype)
    brier = (probabilities - one_hot).square().sum(dim=1).mean()
    return {"ece": float(ece), "brier": float(brier)}


def evaluate(model, loader, device) -> dict[str, float | list[float]]:
    import torch
    import torch.nn.functional as functional

    model.eval()
    logits_parts = []
    target_parts = []
    started = time.perf_counter()
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=False)
            if bool(getattr(model, "_twistedmerge_channels_last", False)):
                images = images.contiguous(memory_format=torch.channels_last)
            targets = targets.to(device, non_blocking=False)
            logits_parts.append(model(images).detach().cpu())
            target_parts.append(targets.detach().cpu())
    synchronize(device)
    elapsed = time.perf_counter() - started
    logits = torch.cat(logits_parts)
    targets = torch.cat(target_parts)
    predictions = logits.argmax(dim=1)
    class_accuracy = []
    for class_index in range(logits.shape[1]):
        selected = targets.eq(class_index)
        class_accuracy.append(float(predictions[selected].eq(targets[selected]).float().mean()))
    calibration = calibration_metrics(logits, targets)
    return {
        "accuracy": float(predictions.eq(targets).float().mean()),
        "nll": float(functional.cross_entropy(logits, targets)),
        "ece": calibration["ece"],
        "brier": calibration["brier"],
        "worst_class_accuracy": min(class_accuracy),
        "class_accuracy": class_accuracy,
        "examples": int(targets.numel()),
        "evaluation_seconds": elapsed,
    }


def _learning_rate_multiplier(epoch: int, recipe: TrainingRecipe) -> float:
    if epoch < recipe.warmup_epochs:
        return float(epoch + 1) / max(recipe.warmup_epochs, 1)
    progress = (epoch - recipe.warmup_epochs) / max(recipe.epochs - recipe.warmup_epochs, 1)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def _atomic_torch_save(payload: dict, path: Path) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def train_resnet18(
    *,
    seed: int,
    recipe: TrainingRecipe,
    train_loader,
    validation_loader,
    device,
    checkpoint_dir: Path,
) -> tuple[object, list[dict], dict]:
    """Train or resume one model, saving a durable checkpoint each epoch."""

    import torch
    import torch.nn.functional as functional

    set_seed(seed)
    model = make_cifar_resnet18().to(device)
    if recipe.channels_last:
        model = model.to(memory_format=torch.channels_last)
        model._twistedmerge_channels_last = True
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=recipe.learning_rate,
        momentum=recipe.momentum,
        weight_decay=recipe.weight_decay,
        nesterov=True,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda epoch: _learning_rate_multiplier(epoch, recipe)
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    last_path = checkpoint_dir / "last.pt"
    best_path = checkpoint_dir / "best.pt"
    history: list[dict] = []
    start_epoch = 0
    best_accuracy = -math.inf
    if last_path.exists():
        payload = torch.load(last_path, map_location=device, weights_only=False)
        if payload.get("recipe") != recipe.to_dict() or int(payload.get("seed", -1)) != seed:
            raise ValueError(f"checkpoint recipe or seed mismatch: {last_path}")
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        start_epoch = int(payload["completed_epoch"]) + 1
        best_accuracy = float(payload["best_validation_accuracy"])
        history = list(payload.get("history", []))

    total_started = time.perf_counter()
    for epoch in range(start_epoch, recipe.epochs):
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_examples = 0
        synchronize(device)
        epoch_started = time.perf_counter()
        for images, targets in train_loader:
            images = images.to(device, non_blocking=False)
            if recipe.channels_last:
                images = images.contiguous(memory_format=torch.channels_last)
            targets = targets.to(device, non_blocking=False)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = functional.cross_entropy(logits, targets, label_smoothing=recipe.label_smoothing)
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.detach().cpu()) * int(targets.numel())
            train_correct += int(logits.argmax(dim=1).eq(targets).sum().detach().cpu())
            train_examples += int(targets.numel())
        scheduler.step()
        synchronize(device)
        training_seconds = time.perf_counter() - epoch_started
        validation = evaluate(model, validation_loader, device)
        row = {
            "seed": seed,
            "epoch": epoch + 1,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train_loss": train_loss_sum / max(train_examples, 1),
            "train_accuracy": train_correct / max(train_examples, 1),
            "validation_accuracy": validation["accuracy"],
            "validation_nll": validation["nll"],
            "validation_ece": validation["ece"],
            "validation_brier": validation["brier"],
            "validation_worst_class_accuracy": validation["worst_class_accuracy"],
            "training_seconds": training_seconds,
            "validation_seconds": validation["evaluation_seconds"],
            "peak_rss_mb": peak_rss_mb(),
        }
        history.append(row)
        if float(validation["accuracy"]) > best_accuracy:
            best_accuracy = float(validation["accuracy"])
            _atomic_torch_save(
                {
                    "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                    "seed": seed,
                    "completed_epoch": epoch,
                    "best_validation_accuracy": best_accuracy,
                    "recipe": recipe.to_dict(),
                },
                best_path,
            )
        _atomic_torch_save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "seed": seed,
                "completed_epoch": epoch,
                "best_validation_accuracy": best_accuracy,
                "recipe": recipe.to_dict(),
                "history": history,
            },
            last_path,
        )
        print(
            f"seed={seed} epoch={epoch + 1}/{recipe.epochs} "
            f"train={row['train_accuracy']:.4f} val={row['validation_accuracy']:.4f} "
            f"seconds={training_seconds:.2f}",
            flush=True,
        )

    best_payload = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best_payload["model"])
    final_validation = evaluate(model, validation_loader, device)
    resources = {
        "seed": seed,
        "parameter_count": parameter_count(model),
        "training_seconds_this_invocation": time.perf_counter() - total_started,
        "recorded_training_seconds_all_epochs": sum(float(row["training_seconds"]) for row in history),
        "recorded_validation_seconds_all_epochs": sum(float(row["validation_seconds"]) for row in history),
        "peak_rss_mb": peak_rss_mb(),
        "checkpoint_bytes": best_path.stat().st_size,
        "inference_multiplier": 1.0,
        "branches": 1,
        "validation_evaluations": len(history),
        "test_evaluations": 0,
        "device": str(device),
        "best_checkpoint": str(best_path),
    }
    return model, history, {"validation": final_validation, "resources": resources}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_manifest(paths: Iterable[Path]) -> list[dict]:
    rows = []
    for path in sorted(set(paths)):
        rows.append({"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return rows
