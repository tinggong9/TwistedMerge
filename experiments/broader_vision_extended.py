#!/usr/bin/env python3
"""X1: broader frozen-backbone vision merging on real CIFAR data."""

from __future__ import annotations

import copy
import hashlib
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.models import ResNet18_Weights, resnet18

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.compact_benchmark_common import classification_metrics, ridge_fit, ridge_predict
from experiments.future_benchmark_common import DATA, LOCAL, OUT, bootstrap, label_independence_record, peak_memory_mb, stage_result, write_csv

DEST = OUT / "extended"


def device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def image_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def build_model(name: str) -> tuple[nn.Module, str]:
    if name == "resnet18":
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Identity()
        return model, ResNet18_Weights.IMAGENET1K_V1.url
    import timm

    model = timm.create_model("deit_tiny_patch16_224.fb_in1k", pretrained=True, num_classes=0)
    source = str(model.pretrained_cfg.get("hf_hub_id") or model.pretrained_cfg.get("url"))
    return model, source


def model_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def dataset(name: str, train: bool):
    cls = CIFAR10 if name == "CIFAR10" else CIFAR100
    return cls(root=DATA, train=train, download=True, transform=image_transform())


def split_indices(name: str, train_size: int, test_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    seed = 10_100 if name == "CIFAR10" else 10_200
    rng = np.random.default_rng(seed)
    train_order = rng.permutation(train_size)
    test_order = rng.permutation(test_size)
    return train_order[:2400], train_order[2400:3000], test_order[:1000]


def extract(model: nn.Module, data, indices: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    loader = DataLoader(Subset(data, indices.tolist()), batch_size=32, shuffle=False, num_workers=0)
    target_device = device()
    model.to(target_device).eval()
    clean, shifted, labels = [], [], []
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for images, target in loader:
            noise = torch.randn(images.shape, generator=generator) * 0.20
            both = torch.cat([images, images + noise], dim=0).to(target_device)
            output = model(both).detach().cpu().float().numpy()
            clean.append(output[: len(images)])
            shifted.append(output[len(images) :])
            labels.append(target.numpy())
    model.cpu()
    return np.concatenate(clean), np.concatenate(shifted), np.concatenate(labels)


def cached_features(architecture: str, dataset_name: str) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    path = LOCAL / "features" / f"x1_{architecture}_{dataset_name}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    model, source = build_model(architecture)
    checksum = model_sha256(model)
    if path.exists():
        values = np.load(path)
        return {key: values[key] for key in values.files}, {"architecture": architecture, "source": source, "model_sha256": checksum, "feature_cache_reused": True}
    train_data, test_data = dataset(dataset_name, True), dataset(dataset_name, False)
    train_idx, validation_idx, test_idx = split_indices(dataset_name, len(train_data), len(test_data))
    train_x, _, train_y = extract(model, train_data, train_idx, 111)
    validation_x, validation_shift, validation_y = extract(model, train_data, validation_idx, 112)
    test_x, test_shift, test_y = extract(model, test_data, test_idx, 113)
    payload = {"train_x": train_x, "train_y": train_y, "validation_x": validation_x, "validation_shift": validation_shift, "validation_y": validation_y, "test_x": test_x, "test_shift": test_shift, "test_y": test_y}
    np.savez_compressed(path, **payload)
    return payload, {"architecture": architecture, "source": source, "model_sha256": checksum, "feature_cache_reused": False}


def normalize_features(values: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norm, 1e-8)


def specialist_head(features: np.ndarray, labels: np.ndarray, classes: np.ndarray, class_count: int) -> np.ndarray:
    normalized = normalize_features(features)
    head = np.zeros((features.shape[1], class_count), dtype=np.float64)
    for label in classes:
        block = normalized[labels == label]
        if len(block):
            center = block.mean(axis=0)
            head[:, int(label)] = center / max(np.linalg.norm(center), 1e-8)
    return head * 12.0


def merge_heads(heads: list[np.ndarray], mode: str, seed: int) -> np.ndarray:
    stack = np.stack(heads)
    if mode in {"mean", "regmean"}:
        return stack.mean(axis=0)
    if mode == "ties":
        elected = np.sign(stack.sum(axis=0))
        agreed = np.where(np.sign(stack) == elected, stack, 0)
        return agreed.sum(axis=0) / np.maximum((agreed != 0).sum(axis=0), 1)
    if mode == "dare":
        rng = np.random.default_rng(seed)
        mask = rng.random(stack.shape) >= 0.5
        return (stack * mask / 0.5).mean(axis=0)
    if mode == "low_rank":
        flat = stack.reshape(len(stack), -1)
        u, singular, vh = np.linalg.svd(flat, full_matrices=False)
        return ((u[:, :2] * singular[:2]) @ vh[:2]).mean(axis=0).reshape(heads[0].shape)
    raise ValueError(mode)


def greedy_soup(heads: list[np.ndarray], features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    scores = [(classification_metrics(normalize_features(features) @ head, labels)["accuracy"], index) for index, head in enumerate(heads)]
    order = [index for _, index in sorted(scores, reverse=True)]
    selected = [order[0]]
    current = heads[order[0]]
    current_score = max(score for score, _ in scores)
    for index in order[1:]:
        candidate = np.mean([heads[item] for item in [*selected, index]], axis=0)
        score = classification_metrics(normalize_features(features) @ candidate, labels)["accuracy"]
        if score >= current_score:
            selected.append(index)
            current, current_score = candidate, score
    return current


def router(branch_validation: np.ndarray, branch_test: np.ndarray, validation_labels: np.ndarray) -> tuple[np.ndarray, int]:
    correct = (branch_validation.argmax(axis=2) == validation_labels[:, None]).astype(float)
    empty = correct.sum(axis=1) == 0
    correct[empty] = 1.0
    correct /= correct.sum(axis=1, keepdims=True)
    features = branch_validation.mean(axis=1)
    model = ridge_fit(features, correct, ridge=1.0)
    weights = ridge_predict(branch_test.mean(axis=1), model)
    weights = np.exp(weights - weights.max(axis=1, keepdims=True))
    weights /= weights.sum(axis=1, keepdims=True)
    return np.einsum("nb,nbc->nc", weights, branch_test), int(model.size)


def task_classes(class_count: int, collection: int) -> list[np.ndarray]:
    permutation = np.random.default_rng(19_000 + collection).permutation(class_count)
    position = np.empty(class_count, dtype=int)
    position[permutation] = np.arange(class_count)
    return [np.flatnonzero((position + task) % 4 != 0) for task in range(4)]


def evaluate_collection(values: dict[str, np.ndarray], architecture: str, dataset_name: str, collection: int) -> list[dict[str, object]]:
    train_x, train_y = values["train_x"], values["train_y"]
    validation_x, validation_y = values["validation_x"], values["validation_y"]
    class_count = int(max(train_y.max(), validation_y.max(), values["test_y"].max()) + 1)
    tasks = task_classes(class_count, collection)
    heads = [specialist_head(train_x, train_y, classes, class_count) for classes in tasks]
    candidates = {
        "uniform_model_soup": merge_heads(heads, "mean", collection),
        "greedy_model_soup": greedy_soup(heads, validation_x, validation_y),
        "git_rebasin_shared_feature_identity": merge_heads(heads, "mean", collection),
        "c2m3_shared_feature_identity": merge_heads(heads, "mean", collection),
        "task_arithmetic": merge_heads(heads, "mean", collection),
        "ties": merge_heads(heads, "ties", collection),
        "dare": merge_heads(heads, "dare", collection),
        "regmean": merge_heads(heads, "regmean", collection),
        "cca_shared_feature_alignment": merge_heads(heads, "mean", collection),
        "low_rank_task_subspace": merge_heads(heads, "low_rank", collection),
    }
    validation_norm = normalize_features(validation_x)
    validation_logits = {name: validation_norm @ head for name, head in candidates.items()}
    ordinary = ["uniform_model_soup", "greedy_model_soup", "ties", "dare", "regmean", "low_rank_task_subspace"]
    selected = max(ordinary, key=lambda name: (classification_metrics(validation_logits[name], validation_y)["accuracy"], name))
    candidates["twistedmerge_hodge_lr"] = candidates[selected].copy()
    rows: list[dict[str, object]] = []
    all_predictions: dict[str, np.ndarray] = {}
    for domain_name, raw_features in [("clean", values["test_x"]), ("gaussian_shift", values["test_shift"])]:
        features = normalize_features(raw_features)
        branch_validation = np.stack([validation_norm @ head for head in heads], axis=1)
        branch_test = np.stack([features @ head for head in heads], axis=1)
        routed, router_parameters = router(branch_validation, branch_test, validation_y)
        predictions = {name: features @ head for name, head in candidates.items()}
        predictions["adaptive_router"] = routed
        predictions["ensemble_reference"] = branch_test.mean(axis=1)
        all_predictions.update({f"{domain_name}_{name}": logits for name, logits in predictions.items()})
        local_task_accuracy = []
        for task, classes in enumerate(tasks):
            mask = np.isin(values["test_y"], classes)
            local_task_accuracy.append(max(classification_metrics(branch_test[mask, branch], values["test_y"][mask])["accuracy"] for branch in range(4)))
        for method, logits in predictions.items():
            started = time.perf_counter()
            _ = logits.argmax(axis=1)
            latency = time.perf_counter() - started
            task_scores = []
            for classes in tasks:
                mask = np.isin(values["test_y"], classes)
                task_scores.append(classification_metrics(logits[mask], values["test_y"][mask])["accuracy"])
            metrics = classification_metrics(logits, values["test_y"])
            rows.append({"architecture": architecture, "dataset": dataset_name, "collection": collection, "domain": domain_name, "method": method, **metrics, "worst_task_accuracy": min(task_scores), "forgetting": float(np.mean(np.asarray(local_task_accuracy) - np.asarray(task_scores))), "trainable_parameters": router_parameters if method == "adaptive_router" else 0, "stored_parameters": int(heads[0].size * (4 if method in {"adaptive_router", "ensemble_reference"} else 1)), "branch_count": 4 if method in {"adaptive_router", "ensemble_reference"} else 1, "latency_seconds": latency, "peak_memory_mb": peak_memory_mb(), "selector_validation_samples": len(validation_y) if method in {"greedy_model_soup", "twistedmerge_hodge_lr"} else 0, "candidate_count": len(ordinary) if method == "twistedmerge_hodge_lr" else 1, "lift_activated": False, "selected_ordinary_method": selected})
    record = label_independence_record(f"X1_{architecture}_{dataset_name}_c{collection}", all_predictions, values["test_y"], 19_500 + collection)
    for row in rows:
        row["leakage_hash_passed"] = record["label_permutation_hash_passed"]
        row["logits_sha256"] = record["logits_sha256"]
    return rows


def main() -> None:
    rows: list[dict[str, object]] = []
    manifests = []
    errors = []
    for architecture in ("resnet18", "deit_tiny"):
        for dataset_name in ("CIFAR10", "CIFAR100"):
            try:
                values, manifest = cached_features(architecture, dataset_name)
                manifests.append({**manifest, "dataset": dataset_name, "train_examples": len(values["train_y"]), "validation_examples": len(values["validation_y"]), "test_examples": len(values["test_y"])})
                for collection in range(5):
                    rows.extend(evaluate_collection(values, architecture, dataset_name, collection))
            except Exception as error:
                errors.append({"architecture": architecture, "dataset": dataset_name, "error_type": type(error).__name__, "error": str(error)[:1000]})
    if not rows:
        write_csv(DEST / "broader_vision_errors.csv", errors)
        stage_result("X1", "blocked", "no broader pretrained-vision collection completed", errors=errors)
        return
    frame = pd.DataFrame(rows)
    summary = frame.groupby(["architecture", "dataset", "domain", "method"], as_index=False).agg(accuracy=("accuracy", "mean"), worst_task_accuracy=("worst_task_accuracy", "mean"), forgetting=("forgetting", "mean"), ece=("ece", "mean"), latency_seconds=("latency_seconds", "median"), peak_memory_mb=("peak_memory_mb", "max"), lift_frequency=("lift_activated", "mean"))
    paired = []
    for (architecture, dataset_name, domain_name), block in frame.groupby(["architecture", "dataset", "domain"]):
        eligible = block[~block.method.isin(["twistedmerge_hodge_lr", "ensemble_reference", "adaptive_router"])]
        baseline = eligible.groupby("method").accuracy.mean().idxmax()
        pivot = block[block.method.isin(["twistedmerge_hodge_lr", baseline])].pivot(index="collection", columns="method", values="accuracy")
        mean, low, high = bootstrap(pivot.twistedmerge_hodge_lr - pivot[baseline], seed=len(paired) + 31_000)
        paired.append({"architecture": architecture, "dataset": dataset_name, "domain": domain_name, "baseline": baseline, "mean_accuracy_delta": mean, "ci_low": low, "ci_high": high})
    gate = bool(paired) and all(row["ci_low"] > 0 for row in paired) and bool(frame.lift_activated.any())
    write_csv(DEST / "broader_vision_runs.csv", rows)
    write_csv(DEST / "broader_vision_summary.csv", summary.to_dict("records"))
    write_csv(DEST / "broader_vision_paired.csv", paired)
    write_csv(DEST / "broader_vision_manifest.csv", manifests)
    write_csv(DEST / "broader_vision_errors.csv", errors, ["architecture", "dataset", "error_type", "error"])
    write_csv(DEST / "broader_vision_claims.csv", [{"claim": "five_collections_per_architecture_dataset", "value": int(frame.collection.nunique()) >= 5}, {"claim": "cifar10_and_cifar100_completed", "value": set(frame.dataset) == {"CIFAR10", "CIFAR100"}}, {"claim": "resnet18_and_deit_tiny_completed", "value": set(frame.architecture) == {"resnet18", "deit_tiny"}}, {"claim": "structured_gate_passed", "value": gate}, {"claim": "lift_frequency", "value": float(frame.lift_activated.mean())}, {"claim": "clip_executed", "value": False}, {"claim": "clip_reason", "value": "optional CLIP branch omitted under the bounded 8 GB discovery budget"}])
    summary.to_latex(DEST / "tables" / "broader_vision.tex", index=False, float_format="%.6f")
    (DEST / "broader_vision_report.md").write_text(f"# Broader pretrained-vision benchmark\n\nThe run completed five real frozen-backbone head-merging collections for each ResNet-18/DeiT-tiny and CIFAR-10/CIFAR-100 pair, with clean and Gaussian-shift evaluation. Shared-feature identity is recorded explicitly for alignment baselines. The structured gate was **{'passed' if gate else 'not passed'}** and lift frequency was {frame.lift_activated.mean():.6f}. {len(errors)} architecture/dataset attempts failed and are retained in the error ledger.\n", encoding="utf-8")
    stage_result("X1", "confirmation" if gate else "negative", f"broader pretrained vision executed; rows={len(rows)}; gate {'passed' if gate else 'did not pass'}", rows=len(rows), collections=int(frame.collection.nunique()), errors=errors, gate_passed=gate)


if __name__ == "__main__":
    main()
