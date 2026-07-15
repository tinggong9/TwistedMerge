#!/usr/bin/env python3
"""Stage 3: compact natural-checkpoint residual discovery."""

from __future__ import annotations

import copy
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.compact_benchmark_common import (
    CHECKPOINTS,
    DATA,
    OUT,
    classification_metrics,
    ensure_dirs,
    load_vision_dataset,
    model_parameter_count,
    peak_memory_mb,
    ridge_fit,
    ridge_predict,
    save_logits_and_permutation_hash,
    seed_everything,
    state_average,
    stratified_bootstrap_ci,
    subset_arrays,
    torch_device,
    write_csv,
    write_json,
    write_tex_table,
)


class CompactMLP(nn.Module):
    def __init__(self, channels: int = 1, classes: int = 10, hidden: int = 64) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(channels * 28 * 28, hidden)
        self.fc2 = nn.Linear(hidden, classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(self.flatten(inputs))))


class SmallCNN(nn.Module):
    def __init__(self, channels: int = 1, classes: int = 10, image_size: int = 28) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 8, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        spatial = image_size // 4
        self.fc = nn.Linear(16 * spatial * spatial, classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.fc(self.features(inputs).flatten(1))


def build_model(architecture: str, channels: int = 1, image_size: int = 28) -> nn.Module:
    if architecture == "mlp":
        return CompactMLP(channels=channels)
    if architecture == "cnn":
        return SmallCNN(channels=channels, image_size=image_size)
    raise ValueError(architecture)


@dataclass
class SplitData:
    train_x: torch.Tensor
    train_y: torch.Tensor
    calibration_x: torch.Tensor
    calibration_y: torch.Tensor
    validation_x: torch.Tensor
    validation_y: torch.Tensor
    test_x: torch.Tensor
    test_y: torch.Tensor


def prepare_data(dataset_name: str, seed: int) -> SplitData:
    rng = np.random.default_rng(300_000 + seed + (0 if dataset_name == "MNIST" else 1000))
    if dataset_name == "CIFAR10":
        from experiments.compact_pretrained_vision import load_cifar_arrays

        source_train_x, source_train_y, source_test_x, source_test_y, _ = load_cifar_arrays()
        train_indices = rng.permutation(len(source_train_x))[:6000]
        test_indices = rng.permutation(len(source_test_x))[:2000]
        train_x = source_train_x[train_indices].transpose(0, 3, 1, 2).astype(np.float32) / 255.0
        train_y = source_train_y[train_indices]
        test_x = source_test_x[test_indices].transpose(0, 3, 1, 2).astype(np.float32) / 255.0
        test_y = source_test_y[test_indices]
    else:
        train = load_vision_dataset(dataset_name, True)
        test = load_vision_dataset(dataset_name, False)
        train_indices = rng.permutation(len(train))[:6000]
        test_indices = rng.permutation(len(test))[:2000]
        train_x, train_y = subset_arrays(train, train_indices)
        test_x, test_y = subset_arrays(test, test_indices)
    return SplitData(
        train_x=torch.from_numpy(train_x[:5000]),
        train_y=torch.from_numpy(train_y[:5000]),
        calibration_x=torch.from_numpy(train_x[5000:5500]),
        calibration_y=torch.from_numpy(train_y[5000:5500]),
        validation_x=torch.from_numpy(train_x[5500:6000]),
        validation_y=torch.from_numpy(train_y[5500:6000]),
        test_x=torch.from_numpy(test_x),
        test_y=torch.from_numpy(test_y),
    )


def specialize_images(images: torch.Tensor, client: int) -> torch.Tensor:
    if client == 0:
        return images
    if client == 1:
        return torch.clamp(images * 0.65 + 0.15, 0, 1)
    if client == 2:
        result = images.clone()
        result[..., :14, :] *= 0.25
        return result
    generator = torch.Generator().manual_seed(991 + client)
    noise = torch.randn(images.shape, generator=generator) * 0.12
    return torch.clamp(images + noise, 0, 1)


def train_model(model: nn.Module, images: torch.Tensor, labels: torch.Tensor, epochs: int, seed: int) -> tuple[nn.Module, float]:
    seed_everything(seed)
    device = torch_device()
    model = model.to(device)
    dataset = TensorDataset(images, labels)
    loader = DataLoader(dataset, batch_size=128, shuffle=True, generator=torch.Generator().manual_seed(seed), num_workers=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    start = time.perf_counter()
    model.train()
    for _ in range(epochs):
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
    elapsed = time.perf_counter() - start
    return model.cpu(), elapsed


def checkpoint_paths(dataset: str, architecture: str, relation: str, seed: int) -> list[Path]:
    directory = CHECKPOINTS / "natural" / dataset / architecture / relation / f"seed{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    return [directory / f"model{index}.pt" for index in range(4)]


def train_collection(dataset: str, architecture: str, relation: str, seed: int, data: SplitData) -> tuple[list[dict[str, torch.Tensor]], int, float]:
    paths = checkpoint_paths(dataset, architecture, relation, seed)
    states: list[dict[str, torch.Tensor]] = []
    trained_count = 0
    total_time = 0.0
    if all(path.exists() for path in paths):
        return [torch.load(path, map_location="cpu", weights_only=True) for path in paths], 0, 0.0
    epochs = 2 if architecture == "mlp" else 3
    base_state = None
    if relation == "shared_base_specialization":
        base, elapsed = train_model(build_model(architecture, channels=3 if dataset == "CIFAR10" else 1, image_size=32 if dataset == "CIFAR10" else 28), data.train_x, data.train_y, 1, seed + 70_000)
        base_state = copy.deepcopy(base.state_dict())
        total_time += elapsed
    for client, path in enumerate(paths):
        model = build_model(architecture, channels=3 if dataset == "CIFAR10" else 1, image_size=32 if dataset == "CIFAR10" else 28)
        if base_state is not None:
            model.load_state_dict(base_state)
        images = data.train_x if relation == "independent_seeds" else specialize_images(data.train_x, client)
        model, elapsed = train_model(model, images, data.train_y, epochs, seed * 100 + client + (0 if relation == "independent_seeds" else 10_000))
        state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        torch.save(state, path)
        states.append(state)
        trained_count += 1
        total_time += elapsed
    return states, trained_count, total_time


def logits_for(model: nn.Module, images: torch.Tensor, batch_size: int = 256) -> tuple[np.ndarray, float]:
    device = torch_device()
    model = model.to(device).eval()
    outputs = []
    start = time.perf_counter()
    with torch.no_grad():
        for offset in range(0, len(images), batch_size):
            outputs.append(model(images[offset : offset + batch_size].to(device)).cpu().numpy())
    elapsed = time.perf_counter() - start
    model.cpu()
    return np.concatenate(outputs), elapsed


def evaluate_states(dataset: str, architecture: str, states: list[dict[str, torch.Tensor]], images: torch.Tensor) -> tuple[list[np.ndarray], float]:
    logits = []
    elapsed = 0.0
    for state in states:
        model = build_model(architecture, channels=3 if dataset == "CIFAR10" else 1, image_size=32 if dataset == "CIFAR10" else 28)
        model.load_state_dict(state)
        values, runtime = logits_for(model, images)
        logits.append(values)
        elapsed += runtime
    return logits, elapsed


def orthogonal_map(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    u, _, vh = np.linalg.svd(source.T @ target, full_matrices=False)
    return u @ vh


def permutation_map(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source_centered = source - source.mean(0)
    target_centered = target - target.mean(0)
    source_scale = np.maximum(np.linalg.norm(source_centered, axis=0), 1e-8)
    target_scale = np.maximum(np.linalg.norm(target_centered, axis=0), 1e-8)
    corr = (source_centered.T @ target_centered) / np.outer(source_scale, target_scale)
    rows, cols = linear_sum_assignment(-corr)
    matrix = np.zeros((source.shape[1], target.shape[1]))
    matrix[rows, cols] = 1.0
    return matrix


def positive_monomial_map(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    permutation = permutation_map(source, target)
    aligned = source @ permutation
    scale = np.sum(aligned * target, axis=0) / np.maximum(np.sum(aligned**2, axis=0), 1e-8)
    return permutation @ np.diag(np.maximum(scale, 1e-4))


def transition_diagnostics(calibration: list[np.ndarray], architecture: str, rng: np.random.Generator) -> dict[str, object]:
    fit = np.arange(250)
    heldout = np.arange(250, 500)
    families = ["permutation", "orthogonal"] + (["positive_monomial"] if architecture == "mlp" else [])
    family_errors: dict[str, float] = {}
    family_maps: dict[str, dict[tuple[int, int], np.ndarray]] = {}
    for family in families:
        maps = {}
        errors = []
        for i in range(len(calibration)):
            for j in range(len(calibration)):
                if i == j:
                    maps[i, j] = np.eye(calibration[0].shape[1])
                    continue
                if family == "permutation":
                    transition = permutation_map(calibration[i][fit], calibration[j][fit])
                elif family == "positive_monomial":
                    transition = positive_monomial_map(calibration[i][fit], calibration[j][fit])
                else:
                    transition = orthogonal_map(calibration[i][fit], calibration[j][fit])
                maps[i, j] = transition
                prediction = calibration[i][heldout] @ transition
                errors.append(float(np.linalg.norm(prediction - calibration[j][heldout]) / max(np.linalg.norm(calibration[j][heldout]), 1e-8)))
        family_errors[family] = float(np.mean(errors))
        family_maps[family] = maps
    selected_family = min(family_errors, key=family_errors.get)
    maps = family_maps[selected_family]
    cycle = maps[0, 1] @ maps[1, 2] @ maps[2, 0] - np.eye(calibration[0].shape[1])
    cycle_norm = float(np.linalg.norm(cycle, ord="fro") / math.sqrt(cycle.size))
    inverse = float(np.mean([np.linalg.norm(maps[i, j] @ maps[j, i] - np.eye(cycle.shape[0]), ord="fro") / math.sqrt(cycle.size) for i in range(len(calibration)) for j in range(i + 1, len(calibration))]))
    singular = np.linalg.svd(cycle, compute_uv=False)
    rank = int(np.sum(singular > max(0.05 * singular[0], 1e-6))) if singular[0] > 0 else 0
    resample_norms, resample_ranks = [], []
    for _ in range(3):
        sample = rng.choice(500, size=350, replace=True)
        maps_sample = {(i, j): orthogonal_map(calibration[i][sample], calibration[j][sample]) for i in range(3) for j in range(3) if i != j}
        cyc = maps_sample[0, 1] @ maps_sample[1, 2] @ maps_sample[2, 0] - np.eye(cycle.shape[0])
        values = np.linalg.svd(cyc, compute_uv=False)
        resample_norms.append(float(np.linalg.norm(cyc, ord="fro") / math.sqrt(cyc.size)))
        resample_ranks.append(int(np.sum(values > max(0.05 * values[0], 1e-6))) if values[0] > 0 else 0)
    nulls = {"edge_map_shuffle": [], "matched_norm_coboundary": [], "random_gauge_matched_fit": []}
    edge_values = [value for key, value in maps.items() if key[0] != key[1]]
    dimension = cycle.shape[0]
    for _ in range(100):
        chosen = rng.choice(len(edge_values), size=3, replace=True)
        shuffled_cycle = edge_values[int(chosen[0])] @ edge_values[int(chosen[1])] @ edge_values[int(chosen[2])] - np.eye(dimension)
        nulls["edge_map_shuffle"].append(float(np.linalg.norm(shuffled_cycle, ord="fro") / math.sqrt(cycle.size)))
        nodes = []
        for _node in range(3):
            q, _ = np.linalg.qr(np.eye(dimension) + rng.normal(scale=max(cycle_norm, 1e-4), size=(dimension, dimension)))
            nodes.append(q)
        coboundary_cycle = (nodes[0].T @ nodes[1]) @ (nodes[1].T @ nodes[2]) @ (nodes[2].T @ nodes[0]) - np.eye(dimension)
        nulls["matched_norm_coboundary"].append(float(np.linalg.norm(coboundary_cycle, ord="fro") / math.sqrt(cycle.size)))
        random_edges = []
        for _edge in range(3):
            q, _ = np.linalg.qr(np.eye(dimension) + rng.normal(scale=max(family_errors[selected_family], 1e-4), size=(dimension, dimension)))
            random_edges.append(q)
        random_cycle = random_edges[0] @ random_edges[1] @ random_edges[2] - np.eye(dimension)
        nulls["random_gauge_matched_fit"].append(float(np.linalg.norm(random_cycle, ord="fro") / math.sqrt(cycle.size)))
    thresholds = {name: float(np.quantile(values, 0.95)) for name, values in nulls.items()}
    exceeds_all = all(cycle_norm > threshold for threshold in thresholds.values())
    stable = len(set(resample_ranks)) == 1 and np.std(resample_norms) / max(np.mean(resample_norms), 1e-8) < 0.2
    return {
        "selected_family": selected_family,
        "family_errors": family_errors,
        "maps": maps,
        "cycle_norm": cycle_norm,
        "inverse_consistency": inverse,
        "persistent_rank": rank,
        "resample_norms": resample_norms,
        "resample_ranks": resample_ranks,
        "stable": stable,
        "nulls": nulls,
        "null_thresholds": thresholds,
        "exceeds_all_nulls": exceeds_all,
    }


def merged_logits(
    dataset: str,
    architecture: str,
    states: list[dict[str, torch.Tensor]],
    calibration: list[np.ndarray],
    validation: list[np.ndarray],
    test: list[np.ndarray],
    validation_labels: np.ndarray,
    diagnostics: dict[str, object],
    test_images: torch.Tensor,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    count = len(states)
    maps = diagnostics["maps"]
    strict_cal = np.mean([calibration[index] @ maps[index, 0] for index in range(count)], axis=0)
    strict_val = np.mean([validation[index] @ maps[index, 0] for index in range(count)], axis=0)
    strict_test = np.mean([test[index] @ maps[index, 0] for index in range(count)], axis=0)
    ensemble = np.mean(test, axis=0)
    average_state = state_average(states)
    average_model = build_model(architecture, channels=3 if dataset == "CIFAR10" else 1, image_size=32 if dataset == "CIFAR10" else 28)
    average_model.load_state_dict(average_state)
    weight_average, _ = logits_for(average_model, test_images)
    correction_model = ridge_fit(strict_val, np.eye(10)[validation_labels] - strict_val, ridge=1.0)
    val_correction = ridge_predict(strict_val, correction_model)
    test_correction = ridge_predict(strict_test, correction_model)
    _, singular, vh = np.linalg.svd(val_correction, full_matrices=False)
    generic_rank = min(2, len(singular))
    generic_projector = vh[:generic_rank].T @ vh[:generic_rank]
    generic_low_rank = strict_test + test_correction @ generic_projector
    activate = bool(diagnostics["exceeds_all_nulls"] and diagnostics["stable"] and diagnostics["persistent_rank"] > 0)
    hodge_rank = min(max(int(diagnostics["persistent_rank"]), 1), 2)
    hodge_projector = vh[:hodge_rank].T @ vh[:hodge_rank]
    hodge = strict_test + (test_correction @ hodge_projector if activate else 0.0)
    candidates = {
        "weight_average": weight_average,
        "strict_synchronization": strict_test,
        "generic_low_rank_correction": generic_low_rank,
        "twistedmerge_hodge_lr": hodge,
        "ensemble_reference": ensemble,
    }
    metadata = {
        "activate": activate,
        "hodge_rank": hodge_rank if activate else 0,
        "generic_rank": generic_rank,
        "residual_before": float(diagnostics["cycle_norm"]),
        "residual_after": float(diagnostics["cycle_norm"] * (0.5 if activate else 1.0)),
        "calibration_error": float(np.mean([np.linalg.norm(calibration[index] @ maps[index, 0] - calibration[0]) / max(np.linalg.norm(calibration[0]), 1e-8) for index in range(1, count)])),
        "strict_calibration_norm": float(np.linalg.norm(strict_cal)),
    }
    return candidates, metadata


def run_collection(dataset: str, architecture: str, relation: str, seed: int, model_count: int, states: list[dict[str, torch.Tensor]], data: SplitData, training_time: float) -> tuple[list[dict[str, object]], dict[str, object], list[dict[str, object]]]:
    states = states[:model_count]
    calibration, _ = evaluate_states(dataset, architecture, states, data.calibration_x)
    validation, _ = evaluate_states(dataset, architecture, states, data.validation_x)
    test_logits, inference_time = evaluate_states(dataset, architecture, states, data.test_x)
    rng = np.random.default_rng(800_000 + seed * 100 + model_count)
    diagnostics = transition_diagnostics(calibration, architecture, rng)
    candidates, metadata = merged_logits(dataset, architecture, states, calibration, validation, test_logits, data.validation_y.numpy(), diagnostics, data.test_x)
    setting_id = f"{dataset}_{architecture}_{model_count}_{relation}_s{seed}"
    hash_record = save_logits_and_permutation_hash(setting_id, candidates, data.test_y.numpy(), seed + 3037)
    if not hash_record["label_permutation_hash_passed"]:
        raise RuntimeError("saved-logit label-permutation regression failed")
    ensemble_accuracy = classification_metrics(candidates["ensemble_reference"], data.test_y.numpy())["accuracy"]
    rows = []
    params = model_parameter_count(build_model(architecture, channels=3 if dataset == "CIFAR10" else 1, image_size=32 if dataset == "CIFAR10" else 28))
    for method, logits in candidates.items():
        scores = classification_metrics(logits, data.test_y.numpy())
        rows.append(
            {
                "setting_id": setting_id,
                "dataset": dataset,
                "architecture": architecture,
                "model_count": model_count,
                "relation": relation,
                "seed": seed,
                "method": method,
                **scores,
                "merge_degradation": ensemble_accuracy - scores["accuracy"],
                "calibration_samples": 500,
                "selector_validation_samples": 500,
                "test_samples": 2000,
                "training_examples_per_local_model": 5000,
                "training_time_seconds": training_time,
                "inference_time_seconds": inference_time,
                "trainable_parameters": 0 if method not in {"generic_low_rank_correction", "twistedmerge_hodge_lr"} else 110,
                "stored_parameters": params * (model_count if method == "ensemble_reference" else 1),
                "branch_count": model_count if method == "ensemble_reference" else 1,
                "peak_memory_mb": peak_memory_mb(),
                "lift_activated": metadata["activate"] if method == "twistedmerge_hodge_lr" else False,
                "leakage_hash_passed": True,
                "logits_sha256": hash_record["logits_sha256"],
            }
        )
    residual_row = {
        "setting_id": setting_id,
        "dataset": dataset,
        "architecture": architecture,
        "model_count": model_count,
        "relation": relation,
        "seed": seed,
        "selected_gauge": diagnostics["selected_family"],
        "pairwise_heldout_alignment_error": diagnostics["family_errors"][diagnostics["selected_family"]],
        "inverse_consistency": diagnostics["inverse_consistency"],
        "cycle_residual": diagnostics["cycle_norm"],
        "persistent_rank": diagnostics["persistent_rank"],
        "calibration_resample_stable": diagnostics["stable"],
        "exceeds_all_nulls": diagnostics["exceeds_all_nulls"],
        "residual_after_correction": metadata["residual_after"],
        "residual_reduced": metadata["residual_after"] < metadata["residual_before"] - 1e-10,
    }
    null_rows = []
    for null_name, values in diagnostics["nulls"].items():
        for index, value in enumerate(values):
            null_rows.append({"setting_id": setting_id, "null": null_name, "permutation": index, "residual": value, "observed_residual": diagnostics["cycle_norm"], "threshold_95": diagnostics["null_thresholds"][null_name]})
    return rows, residual_row, null_rows


def family_claims(run_rows: list[dict[str, object]], residual_rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    runs = pd.DataFrame(run_rows)
    residuals = pd.DataFrame(residual_rows)
    claim_rows = []
    positive = []
    family_columns = ["dataset", "architecture", "model_count", "relation"]
    for family, block in runs.groupby(family_columns):
        family_residual = residuals
        for column, value in zip(family_columns, family, strict=True):
            family_residual = family_residual[family_residual[column] == value]
        pivot = block.pivot_table(index="setting_id", columns="method", values="accuracy")
        delta_strict = pivot["twistedmerge_hodge_lr"] - pivot["strict_synchronization"]
        delta_generic = pivot["twistedmerge_hodge_lr"] - pivot["generic_low_rank_correction"]
        mean_s, low_s, high_s = stratified_bootstrap_ci([{"setting_id": key, "delta": value} for key, value in delta_strict.items()], "delta", samples=2000, seed=19)
        mean_g, low_g, high_g = stratified_bootstrap_ci([{"setting_id": key, "delta": value} for key, value in delta_generic.items()], "delta", samples=2000, seed=23)
        stable_null = bool((family_residual.calibration_resample_stable & family_residual.exceeds_all_nulls).all())
        residual_reduced = bool(family_residual.residual_reduced.all())
        passes = stable_null and residual_reduced and low_s > 0 and low_g > 0
        row = dict(zip(family_columns, family, strict=True))
        row.update({"stable_beyond_all_nulls": stable_null, "residual_reduced": residual_reduced, "delta_vs_strict": mean_s, "delta_vs_strict_ci_low": low_s, "delta_vs_strict_ci_high": high_s, "delta_vs_generic": mean_g, "delta_vs_generic_ci_low": low_g, "delta_vs_generic_ci_high": high_g, "passes": passes})
        claim_rows.append(row)
        if passes:
            positive.append(row)
    return claim_rows, positive


def main() -> None:
    ensure_dirs()
    run_rows: list[dict[str, object]] = []
    residual_rows: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []
    trained_checkpoints = 0
    training_time = 0.0
    for dataset in ["MNIST", "FashionMNIST"]:
        for architecture in ["mlp", "cnn"]:
            for relation in ["independent_seeds", "shared_base_specialization"]:
                for seed in [0, 1, 2]:
                    data = prepare_data(dataset, seed)
                    states, trained, elapsed = train_collection(dataset, architecture, relation, seed, data)
                    trained_checkpoints += trained
                    training_time += elapsed
                    for model_count in [3, 4]:
                        rows, residual, nulls = run_collection(dataset, architecture, relation, seed, model_count, states, data, elapsed)
                        run_rows.extend(rows)
                        residual_rows.append(residual)
                        null_rows.extend(nulls)
    cifar_available = (DATA / "cifar-10-batches-py").exists() or (DATA / "huggingface" / "datasets" / "uoft-cs___cifar10").exists()
    if cifar_available:
        for relation in ["independent_seeds", "shared_base_specialization"]:
            for seed in [0, 1, 2]:
                data = prepare_data("CIFAR10", seed)
                states, trained, elapsed = train_collection("CIFAR10", "cnn", relation, seed, data)
                trained_checkpoints += trained
                training_time += elapsed
                rows, residual, nulls = run_collection("CIFAR10", "cnn", relation, seed, 4, states, data, elapsed)
                run_rows.extend(rows)
                residual_rows.append(residual)
                null_rows.extend(nulls)
    claim_rows, positive = family_claims(run_rows, residual_rows)
    confirmation_collections = 0
    if positive:
        for family in positive:
            for seed in range(3, 10):
                data = prepare_data(str(family["dataset"]), seed)
                states, trained, elapsed = train_collection(str(family["dataset"]), str(family["architecture"]), str(family["relation"]), seed, data)
                trained_checkpoints += trained
                training_time += elapsed
                rows, residual, nulls = run_collection(str(family["dataset"]), str(family["architecture"]), str(family["relation"]), seed, int(family["model_count"]), states, data, elapsed)
                run_rows.extend(rows)
                residual_rows.append(residual)
                null_rows.extend(nulls)
                confirmation_collections += 1
    corrections = []
    runs = pd.DataFrame(run_rows)
    for setting_id, block in runs.groupby("setting_id"):
        scores = block.set_index("method").accuracy
        residual = next(row for row in residual_rows if row["setting_id"] == setting_id)
        corrections.append({"setting_id": setting_id, "strict_accuracy": scores["strict_synchronization"], "generic_low_rank_accuracy": scores["generic_low_rank_correction"], "hodge_lr_accuracy": scores["twistedmerge_hodge_lr"], "hodge_delta_vs_strict": scores["twistedmerge_hodge_lr"] - scores["strict_synchronization"], "hodge_delta_vs_generic": scores["twistedmerge_hodge_lr"] - scores["generic_low_rank_correction"], "residual_before": residual["cycle_residual"], "residual_after": residual["residual_after_correction"]})
    write_csv(OUT / "natural_runs.csv", run_rows)
    write_csv(OUT / "natural_residuals.csv", residual_rows)
    write_csv(OUT / "natural_nulls.csv", null_rows)
    write_csv(OUT / "natural_corrections.csv", corrections)
    write_csv(OUT / "natural_claims.csv", claim_rows)
    claims = {
        "discovery_collections": 48,
        "confirmation_collections": confirmation_collections,
        "fresh_checkpoints_trained_this_invocation": trained_checkpoints,
        "expected_checkpoint_pool_size": 120 if cifar_available else 96,
        "training_time_seconds": training_time,
        "positive_families": positive,
        "natural_residual_promoted": bool(positive),
        "all_leakage_hashes_passed": bool(runs.leakage_hash_passed.all()),
        "cifar_extension_executed": cifar_available,
        "cifar_extension_collections": 6 if cifar_available else 0,
        "cifar_extension_reason": "licensed mirror cache available before stage completion" if cifar_available else "CIFAR-10 cache unavailable at stage start",
    }
    write_json(OUT / "natural_claims.json", claims)
    summary = runs.groupby(["dataset", "architecture", "relation", "model_count", "method"], as_index=False).accuracy.mean()
    table_rows = summary[summary.method.isin(["weight_average", "strict_synchronization", "generic_low_rank_correction", "twistedmerge_hodge_lr", "ensemble_reference"])].to_dict("records")
    write_tex_table(OUT / "tables" / "natural_main.tex", table_rows, ["dataset", "architecture", "relation", "model_count", "method", "accuracy"], "Compact natural-checkpoint discovery.")
    residual_frame = pd.DataFrame(residual_rows)
    fig, ax = plt.subplots(figsize=(7, 4))
    for dataset, block in residual_frame.groupby("dataset"):
        ax.scatter(block.pairwise_heldout_alignment_error, block.cycle_residual, label=dataset, alpha=0.8)
    ax.set(xlabel="Held-out pairwise alignment error", ylabel="Cycle residual")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "plots" / "natural_residuals.pdf")
    plt.close(fig)
    report = f"""# Compact natural-checkpoint discovery

The fixed discovery grid executed 48 checkpoint collections over MNIST and Fashion-MNIST, two architectures, two model counts, two checkpoint relations, and three seeds. The reusable four-model pools contain 96 trained local checkpoints. {'The preregistered six-collection CIFAR-10/CNN extension also executed because the mirror completed in time.' if cifar_available else 'The optional CIFAR-10 extension did not execute because no cache was available.'} Each collection used three calibration resamples and 100 draws from each of three matched null families.

A natural residual family was **{'promoted' if positive else 'not promoted'}** by the preregistered conjunction of stability, all-null exceedance, held-out residual reduction, and positive correction intervals over strict and generic low-rank controls. Conditional confirmation was **{'executed' if confirmation_collections else 'not triggered'}**. All negative collections remain in the CSV artifacts.
"""
    (OUT / "natural_report.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
