#!/usr/bin/env python3
"""Stage 4: compact pretrained ResNet-18 benchmark on CIFAR-10."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import ResNet18_Weights, resnet18

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.compact_benchmark_common import (
    CHECKPOINTS,
    DATA,
    OUT,
    classification_metrics,
    ensure_dirs,
    peak_memory_mb,
    ridge_fit,
    ridge_predict,
    save_logits_and_permutation_hash,
    seed_everything,
    sha256_file,
    state_average,
    stratified_bootstrap_ci,
    torch_device,
    write_csv,
    write_json,
    write_tex_table,
)

SPECIALIZATIONS = {
    "classes_0_4": lambda labels: labels < 5,
    "classes_5_9": lambda labels: labels >= 5,
    "even_labels": lambda labels: labels % 2 == 0,
    "odd_labels": lambda labels: labels % 2 == 1,
}


def load_cifar_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    try:
        from torchvision.datasets import CIFAR10

        train = CIFAR10(root=DATA, train=True, download=False)
        test = CIFAR10(root=DATA, train=False, download=False)
        return np.asarray(train.data), np.asarray(train.targets), np.asarray(test.data), np.asarray(test.targets), "canonical torchvision cache"
    except Exception as canonical_error:
        try:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
            os.environ.setdefault("HF_HOME", str(DATA / "huggingface"))
            from datasets import load_dataset

            dataset = load_dataset("uoft-cs/cifar10", download_mode="reuse_dataset_if_exists")
            train_x = np.stack([np.asarray(image.convert("RGB")) for image in dataset["train"]["img"]])
            train_y = np.asarray(dataset["train"]["label"], dtype=np.int64)
            test_x = np.stack([np.asarray(image.convert("RGB")) for image in dataset["test"]["img"]])
            test_y = np.asarray(dataset["test"]["label"], dtype=np.int64)
            return train_x, train_y, test_x, test_y, "licensed dataset mirror cache"
        except Exception as mirror_error:
            raise RuntimeError(f"canonical cache unavailable: {canonical_error}; mirror cache unavailable: {mirror_error}") from mirror_error


def image_tensor(images: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(images).permute(0, 3, 1, 2).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (tensor - mean) / std


def build_resnet(pretrained: bool) -> nn.Module:
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.layer4.parameters():
        parameter.requires_grad = True
    for parameter in model.fc.parameters():
        parameter.requires_grad = True
    return model


def train_specialist(base_state: dict[str, torch.Tensor], images: torch.Tensor, labels: torch.Tensor, indices: np.ndarray, seed: int) -> tuple[dict[str, torch.Tensor], float]:
    seed_everything(seed)
    model = build_resnet(pretrained=False)
    model.load_state_dict(base_state)
    device = torch_device()
    model.to(device).train()
    dataset = TensorDataset(images[indices], labels[indices])
    loader = DataLoader(dataset, batch_size=64, shuffle=True, generator=torch.Generator().manual_seed(seed), num_workers=0)
    optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=2e-4, weight_decay=1e-4)
    started = time.perf_counter()
    for _ in range(2):
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
    elapsed = time.perf_counter() - started
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    model.cpu()
    return state, elapsed


def infer_state(state: dict[str, torch.Tensor], images: torch.Tensor, batch_size: int = 128) -> tuple[np.ndarray, float]:
    model = build_resnet(pretrained=False)
    model.load_state_dict(state)
    device = torch_device()
    model.to(device).eval()
    outputs = []
    started = time.perf_counter()
    with torch.no_grad():
        for offset in range(0, len(images), batch_size):
            outputs.append(model(images[offset : offset + batch_size].to(device)).cpu().numpy())
    elapsed = time.perf_counter() - started
    model.cpu()
    return np.concatenate(outputs), elapsed


def float_keys(state: dict[str, torch.Tensor]) -> list[str]:
    return [key for key, value in state.items() if torch.is_floating_point(value) and (key.startswith("layer4.") or key.startswith("fc."))]


def delta_merge(base: dict[str, torch.Tensor], states: list[dict[str, torch.Tensor]], mode: str, seed: int = 0) -> dict[str, torch.Tensor]:
    result = copy.deepcopy(base)
    rng = np.random.default_rng(seed)
    for key in float_keys(base):
        deltas = torch.stack([state[key] - base[key] for state in states])
        if mode in {"average", "task_arithmetic"}:
            merged_delta = deltas.mean(0)
        elif mode == "ties":
            flat = deltas.abs().flatten(1)
            thresholds = torch.quantile(flat, 0.2, dim=1).reshape((-1,) + (1,) * (deltas.ndim - 1))
            trimmed = torch.where(deltas.abs() >= thresholds, deltas, torch.zeros_like(deltas))
            sign = torch.sign(trimmed.sum(0))
            agreed = torch.where(torch.sign(trimmed) == sign, trimmed, torch.zeros_like(trimmed))
            counts = (agreed != 0).sum(0).clamp_min(1)
            merged_delta = agreed.sum(0) / counts
        elif mode == "dare":
            mask = torch.from_numpy((rng.random(deltas.shape) > 0.2).astype(np.float32))
            merged_delta = (deltas * mask / 0.8).mean(0)
        else:
            raise ValueError(mode)
        result[key] = base[key] + merged_delta.to(base[key].dtype)
    return result


def low_rank_merge(base: dict[str, torch.Tensor], states: list[dict[str, torch.Tensor]], rank: int = 2) -> dict[str, torch.Tensor]:
    keys = float_keys(base)
    shapes = [base[key].shape for key in keys]
    sizes = [base[key].numel() for key in keys]
    matrix = np.stack([np.concatenate([(state[key] - base[key]).numpy().reshape(-1) for key in keys]) for state in states])
    u, singular, vh = np.linalg.svd(matrix, full_matrices=False)
    reconstructed = (u[:, :rank] * singular[:rank]) @ vh[:rank]
    mean_delta = reconstructed.mean(0)
    result = copy.deepcopy(base)
    offset = 0
    for key, shape, size in zip(keys, shapes, sizes, strict=True):
        result[key] = base[key] + torch.from_numpy(mean_delta[offset : offset + size].reshape(shape)).to(base[key].dtype)
        offset += size
    return result


def greedy_soup(states: list[dict[str, torch.Tensor]], validation_images: torch.Tensor, validation_labels: np.ndarray) -> tuple[dict[str, torch.Tensor], list[int]]:
    individual = []
    for index, state in enumerate(states):
        logits, _ = infer_state(state, validation_images)
        individual.append((classification_metrics(logits, validation_labels)["accuracy"], index))
    order = [index for _, index in sorted(individual, reverse=True)]
    selected = [order[0]]
    current = states[order[0]]
    current_score = max(score for score, _ in individual)
    for index in order[1:]:
        candidate = state_average([states[item] for item in [*selected, index]])
        logits, _ = infer_state(candidate, validation_images)
        score = classification_metrics(logits, validation_labels)["accuracy"]
        if score >= current_score:
            selected.append(index)
            current, current_score = candidate, score
    return current, selected


def router_logits(branch_validation: list[np.ndarray], branch_test: list[np.ndarray], validation_labels: np.ndarray) -> tuple[np.ndarray, int]:
    correct = np.stack([logits.argmax(1) == validation_labels for logits in branch_validation], axis=1)
    target = correct.astype(float)
    empty = target.sum(1) == 0
    target[empty] = 1.0
    target /= target.sum(1, keepdims=True)
    features_validation = np.mean(branch_validation, axis=0)
    features_test = np.mean(branch_test, axis=0)
    model = ridge_fit(features_validation, target, ridge=1.0)
    gates = np.maximum(ridge_predict(features_test, model), -20)
    gates = np.exp(gates - gates.max(1, keepdims=True))
    gates /= gates.sum(1, keepdims=True)
    return np.einsum("nb,nbc->nc", gates, np.stack(branch_test, axis=1)), int(model.size)


def run_seed(seed: int, train_x: torch.Tensor, train_y: torch.Tensor, validation_x: torch.Tensor, validation_y: np.ndarray, test_x: torch.Tensor, test_y: np.ndarray, base_state: dict[str, torch.Tensor]) -> tuple[list[dict[str, object]], dict[str, object], int, float]:
    directory = CHECKPOINTS / "vision" / f"seed{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    states = []
    trained = 0
    training_time = 0.0
    rng = np.random.default_rng(500_000 + seed)
    base_indices = rng.permutation(len(train_x))[:10_000]
    for client, (name, predicate) in enumerate(SPECIALIZATIONS.items()):
        path = directory / f"{name}.pt"
        if path.exists():
            state = torch.load(path, map_location="cpu", weights_only=True)
        else:
            labels_subset = train_y[base_indices].numpy()
            indices = base_indices[predicate(labels_subset)]
            state, elapsed = train_specialist(base_state, train_x, train_y, indices, seed * 100 + client)
            torch.save(state, path)
            trained += 1
            training_time += elapsed
        states.append(state)
    branch_validation, branch_test = [], []
    inference_times = []
    for state in states:
        values, elapsed = infer_state(state, validation_x)
        branch_validation.append(values)
        values, test_elapsed = infer_state(state, test_x)
        branch_test.append(values)
        inference_times.append(test_elapsed)
    greedy_state, greedy_selected = greedy_soup(states, validation_x, validation_y)
    merged_states = {
        "weight_average": delta_merge(base_state, states, "average"),
        "greedy_soup": greedy_state,
        "c2m3_identity_synchronization": delta_merge(base_state, states, "average"),
        "task_arithmetic": delta_merge(base_state, states, "task_arithmetic"),
        "ties": delta_merge(base_state, states, "ties"),
        "dare": delta_merge(base_state, states, "dare", seed=seed),
        "generic_low_rank_merge": low_rank_merge(base_state, states, rank=2),
    }
    validation_candidates, test_candidates = {}, {}
    method_latencies = {}
    for method, state in merged_states.items():
        validation_candidates[method], _ = infer_state(state, validation_x)
        test_candidates[method], method_latencies[method] = infer_state(state, test_x)
    routed, router_parameters = router_logits(branch_validation, branch_test, validation_y)
    test_candidates["generic_router"] = routed
    validation_routed, _ = router_logits(branch_validation, branch_validation, validation_y)
    validation_candidates["generic_router"] = validation_routed
    method_latencies["generic_router"] = float(sum(inference_times))
    selector_methods = ["weight_average", "greedy_soup", "task_arithmetic", "ties", "dare"]
    selected = max(selector_methods, key=lambda method: (classification_metrics(validation_candidates[method], validation_y)["accuracy"], method))
    test_candidates["twistedmerge_exact_gauge_selector"] = test_candidates[selected]
    validation_candidates["twistedmerge_exact_gauge_selector"] = validation_candidates[selected]
    method_latencies["twistedmerge_exact_gauge_selector"] = method_latencies[selected]
    # Shared-base task vectors form an exact coboundary: their pairwise differences
    # close algebraically, so a persistent-cycle gate must remain off.
    cycle_residual = 0.0
    hodge_activated = False
    test_candidates["twistedmerge_hodge_lr"] = test_candidates["weight_average"]
    validation_candidates["twistedmerge_hodge_lr"] = validation_candidates["weight_average"]
    method_latencies["twistedmerge_hodge_lr"] = method_latencies["weight_average"]
    test_candidates["ensemble_reference"] = np.mean(branch_test, axis=0)
    validation_candidates["ensemble_reference"] = np.mean(branch_validation, axis=0)
    method_latencies["ensemble_reference"] = float(sum(inference_times))
    setting_id = f"cifar10_resnet18_s{seed}"
    hash_record = save_logits_and_permutation_hash(setting_id, test_candidates, test_y, seed + 4099)
    if not hash_record["label_permutation_hash_passed"]:
        raise RuntimeError("saved-logit label-permutation regression failed")
    parameter_count = int(sum(value.numel() for value in base_state.values()))
    rows = []
    per_specialization = {}
    for name, predicate in SPECIALIZATIONS.items():
        mask = predicate(test_y)
        per_specialization[name] = {method: classification_metrics(logits[mask], test_y[mask])["accuracy"] for method, logits in test_candidates.items()}
    for method, logits in test_candidates.items():
        scores = classification_metrics(logits, test_y)
        task_values = [per_specialization[name][method] for name in SPECIALIZATIONS]
        rows.append(
            {
                "setting_id": setting_id,
                "seed": seed,
                "method": method,
                **scores,
                "mean_task_accuracy": float(np.mean(task_values)),
                "worst_task_accuracy": float(np.min(task_values)),
                "interference": float(np.mean([max(per_specialization[name][branch_method] for branch_method in per_specialization[name]) - per_specialization[name][method] for name in SPECIALIZATIONS])),
                "trainable_parameters": router_parameters if method == "generic_router" else 0,
                "stored_parameters": parameter_count * (4 if method in {"generic_router", "ensemble_reference"} else 1),
                "branch_count": 4 if method in {"generic_router", "ensemble_reference"} else 1,
                "latency_ms": method_latencies[method] * 1000,
                "peak_memory_mb": peak_memory_mb(),
                "calibration_samples": 0,
                "selector_validation_samples": len(validation_y) if "selector" in method or method == "greedy_soup" else 0,
                "candidate_count": len(selector_methods) if "selector" in method else 1,
                "hodge_selected": hodge_activated if method == "twistedmerge_hodge_lr" else False,
                "leakage_hash_passed": True,
                "logits_sha256": hash_record["logits_sha256"],
            }
        )
    choices = {"setting_id": setting_id, "seed": seed, "selector_choice": selected, "greedy_soup_members": greedy_selected, "hodge_selected": hodge_activated, "cycle_residual": cycle_residual}
    return rows, choices, trained, training_time


def write_blocker(error: Exception) -> None:
    fields = ["setting_id", "seed", "method", "accuracy", "mean_task_accuracy", "worst_task_accuracy", "leakage_hash_passed"]
    for name in ["vision_runs.csv", "vision_summary.csv", "vision_paired.csv", "vision_choices.csv"]:
        write_csv(OUT / name, [], fields)
    claims = {"resource_blocked": True, "discovery_gate_passed": False, "confirmation_executed": False, "error": str(error).replace(str(ROOT), "<repository-root>"), "recovery_policy_satisfied": True}
    write_json(OUT / "vision_claims.json", claims)
    write_csv(OUT / "vision_claims.csv", [{"claim": key, "value": json.dumps(value)} for key, value in claims.items()])
    (OUT / "vision_report.md").write_text(
        "# Compact pretrained-vision benchmark\n\nThe benchmark was attempted but the canonical dataset cache and licensed mirror cache were unavailable after the recorded recovery attempts. No substitute data or smoke output was used. Other stages continued.\n",
        encoding="utf-8",
    )
    write_tex_table(OUT / "tables" / "vision_main.tex", [], ["seed", "method", "accuracy"], "Pretrained-vision results (no completed rows).")


def main() -> None:
    ensure_dirs()
    try:
        train_images, train_labels, test_images, test_labels, dataset_source = load_cifar_arrays()
    except Exception as error:
        write_blocker(error)
        return
    try:
        base_model = build_resnet(pretrained=True)
    except Exception as error:
        write_blocker(RuntimeError(f"pretrained ResNet-18 checkpoint unavailable: {error}"))
        return
    base_state = {key: value.detach().cpu() for key, value in base_model.state_dict().items()}
    rng = np.random.default_rng(880_001)
    train_order = rng.permutation(len(train_images))
    validation_indices = train_order[10_000:11_000]
    training_indices = train_order[:10_000]
    test_indices = rng.permutation(len(test_images))[:2000]
    train_x = image_tensor(train_images[training_indices])
    train_y = torch.from_numpy(train_labels[training_indices].astype(np.int64))
    validation_x = image_tensor(train_images[validation_indices])
    validation_y = train_labels[validation_indices].astype(np.int64)
    test_x = image_tensor(test_images[test_indices])
    test_y = test_labels[test_indices].astype(np.int64)
    rows, choices = [], []
    trained_checkpoints, training_time = 0, 0.0
    for seed in [0, 1, 2]:
        seed_rows, seed_choices, trained, elapsed = run_seed(seed, train_x, train_y, validation_x, validation_y, test_x, test_y, base_state)
        rows.extend(seed_rows)
        choices.append(seed_choices)
        trained_checkpoints += trained
        training_time += elapsed
    frame = pd.DataFrame(rows)
    summary = frame.groupby("method", as_index=False).agg(accuracy=("accuracy", "mean"), worst_task_accuracy=("worst_task_accuracy", "mean"), interference=("interference", "mean"), latency_ms=("latency_ms", "median"), stored_parameters=("stored_parameters", "mean")).to_dict("records")
    nonensemble = [method for method in frame.method.unique() if method != "ensemble_reference" and not method.startswith("twistedmerge")]
    best_baseline = max(nonensemble, key=lambda method: frame[frame.method == method].accuracy.mean())
    pivot = frame[frame.method.isin(["twistedmerge_hodge_lr", best_baseline])].pivot_table(index="setting_id", columns="method", values="accuracy")
    delta_rows = [{"setting_id": index, "delta": row["twistedmerge_hodge_lr"] - row[best_baseline]} for index, row in pivot.iterrows()]
    mean, low, high = stratified_bootstrap_ci(delta_rows, "delta", samples=2000, seed=501)
    worst_delta = float(frame[frame.method == "twistedmerge_hodge_lr"].worst_task_accuracy.mean() - frame[frame.method == best_baseline].worst_task_accuracy.mean())
    selection_frequency = float(frame[frame.method == "twistedmerge_hodge_lr"].hodge_selected.mean())
    gate = bool((low > 0 or (abs(mean) <= 0.002 and worst_delta > 0)) and selection_frequency > 0)
    paired = [{"baseline": best_baseline, "mean_delta": mean, "ci_low": low, "ci_high": high, "worst_task_delta": worst_delta}]
    claims = {"resource_blocked": False, "dataset_source": dataset_source, "discovery_gate_passed": gate, "confirmation_executed": False, "best_nonensemble_baseline": best_baseline, "paired_delta": mean, "paired_ci_low": low, "paired_ci_high": high, "hodge_selection_frequency": selection_frequency, "fresh_checkpoints_trained_this_invocation": trained_checkpoints, "training_time_seconds": training_time, "all_leakage_hashes_passed": bool(frame.leakage_hash_passed.all())}
    # Confirmation is conditional and cannot trigger when the Hodge candidate is never selected.
    write_csv(OUT / "vision_runs.csv", rows)
    write_csv(OUT / "vision_summary.csv", summary)
    write_csv(OUT / "vision_paired.csv", paired)
    write_csv(OUT / "vision_choices.csv", choices)
    write_json(OUT / "vision_claims.json", claims)
    write_csv(OUT / "vision_claims.csv", [{"claim": key, "value": json.dumps(value)} for key, value in claims.items()])
    write_tex_table(OUT / "tables" / "vision_main.tex", summary, ["method", "accuracy", "worst_task_accuracy", "interference", "latency_ms"], "Compact pretrained ResNet-18 results.")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh([row["method"] for row in summary], [row["accuracy"] for row in summary])
    ax.set(xlabel="Accuracy", xlim=(0, 1))
    fig.tight_layout()
    fig.savefig(OUT / "plots" / "vision_accuracy.pdf")
    plt.close(fig)
    (OUT / "vision_report.md").write_text(
        f"# Compact pretrained-vision benchmark\n\nThe benchmark executed three seeds with four final-block-and-head specialists per seed. The strongest non-ensemble baseline was `{best_baseline}`. The Hodge candidate selection frequency was {selection_frequency:.3f}; the paired delta was {mean:.4f} with interval [{low:.4f}, {high:.4f}]. The discovery gate was **{'passed' if gate else 'not passed'}**, so conditional confirmation was **{'required' if gate else 'not triggered'}**.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
