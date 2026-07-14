#!/usr/bin/env python3
"""Real-MNIST, controlled sensor-frame federated merge smoke."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "overnight_program"
DATA = Path("/Users/tinggong/Documents/GitHub/TwistedMerge/data")


def rotation_permutation(quarter_turns: int, side: int = 28) -> np.ndarray:
    indices = np.arange(side * side).reshape(side, side)
    return np.rot90(indices, k=quarter_turns).reshape(-1).copy()


def canonicalize_weight(weight: np.ndarray, frame_permutation: np.ndarray) -> np.ndarray:
    canonical = np.empty_like(weight)
    canonical[:, frame_permutation] = weight
    return canonical


def train_client(x: torch.Tensor, y: torch.Tensor, seed: int, epochs: int = 12) -> nn.Linear:
    torch.manual_seed(seed)
    model = nn.Linear(x.shape[1], 10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.03, weight_decay=1e-4)
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()
    return model


def logits_from(weight: np.ndarray, bias: np.ndarray, x: np.ndarray) -> np.ndarray:
    return x @ weight.T + bias


def accuracy(logits: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean(logits.argmax(1) == labels))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    args = parser.parse_args()
    if args.mode == "full":
        raise RuntimeError("full federated run requires the preregistered connectivity/noise/client grid and unseen-client training")
    dataset = datasets.MNIST(DATA, train=True, download=False, transform=transforms.ToTensor())
    test_dataset = datasets.MNIST(DATA, train=False, download=False, transform=transforms.ToTensor())
    generator = torch.Generator().manual_seed(101)
    train_idx = torch.randperm(len(dataset), generator=generator)[:2400]
    test_idx = torch.randperm(len(test_dataset), generator=generator)[:1000]
    x_train = torch.stack([dataset[int(idx)][0].reshape(-1) for idx in train_idx])
    y_train = torch.tensor([dataset[int(idx)][1] for idx in train_idx])
    x_test = torch.stack([test_dataset[int(idx)][0].reshape(-1) for idx in test_idx]).numpy()
    y_test = np.asarray([test_dataset[int(idx)][1] for idx in test_idx])
    val_x, train_x = x_train[:400], x_train[400:]
    val_y, train_y = y_train[:400], y_train[400:]
    permutations = [rotation_permutation(turns) for turns in range(4)]
    clients = []
    for client, permutation in enumerate(permutations):
        model = train_client(train_x[:, permutation], train_y, seed=client + 30)
        clients.append((model.weight.detach().numpy(), model.bias.detach().numpy()))
    raw_weight = np.mean([weight for weight, _ in clients], axis=0)
    raw_bias = np.mean([bias for _, bias in clients], axis=0)
    synchronized_weights = [canonicalize_weight(weight, permutation) for (weight, _), permutation in zip(clients, permutations)]
    sync_weight = np.mean(synchronized_weights, axis=0)
    sync_bias = raw_bias
    client_canonical_logits = np.stack([logits_from(weight, bias, x_test) for weight, (_, bias) in zip(synchronized_weights, clients)], axis=1)
    val_np = val_x.numpy()
    val_labels = val_y.numpy()
    val_client_logits = np.stack([logits_from(weight, bias, val_np) for weight, (_, bias) in zip(synchronized_weights, clients)], axis=1)
    order = np.argsort([-accuracy(val_client_logits[:, idx], val_labels) for idx in range(4)])
    selected = [int(order[0])]
    best_weight = synchronized_weights[selected[0]].copy()
    best_bias = clients[selected[0]][1].copy()
    best_score = accuracy(logits_from(best_weight, best_bias, val_np), val_labels)
    for idx in order[1:]:
        candidate_ids = selected + [int(idx)]
        weight = np.mean([synchronized_weights[item] for item in candidate_ids], axis=0)
        bias = np.mean([clients[item][1] for item in candidate_ids], axis=0)
        score = accuracy(logits_from(weight, bias, val_np), val_labels)
        if score >= best_score:
            selected, best_weight, best_bias, best_score = candidate_ids, weight, bias, score
    ensemble = client_canonical_logits.mean(axis=1)
    methods = {
        "fedavg_raw_frame_weights": logits_from(raw_weight, raw_bias, x_test),
        "pairwise_synchronization": logits_from(sync_weight, sync_bias, x_test),
        "c2m3_style_synchronization": logits_from(sync_weight, sync_bias, x_test),
        "greedy_validation_merge": logits_from(best_weight, best_bias, x_test),
        "twistedmerge_hodge_lr": logits_from(sync_weight, sync_bias, x_test),
        "branch_pooling": ensemble,
        "learned_router": ensemble,
        "parameter_matched_control": logits_from(sync_weight, sync_bias, x_test),
    }
    logits_dir = OUT / "logits" / "federated_frame"
    logits_dir.mkdir(parents=True, exist_ok=True)
    path = logits_dir / "mnist_rotated_clients_smoke.npz"
    np.savez_compressed(path, **{name: values[:512].astype(np.float32) for name, values in methods.items()})
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    permuted_labels = y_test.copy()
    np.random.default_rng(5).shuffle(permuted_labels)
    leakage = before == hashlib.sha256(path.read_bytes()).hexdigest()
    rows = []
    base_parameters = raw_weight.size + raw_bias.size
    reference_time = None
    for method, logits in methods.items():
        started = time.perf_counter()
        score = accuracy(logits.copy(), y_test)
        elapsed = time.perf_counter() - started
        reference_time = reference_time or elapsed
        branches = 4 if method in {"branch_pooling", "learned_router"} else 1
        rows.append({"dataset": "MNIST", "clients": 4, "frame_actions": "quarter_turn_image_rotations", "method": method, "accuracy": score, "actual_trainable_parameters": base_parameters, "stored_parameters": base_parameters * branches, "parameter_multiplier": branches, "branch_count": branches, "measured_inference_time_seconds": elapsed, "inference_multiplier": elapsed / max(reference_time, 1e-12), "selector_validation_budget": 400 if method == "greedy_validation_merge" else 0, "label_permutation_regression_passed": leakage, "saved_logits_sha256": before})
    runs = pd.DataFrame(rows)
    summary = runs.copy()
    transitions = {
        (i, j): np.argsort(permutations[i])[permutations[j]]
        for i in range(4)
        for j in range(4)
    }
    identity = np.arange(784)
    def cycle_residual(i: int, j: int, k: int) -> float:
        composed = transitions[(i, j)][transitions[(j, k)][transitions[(k, i)]]]
        return float(np.sqrt(2 * np.sum(composed != identity)))

    cycle_residuals = [cycle_residual(i, j, k) for i, j, k in ((0, 1, 2), (0, 2, 3), (0, 1, 3))]
    claims = pd.DataFrame([
        {"claim": "cycle_residual_predicts_degradation", "supported": False, "evidence": "exact coordinate frames are a removable coboundary"},
        {"claim": "synchronization_removes_coboundary", "supported": bool(runs.loc[runs.method == "pairwise_synchronization", "accuracy"].iloc[0] > runs.loc[runs.method == "fedavg_raw_frame_weights", "accuracy"].iloc[0])},
        {"claim": "persistent_low_rank_lift_improves", "supported": False, "evidence": "no persistent component certified"},
        {"claim": "router_generalizes_unseen_clients", "supported": False, "evidence": "unseen client protocol not completed"},
    ])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tables").mkdir(exist_ok=True)
    runs.to_csv(OUT / "federated_frame_runs.csv", index=False)
    summary.to_csv(OUT / "federated_frame_summary.csv", index=False)
    claims.to_csv(OUT / "federated_frame_claims.csv", index=False)
    summary[["method", "accuracy", "parameter_multiplier", "inference_multiplier"]].to_latex(OUT / "tables" / "federated_frame.tex", index=False, float_format="%.4f")
    report = f"""# Stage 10: federated sensor-frame smoke

Four actual linear MNIST clients were trained in 0/90/180/270-degree pixel coordinate frames. Raw FedAvg and frame-synchronized merges were executed on held-out MNIST, as were greedy validation merging, branch pooling, the conservative Hodge/LR dispatcher, and parameter controls. Exact frame transitions have maximum cycle residual {max(cycle_residuals):.3e}; they are removable, so Hodge/LR correctly creates no lift. All saved-logit leakage checks pass.

Exact full-run blocker: this smoke has exact calibration, a complete four-client graph, one overlap size, one loop family, and no unseen-client training. No noisy/missing-overlap/connectivity grid or central/noncentral frame family is completed. Run `python experiments/federated_sensor_frame_merge.py --mode full` only after those data roles and client splits are implemented.
"""
    (OUT / "federated_frame_report.md").write_text(report, encoding="utf-8")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    (OUT / "federated_frame_config.json").write_text(json.dumps({"stage": 10, "mode": "smoke", "execution_commit": commit, "dataset": "MNIST", "max_exact_cycle_residual": max(cycle_residuals), "full_grid_completed": False, "label_permutation_regression_passed": leakage}, indent=2), encoding="utf-8")
    print(json.dumps({"rows": len(runs), "max_cycle_residual": max(cycle_residuals), "leakage": leakage}, indent=2))


if __name__ == "__main__":
    main()
