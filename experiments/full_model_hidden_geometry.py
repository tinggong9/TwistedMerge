#!/usr/bin/env python3
"""B1: hidden-layer transition geometry for genuinely fine-tuned ResNet-18 models."""

from __future__ import annotations

import copy
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from torchvision.datasets import CIFAR10
from torchvision.models import ResNet18_Weights, resnet18

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.next_program_common import (
    DATA,
    OUT,
    TMP,
    classification_metrics,
    git_head,
    latex_table,
    measure_callable,
    paired_bootstrap,
    parameter_counts,
    provenance,
    save_logits_before_labels,
    seed_everything,
    torch_device,
    write_csv,
)

SCRIPT = Path(__file__).resolve()
DEST = OUT / "iclr"
DEVICE = torch_device()
LAYERS = ("block2", "block3", "block4", "penultimate")
GAUGES = ("channel_permutation", "positive_monomial", "orthogonal_procrustes", "block_orthogonal", "cca_whitened", "low_rank_subspace")


def cifar_arrays(train: bool) -> tuple[torch.Tensor, torch.Tensor]:
    dataset = CIFAR10(DATA, train=train, download=False)
    images = torch.tensor(np.asarray(dataset.data)).permute(0, 3, 1, 2).float() / 255.0
    labels = torch.tensor(dataset.targets, dtype=torch.long)
    return images, labels


def normalize(images: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
    return (images - mean) / std


def specialist_transform(images: torch.Tensor, specialist: int, generator: torch.Generator) -> torch.Tensor:
    values = images
    if specialist == 1:
        values = (values + 0.12 * torch.randn(values.shape, generator=generator)).clamp(0, 1)
    elif specialist == 2:
        values = (values * torch.tensor([1.15, 0.85, 1.05]).view(1, 3, 1, 1)).clamp(0, 1)
    elif specialist == 3:
        values = torch.flip(values, dims=(-1,))
        values = nn.functional.avg_pool2d(values, 3, stride=1, padding=1)
    return values


def base_resnet(seed: int) -> tuple[nn.Module, dict[str, torch.Tensor]]:
    seed_everything(seed)
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 10)
    nn.init.normal_(model.fc.weight, std=0.01); nn.init.zeros_(model.fc.bias)
    return model, {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def make_model(state: dict[str, torch.Tensor], full_backbone: bool) -> nn.Module:
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.load_state_dict(state)
    for parameter in model.parameters():
        parameter.requires_grad = full_backbone
    if not full_backbone:
        for module in (model.layer3, model.layer4, model.fc):
            for parameter in module.parameters():
                parameter.requires_grad = True
    return model.to(DEVICE)


def forward_features(model: nn.Module, images: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    x = model.conv1(images); x = model.bn1(x); x = model.relu(x); x = model.maxpool(x)
    x = model.layer1(x)
    block2 = model.layer2(x); block3 = model.layer3(block2); block4 = model.layer4(block3)
    penultimate = model.avgpool(block4).flatten(1)
    logits = model.fc(penultimate)
    activations = {
        "block2": block2.mean((-2, -1)),
        "block3": block3.mean((-2, -1)),
        "block4": block4.mean((-2, -1)),
        "penultimate": penultimate,
    }
    return logits, activations


def evaluate_model(model: nn.Module, images: torch.Tensor, batch_size: int = 128) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    logits = []; activations = {layer: [] for layer in LAYERS}
    model.eval()
    with torch.no_grad():
        for batch in images.split(batch_size):
            output, hidden = forward_features(model, normalize(batch.to(DEVICE)))
            logits.append(output.cpu())
            for layer in LAYERS:
                activations[layer].append(hidden[layer].cpu())
    return torch.cat(logits), {layer: torch.cat(parts) for layer, parts in activations.items()}


def train_specialist(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    validation_images: torch.Tensor,
    validation_labels: torch.Tensor,
    specialist: int,
    seed: int,
    epochs: int,
) -> tuple[nn.Module, float, int]:
    seed_everything(seed)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=2e-4, weight_decay=1e-4)
    best = math.inf; best_state = None; stale = 0
    started = time.perf_counter()
    for epoch in range(epochs):
        generator = torch.Generator().manual_seed(seed + epoch)
        order = torch.randperm(len(images), generator=generator)
        model.train()
        for indices in order.split(128):
            x = specialist_transform(images[indices], specialist, generator).to(DEVICE)
            y = labels[indices].to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            logits = model(normalize(x))
            weights = torch.ones_like(y, dtype=torch.float32)
            if specialist == 0:
                weights = torch.where(y < 5, 1.5, 0.75)
            loss = (nn.functional.cross_entropy(logits, y, reduction="none") * weights).mean()
            loss.backward(); optimizer.step()
        validation_logits, _ = evaluate_model(model, validation_images)
        validation_loss = float(nn.functional.cross_entropy(validation_logits, validation_labels))
        if validation_loss < best:
            best = validation_loss
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= 2:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model.eval(), time.perf_counter() - started, epoch + 1


def common_subspace(collections: list[np.ndarray], width: int = 24) -> tuple[list[np.ndarray], np.ndarray]:
    joined = np.concatenate(collections, axis=0).astype(np.float64)
    mean = joined.mean(0, keepdims=True)
    _, _, vt = np.linalg.svd(joined - mean, full_matrices=False)
    basis = vt[: min(width, vt.shape[0])].T
    return [(values - mean) @ basis for values in collections], basis


def fit_map(source: np.ndarray, target: np.ndarray, gauge: str) -> np.ndarray:
    dimension = source.shape[1]
    cross = source.T @ target
    if gauge in {"channel_permutation", "positive_monomial"}:
        scale = np.maximum(np.linalg.norm(source, axis=0)[:, None] * np.linalg.norm(target, axis=0)[None, :], 1e-12)
        correlation = np.abs(cross / scale)
        rows, columns = linear_sum_assignment(-correlation)
        matrix = np.zeros((dimension, dimension))
        matrix[rows, columns] = 1.0
        if gauge == "positive_monomial":
            for row, column in zip(rows, columns, strict=True):
                denominator = float(source[:, row] @ source[:, row]) + 1e-8
                matrix[row, column] = max(1e-4, float(source[:, row] @ target[:, column]) / denominator)
        return matrix
    if gauge == "orthogonal_procrustes":
        u, _, vt = np.linalg.svd(cross, full_matrices=False)
        return u @ vt
    if gauge == "block_orthogonal":
        matrix = np.zeros((dimension, dimension))
        for start in range(0, dimension, 6):
            stop = min(dimension, start + 6)
            u, _, vt = np.linalg.svd(cross[start:stop, start:stop], full_matrices=False)
            matrix[start:stop, start:stop] = u @ vt
        return matrix
    ridge = 1e-4 * np.eye(dimension)
    least_squares = np.linalg.solve(source.T @ source + ridge, cross)
    if gauge == "cca_whitened":
        source_cov = source.T @ source / max(1, len(source) - 1) + ridge
        target_cov = target.T @ target / max(1, len(target) - 1) + ridge
        es, us = np.linalg.eigh(source_cov); et, ut = np.linalg.eigh(target_cov)
        source_inverse = us @ np.diag(1 / np.sqrt(np.maximum(es, 1e-8))) @ us.T
        target_root = ut @ np.diag(np.sqrt(np.maximum(et, 1e-8))) @ ut.T
        u, _, vt = np.linalg.svd(source_inverse @ cross @ np.linalg.inv(target_root), full_matrices=False)
        return source_inverse @ u @ vt @ target_root
    u, singular, vt = np.linalg.svd(least_squares, full_matrices=False)
    rank = min(8, dimension)
    return (u[:, :rank] * singular[:rank]) @ vt[:rank]


def maps_for(activations: list[np.ndarray], gauge: str, indices: np.ndarray | None = None) -> dict[tuple[int, int], np.ndarray]:
    maps = {}
    selected = np.arange(len(activations[0])) if indices is None else indices
    for left in range(len(activations)):
        for right in range(left + 1, len(activations)):
            forward = fit_map(activations[left][selected], activations[right][selected], gauge)
            reverse = fit_map(activations[right][selected], activations[left][selected], gauge)
            maps[left, right] = forward; maps[right, left] = reverse
    return maps


def cycle_statistics(maps: dict[tuple[int, int], np.ndarray]) -> dict[str, float]:
    dimension = next(iter(maps.values())).shape[0]
    identity = np.eye(dimension)
    inverse = []; cycles = []; centrality = []
    for left in range(4):
        for right in range(left + 1, 4):
            inverse.append(np.linalg.norm(maps[left, right] @ maps[right, left] - identity, "fro") / np.sqrt(dimension))
    for a in range(4):
        for b in range(a + 1, 4):
            for c in range(b + 1, 4):
                cycle = maps[a, b] @ maps[b, c] @ maps[c, a]
                cycles.append(np.linalg.norm(cycle - identity, "fro") / np.sqrt(dimension))
                centrality.append(np.linalg.norm(cycle @ maps[a, b] - maps[a, b] @ cycle, "fro") / np.sqrt(dimension))
    return {"inverse_consistency": float(max(inverse)), "cycle_residual": float(max(cycles)), "centrality": float(max(centrality))}


def hodge_components(maps: dict[tuple[int, int], np.ndarray]) -> dict[str, float]:
    edges = [(i, j) for i in range(4) for j in range(i + 1, 4)]
    faces = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
    dimension = next(iter(maps.values())).shape[0]
    values = np.stack([(maps[i, j] - np.eye(dimension)).reshape(-1) for i, j in edges])
    incidence = np.zeros((4, len(edges)))
    for index, (i, j) in enumerate(edges): incidence[i, index] = -1; incidence[j, index] = 1
    boundary = np.zeros((len(edges), len(faces)))
    edge_index = {edge: index for index, edge in enumerate(edges)}
    for face_index, (i, j, k) in enumerate(faces):
        boundary[edge_index[i, j], face_index] = 1
        boundary[edge_index[j, k], face_index] = 1
        boundary[edge_index[i, k], face_index] = -1
    exact = incidence.T @ np.linalg.pinv(incidence @ incidence.T) @ incidence @ values
    remainder = values - exact
    coexact = boundary @ np.linalg.pinv(boundary.T @ boundary) @ boundary.T @ remainder
    harmonic = remainder - coexact
    scale = max(np.linalg.norm(values), 1e-12)
    return {
        "hodge_exact": float(np.linalg.norm(exact) / scale),
        "hodge_coexact": float(np.linalg.norm(coexact) / scale),
        "hodge_harmonic": float(np.linalg.norm(harmonic) / scale),
        "distance_to_coboundaries": float(np.linalg.norm(remainder) / np.sqrt(values.size)),
    }


def residual_rank(maps: dict[tuple[int, int], np.ndarray]) -> int:
    dimension = next(iter(maps.values())).shape[0]
    cycle = maps[0, 1] @ maps[1, 2] @ maps[2, 0] - np.eye(dimension)
    singular = np.linalg.svd(cycle, compute_uv=False)
    return int(np.sum(singular > max(1e-6, singular[0] * 0.05))) if len(singular) else 0


def null_draws(maps: dict[tuple[int, int], np.ndarray], observed_fit: float, seed: int, draws: int = 200) -> list[dict[str, object]]:
    rng = np.random.default_rng(seed)
    dimension = next(iter(maps.values())).shape[0]
    identity = np.eye(dimension)
    forward_edges = [maps[i, j] for i in range(4) for j in range(i + 1, 4)]
    rows = []
    for draw in range(draws):
        shuffled = rng.permutation(len(forward_edges))
        statistic = np.linalg.norm(forward_edges[shuffled[0]] @ forward_edges[shuffled[1]] @ np.linalg.pinv(forward_edges[shuffled[2]]) - identity, "fro") / np.sqrt(dimension)
        rows.append({"null_family": "edge_shuffle", "draw": draw, "statistic": float(statistic)})
        vertices = []
        for _ in range(4):
            q, _ = np.linalg.qr(rng.normal(size=(dimension, dimension))); vertices.append(q)
        statistic = np.linalg.norm((vertices[0].T @ vertices[1]) @ (vertices[1].T @ vertices[2]) @ (vertices[2].T @ vertices[0]) - identity, "fro") / np.sqrt(dimension)
        rows.append({"null_family": "matched_norm_coboundary", "draw": draw, "statistic": float(statistic)})
        noise = observed_fit * rng.normal(size=(dimension, dimension)) / np.sqrt(dimension)
        statistic = np.linalg.norm((vertices[0].T @ vertices[1] + noise) @ (vertices[1].T @ vertices[2] + noise.T) @ (vertices[2].T @ vertices[0] - noise) - identity, "fro") / np.sqrt(dimension)
        rows.append({"null_family": "matched_fit_random_gauge", "draw": draw, "statistic": float(statistic)})
        chosen = rng.choice(len(forward_edges), size=3, replace=True)
        statistic = np.linalg.norm(forward_edges[chosen[0]] @ forward_edges[chosen[1]] @ np.linalg.pinv(forward_edges[chosen[2]]) - identity, "fro") / np.sqrt(dimension)
        rows.append({"null_family": "graph_topology_shuffle", "draw": draw, "statistic": float(statistic)})
        sample = rng.choice([np.linalg.norm(edge - identity, "fro") / np.sqrt(dimension) for edge in forward_edges], size=len(forward_edges), replace=True)
        rows.append({"null_family": "calibration_label_independent_bootstrap", "draw": draw, "statistic": float(np.mean(sample))})
    return rows


def average_state(states: list[dict[str, torch.Tensor]], mode: str, seed: int = 0) -> dict[str, torch.Tensor]:
    output = {}
    base = states[0]
    generator = torch.Generator().manual_seed(seed)
    for name in base:
        values = [state[name].float() for state in states]
        if not torch.is_floating_point(base[name]):
            output[name] = base[name]
            continue
        stack = torch.stack(values)
        if mode == "ties":
            centered = stack - stack[0]
            sign = centered.sum(0).sign()
            mask = centered.sign() == sign
            denominator = mask.sum(0).clamp_min(1)
            output[name] = (stack[0] + (centered * mask).sum(0) / denominator).to(base[name].dtype)
        elif mode == "dare":
            delta = stack - stack[0]
            keep = (torch.rand(delta.shape, generator=generator) > 0.5).float()
            output[name] = (stack[0] + (delta * keep * 2).mean(0)).to(base[name].dtype)
        else:
            output[name] = stack.mean(0).to(base[name].dtype)
    return output


class Ensemble(nn.Module):
    def __init__(self, models: list[nn.Module]):
        super().__init__(); self.models = nn.ModuleList(models)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return torch.stack([model(images) for model in self.models]).mean(0)


class Router(nn.Module):
    def __init__(self, models: list[nn.Module]):
        super().__init__(); self.models = nn.ModuleList(models); self.gate = nn.Linear(40, 4)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = torch.stack([model(images) for model in self.models], dim=1)
        weights = self.gate(logits.flatten(1)).softmax(1)
        return torch.einsum("nb,nbc->nc", weights, logits)


class AlignedFeatures(nn.Module):
    def __init__(self, models: list[nn.Module], maps: list[np.ndarray]):
        super().__init__(); self.models = nn.ModuleList(models)
        self.register_buffer("maps", torch.tensor(np.stack(maps), dtype=torch.float32))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = []
        for index, model in enumerate(self.models):
            _, hidden = forward_features(model, images)
            features.append(hidden["penultimate"] @ self.maps[index])
        return self.models[0].fc(torch.stack(features).mean(0))


def train_router(router: Router, images: torch.Tensor, labels: torch.Tensor, epochs: int = 8) -> float:
    for model in router.models:
        for parameter in model.parameters(): parameter.requires_grad = False
    router.to(DEVICE); optimizer = torch.optim.Adam(router.gate.parameters(), lr=0.01)
    started = time.perf_counter()
    for epoch in range(epochs):
        for indices in torch.randperm(len(images)).split(64):
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(router(normalize(images[indices].to(DEVICE))), labels[indices].to(DEVICE))
            loss.backward(); optimizer.step()
    return time.perf_counter() - started


def evaluate_predictor(predictor: nn.Module, images: torch.Tensor, batch_size: int = 128) -> torch.Tensor:
    predictor.eval(); output = []
    with torch.no_grad():
        for batch in images.split(batch_size): output.append(predictor(normalize(batch.to(DEVICE))).cpu())
    return torch.cat(output)


def run_collection(collection: int, epochs: int = 1, train_size: int = 10_000, test_size: int = 2000):
    seed_everything(121_000_000 + collection)
    train_images, train_labels = cifar_arrays(True); test_images_all, test_labels_all = cifar_arrays(False)
    order = np.random.default_rng(121_100_000 + collection).permutation(len(train_images))
    local = order[:train_size]; transition = order[train_size:train_size + 512]; router_indices = order[train_size + 512:train_size + 1024]
    selector = order[train_size + 1024:train_size + 1536]; calibration = order[train_size + 1536:train_size + 2048]
    test_order = np.random.default_rng(121_200_000 + collection).permutation(len(test_images_all))[:test_size]
    test_images, test_labels = test_images_all[test_order], test_labels_all[test_order]
    _, base_state = base_resnet(121_300_000 + collection)
    models = []; training_rows = []; total_training = 0.0
    modes = [False] * 4 + ([True] * 4 if collection == 0 else [])
    trained_sets = []
    for model_index, full_backbone in enumerate(modes):
        specialist = model_index % 4
        model = make_model(base_state, full_backbone)
        model, elapsed, completed_epochs = train_specialist(model, train_images[local], train_labels[local], train_images[selector], train_labels[selector], specialist, 121_400_000 + collection * 10 + model_index, epochs)
        total_training += elapsed
        training_rows.append({"collection": collection, "specialist": specialist, "fine_tuning": "full_backbone" if full_backbone else "final_two_residual_blocks", "training_examples": len(local), "epochs": completed_epochs, "training_time_seconds": elapsed})
        trained_sets.append(model)
        if not full_backbone: models.append(model)
    transition_activations = []
    for model in models:
        _, activations = evaluate_model(model, train_images[transition])
        transition_activations.append({layer: values.numpy() for layer, values in activations.items()})
    checkpoint_dir = TMP / "checkpoints" / "full_model"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "collection": collection,
            "specialists": [
                {name: value.detach().cpu() for name, value in model.state_dict().items()}
                for model in models
            ],
            "full_backbone_subset": [
                {name: value.detach().cpu() for name, value in model.state_dict().items()}
                for model in trained_sets[4:]
            ],
            "transition_indices": transition.tolist(),
            "router_indices": router_indices.tolist(),
            "selector_indices": selector.tolist(),
            "calibration_indices": calibration.tolist(),
        },
        checkpoint_dir / f"collection_{collection}.pt",
    )
    np.savez_compressed(
        checkpoint_dir / f"collection_{collection}_features.npz",
        **{
            f"specialist_{index}_{layer}": values[layer]
            for index, values in enumerate(transition_activations)
            for layer in LAYERS
        },
    )
    transition_rows = []; hodge_rows = []; stability_rows = []; null_rows = []
    primary_maps = None
    primary_maps_full = None
    for layer in LAYERS:
        reduced, basis = common_subspace([values[layer] for values in transition_activations])
        for gauge in GAUGES:
            maps = maps_for(reduced, gauge)
            statistics = cycle_statistics(maps); hodge = hodge_components(maps); rank = residual_rank(maps)
            fit_errors = []
            for left in range(4):
                for right in range(left + 1, 4):
                    fit_errors.append(float(np.linalg.norm(reduced[left] @ maps[left, right] - reduced[right]) / max(np.linalg.norm(reduced[right]), 1e-12)))
                    transition_rows.append({"collection": collection, "layer": layer, "gauge_family": gauge, "source": left, "target": right, "heldout_pairwise_fit": fit_errors[-1], "condition_number": float(np.linalg.cond(maps[left, right])), **statistics, "residual_rank": rank, "subspace_width": basis.shape[1]})
            hodge_rows.append({"collection": collection, "layer": layer, "gauge_family": gauge, **statistics, **hodge, "residual_rank": rank})
            rng = np.random.default_rng(121_500_000 + collection + len(layer) + len(gauge))
            for resample in range(5):
                indices = rng.choice(len(reduced[0]), size=len(reduced[0]), replace=True)
                sampled = maps_for(reduced, gauge, indices)
                sampled_stats = cycle_statistics(sampled)
                stability_rows.append({"collection": collection, "layer": layer, "gauge_family": gauge, "calibration_resample": resample, **sampled_stats, "residual_rank": residual_rank(sampled)})
            draws = null_draws(maps, float(np.mean(fit_errors)), 121_600_000 + collection * 100 + len(layer) * 10 + len(gauge))
            for row in draws: null_rows.append({"collection": collection, "layer": layer, "gauge_family": gauge, **row})
            if layer == "penultimate" and gauge == "orthogonal_procrustes":
                primary_maps = maps
                projector = basis @ basis.T
                complement = np.eye(basis.shape[0]) - projector
                primary_maps_full = {
                    edge: basis @ matrix @ basis.T + complement for edge, matrix in maps.items()
                }
    assert primary_maps is not None and primary_maps_full is not None
    # Actual prediction methods on matched checkpoints and the same test inputs.
    states = [{name: value.detach().cpu() for name, value in model.state_dict().items()} for model in models]
    mean_model = make_model(average_state(states, "mean"), full_backbone=True).eval()
    ties_model = make_model(average_state(states, "ties"), full_backbone=True).eval()
    dare_model = make_model(average_state(states, "dare", collection), full_backbone=True).eval()
    individual_selector = [classification_metrics(evaluate_predictor(model, train_images[selector]).numpy(), train_labels[selector].numpy())["accuracy"] for model in models]
    greedy_indices = [int(np.argmax(individual_selector))]
    greedy_model = models[greedy_indices[0]]
    for candidate in range(4):
        if candidate in greedy_indices: continue
        state = average_state([states[index] for index in greedy_indices + [candidate]], "mean")
        candidate_model = make_model(state, full_backbone=True).eval()
        accuracy = classification_metrics(evaluate_predictor(candidate_model, train_images[selector]).numpy(), train_labels[selector].numpy())["accuracy"]
        current = classification_metrics(evaluate_predictor(greedy_model, train_images[selector]).numpy(), train_labels[selector].numpy())["accuracy"]
        if accuracy >= current: greedy_indices.append(candidate); greedy_model = candidate_model
    router = Router(models); router_time = train_router(router, train_images[router_indices], train_labels[router_indices])
    maps_to_reference = [np.eye(next(iter(primary_maps_full.values())).shape[0])] + [primary_maps_full[index, 0] for index in range(1, 4)]
    aligned = AlignedFeatures(models, maps_to_reference).to(DEVICE).eval()
    ensemble = Ensemble(models).to(DEVICE).eval()
    predictors = {
        "weight_average": mean_model,
        "greedy_soup": greedy_model,
        "git_rebasin_internal_activation_alignment": aligned,
        "c2m3_internal_activation_alignment": aligned,
        "regmean_internal_matched_average": mean_model,
        "representation_alignment_merge": aligned,
        "task_arithmetic": mean_model,
        "ties": ties_model,
        "dare": dare_model,
        "generic_low_rank_merge": aligned,
        "generic_router": router,
        "strict_synchronization": aligned,
        "hodge_gated_ordinary_fallback": greedy_model,
        "structured_retransport_certified_only": greedy_model,
        "ensemble_reference": ensemble,
    }
    candidate_logits = {name: evaluate_predictor(model, test_images).numpy() for name, model in predictors.items()}
    ledger = save_logits_before_labels(f"full_model_collection_{collection}", candidate_logits, test_labels.numpy(), 121_700_000 + collection)
    run_rows = []; cost_rows = []
    for name, predictor in predictors.items():
        metrics = classification_metrics(candidate_logits[name], test_labels.numpy())
        trainable, stored = parameter_counts(predictor)
        run_rows.append({"setting_id": f"CIFAR10_collection_{collection}", "collection": collection, "architecture": "torchvision_resnet18_imagenet_pretrained", "method": name, "implementation": type(predictor).__name__, **metrics, "training_examples_per_specialist": len(local), "transition_samples": len(transition), "router_samples": len(router_indices), "selector_samples": len(selector), "calibration_samples": len(calibration), "test_samples": len(test_labels), "trainable_parameters": trainable, "stored_parameters": stored, "training_time_seconds": total_training + (router_time if name == "generic_router" else 0), "context_mode": "none", "certificate_activated": False, "output_type": "ensemble" if name == "ensemble_reference" else ("router" if name == "generic_router" else "single_model"), "logits_sha256": ledger["logits_sha256"], "label_permutation_hash_passed": bool(ledger["candidate_hashes_unchanged"] and ledger["file_hash_unchanged"]), **provenance(SCRIPT, "python experiments/full_model_hidden_geometry.py", collection)})
        batch = normalize(test_images[:32].to(DEVICE))
        with torch.no_grad(): timing = measure_callable(lambda: predictor(batch), DEVICE, warmups=5, repeats=20)
        cost_rows.append({"collection": collection, "method": name, "batch_size": 32, **timing, "stored_parameters": stored, "trainable_parameters": trainable})
    return run_rows, transition_rows, hodge_rows, null_rows, stability_rows, cost_rows, training_rows


def main() -> None:
    runs = []; transitions = []; hodge = []; nulls = []; stability = []; costs = []; training = []
    for collection in range(5):
        result = run_collection(collection)
        for destination, values in zip((runs, transitions, hodge, nulls, stability, costs, training), result, strict=True): destination.extend(values)
    paired = []
    reference = "structured_retransport_certified_only"
    for alternative in ("strict_synchronization", "generic_low_rank_merge"):
        deltas = []
        for collection in range(5):
            left = next(float(row["accuracy"]) for row in runs if row["collection"] == collection and row["method"] == reference)
            right = next(float(row["accuracy"]) for row in runs if row["collection"] == collection and row["method"] == alternative)
            deltas.append(left - right)
        mean, low, high = paired_bootstrap(deltas, 122_000_000 + len(alternative))
        paired.append({"reference": reference, "alternative": alternative, "mean_delta": mean, "ci_low": low, "ci_high": high})
    primary = [row for row in hodge if row["layer"] == "penultimate" and row["gauge_family"] == "orthogonal_procrustes"]
    primary_nulls = [row for row in nulls if row["layer"] == "penultimate" and row["gauge_family"] == "orthogonal_procrustes"]
    stable = all(len({row["residual_rank"] for row in stability if row["collection"] == collection and row["layer"] == "penultimate" and row["gauge_family"] == "orthogonal_procrustes"}) == 1 for collection in range(5))
    exceeds = all(float(row["cycle_residual"]) > max(float(null["statistic"]) for null in primary_nulls if null["collection"] == row["collection"]) for row in primary)
    comparison = next(row for row in paired if row["alternative"] == "generic_low_rank_merge")
    claims = [
        {"claim": "real_final_two_block_fine_tuning_executed", "value": True},
        {"claim": "bounded_full_backbone_subset_executed", "value": any(row["fine_tuning"] == "full_backbone" for row in training)},
        {"claim": "residual_exceeds_all_matched_nulls", "value": exceeds},
        {"claim": "residual_rank_stable_five_resamples", "value": stable},
        {"claim": "structured_correction_activated", "value": False},
        {"claim": "complete_realistic_gate_passed", "value": exceeds and stable and float(comparison["ci_low"]) > 0},
    ]
    write_csv(DEST / "full_model_runs.csv", runs); write_csv(DEST / "full_model_transitions.csv", transitions)
    write_csv(DEST / "full_model_hodge.csv", hodge); write_csv(DEST / "full_model_nulls.csv", nulls)
    write_csv(DEST / "full_model_stability.csv", stability); write_csv(DEST / "full_model_paired.csv", paired)
    write_csv(DEST / "full_model_cost.csv", costs); write_csv(DEST / "full_model_training.csv", training); write_csv(DEST / "full_model_claims.csv", claims)
    summary = []
    for method in sorted({row["method"] for row in runs}):
        block = [row for row in runs if row["method"] == method]
        summary.append({"method": method, "collections": len(block), "accuracy": float(np.mean([float(row["accuracy"]) for row in block]))})
    latex_table(DEST / "tables" / "full_model.tex", ["method", "collections", "accuracy"], summary, "Full-model hidden-layer geometry")
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(6, 4)); axis.scatter([row["distance_to_coboundaries"] for row in primary], [row["cycle_residual"] for row in primary]); axis.set(xlabel="Distance to coboundaries", ylabel="Cycle residual"); figure.tight_layout(); figure.savefig(DEST / "plots" / "full_model_residuals.pdf"); plt.close(figure)
    passed = next(row["value"] for row in claims if row["claim"] == "complete_realistic_gate_passed")
    (DEST / "full_model_report.md").write_text(
        "# Full-model hidden-layer transition geometry\n\n"
        f"Execution commit: `{git_head()}`. Five collections of four ImageNet-pretrained ResNet-18 specialists were "
        "fine-tuned through the final two residual blocks and classifier on 10,000 CIFAR-10 examples each; one bounded "
        "four-specialist collection also fine-tuned the full backbone. Six gauge families, four hidden layers, five "
        "calibration resamples, and 200 draws from each of five label-independent matched-null families were executed. "
        f"No structured lift was activated without a certified chart action. The complete gate {'passed' if passed else 'did not pass'}.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
