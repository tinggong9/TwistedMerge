#!/usr/bin/env python3
"""Execute the preregistered natural model-lineage holonomy experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from torch import nn
from torchvision import __version__ as torchvision_version
from torchvision.datasets import CIFAR10
from torchvision.models import ResNet18_Weights, resnet18

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.holonomy_application_corpus import LowRankChartAdapter, classification_metrics
from src.lineage_merge_audit import (
    aligned_rank_bounded_merge,
    binary_prediction_metrics,
    double_holdout_logistic,
    double_holdout_ridge,
    harmful_merge_label,
    logits_for_domains,
    raw_parameter_average,
    score_domain_logits,
    seed_bootstrap_interval,
    validation_selected_interpolation,
)
from src.lineage_transport_sync import (
    TRANSITION_METHODS,
    bootstrap_loop_products,
    bootstrap_transition_instability,
    commutator_distance,
    fit_transition,
    inverse_consistency,
    loop_product,
    loop_statistics,
    normalized_residual,
    select_transition,
    synchronize_frames,
)
from src.model_lineage_holonomy import (
    TASKS,
    adapt_on_task,
    adjacent_swap_pairs,
    apply_task_corruption,
    branch_pairs,
    deterministic_split_indices,
    evaluate_domains,
    feature_discrepancy,
    lineage_edges,
    lineage_nodes,
    order_comparison_pairs,
    order_sensitivity_score,
    parameter_distance,
    prediction_disagreement,
    representations,
    state_bytes,
    state_dict_sha256,
    state_parameter_count,
    two_task_squares,
)


REPORT_ROOT = ROOT / "reports" / "model_lineage_holonomy"
ARTIFACT_ROOT = ROOT / "reports" / "tmp" / "model_lineage_holonomy"
SOURCE_CORPUS_ROOT = Path(
    "/Users/tinggong/Documents/Codex/2026-07-19/holonomy-applications/work/"
    "TwistedMerge-holonomy-applications/reports/tmp/holonomy_applications/"
    "shared_corpus_confirmatory"
)
SOURCE_FEATURE_CACHE = SOURCE_CORPUS_ROOT / "projected_features.pt"

FROZEN_CONFIG = {
    "schema_version": 1,
    "evidence_label": "natural_model_lineage",
    "source_commit": "9c91bc707d1f44beb36fe0fdce43af9ce1be79ed",
    "split_seed": 7192026,
    "pilot_seeds": [0, 1, 2],
    "extension_seeds": [3, 4],
    "tasks": {
        "A": {"kind": "gaussian_noise", "sigma": 0.15, "identity_seed_base": 42101},
        "B": {"kind": "gaussian_blur", "kernel": 5, "sigma": 1.0},
        "C": {
            "kind": "color_contrast",
            "contrast": 1.35,
            "rgb_scale": [1.10, 0.90, 1.00],
            "rgb_offset": [0.04, -0.02, 0.00],
        },
    },
    "split_sizes": {
        "adaptation_train": 2500,
        "transport_fit": 384,
        "transport_validation": 384,
        "transport_test": 384,
        "model_validation": 512,
        "application_test": 1000,
    },
    "adaptation": {
        "epochs_per_edge": 12,
        "batch_size": 256,
        "learning_rate": 0.003,
        "weight_decay": 0.0001,
        "rank": 4,
        "feature_dim": 64,
    },
    "representation_layers": {"early": 32, "mid": 32, "late": 64, "adapter": 64, "penultimate": 64},
    "transition_methods": list(TRANSITION_METHODS),
    "transition_bootstrap_samples": 100,
    "seed_bootstrap_samples": 2000,
    "stable_loop": {
        "identity_ci_low": 0.05,
        "commutator_ci_low": 0.03,
        "maximum_ci_width": 0.15,
        "maximum_edge_validation_residual": 0.35,
        "maximum_condition_number": 1e4,
    },
    "cycle_policy": {
        "correction_sync_residual": 0.20,
        "abstention_sync_residual": 0.35,
        "abstention_loop_ci_width": 0.35,
    },
    "harmful_merge": {"mean_margin": 0.01, "worst_domain_margin": 0.02},
    "interpolation_grid": [round(value, 1) for value in np.linspace(0.0, 1.0, 11)],
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def git_output(*arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], cwd=ROOT, text=True).strip()


def device_for(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def peak_memory_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def raw_images(dataset: CIFAR10, indices: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.asarray(dataset.data)[indices]).permute(0, 3, 1, 2).float().div_(255.0)


def encoder_inputs(images: torch.Tensor, weights: ResNet18_Weights) -> torch.Tensor:
    resized = nn.functional.interpolate(images, size=(64, 64), mode="bilinear", align_corners=False)
    mean = torch.tensor(weights.transforms().mean).view(1, 3, 1, 1)
    std = torch.tensor(weights.transforms().std).view(1, 3, 1, 1)
    return (resized - mean) / std


def deterministic_projection(input_dim: int, output_dim: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    matrix = torch.randn(input_dim, output_dim, generator=generator, dtype=torch.float64)
    q, _r = torch.linalg.qr(matrix, mode="reduced")
    return q.float()


def extract_encoder_layers(
    encoder: nn.Module,
    images: torch.Tensor,
    weights: ResNet18_Weights,
    batch_size: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    captured: dict[str, torch.Tensor] = {}

    def capture(name: str):
        def hook(_module, _inputs, output):
            captured[name] = nn.functional.adaptive_avg_pool2d(output, 1).flatten(1)

        return hook

    handles = [encoder.layer1.register_forward_hook(capture("early")), encoder.layer3.register_forward_hook(capture("mid"))]
    parts: dict[str, list[torch.Tensor]] = defaultdict(list)
    encoder.eval().to(device)
    with torch.no_grad():
        for batch in images.split(batch_size):
            late = encoder(encoder_inputs(batch, weights).to(device))
            parts["early"].append(captured["early"].cpu())
            parts["mid"].append(captured["mid"].cpu())
            parts["late_raw"].append(late.cpu())
    for handle in handles:
        handle.remove()
    encoder.cpu()
    return {name: torch.cat(values) for name, values in parts.items()}


def build_feature_cache(
    data_dir: Path,
    cache_path: Path,
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[dict, dict[str, object]]:
    if cache_path.is_file():
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        if payload.get("config_sha256") != hashlib.sha256(
            json.dumps(FROZEN_CONFIG, sort_keys=True).encode("utf-8")
        ).hexdigest():
            raise RuntimeError("existing feature cache does not match the frozen configuration")
        return payload, {"feature_cache_reused": True, "feature_extraction_seconds": 0.0}

    source_payload = torch.load(SOURCE_FEATURE_CACHE, map_location="cpu", weights_only=False)
    source_projection = source_payload["projection"]
    train_dataset = CIFAR10(data_dir, train=True, download=False)
    test_dataset = CIFAR10(data_dir, train=False, download=False)
    splits = deterministic_split_indices(len(train_dataset), len(test_dataset))
    weights = ResNet18_Weights.IMAGENET1K_V1
    encoder = resnet18(weights=weights)
    encoder.fc = nn.Identity()
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    early_projection = deterministic_projection(64, 32, 91001)
    mid_projection = deterministic_projection(256, 32, 91002)
    features: dict[str, dict[str, dict[str, torch.Tensor]]] = {}
    started = time.perf_counter()
    for split_name, indices in splits.items():
        dataset = test_dataset if split_name == "application_test" else train_dataset
        base_images = raw_images(dataset, indices)
        features[split_name] = {}
        for task in TASKS:
            corrupted = apply_task_corruption(base_images, task, indices)
            raw = extract_encoder_layers(encoder, corrupted, weights, batch_size, device)
            features[split_name][task] = {
                "early": raw["early"] @ early_projection,
                "mid": raw["mid"] @ mid_projection,
                "late": (
                    (raw["late_raw"] - source_projection["mean"])
                    @ source_projection["projection"]
                    / source_projection["scale"]
                ).float(),
            }

    for layer in ("early", "mid"):
        training = torch.cat([features["adaptation_train"][task][layer] for task in TASKS])
        mean = training.mean(0)
        scale = training.std(0).clamp_min(1e-6)
        for split_values in features.values():
            for task_values in split_values.values():
                task_values[layer] = ((task_values[layer] - mean) / scale).float()

    config_hash = hashlib.sha256(json.dumps(FROZEN_CONFIG, sort_keys=True).encode("utf-8")).hexdigest()
    payload = {
        "schema_version": 1,
        "config_sha256": config_hash,
        "features": features,
        "splits": {name: torch.from_numpy(values) for name, values in splits.items()},
        "source_projection": source_projection,
        "source_feature_cache": str(SOURCE_FEATURE_CACHE),
        "source_feature_cache_sha256": sha256_file(SOURCE_FEATURE_CACHE),
        "early_projection": early_projection,
        "mid_projection": mid_projection,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache_path)
    return payload, {
        "feature_cache_reused": False,
        "feature_extraction_seconds": float(time.perf_counter() - started),
    }


def load_m0(seed: int) -> tuple[LowRankChartAdapter, Path, str]:
    path = SOURCE_CORPUS_ROOT / f"adapter_seed_{seed}.pt"
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    model = LowRankChartAdapter(int(bundle["feature_dim"]), int(bundle["rank"]))
    model.load_state_dict(bundle["states"]["0"])
    model.eval()
    return model, path, sha256_file(path)


def relevant_edges() -> tuple[tuple[str, str], ...]:
    edges = list(lineage_edges(directed_both_ways=True))
    for loop in two_task_squares().values():
        for edge in zip(loop[:-1], loop[1:], strict=True):
            edges.extend((edge, (edge[1], edge[0])))
    for left, right in (*adjacent_swap_pairs(), ("AB", "BA"), ("AC", "CA"), ("BC", "CB")):
        edges.extend(((f"M_{left}", f"M_{right}"), (f"M_{right}", f"M_{left}")))
    for left, right in branch_pairs():
        edges.extend(((f"M_{left}", f"M_{right}"), (f"M_{right}", f"M_{left}")))
    return tuple(dict.fromkeys(edges))


def loop_definitions() -> dict[str, tuple[str, ...]]:
    loops = dict(two_task_squares())
    for left, right in adjacent_swap_pairs():
        loops[f"swap_{left}_{right}"] = (
            "M0",
            f"M_{left[0]}",
            f"M_{left[:2]}",
            f"M_{left}",
            f"M_{right}",
            f"M_{right[:2]}",
            f"M_{right[0]}",
            "M0",
        )
    return loops


def stack_anchor(
    feature_cache: Mapping[str, Mapping[str, Mapping[str, torch.Tensor]]],
    split: str,
    task_layer_values: Mapping[str, Mapping[str, torch.Tensor]],
    layer: str,
) -> np.ndarray:
    if layer in {"early", "mid", "late"}:
        return torch.cat([feature_cache[split][task][layer] for task in TASKS]).numpy()
    return torch.cat([task_layer_values[task][layer] for task in TASKS]).numpy()


def prepare_seed(
    seed: int,
    feature_payload: dict,
    train_dataset: CIFAR10,
    output_dir: Path,
    artifact_dir: Path,
) -> dict[str, object]:
    features = feature_payload["features"]
    splits = {name: value.numpy() for name, value in feature_payload["splits"].items()}
    labels_all = torch.tensor(np.asarray(train_dataset.targets), dtype=torch.long)
    train_labels = labels_all[torch.from_numpy(splits["adaptation_train"])]
    validation_labels = labels_all[torch.from_numpy(splits["model_validation"])]
    models: dict[str, LowRankChartAdapter] = {}
    validation_metrics: dict[str, dict[str, dict[str, float]]] = {}
    test_logits: dict[str, dict[str, torch.Tensor]] = {}
    node_representations: dict[str, dict[str, dict[str, np.ndarray]]] = {
        split: {} for split in ("transport_fit", "transport_validation", "transport_test")
    }
    lineage_rows: list[dict[str, object]] = []
    checkpoint_rows: list[dict[str, object]] = []
    artifact_rows: list[dict[str, object]] = []
    seed_artifact = artifact_dir / f"seed_{seed}"
    seed_artifact.mkdir(parents=True, exist_ok=True)
    m0, source_path, source_hash = load_m0(seed)
    models["M0"] = m0

    for node in lineage_nodes():
        if node.name != "M0":
            parent = models[node.parent]
            result = adapt_on_task(
                parent,
                features["adaptation_train"][node.appended_task]["late"],
                train_labels,
                independent_seed=seed,
                task=node.appended_task,
                epochs=12,
                batch_size=256,
                learning_rate=0.003,
                weight_decay=0.0001,
            )
            models[node.name] = result.model
            edge_seconds = result.wall_seconds
            optimizer_steps = result.optimizer_steps
            final_training_loss = result.final_training_loss
        else:
            edge_seconds = 0.0
            optimizer_steps = 0
            final_training_loss = float("nan")
        model = models[node.name]
        validation_metrics[node.name] = evaluate_domains(
            model,
            {task: features["model_validation"][task]["late"] for task in TASKS},
            validation_labels,
        )
        checkpoint_path = seed_artifact / f"{node.name}_checkpoint.pt"
        torch.save(
            {
                "schema_version": 1,
                "seed": seed,
                "node": node.name,
                "order": node.order,
                "parent": node.parent,
                "appended_task": node.appended_task,
                "state": model.state_dict(),
            },
            checkpoint_path,
        )
        checkpoint_hash = sha256_file(checkpoint_path)
        test_logits[node.name] = logits_for_domains(
            model, {task: features["application_test"][task]["late"] for task in TASKS}
        )
        logits_path = seed_artifact / f"{node.name}_test_logits.npz"
        np.savez_compressed(logits_path, **{task: value.numpy() for task, value in test_logits[node.name].items()})
        logits_hash = sha256_file(logits_path)

        representation_payload: dict[str, np.ndarray] = {}
        for split in node_representations:
            task_values = {}
            for task in TASKS:
                late = features[split][task]["late"]
                learned = representations(model, late)
                task_values[task] = learned
                for layer in ("early", "mid", "late"):
                    representation_payload[f"{split}_{task}_{layer}"] = features[split][task][layer].numpy()
                for layer in ("adapter", "penultimate"):
                    representation_payload[f"{split}_{task}_{layer}"] = learned[layer].numpy()
            node_representations[split][node.name] = {
                layer: stack_anchor(features, split, task_values, layer)
                for layer in ("early", "mid", "late", "adapter", "penultimate")
            }
        representation_path = seed_artifact / f"{node.name}_representations.npz"
        np.savez_compressed(representation_path, **representation_payload)
        representation_hash = sha256_file(representation_path)
        trainable, model_total = state_parameter_count(model)
        first_task = node.order[0] if node.order else ""
        first_task_peak = (
            validation_metrics[f"M_{first_task}"][first_task]["accuracy"] if first_task else float("nan")
        )
        first_task_current = (
            validation_metrics[node.name][first_task]["accuracy"] if first_task else float("nan")
        )
        forgetting = first_task_peak - first_task_current if first_task else float("nan")
        metrics = validation_metrics[node.name]
        lineage_rows.append(
            {
                "evidence_label": "natural_model_lineage",
                "mode": "pending",
                "seed": seed,
                "node": node.name,
                "task_order": node.order,
                "parent": node.parent or "",
                "appended_task": node.appended_task or "",
                "depth": node.depth,
                "training_examples_per_edge": 2500 if node.depth else 0,
                "epochs_per_edge": 12 if node.depth else 0,
                "optimizer_steps_this_edge": optimizer_steps,
                "edge_training_seconds": edge_seconds,
                "final_training_loss": final_training_loss,
                "validation_mean_accuracy": float(np.mean([metrics[task]["accuracy"] for task in TASKS])),
                "validation_worst_accuracy": float(np.min([metrics[task]["accuracy"] for task in TASKS])),
                **{f"validation_{task.lower()}_accuracy": metrics[task]["accuracy"] for task in TASKS},
                "first_task_forgetting": forgetting,
                "state_sha256": state_dict_sha256(model),
                "checkpoint_sha256": checkpoint_hash,
                "trainable_parameters": trainable,
                "adapter_total_parameters": model_total,
                "frozen_encoder_parameters": 11176512,
                "stored_bytes": state_bytes(model),
                "source_m0_path": str(source_path) if node.name == "M0" else "",
                "source_m0_sha256": source_hash if node.name == "M0" else "",
            }
        )
        checkpoint_rows.append(
            {
                "seed": seed,
                "node": node.name,
                "parent": node.parent or "",
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_sha256": checkpoint_hash,
                "checkpoint_bytes": checkpoint_path.stat().st_size,
                "representations_path": str(representation_path),
                "representations_sha256": representation_hash,
                "representations_bytes": representation_path.stat().st_size,
                "test_logits_path": str(logits_path),
                "test_logits_sha256": logits_hash,
                "test_logits_bytes": logits_path.stat().st_size,
                "test_logits_hashed_before_labels": True,
            }
        )
        artifact_rows.extend(
            [
                artifact_row("checkpoint", checkpoint_path, seed, node.name),
                artifact_row("representations", representation_path, seed, node.name),
                artifact_row("test_logits_before_labels", logits_path, seed, node.name),
            ]
        )
    return {
        "seed": seed,
        "models": models,
        "validation_metrics": validation_metrics,
        "test_logits": test_logits,
        "representations": node_representations,
        "lineage_rows": lineage_rows,
        "checkpoint_rows": checkpoint_rows,
        "artifact_rows": artifact_rows,
        "source_hash": source_hash,
        "test_labels_loaded": False,
    }


def artifact_row(kind: str, path: Path, seed: int | str, node: str = "") -> dict[str, object]:
    return {
        "artifact_kind": kind,
        "seed": seed,
        "node_or_family": node,
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def analyze_transports(context: dict[str, object], artifact_dir: Path) -> dict[str, list[dict[str, object]]]:
    seed = int(context["seed"])
    representations_by_split = context["representations"]
    edges = relevant_edges()
    layers = ("early", "mid", "late", "adapter", "penultimate")
    selected: dict[tuple[str, tuple[str, str]], object] = {}
    selected_validation: dict[tuple[str, tuple[str, str]], float] = {}
    selected_test: dict[tuple[str, tuple[str, str]], float] = {}
    pairwise_rows: list[dict[str, object]] = []
    transition_rows: list[dict[str, object]] = []
    matrices: dict[str, np.ndarray] = {}
    frozen_cache = {}
    for layer in layers:
        for edge_index, edge in enumerate(edges):
            source, target = edge
            fit_source = representations_by_split["transport_fit"][source][layer]
            fit_target = representations_by_split["transport_fit"][target][layer]
            validation_source = representations_by_split["transport_validation"][source][layer]
            validation_target = representations_by_split["transport_validation"][target][layer]
            if layer in {"early", "mid", "late"} and layer in frozen_cache:
                selected_fit, fits, validation, instability = frozen_cache[layer]
            else:
                selected_fit, fits, validation = select_transition(
                    fit_source, fit_target, validation_source, validation_target
                )
                instability = bootstrap_transition_instability(
                    fit_source,
                    fit_target,
                    selected_fit.method,
                    selected_fit.matrix,
                    samples=100,
                    seed=930000 + seed * 10000 + edge_index * 10 + layers.index(layer),
                )
                if layer in {"early", "mid", "late"}:
                    frozen_cache[layer] = selected_fit, fits, validation, instability
            selected[(layer, edge)] = selected_fit
            selected_validation[(layer, edge)] = float(validation[selected_fit.method])
            selected_test[(layer, edge)] = normalized_residual(
                representations_by_split["transport_test"][source][layer],
                representations_by_split["transport_test"][target][layer],
                selected_fit.matrix,
            )
            matrix_key = f"{layer}__{source}__{target}"
            matrices[matrix_key] = selected_fit.matrix.astype(np.float32)
            for method, fit in fits.items():
                pairwise_rows.append(
                    {
                        "seed": seed,
                        "layer": layer,
                        "source": source,
                        "target": target,
                        "path_relation": path_relation(source, target),
                        "method": method,
                        "selected": method == selected_fit.method,
                        "fit_residual": fit.fit_residual,
                        "transport_validation_residual": validation[method],
                        "transport_test_residual": normalized_residual(
                            representations_by_split["transport_test"][source][layer],
                            representations_by_split["transport_test"][target][layer],
                            fit.matrix,
                        ),
                        "condition_number": fit.condition_number,
                        "effective_rank": fit.effective_rank,
                        "singular_value_spread": fit.singular_value_spread,
                        "inverse_consistency": float("nan"),
                        "bootstrap_instability_mean": instability[0] if method == selected_fit.method else float("nan"),
                        "bootstrap_instability_ci_low": instability[1] if method == selected_fit.method else float("nan"),
                        "bootstrap_instability_ci_high": instability[2] if method == selected_fit.method else float("nan"),
                    }
                )
            transition_rows.append(
                {
                    "seed": seed,
                    "layer": layer,
                    "source": source,
                    "target": target,
                    "path_relation": path_relation(source, target),
                    "selected_method": selected_fit.method,
                    "matrix_artifact_key": matrix_key,
                    "fit_residual": selected_fit.fit_residual,
                    "transport_validation_residual": selected_validation[(layer, edge)],
                    "transport_test_residual": selected_test[(layer, edge)],
                    "condition_number": selected_fit.condition_number,
                    "effective_rank": selected_fit.effective_rank,
                    "singular_value_spread": selected_fit.singular_value_spread,
                    "inverse_consistency": float("nan"),
                    "bootstrap_instability_mean": instability[0],
                    "bootstrap_instability_ci_low": instability[1],
                    "bootstrap_instability_ci_high": instability[2],
                }
            )
    transition_lookup = {(row["layer"], row["source"], row["target"]): row for row in transition_rows}
    for layer in layers:
        for source, target in edges:
            reverse = selected.get((layer, (target, source)))
            if reverse is None:
                continue
            value = inverse_consistency(selected[(layer, (source, target))].matrix, reverse.matrix)
            transition_lookup[(layer, source, target)]["inverse_consistency"] = value
            for row in pairwise_rows:
                if row["layer"] == layer and row["source"] == source and row["target"] == target and row["selected"]:
                    row["inverse_consistency"] = value
                    break
    matrix_path = artifact_dir / f"transport_maps_seed_{seed}.npz"
    np.savez_compressed(matrix_path, **matrices)
    matrix_hash = sha256_file(matrix_path)
    for row in transition_rows:
        row["matrix_artifact_path"] = str(matrix_path)
        row["matrix_artifact_sha256"] = matrix_hash

    loop_rows: list[dict[str, object]] = []
    commutator_rows: list[dict[str, object]] = []
    loop_products_by_layer: dict[tuple[str, str], np.ndarray] = {}
    loop_bootstrap_products: dict[tuple[str, str], np.ndarray] = {}
    for layer_index, layer in enumerate(layers):
        for loop_index, (loop_name, loop) in enumerate(loop_definitions().items()):
            loop_edges = tuple(zip(loop[:-1], loop[1:], strict=True))
            transitions = {edge: selected[(layer, edge)].matrix for edge in loop_edges}
            methods = {edge: selected[(layer, edge)].method for edge in loop_edges}
            product = loop_product(transitions, loop)
            if layer in {"early", "mid", "late"}:
                test_products = np.repeat(product[None, :, :], 100, axis=0)
                validation_products = test_products.copy()
            else:
                test_products = bootstrap_loop_products(
                    {node: representations_by_split["transport_test"][node][layer] for node in set(loop)},
                    loop,
                    methods,
                    samples=100,
                    seed=950000 + seed * 10000 + layer_index * 100 + loop_index,
                )
                validation_products = bootstrap_loop_products(
                    {node: representations_by_split["transport_validation"][node][layer] for node in set(loop)},
                    loop,
                    methods,
                    samples=100,
                    seed=960000 + seed * 10000 + layer_index * 100 + loop_index,
                )
            test_distances = np.asarray([loop_statistics(value)["identity_distance"] for value in test_products])
            validation_distances = np.asarray([loop_statistics(value)["identity_distance"] for value in validation_products])
            stats = loop_statistics(product)
            max_validation = max(selected_validation[(layer, edge)] for edge in loop_edges)
            max_condition = max(selected[(layer, edge)].condition_number for edge in loop_edges)
            test_low, test_high = np.quantile(test_distances, (0.025, 0.975))
            validation_low, validation_high = np.quantile(validation_distances, (0.025, 0.975))
            stable = bool(
                test_low > 0.05
                and test_high - test_low <= 0.15
                and max_validation <= 0.35
                and max_condition <= 1e4
            )
            loop_rows.append(
                {
                    "seed": seed,
                    "layer": layer,
                    "loop_name": loop_name,
                    "loop_vertices": "->".join(loop),
                    **stats,
                    "transport_test_bootstrap_mean": float(test_distances.mean()),
                    "transport_test_bootstrap_ci_low": float(test_low),
                    "transport_test_bootstrap_ci_high": float(test_high),
                    "transport_validation_bootstrap_mean": float(validation_distances.mean()),
                    "transport_validation_bootstrap_ci_low": float(validation_low),
                    "transport_validation_bootstrap_ci_high": float(validation_high),
                    "transport_validation_stability_width": float(validation_high - validation_low),
                    "maximum_edge_validation_residual": max_validation,
                    "maximum_edge_condition_number": max_condition,
                    "stable_nonidentity": stable,
                    "centrality_taxonomy_only": float(
                        np.linalg.norm(product - np.trace(product) / product.shape[0] * np.eye(product.shape[0]))
                        / max(np.linalg.norm(np.eye(product.shape[0])), 1e-12)
                    ),
                    "brauer_class_claimed": False,
                }
            )
            loop_products_by_layer[(layer, loop_name)] = product
            loop_bootstrap_products[(layer, loop_name)] = test_products
        names = list(loop_definitions())
        layer_loop_rows = {
            row["loop_name"]: row
            for row in loop_rows
            if row["seed"] == seed and row["layer"] == layer
        }
        for left_index, left_name in enumerate(names):
            for right_name in names[left_index + 1 :]:
                left_product = loop_products_by_layer[(layer, left_name)]
                right_product = loop_products_by_layer[(layer, right_name)]
                bootstrap_values = np.asarray(
                    [
                        commutator_distance(left, right)
                        for left, right in zip(
                            loop_bootstrap_products[(layer, left_name)],
                            loop_bootstrap_products[(layer, right_name)],
                            strict=True,
                        )
                    ]
                )
                low, high = np.quantile(bootstrap_values, (0.025, 0.975))
                left_quality = layer_loop_rows[left_name]
                right_quality = layer_loop_rows[right_name]
                edge_quality_passed = bool(
                    max(
                        left_quality["maximum_edge_validation_residual"],
                        right_quality["maximum_edge_validation_residual"],
                    )
                    <= 0.35
                    and max(
                        left_quality["maximum_edge_condition_number"],
                        right_quality["maximum_edge_condition_number"],
                    )
                    <= 1e4
                    and max(
                        left_quality["transport_test_bootstrap_ci_high"]
                        - left_quality["transport_test_bootstrap_ci_low"],
                        right_quality["transport_test_bootstrap_ci_high"]
                        - right_quality["transport_test_bootstrap_ci_low"],
                    )
                    <= 0.15
                )
                commutator_rows.append(
                    {
                        "seed": seed,
                        "layer": layer,
                        "left_loop": left_name,
                        "right_loop": right_name,
                        "commutator_distance": commutator_distance(left_product, right_product),
                        "bootstrap_mean": float(bootstrap_values.mean()),
                        "bootstrap_ci_low": float(low),
                        "bootstrap_ci_high": float(high),
                        "edge_quality_passed": edge_quality_passed,
                        "stable_noncommuting": bool(low > 0.03 and edge_quality_passed),
                    }
                )
    return {
        "pairwise_rows": pairwise_rows,
        "transition_rows": transition_rows,
        "loop_rows": loop_rows,
        "commutator_rows": commutator_rows,
        "selected": selected,
        "selected_validation": selected_validation,
        "selected_test": selected_test,
        "transport_artifact": artifact_row("transport_matrix_bundle", matrix_path, seed),
    }


def path_relation(source: str, target: str) -> str:
    node_map = {node.name: node for node in lineage_nodes()}
    if node_map.get(target) and node_map[target].parent == source:
        return "forward_lineage"
    if node_map.get(source) and node_map[source].parent == target:
        return "reverse_lineage"
    if source.startswith("M_") and target.startswith("M_") and len(source) == len(target):
        return "same_depth_closing"
    return "loop_closing"


def mean_validation_accuracy(model: LowRankChartAdapter, features: dict, labels: torch.Tensor) -> float:
    metrics = evaluate_domains(model, {task: features["model_validation"][task]["late"] for task in TASKS}, labels)
    return float(np.mean([metrics[task]["accuracy"] for task in TASKS]))


def merge_context(
    context: dict[str, object],
    transport: dict[str, object],
    feature_payload: dict,
    train_dataset: CIFAR10,
    artifact_dir: Path,
) -> dict[str, object]:
    seed = int(context["seed"])
    models = context["models"]
    features = feature_payload["features"]
    splits = {name: value.numpy() for name, value in feature_payload["splits"].items()}
    validation_labels = torch.tensor(np.asarray(train_dataset.targets), dtype=torch.long)[
        torch.from_numpy(splits["model_validation"])
    ]
    selected = transport["selected"]
    selected_validation = transport["selected_validation"]
    loop_lookup = {(row["layer"], row["loop_name"]): row for row in transport["loop_rows"]}
    merge_bundles = {}
    capacity_rows = []
    for left_task, right_task in branch_pairs():
        family = left_task + right_task
        left_name, right_name = f"M_{left_task}", f"M_{right_task}"
        left, right = models[left_name], models[right_name]
        timings = {}

        def timed(name: str, function):
            started = time.perf_counter()
            value = function()
            timings[name] = time.perf_counter() - started
            return value

        raw = timed("raw_parameter_average", lambda: raw_parameter_average((left, right)))
        pair_fit = selected[("penultimate", (left_name, right_name))]
        pairwise = timed(
            "pairwise_reference_alignment",
            lambda: aligned_rank_bounded_merge((left, right), (pair_fit.matrix.T, np.eye(64))),
        )
        ordinary_nodes = ("M0", left_name, right_name)
        ordinary_transitions = {
            edge: selected[("penultimate", edge)].matrix
            for edge in relevant_edges()
            if edge[0] in ordinary_nodes and edge[1] in ordinary_nodes
        }
        sync_started = time.perf_counter()
        ordinary_sync = synchronize_frames(ordinary_transitions, ordinary_nodes)
        ordinary_sync_seconds = time.perf_counter() - sync_started
        ordinary = timed(
            "ordinary_global_synchronization",
            lambda: aligned_rank_bounded_merge(
                (left, right),
                (ordinary_sync.maps_to_common[left_name], ordinary_sync.maps_to_common[right_name]),
            ),
        )
        greedy, interpolation_weight, interpolation_evaluations = timed(
            "validation_selected_interpolation",
            lambda: validation_selected_interpolation(
                left,
                right,
                lambda model: mean_validation_accuracy(model, features, validation_labels),
                FROZEN_CONFIG["interpolation_grid"],
            ),
        )
        ordinary_candidates = {
            "raw_parameter_average": raw,
            "pairwise_reference_alignment": pairwise,
            "ordinary_global_synchronization": ordinary,
            "validation_selected_interpolation": greedy,
        }
        ordinary_validation = {
            name: mean_validation_accuracy(model, features, validation_labels)
            for name, model in ordinary_candidates.items()
        }
        fallback_name = max(ordinary_validation, key=lambda name: (ordinary_validation[name], name))
        fallback = ordinary_candidates[fallback_name]
        cycle_nodes = ("M0", left_name, right_name, f"M_{family}", f"M_{family[::-1]}")
        cycle_transitions = {
            edge: selected[("penultimate", edge)].matrix
            for edge in relevant_edges()
            if edge[0] in cycle_nodes and edge[1] in cycle_nodes
        }
        sync_started = time.perf_counter()
        cycle_sync = synchronize_frames(cycle_transitions, cycle_nodes)
        cycle_sync_seconds = time.perf_counter() - sync_started
        square = loop_lookup[("penultimate", f"{family}_square")]
        validation_low = float(square["transport_validation_bootstrap_ci_low"])
        validation_width = float(square["transport_validation_stability_width"])
        max_condition = float(square["maximum_edge_condition_number"])
        correction_allowed = bool(
            validation_low > 0.05
            and validation_width <= 0.15
            and float(square["maximum_edge_validation_residual"]) <= 0.35
            and max_condition <= 1e4
            and cycle_sync.connection_residual <= 0.20
        )
        abstain = bool(
            max_condition > 1e4
            or validation_width > 0.35
            or cycle_sync.connection_residual > 0.35
        )
        if correction_allowed:
            cycle_model = timed(
                "cycle_aware_synchronization",
                lambda: aligned_rank_bounded_merge(
                    (left, right),
                    (cycle_sync.maps_to_common[left_name], cycle_sync.maps_to_common[right_name]),
                ),
            )
            cycle_action = "correct"
        elif abstain:
            cycle_model = left if ordinary_validation["raw_parameter_average"] >= ordinary_validation[fallback_name] else fallback
            branch_scores = {
                left_name: mean_validation_accuracy(left, features, validation_labels),
                right_name: mean_validation_accuracy(right, features, validation_labels),
            }
            cycle_model = left if branch_scores[left_name] >= branch_scores[right_name] else right
            timings["cycle_aware_synchronization"] = 0.0
            cycle_action = "abstain"
        else:
            cycle_model = fallback
            timings["cycle_aware_synchronization"] = 0.0
            cycle_action = f"fallback:{fallback_name}"
        sequential_names = (f"M_{family}", f"M_{family[::-1]}")
        sequential_scores = {
            name: mean_validation_accuracy(models[name], features, validation_labels) for name in sequential_names
        }
        sequential_name = max(sequential_scores, key=lambda name: (sequential_scores[name], name))
        methods = {
            **ordinary_candidates,
            "cycle_aware_synchronization": cycle_model,
            "fallback_only_policy": fallback,
            "joint_sequential_adaptation_oracle": models[sequential_name],
        }
        method_logits = {
            name: logits_for_domains(
                model, {task: features["application_test"][task]["late"] for task in TASKS}
            )
            for name, model in methods.items()
        }
        method_logits["prediction_ensemble_upper_bound"] = {
            task: 0.5 * (context["test_logits"][left_name][task] + context["test_logits"][right_name][task])
            for task in TASKS
        }
        latency_features = features["model_validation"]["A"]["late"]
        inference_latencies = {}
        with torch.no_grad():
            for method, model in methods.items():
                started = time.perf_counter()
                for _ in range(3):
                    _ = model(latency_features)
                inference_latencies[method] = (time.perf_counter() - started) / 3.0
            started = time.perf_counter()
            for _ in range(3):
                _ = left(latency_features)
                _ = right(latency_features)
            inference_latencies["prediction_ensemble_upper_bound"] = (time.perf_counter() - started) / 3.0
        logits_path = artifact_dir / f"merge_logits_seed_{seed}_{family}.npz"
        np.savez_compressed(
            logits_path,
            **{
                f"{method}__{task}": logits.numpy()
                for method, values in method_logits.items()
                for task, logits in values.items()
            },
        )
        merge_bundles[family] = {
            "left_name": left_name,
            "right_name": right_name,
            "method_logits": method_logits,
            "models": methods,
            "cycle_action": cycle_action,
            "fallback_method": fallback_name,
            "sequential_oracle_node": sequential_name,
            "interpolation_weight": interpolation_weight,
            "interpolation_evaluations": interpolation_evaluations,
            "ordinary_sync_residual": ordinary_sync.connection_residual,
            "cycle_sync_residual": cycle_sync.connection_residual,
            "pairwise_validation_residual": float(
                np.mean(
                    [
                        selected_validation[("penultimate", (left_name, right_name))],
                        selected_validation[("penultimate", (right_name, left_name))],
                    ]
                )
            ),
            "pairwise_inverse_consistency": inverse_consistency(
                selected[("penultimate", (left_name, right_name))].matrix,
                selected[("penultimate", (right_name, left_name))].matrix,
            ),
            "loop_row": square,
            "logits_path": logits_path,
            "logits_hash": sha256_file(logits_path),
            "timings": timings,
        }
        process_peak = peak_memory_bytes()
        for method in (*methods, "prediction_ensemble_upper_bound"):
            branch_count = 2 if method == "prediction_ensemble_upper_bound" else 1
            validation_count = 11 if method == "validation_selected_interpolation" else 0
            if method == "fallback_only_policy":
                validation_count = 4
            if method == "joint_sequential_adaptation_oracle":
                validation_count = 2
            model = methods.get(method)
            trainable = 2 * 64 * 4 + 10 * 64 + 10 if model is not None else 2 * (2 * 64 * 4 + 10 * 64 + 10)
            stored = state_bytes(model) if model is not None else state_bytes(left) + state_bytes(right)
            capacity_rows.append(
                {
                    "seed": seed,
                    "branch_family": family,
                    "method": method,
                    "trainable_parameters": trainable,
                    "total_parameters_including_frozen_encoder": 11176512 + trainable,
                    "branch_count": branch_count,
                    "inference_multiplier": float(branch_count),
                    "stored_bytes": stored,
                    "transport_fit_cost_seconds": 0.0,
                    "synchronization_cost_seconds": (
                        cycle_sync_seconds if method == "cycle_aware_synchronization" else ordinary_sync_seconds if method == "ordinary_global_synchronization" else 0.0
                    ),
                    "merge_cost_seconds": timings.get(method, 0.0),
                    "peak_memory_bytes": process_peak,
                    "inference_latency_seconds": inference_latencies[method],
                    "validation_evaluation_count": validation_count,
                    "additional_training_control": method == "joint_sequential_adaptation_oracle",
                }
            )
    return {
        "bundles": merge_bundles,
        "capacity_rows": capacity_rows,
        "artifacts": [
            artifact_row("merge_logits_before_labels", value["logits_path"], seed, family)
            for family, value in merge_bundles.items()
        ],
    }


def score_context(
    context: dict[str, object],
    transport: dict[str, object],
    merges: dict[str, object],
    test_labels: torch.Tensor,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    seed = int(context["seed"])
    models = context["models"]
    checkpoint_scores = {
        node: score_domain_logits(logits, test_labels) for node, logits in context["test_logits"].items()
    }
    for row in context["lineage_rows"]:
        score = checkpoint_scores[row["node"]]
        row.update({f"test_{name}": value for name, value in score.items()})
        row["mode"] = "scored_after_logit_hash"
    loop_lookup = {(row["layer"], row["loop_name"]): row for row in transport["loop_rows"]}
    commutator_by_loop = defaultdict(float)
    for row in transport["commutator_rows"]:
        if row["layer"] == "penultimate":
            commutator_by_loop[row["left_loop"]] = max(commutator_by_loop[row["left_loop"]], row["commutator_distance"])
            commutator_by_loop[row["right_loop"]] = max(commutator_by_loop[row["right_loop"]], row["commutator_distance"])
    order_rows = []
    for left_order, right_order in order_comparison_pairs():
        left_name, right_name = f"M_{left_order}", f"M_{right_order}"
        left_score, right_score = checkpoint_scores[left_name], checkpoint_scores[right_name]
        left_first = checkpoint_scores[f"M_{left_order[0]}"][f"{left_order[0].lower()}_accuracy"]
        right_first = checkpoint_scores[f"M_{right_order[0]}"][f"{right_order[0].lower()}_accuracy"]
        left_forgetting = left_first - left_score[f"{left_order[0].lower()}_accuracy"]
        right_forgetting = right_first - right_score[f"{right_order[0].lower()}_accuracy"]
        left_logits = torch.cat([context["test_logits"][left_name][task] for task in TASKS])
        right_logits = torch.cat([context["test_logits"][right_name][task] for task in TASKS])
        left_features = context["representations"]["transport_test"][left_name]["penultimate"]
        right_features = context["representations"]["transport_test"][right_name]["penultimate"]
        checkpoint_gap = parameter_distance(models[left_name], models[right_name])
        disagreement = prediction_disagreement(left_logits, right_logits)
        feature_gap = feature_discrepancy(torch.from_numpy(left_features), torch.from_numpy(right_features))
        loop_name = f"{''.join(sorted(set(left_order)))}_square" if len(left_order) == 2 else f"swap_{left_order}_{right_order}"
        loop_row = loop_lookup[("penultimate", loop_name)]
        edge = (left_name, right_name)
        pair_fit = transport["selected"][("penultimate", edge)]
        reverse_fit = transport["selected"][("penultimate", (right_name, left_name))]
        pair_residual = float(
            np.mean(
                [
                    transport["selected_validation"][("penultimate", edge)],
                    transport["selected_validation"][("penultimate", (right_name, left_name))],
                ]
            )
        )
        score = order_sensitivity_score(
            mean_accuracy_delta=left_score["mean_accuracy"] - right_score["mean_accuracy"],
            worst_accuracy_delta=left_score["worst_domain_accuracy"] - right_score["worst_domain_accuracy"],
            forgetting_delta=left_forgetting - right_forgetting,
            disagreement=disagreement,
            feature_difference=feature_gap,
            ece_delta=left_score["mean_ece"] - right_score["mean_ece"],
            checkpoint_distance=checkpoint_gap,
        )
        order_rows.append(
            {
                "seed": seed,
                "order_family": f"{left_order}_vs_{right_order}",
                "loop_id": f"{seed}:{loop_name}",
                "left_node": left_name,
                "right_node": right_name,
                "left_mean_accuracy": left_score["mean_accuracy"],
                "right_mean_accuracy": right_score["mean_accuracy"],
                "absolute_mean_accuracy_delta": abs(left_score["mean_accuracy"] - right_score["mean_accuracy"]),
                "absolute_worst_accuracy_delta": abs(left_score["worst_domain_accuracy"] - right_score["worst_domain_accuracy"]),
                "left_first_task_forgetting": left_forgetting,
                "right_first_task_forgetting": right_forgetting,
                "absolute_forgetting_delta": abs(left_forgetting - right_forgetting),
                "prediction_disagreement": disagreement,
                "feature_discrepancy": feature_gap,
                "absolute_ece_delta": abs(left_score["mean_ece"] - right_score["mean_ece"]),
                "parameter_distance": checkpoint_gap,
                "pairwise_transport_residual": pair_residual,
                "inverse_consistency": inverse_consistency(pair_fit.matrix, reverse_fit.matrix),
                "loop_identity_distance": loop_row["identity_distance"],
                "loop_spectral_radius": loop_row["spectral_radius"],
                "loop_singular_spread": loop_row["singular_value_spread"],
                "loop_commutator_max": commutator_by_loop[loop_name],
                "loop_stable_nonidentity": loop_row["stable_nonidentity"],
                "order_sensitivity_score": score,
            }
        )
    merge_rows = []
    for family, bundle in merges["bundles"].items():
        left_score = checkpoint_scores[bundle["left_name"]]
        right_score = checkpoint_scores[bundle["right_name"]]
        scored = {
            method: score_domain_logits(values, test_labels)
            for method, values in bundle["method_logits"].items()
        }
        raw_score = scored["raw_parameter_average"]
        harmful = harmful_merge_label(
            raw_score["mean_accuracy"],
            raw_score["worst_domain_accuracy"],
            left_score["mean_accuracy"],
            right_score["mean_accuracy"],
            left_score["worst_domain_accuracy"],
            right_score["worst_domain_accuracy"],
        )
        best_deployable = max(
            score["mean_accuracy"]
            for method, score in scored.items()
            if method not in {"prediction_ensemble_upper_bound", "joint_sequential_adaptation_oracle"}
        )
        best_individual = max(left_score["mean_accuracy"], right_score["mean_accuracy"])
        left_logits = torch.cat([context["test_logits"][bundle["left_name"]][task] for task in TASKS])
        right_logits = torch.cat([context["test_logits"][bundle["right_name"]][task] for task in TASKS])
        loop_row = bundle["loop_row"]
        for method, score in scored.items():
            merge_rows.append(
                {
                    "seed": seed,
                    "branch_family": family,
                    "loop_id": f"{seed}:{family}_square",
                    "method": method,
                    **score,
                    "ordinary_raw_harmful": harmful,
                    "cycle_action": bundle["cycle_action"],
                    "fallback_method": bundle["fallback_method"],
                    "interpolation_weight": bundle["interpolation_weight"],
                    "validation_evaluation_count": bundle["interpolation_evaluations"] if method == "validation_selected_interpolation" else 0,
                    "parameter_distance": parameter_distance(models[bundle["left_name"]], models[bundle["right_name"]]),
                    "prediction_disagreement": prediction_disagreement(left_logits, right_logits),
                    "pairwise_transport_residual": bundle["pairwise_validation_residual"],
                    "inverse_consistency": bundle["pairwise_inverse_consistency"],
                    "loop_identity_distance": loop_row["identity_distance"],
                    "loop_spectral_radius": loop_row["spectral_radius"],
                    "loop_singular_spread": loop_row["singular_value_spread"],
                    "loop_stable_nonidentity": loop_row["stable_nonidentity"],
                    "ordinary_sync_residual": bundle["ordinary_sync_residual"],
                    "cycle_sync_residual": bundle["cycle_sync_residual"],
                    "regret_vs_best_deployable": best_deployable - score["mean_accuracy"],
                    "regret_if_ordinary_merge": best_deployable - raw_score["mean_accuracy"],
                    "regret_if_abstain": best_deployable - best_individual,
                    "additional_training_control": method == "joint_sequential_adaptation_oracle",
                    "nondeployable_or_upper_bound": method in {"prediction_ensemble_upper_bound", "joint_sequential_adaptation_oracle"},
                    "test_logits_hashed_before_labels": True,
                }
            )
    context["test_labels_loaded"] = True
    return order_rows, merge_rows


FEATURE_SETS = {
    "parameter_distance_only": ["parameter_distance"],
    "prediction_disagreement_only": ["prediction_disagreement"],
    "pairwise_transport_residuals_only": ["pairwise_transport_residual"],
    "pairwise_plus_inverse_consistency": ["pairwise_transport_residual", "inverse_consistency"],
    "loop_holonomy_features": ["loop_identity_distance", "loop_spectral_radius", "loop_singular_spread"],
    "pairwise_plus_holonomy": ["pairwise_transport_residual", "inverse_consistency", "loop_identity_distance", "loop_spectral_radius", "loop_singular_spread"],
    "all_features": ["parameter_distance", "prediction_disagreement", "pairwise_transport_residual", "inverse_consistency", "loop_identity_distance", "loop_spectral_radius", "loop_singular_spread"],
}


def prediction_analysis(order: pd.DataFrame, merges: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, bool]]:
    rows = []
    paired = []
    raw = merges[merges["method"] == "raw_parameter_average"].reset_index(drop=True)
    classification_predictions = {}
    regression_predictions = {}
    for feature_name, columns in FEATURE_SETS.items():
        probabilities, audit = double_holdout_logistic(
            raw[columns].to_numpy(),
            raw["ordinary_raw_harmful"].astype(int).to_numpy(),
            raw["seed"].to_numpy(),
            raw["branch_family"].to_numpy(),
            raw["loop_id"].to_numpy(),
        )
        classification_predictions[feature_name] = probabilities
        metrics = binary_prediction_metrics(raw["ordinary_raw_harmful"].astype(int), probabilities)
        action_regret = np.where(
            probabilities >= 0.5,
            raw["regret_if_abstain"].to_numpy(),
            raw["regret_if_ordinary_merge"].to_numpy(),
        )
        rows.append(
            {
                "prediction_task": "harmful_merge",
                "feature_set": feature_name,
                "rows": len(raw),
                "independent_seeds": raw["seed"].nunique(),
                "auroc": metrics.auroc,
                "auprc": metrics.auprc,
                "brier": metrics.brier,
                "ece": metrics.ece,
                "accuracy": metrics.accuracy,
                "harmful_merge_avoidance": metrics.harmful_avoidance,
                "merge_abstain_regret": float(np.mean(action_regret)),
                "spearman_order_sensitivity": float("nan"),
                "double_holdout_folds": len(audit),
                "all_seed_exclusions_passed": all(value["seed_excluded_from_train"] for value in audit),
                "all_family_exclusions_passed": all(value["family_excluded_from_train"] for value in audit),
                "all_loop_exclusions_passed": all(value["loop_excluded_from_train"] for value in audit),
            }
        )
        predictions, ridge_audit = double_holdout_ridge(
            order[columns].to_numpy(),
            order["order_sensitivity_score"].to_numpy(),
            order["seed"].to_numpy(),
            order["order_family"].to_numpy(),
            order["loop_id"].to_numpy(),
        )
        regression_predictions[feature_name] = predictions
        correlation = spearmanr(order["order_sensitivity_score"], predictions).statistic
        rows.append(
            {
                "prediction_task": "order_sensitivity",
                "feature_set": feature_name,
                "rows": len(order),
                "independent_seeds": order["seed"].nunique(),
                "auroc": float("nan"),
                "auprc": float("nan"),
                "brier": float(np.mean((predictions - order["order_sensitivity_score"].to_numpy()) ** 2)),
                "ece": float("nan"),
                "accuracy": float("nan"),
                "harmful_merge_avoidance": float("nan"),
                "merge_abstain_regret": float("nan"),
                "spearman_order_sensitivity": float(correlation),
                "double_holdout_folds": len(ridge_audit),
                "all_seed_exclusions_passed": all(value["seed_excluded_from_train"] for value in ridge_audit),
                "all_family_exclusions_passed": all(value["family_excluded_from_train"] for value in ridge_audit),
                "all_loop_exclusions_passed": all(value["loop_excluded_from_train"] for value in ridge_audit),
            }
        )

    pair_name = "pairwise_plus_inverse_consistency"
    holonomy_name = "pairwise_plus_holonomy"
    rng = np.random.default_rng(10072026)
    seed_values = sorted(raw["seed"].unique())
    auc_deltas, auprc_deltas = [], []
    target = raw["ordinary_raw_harmful"].astype(int).to_numpy()
    for _ in range(2000):
        sampled = rng.choice(seed_values, size=len(seed_values), replace=True)
        indices = np.concatenate([np.flatnonzero(raw["seed"].to_numpy() == seed) for seed in sampled])
        if len(np.unique(target[indices])) < 2:
            continue
        pair_metrics = binary_prediction_metrics(target[indices], classification_predictions[pair_name][indices])
        holo_metrics = binary_prediction_metrics(target[indices], classification_predictions[holonomy_name][indices])
        auc_deltas.append(holo_metrics.auroc - pair_metrics.auroc)
        auprc_deltas.append(holo_metrics.auprc - pair_metrics.auprc)
    for metric_name, values in (("auroc", auc_deltas), ("auprc", auprc_deltas)):
        values_array = np.asarray(values)
        paired.append(
            {
                "comparison": f"H2_{metric_name}_pairwise_plus_holonomy_minus_pairwise_only",
                "mean_delta": float(values_array.mean()) if len(values_array) else float("nan"),
                "ci_low": float(np.quantile(values_array, 0.025)) if len(values_array) else float("nan"),
                "ci_high": float(np.quantile(values_array, 0.975)) if len(values_array) else float("nan"),
                "independent_seeds": len(seed_values),
                "status": "evaluated" if len(values_array) else "not_estimable_single_outcome_class",
            }
        )

    controls = ["parameter_distance", "prediction_disagreement", "pairwise_transport_residual", "inverse_consistency"]
    holonomy = ["loop_identity_distance", "loop_spectral_radius", "loop_singular_spread"]
    coefficient_values = []
    order_seeds = sorted(order["seed"].unique())
    for _ in range(2000):
        sampled = rng.choice(order_seeds, size=len(order_seeds), replace=True)
        indices = np.concatenate([np.flatnonzero(order["seed"].to_numpy() == seed) for seed in sampled])
        x = order[controls + holonomy].to_numpy()[indices]
        y = order["order_sensitivity_score"].to_numpy()[indices]
        scaler = StandardScaler().fit(x)
        model = Ridge(alpha=1.0).fit(scaler.transform(x), y)
        coefficient_values.append(float(model.coef_[len(controls)]))
    coefficient_array = np.asarray(coefficient_values)
    paired.append(
        {
            "comparison": "H1_incremental_loop_identity_coefficient_controlling_pairwise_drift",
            "mean_delta": float(coefficient_array.mean()),
            "ci_low": float(np.quantile(coefficient_array, 0.025)),
            "ci_high": float(np.quantile(coefficient_array, 0.975)),
            "independent_seeds": len(order_seeds),
            "status": "evaluated",
        }
    )

    ordinary = merges[merges["method"] == "ordinary_global_synchronization"]
    cycle = merges[merges["method"] == "cycle_aware_synchronization"]
    joined = ordinary.merge(cycle, on=["seed", "branch_family"], suffixes=("_ordinary", "_cycle"))
    corrected = joined[joined["cycle_action_cycle"] == "correct"]
    deltas_by_seed = (
        corrected.groupby("seed").apply(
            lambda frame: float((frame["mean_accuracy_cycle"] - frame["mean_accuracy_ordinary"]).mean()),
            include_groups=False,
        ).to_dict()
        if len(corrected)
        else {}
    )
    h3_mean, h3_low, h3_high = (
        seed_bootstrap_interval(deltas_by_seed, samples=2000, seed=2003)
        if deltas_by_seed
        else (float("nan"), float("nan"), float("nan"))
    )
    paired.append(
        {
            "comparison": "H3_cycle_aware_minus_ordinary_sync_mean_accuracy",
            "mean_delta": h3_mean,
            "ci_low": h3_low,
            "ci_high": h3_high,
            "independent_seeds": len(deltas_by_seed),
            "status": "evaluated_corrections" if deltas_by_seed else "not_estimable_no_cycle_corrections",
        }
    )
    h1 = paired[-2]["ci_low"] > 0 or paired[-2]["ci_high"] < 0
    h2_rows = paired[:2]
    h2 = any(row["ci_low"] > 0 for row in h2_rows)
    h3 = bool(len(deltas_by_seed) >= 2 and h3_low > 0)
    raw_rows = merges[merges["method"] == "raw_parameter_average"]
    cycle_rows = merges[merges["method"] == "cycle_aware_synchronization"]
    h4_join = raw_rows.merge(
        cycle_rows[["seed", "branch_family", "mean_accuracy", "cycle_action"]],
        on=["seed", "branch_family"],
        suffixes=("_raw", "_cycle"),
    )
    h4_qualifying = h4_join[
        (h4_join["pairwise_transport_residual"] <= 0.35)
        & h4_join["loop_stable_nonidentity"].astype(bool)
        & h4_join["ordinary_raw_harmful"].astype(bool)
        & (
            (
                (h4_join["cycle_action_cycle"] == "correct")
                & (h4_join["mean_accuracy_cycle"] > h4_join["mean_accuracy_raw"])
            )
            | h4_join["cycle_action_cycle"].str.startswith("abstain")
        )
    ]
    h4 = h4_qualifying["seed"].nunique() >= 2
    paired.append(
        {
            "comparison": "H4_repeated_conflict_detection",
            "mean_delta": float(len(h4_qualifying)),
            "ci_low": float(h4_qualifying["seed"].nunique()),
            "ci_high": float(raw_rows["seed"].nunique()),
            "independent_seeds": raw_rows["seed"].nunique(),
            "status": "evaluated",
        }
    )
    return pd.DataFrame(rows), pd.DataFrame(paired), {"H1": bool(h1), "H2": bool(h2), "H3": bool(h3), "H4": bool(h4)}


def aggregate_existing(output_dir: Path, name: str, new_rows: list[dict[str, object]], include_pilot: bool) -> pd.DataFrame:
    current = pd.DataFrame(new_rows)
    if include_pilot:
        pilot_path = REPORT_ROOT / "pilot" / name
        if not pilot_path.is_file():
            raise FileNotFoundError(f"confirmatory extension requires pilot artifact: {pilot_path}")
        prior = pd.read_csv(pilot_path)
        current = pd.concat([prior, current], ignore_index=True)
    current.to_csv(output_dir / name, index=False)
    return current


def make_plots(output_dir: Path, loops: pd.DataFrame, order: pd.DataFrame, merges: pd.DataFrame, predictions: pd.DataFrame) -> None:
    plots = output_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(5.2, 3.7))
    axis.scatter(order["loop_identity_distance"], order["order_sensitivity_score"], c=order["seed"], cmap="viridis", s=28)
    axis.set_xlabel("Penultimate loop distance from identity")
    axis.set_ylabel("Order-sensitivity score")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(plots / f"holonomy_vs_order_sensitivity.{suffix}", dpi=220)
    plt.close(fig)

    summary = merges[~merges["nondeployable_or_upper_bound"].astype(bool)].groupby("method")["mean_accuracy"].mean().sort_values()
    fig, axis = plt.subplots(figsize=(6.4, 3.9))
    summary.plot.barh(ax=axis, color="#4472C4")
    axis.set_xlabel("Mean accuracy across A/B/C")
    axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(plots / f"merge_method_accuracy.{suffix}", dpi=220)
    plt.close(fig)

    classification = predictions[predictions["prediction_task"] == "harmful_merge"]
    fig, axis = plt.subplots(figsize=(6.4, 3.9))
    axis.barh(classification["feature_set"], classification["auroc"], color="#70AD47")
    axis.axvline(0.5, color="black", linewidth=0.8, linestyle="--")
    axis.set_xlabel("Double-held-out AUROC")
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(plots / f"harmful_merge_prediction.{suffix}", dpi=220)
    plt.close(fig)


def order_family_loop(value: str) -> str:
    left, right = value.split("_vs_")
    return f"{''.join(sorted(set(left)))}_square" if len(left) == 2 else f"swap_{left}_{right}"


def write_tables(output_dir: Path, order: pd.DataFrame, merges: pd.DataFrame, predictions: pd.DataFrame) -> None:
    tables = output_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    order.groupby("order_family", as_index=False)[["order_sensitivity_score", "loop_identity_distance"]].mean().to_latex(
        tables / "order_sensitivity.tex", index=False, float_format="%.4f"
    )
    merges.groupby("method", as_index=False)[["mean_accuracy", "worst_domain_accuracy", "regret_vs_best_deployable"]].mean().to_latex(
        tables / "merge_results.tex", index=False, float_format="%.4f"
    )
    predictions.to_latex(tables / "prediction_results.tex", index=False, float_format="%.4f")


def write_reports(
    output_dir: Path,
    mode: str,
    lineage: pd.DataFrame,
    loops: pd.DataFrame,
    commutators: pd.DataFrame,
    order: pd.DataFrame,
    merges: pd.DataFrame,
    predictions: pd.DataFrame,
    paired: pd.DataFrame,
    gates: dict[str, bool],
    failures: pd.DataFrame,
    command: str,
) -> None:
    stable_loops = int(loops["stable_nonidentity"].astype(bool).sum())
    stable_commutators = int(commutators["stable_noncommuting"].astype(bool).sum())
    cycle_rows = merges[merges["method"] == "cycle_aware_synchronization"]
    action_counts = cycle_rows["cycle_action"].value_counts().to_dict()
    harmful_count = int(
        merges[merges["method"] == "raw_parameter_average"]["ordinary_raw_harmful"].astype(bool).sum()
    )
    decision = "positive gate" if any(gates.values()) else "no preregistered gate"
    report = f"""# Model-lineage holonomy report

Mode: **{mode}**. Decision: **{decision} passed**.

## Execution

`{command}`

- Independent seeds: {sorted(lineage['seed'].unique().tolist())}
- Checkpoints: {len(lineage)}
- Stable nonidentity loop/layer rows: {stable_loops} / {len(loops)}
- Stable noncommutator rows: {stable_commutators} / {len(commutators)}
- Harmful raw branch merges: {harmful_count} / {len(cycle_rows)}
- Cycle-policy actions: `{action_counts}`
- Failures: {len(failures)}
- Gates: `{gates}`

All transition estimators were selected by unlabeled transport-validation residual. Application test logits were saved and hashed before labels were loaded in each execution phase. Seeds are the inferential unit.

## Boundary

These are natural learning-path representation loops, not Brauer classes or topological certificates. Three seeds permit pilot wording only; only the five-seed aggregate is assessed against the confirmatory gates.
"""
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    negative = (
        "# Negative results\n\n"
        + (
            "In this model-lineage setting, task-order path dependence is adequately explained by pairwise drift, parameter distance, or prediction disagreement; loop holonomy adds no reliable practical value.\n"
            if not any(gates.values()) and mode == "confirmatory"
            else "The phase-specific negative rows and failed gates are retained without opening another dataset, backbone, corruption, predictor, or cycle estimator.\n"
        )
        + "\nNo loop is called a Brauer class, and controlled projective evidence is not used to validate this natural application.\n"
    )
    (output_dir / "negative_results.md").write_text(negative, encoding="utf-8")
    if mode == "pilot":
        strongest = "No main-paper claim is available from the three-seed pilot; complete the frozen two-seed extension without broadening the design."
        stopping_answer = "No. Complete seeds 3-4 as preregistered, but do not add another dataset, corruption, backbone, predictor, or cycle estimator."
    else:
        strongest = (
            "At least one preregistered gate passed; only the exact passing gate and its five-seed interval may be stated."
            if any(gates.values())
            else "Different learning orders produced measurable terminal differences, but holonomy supplied no reliable incremental predictive or corrective value beyond pairwise diagnostics."
        )
        stopping_answer = (
            "No only for the exact passing gate, without broadening scope."
            if any(gates.values())
            else "Yes. The preregistered stopping rule applies."
        )
    assessment = f"""# Final assessment: model-lineage holonomy

Mode: **{mode}**. Gate status: `{gates}`.

1. **Were stable nonidentity loop holonomies observed?** {'Yes' if stable_loops else 'No'}; {stable_loops} loop/layer rows passed the frozen stability definition.
2. **Were any independent loop holonomies noncommuting?** {'Yes' if stable_commutators else 'No'}; {stable_commutators} commutator rows passed the frozen interval threshold.
3. **Did holonomy correlate with task-order dependence?** {'Yes, under H1.' if gates['H1'] else 'No confirmatory incremental association passed H1.'}
4. **Did holonomy add information beyond pairwise drift?** {'Yes, under a frozen held-out gate.' if gates['H1'] or gates['H2'] else 'No.'}
5. **Did holonomy predict harmful branch merges?** {'Yes, H2 passed.' if gates['H2'] else f'No held-out H2 improvement was established; {harmful_count} harmful raw merge rows were observed.'}
6. **Did cycle-aware correction improve merging?** {'Yes, H3 passed.' if gates['H3'] else 'No paired seed-level H3 improvement was established.'}
7. **Did conservative abstention reduce regret?** {'A repeated H4 conflict class passed.' if gates['H4'] else 'No repeated H4 conflict class was established.'}
8. **Which layers carried the strongest stable signal?** {strongest_layer(loops)}.
9. **What is the strongest main-paper claim?** {strongest}
10. **Should holonomy application experiments stop?** {stopping_answer}

## Integrity

- Test-logit-before-label flag: `{bool(merges['test_logits_hashed_before_labels'].astype(bool).all())}`.
- Double seed/family/loop holdout flags: `{bool(predictions[['all_seed_exclusions_passed','all_family_exclusions_passed','all_loop_exclusions_passed']].astype(bool).all().all())}`.
- Failure rows: `{len(failures)}`.
- No manuscript, LaTeX, bibliography, protected worktree, or prior evidence artifact was modified.
"""
    (output_dir / "final_assessment.md").write_text(assessment, encoding="utf-8")
    brief = f"""# Paper-editor evidence brief: model-lineage holonomy

## Recommended wording

{strongest}

## Required limitations

- Five independent training seeds are the inferential units; lineage rows and layers are not replicates.
- The model is one frozen ResNet-18 encoder with rank-4 feature adapters and a classifier on three deterministic CIFAR-10 corruptions.
- Prediction is double held out by seed and task-order family.
- Ensemble and sequential-adaptation rows are upper-bound or additional-training controls.
- No natural Brauer, topology, universal continual-learning, or broad model-merging claim is permitted.

## Gate status

`{gates}`
"""
    (output_dir / "paper_editor_evidence_brief.md").write_text(brief, encoding="utf-8")


def strongest_layer(loops: pd.DataFrame) -> str:
    stable = loops[loops["stable_nonidentity"].astype(bool)]
    if stable.empty:
        raw = loops.groupby("layer")["identity_distance"].mean().sort_values(ascending=False)
        return f"No layer passed the stability gate; `{raw.index[0]}` had the largest raw distance ({raw.iloc[0]:.4f}) but is not a stable signal"
    summary = loops.groupby("layer")["identity_distance"].mean().sort_values(ascending=False)
    return f"`{summary.index[0]}` had the largest mean identity distance ({summary.iloc[0]:.4f}); stability gates remain controlling"


def build_manifest(output_dir: Path, external_rows: list[dict[str, object]]) -> pd.DataFrame:
    rows = list(external_rows)
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "artifact_manifest.csv":
            rows.append(
                {
                    "artifact_kind": "committed_report_artifact",
                    "seed": "aggregate",
                    "node_or_family": "",
                    "path": str(path.relative_to(ROOT)),
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                }
            )
    frame = pd.DataFrame(rows).drop_duplicates(subset=["path", "sha256"], keep="last")
    frame.to_csv(output_dir / "artifact_manifest.csv", index=False)
    return frame


def verify_manifest(frame: pd.DataFrame) -> None:
    for row in frame.to_dict("records"):
        path = Path(row["path"])
        if not path.is_absolute():
            path = ROOT / path
        if not path.is_file() or path.stat().st_size != int(row["bytes"]) or sha256_file(path) != row["sha256"]:
            raise RuntimeError(f"artifact manifest verification failed: {path}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("pilot", "confirmatory"), required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("/Users/tinggong/Documents/GitHub/TwistedMerge/data"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--encoder-batch-size", type=int, default=128)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    seeds = FROZEN_CONFIG["pilot_seeds"] if args.mode == "pilot" else FROZEN_CONFIG["extension_seeds"]
    output_dir = REPORT_ROOT / "pilot" if args.mode == "pilot" else REPORT_ROOT
    artifact_dir = ARTIFACT_ROOT / args.mode
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    command = " ".join([sys.executable, *sys.argv])
    config = {
        **FROZEN_CONFIG,
        "mode": args.mode,
        "executed_seeds": seeds,
        "command": command,
        "execution_commit": git_output("rev-parse", "HEAD"),
        "source_sha256": sha256_file(Path(__file__)),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "torchvision": torchvision_version,
        "numpy": np.__version__,
        "device": str(device_for(args.device)),
        "physical_memory_bytes": psutil.virtual_memory().total,
    }
    write_json(output_dir / "config.json", config)
    source_archive = args.data_dir / "cifar-10-python.tar.gz"
    weights = ResNet18_Weights.IMAGENET1K_V1
    weights_path = Path(torch.hub.get_dir()) / "checkpoints" / Path(weights.url).name
    if not source_archive.is_file() or not weights_path.is_file() or not SOURCE_FEATURE_CACHE.is_file():
        raise FileNotFoundError("required cached dataset, encoder, or source feature artifact is missing")
    cache_path = ARTIFACT_ROOT / "shared_corrupted_features.pt"
    feature_payload, feature_metadata = build_feature_cache(
        args.data_dir.resolve(), cache_path, device=device_for(args.device), batch_size=args.encoder_batch_size
    )
    split_path = output_dir / "split_manifest.json"
    write_json(split_path, {name: [int(value) for value in tensor] for name, tensor in feature_payload["splits"].items()})
    config.update(
        {
            **feature_metadata,
            "dataset_archive_path": str(source_archive),
            "dataset_archive_sha256": sha256_file(source_archive),
            "encoder_weights_path": str(weights_path),
            "encoder_weights_sha256": sha256_file(weights_path),
            "source_feature_cache_sha256": sha256_file(SOURCE_FEATURE_CACHE),
            "feature_cache_path": str(cache_path),
            "feature_cache_sha256": sha256_file(cache_path),
            "split_manifest_sha256": sha256_file(split_path),
        }
    )
    write_json(output_dir / "config.json", config)
    train_dataset = CIFAR10(args.data_dir, train=True, download=False)
    test_dataset = CIFAR10(args.data_dir, train=False, download=False)
    contexts, transports, merge_contexts = [], [], []
    failures = []
    external_artifacts = [
        artifact_row("shared_feature_cache", cache_path, "shared"),
        artifact_row("split_manifest", split_path, "shared"),
    ]
    if args.mode == "confirmatory" and (REPORT_ROOT / "pilot" / "artifact_manifest.csv").is_file():
        external_artifacts.extend(
            pd.read_csv(REPORT_ROOT / "pilot" / "artifact_manifest.csv").to_dict("records")
        )
    for seed in seeds:
        try:
            context = prepare_seed(seed, feature_payload, train_dataset, output_dir, artifact_dir)
            transport = analyze_transports(context, artifact_dir)
            merges = merge_context(context, transport, feature_payload, train_dataset, artifact_dir)
            contexts.append(context)
            transports.append(transport)
            merge_contexts.append(merges)
            external_artifacts.extend(context["artifact_rows"])
            external_artifacts.append(transport["transport_artifact"])
            external_artifacts.extend(merges["artifacts"])
        except Exception as error:
            failures.append(
                {
                    "mode": args.mode,
                    "seed": seed,
                    "stage": "seed_pipeline",
                    "error_type": type(error).__name__,
                    "message": str(error),
                }
            )
    if len(contexts) != len(seeds):
        pd.DataFrame(failures).to_csv(output_dir / "failure_log.csv", index=False)
        raise RuntimeError("one or more independent seed pipelines failed")

    # This is the first access to application-test labels in this execution phase.
    test_indices = feature_payload["splits"]["application_test"].numpy()
    test_labels = torch.tensor(np.asarray(test_dataset.targets), dtype=torch.long)[torch.from_numpy(test_indices)]
    order_rows, merge_rows = [], []
    for context, transport, merges in zip(contexts, transports, merge_contexts, strict=True):
        seed_order, seed_merge = score_context(context, transport, merges, test_labels)
        order_rows.extend(seed_order)
        merge_rows.extend(seed_merge)

    include_pilot = args.mode == "confirmatory"
    lineage = aggregate_existing(output_dir, "lineage_manifest.csv", [row for context in contexts for row in context["lineage_rows"]], include_pilot)
    checkpoints = aggregate_existing(output_dir, "checkpoint_manifest.csv", [row for context in contexts for row in context["checkpoint_rows"]], include_pilot)
    pairwise = aggregate_existing(output_dir, "pairwise_residuals.csv", [row for value in transports for row in value["pairwise_rows"]], include_pilot)
    transition_maps = aggregate_existing(output_dir, "transport_maps.csv", [row for value in transports for row in value["transition_rows"]], include_pilot)
    loops = aggregate_existing(output_dir, "loop_holonomy.csv", [row for value in transports for row in value["loop_rows"]], include_pilot)
    commutators = aggregate_existing(output_dir, "loop_commutators.csv", [row for value in transports for row in value["commutator_rows"]], include_pilot)
    order = aggregate_existing(output_dir, "order_sensitivity.csv", order_rows, include_pilot)
    merges = aggregate_existing(output_dir, "merge_results.csv", merge_rows, include_pilot)
    capacity = aggregate_existing(output_dir, "capacity_cost.csv", [row for value in merge_contexts for row in value["capacity_rows"]], include_pilot)
    _ = checkpoints, pairwise, transition_maps, capacity
    prediction_results, paired_statistics, gates = prediction_analysis(order, merges)
    prediction_results.to_csv(output_dir / "prediction_results.csv", index=False)
    paired_statistics.to_csv(output_dir / "paired_statistics.csv", index=False)
    failure_frame = pd.DataFrame(failures, columns=("mode", "seed", "stage", "error_type", "message"))
    failure_frame.to_csv(output_dir / "failure_log.csv", index=False)
    make_plots(output_dir, loops, order, merges, prediction_results)
    write_tables(output_dir, order, merges, prediction_results)
    write_reports(
        output_dir,
        args.mode,
        lineage,
        loops,
        commutators,
        order,
        merges,
        prediction_results,
        paired_statistics,
        gates,
        failure_frame,
        command,
    )
    manifest = build_manifest(output_dir, external_artifacts)
    verify_manifest(manifest)
    print(json.dumps({"mode": args.mode, "seeds": seeds, "gates": gates, "failures": len(failures)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
