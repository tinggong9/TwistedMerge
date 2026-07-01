#!/usr/bin/env python
"""Same-base task-vector benchmark for soup and task-vector baselines.

This benchmark is deliberately separate from the independent-seed/rebasin
experiments.  It trains one common base model, fine-tunes task copies from that
base, and then evaluates task-vector methods whose assumptions require a shared
coordinate system.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model_merging_benchmark import (  # noqa: E402
    average_models,
    clone_model,
    compute_layerwise_pairwise_permutations,
    cycle_score,
    device_from_arg,
    evaluate_model,
    format_markdown_table,
    greedy_soup,
    load_dataset,
    make_loader,
    make_model,
    primary_alignment_layer,
    primary_pairwise_permutations,
    require_torch,
    save_checkpoint,
    set_seed,
    synchronize_permutations,
    train_model,
)


METHODS = [
    "base_model",
    "individual_finetuned_mean",
    "weight_average",
    "greedy_soup",
    "slerp_sequential",
    "task_arithmetic",
    "ties_merging",
    "dare",
    "git_rebasin_pairwise_secondary_not_run",
    "c2m3_synchronization_secondary_not_run",
]


@dataclass(frozen=True)
class TaskDef:
    name: str
    classes: tuple[int, ...]


TASK_PRESETS = {
    "mnist_digit_subsets": (
        TaskDef("digits_0_4", (0, 1, 2, 3, 4)),
        TaskDef("digits_5_9", (5, 6, 7, 8, 9)),
        TaskDef("even_digits", (0, 2, 4, 6, 8)),
    ),
    "mnist_four_subsets": (
        TaskDef("digits_0_4", (0, 1, 2, 3, 4)),
        TaskDef("digits_5_9", (5, 6, 7, 8, 9)),
        TaskDef("even_digits", (0, 2, 4, 6, 8)),
        TaskDef("odd_digits", (1, 3, 5, 7, 9)),
    ),
    "fashion_class_subsets": (
        TaskDef("fashion_low", (0, 1, 2, 3, 4)),
        TaskDef("fashion_high", (5, 6, 7, 8, 9)),
        TaskDef("fashion_even", (0, 2, 4, 6, 8)),
    ),
}


def parse_seeds(text: str) -> list[int]:
    seeds: list[int] = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            start, end = part.split(":", 1)
            seeds.extend(range(int(start), int(end)))
        elif "-" in part:
            start, end = part.split("-", 1)
            seeds.extend(range(int(start), int(end) + 1))
        else:
            seeds.append(int(part))
    return seeds


def safe_float(value) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def split_indices(n_items: int, val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    torch, _, _ = require_torch()
    generator = torch.Generator()
    generator.manual_seed(seed)
    indices = torch.randperm(n_items, generator=generator).tolist()
    n_val = max(1, int(round(n_items * val_fraction)))
    return indices[n_val:], indices[:n_val]


def sample_indices(indices: list[int], max_items: int, seed: int) -> list[int]:
    if max_items <= 0 or len(indices) <= max_items:
        return list(indices)
    rng = np.random.default_rng(seed)
    picked = rng.choice(np.asarray(indices, dtype=int), size=max_items, replace=False)
    return [int(item) for item in picked.tolist()]


def subset_by_classes(dataset, base_indices: list[int], classes: tuple[int, ...], max_items: int, seed: int) -> list[int]:
    wanted = set(int(item) for item in classes)
    kept = []
    for idx in base_indices:
        _x, y = dataset[idx]
        if int(y) in wanted:
            kept.append(int(idx))
    return sample_indices(kept, max_items, seed)


def make_subset(dataset, indices: list[int]):
    torch, _, _ = require_torch()
    return torch.utils.data.Subset(dataset, indices)


def combined_loader(task_loaders: dict[str, object], batch_size: int, seed: int):
    torch, _, _ = require_torch()
    datasets = [loader.dataset for loader in task_loaders.values()]
    return make_loader(torch.utils.data.ConcatDataset(datasets), batch_size, shuffle=False, seed=seed)


def state_vector(model) -> tuple[np.ndarray, list[tuple[str, tuple[int, ...], np.dtype]]]:
    pieces = []
    meta = []
    for name, tensor in model.state_dict().items():
        arr = tensor.detach().cpu().numpy()
        pieces.append(arr.reshape(-1).astype(np.float64))
        meta.append((name, tuple(arr.shape), arr.dtype))
    return np.concatenate(pieces), meta


def vector_to_model(vector: np.ndarray, meta, architecture: str, spec, width: int):
    torch, _, _ = require_torch()
    model = make_model(architecture, spec, width)
    state = {}
    offset = 0
    for name, shape, dtype in meta:
        size = int(np.prod(shape))
        arr = vector[offset : offset + size].reshape(shape).astype(dtype)
        state[name] = torch.tensor(arr)
        offset += size
    model.load_state_dict(state)
    return model


def average_vectors(vectors: list[np.ndarray]) -> np.ndarray:
    return np.stack(vectors, axis=0).mean(axis=0)


def slerp(v0: np.ndarray, v1: np.ndarray, t: float) -> np.ndarray:
    n0 = float(np.linalg.norm(v0))
    n1 = float(np.linalg.norm(v1))
    if n0 <= 1e-12 or n1 <= 1e-12:
        return (1.0 - t) * v0 + t * v1
    u0 = v0 / n0
    u1 = v1 / n1
    dot = float(np.clip(np.dot(u0, u1), -1.0, 1.0))
    if abs(dot) > 0.9995:
        return (1.0 - t) * v0 + t * v1
    theta = math.acos(dot)
    out = math.sin((1.0 - t) * theta) / math.sin(theta) * v0 + math.sin(t * theta) / math.sin(theta) * v1
    return out


def sequential_slerp(vectors: list[np.ndarray]) -> np.ndarray:
    out = vectors[0].copy()
    for idx, vec in enumerate(vectors[1:], start=2):
        out = slerp(out, vec, 1.0 / idx)
    return out


def task_arithmetic_vector(base: np.ndarray, deltas: list[np.ndarray], scale: float) -> np.ndarray:
    return base + scale * np.stack(deltas, axis=0).mean(axis=0)


def ties_vector(base: np.ndarray, deltas: list[np.ndarray], density: float, scale: float) -> np.ndarray:
    delta_matrix = np.stack(deltas, axis=0)
    masked = np.zeros_like(delta_matrix)
    keep_count = max(1, int(round(density * delta_matrix.shape[1])))
    for idx, delta in enumerate(delta_matrix):
        if keep_count >= len(delta):
            mask = np.ones(len(delta), dtype=bool)
        else:
            threshold = np.partition(np.abs(delta), -keep_count)[-keep_count]
            mask = np.abs(delta) >= threshold
        masked[idx, mask] = delta[mask]
    elected = np.sign(masked.sum(axis=0))
    elected[elected == 0.0] = np.sign(delta_matrix.sum(axis=0))[elected == 0.0]
    selected = np.where(np.sign(masked) == elected[None, :], masked, 0.0)
    counts = np.maximum((selected != 0.0).sum(axis=0), 1)
    merged = selected.sum(axis=0) / counts
    return base + scale * merged


def dare_vector(base: np.ndarray, deltas: list[np.ndarray], drop_rate: float, scale: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    keep_prob = max(1e-6, 1.0 - drop_rate)
    masked = []
    for idx, delta in enumerate(deltas):
        mask = rng.random(len(delta)) < keep_prob
        masked.append(np.where(mask, delta / keep_prob, 0.0))
    return base + scale * np.stack(masked, axis=0).mean(axis=0)


def evaluate_across_tasks(model, task_loaders: dict[str, object], device) -> dict:
    per_task = {}
    for task_name, loader in task_loaders.items():
        metrics = evaluate_model(model, loader, device)
        per_task[task_name] = metrics
    accuracies = [metrics["accuracy"] for metrics in per_task.values()]
    losses = [metrics["loss"] for metrics in per_task.values()]
    return {
        "average_accuracy": float(np.mean(accuracies)),
        "worst_accuracy": float(np.min(accuracies)),
        "average_loss": float(np.mean(losses)),
        "per_task": per_task,
    }


def validation_score(model, val_loaders: dict[str, object], device) -> tuple[float, float]:
    metrics = evaluate_across_tasks(model, val_loaders, device)
    return float(metrics["average_accuracy"]), float(metrics["average_loss"])


def select_candidate(
    name: str,
    candidates: list[tuple[dict, object]],
    val_loaders: dict[str, object],
    device,
) -> tuple[dict, object, dict]:
    rows = []
    best = None
    for metadata, model in candidates:
        acc, loss = validation_score(model, val_loaders, device)
        row = {**metadata, "validation_accuracy": acc, "validation_loss": loss}
        rows.append(row)
        key = (acc, -loss)
        if best is None or key > best[0]:
            best = (key, row, model)
    assert best is not None
    return best[1], best[2], {"method": name, "candidates": rows}


def candidate_grid_records(base: dict, method: str, selection: dict, selected: dict) -> list[dict]:
    candidate_base = candidate_grid_base(base)
    records = []
    selected_params = {
        key: value
        for key, value in selected.items()
        if key not in {"validation_accuracy", "validation_loss", "selection_trace"}
    }
    for rank, candidate in enumerate(selection.get("candidates", []), start=1):
        params = {
            key: value
            for key, value in candidate.items()
            if key not in {"validation_accuracy", "validation_loss"}
        }
        is_selected = all(candidate.get(key) == value for key, value in selected_params.items())
        records.append(
            {
                **candidate_base,
                "method": method,
                "candidate_kind": "validation_hyperparameter_grid",
                "candidate_rank": rank,
                "candidate_model_index": "",
                "candidate_params_json": json.dumps(params, sort_keys=True, separators=(",", ":")),
                "scale": candidate.get("scale", float("nan")),
                "density": candidate.get("density", float("nan")),
                "drop_rate": candidate.get("drop_rate", float("nan")),
                "validation_accuracy": candidate.get("validation_accuracy", float("nan")),
                "validation_loss": candidate.get("validation_loss", float("nan")),
                "accepted": bool(is_selected),
                "selected": bool(is_selected),
                "decision_reason": "selected_by_validation_accuracy_then_loss" if is_selected else "not_selected_validation_grid",
                "uses_test_metrics_for_selection": False,
            }
        )
    return records


def greedy_candidate_records(base: dict, trajectory: list[dict]) -> list[dict]:
    candidate_base = candidate_grid_base(base)
    records = []
    for item in trajectory:
        records.append(
            {
                **candidate_base,
                "method": "greedy_soup",
                "candidate_kind": "greedy_soup_validation_trajectory",
                "candidate_rank": int(item["candidate_rank"]),
                "candidate_model_index": int(item["candidate_model_index"]),
                "candidate_params_json": json.dumps(
                    {
                        "candidate_order": item.get("candidate_order", []),
                        "soup_indices_before": item.get("soup_indices_before", []),
                        "soup_indices_after": item.get("soup_indices_after", []),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "scale": float("nan"),
                "density": float("nan"),
                "drop_rate": float("nan"),
                "validation_accuracy": item.get("candidate_soup_validation_accuracy", float("nan")),
                "validation_loss": item.get("candidate_soup_validation_loss", float("nan")),
                "accepted": bool(item.get("accepted", False)),
                "selected": bool(item.get("is_final_selection", False)),
                "decision_reason": item.get("decision_reason", ""),
                "uses_test_metrics_for_selection": False,
            }
        )
    return records


def candidate_grid_base(base: dict) -> dict:
    """Metadata allowed in the validation-selection candidate grid.

    Keep this free of test metrics. The benchmark table stores test evaluation,
    but selector candidate rows should remain validation-only plus static
    setting/task-vector diagnostics.
    """

    allowed = [
        "setting_id",
        "run_id",
        "dataset",
        "task_preset",
        "architecture",
        "width",
        "n_tasks",
        "seed",
        "base_epochs",
        "finetune_epochs",
        "max_train_samples",
        "max_test_samples",
        "task_definitions_json",
        "task_vector_sign_conflict_fraction",
        "task_vector_active_fraction",
        "task_vector_mean_pairwise_cosine",
        "task_vector_min_pairwise_cosine",
        "triangle_cycle_score",
        "sync_disagreement",
    ]
    return {key: base[key] for key in allowed if key in base}


def sign_conflict_stats(deltas: list[np.ndarray]) -> dict[str, float]:
    matrix = np.stack(deltas, axis=0)
    threshold = max(float(np.median(np.abs(matrix))) * 1e-3, 1e-10)
    pos = (matrix > threshold).any(axis=0)
    neg = (matrix < -threshold).any(axis=0)
    active = (np.abs(matrix) > threshold).any(axis=0)
    conflict = pos & neg
    norms = np.maximum(np.linalg.norm(matrix, axis=1), 1e-12)
    cosine = (matrix @ matrix.T) / (norms[:, None] * norms[None, :])
    off_diag = cosine[np.triu_indices_from(cosine, k=1)]
    return {
        "task_vector_sign_conflict_fraction": float(conflict.sum() / max(active.sum(), 1)),
        "task_vector_active_fraction": float(active.mean()),
        "task_vector_mean_pairwise_cosine": float(off_diag.mean()) if len(off_diag) else float("nan"),
        "task_vector_min_pairwise_cosine": float(off_diag.min()) if len(off_diag) else float("nan"),
    }


def cycle_diagnostics(models: list, architecture: str, loader, device, width: int) -> dict[str, float]:
    if len(models) < 3:
        return {"triangle_cycle_score": 0.0, "sync_disagreement": 0.0}
    try:
        pairwise_by_layer = compute_layerwise_pairwise_permutations(models, architecture, loader, device, "weight")
        pairwise = primary_pairwise_permutations(pairwise_by_layer, architecture)
        primary_layer = primary_alignment_layer(architecture)
        primary_width = width
        if primary_layer == "hidden1" and hasattr(models[0], "hidden1"):
            primary_width = int(models[0].hidden1.out_features)
        elif primary_layer == "hidden2" and hasattr(models[0], "hidden2"):
            primary_width = int(models[0].hidden2.out_features)
        score, _rows = cycle_score(pairwise, len(models), primary_width)
        _ref, _q, sync_disagreement = synchronize_permutations(pairwise, len(models))
        return {"triangle_cycle_score": float(score), "sync_disagreement": float(sync_disagreement)}
    except Exception:
        return {"triangle_cycle_score": float("nan"), "sync_disagreement": float("nan")}


def method_record(
    *,
    base: dict,
    method: str,
    method_role: str,
    model,
    val_loaders: dict[str, object],
    test_loaders: dict[str, object],
    device,
    oracle_average_accuracy: float,
    selected_metadata: dict | None = None,
    soup_trajectory: list[dict] | None = None,
    status: str = "ok",
) -> dict:
    if model is None:
        metrics = {
            "average_accuracy": float("nan"),
            "worst_accuracy": float("nan"),
            "average_loss": float("nan"),
            "per_task": {},
        }
        val_metrics = metrics
    else:
        val_metrics = evaluate_across_tasks(model, val_loaders, device)
        metrics = evaluate_across_tasks(model, test_loaders, device)
    selected_metadata = selected_metadata or {}
    trajectory = soup_trajectory or []
    accepted = [
        {
            "candidate_rank": int(item["candidate_rank"]),
            "candidate_model_index": int(item["candidate_model_index"]),
            "accepted": bool(item["accepted"]),
            "decision_reason": item["decision_reason"],
            "validation_accuracy_margin_after_minus_before": (
                float(item["validation_accuracy_margin_after_minus_before"])
                if math.isfinite(float(item["validation_accuracy_margin_after_minus_before"]))
                else None
            ),
        }
        for item in trajectory
    ]
    return {
        **base,
        "method": method,
        "method_role": method_role,
        "status": status,
        "average_test_accuracy": metrics["average_accuracy"],
        "worst_task_accuracy": metrics["worst_accuracy"],
        "average_test_loss": metrics["average_loss"],
        "validation_selected_accuracy": val_metrics["average_accuracy"],
        "validation_selected_loss": val_metrics["average_loss"],
        "interference_score": oracle_average_accuracy - metrics["average_accuracy"]
        if math.isfinite(metrics["average_accuracy"])
        else float("nan"),
        "per_task_test_accuracy_json": json.dumps(
            {task: values["accuracy"] for task, values in metrics["per_task"].items()},
            sort_keys=True,
            separators=(",", ":"),
        ),
        "per_task_val_accuracy_json": json.dumps(
            {task: values["accuracy"] for task, values in val_metrics["per_task"].items()},
            sort_keys=True,
            separators=(",", ":"),
        ),
        "selected_hyperparameters_json": json.dumps(selected_metadata, sort_keys=True, separators=(",", ":")),
        "greedy_soup_acceptance_json": json.dumps(accepted, sort_keys=True, separators=(",", ":")),
        "capacity_matched": method not in {"individual_oracle_per_task"},
        "single_model": method not in {"individual_oracle_per_task"},
        "uses_validation_data": method in {"greedy_soup", "task_arithmetic", "ties_merging", "dare"},
        "common_base_required": method in {"slerp_sequential", "task_arithmetic", "ties_merging", "dare"},
    }


def save_model_checkpoint(model, path: Path, metadata: dict) -> None:
    save_checkpoint(model, path, metadata)


def run_one_seed(args, dataset_name: str, seed: int, task_defs: tuple[TaskDef, ...]) -> tuple[list[dict], list[dict]]:
    torch, _, _ = require_torch()
    device = device_from_arg(args.device)
    set_seed(seed)
    spec, train_base, test_base = load_dataset(
        dataset_name,
        args.data_dir,
        args.max_train_samples,
        args.max_test_samples,
        args.dataset_seed,
        augmentation=args.augmentation,
    )
    train_indices, val_indices = split_indices(len(train_base), args.val_fraction, args.dataset_seed + 17 + seed)
    base_train_indices = sample_indices(train_indices, args.max_base_train_samples, seed + 11)
    base_train_loader = make_loader(make_subset(train_base, base_train_indices), args.batch_size, shuffle=True, seed=seed + 101)

    setting_id = f"{dataset_name}_{args.task_preset}_{args.architecture}_W{args.width}_N{len(task_defs)}"
    run_id = f"{setting_id}_seed{seed}"
    checkpoint_dir = args.reports_dir / "checkpoints" / "same_base_task_vector" / setting_id / f"seed{seed}"

    base_model = make_model(args.architecture, spec, args.width)
    train_model(
        base_model,
        base_train_loader,
        args.base_epochs,
        args.lr,
        device,
        optimizer=args.optimizer,
        weight_decay=args.weight_decay,
        scheduler=args.scheduler,
        step_size=args.step_size,
        gamma=args.gamma,
    )
    base_model.to("cpu")
    save_model_checkpoint(
        base_model,
        checkpoint_dir / "base.pt",
        {
            "dataset": dataset_name,
            "seed": seed,
            "architecture": args.architecture,
            "width": args.width,
            "task_preset": args.task_preset,
            "checkpoint_role": "common_base",
        },
    )

    task_train_loaders = {}
    task_val_loaders = {}
    task_test_loaders = {}
    task_rows = []
    for task_idx, task in enumerate(task_defs):
        train_task_indices = subset_by_classes(
            train_base,
            train_indices,
            task.classes,
            args.max_task_train_samples,
            seed + 1000 + task_idx,
        )
        val_task_indices = subset_by_classes(
            train_base,
            val_indices,
            task.classes,
            args.max_task_val_samples,
            seed + 2000 + task_idx,
        )
        test_task_indices = subset_by_classes(
            test_base,
            list(range(len(test_base))),
            task.classes,
            args.max_task_test_samples,
            seed + 3000 + task_idx,
        )
        task_train_loaders[task.name] = make_loader(
            make_subset(train_base, train_task_indices),
            args.batch_size,
            shuffle=True,
            seed=seed + 4000 + task_idx,
        )
        task_val_loaders[task.name] = make_loader(
            make_subset(train_base, val_task_indices),
            args.batch_size,
            shuffle=False,
            seed=seed + 5000 + task_idx,
        )
        task_test_loaders[task.name] = make_loader(
            make_subset(test_base, test_task_indices),
            args.batch_size,
            shuffle=False,
            seed=seed + 6000 + task_idx,
        )
        task_rows.append(
            {
                "task_name": task.name,
                "classes": list(task.classes),
                "train_samples": len(train_task_indices),
                "val_samples": len(val_task_indices),
                "test_samples": len(test_task_indices),
            }
        )

    task_models = []
    individual_diag_test = []
    for task_idx, task in enumerate(task_defs):
        model = clone_model(base_model, args.architecture, spec, args.width)
        train_model(
            model,
            task_train_loaders[task.name],
            args.finetune_epochs,
            args.finetune_lr,
            device,
            optimizer=args.optimizer,
            weight_decay=args.weight_decay,
            scheduler=args.scheduler,
            step_size=args.step_size,
            gamma=args.gamma,
        )
        model.to("cpu")
        save_model_checkpoint(
            model,
            checkpoint_dir / f"task_{task_idx}_{task.name}.pt",
            {
                "dataset": dataset_name,
                "seed": seed,
                "architecture": args.architecture,
                "width": args.width,
                "task_preset": args.task_preset,
                "task_name": task.name,
                "task_classes": list(task.classes),
                "checkpoint_role": "fine_tuned_task",
            },
        )
        task_models.append(model)
        individual_diag_test.append(evaluate_model(model, task_test_loaders[task.name], device)["accuracy"])

    oracle_average_accuracy = float(np.mean(individual_diag_test))
    base_vector, meta = state_vector(base_model)
    task_vectors = [state_vector(model)[0] for model in task_models]
    deltas = [vec - base_vector for vec in task_vectors]
    sign_stats = sign_conflict_stats(deltas)
    combined_val_loader = combined_loader(task_val_loaders, args.batch_size, seed + 7000)
    cycle_stats = cycle_diagnostics(task_models, args.architecture, combined_val_loader, device, args.width)

    base = {
        "setting_id": setting_id,
        "run_id": run_id,
        "dataset": dataset_name,
        "task_preset": args.task_preset,
        "architecture": args.architecture,
        "width": args.width,
        "n_tasks": len(task_defs),
        "seed": seed,
        "base_epochs": args.base_epochs,
        "finetune_epochs": args.finetune_epochs,
        "max_train_samples": args.max_train_samples,
        "max_test_samples": args.max_test_samples,
        "task_definitions_json": json.dumps(task_rows, sort_keys=True, separators=(",", ":")),
        "base_checkpoint": str(checkpoint_dir / "base.pt"),
        "task_checkpoints_json": json.dumps(
            [str(checkpoint_dir / f"task_{idx}_{task.name}.pt") for idx, task in enumerate(task_defs)],
            separators=(",", ":"),
        ),
        "oracle_average_task_accuracy": oracle_average_accuracy,
        **sign_stats,
        **cycle_stats,
    }

    rows: list[dict] = []
    candidate_rows: list[dict] = []
    rows.append(
        method_record(
            base=base,
            method="base_model",
            method_role="common_base_reference",
            model=base_model,
            val_loaders=task_val_loaders,
            test_loaders=task_test_loaders,
            device=device,
            oracle_average_accuracy=oracle_average_accuracy,
        )
    )

    for task_idx, task in enumerate(task_defs):
        rows.append(
            method_record(
                base=base,
                method=f"individual_finetuned_{task.name}",
                method_role="individual_task_model",
                model=task_models[task_idx],
                val_loaders=task_val_loaders,
                test_loaders=task_test_loaders,
                device=device,
                oracle_average_accuracy=oracle_average_accuracy,
                selected_metadata={"task_name": task.name, "task_index": task_idx},
            )
        )
    rows.append(
        method_record(
            base=base,
            method="individual_finetuned_mean",
            method_role="oracle_summary_not_single_model",
            model=None,
            val_loaders=task_val_loaders,
            test_loaders=task_test_loaders,
            device=device,
            oracle_average_accuracy=oracle_average_accuracy,
            selected_metadata={"oracle_average_task_accuracy": oracle_average_accuracy},
            status="oracle_summary",
        )
    )
    rows[-1]["average_test_accuracy"] = oracle_average_accuracy
    rows[-1]["worst_task_accuracy"] = float(np.min(individual_diag_test))
    rows[-1]["interference_score"] = 0.0
    rows[-1]["single_model"] = False
    rows[-1]["capacity_matched"] = False

    weight_model = average_models(task_models, args.architecture, spec, args.width)
    rows.append(
        method_record(
            base=base,
            method="weight_average",
            method_role="same_base_weight_average",
            model=weight_model,
            val_loaders=task_val_loaders,
            test_loaders=task_test_loaders,
            device=device,
            oracle_average_accuracy=oracle_average_accuracy,
        )
    )

    soup_model, soup_indices, _soup_test, soup_trajectory = greedy_soup(
        task_models,
        combined_val_loader,
        combined_val_loader,
        device,
        args.architecture,
        spec,
        args.width,
        return_trajectory=True,
    )
    candidate_rows.extend(greedy_candidate_records(base, soup_trajectory))
    rows.append(
        method_record(
            base=base,
            method="greedy_soup",
            method_role="model_soups_validation_descent",
            model=soup_model,
            val_loaders=task_val_loaders,
            test_loaders=task_test_loaders,
            device=device,
            oracle_average_accuracy=oracle_average_accuracy,
            selected_metadata={"selection_indices": soup_indices},
            soup_trajectory=soup_trajectory,
        )
    )

    slerp_model = vector_to_model(sequential_slerp(task_vectors), meta, args.architecture, spec, args.width)
    rows.append(
        method_record(
            base=base,
            method="slerp_sequential",
            method_role="same_base_interpolation",
            model=slerp_model,
            val_loaders=task_val_loaders,
            test_loaders=task_test_loaders,
            device=device,
            oracle_average_accuracy=oracle_average_accuracy,
        )
    )

    task_arithmetic_candidates = []
    for scale in args.task_arithmetic_scales:
        model = vector_to_model(task_arithmetic_vector(base_vector, deltas, scale), meta, args.architecture, spec, args.width)
        task_arithmetic_candidates.append(({"scale": scale}, model))
    selected, task_arithmetic_model, selection = select_candidate("task_arithmetic", task_arithmetic_candidates, task_val_loaders, device)
    candidate_rows.extend(candidate_grid_records(base, "task_arithmetic", selection, selected))
    rows.append(
        method_record(
            base=base,
            method="task_arithmetic",
            method_role="same_base_task_vector",
            model=task_arithmetic_model,
            val_loaders=task_val_loaders,
            test_loaders=task_test_loaders,
            device=device,
            oracle_average_accuracy=oracle_average_accuracy,
            selected_metadata={**selected, "selection_trace": selection["candidates"]},
        )
    )

    ties_candidates = []
    for density in args.ties_densities:
        for scale in args.ties_scales:
            model = vector_to_model(ties_vector(base_vector, deltas, density, scale), meta, args.architecture, spec, args.width)
            ties_candidates.append(({"density": density, "scale": scale}, model))
    selected, ties_model, selection = select_candidate("ties_merging", ties_candidates, task_val_loaders, device)
    candidate_rows.extend(candidate_grid_records(base, "ties_merging", selection, selected))
    rows.append(
        method_record(
            base=base,
            method="ties_merging",
            method_role="same_base_task_vector",
            model=ties_model,
            val_loaders=task_val_loaders,
            test_loaders=task_test_loaders,
            device=device,
            oracle_average_accuracy=oracle_average_accuracy,
            selected_metadata={**selected, "selection_trace": selection["candidates"]},
        )
    )

    dare_candidates = []
    for drop_rate in args.dare_drop_rates:
        for scale in args.dare_scales:
            model = vector_to_model(
                dare_vector(base_vector, deltas, drop_rate, scale, seed + int(10000 * drop_rate) + int(100 * scale)),
                meta,
                args.architecture,
                spec,
                args.width,
            )
            dare_candidates.append(({"drop_rate": drop_rate, "scale": scale}, model))
    selected, dare_model, selection = select_candidate("dare", dare_candidates, task_val_loaders, device)
    candidate_rows.extend(candidate_grid_records(base, "dare", selection, selected))
    rows.append(
        method_record(
            base=base,
            method="dare",
            method_role="same_base_task_vector",
            model=dare_model,
            val_loaders=task_val_loaders,
            test_loaders=task_test_loaders,
            device=device,
            oracle_average_accuracy=oracle_average_accuracy,
            selected_metadata={**selected, "selection_trace": selection["candidates"]},
        )
    )

    for method in ["git_rebasin_pairwise_secondary_not_run", "c2m3_synchronization_secondary_not_run"]:
        rows.append(
            method_record(
                base=base,
                method=method,
                method_role="secondary_rebasin_diagnostic_not_run",
                model=None,
                val_loaders=task_val_loaders,
                test_loaders=task_test_loaders,
                device=device,
                oracle_average_accuracy=oracle_average_accuracy,
                selected_metadata={
                    "reason": "common_base_task_vector_setup_has_no_independent_seed_permutation_regime",
                },
                status="not_run_secondary_regime_mismatch",
            )
        )

    baselines = {row["method"]: row["average_test_accuracy"] for row in rows}
    for row in rows:
        row["delta_vs_greedy_soup"] = row["average_test_accuracy"] - baselines.get("greedy_soup", float("nan"))
        row["delta_vs_task_arithmetic"] = row["average_test_accuracy"] - baselines.get("task_arithmetic", float("nan"))
        row["delta_vs_ties"] = row["average_test_accuracy"] - baselines.get("ties_merging", float("nan"))
        row["delta_vs_dare"] = row["average_test_accuracy"] - baselines.get("dare", float("nan"))

    for model in [base_model, *task_models]:
        model.to("cpu")
    return rows, candidate_rows


def bootstrap_ci(values: np.ndarray, samples: int, seed: int) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(max(samples, 1)):
        idx = rng.integers(0, len(values), len(values))
        draws.append(float(values[idx].mean()))
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def summarize(rows: pd.DataFrame, bootstrap_samples: int) -> pd.DataFrame:
    records = []
    if rows.empty:
        return pd.DataFrame()
    group_cols = ["dataset", "task_preset", "architecture", "width", "n_tasks", "method"]
    for key, group in rows.groupby(group_cols, dropna=False):
        dataset, task_preset, architecture, width, n_tasks, method = key
        ok = group[group["status"].astype(str).isin(["ok", "oracle_summary"])]
        values = pd.to_numeric(ok["average_test_accuracy"], errors="coerce").to_numpy(dtype=float)
        delta_greedy = pd.to_numeric(ok["delta_vs_greedy_soup"], errors="coerce").to_numpy(dtype=float)
        delta_task = pd.to_numeric(ok["delta_vs_task_arithmetic"], errors="coerce").to_numpy(dtype=float)
        delta_ties = pd.to_numeric(ok["delta_vs_ties"], errors="coerce").to_numpy(dtype=float)
        delta_dare = pd.to_numeric(ok["delta_vs_dare"], errors="coerce").to_numpy(dtype=float)
        ci_low, ci_high = bootstrap_ci(values, bootstrap_samples, 101)
        dg_low, dg_high = bootstrap_ci(delta_greedy, bootstrap_samples, 102)
        dt_low, dt_high = bootstrap_ci(delta_task, bootstrap_samples, 103)
        tt_low, tt_high = bootstrap_ci(delta_ties, bootstrap_samples, 104)
        dd_low, dd_high = bootstrap_ci(delta_dare, bootstrap_samples, 105)
        records.append(
            {
                "dataset": dataset,
                "task_preset": task_preset,
                "architecture": architecture,
                "width": int(width),
                "n_tasks": int(n_tasks),
                "method": method,
                "method_role": group["method_role"].iloc[0],
                "status": group["status"].iloc[0],
                "n_rows": int(len(group)),
                "n_ok_rows": int(len(ok)),
                "n_unique_seeds": int(ok["seed"].nunique()) if not ok.empty else 0,
                "mean_average_test_accuracy": float(np.nanmean(values)) if len(values) else float("nan"),
                "accuracy_ci_low": ci_low,
                "accuracy_ci_high": ci_high,
                "mean_worst_task_accuracy": float(pd.to_numeric(ok["worst_task_accuracy"], errors="coerce").mean()) if not ok.empty else float("nan"),
                "mean_validation_selected_accuracy": float(pd.to_numeric(ok["validation_selected_accuracy"], errors="coerce").mean()) if not ok.empty else float("nan"),
                "mean_interference_score": float(pd.to_numeric(ok["interference_score"], errors="coerce").mean()) if not ok.empty else float("nan"),
                "mean_base_epochs": float(pd.to_numeric(ok["base_epochs"], errors="coerce").mean()) if not ok.empty else float("nan"),
                "mean_finetune_epochs": float(pd.to_numeric(ok["finetune_epochs"], errors="coerce").mean()) if not ok.empty else float("nan"),
                "mean_delta_vs_greedy_soup": float(np.nanmean(delta_greedy)) if len(delta_greedy) else float("nan"),
                "delta_vs_greedy_ci_low": dg_low,
                "delta_vs_greedy_ci_high": dg_high,
                "mean_delta_vs_task_arithmetic": float(np.nanmean(delta_task)) if len(delta_task) else float("nan"),
                "delta_vs_task_arithmetic_ci_low": dt_low,
                "delta_vs_task_arithmetic_ci_high": dt_high,
                "mean_delta_vs_ties": float(np.nanmean(delta_ties)) if len(delta_ties) else float("nan"),
                "delta_vs_ties_ci_low": tt_low,
                "delta_vs_ties_ci_high": tt_high,
                "mean_delta_vs_dare": float(np.nanmean(delta_dare)) if len(delta_dare) else float("nan"),
                "delta_vs_dare_ci_low": dd_low,
                "delta_vs_dare_ci_high": dd_high,
                "mean_sign_conflict_fraction": float(pd.to_numeric(ok["task_vector_sign_conflict_fraction"], errors="coerce").mean()) if not ok.empty else float("nan"),
                "mean_triangle_cycle_score": float(pd.to_numeric(ok["triangle_cycle_score"], errors="coerce").mean()) if not ok.empty else float("nan"),
            }
        )
    summary = pd.DataFrame(records)
    if not summary.empty:
        setting_cols = ["dataset", "task_preset", "architecture", "width", "n_tasks"]
        summary["best_single_model_method_by_mean_accuracy"] = ""
        summary["claim_decision"] = "descriptive_no_general_superiority_claim"
        summary["claim_boundary"] = "same-base task-vector benchmark; no broad superiority claim"
        for setting_key, setting_group in summary.groupby(setting_cols, dropna=False):
            setting_mask = np.ones(len(summary), dtype=bool)
            for col, value in zip(setting_cols, setting_key):
                setting_mask &= summary[col].eq(value).to_numpy()
            comparable = setting_group[
                (setting_group["n_ok_rows"] > 0)
                & setting_group["status"].astype(str).eq("ok")
                & ~setting_group["method"].astype(str).str.startswith("individual_finetuned_")
            ].copy()
            if comparable.empty:
                continue
            best = comparable.sort_values(["mean_average_test_accuracy", "mean_worst_task_accuracy"], ascending=[False, False]).iloc[0]
            best_method = str(best["method"])
            n_seeds = int(best["n_unique_seeds"])
            if n_seeds < 20:
                decision = "descriptive_below_20_seed_gate"
            elif best_method == "greedy_soup":
                decision = "greedy_soup_empirical_descent_best_in_exact_setting"
            elif best_method in {"task_arithmetic", "ties_merging", "dare", "slerp_sequential"}:
                if safe_float(best["delta_vs_greedy_ci_low"]) > 0.0:
                    decision = "supported_exact_setting_delta_vs_greedy"
                else:
                    decision = "descriptive_best_mean_ci_overlaps_greedy"
            else:
                decision = "descriptive_no_general_superiority_claim"
            boundary = (
                f"exact setting only; n_unique_seeds={n_seeds}; "
                "same-base task-vector regime, not independent-seed rebasin"
            )
            summary.loc[setting_mask, "best_single_model_method_by_mean_accuracy"] = best_method
            summary.loc[setting_mask, "claim_decision"] = decision
            summary.loc[setting_mask, "claim_boundary"] = boundary
    return summary


def md_table(df: pd.DataFrame, columns: list[str], max_rows: int = 40) -> str:
    if df.empty:
        return "_No rows._"
    view = df.head(max_rows).copy()
    for col in columns:
        if col not in view.columns:
            view[col] = ""
    return format_markdown_table(view[columns].to_dict("records"), columns)


def plot_deltas(summary: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    ok = summary[(summary["n_ok_rows"] > 0) & (~summary["method"].astype(str).str.startswith("individual_finetuned_"))].copy()
    ok = ok[~ok["method"].isin(["individual_finetuned_mean"])]
    fig, ax = plt.subplots(figsize=(12, max(5.2, 0.26 * max(len(ok), 1))))
    if ok.empty:
        ax.text(0.5, 0.5, "No summary rows", ha="center", va="center")
        ax.set_axis_off()
    else:
        ok = ok.sort_values("mean_delta_vs_greedy_soup")
        x = np.arange(len(ok))
        y = ok["mean_delta_vs_greedy_soup"].to_numpy(dtype=float)
        lo = ok["delta_vs_greedy_ci_low"].to_numpy(dtype=float)
        hi = ok["delta_vs_greedy_ci_high"].to_numpy(dtype=float)
        err = np.vstack([np.maximum(y - lo, 0.0), np.maximum(hi - y, 0.0)])
        labels = [
            f"{row.dataset}/{row.task_preset}/W{int(row.width)}\n{row.method}"
            for row in ok.itertuples(index=False)
        ]
        ax.barh(x, y, color="tab:blue", alpha=0.78)
        ax.errorbar(y, x, xerr=err, fmt="none", ecolor="black", capsize=3, linewidth=0.8)
        ax.axvline(0.0, color="black", linewidth=0.8)
        ax.set_yticks(x)
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel("mean accuracy delta vs greedy soup")
        ax.set_title("Same-base task-vector methods by fixed setting")
        ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_latex(summary: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = summary[summary["n_ok_rows"] > 0].copy()
    keep_methods = ["base_model", "weight_average", "greedy_soup", "slerp_sequential", "task_arithmetic", "ties_merging", "dare"]
    ok = ok[ok["method"].isin(keep_methods)]
    lines = [
        r"\begin{tabular}{lllrrrr}",
        r"\toprule",
        r"Setting & Method & Seeds & Avg. acc. & Worst acc. & $\Delta$ Soup & Interference \\",
        r"\midrule",
    ]
    for row in ok.sort_values(["dataset", "task_preset", "width", "mean_average_test_accuracy"], ascending=[True, True, True, False]).itertuples(index=False):
        setting = f"{row.dataset}/{row.task_preset}/W{int(row.width)}".replace("_", "\\_")
        method_name = str(row.method).replace("_", "\\_")
        lines.append(
            f"{setting} & "
            f"{method_name} & "
            f"{int(row.n_unique_seeds)} & "
            f"{row.mean_average_test_accuracy:.4f} & "
            f"{row.mean_worst_task_accuracy:.4f} & "
            f"{row.mean_delta_vs_greedy_soup:.4f} & "
            f"{row.mean_interference_score:.4f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_report(args, rows: pd.DataFrame, summary: pd.DataFrame, candidate_grid: pd.DataFrame, path: Path) -> None:
    best = summary[
        (summary["n_ok_rows"] > 0)
        & summary["status"].astype(str).eq("ok")
        & ~summary["method"].astype(str).str.startswith("individual_finetuned_")
    ].sort_values("mean_average_test_accuracy", ascending=False).head(1)
    oracle = summary[summary["method"].eq("individual_finetuned_mean")].sort_values(
        "mean_average_test_accuracy",
        ascending=False,
    ).head(1)
    if best.empty:
        headline = "No completed method rows."
    else:
        row = best.iloc[0]
        headline = (
            f"Best single-model mean accuracy across completed settings: `{row['method']}` "
            f"on `{row['dataset']}/{row['task_preset']}/W{int(row['width'])}` "
            f"at {row['mean_average_test_accuracy']:.4f}; claim decision `{row.get('claim_decision', 'descriptive')}`."
        )
        if not oracle.empty:
            oracle_row = oracle.iloc[0]
            headline += (
                f" The strongest per-task fine-tuned oracle summary row is `{oracle_row['mean_average_test_accuracy']:.4f}` "
                "and is not a single merged model."
            )
    completed = summary[summary["n_ok_rows"] > 0].drop_duplicates(["dataset", "task_preset", "architecture", "width", "n_tasks"])
    completed_settings = md_table(
        completed,
        ["dataset", "task_preset", "architecture", "width", "n_tasks", "n_unique_seeds", "mean_base_epochs", "mean_finetune_epochs", "claim_decision", "claim_boundary"],
        40,
    )
    report = f"""# Same-Base Task-Vector Benchmark

Generated by `experiments/same_base_task_vector_benchmark.py`.

## Exact Command

```bash
{args.command_string}
```

## Scope

This benchmark trains a common `mlp2` base checkpoint, fine-tunes task copies from that base, and compares methods that require a shared coordinate system: Model Soups, SLERP, Task Arithmetic, TIES, and DARE.

This is not an independent-seed/rebasin benchmark. Git-ReBasin and C2M3 rows are recorded as secondary not-run diagnostics because the current setup uses a common base and does not intentionally create independent permutation mismatch.

No paper prose is written here. Claim decisions are exact-setting, validation-safe, and gated by paired bootstrap confidence intervals and completed seed counts.

## Headline

{headline}

## Outputs

- `reports/csv/same_base_task_vector_benchmark.csv`
- `reports/csv/same_base_task_vector_summary.csv`
- `reports/csv/same_base_task_vector_candidate_grid.csv`
- `reports/same_base_task_vector_report.md`
- `reports/plots/same_base_task_vector_deltas.pdf`
- `reports/tables/same_base_task_vector_table.tex`

## Completed Settings

{completed_settings}

## Method Summary

{md_table(summary, ["dataset", "task_preset", "width", "method", "status", "n_ok_rows", "n_unique_seeds", "mean_average_test_accuracy", "accuracy_ci_low", "accuracy_ci_high", "mean_worst_task_accuracy", "mean_validation_selected_accuracy", "mean_interference_score", "mean_delta_vs_greedy_soup", "delta_vs_greedy_ci_low", "delta_vs_greedy_ci_high", "claim_decision"], 120)}

## Task-Vector Diagnostics

{md_table(summary.drop_duplicates(["dataset", "task_preset", "width"]), ["dataset", "task_preset", "width", "mean_sign_conflict_fraction", "mean_triangle_cycle_score", "claim_decision"], 40)}

## Validation Candidate Grid

Task Arithmetic scale, TIES density/scale, and DARE drop-rate/scale are selected by validation accuracy and validation loss only. The candidate grid contains no test metrics and records greedy-soup accept/reject validation decisions separately.

{md_table(candidate_grid, ["dataset", "task_preset", "width", "seed", "method", "candidate_kind", "candidate_rank", "candidate_params_json", "validation_accuracy", "validation_loss", "accepted", "selected", "uses_test_metrics_for_selection"], 30)}

## Greedy Soup Acceptance Sample

{md_table(rows[rows["method"].eq("greedy_soup")], ["run_id", "method", "average_test_accuracy", "worst_task_accuracy", "validation_selected_accuracy", "greedy_soup_acceptance_json"], 10)}

## Claim Boundaries

- If greedy soup wins, interpret it as empirical validation descent selecting compatible task vectors in this same-base regime.
- If Task Arithmetic, TIES, or DARE wins, interpret it as task-vector algebra outperforming simple validation descent in this same-base regime.
- This run does not support general superiority unless paired confidence intervals and multiple task families support the exact claim.
- C2M3/Git-ReBasin are not primary baselines here because this benchmark intentionally avoids independent-seed permutation mismatch.
"""
    path.write_text(report, encoding="utf-8")


def update_claims_audit(summary: pd.DataFrame, path: Path) -> None:
    if not path.exists():
        return
    best = summary[
        (summary["n_ok_rows"] > 0)
        & summary["status"].astype(str).eq("ok")
        & ~summary["method"].astype(str).str.startswith("individual_finetuned_")
    ].sort_values("mean_average_test_accuracy", ascending=False).head(1)
    if best.empty:
        status = "Not yet supported"
        evidence = "the same-base task-vector benchmark did not produce completed rows"
    else:
        row = best.iloc[0]
        completed_settings = summary[summary["n_ok_rows"] > 0].drop_duplicates(["dataset", "task_preset", "architecture", "width", "n_tasks"])
        n_completed = int(len(completed_settings))
        min_seeds = int(completed_settings["n_unique_seeds"].min()) if n_completed else 0
        status = "Supported descriptive"
        evidence = (
            f"`reports/same_base_task_vector_report.md` records a same-base task-vector benchmark; "
            f"{n_completed} completed fixed settings with minimum `{min_seeds}` seeds; "
            f"best mean method `{row['method']}` on `{row['dataset']}/{row['task_preset']}/W{int(row['width'])}` "
            f"with mean accuracy `{row['mean_average_test_accuracy']:.4f}`; "
            "the report preserves validation-only selection and no-broad-superiority boundaries."
        )
    audit_row = (
        "| Same-base task-vector baselines are evaluated separately from independent-seed rebasin baselines. "
        f"| {status} | {evidence} |"
    )
    text = path.read_text(encoding="utf-8")
    marker = "Same-base task-vector baselines are evaluated separately from independent-seed rebasin baselines."
    lines = text.splitlines()
    replaced = False
    for idx, line in enumerate(lines):
        if marker in line:
            lines[idx] = audit_row
            replaced = True
            break
    if replaced:
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    insert_marker = "<!-- prompt10-claim-audit:start -->"
    if insert_marker in text:
        text = text.replace(insert_marker, audit_row + "\n\n" + insert_marker)
    else:
        text = text.rstrip() + "\n" + audit_row + "\n"
    path.write_text(text, encoding="utf-8")


def parse_float_list(text: str) -> list[float]:
    return [float(item.strip()) for item in str(text).split(",") if item.strip()]


def parse_csv(text: str, cast=str) -> list:
    return [cast(item.strip()) for item in str(text).split(",") if item.strip()]


def parse_int_list(text: str) -> list[int]:
    return [int(item.strip()) for item in str(text).split(",") if item.strip()]


def compatible_task_preset(dataset_name: str, task_preset: str) -> bool:
    if task_preset.startswith("fashion"):
        return dataset_name == "fashion_mnist"
    if task_preset.startswith("mnist"):
        return dataset_name == "mnist"
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="mnist")
    parser.add_argument("--task-preset", default="mnist_digit_subsets", choices=sorted(TASK_PRESETS))
    parser.add_argument("--task-presets", default="")
    parser.add_argument("--architecture", default="mlp2", choices=["mlp2"])
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--widths", default="")
    parser.add_argument("--seeds", default="7200:7203")
    parser.add_argument("--base-epochs", type=int, default=3)
    parser.add_argument("--finetune-epochs", type=int, default=2)
    parser.add_argument("--max-train-samples", type=int, default=6000)
    parser.add_argument("--max-test-samples", type=int, default=2000)
    parser.add_argument("--max-base-train-samples", type=int, default=5000)
    parser.add_argument("--max-task-train-samples", type=int, default=1800)
    parser.add_argument("--max-task-val-samples", type=int, default=600)
    parser.add_argument("--max-task-test-samples", type=int, default=600)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--finetune-lr", type=float, default=5e-4)
    parser.add_argument("--optimizer", default="adamw", choices=["adam", "adamw", "sgd"])
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--scheduler", default="cosine", choices=["none", "cosine", "step"])
    parser.add_argument("--step-size", type=int, default=3)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--augmentation", default="none")
    parser.add_argument("--dataset-seed", type=int, default=314159)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "reports")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--task-arithmetic-scales", default="0.25,0.5,0.75,1.0,1.25")
    parser.add_argument("--ties-densities", default="0.2,0.5,1.0")
    parser.add_argument("--ties-scales", default="0.5,1.0,1.25")
    parser.add_argument("--dare-drop-rates", default="0.1,0.3,0.5")
    parser.add_argument("--dare-scales", default="0.5,1.0,1.25")
    parser.add_argument("--update-claims-audit", action="store_true", default=True)
    parser.add_argument("--no-update-claims-audit", action="store_false", dest="update_claims_audit")
    args = parser.parse_args()
    args.task_arithmetic_scales = parse_float_list(args.task_arithmetic_scales)
    args.ties_densities = parse_float_list(args.ties_densities)
    args.ties_scales = parse_float_list(args.ties_scales)
    args.dare_drop_rates = parse_float_list(args.dare_drop_rates)
    args.dare_scales = parse_float_list(args.dare_scales)
    args.task_presets = parse_csv(args.task_presets, str) if args.task_presets else [args.task_preset]
    args.width_list = parse_int_list(args.widths) if args.widths else [int(args.width)]
    args.command_string = " ".join([sys.executable, *sys.argv])
    return args


def main() -> None:
    args = parse_args()
    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    seeds = parse_seeds(args.seeds)
    all_rows = []
    all_candidate_rows = []
    for dataset_name in datasets:
        for task_preset in args.task_presets:
            if not compatible_task_preset(dataset_name, task_preset):
                print(f"skipping incompatible setting dataset={dataset_name} task_preset={task_preset}", flush=True)
                continue
            task_defs = TASK_PRESETS[task_preset]
            for width in args.width_list:
                run_args = argparse.Namespace(**vars(args))
                run_args.task_preset = task_preset
                run_args.width = int(width)
                for seed in seeds:
                    print(f"running {dataset_name} preset={task_preset} width={width} seed {seed}", flush=True)
                    rows, candidate_rows = run_one_seed(run_args, dataset_name, seed, task_defs)
                    all_rows.extend(rows)
                    all_candidate_rows.extend(candidate_rows)
    rows_df = pd.DataFrame(all_rows)
    candidate_df = pd.DataFrame(all_candidate_rows)
    summary_df = summarize(rows_df, args.bootstrap_samples)

    csv_dir = args.reports_dir / "csv"
    plot_dir = args.reports_dir / "plots"
    table_dir = args.reports_dir / "tables"
    csv_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    rows_path = csv_dir / "same_base_task_vector_benchmark.csv"
    summary_path = csv_dir / "same_base_task_vector_summary.csv"
    candidate_grid_path = csv_dir / "same_base_task_vector_candidate_grid.csv"
    report_path = args.reports_dir / "same_base_task_vector_report.md"
    plot_path = plot_dir / "same_base_task_vector_deltas.pdf"
    table_path = table_dir / "same_base_task_vector_table.tex"
    rows_df.to_csv(rows_path, index=False, lineterminator="\n")
    summary_df.to_csv(summary_path, index=False, lineterminator="\n")
    candidate_df.to_csv(candidate_grid_path, index=False, lineterminator="\n")
    plot_deltas(summary_df, plot_path)
    write_latex(summary_df, table_path)
    write_report(args, rows_df, summary_df, candidate_df, report_path)
    if args.update_claims_audit:
        update_claims_audit(summary_df, args.reports_dir / "claims_audit.md")
    print(f"wrote {rows_path}")
    print(f"wrote {summary_path}")
    print(f"wrote {candidate_grid_path}")
    print(f"wrote {report_path}")
    print(f"wrote {plot_path}")
    print(f"wrote {table_path}")


if __name__ == "__main__":
    main()
