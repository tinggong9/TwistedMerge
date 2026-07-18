#!/usr/bin/env python3
"""Budget-controlled selector attribution on untouched MNIST MLP groups.

The experiment is deliberately separate from the historical selector sweep.
It trains new checkpoint groups, freezes A0--A5 from validation data, and only
then evaluates the candidate models on test data.  A6 is an explicitly
non-deployable test oracle.  The official Git Re-Basin and C2M3 entries use the
same pinned, adapter-assisted official cores as the existing official-baseline
integration; they are never relabelled as unmodified end-to-end runs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import resource
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.external_baseline_comparison import split_train_val  # noqa: E402
from experiments.post_iclr_official_baseline_integration import (  # noqa: E402
    git_rebasin_align,
    model_from_state,
    preservation_metrics,
)
from src.improved_monomial_merge import (  # noqa: E402
    build_scaled_models,
    greedy_soup_with_metadata,
    reference_log_scales_from_features,
)
from src.model_merging_benchmark import (  # noqa: E402
    average_models,
    collect_features,
    device_from_arg,
    load_dataset,
    make_loader,
    make_model,
    permute_model_to_reference,
    require_torch,
    set_seed,
    synchronize_permutations,
    train_model,
)
from src.official_baseline_adapters import (  # noqa: E402
    average_state_dicts,
    official_c2m3_synchronized_states,
)
from src.structure_group_ladder import estimate_pairwise_permutations_from_activations  # noqa: E402


PHASE = "selector_attribution"
DEFAULT_OFFICIAL_ROOT = Path(
    "/Users/tinggong/Documents/Codex/2026-07-17/files-mentioned-by-the-user-you/"
    "work/official-baseline-sources"
)
DEFAULT_JAX_PYTHON = Path(
    "/Users/tinggong/Documents/Codex/2026-07-17/files-mentioned-by-the-user-you/"
    "work/official-git-rebasin-py312-venv/bin/python"
)

VARIANT_LABELS = {
    "A0": "ordinary greedy baseline",
    "A1": "official-synchronization pool",
    "A2": "gauge-only augmentation",
    "A3": "gauge plus soup augmentation",
    "A4": "diagnostic-only augmentation",
    "A5": "full TwistedMerge selector",
    "B0": "budget-matched ordinary soup control",
    "A6": "test oracle upper bound",
}

TWISTEDMERGE_FAMILIES = {
    "permutation_gauge",
    "positive_monomial_gauge",
    "permutation_gauge_soup",
    "monomial_gauge_soup",
    "union_gauge_soup",
}


@dataclass
class Candidate:
    name: str
    family: str
    model: object
    output_type: str
    tm_specific: bool
    merge_seconds: float
    generation_validation_evals: int
    details: dict = field(default_factory=dict)
    val_accuracy: float = math.nan
    val_loss: float = math.nan
    val_correct: np.ndarray | None = None
    val_losses: np.ndarray | None = None
    test_accuracy: float = math.nan
    test_loss: float = math.nan
    test_seconds: float = math.nan


def parse_csv(text: str, cast=str) -> list:
    return [cast(item.strip()) for item in text.split(",") if item.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def write_csv(path: Path, rows: list[dict], fields: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def metric_key(name: str, metrics: dict[str, dict[str, float]]) -> tuple[float, float, str]:
    item = metrics[name]
    return (-float(item["accuracy"]), float(item["loss"]), name)


def validation_choice(pool: Sequence[str], metrics: dict[str, dict[str, float]]) -> str:
    """Choose by validation accuracy, loss, then stable name; never reads test."""

    available = [name for name in pool if name in metrics]
    if not available:
        raise ValueError("selector pool has no evaluated candidates")
    return min(available, key=lambda name: metric_key(name, metrics))


def oracle_choice(pool: Sequence[str], test_metrics: dict[str, dict[str, float]]) -> str:
    """Test-only choice for A6; callers must label it non-deployable."""

    return validation_choice(pool, test_metrics)


def diagnostic_choice(
    a0_pool: Sequence[str],
    a1_pool: Sequence[str],
    metrics: dict[str, dict[str, float]],
    residual: float,
    threshold: float,
) -> str:
    """Frozen A4 rule: high residual falls back to A0, otherwise use A1."""

    pool = a0_pool if float(residual) > float(threshold) else a1_pool
    return validation_choice(pool, metrics)


def budget_match_names(base_pool: Sequence[str], target_size: int) -> list[str]:
    """Return a stable exact-size prefix and reject under-filled controls."""

    unique = list(dict.fromkeys(base_pool))
    if len(unique) < target_size:
        raise ValueError(f"budget control has {len(unique)} candidates, needs {target_size}")
    return unique[:target_size]


def bootstrap_group_ci(
    rows: pd.DataFrame,
    value_column: str,
    *,
    group_column: str = "seed",
    n_bootstrap: int = 4000,
    seed: int = 271828,
) -> tuple[float, float]:
    """Bootstrap independent training groups, averaging dependent settings within group."""

    grouped = rows.groupby(group_column, dropna=False)[value_column].mean().to_numpy(float)
    grouped = grouped[np.isfinite(grouped)]
    if grouped.size == 0:
        return math.nan, math.nan
    if grouped.size == 1 or n_bootstrap <= 0:
        value = float(grouped.mean())
        return value, value
    rng = np.random.default_rng(seed)
    samples = rng.choice(grouped, size=(n_bootstrap, grouped.size), replace=True).mean(axis=1)
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def weighted_average_models(models: Sequence, weights: np.ndarray, spec, width: int):
    torch, _, _ = require_torch()
    weights = np.asarray(weights, dtype=float)
    if len(models) != len(weights) or not np.isclose(weights.sum(), 1.0):
        raise ValueError("weights must match models and sum to one")
    merged = make_model("mlp", spec, width)
    states = [model.state_dict() for model in models]
    state = merged.state_dict()
    with torch.no_grad():
        for key in state:
            values = [src[key].detach().cpu() * float(weight) for src, weight in zip(states, weights)]
            state[key].copy_(torch.stack(values).sum(dim=0))
    merged.load_state_dict(state)
    return merged


def evaluate_arrays(model, loader, device) -> tuple[dict[str, float], np.ndarray, np.ndarray, float]:
    torch, _, _ = require_torch()
    model.to(device)
    model.eval()
    correct = []
    losses = []
    start = time.perf_counter()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            batch_losses = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
            correct.append((logits.argmax(dim=1) == labels).detach().cpu().numpy().astype(float))
            losses.append(batch_losses.detach().cpu().numpy().astype(float))
    elapsed = time.perf_counter() - start
    correct_array = np.concatenate(correct)
    loss_array = np.concatenate(losses)
    return (
        {"accuracy": float(correct_array.mean()), "loss": float(loss_array.mean())},
        correct_array,
        loss_array,
        elapsed,
    )


def add_candidate(
    candidates: dict[str, Candidate],
    candidate: Candidate,
    val_loader,
    device,
) -> None:
    if candidate.name in candidates:
        raise ValueError(f"duplicate candidate {candidate.name}")
    metrics, correct, losses, _elapsed = evaluate_arrays(candidate.model, val_loader, device)
    candidate.val_accuracy = metrics["accuracy"]
    candidate.val_loss = metrics["loss"]
    candidate.val_correct = correct
    candidate.val_losses = losses
    candidates[candidate.name] = candidate


def model_parameter_count(model) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def model_bytes(model) -> int:
    return int(sum(value.numel() * value.element_size() for value in model.state_dict().values()))


def dataset_files(data_dir: Path) -> list[Path]:
    mnist = data_dir / "MNIST"
    return sorted(path for path in mnist.rglob("*") if path.is_file()) if mnist.exists() else []


def dataset_checksum(data_dir: Path) -> str:
    digest = hashlib.sha256()
    for path in dataset_files(data_dir):
        digest.update(str(path.relative_to(data_dir)).encode())
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def checkpoint_path(args, setting_id: str, model_index: int) -> Path:
    return args.checkpoint_root / args.stage / setting_id / f"model_{model_index}.pt"


def train_checkpoint_group(args, spec, train_subset, seed: int, n_models: int, width: int, device, setting_id: str):
    torch, _, _ = require_torch()
    models = []
    rows = []
    start_group = time.perf_counter()
    for model_index in range(n_models):
        path = checkpoint_path(args, setting_id, model_index)
        model_seed = seed + 1000 * model_index + 17 * width + n_models
        start = time.perf_counter()
        if args.reuse_checkpoints and path.exists():
            payload = torch.load(path, map_location="cpu")
            model = make_model("mlp", spec, width)
            model.load_state_dict(payload["state_dict"])
            reused = True
        else:
            set_seed(model_seed)
            model = make_model("mlp", spec, width)
            train_loader = make_loader(train_subset, args.batch_size, shuffle=True, seed=model_seed + 11)
            train_model(model, train_loader, args.epochs, args.lr, device)
            model.to("cpu")
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "metadata": {
                        "phase": PHASE,
                        "stage": args.stage,
                        "setting_id": setting_id,
                        "seed": seed,
                        "model_seed": model_seed,
                        "model_index": model_index,
                        "n_models": n_models,
                        "width": width,
                        "epochs": args.epochs,
                        "lr": args.lr,
                        "max_train_samples": args.max_train_samples,
                        "test_unseen_during_training": True,
                    },
                },
                path,
            )
            reused = False
        model.to("cpu")
        models.append(model)
        rows.append(
            {
                "setting_id": setting_id,
                "model_index": model_index,
                "model_seed": model_seed,
                "checkpoint": str(path),
                "checkpoint_sha256": sha256_file(path),
                "checkpoint_bytes": path.stat().st_size,
                "reused": reused,
                "training_seconds": time.perf_counter() - start,
            }
        )
    return models, rows, time.perf_counter() - start_group


def pool_definitions(candidates: dict[str, Candidate], n_models: int) -> dict[str, list[str]]:
    individuals = [f"individual_{index}" for index in range(n_models)]
    a0 = individuals + ["greedy_soup", "weight_average"]
    a1 = a0 + ["official_git_rebasin", "official_c2m3"]
    a2 = a0 + ["permutation_gauge_merge", "positive_monomial_gauge_merge"]
    a3 = a0 + [
        "permutation_gauge_merge",
        "positive_monomial_gauge_merge",
        "permutation_gauge_soup",
        "monomial_gauge_soup",
        "union_gauge_soup",
    ]
    a5 = list(dict.fromkeys(a1 + a2 + a3))
    return {
        "A0": [name for name in a0 if name in candidates],
        "A1": [name for name in a1 if name in candidates],
        "A2": [name for name in a2 if name in candidates],
        "A3": [name for name in a3 if name in candidates],
        "A5": [name for name in a5 if name in candidates],
    }


def add_soup_candidate(
    candidates: dict[str, Candidate],
    *,
    name: str,
    family: str,
    models: Sequence,
    labels: Sequence[str],
    val_loader,
    device,
    spec,
    width: int,
    tm_specific: bool,
) -> None:
    start = time.perf_counter()
    soup = greedy_soup_with_metadata(
        models,
        labels,
        val_loader,
        val_loader,
        device,
        "mlp",
        spec,
        width,
        evaluate_test=False,
    )
    candidate = Candidate(
        name=name,
        family=family,
        model=soup.model,
        output_type="same_capacity_model_soup",
        tm_specific=tm_specific,
        merge_seconds=time.perf_counter() - start,
        generation_validation_evals=2 * len(models),
        details={
            "selected_indices": soup.selected_indices,
            "selected_labels": soup.selected_labels,
            "ingredient_count": len(soup.selected_indices),
        },
    )
    add_candidate(candidates, candidate, val_loader, device)


def add_official_candidates(
    args,
    candidates: dict[str, Candidate],
    models: Sequence,
    spec,
    width: int,
    seed: int,
    val_loader,
    device,
    failures: list[dict],
    setting_id: str,
) -> None:
    states = [model.state_dict() for model in models]

    start = time.perf_counter()
    try:
        aligned = [states[0]]
        worker_metadata = []
        for index, state in enumerate(states[1:], start=1):
            converted, metadata = git_rebasin_align(args, states[0], state, seed=seed, index=index)
            aligned.append(converted)
            worker_metadata.append(metadata)
        preservation_error, disagreement = preservation_metrics(states, aligned, "mlp", spec, width, val_loader)
        merged_state = average_state_dicts(aligned)
        model = model_from_state(merged_state, "mlp", spec, width)
        add_candidate(
            candidates,
            Candidate(
                "official_git_rebasin",
                "official_synchronization",
                model,
                "same_capacity_single_model",
                False,
                time.perf_counter() - start,
                0,
                {
                    "implementation_kind": "adapter_assisted_official_core",
                    "source_commit": git_output("-C", str(args.official_root / "git-re-basin"), "rev-parse", "HEAD"),
                    "max_iter": args.git_rebasin_max_iter,
                    "functional_preservation_max_abs_error": preservation_error,
                    "prediction_disagreement": disagreement,
                    "worker_metadata": worker_metadata,
                },
            ),
            val_loader,
            device,
        )
    except Exception as error:  # official failure must remain visible
        failures.append(
            {
                "setting_id": setting_id,
                "candidate": "official_git_rebasin",
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )

    start = time.perf_counter()
    try:
        aligned, permutations, optimization = official_c2m3_synchronized_states(
            states,
            args.official_root / "c2m3",
            max_iter=args.c2m3_max_iter,
        )
        preservation_error, disagreement = preservation_metrics(states, aligned, "mlp", spec, width, val_loader)
        merged_state = average_state_dicts(aligned)
        model = model_from_state(merged_state, "mlp", spec, width)
        add_candidate(
            candidates,
            Candidate(
                "official_c2m3",
                "official_synchronization",
                model,
                "same_capacity_single_model",
                False,
                time.perf_counter() - start,
                0,
                {
                    "implementation_kind": "adapter_assisted_official_core",
                    "source_commit": git_output("-C", str(args.official_root / "c2m3"), "rev-parse", "HEAD"),
                    "max_iter": args.c2m3_max_iter,
                    "functional_preservation_max_abs_error": preservation_error,
                    "prediction_disagreement": disagreement,
                    "permutations": permutations,
                    "objective_values": [float(value) for value in optimization["obj_values"]],
                },
            ),
            val_loader,
            device,
        )
    except Exception as error:
        failures.append(
            {
                "setting_id": setting_id,
                "candidate": "official_c2m3",
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )


def resampled_metrics(candidates: dict[str, Candidate], indices: np.ndarray) -> dict[str, dict[str, float]]:
    output = {}
    for name, candidate in candidates.items():
        output[name] = {
            "accuracy": float(candidate.val_correct[indices].mean()),
            "loss": float(candidate.val_losses[indices].mean()),
        }
    return output


def selector_stability(
    candidates: dict[str, Candidate],
    pools: dict[str, list[str]],
    residual: float,
    threshold: float,
    seed: int,
    n_resamples: int,
) -> dict[str, dict]:
    n_val = len(next(iter(candidates.values())).val_correct)
    rng = np.random.default_rng(seed + 424242)
    choices = {variant: [] for variant in ["A0", "A1", "A2", "A3", "A4", "A5"]}
    for _ in range(n_resamples):
        indices = rng.choice(n_val, size=n_val, replace=True)
        metrics = resampled_metrics(candidates, indices)
        for variant in ["A0", "A1", "A2", "A3", "A5"]:
            choices[variant].append(validation_choice(pools[variant], metrics))
        choices["A4"].append(diagnostic_choice(pools["A0"], pools["A1"], metrics, residual, threshold))
    return {
        variant: {
            "resample_choices": values,
            "modal_fraction": max(values.count(item) for item in set(values)) / max(len(values), 1),
            "unique_choices": len(set(values)),
        }
        for variant, values in choices.items()
    }


def run_setting(args, spec, train_data, test_data, seed: int, n_models: int, width: int, diagnostic_threshold: float):
    device = device_from_arg(args.device)
    setting_id = f"post_iclr_v2_mnist_mlp_N{n_models}_W{width}_S{seed}"
    train_subset, val_subset = split_train_val(train_data, args.val_fraction, seed + 77)
    val_loader = make_loader(val_subset, args.batch_size, shuffle=False, seed=seed + 700)
    test_loader = make_loader(test_data, args.batch_size, shuffle=False, seed=seed + 999)
    match_loader = make_loader(train_subset, args.batch_size, shuffle=False, seed=seed + 501)
    split_hash = sha256_json(
        {
            "train": list(map(int, train_subset.indices)),
            "validation": list(map(int, val_subset.indices)),
            "test_dataset_seed": args.dataset_seed,
            "max_test_samples": args.max_test_samples,
        }
    )

    models, checkpoint_rows, training_seconds = train_checkpoint_group(
        args, spec, train_subset, seed, n_models, width, device, setting_id
    )
    candidates: dict[str, Candidate] = {}
    failures: list[dict] = []

    for index, model in enumerate(models):
        add_candidate(
            candidates,
            Candidate(
                f"individual_{index}",
                "individual",
                model,
                "same_capacity_single_model",
                False,
                0.0,
                0,
                {"model_index": index},
            ),
            val_loader,
            device,
        )

    start = time.perf_counter()
    weight_average = average_models(models, "mlp", spec, width)
    add_candidate(
        candidates,
        Candidate(
            "weight_average",
            "ordinary_average",
            weight_average,
            "same_capacity_single_model",
            False,
            time.perf_counter() - start,
            0,
        ),
        val_loader,
        device,
    )
    add_soup_candidate(
        candidates,
        name="greedy_soup",
        family="ordinary_soup",
        models=models,
        labels=[f"original:{index}" for index in range(n_models)],
        val_loader=val_loader,
        device=device,
        spec=spec,
        width=width,
        tm_specific=False,
    )

    add_official_candidates(
        args,
        candidates,
        models,
        spec,
        width,
        seed,
        val_loader,
        device,
        failures,
        setting_id,
    )

    start = time.perf_counter()
    features = {index: collect_features(model, match_loader, device) for index, model in enumerate(models)}
    pairwise = estimate_pairwise_permutations_from_activations(features, n_models, width)
    ref, synced, sync_disagreement = synchronize_permutations(pairwise, n_models)
    aligned = [
        permute_model_to_reference(model, "mlp", spec, width, synced[index])
        for index, model in enumerate(models)
    ]
    permutation_merge = average_models(aligned, "mlp", spec, width)
    permutation_seconds = time.perf_counter() - start
    add_candidate(
        candidates,
        Candidate(
            "permutation_gauge_merge",
            "permutation_gauge",
            permutation_merge,
            "same_capacity_single_model",
            True,
            permutation_seconds,
            0,
            {"reference": ref, "synchronization_disagreement": sync_disagreement},
        ),
        val_loader,
        device,
    )

    start = time.perf_counter()
    log_scales = reference_log_scales_from_features(features, synced, ref=ref, width=width)
    scaled = build_scaled_models(models, spec, width, synced, log_scales)
    monomial_merge = average_models(scaled, "mlp", spec, width)
    monomial_seconds = time.perf_counter() - start
    add_candidate(
        candidates,
        Candidate(
            "positive_monomial_gauge_merge",
            "positive_monomial_gauge",
            monomial_merge,
            "same_capacity_single_model",
            True,
            monomial_seconds,
            0,
            {
                "scale_source": "reference_activation_norm",
                "mean_abs_log_scale": float(np.mean(np.abs(log_scales))),
                "max_abs_log_scale": float(np.max(np.abs(log_scales))),
            },
        ),
        val_loader,
        device,
    )

    add_soup_candidate(
        candidates,
        name="permutation_gauge_soup",
        family="permutation_gauge_soup",
        models=aligned,
        labels=[f"permutation:{index}" for index in range(n_models)],
        val_loader=val_loader,
        device=device,
        spec=spec,
        width=width,
        tm_specific=True,
    )
    add_soup_candidate(
        candidates,
        name="monomial_gauge_soup",
        family="monomial_gauge_soup",
        models=scaled,
        labels=[f"monomial:{index}" for index in range(n_models)],
        val_loader=val_loader,
        device=device,
        spec=spec,
        width=width,
        tm_specific=True,
    )
    union_models = [*models, *aligned, *scaled]
    union_labels = (
        [f"original:{index}" for index in range(n_models)]
        + [f"permutation:{index}" for index in range(n_models)]
        + [f"monomial:{index}" for index in range(n_models)]
    )
    add_soup_candidate(
        candidates,
        name="union_gauge_soup",
        family="union_gauge_soup",
        models=union_models,
        labels=union_labels,
        val_loader=val_loader,
        device=device,
        spec=spec,
        width=width,
        tm_specific=True,
    )

    pools = pool_definitions(candidates, n_models)
    a5_target = len(pools["A5"])
    rng = np.random.default_rng(seed + 104729 * n_models + width)
    ordinary_pool = list(pools["A0"])
    ordinary_index = 0
    while len(ordinary_pool) < a5_target:
        weights = rng.dirichlet(np.ones(n_models))
        start = time.perf_counter()
        model = weighted_average_models(models, weights, spec, width)
        name = f"ordinary_weighted_soup_{ordinary_index}"
        add_candidate(
            candidates,
            Candidate(
                name,
                "ordinary_soup",
                model,
                "same_capacity_model_soup",
                False,
                time.perf_counter() - start,
                0,
                {"frozen_weights": weights.tolist(), "generation_seed": seed + 104729 * n_models + width},
            ),
            val_loader,
            device,
        )
        ordinary_pool.append(name)
        ordinary_index += 1
    pools["B0"] = budget_match_names(ordinary_pool, a5_target)

    # Freeze all deployable choices before any candidate is evaluated on test.
    val_metrics = {
        name: {"accuracy": candidate.val_accuracy, "loss": candidate.val_loss}
        for name, candidate in candidates.items()
    }
    choices = {variant: validation_choice(pools[variant], val_metrics) for variant in ["A0", "A1", "A2", "A3", "A5", "B0"]}
    choices["A4"] = diagnostic_choice(
        pools["A0"], pools["A1"], val_metrics, sync_disagreement, diagnostic_threshold
    )
    pools["A4"] = pools["A1"]
    stability = selector_stability(
        candidates,
        pools,
        sync_disagreement,
        diagnostic_threshold,
        seed,
        args.stability_resamples,
    )

    for candidate in candidates.values():
        metrics, _correct, _losses, elapsed = evaluate_arrays(candidate.model, test_loader, device)
        candidate.test_accuracy = metrics["accuracy"]
        candidate.test_loss = metrics["loss"]
        candidate.test_seconds = elapsed

    test_metrics = {
        name: {"accuracy": candidate.test_accuracy, "loss": candidate.test_loss}
        for name, candidate in candidates.items()
    }
    choices["A6"] = oracle_choice(pools["A5"], test_metrics)
    pools["A6"] = pools["A5"]

    selector_rows = []
    greedy_test = candidates["greedy_soup"].test_accuracy
    oracle_test = candidates[choices["A6"]].test_accuracy
    for variant in ["A0", "A1", "A2", "A3", "A4", "A5", "B0", "A6"]:
        selected = candidates[choices[variant]]
        selector_rows.append(
            {
                "setting_id": setting_id,
                "seed": seed,
                "n_models": n_models,
                "width": width,
                "selector": variant,
                "selector_label": VARIANT_LABELS[variant],
                "selected_candidate": selected.name,
                "selected_family": selected.family,
                "selected_tm_specific": selected.tm_specific,
                "test_accuracy": selected.test_accuracy,
                "test_loss": selected.test_loss,
                "val_accuracy": selected.val_accuracy,
                "val_loss": selected.val_loss,
                "regret_vs_greedy_soup": greedy_test - selected.test_accuracy,
                "regret_vs_oracle": oracle_test - selected.test_accuracy,
                "pool_size": len(pools[variant]),
                "selection_validation_evals": 0 if variant == "A6" else len(pools[variant]),
                "oracle_uses_test": variant == "A6",
                "deployable": variant != "A6",
                "validation_test_best_agreement": selected.name == choices["A6"],
                "stability_modal_fraction": math.nan if variant in {"B0", "A6"} else stability[variant]["modal_fraction"],
                "stability_unique_choices": math.nan if variant in {"B0", "A6"} else stability[variant]["unique_choices"],
                "sync_disagreement": sync_disagreement,
                "diagnostic_threshold": diagnostic_threshold,
                "diagnostic_fallback_to_a0": variant == "A4" and sync_disagreement > diagnostic_threshold,
                "split_sha256": split_hash,
            }
        )

    selected_by = {
        name: ";".join(variant for variant, choice in choices.items() if choice == name)
        for name in candidates
    }
    run_rows = []
    resource_rows = []
    peak_memory_mb = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0)
    for candidate in candidates.values():
        row = {
            "setting_id": setting_id,
            "seed": seed,
            "n_models": n_models,
            "width": width,
            "candidate": candidate.name,
            "family": candidate.family,
            "tm_specific": candidate.tm_specific,
            "output_type": candidate.output_type,
            "val_accuracy": candidate.val_accuracy,
            "val_loss": candidate.val_loss,
            "test_accuracy": candidate.test_accuracy,
            "test_loss": candidate.test_loss,
            "selected_by": selected_by[candidate.name],
            "merge_seconds": candidate.merge_seconds,
            "generation_validation_evals": candidate.generation_validation_evals,
            "details": json.dumps(candidate.details, sort_keys=True),
            "split_sha256": split_hash,
            "status": "evaluated",
        }
        run_rows.append(row)
        resource_rows.append(
            {
                "setting_id": setting_id,
                "candidate": candidate.name,
                "output_type": candidate.output_type,
                "parameter_multiplier": 1.0,
                "active_parameters": model_parameter_count(candidate.model),
                "stored_checkpoint_bytes": model_bytes(candidate.model),
                "inference_multiplier": 1.0,
                "branches": 1,
                "same_capacity_single_model": True,
                "merge_compute_seconds": candidate.merge_seconds,
                "training_compute_seconds_group": training_seconds,
                "test_inference_seconds": candidate.test_seconds,
                "test_samples": len(test_data),
                "peak_process_memory_mb": peak_memory_mb,
                "generation_validation_evals": candidate.generation_validation_evals,
                "selector_validation_eval": 1,
            }
        )

    budget_rows = []
    for variant in ["A0", "A1", "A2", "A3", "A4", "A5", "B0", "A6"]:
        pool = pools[variant]
        budget_rows.append(
            {
                "setting_id": setting_id,
                "selector": variant,
                "candidate_pool_size": len(pool),
                "selection_validation_evals": 0 if variant == "A6" else len(pool),
                "generation_validation_evals": sum(candidates[name].generation_validation_evals for name in pool),
                "hyperparameter_configurations": 0,
                "candidate_generation_merge_seconds": sum(candidates[name].merge_seconds for name in pool),
                "same_pool_size_as_a5": len(pool) == len(pools["A5"]),
                "same_selection_validation_evals_as_a5": variant != "A6" and len(pool) == len(pools["A5"]),
                "generation_compute_exactly_matched_to_a5": variant == "A5",
                "budget_note": (
                    "exact candidate-count and selector-evaluation match; generation kernels differ and are reported"
                    if variant == "B0"
                    else "A6 uses test and has no validation-selection budget"
                    if variant == "A6"
                    else "native variant budget"
                ),
            }
        )

    setting_row = {
        "setting_id": setting_id,
        "seed": seed,
        "n_models": n_models,
        "width": width,
        "sync_disagreement": sync_disagreement,
        "diagnostic_threshold": diagnostic_threshold,
        "train_size": len(train_subset),
        "validation_size": len(val_subset),
        "test_size": len(test_data),
        "training_seconds": training_seconds,
        "candidate_count": len(candidates),
        "failed_candidates": len(failures),
        "split_sha256": split_hash,
    }
    return {
        "runs": run_rows,
        "selectors": selector_rows,
        "budgets": budget_rows,
        "resources": resource_rows,
        "checkpoints": checkpoint_rows,
        "failures": failures,
        "setting": setting_row,
        "stability": [
            {
                "setting_id": setting_id,
                "selector": variant,
                **values,
                "resample_choices": json.dumps(values["resample_choices"]),
            }
            for variant, values in stability.items()
        ],
    }


def selector_summary(selectors: pd.DataFrame) -> list[dict]:
    rows = []
    for selector, group in selectors.groupby("selector", sort=False):
        values = group["test_accuracy"].to_numpy(float)
        rows.append(
            {
                "selector": selector,
                "selector_label": VARIANT_LABELS[selector],
                "n_exact_settings": len(group),
                "n_training_groups": group["seed"].nunique(),
                "failed_runs": 0,
                "mean_test_accuracy": float(np.mean(values)),
                "median_test_accuracy": float(np.median(values)),
                "std_test_accuracy": float(np.std(values, ddof=1)) if len(values) > 1 else math.nan,
                "se_test_accuracy": float(np.std(values, ddof=1) / math.sqrt(len(values))) if len(values) > 1 else math.nan,
                "mean_test_loss": float(group["test_loss"].mean()),
                "mean_regret_vs_greedy_soup": float(group["regret_vs_greedy_soup"].mean()),
                "mean_regret_vs_oracle": float(group["regret_vs_oracle"].mean()),
                "worst_setting_accuracy": float(group["test_accuracy"].min()),
                "tm_specific_selection_frequency": float(group["selected_tm_specific"].mean()),
                "validation_test_best_agreement": float(group["validation_test_best_agreement"].mean()),
                "mean_stability_modal_fraction": float(group["stability_modal_fraction"].mean()),
                "deployable": bool(group["deployable"].all()),
            }
        )
    return rows


def paired_comparisons(selectors: pd.DataFrame, n_bootstrap: int) -> list[dict]:
    wide_acc = selectors.pivot(index=["setting_id", "seed", "n_models", "width"], columns="selector", values="test_accuracy").reset_index()
    wide_loss = selectors.pivot(index=["setting_id", "seed", "n_models", "width"], columns="selector", values="test_loss").reset_index()
    rows = []
    for baseline in ["A0", "A1", "A2", "A3", "A4", "B0"]:
        clean = wide_acc.dropna(subset=["A5", baseline]).copy()
        clean["delta"] = clean["A5"] - clean[baseline]
        low, high = bootstrap_group_ci(clean, "delta", n_bootstrap=n_bootstrap, seed=9910 + len(rows))
        losses = wide_loss.dropna(subset=["A5", baseline]).copy()
        loss_delta = losses["A5"] - losses[baseline]
        delta = clean["delta"]
        standard = float(delta.std(ddof=1)) if len(delta) > 1 else math.nan
        rows.append(
            {
                "comparison": f"A5_minus_{baseline}",
                "method": "A5",
                "baseline": baseline,
                "n_exact_settings": len(clean),
                "n_training_groups": clean["seed"].nunique(),
                "paired_mean_accuracy_delta": float(delta.mean()),
                "paired_accuracy_delta_ci_low": low,
                "paired_accuracy_delta_ci_high": high,
                "paired_mean_loss_delta": float(loss_delta.mean()),
                "wins": int((delta > 0).sum()),
                "ties": int((delta == 0).sum()),
                "losses": int((delta < 0).sum()),
                "paired_effect_size_dz": float(delta.mean() / standard) if standard and np.isfinite(standard) else math.nan,
                "bootstrap_unit": "independent training-group seed; N and width settings averaged within seed",
            }
        )
    a4 = wide_acc.dropna(subset=["A4", "A1"]).copy()
    a4["delta"] = a4["A4"] - a4["A1"]
    low, high = bootstrap_group_ci(a4, "delta", n_bootstrap=n_bootstrap, seed=10001)
    rows.append(
        {
            "comparison": "A4_minus_A1_same_pool_diagnostic_rule",
            "method": "A4",
            "baseline": "A1",
            "n_exact_settings": len(a4),
            "n_training_groups": a4["seed"].nunique(),
            "paired_mean_accuracy_delta": float(a4["delta"].mean()),
            "paired_accuracy_delta_ci_low": low,
            "paired_accuracy_delta_ci_high": high,
            "wins": int((a4["delta"] > 0).sum()),
            "ties": int((a4["delta"] == 0).sum()),
            "losses": int((a4["delta"] < 0).sum()),
            "bootstrap_unit": "independent training-group seed; N and width settings averaged within seed",
        }
    )
    return rows


def selection_counts(selectors: pd.DataFrame) -> list[dict]:
    rows = []
    for (selector, family, candidate), group in selectors.groupby(
        ["selector", "selected_family", "selected_candidate"], dropna=False
    ):
        rows.append(
            {
                "selector": selector,
                "selected_family": family,
                "selected_candidate": candidate,
                "count": len(group),
                "frequency": len(group) / len(selectors[selectors["selector"].eq(selector)]),
                "tm_specific": bool(group["selected_tm_specific"].iloc[0]),
                "conditional_mean_gain_vs_a0": math.nan,
            }
        )
    a0 = selectors[selectors["selector"].eq("A0")][["setting_id", "test_accuracy"]].rename(columns={"test_accuracy": "a0_accuracy"})
    a5 = selectors[selectors["selector"].eq("A5")].merge(a0, on="setting_id", how="left")
    a5["gain"] = a5["test_accuracy"] - a5["a0_accuracy"]
    conditional = a5.groupby("selected_candidate")["gain"].mean().to_dict()
    for row in rows:
        if row["selector"] == "A5":
            row["conditional_mean_gain_vs_a0"] = conditional.get(row["selected_candidate"], math.nan)
    return rows


def make_plots(output_dir: Path, selectors: pd.DataFrame, paired: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots = output_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    order = ["A0", "A1", "A2", "A3", "A4", "A5", "B0", "A6"]
    means = selectors.groupby("selector")["test_accuracy"].mean().reindex(order)
    errors = selectors.groupby("selector")["test_accuracy"].sem().reindex(order).fillna(0.0)
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    colors = ["#6b7280" if item not in {"A5", "B0"} else "#2563eb" if item == "A5" else "#f59e0b" for item in order]
    ax.bar(order, means, yerr=errors, capsize=3, color=colors)
    ax.set_ylabel("MNIST test accuracy")
    ax.set_title("Selector attribution on untouched checkpoint groups")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots / "selector_accuracy.png", dpi=220)
    fig.savefig(plots / "selector_accuracy.pdf")
    plt.close(fig)

    row = paired[paired["comparison"].eq("A5_minus_B0")].iloc[0]
    paired_settings = selectors.pivot(index=["setting_id", "n_models", "width"], columns="selector", values="test_accuracy").reset_index()
    paired_settings["delta"] = paired_settings["A5"] - paired_settings["B0"]
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    x = np.arange(len(paired_settings))
    ax.axhline(0.0, color="black", linewidth=1)
    points = ax.scatter(x, paired_settings["delta"], c=paired_settings["width"], cmap="viridis", s=28)
    ax.axhspan(row["paired_accuracy_delta_ci_low"], row["paired_accuracy_delta_ci_high"], color="#2563eb", alpha=0.15)
    ax.axhline(row["paired_mean_accuracy_delta"], color="#2563eb", linestyle="--", label="paired mean")
    ax.set_xlabel("Exact checkpoint setting")
    ax.set_ylabel("A5 - budget-matched ordinary accuracy")
    ax.set_title("Budget-controlled full-selector attribution")
    ax.legend()
    colorbar = fig.colorbar(points, ax=ax, pad=0.015)
    colorbar.set_label("MLP width")
    colorbar.set_ticks(sorted(paired_settings["width"].unique()))
    fig.tight_layout()
    fig.savefig(plots / "paired_delta_a5_vs_budget_matched.png", dpi=220)
    fig.savefig(plots / "paired_delta_a5_vs_budget_matched.pdf")
    plt.close(fig)


def format_value(value) -> str:
    if isinstance(value, (float, np.floating)):
        return "NA" if not np.isfinite(value) else f"{float(value):.4f}"
    return str(value)


def latex_table(path: Path, summary: pd.DataFrame) -> None:
    rows = []
    for item in summary.itertuples(index=False):
        rows.append(
            f"{item.selector} & {item.n_exact_settings} & {item.mean_test_accuracy:.4f} & "
            f"{item.mean_regret_vs_greedy_soup:.4f} & {item.mean_regret_vs_oracle:.4f} & "
            f"{item.tm_specific_selection_frequency:.3f} \\\\"
        )
    text = """% Generated by experiments/post_iclr_selector_attribution.py
\\begin{table}[t]
\\centering
\\caption{Selector attribution on new, untouched MNIST MLP checkpoint groups. A6 is a non-deployable test oracle; B0 matches A5's candidate count and selector validation evaluations with ordinary soups.}
\\label{tab:post-iclr-selector-attribution}
\\begin{tabular}{lrrrrr}
\\toprule
Selector & Settings & Accuracy & Regret vs. soup & Regret vs. oracle & TM selection \\\\
\\midrule
""" + "\n".join(rows) + """
\\bottomrule
\\end{tabular}
\\end{table}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def claim_status(selectors: pd.DataFrame, paired: pd.DataFrame) -> tuple[str, list[dict]]:
    budget = paired[paired["comparison"].eq("A5_minus_B0")].iloc[0]
    diag = paired[paired["comparison"].eq("A4_minus_A1_same_pool_diagnostic_rule")].iloc[0]
    a5 = selectors[selectors["selector"].eq("A5")]
    b0 = selectors[selectors["selector"].eq("B0")]
    a0 = selectors[selectors["selector"].eq("A0")][["setting_id", "test_accuracy"]].rename(columns={"test_accuracy": "a0"})
    conditional = a5[a5["selected_tm_specific"]].merge(a0, on="setting_id", how="left")
    conditional_gain = float((conditional["test_accuracy"] - conditional["a0"]).mean()) if len(conditional) else math.nan
    criteria = [
        {
            "criterion": "A5 beats B0 with positive group-bootstrap CI",
            "passed": bool(budget["paired_accuracy_delta_ci_low"] > 0),
            "value": float(budget["paired_mean_accuracy_delta"]),
            "ci_low": float(budget["paired_accuracy_delta_ci_low"]),
            "ci_high": float(budget["paired_accuracy_delta_ci_high"]),
        },
        {
            "criterion": "TM-specific choice has nontrivial frequency and positive conditional gain",
            "passed": bool(a5["selected_tm_specific"].mean() >= 0.1 and np.isfinite(conditional_gain) and conditional_gain > 0),
            "value": float(a5["selected_tm_specific"].mean()),
            "conditional_gain": conditional_gain,
        },
        {
            "criterion": "A4 residual rule reduces regret with same pool",
            "passed": bool(diag["paired_accuracy_delta_ci_low"] > 0),
            "value": float(diag["paired_mean_accuracy_delta"]),
            "ci_low": float(diag["paired_accuracy_delta_ci_low"]),
            "ci_high": float(diag["paired_accuracy_delta_ci_high"]),
        },
        {
            "criterion": "A5 improves worst setting without material mean loss",
            "passed": bool(a5["test_accuracy"].min() > b0["test_accuracy"].min() and a5["test_accuracy"].mean() >= b0["test_accuracy"].mean() - 0.001),
            "a5_worst": float(a5["test_accuracy"].min()),
            "b0_worst": float(b0["test_accuracy"].min()),
        },
    ]
    status = "TwistedMerge-specific selector gain supported" if any(row["passed"] for row in criteria) else "enriched-pool selection; no TwistedMerge-specific algorithmic gain established"
    return status, criteria


def markdown_table(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    rows = ["| " + " | ".join(format_value(row[column]) for column in columns) + " |" for _, row in frame[columns].iterrows()]
    return "\n".join([header, divider, *rows])


def write_report(
    output_dir: Path,
    args,
    selectors: pd.DataFrame,
    summary: pd.DataFrame,
    paired: pd.DataFrame,
    counts: pd.DataFrame,
    failures: pd.DataFrame,
    status: str,
    criteria: list[dict],
) -> None:
    a5_b0 = paired[paired["comparison"].eq("A5_minus_B0")].iloc[0]
    a5 = selectors[selectors["selector"].eq("A5")]
    conditional = counts[(counts["selector"].eq("A5")) & counts["tm_specific"]]
    text = f"""# Post-ICLR selector-attribution report

## Verdict

**{status}.**

The confirmatory unit is the independent training-group seed. Settings sharing a seed across model count and width are averaged before the paired bootstrap. The study contains `{selectors['seed'].nunique()}` independent groups and `{selectors['setting_id'].nunique()}` exact checkpoint settings. Failed official candidates: `{len(failures)}`.

## Frozen protocol

- Stage: `{args.stage}`.
- Seeds: `{args.seeds}`; model counts: `{args.model_counts}`; widths: `{args.widths}`.
- Recipe: Adam, learning rate `{args.lr}`, `{args.epochs}` epochs, `{args.max_train_samples}` sampled MNIST training examples, validation fraction `{args.val_fraction}`.
- Every selector sees the identical checkpoint group in a setting.
- A0--A5 and B0 are frozen from validation metrics before test evaluation. A6 alone uses test metrics and is an oracle upper bound.
- B0 exactly matches A5's candidate count and selector validation-evaluation count. Candidate-generation kernels and compute are not exactly equal; `budget_audit.csv` reports the difference.
- A4 uses the frozen residual threshold `{float(a5['diagnostic_threshold'].iloc[0]):.8g}`: above threshold it falls back to A0; otherwise it chooses from A1.
- A5 includes no lift: the current claim ledger does not certify a natural-MNIST lift candidate.
- Git Re-Basin and C2M3 are adapter-assisted official cores, not unmodified end-to-end runs.

Smoke command:

```bash
{args.python_command} experiments/post_iclr_selector_attribution.py --stage smoke --data-dir {args.data_dir} --official-root {args.official_root} --jax-python {args.jax_python}
```

Confirmatory command:

```bash
{args.python_command} experiments/post_iclr_selector_attribution.py --stage confirmatory --data-dir {args.data_dir} --official-root {args.official_root} --jax-python {args.jax_python}
```

## Selector summary

{markdown_table(summary, ['selector', 'n_exact_settings', 'n_training_groups', 'mean_test_accuracy', 'median_test_accuracy', 'std_test_accuracy', 'mean_test_loss', 'mean_regret_vs_greedy_soup', 'mean_regret_vs_oracle', 'worst_setting_accuracy', 'tm_specific_selection_frequency', 'validation_test_best_agreement'])}

## Paired attribution

{markdown_table(paired, ['comparison', 'n_exact_settings', 'n_training_groups', 'paired_mean_accuracy_delta', 'paired_accuracy_delta_ci_low', 'paired_accuracy_delta_ci_high', 'wins', 'ties', 'losses', 'paired_effect_size_dz'])}

The primary controlled comparison A5 - B0 is `{a5_b0['paired_mean_accuracy_delta']:.4f}` with group-bootstrap 95% CI `[{a5_b0['paired_accuracy_delta_ci_low']:.4f}, {a5_b0['paired_accuracy_delta_ci_high']:.4f}]`.

## A5 selections

{markdown_table(counts[counts['selector'].eq('A5')], ['selected_family', 'selected_candidate', 'count', 'frequency', 'tm_specific', 'conditional_mean_gain_vs_a0'])}

TwistedMerge-specific selection frequency is `{a5['selected_tm_specific'].mean():.3f}`. Conditional rows are reported even when negative or absent.

## Preregistered success gates

```json
{json.dumps(criteria, indent=2, sort_keys=True)}
```

## Budget and stability interpretation

Pool size and final selector evaluation count are exactly controlled by B0. Generation compute cannot be made identical because official synchronization, exact gauges, and greedy soup construction use different kernels; it is timed and counted rather than hidden. Validation-resampling stability is computed from per-example validation losses and correctness without touching test labels.

## Capacity and cost

Every A0--A6 candidate is a single, same-width MLP at inference. Soup candidates are materialized as one averaged model. There are no ensembles, wider models, branch lifts, or rank lifts. Per-candidate parameters, stored bytes, latency, merge time, training time, peak process memory, branches, and validation evaluations are in `resource_accounting.csv`.

## Reproducibility and negative results

`config.json`, `checkpoint_manifest.csv`, `failure_log.csv`, and `artifact_manifest.csv` record the recipe, split and dataset checksums, checkpoint provenance, external source commits, environment, commands, and output hashes. Official failures are never replaced by internal methods. Per-setting unfavorable results remain in `runs.csv` and `selectors.csv`.

![Selector accuracy](plots/selector_accuracy.png)

![A5 versus budget-matched ordinary control](plots/paired_delta_a5_vs_budget_matched.png)
"""
    (output_dir / "report.md").write_text(text, encoding="utf-8")


def artifact_manifest(output_dir: Path, extra_paths: Sequence[Path]) -> list[dict]:
    files = sorted(path for path in output_dir.rglob("*") if path.is_file() and path.name != "artifact_manifest.csv")
    files.extend(path for path in extra_paths if path.exists())
    rows = []
    for path in sorted(set(files)):
        rows.append(
            {
                "path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return rows


def frozen_protocol() -> dict:
    return {
        "phase": PHASE,
        "confirmatory_seeds": list(range(9300, 9310)),
        "model_counts": [3, 4],
        "widths": [32, 64],
        "epochs": 3,
        "lr": 0.001,
        "max_train_samples": 5000,
        "max_test_samples": 0,
        "dataset_seed": 8128,
        "val_fraction": 0.2,
        "batch_size": 128,
        "optimizer": "adam",
        "augmentation": "none",
        "selection_tie_break": "validation accuracy descending, validation loss ascending, stable candidate name",
        "a4_rule": "if synchronization disagreement exceeds pilot median, use A0; otherwise use A1",
        "a5_lift_policy": "exclude: no certified natural-MNIST lift in current claim ledger",
        "test_policy": "freeze A0-A5/B0 choices before first test evaluation; A6 only is test-selected",
        "bootstrap_unit": "independent training-group seed",
    }


def stage_defaults(args) -> None:
    if args.stage == "smoke":
        args.seeds = args.seeds or "9100"
        args.model_counts = args.model_counts or "3"
        args.widths = args.widths or "32"
        args.epochs = args.epochs if args.epochs is not None else 1
        args.max_train_samples = args.max_train_samples if args.max_train_samples is not None else 512
        args.max_test_samples = args.max_test_samples if args.max_test_samples is not None else 512
        args.bootstrap_samples = min(args.bootstrap_samples, 200)
        args.stability_resamples = min(args.stability_resamples, 2)
    elif args.stage == "pilot":
        args.seeds = args.seeds or "9100,9101"
        args.model_counts = args.model_counts or "3,4"
        args.widths = args.widths or "32"
        args.epochs = args.epochs if args.epochs is not None else 2
        args.max_train_samples = args.max_train_samples if args.max_train_samples is not None else 2000
        args.max_test_samples = args.max_test_samples if args.max_test_samples is not None else 1000
        args.bootstrap_samples = min(args.bootstrap_samples, 500)
    else:
        frozen_path = args.output_root / "frozen_config.json"
        if not frozen_path.exists():
            raise FileNotFoundError(f"confirmatory run requires pilot-frozen protocol: {frozen_path}")
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
        protocol = frozen["protocol"]
        args.seeds = args.seeds or ",".join(map(str, protocol["confirmatory_seeds"]))
        args.model_counts = args.model_counts or ",".join(map(str, protocol["model_counts"]))
        args.widths = args.widths or ",".join(map(str, protocol["widths"]))
        args.epochs = args.epochs if args.epochs is not None else int(protocol["epochs"])
        args.max_train_samples = args.max_train_samples if args.max_train_samples is not None else int(protocol["max_train_samples"])
        args.max_test_samples = args.max_test_samples if args.max_test_samples is not None else int(protocol["max_test_samples"])
        args.diagnostic_threshold = float(frozen["diagnostic_threshold"])


def environment() -> dict:
    import matplotlib
    import torch
    import torchvision

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "matplotlib": matplotlib.__version__,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["smoke", "pilot", "confirmatory"], required=True)
    parser.add_argument("--seeds", default="")
    parser.add_argument("--model-counts", default="")
    parser.add_argument("--widths", default="")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-test-samples", type=int)
    parser.add_argument("--dataset-seed", type=int, default=8128)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--bootstrap-samples", type=int, default=4000)
    parser.add_argument("--stability-resamples", type=int, default=5)
    parser.add_argument("--diagnostic-threshold", type=float, default=0.0)
    parser.add_argument("--git-rebasin-max-iter", type=int, default=100)
    parser.add_argument("--c2m3-max-iter", type=int, default=30)
    parser.add_argument("--official-root", type=Path, default=DEFAULT_OFFICIAL_ROOT)
    parser.add_argument("--jax-python", type=Path, default=DEFAULT_JAX_PYTHON)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-root", type=Path, default=ROOT / "reports" / "post_iclr_v2" / PHASE)
    parser.add_argument("--checkpoint-root", type=Path, default=ROOT / "checkpoints" / "post_iclr_selector_attribution")
    parser.add_argument("--reuse-checkpoints", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    stage_defaults(args)
    args.command = " ".join([sys.executable, *sys.argv])
    args.python_command = sys.executable

    output_dir = args.output_root if args.stage == "confirmatory" else args.output_root / "stages" / args.stage
    output_dir.mkdir(parents=True, exist_ok=True)
    spec, train_data, test_data = load_dataset(
        "mnist",
        args.data_dir,
        args.max_train_samples,
        args.max_test_samples,
        args.dataset_seed,
        augmentation="none",
    )

    all_results = {key: [] for key in ["runs", "selectors", "budgets", "resources", "checkpoints", "failures", "setting", "stability"]}
    for seed in parse_csv(args.seeds, int):
        for n_models in parse_csv(args.model_counts, int):
            for width in parse_csv(args.widths, int):
                print(f"[{args.stage}] seed={seed} N={n_models} W={width}", flush=True)
                result = run_setting(
                    args,
                    spec,
                    train_data,
                    test_data,
                    seed,
                    n_models,
                    width,
                    args.diagnostic_threshold,
                )
                for key, rows in result.items():
                    all_results[key].extend(rows if isinstance(rows, list) else [rows])

    if args.stage == "pilot":
        threshold = float(pd.DataFrame(all_results["setting"])["sync_disagreement"].median())
        frozen = {
            "frozen_after_stage": "pilot",
            "diagnostic_threshold": threshold,
            "threshold_source": "median synchronization disagreement across pilot settings; no test metric used",
            "pilot_seeds": parse_csv(args.seeds, int),
            "protocol": frozen_protocol(),
            "frozen_at_git_commit": git_output("rev-parse", "HEAD"),
        }
        args.output_root.mkdir(parents=True, exist_ok=True)
        (args.output_root / "frozen_config.json").write_text(json.dumps(frozen, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    selectors = pd.DataFrame(all_results["selectors"])
    summary = pd.DataFrame(selector_summary(selectors))
    paired = pd.DataFrame(paired_comparisons(selectors, args.bootstrap_samples))
    counts = pd.DataFrame(selection_counts(selectors))
    failures = pd.DataFrame(all_results["failures"])
    if failures.empty:
        failures = pd.DataFrame(columns=["setting_id", "candidate", "status", "error_type", "error"])

    write_csv(output_dir / "runs.csv", all_results["runs"])
    write_csv(output_dir / "selectors.csv", all_results["selectors"])
    write_csv(output_dir / "summary.csv", summary.to_dict(orient="records"))
    write_csv(output_dir / "paired.csv", paired.to_dict(orient="records"))
    write_csv(output_dir / "selection_counts.csv", counts.to_dict(orient="records"))
    write_csv(output_dir / "budget_audit.csv", all_results["budgets"])
    write_csv(output_dir / "failure_log.csv", failures.to_dict(orient="records"), fields=list(failures.columns))
    write_csv(output_dir / "resource_accounting.csv", all_results["resources"])
    write_csv(output_dir / "checkpoint_manifest.csv", all_results["checkpoints"])
    write_csv(output_dir / "settings.csv", all_results["setting"])
    write_csv(output_dir / "stability.csv", all_results["stability"])

    status, criteria = claim_status(selectors, paired)
    (output_dir / "claim_status_update.json").write_text(
        json.dumps({"status": status, "criteria": criteria}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    make_plots(output_dir, selectors, paired)
    latex_path = ROOT / "reports" / "tables" / "post_iclr_selector_attribution.tex"
    if args.stage == "confirmatory":
        latex_table(latex_path, summary)

    config = {
        "phase": PHASE,
        "stage": args.stage,
        "exact_command": args.command,
        "smoke_command": f"{sys.executable} experiments/post_iclr_selector_attribution.py --stage smoke --data-dir {args.data_dir} --official-root {args.official_root} --jax-python {args.jax_python}",
        "pilot_command": f"{sys.executable} experiments/post_iclr_selector_attribution.py --stage pilot --data-dir {args.data_dir} --official-root {args.official_root} --jax-python {args.jax_python}",
        "confirmatory_command": f"{sys.executable} experiments/post_iclr_selector_attribution.py --stage confirmatory --data-dir {args.data_dir} --official-root {args.official_root} --jax-python {args.jax_python}",
        "git_commit_at_execution": git_output("rev-parse", "HEAD"),
        "git_worktree_dirty_at_execution": bool(git_output("status", "--porcelain")),
        "environment": environment(),
        "dataset": "torchvision MNIST; ToTensor only",
        "dataset_files": [str(path) for path in dataset_files(args.data_dir)],
        "dataset_sha256": dataset_checksum(args.data_dir),
        "dataset_seed": args.dataset_seed,
        "seeds": parse_csv(args.seeds, int),
        "model_initialization_seed_formula": "group_seed + 1000*model_index + 17*width + n_models",
        "model_counts": parse_csv(args.model_counts, int),
        "widths": parse_csv(args.widths, int),
        "epochs": args.epochs,
        "lr": args.lr,
        "optimizer": "adam",
        "max_train_samples": args.max_train_samples,
        "max_test_samples": args.max_test_samples,
        "val_fraction": args.val_fraction,
        "batch_size": args.batch_size,
        "device": args.device,
        "diagnostic_threshold": args.diagnostic_threshold,
        "stability_resamples": args.stability_resamples,
        "official_root": str(args.official_root),
        "jax_python": str(args.jax_python),
        "git_rebasin_source_commit": git_output("-C", str(args.official_root / "git-re-basin"), "rev-parse", "HEAD"),
        "git_rebasin_source_dirty": bool(git_output("-C", str(args.official_root / "git-re-basin"), "status", "--porcelain")),
        "c2m3_source_commit": git_output("-C", str(args.official_root / "c2m3"), "rev-parse", "HEAD"),
        "c2m3_source_dirty": bool(git_output("-C", str(args.official_root / "c2m3"), "status", "--porcelain")),
        "lift_certification_gate": False,
        "lift_exclusion_reason": "current claim ledger has no certified natural-MNIST lift candidate",
        "test_selection_policy": "A0-A5/B0 frozen before test evaluation; A6 only is test-selected",
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(output_dir, args, selectors, summary, paired, counts, failures, status, criteria)
    manifest = artifact_manifest(output_dir, [latex_path] if args.stage == "confirmatory" else [])
    write_csv(output_dir / "artifact_manifest.csv", manifest)
    print(json.dumps({"stage": args.stage, "settings": selectors["setting_id"].nunique(), "status": status}, sort_keys=True))


if __name__ == "__main__":
    main()
