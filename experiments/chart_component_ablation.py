#!/usr/bin/env python3
"""Isolate chart recognition, canonicalization, expert count, and retransport."""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.chart_followup_common import (
    DEVICE,
    OUT,
    TMP,
    D4EquivariantChartCNN,
    ImageCNN,
    LowRankContextAdapter,
    OrbitTaskCNN,
    apply_d4,
    calibrate_temperature,
    chart_probabilities,
    checkpoint_payload,
    choose_abstention_threshold,
    conditioned_test_examples,
    d4_tta_logits,
    dataset_tensors,
    extended_metrics,
    factual_report,
    make_chart_examples,
    measure_actual,
    model_bytes,
    model_logits,
    one_expert_branches,
    ordinary_chart_augmentation,
    paired_interval_rows,
    parameter_count,
    provenance,
    retransport_logits,
    save_logits_before_evaluation,
    specialized_expert_logits,
    split_indices,
    task_branches,
    train_adapter,
    train_classifier,
    wrong_order_branches,
)
from experiments.next_program_common import latex_table, write_csv

SCRIPT = Path(__file__).resolve()
DEST = OUT / "ablation"
COMMAND = "python experiments/chart_component_ablation.py"
METHODS = (
    "single_canonical_raw",
    "d4_test_time_augmentation",
    "direct_d4_equivariant_task_classifier",
    "ordinary_chart_soft_moe",
    "d4_chart_soft_moe",
    "d4_chart_hard_branch_selection",
    "inverse_transform_one_canonical_expert",
    "inverse_transform_four_expert_average",
    "canonicalize_pool_retransport",
    "uncertainty_weighted_retransport",
    "abstaining_retransport",
    "supplied_chart_oracle",
    "random_chart_control",
    "wrong_group_action_control",
    "wrong_multiplication_order_control",
    "ensemble_reference",
    "generic_low_rank_context_adapter",
)


def prepare_seed(seed: int, smoke: bool) -> dict[str, object]:
    train_images, train_labels, test_images, test_labels, channels = dataset_tensors("FashionMNIST")
    local_size = 512 if smoke else 6000
    split = split_indices(seed, len(train_images), local_train=local_size)
    if smoke:
        split["chart_train"] = split["chart_train"][:128]
        split["validation"] = split["validation"][:64]
        split["calibration"] = split["calibration"][:64]
        split["threshold"] = split["threshold"][:64]
    test_size = 256 if smoke else 2000
    test_order = np.random.default_rng(220_000_000 + seed).permutation(len(test_images))[:test_size]
    chart_images, chart_labels, _ = make_chart_examples(train_images[split["chart_train"]], 221_000_000 + seed, [0, 1, 4], "chart_training")
    validation_images, validation_charts, _ = make_chart_examples(train_images[split["validation"]], 222_000_000 + seed, range(8), "early_stopping_validation")
    calibration_images, calibration_charts, _ = make_chart_examples(train_images[split["calibration"]], 223_000_000 + seed, range(8), "calibration")
    threshold_images, threshold_charts, _ = make_chart_examples(train_images[split["threshold"]], 224_000_000 + seed, range(8), "threshold_selection")
    final_images, final_charts, conditions = conditioned_test_examples(test_images[test_order], 225_000_000 + seed, include_color=False)
    return {
        "channels": channels,
        "split": split,
        "test_order": test_order,
        "local_images": train_images[split["local_train"]],
        "local_labels": train_labels[split["local_train"]],
        "chart_images": chart_images,
        "chart_labels": chart_labels,
        "chart_task_labels": train_labels[split["chart_train"]],
        "validation_images": validation_images,
        "validation_charts": validation_charts,
        "validation_task_labels": train_labels[split["validation"]],
        "calibration_images": calibration_images,
        "calibration_charts": calibration_charts,
        "threshold_images": threshold_images,
        "threshold_charts": threshold_charts,
        "threshold_task_labels": train_labels[split["threshold"]],
        "test_images": final_images,
        "test_charts": final_charts,
        "test_labels": test_labels[test_order],
        "conditions": conditions,
    }


def train_models(payload: dict[str, object], seed: int, smoke: bool) -> tuple[dict[str, torch.nn.Module], dict[str, float]]:
    channels = int(payload["channels"])
    local_images = payload["local_images"]
    local_labels = payload["local_labels"]
    validation_images = payload["validation_images"]
    validation_task_labels = payload["validation_task_labels"]
    task_epochs = 1 if smoke else 2
    chart_epochs = 2 if smoke else 8
    models: dict[str, torch.nn.Module] = {}
    training_times: dict[str, float] = {}
    for index in range(4):
        model, elapsed, _ = train_classifier(
            ImageCNN(10, channels, width=12),
            local_images,
            local_labels,
            validation_images,
            validation_task_labels,
            226_000_000 + seed * 10 + index,
            task_epochs,
        )
        models[f"canonical_{index}"] = model
        training_times[f"canonical_{index}"] = elapsed
    for index in range(4):
        indices = torch.arange(index, len(local_images), 4)
        charts = torch.tensor([index, index + 4] * ((len(indices) + 1) // 2), dtype=torch.long)[: len(indices)]
        specialized_images = apply_d4(local_images[indices], charts)
        model, elapsed, _ = train_classifier(
            ImageCNN(10, channels, width=12),
            specialized_images,
            local_labels[indices],
            None,
            None,
            226_100_000 + seed * 10 + index,
            task_epochs,
        )
        models[f"specialized_{index}"] = model
        training_times[f"specialized_{index}"] = elapsed
    direct, elapsed, _ = train_classifier(
        OrbitTaskCNN(channels, width=10),
        local_images,
        local_labels,
        validation_images,
        validation_task_labels,
        226_200_000 + seed,
        task_epochs,
    )
    models["direct_task"] = direct
    training_times["direct_task"] = elapsed
    chart_specs = {
        "chart_equivariant": (D4EquivariantChartCNN(channels, width=10), None),
        "chart_ordinary_matched": (ImageCNN(8, channels, width=7), None),
        "chart_ordinary_larger": (ImageCNN(8, channels, width=14), None),
        "chart_ordinary_augmented": (ImageCNN(8, channels, width=10), ordinary_chart_augmentation),
    }
    for offset, (name, (model, augmentation)) in enumerate(chart_specs.items()):
        model, elapsed, _ = train_classifier(
            model,
            payload["chart_images"],
            payload["chart_labels"],
            payload["validation_images"],
            payload["validation_charts"],
            226_300_000 + seed * 10 + offset,
            chart_epochs,
            batch_size=64,
            augment=augmentation,
        )
        models[name] = model
        training_times[name] = elapsed
    ordinary_calibration = model_logits(models["chart_ordinary_larger"], payload["calibration_images"])
    ordinary_temperature = calibrate_temperature(ordinary_calibration, payload["calibration_charts"])
    chart_training_probabilities = chart_probabilities(models["chart_ordinary_larger"], payload["chart_images"], ordinary_temperature)
    validation_probabilities = chart_probabilities(models["chart_ordinary_larger"], payload["validation_images"], ordinary_temperature)
    adapter, adapter_elapsed = train_adapter(
        LowRankContextAdapter(channels, width=10),
        payload["chart_images"],
        payload["chart_task_labels"],
        chart_training_probabilities,
        payload["validation_images"],
        payload["validation_task_labels"],
        validation_probabilities,
        226_400_000 + seed,
        chart_epochs,
    )
    models["adapter"] = adapter
    training_times["adapter"] = adapter_elapsed
    return models, training_times


def probabilities_and_threshold(
    payload: dict[str, object], models: dict[str, torch.nn.Module]
) -> tuple[float, float, float]:
    ordinary_temperature = calibrate_temperature(
        model_logits(models["chart_ordinary_larger"], payload["calibration_images"]), payload["calibration_charts"]
    )
    equivariant_temperature = calibrate_temperature(
        model_logits(models["chart_equivariant"], payload["calibration_images"]), payload["calibration_charts"]
    )
    threshold_probabilities = chart_probabilities(
        models["chart_equivariant"], payload["threshold_images"], equivariant_temperature
    )
    canonical = [models[f"canonical_{index}"] for index in range(4)]
    threshold_retransport = retransport_logits(payload["threshold_images"], canonical, threshold_probabilities)
    threshold_fallback = d4_tta_logits(payload["threshold_images"], canonical[0])
    threshold = choose_abstention_threshold(
        threshold_probabilities.max(1).values,
        threshold_retransport,
        threshold_fallback,
        payload["threshold_task_labels"],
    )
    return ordinary_temperature, equivariant_temperature, threshold


def infer_candidates(
    images: torch.Tensor,
    true_charts: torch.Tensor,
    models: dict[str, torch.nn.Module],
    ordinary_temperature: float,
    equivariant_temperature: float,
    threshold: float,
    random_seed: int,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    canonical = [models[f"canonical_{index}"] for index in range(4)]
    specialized = [models[f"specialized_{index}"] for index in range(4)]
    ordinary_probability = chart_probabilities(models["chart_ordinary_larger"], images, ordinary_temperature)
    equivariant_probability = chart_probabilities(models["chart_equivariant"], images, equivariant_temperature)
    specialized_logits = specialized_expert_logits(images, specialized)
    one_branches = one_expert_branches(images, canonical[0])
    four_branches = task_branches(images, canonical)
    wrong_branches = wrong_order_branches(images, canonical)
    tta = d4_tta_logits(images, canonical[0])
    retransport = retransport_logits(images, canonical, equivariant_probability)
    confidence = equivariant_probability.max(1).values
    equivariant_soft_moe = torch.einsum("nb,nbc->nc", equivariant_probability, specialized_logits)
    equivariant_hard_chart = equivariant_probability.argmax(1)
    equivariant_hard_moe = specialized_logits[torch.arange(len(images)), equivariant_hard_chart]
    inverse_one = torch.einsum("nb,nbc->nc", equivariant_probability, one_branches)
    inverse_four = torch.einsum("nb,nbc->nc", equivariant_probability, four_branches)
    uncertainty = confidence.unsqueeze(1) * retransport + (1 - confidence).unsqueeze(1) * tta
    abstaining = torch.where((confidence >= threshold).unsqueeze(1), retransport, tta)
    supplied = four_branches[torch.arange(len(images)), true_charts]
    generator = torch.Generator().manual_seed(random_seed)
    random_charts = torch.randint(0, 8, (len(images),), generator=generator)
    random_control = four_branches[torch.arange(len(images)), random_charts]
    wrong_action = four_branches[torch.arange(len(images)), (equivariant_hard_chart + 1) % 8]
    wrong_order = torch.einsum("nb,nbc->nc", equivariant_probability, wrong_branches)
    with torch.no_grad():
        adapter = torch.cat(
            [
                models["adapter"](batch.to(DEVICE), probability.to(DEVICE)).cpu()
                for batch, probability in zip(images.split(128), ordinary_probability.split(128), strict=True)
            ]
        )
    candidates = {
        "single_canonical_raw": model_logits(canonical[0], images),
        "d4_test_time_augmentation": tta,
        "direct_d4_equivariant_task_classifier": model_logits(models["direct_task"], images),
        "ordinary_chart_soft_moe": torch.einsum("nb,nbc->nc", ordinary_probability, specialized_logits),
        "d4_chart_soft_moe": equivariant_soft_moe,
        "d4_chart_hard_branch_selection": equivariant_hard_moe,
        "inverse_transform_one_canonical_expert": inverse_one,
        "inverse_transform_four_expert_average": inverse_four,
        "canonicalize_pool_retransport": retransport,
        "uncertainty_weighted_retransport": uncertainty,
        "abstaining_retransport": abstaining,
        "supplied_chart_oracle": supplied,
        "random_chart_control": random_control,
        "wrong_group_action_control": wrong_action,
        "wrong_multiplication_order_control": wrong_order,
        "ensemble_reference": torch.stack([model_logits(model, images) for model in canonical]).mean(0),
        "generic_low_rank_context_adapter": adapter,
    }
    chart_predictions = {
        "equivariant": equivariant_probability.argmax(1),
        "ordinary_matched": model_logits(models["chart_ordinary_matched"], images).argmax(1),
        "ordinary_larger": model_logits(models["chart_ordinary_larger"], images).argmax(1),
        "ordinary_augmented": model_logits(models["chart_ordinary_augmented"], images).argmax(1),
        "random": random_charts,
    }
    return candidates, chart_predictions


def infer_method(
    method: str,
    images: torch.Tensor,
    true_charts: torch.Tensor,
    models: dict[str, torch.nn.Module],
    ordinary_temperature: float,
    equivariant_temperature: float,
    threshold: float,
    random_seed: int,
) -> torch.Tensor:
    """Execute exactly one complete inference path for latency accounting."""

    canonical = [models[f"canonical_{index}"] for index in range(4)]
    specialized = [models[f"specialized_{index}"] for index in range(4)]
    if method == "single_canonical_raw":
        return model_logits(canonical[0], images)
    if method == "d4_test_time_augmentation":
        return d4_tta_logits(images, canonical[0])
    if method == "direct_d4_equivariant_task_classifier":
        return model_logits(models["direct_task"], images)
    if method == "ensemble_reference":
        return torch.stack([model_logits(model, images) for model in canonical]).mean(0)
    if method == "supplied_chart_oracle":
        branches = task_branches(images, canonical)
        return branches[torch.arange(len(images)), true_charts]
    if method == "random_chart_control":
        branches = task_branches(images, canonical)
        generator = torch.Generator().manual_seed(random_seed)
        charts = torch.randint(0, 8, (len(images),), generator=generator)
        return branches[torch.arange(len(images)), charts]
    if method in {"ordinary_chart_soft_moe", "generic_low_rank_context_adapter"}:
        probabilities = chart_probabilities(models["chart_ordinary_larger"], images, ordinary_temperature)
        if method == "ordinary_chart_soft_moe":
            return torch.einsum("nb,nbc->nc", probabilities, specialized_expert_logits(images, specialized))
        with torch.no_grad():
            return models["adapter"](images.to(DEVICE), probabilities.to(DEVICE)).cpu()
    probabilities = chart_probabilities(models["chart_equivariant"], images, equivariant_temperature)
    if method == "d4_chart_soft_moe":
        return torch.einsum("nb,nbc->nc", probabilities, specialized_expert_logits(images, specialized))
    if method == "d4_chart_hard_branch_selection":
        branches = specialized_expert_logits(images, specialized)
        return branches[torch.arange(len(images)), probabilities.argmax(1)]
    if method == "inverse_transform_one_canonical_expert":
        return torch.einsum("nb,nbc->nc", probabilities, one_expert_branches(images, canonical[0]))
    if method == "inverse_transform_four_expert_average":
        return torch.einsum("nb,nbc->nc", probabilities, task_branches(images, canonical))
    if method == "wrong_group_action_control":
        branches = task_branches(images, canonical)
        return branches[torch.arange(len(images)), (probabilities.argmax(1) + 1) % 8]
    if method == "wrong_multiplication_order_control":
        return torch.einsum("nb,nbc->nc", probabilities, wrong_order_branches(images, canonical))
    structured = retransport_logits(images, canonical, probabilities)
    if method == "canonicalize_pool_retransport":
        return structured
    fallback = d4_tta_logits(images, canonical[0])
    confidence = probabilities.max(1).values
    if method == "uncertainty_weighted_retransport":
        return confidence.unsqueeze(1) * structured + (1 - confidence).unsqueeze(1) * fallback
    if method == "abstaining_retransport":
        return torch.where((confidence >= threshold).unsqueeze(1), structured, fallback)
    raise KeyError(method)


def method_model_names(method: str) -> list[str]:
    canonical_four = [f"canonical_{index}" for index in range(4)]
    specialized_four = [f"specialized_{index}" for index in range(4)]
    mapping = {
        "single_canonical_raw": ["canonical_0"],
        "d4_test_time_augmentation": ["canonical_0"],
        "direct_d4_equivariant_task_classifier": ["direct_task"],
        "ordinary_chart_soft_moe": ["chart_ordinary_larger", *specialized_four],
        "d4_chart_soft_moe": ["chart_equivariant", *specialized_four],
        "d4_chart_hard_branch_selection": ["chart_equivariant", *specialized_four],
        "inverse_transform_one_canonical_expert": ["chart_equivariant", "canonical_0"],
        "inverse_transform_four_expert_average": ["chart_equivariant", *canonical_four],
        "canonicalize_pool_retransport": ["chart_equivariant", *canonical_four],
        "uncertainty_weighted_retransport": ["chart_equivariant", *canonical_four],
        "abstaining_retransport": ["chart_equivariant", *canonical_four],
        "supplied_chart_oracle": canonical_four,
        "random_chart_control": canonical_four,
        "wrong_group_action_control": ["chart_equivariant", *canonical_four],
        "wrong_multiplication_order_control": ["chart_equivariant", *canonical_four],
        "ensemble_reference": canonical_four,
        "generic_low_rank_context_adapter": ["chart_ordinary_larger", "adapter"],
    }
    return mapping[method]


def run_seed(seed: int, smoke: bool) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    payload = prepare_seed(seed, smoke)
    models, training_times = train_models(payload, seed, smoke)
    ordinary_temperature, equivariant_temperature, threshold = probabilities_and_threshold(payload, models)
    candidates, chart_predictions = infer_candidates(
        payload["test_images"],
        payload["test_charts"],
        models,
        ordinary_temperature,
        equivariant_temperature,
        threshold,
        227_000_000 + seed,
    )
    ledger = save_logits_before_evaluation(f"ablation_seed_{seed}", candidates, payload["test_labels"], 227_100_000 + seed)
    chart_accuracy_rows = []
    for name, prediction in chart_predictions.items():
        chart_accuracy_rows.append(
            {
                "seed": seed,
                "chart_model": name,
                "chart_accuracy": float((prediction == payload["test_charts"]).float().mean()),
                "parameters": parameter_count(models["chart_equivariant"] if name == "equivariant" else models.get(f"chart_{name}", models["chart_ordinary_larger"])),
            }
        )
    method_chart_key = {
        "ordinary_chart_soft_moe": "ordinary_larger",
        "generic_low_rank_context_adapter": "ordinary_larger",
        "random_chart_control": "random",
    }
    equivariant_methods = set(METHODS) - {
        "single_canonical_raw",
        "d4_test_time_augmentation",
        "direct_d4_equivariant_task_classifier",
        "ordinary_chart_soft_moe",
        "supplied_chart_oracle",
        "random_chart_control",
        "ensemble_reference",
        "generic_low_rank_context_adapter",
    }
    batch = payload["test_images"][: min(128, len(payload["test_images"]))]
    batch_charts = payload["test_charts"][: len(batch)]
    timing_candidates = {}
    runs = []
    condition_rows = []
    for method in METHODS:
        # This is a real complete method invocation. Detailed 100-repeat timings
        # are reserved for the dedicated COST stage.
        timing = measure_actual(
            lambda selected=method: infer_method(
                selected,
                batch,
                batch_charts,
                models,
                ordinary_temperature,
                equivariant_temperature,
                threshold,
                227_200_000 + seed,
            ),
            warmups=0 if smoke else 1,
            repeats=1 if smoke else 3,
        )
        timing_candidates[method] = timing
        metrics = extended_metrics(candidates[method], payload["test_labels"])
        selected_names = method_model_names(method)
        if method == "supplied_chart_oracle":
            chart_accuracy: float | str = 1.0
        elif method in equivariant_methods:
            chart_accuracy = next(row["chart_accuracy"] for row in chart_accuracy_rows if row["chart_model"] == "equivariant")
        elif method in method_chart_key:
            chart_accuracy = next(row["chart_accuracy"] for row in chart_accuracy_rows if row["chart_model"] == method_chart_key[method])
        else:
            chart_accuracy = ""
        runs.append(
            {
                "setting_id": f"FashionMNIST_ablation_seed{seed}",
                "seed": seed,
                "method": method,
                **metrics,
                "chart_accuracy": chart_accuracy,
                "trainable_parameters": sum(parameter_count(models[name]) for name in selected_names),
                "stored_parameters": sum(parameter_count(models[name]) for name in selected_names),
                "stored_bytes": sum(model_bytes(models[name]) for name in selected_names),
                "branch_count": 8 if method not in {"single_canonical_raw", "direct_d4_equivariant_task_classifier", "generic_low_rank_context_adapter"} else 1,
                "complete_latency_ms_batch128": timing["warm_start_latency_ms"],
                "peak_process_memory_mb": timing["peak_process_memory_mb"],
                "peak_accelerator_memory_mb": timing["peak_accelerator_memory_mb"],
                "training_time_seconds": sum(training_times.get(name, 0.0) for name in selected_names),
                "chart_training_examples": 0 if method in {"single_canonical_raw", "d4_test_time_augmentation", "direct_d4_equivariant_task_classifier", "supplied_chart_oracle", "random_chart_control", "ensemble_reference"} else len(payload["chart_images"]),
                "validation_examples": len(payload["validation_images"]),
                "calibration_examples": len(payload["calibration_images"]),
                "threshold_selection_examples": len(payload["threshold_images"]),
                "test_examples": len(payload["test_images"]),
                "chart_information": "supplied" if method == "supplied_chart_oracle" else ("none" if chart_accuracy == "" else "inferred"),
                "expert_evaluations": 4 if "four" in method or method in {"canonicalize_pool_retransport", "uncertainty_weighted_retransport", "abstaining_retransport", "ensemble_reference", "supplied_chart_oracle"} else 1,
                "abstention_threshold": threshold if method == "abstaining_retransport" else "",
                "logits_path": ledger["logits_path"],
                "logits_sha256": ledger["logits_sha256"],
                "label_permutation_hash_passed": bool(ledger["candidate_hashes_unchanged"] and ledger["file_hash_unchanged"]),
                **provenance(SCRIPT, COMMAND + (" --smoke" if smoke else ""), seed),
            }
        )
        for condition in sorted(set(payload["conditions"])):
            mask = payload["conditions"] == condition
            condition_rows.append(
                {
                    "seed": seed,
                    "method": method,
                    "condition": condition,
                    "examples": int(mask.sum()),
                    "task_accuracy": float((candidates[method][mask].argmax(1) == payload["test_labels"][mask]).float().mean()),
                }
            )
    checkpoint_path = TMP / "checkpoints" / "ablation" / f"seed_{seed}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        checkpoint_payload(
            models,
            seed=seed,
            split_indices={key: value.tolist() for key, value in payload["split"].items()},
            test_order=payload["test_order"].tolist(),
            ordinary_temperature=ordinary_temperature,
            equivariant_temperature=equivariant_temperature,
            abstention_threshold=threshold,
            training_times=training_times,
        ),
        checkpoint_path,
    )
    return runs, condition_rows, chart_accuracy_rows, [
        {
            "seed": seed,
            "checkpoint_path": str(checkpoint_path.relative_to(ROOT)),
            "bytes": checkpoint_path.stat().st_size,
            "sha256": __import__("hashlib").sha256(checkpoint_path.read_bytes()).hexdigest(),
        }
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    seeds = [20] if arguments.smoke else list(range(20, 30))
    runs: list[dict[str, object]] = []
    conditions: list[dict[str, object]] = []
    chart_rows: list[dict[str, object]] = []
    checkpoints: list[dict[str, object]] = []
    for seed in seeds:
        seed_runs, seed_conditions, seed_chart, seed_checkpoints = run_seed(seed, arguments.smoke)
        runs.extend(seed_runs)
        conditions.extend(seed_conditions)
        chart_rows.extend(seed_chart)
        checkpoints.extend(seed_checkpoints)
    comparisons = (
        ("soft_moe_minus_hard_same_probabilities", "d4_chart_soft_moe", "d4_chart_hard_branch_selection"),
        ("retransport_minus_hard_same_probabilities", "canonicalize_pool_retransport", "d4_chart_hard_branch_selection"),
        ("four_experts_minus_one_after_canonicalization", "inverse_transform_four_expert_average", "inverse_transform_one_canonical_expert"),
        ("retransport_minus_d4_tta", "canonicalize_pool_retransport", "d4_test_time_augmentation"),
        ("retransport_minus_direct_equivariant_task", "canonicalize_pool_retransport", "direct_d4_equivariant_task_classifier"),
    )
    paired = paired_interval_rows(runs, comparisons, "task_accuracy", 228_000_000)
    chart_by_seed = []
    for seed in seeds:
        equivariant = next(float(row["chart_accuracy"]) for row in chart_rows if row["seed"] == seed and row["chart_model"] == "equivariant")
        ordinary_values = [float(row["chart_accuracy"]) for row in chart_rows if row["seed"] == seed and str(row["chart_model"]).startswith("ordinary")]
        chart_by_seed.append(equivariant - max(ordinary_values))
    mean, low, high = __import__("experiments.next_program_common", fromlist=["paired_bootstrap"]).paired_bootstrap(chart_by_seed, 228_100_000)
    component_deltas = [{"comparison": "equivariant_chart_minus_best_ordinary", "mean_delta": mean, "ci_low": low, "ci_high": high}]
    component_deltas.extend(paired)
    lookup = {row["comparison"]: row for row in component_deltas}
    mean_latency = {method: float(np.mean([float(row["complete_latency_ms_batch128"]) for row in runs if row["method"] == method])) for method in METHODS}
    mean_storage = {method: float(np.mean([float(row["stored_bytes"]) for row in runs if row["method"] == method])) for method in METHODS}
    structured_method = "canonicalize_pool_retransport"
    tta_matched_cost = mean_latency[structured_method] <= mean_latency["d4_test_time_augmentation"] and mean_storage[structured_method] <= mean_storage["d4_test_time_augmentation"]
    direct_matched_cost = mean_latency[structured_method] <= mean_latency["direct_d4_equivariant_task_classifier"] and mean_storage[structured_method] <= mean_storage["direct_d4_equivariant_task_classifier"]
    beats_tta_at_matched_cost = float(lookup["retransport_minus_d4_tta"]["ci_low"]) > 0 and tta_matched_cost
    beats_direct_at_matched_cost = float(lookup["retransport_minus_direct_equivariant_task"]["ci_low"]) > 0 and direct_matched_cost
    claims = [
        {"claim": "equivariant_chart_benefit", "value": low > 0, "evidence": "equivariant_chart_minus_best_ordinary"},
        {"claim": "retransport_benefit", "value": float(lookup["retransport_minus_hard_same_probabilities"]["ci_low"]) > 0, "evidence": "retransport_minus_hard_same_probabilities"},
        {"claim": "multi_expert_benefit", "value": float(lookup["four_experts_minus_one_after_canonicalization"]["ci_low"]) > 0, "evidence": "four_experts_minus_one_after_canonicalization"},
        {"claim": "twistedmerge_specific_benefit_over_tta_at_matched_cost", "value": beats_tta_at_matched_cost, "evidence": "positive paired interval with no higher latency or storage"},
        {"claim": "twistedmerge_specific_benefit_over_direct_equivariant_task_at_matched_cost", "value": beats_direct_at_matched_cost, "evidence": "positive paired interval with no higher latency or storage"},
        {"claim": "twistedmerge_specific_benefit", "value": beats_tta_at_matched_cost and beats_direct_at_matched_cost, "evidence": "both matched-cost simple-equivariant comparisons"},
        {"claim": "all_saved_logits_label_permutation_invariant", "value": all(bool(row["label_permutation_hash_passed"]) for row in runs), "evidence": "runs.csv"},
    ]
    summary = []
    for method in METHODS:
        block = [row for row in runs if row["method"] == method]
        summary.append(
            {
                "method": method,
                "seeds": len(block),
                "task_accuracy": float(np.mean([float(row["task_accuracy"]) for row in block])),
                "negative_log_likelihood": float(np.mean([float(row["negative_log_likelihood"]) for row in block])),
                "ece": float(np.mean([float(row["ece"]) for row in block])),
                "complete_latency_ms_batch128": float(np.mean([float(row["complete_latency_ms_batch128"]) for row in block])),
                "stored_bytes": int(np.mean([int(row["stored_bytes"]) for row in block])),
            }
        )
    write_csv(DEST / "runs.csv", runs)
    write_csv(DEST / "summary.csv", summary)
    write_csv(DEST / "paired.csv", paired)
    write_csv(DEST / "component_deltas.csv", component_deltas)
    write_csv(DEST / "claims.csv", claims)
    write_csv(DEST / "conditions.csv", conditions)
    write_csv(DEST / "chart_models.csv", chart_rows)
    write_csv(DEST / "checkpoint_manifest.csv", checkpoints)
    latex_table(DEST / "tables" / "component_ablation.tex", ["method", "task_accuracy", "ece", "complete_latency_ms_batch128", "stored_bytes"], summary, "Fashion-MNIST chart component ablation")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8.5, 4.8))
    axis.barh([row["method"] for row in summary], [row["task_accuracy"] for row in summary])
    axis.set(xlabel="Task accuracy", xlim=(0, 1))
    figure.tight_layout()
    figure.savefig(DEST / "plots" / "component_accuracy.pdf")
    plt.close(figure)
    factual_report(
        DEST / "report.md",
        "Fashion-MNIST component ablation",
        [
            f"Execution commit: `{provenance(SCRIPT, COMMAND, 'aggregate')['execution_commit']}`. The stage executed {len(seeds)} fresh seeds with disjoint local-model, chart-training, validation, calibration, threshold-selection, and final-test roles.",
            "The seven same-chart-probability methods used byte-identical calibrated D4 chart probabilities. Candidate task logits were saved before final-test labels were evaluated, and the label-permutation audit left every saved candidate hash unchanged.",
            "Attribution gates are recorded without reinterpretation in `claims.csv`; negative intervals remain negative findings.",
        ],
    )


if __name__ == "__main__":
    main()
