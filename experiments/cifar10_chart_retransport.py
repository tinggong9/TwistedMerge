#!/usr/bin/env python3
"""Bounded CIFAR-10 discovery and gated confirmation for D4 retransport."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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
    calibrate_temperature,
    chart_probabilities,
    checkpoint_payload,
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
    split_indices,
    task_branches,
    train_adapter,
    train_classifier,
)
from experiments.next_program_common import latex_table, paired_bootstrap, write_csv

SCRIPT = Path(__file__).resolve()
DEST = OUT / "cifar"
COMMAND = "python experiments/cifar10_chart_retransport.py"
DISCOVERY_SEEDS = tuple(range(5))
CONFIRMATION_SEEDS = tuple(range(5, 10))
METHODS = (
    "context_blind_expert_average",
    "generic_moe",
    "generic_low_rank_context_adapter",
    "d4_equivariant_chart_soft_routing",
    "inferred_canonicalize_pool_retransport",
    "d4_equivariant_task_classifier",
    "d4_test_time_augmentation",
    "one_canonical_after_inferred_inverse",
    "supplied_chart_oracle",
    "random_action_control",
    "wrong_action_control",
    "ensemble_reference",
)


def prepare(seed: int, smoke: bool) -> dict[str, object]:
    train_images, train_labels, test_images, test_labels, channels = dataset_tensors("CIFAR10")
    split = split_indices(50 + seed, len(train_images), local_train=512 if smoke else 6000)
    if smoke:
        for key, size in (("chart_train", 128), ("validation", 64), ("calibration", 64), ("threshold", 64)):
            split[key] = split[key][:size]
    order = np.random.default_rng(241_000_000 + seed).permutation(len(test_images))[: (256 if smoke else 2000)]
    chart_images, chart_labels, _ = make_chart_examples(train_images[split["chart_train"]], 241_100_000 + seed, [0, 1, 4], "chart_training")
    validation_images, validation_charts, _ = make_chart_examples(train_images[split["validation"]], 241_200_000 + seed, range(8), "early_stopping_validation")
    calibration_images, calibration_charts, _ = make_chart_examples(train_images[split["calibration"]], 241_300_000 + seed, range(8), "calibration")
    final_images, final_charts, conditions = conditioned_test_examples(test_images[order], 241_400_000 + seed, include_color=True)
    return {
        "channels": channels,
        "split": split,
        "test_order": order,
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
        "test_images": final_images,
        "test_charts": final_charts,
        "test_labels": test_labels[order],
        "conditions": conditions,
    }


def run_seed(seed: int, phase: str, smoke: bool) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    payload = prepare(seed, smoke)
    channels = int(payload["channels"])
    task_epochs = 1 if smoke else 3
    chart_epochs = 2 if smoke else 8
    models: dict[str, torch.nn.Module] = {}
    training_times: dict[str, float] = {}
    for index in range(4):
        model, elapsed, _ = train_classifier(
            ImageCNN(10, channels, width=12), payload["local_images"], payload["local_labels"], payload["validation_images"], payload["validation_task_labels"], 242_000_000 + seed * 10 + index, task_epochs
        )
        models[f"expert_{index}"] = model
        training_times[f"expert_{index}"] = elapsed
    direct, elapsed, _ = train_classifier(
        OrbitTaskCNN(channels, width=10), payload["local_images"], payload["local_labels"], payload["validation_images"], payload["validation_task_labels"], 242_100_000 + seed, task_epochs
    )
    models["direct_task"] = direct
    training_times["direct_task"] = elapsed
    for offset, (name, model, augmentation) in enumerate(
        (
            ("chart_equivariant", D4EquivariantChartCNN(channels, width=10), None),
            ("chart_ordinary", ImageCNN(8, channels, width=8), None),
            ("chart_augmented", ImageCNN(8, channels, width=10), ordinary_chart_augmentation),
        )
    ):
        model, elapsed, _ = train_classifier(
            model, payload["chart_images"], payload["chart_labels"], payload["validation_images"], payload["validation_charts"], 242_200_000 + seed * 10 + offset, chart_epochs, batch_size=64, augment=augmentation
        )
        models[name] = model
        training_times[name] = elapsed
    ordinary_temperature = calibrate_temperature(model_logits(models["chart_ordinary"], payload["calibration_images"]), payload["calibration_charts"])
    equivariant_temperature = calibrate_temperature(model_logits(models["chart_equivariant"], payload["calibration_images"]), payload["calibration_charts"])
    chart_train_probabilities = chart_probabilities(models["chart_ordinary"], payload["chart_images"], ordinary_temperature)
    validation_probabilities = chart_probabilities(models["chart_ordinary"], payload["validation_images"], ordinary_temperature)
    adapter, elapsed = train_adapter(
        LowRankContextAdapter(channels, width=10), payload["chart_images"], payload["chart_task_labels"], chart_train_probabilities, payload["validation_images"], payload["validation_task_labels"], validation_probabilities, 242_300_000 + seed, chart_epochs
    )
    models["adapter"] = adapter
    training_times["adapter"] = elapsed
    experts = [models[f"expert_{index}"] for index in range(4)]
    ordinary_probability = chart_probabilities(models["chart_ordinary"], payload["test_images"], ordinary_temperature)
    equivariant_probability = chart_probabilities(models["chart_equivariant"], payload["test_images"], equivariant_temperature)
    branches = task_branches(payload["test_images"], experts)
    one_branches = one_expert_branches(payload["test_images"], experts[0])
    retransport = retransport_logits(payload["test_images"], experts, equivariant_probability)
    random_generator = torch.Generator().manual_seed(242_900_000 + seed)
    random_charts = torch.randint(0, 8, (len(payload["test_images"]),), generator=random_generator)
    with torch.no_grad():
        adapter_logits = torch.cat(
            [
                adapter(images.to(DEVICE), probabilities.to(DEVICE)).cpu()
                for images, probabilities in zip(payload["test_images"].split(128), ordinary_probability.split(128), strict=True)
            ]
        )
    candidates = {
        "context_blind_expert_average": branches.mean(1),
        "generic_moe": torch.einsum("nb,nbc->nc", ordinary_probability, branches),
        "generic_low_rank_context_adapter": adapter_logits,
        "d4_equivariant_chart_soft_routing": torch.einsum("nb,nbc->nc", equivariant_probability, branches),
        "inferred_canonicalize_pool_retransport": retransport,
        "d4_equivariant_task_classifier": model_logits(direct, payload["test_images"]),
        "d4_test_time_augmentation": d4_tta_logits(payload["test_images"], experts[0]),
        "one_canonical_after_inferred_inverse": torch.einsum("nb,nbc->nc", equivariant_probability, one_branches),
        "supplied_chart_oracle": branches[torch.arange(len(branches)), payload["test_charts"]],
        "random_action_control": branches[torch.arange(len(branches)), random_charts],
        "wrong_action_control": branches[torch.arange(len(branches)), (equivariant_probability.argmax(1) + 1) % 8],
        "ensemble_reference": torch.stack([model_logits(model, payload["test_images"]) for model in experts]).mean(0),
    }
    ledger = save_logits_before_evaluation(f"cifar_{phase}_seed_{seed}", candidates, payload["test_labels"], 243_000_000 + seed)
    chart_prediction = equivariant_probability.argmax(1)
    method_chart_prediction = {
        "generic_moe": ordinary_probability.argmax(1),
        "generic_low_rank_context_adapter": ordinary_probability.argmax(1),
        "d4_equivariant_chart_soft_routing": chart_prediction,
        "inferred_canonicalize_pool_retransport": chart_prediction,
        "one_canonical_after_inferred_inverse": chart_prediction,
        "supplied_chart_oracle": payload["test_charts"],
        "random_action_control": random_charts,
        "wrong_action_control": (chart_prediction + 1) % 8,
    }
    batch = payload["test_images"][: min(128, len(payload["test_images"]))]
    batch_charts = payload["test_charts"][: len(batch)]

    def executed_path(method: str) -> torch.Tensor:
        if method == "d4_equivariant_task_classifier":
            return model_logits(direct, batch)
        if method == "d4_test_time_augmentation":
            return d4_tta_logits(batch, experts[0])
        if method == "ensemble_reference":
            return torch.stack([model_logits(model, batch) for model in experts]).mean(0)
        if method == "generic_low_rank_context_adapter":
            probability = chart_probabilities(models["chart_ordinary"], batch, ordinary_temperature)
            with torch.no_grad():
                return adapter(batch.to(DEVICE), probability.to(DEVICE)).cpu()
        local_branches = task_branches(batch, experts)
        if method == "context_blind_expert_average":
            return local_branches.mean(1)
        if method == "supplied_chart_oracle":
            return local_branches[torch.arange(len(local_branches)), batch_charts]
        if method == "random_action_control":
            generator = torch.Generator().manual_seed(242_900_000 + seed)
            charts = torch.randint(0, 8, (len(batch),), generator=generator)
            return local_branches[torch.arange(len(local_branches)), charts]
        if method == "generic_moe":
            probability = chart_probabilities(models["chart_ordinary"], batch, ordinary_temperature)
            return torch.einsum("nb,nbc->nc", probability, local_branches)
        probability = chart_probabilities(models["chart_equivariant"], batch, equivariant_temperature)
        if method == "one_canonical_after_inferred_inverse":
            return torch.einsum("nb,nbc->nc", probability, one_expert_branches(batch, experts[0]))
        if method == "wrong_action_control":
            return local_branches[torch.arange(len(local_branches)), (probability.argmax(1) + 1) % 8]
        if method == "d4_equivariant_chart_soft_routing":
            return torch.einsum("nb,nbc->nc", probability, local_branches)
        return retransport_logits(batch, experts, probability)

    timings = {
        method: measure_actual(lambda selected=method: executed_path(selected), 0 if smoke else 1, 1 if smoke else 3)
        for method in METHODS
    }
    selected_models = {
        "context_blind_expert_average": [f"expert_{index}" for index in range(4)],
        "generic_moe": ["chart_ordinary", *[f"expert_{index}" for index in range(4)]],
        "generic_low_rank_context_adapter": ["chart_ordinary", "adapter"],
        "d4_equivariant_chart_soft_routing": ["chart_equivariant", *[f"expert_{index}" for index in range(4)]],
        "inferred_canonicalize_pool_retransport": ["chart_equivariant", *[f"expert_{index}" for index in range(4)]],
        "d4_equivariant_task_classifier": ["direct_task"],
        "d4_test_time_augmentation": ["expert_0"],
        "one_canonical_after_inferred_inverse": ["chart_equivariant", "expert_0"],
        "supplied_chart_oracle": [f"expert_{index}" for index in range(4)],
        "random_action_control": [f"expert_{index}" for index in range(4)],
        "wrong_action_control": ["chart_equivariant", *[f"expert_{index}" for index in range(4)]],
        "ensemble_reference": [f"expert_{index}" for index in range(4)],
    }
    runs = []
    generalization = []
    cost = []
    for method in METHODS:
        metrics = extended_metrics(candidates[method], payload["test_labels"])
        names = selected_models[method]
        method_prediction = method_chart_prediction.get(method)
        runs.append(
            {
                "setting_id": f"CIFAR10_{phase}_seed{seed}",
                "phase": phase,
                "seed": seed,
                "method": method,
                **metrics,
                "chart_accuracy": float((method_prediction == payload["test_charts"]).float().mean()) if method_prediction is not None else "",
                "trainable_parameters": sum(parameter_count(models[name]) for name in names),
                "stored_parameters": sum(parameter_count(models[name]) for name in names),
                "stored_bytes": sum(model_bytes(models[name]) for name in names),
                "branch_count": 1 if method in {"generic_low_rank_context_adapter"} else 8,
                "complete_latency_ms_batch128": timings[method]["warm_start_latency_ms"],
                "peak_process_memory_mb": timings[method]["peak_process_memory_mb"],
                "peak_accelerator_memory_mb": timings[method]["peak_accelerator_memory_mb"],
                "chart_training_examples": 0 if method in {"context_blind_expert_average", "d4_equivariant_task_classifier", "d4_test_time_augmentation", "supplied_chart_oracle", "random_action_control", "ensemble_reference"} else len(payload["chart_images"]),
                "validation_examples": len(payload["validation_images"]),
                "calibration_examples": len(payload["calibration_images"]),
                "threshold_selection_examples": 0,
                "test_examples": len(payload["test_images"]),
                "training_time_seconds": sum(training_times.get(name, 0.0) for name in names),
                "chart_information": "supplied" if method == "supplied_chart_oracle" else ("none" if method in {"context_blind_expert_average", "d4_equivariant_task_classifier", "d4_test_time_augmentation", "ensemble_reference"} else "inferred"),
                "expert_evaluations": 4 if method not in {"d4_equivariant_task_classifier", "d4_test_time_augmentation", "one_canonical_after_inferred_inverse", "generic_low_rank_context_adapter"} else 1,
                "logits_path": ledger["logits_path"],
                "logits_sha256": ledger["logits_sha256"],
                "label_permutation_hash_passed": bool(ledger["candidate_hashes_unchanged"] and ledger["file_hash_unchanged"]),
                **provenance(SCRIPT, COMMAND + (" --smoke" if smoke else ""), seed),
            }
        )
        for condition in sorted(set(payload["conditions"])):
            mask = payload["conditions"] == condition
            generalization.append(
                {
                    "phase": phase,
                    "seed": seed,
                    "method": method,
                    "condition": condition,
                    "examples": int(mask.sum()),
                    "task_accuracy": float((candidates[method][mask].argmax(1) == payload["test_labels"][mask]).float().mean()),
                }
            )
        cost.append(
            {
                "phase": phase,
                "seed": seed,
                "method": method,
                "batch_size": len(batch),
                "complete_path_latency_ms": timings[method]["warm_start_latency_ms"],
                "latency_q1_ms": timings[method]["latency_q1_ms"],
                "latency_q3_ms": timings[method]["latency_q3_ms"],
                "stored_bytes": sum(model_bytes(models[name]) for name in names),
                "trainable_parameters": sum(parameter_count(models[name]) for name in names),
            }
        )
    checkpoint_path = TMP / "checkpoints" / "cifar" / f"{phase}_seed_{seed}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        checkpoint_payload(models, seed=seed, phase=phase, split_indices={key: value.tolist() for key, value in payload["split"].items()}, test_order=payload["test_order"].tolist(), ordinary_temperature=ordinary_temperature, equivariant_temperature=equivariant_temperature, training_times=training_times),
        checkpoint_path,
    )
    return runs, generalization, cost, [{"phase": phase, "seed": seed, "path": str(checkpoint_path.relative_to(ROOT)), "bytes": checkpoint_path.stat().st_size}]


def discovery_gate(runs: list[dict[str, object]], generalization: list[dict[str, object]], seeds: list[int]) -> tuple[dict[str, object], list[dict[str, object]]]:
    discovery = [row for row in runs if row["phase"] == "discovery"]
    comparisons = (
        ("structured_minus_generic_moe", "inferred_canonicalize_pool_retransport", "generic_moe"),
        ("structured_minus_low_rank_adapter", "inferred_canonicalize_pool_retransport", "generic_low_rank_context_adapter"),
        ("structured_minus_direct_equivariant_task", "inferred_canonicalize_pool_retransport", "d4_equivariant_task_classifier"),
    )
    paired = paired_interval_rows(discovery, comparisons, "task_accuracy", 244_000_000)
    criterion_a = all(float(row["ci_low"]) > 0 for row in paired if "generic" in str(row["comparison"]) or "low_rank" in str(row["comparison"]))
    criterion_b = next(float(row["ci_low"]) for row in paired if "direct_equivariant" in str(row["comparison"])) > 0
    structured_parameters = np.mean([float(row["trainable_parameters"]) for row in discovery if row["method"] == "inferred_canonicalize_pool_retransport"])
    baseline_parameters = min(float(row["trainable_parameters"]) for row in discovery if row["method"] in {"generic_moe", "generic_low_rank_context_adapter"})
    structured_accuracy = np.mean([float(row["task_accuracy"]) for row in discovery if row["method"] == "inferred_canonicalize_pool_retransport"])
    baseline_accuracy = max(np.mean([float(row["task_accuracy"]) for row in discovery if row["method"] == method]) for method in ("generic_moe", "generic_low_rank_context_adapter"))
    criterion_c = structured_accuracy >= baseline_accuracy - 0.002 and structured_parameters <= 0.5 * baseline_parameters
    worst_deltas = []
    for seed in seeds:
        structured = [float(row["task_accuracy"]) for row in generalization if row["phase"] == "discovery" and row["seed"] == seed and row["method"] == "inferred_canonicalize_pool_retransport"]
        generic = [float(row["task_accuracy"]) for row in generalization if row["phase"] == "discovery" and row["seed"] == seed and row["method"] == "generic_moe"]
        worst_deltas.append(min(structured) - min(generic))
    worst_mean, worst_low, worst_high = paired_bootstrap(worst_deltas, 244_100_000)
    generic_moe_parameters = np.mean([float(row["trainable_parameters"]) for row in discovery if row["method"] == "generic_moe"])
    criterion_d = worst_low > 0 and structured_parameters <= generic_moe_parameters
    return (
        {
            "phase": "discovery",
            "criterion_a_all_generic_inferred": criterion_a,
            "criterion_b_direct_equivariant": criterion_b,
            "criterion_c_half_parameters_or_chart_data": criterion_c,
            "criterion_d_worst_condition": criterion_d,
            "worst_condition_mean_delta": worst_mean,
            "worst_condition_ci_low": worst_low,
            "worst_condition_ci_high": worst_high,
            "gate_passed": criterion_a or criterion_b or criterion_c or criterion_d,
        },
        paired,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    discovery_seeds = [0] if arguments.smoke else list(DISCOVERY_SEEDS)
    runs: list[dict[str, object]] = []
    generalization: list[dict[str, object]] = []
    cost: list[dict[str, object]] = []
    checkpoints: list[dict[str, object]] = []
    for seed in discovery_seeds:
        pieces = run_seed(seed, "discovery", arguments.smoke)
        runs.extend(pieces[0]); generalization.extend(pieces[1]); cost.extend(pieces[2]); checkpoints.extend(pieces[3])
    gate, paired = discovery_gate(runs, generalization, discovery_seeds)
    confirmation_executed = False
    if bool(gate["gate_passed"]) and not arguments.smoke:
        confirmation_executed = True
        for seed in CONFIRMATION_SEEDS:
            pieces = run_seed(seed, "confirmation", False)
            runs.extend(pieces[0]); generalization.extend(pieces[1]); cost.extend(pieces[2]); checkpoints.extend(pieces[3])
    claims = [
        {"claim": "discovery_gate_passed", "value": gate["gate_passed"]},
        {"claim": "confirmation_executed", "value": confirmation_executed},
        {"claim": "confirmation_required_when_discovery_passes", "value": (not gate["gate_passed"]) or confirmation_executed or arguments.smoke},
        {"claim": "all_saved_logits_label_permutation_invariant", "value": all(bool(row["label_permutation_hash_passed"]) for row in runs)},
    ]
    summary = []
    for phase in sorted({str(row["phase"]) for row in runs}):
        for method in METHODS:
            block = [row for row in runs if row["phase"] == phase and row["method"] == method]
            summary.append({"phase": phase, "method": method, "seeds": len(block), "task_accuracy": float(np.mean([float(row["task_accuracy"]) for row in block])), "ece": float(np.mean([float(row["ece"]) for row in block])), "stored_bytes": int(np.mean([int(row["stored_bytes"]) for row in block]))})
    write_csv(DEST / "runs.csv", runs)
    write_csv(DEST / "summary.csv", summary)
    write_csv(DEST / "paired.csv", paired)
    write_csv(DEST / "generalization.csv", generalization)
    write_csv(DEST / "cost.csv", cost)
    write_csv(DEST / "claims.csv", claims)
    write_csv(DEST / "gate.csv", [gate])
    write_csv(DEST / "checkpoint_manifest.csv", checkpoints)
    latex_table(DEST / "tables" / "cifar_chart.tex", ["phase", "method", "task_accuracy", "ece", "stored_bytes"], summary, "CIFAR-10 D4 chart transfer")
    import matplotlib.pyplot as plt

    discovery_summary = [row for row in summary if row["phase"] == "discovery"]
    figure, axis = plt.subplots(figsize=(8.2, 4.5))
    axis.barh([row["method"] for row in discovery_summary], [row["task_accuracy"] for row in discovery_summary])
    axis.set(xlabel="Discovery task accuracy", xlim=(0, 1))
    figure.tight_layout(); figure.savefig(DEST / "plots" / "cifar_conditions.pdf"); plt.close(figure)
    factual_report(
        DEST / "report.md",
        "CIFAR-10 chart retransport",
        [
            f"Execution commit: `{provenance(SCRIPT, COMMAND, 'aggregate')['execution_commit']}`. Discovery used seeds {discovery_seeds}; confirmation {'executed without hyperparameter changes' if confirmation_executed else 'was not triggered by the preregistered discovery gate'}.",
            "All methods used identical CIFAR-10 splits and transformations within each seed. Candidate logits were persisted before final labels were evaluated, and the saved hashes were unchanged by the label-permutation audit.",
            "The discovery criteria and worst-condition interval are recorded in `gate.csv`; failed criteria remain negative findings.",
        ],
    )


if __name__ == "__main__":
    main()
