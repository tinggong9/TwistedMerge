#!/usr/bin/env python3
"""Optional Fashion-MNIST chart-label sample-efficiency experiment."""

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
    dataset_tensors,
    extended_metrics,
    factual_report,
    make_chart_examples,
    model_bytes,
    model_logits,
    one_expert_branches,
    ordinary_chart_augmentation,
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
DEST = OUT / "sample_efficiency"
COMMAND = "python experiments/chart_sample_efficiency.py"
BUDGETS = (32, 64, 128, 256, 512, 1000)
METHODS = (
    "d4_equivariant_chart_cnn",
    "capacity_matched_ordinary_chart_cnn",
    "augmentation_trained_ordinary_chart_cnn",
    "generic_low_rank_adapter",
    "structured_retransport",
    "direct_d4_equivariant_task_cnn",
)


def first_crossing(points: list[tuple[int, float]], threshold: float) -> float | str:
    ordered = sorted(points)
    if ordered[0][1] >= threshold:
        return float(ordered[0][0])
    for (left_n, left_value), (right_n, right_value) in zip(ordered, ordered[1:]):
        if left_value < threshold <= right_value:
            fraction = (threshold - left_value) / max(right_value - left_value, 1e-12)
            return float(left_n + fraction * (right_n - left_n))
    return "not_reached"


def adapter_logits(model: LowRankContextAdapter, images: torch.Tensor, probabilities: torch.Tensor) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        return torch.cat(
            [
                model(batch.to(DEVICE), probability.to(DEVICE)).cpu()
                for batch, probability in zip(images.split(128), probabilities.split(128), strict=True)
            ]
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    seeds = [40] if arguments.smoke else list(range(40, 45))
    budgets = (32, 64) if arguments.smoke else BUDGETS
    train_images, train_labels, test_images, test_labels, channels = dataset_tensors("FashionMNIST")
    runs: list[dict[str, object]] = []
    checkpoint_rows: list[dict[str, object]] = []
    target_by_seed: dict[int, float] = {}
    for seed in seeds:
        split = split_indices(seed, len(train_images), local_train=512 if arguments.smoke else 6000)
        if arguments.smoke:
            for key, size in (("chart_train", 128), ("validation", 64), ("calibration", 64)):
                split[key] = split[key][:size]
        validation_images, validation_charts, _ = make_chart_examples(train_images[split["validation"]], 270_000_000 + seed, range(8), "validation")
        calibration_images, calibration_charts, _ = make_chart_examples(train_images[split["calibration"]], 271_000_000 + seed, range(8), "calibration")
        chart_pool_images, chart_pool_labels, _ = make_chart_examples(train_images[split["chart_train"]], 272_000_000 + seed, range(8), "chart_training")
        order = np.random.default_rng(273_000_000 + seed).permutation(len(test_images))[: 256 if arguments.smoke else 2000]
        test_chart_images, test_charts, conditions = conditioned_test_examples(test_images[order], 274_000_000 + seed, include_color=False)
        experts: list[ImageCNN] = []
        expert_times: list[float] = []
        for index in range(4):
            expert, elapsed, _ = train_classifier(
                ImageCNN(10, channels, width=12),
                train_images[split["local_train"]],
                train_labels[split["local_train"]],
                train_images[split["validation"]],
                train_labels[split["validation"]],
                275_000_000 + seed * 10 + index,
                1 if arguments.smoke else 2,
            )
            experts.append(expert)
            expert_times.append(elapsed)
        direct, direct_time, _ = train_classifier(
            OrbitTaskCNN(channels, width=10),
            train_images[split["local_train"]],
            train_labels[split["local_train"]],
            train_images[split["validation"]],
            train_labels[split["validation"]],
            275_100_000 + seed,
            1 if arguments.smoke else 2,
        )
        fixed_branches = task_branches(test_chart_images, experts)
        one_branches = one_expert_branches(test_chart_images, experts[0])
        oracle = fixed_branches[torch.arange(len(fixed_branches)), test_charts]
        blind = fixed_branches.mean(1)
        oracle_accuracy = float((oracle.argmax(1) == test_labels[order]).float().mean())
        blind_accuracy = float((blind.argmax(1) == test_labels[order]).float().mean())
        target_by_seed[seed] = blind_accuracy + 0.9 * (oracle_accuracy - blind_accuracy)
        for budget in budgets:
            chart_images = chart_pool_images[:budget]
            chart_labels = chart_pool_labels[:budget]
            chart_task_labels = train_labels[split["chart_train"][:budget]]
            model_specs = {
                "equivariant": (D4EquivariantChartCNN(channels, width=10), None),
                "ordinary": (ImageCNN(8, channels, width=10), None),
                "augmented": (ImageCNN(8, channels, width=10), ordinary_chart_augmentation),
            }
            chart_models: dict[str, torch.nn.Module] = {}
            chart_times: dict[str, float] = {}
            temperatures: dict[str, float] = {}
            for offset, (name, (model, augmentation)) in enumerate(model_specs.items()):
                trained, elapsed, _ = train_classifier(
                    model,
                    chart_images,
                    chart_labels,
                    validation_images,
                    validation_charts,
                    276_000_000 + seed * 100 + budget + offset,
                    2 if arguments.smoke else 8,
                    batch_size=min(64, budget),
                    augment=augmentation,
                )
                chart_models[name] = trained
                chart_times[name] = elapsed
                temperatures[name] = calibrate_temperature(model_logits(trained, calibration_images), calibration_charts)
            training_probability = chart_probabilities(chart_models["ordinary"], chart_images, temperatures["ordinary"])
            validation_probability = chart_probabilities(chart_models["ordinary"], validation_images, temperatures["ordinary"])
            adapter, adapter_time = train_adapter(
                LowRankContextAdapter(channels, width=10),
                chart_images,
                chart_task_labels,
                training_probability,
                validation_images,
                train_labels[split["validation"]],
                validation_probability,
                277_000_000 + seed * 100 + budget,
                2 if arguments.smoke else 8,
            )
            probabilities = {
                name: chart_probabilities(chart_models[name], test_chart_images, temperatures[name])
                for name in ("equivariant", "ordinary", "augmented")
            }
            candidates = {
                "d4_equivariant_chart_cnn": torch.einsum("nb,nbc->nc", probabilities["equivariant"], one_branches),
                "capacity_matched_ordinary_chart_cnn": torch.einsum("nb,nbc->nc", probabilities["ordinary"], one_branches),
                "augmentation_trained_ordinary_chart_cnn": torch.einsum("nb,nbc->nc", probabilities["augmented"], one_branches),
                "generic_low_rank_adapter": adapter_logits(adapter, test_chart_images, probabilities["ordinary"]),
                "structured_retransport": retransport_logits(test_chart_images, experts, probabilities["equivariant"]),
                "direct_d4_equivariant_task_cnn": model_logits(direct, test_chart_images),
                "supplied_chart_oracle_diagnostic": oracle,
                "context_blind_diagnostic": blind,
            }
            audit = save_logits_before_evaluation(
                f"sample_efficiency_seed{seed}_budget{budget}", candidates, test_labels[order], 278_000_000 + seed + budget
            )
            model_map = {
                "d4_equivariant_chart_cnn": [chart_models["equivariant"], experts[0]],
                "capacity_matched_ordinary_chart_cnn": [chart_models["ordinary"], experts[0]],
                "augmentation_trained_ordinary_chart_cnn": [chart_models["augmented"], experts[0]],
                "generic_low_rank_adapter": [chart_models["ordinary"], adapter],
                "structured_retransport": [chart_models["equivariant"], *experts],
                "direct_d4_equivariant_task_cnn": [direct],
            }
            probability_name = {
                "d4_equivariant_chart_cnn": "equivariant",
                "capacity_matched_ordinary_chart_cnn": "ordinary",
                "augmentation_trained_ordinary_chart_cnn": "augmented",
                "generic_low_rank_adapter": "ordinary",
                "structured_retransport": "equivariant",
            }
            for method in METHODS:
                logits = candidates[method]
                metrics = extended_metrics(logits, test_labels[order])
                name = probability_name.get(method)
                condition_accuracies = [
                    float((logits[conditions == condition].argmax(1) == test_labels[order][conditions == condition]).float().mean())
                    for condition in sorted(set(conditions))
                ]
                selected_models = model_map[method]
                training_time = direct_time if method == "direct_d4_equivariant_task_cnn" else (
                    chart_times[str(name)] + (adapter_time if method == "generic_low_rank_adapter" else sum(expert_times) if method == "structured_retransport" else expert_times[0])
                )
                runs.append(
                    {
                        "setting_id": f"FashionMNIST_sample_efficiency_seed{seed}_budget{budget}",
                        "seed": seed,
                        "chart_label_budget": budget,
                        "method": method,
                        "chart_accuracy": float((probabilities[str(name)].argmax(1) == test_charts).float().mean()) if name is not None else "",
                        **metrics,
                        "worst_condition_task_accuracy": min(condition_accuracies),
                        "trainable_parameters": sum(parameter_count(model) for model in selected_models),
                        "stored_parameters": sum(parameter_count(model) for model in selected_models),
                        "stored_bytes": sum(model_bytes(model) for model in selected_models),
                        "training_time_seconds": training_time,
                        "chart_training_examples": 0 if method == "direct_d4_equivariant_task_cnn" else budget,
                        "validation_examples": len(validation_images),
                        "calibration_examples": len(calibration_images),
                        "test_examples": len(test_chart_images),
                        "logits_path": audit["logits_path"],
                        "logits_sha256": audit["logits_sha256"],
                        "label_permutation_hash_passed": bool(audit["candidate_hashes_unchanged"] and audit["file_hash_unchanged"]),
                        **provenance(SCRIPT, COMMAND + (" --smoke" if arguments.smoke else ""), seed),
                    }
                )
            checkpoint_path = TMP / "checkpoints" / "sample_efficiency" / f"seed_{seed}_budget_{budget}.pt"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            models = {**{f"expert_{index}": model for index, model in enumerate(experts)}, "direct": direct, **{f"chart_{key}": value for key, value in chart_models.items()}, "adapter": adapter}
            torch.save(checkpoint_payload(models, seed=seed, budget=budget, temperatures=temperatures), checkpoint_path)
            checkpoint_rows.append({"seed": seed, "chart_label_budget": budget, "path": str(checkpoint_path.relative_to(ROOT)), "role": "trained_model_state"})
    summary: list[dict[str, object]] = []
    for budget in budgets:
        for method in METHODS:
            block = [row for row in runs if int(row["chart_label_budget"]) == budget and row["method"] == method]
            summary.append(
                {
                    "chart_label_budget": budget,
                    "method": method,
                    "seeds": len(block),
                    "mean_task_accuracy": float(np.mean([float(row["task_accuracy"]) for row in block])),
                    "std_task_accuracy": float(np.std([float(row["task_accuracy"]) for row in block], ddof=1)) if len(block) > 1 else 0.0,
                    "mean_worst_condition_task_accuracy": float(np.mean([float(row["worst_condition_task_accuracy"]) for row in block])),
                    "mean_chart_accuracy": float(np.mean([float(row["chart_accuracy"]) for row in block if row["chart_accuracy"] != ""])) if any(row["chart_accuracy"] != "" for row in block) else "",
                    "mean_ece": float(np.mean([float(row["ece"]) for row in block])),
                    "mean_trainable_parameters": int(np.mean([int(row["trainable_parameters"]) for row in block])),
                    "mean_training_time_seconds": float(np.mean([float(row["training_time_seconds"]) for row in block])),
                }
            )
    paired: list[dict[str, object]] = []
    for budget in budgets:
        values = []
        for seed in seeds:
            structured = next(float(row["task_accuracy"]) for row in runs if row["seed"] == seed and row["chart_label_budget"] == budget and row["method"] == "structured_retransport")
            ordinary = next(float(row["task_accuracy"]) for row in runs if row["seed"] == seed and row["chart_label_budget"] == budget and row["method"] == "capacity_matched_ordinary_chart_cnn")
            values.append(structured - ordinary)
        mean, low, high = paired_bootstrap(values, 279_000_000 + budget)
        paired.append({"chart_label_budget": budget, "comparison": "structured_minus_ordinary", "collections": len(values), "mean_delta": mean, "ci_low": low, "ci_high": high})
    oracle_gain_target = float(np.mean(list(target_by_seed.values())))
    threshold_rows = []
    for method in METHODS:
        points = [(int(row["chart_label_budget"]), float(row["mean_task_accuracy"])) for row in summary if row["method"] == method]
        for label, threshold in (("task_accuracy_50_percent", 0.50), ("task_accuracy_60_percent", 0.60), ("oracle_gain_90_percent", oracle_gain_target)):
            threshold_rows.append({"method": method, "threshold": label, "target_task_accuracy": threshold, "interpolated_chart_labels_to_threshold": first_crossing(points, threshold)})
    write_csv(DEST / "runs.csv", runs)
    write_csv(DEST / "summary.csv", summary)
    write_csv(DEST / "paired.csv", paired)
    write_csv(DEST / "thresholds.csv", threshold_rows)
    write_csv(DEST / "checkpoint_manifest.csv", checkpoint_rows)
    latex_table(DEST / "tables" / "sample_efficiency.tex", ["chart_label_budget", "method", "mean_task_accuracy", "mean_worst_condition_task_accuracy", "mean_chart_accuracy", "mean_ece"], summary, "Chart-label sample efficiency")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(6.2, 4.6))
    for method in METHODS:
        block = [row for row in summary if row["method"] == method]
        axis.plot([row["chart_label_budget"] for row in block], [row["mean_task_accuracy"] for row in block], marker="o", label=method)
    axis.set(xscale="log", xlabel="Chart-labeled training examples", ylabel="Task accuracy")
    axis.legend(fontsize=6)
    figure.tight_layout()
    figure.savefig(DEST / "plots" / "sample_efficiency.pdf")
    plt.close(figure)
    factual_report(
        DEST / "report.md",
        "Chart-label sample efficiency",
        [
            f"Trained and evaluated {len(runs)} method-seed-budget combinations at chart-label budgets {list(budgets)}.",
            "Each seed uses disjoint local-task, chart-training, early-stopping, calibration, and fixed held-out test roles. Candidate logits were persisted before test labels were used for evaluation.",
            f"Threshold crossings include 50% and 60% task accuracy and 90% of the seed-averaged supplied-chart oracle gain target ({oracle_gain_target:.6f}).",
        ],
    )


if __name__ == "__main__":
    main()
