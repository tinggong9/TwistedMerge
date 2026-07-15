#!/usr/bin/env python3
"""Complete-path Fashion-MNIST latency, memory, storage, and Pareto audit."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.chart_component_ablation import infer_method, method_model_names, prepare_seed
from experiments.chart_followup_common import (
    DEVICE,
    OUT,
    TMP,
    D4EquivariantChartCNN,
    ImageCNN,
    LowRankContextAdapter,
    OrbitTaskCNN,
    chart_probabilities,
    factual_report,
    inverse_d4,
    measure_actual,
    model_bytes,
    model_logits,
    parameter_count,
    provenance,
    retransport_logits,
    specialized_expert_logits,
    task_branches,
    task_feature_branches,
)
from experiments.next_program_common import latex_table, write_csv

SCRIPT = Path(__file__).resolve()
DEST = OUT / "cost"
COMMAND = "python experiments/fashion_complete_cost_audit.py"
BATCH_SIZES = (1, 8, 32, 128)
METHOD_MAP = {
    "single_ordinary_cnn": "single_canonical_raw",
    "direct_d4_equivariant_task_cnn": "direct_d4_equivariant_task_classifier",
    "d4_test_time_augmentation": "d4_test_time_augmentation",
    "generic_low_rank_context_adapter": "generic_low_rank_context_adapter",
    "generic_moe": "ordinary_chart_soft_moe",
    "d4_chart_soft_routing": "d4_chart_soft_moe",
    "hard_branch_selection": "d4_chart_hard_branch_selection",
    "canonicalize_pool_retransport": "canonicalize_pool_retransport",
    "uncertainty_weighted_retransport": "uncertainty_weighted_retransport",
    "abstaining_retransport": "abstaining_retransport",
    "ensemble": "ensemble_reference",
    "supplied_chart_oracle_diagnostic": "supplied_chart_oracle",
}


def instantiate_models() -> dict[str, torch.nn.Module]:
    models: dict[str, torch.nn.Module] = {}
    for index in range(4):
        models[f"canonical_{index}"] = ImageCNN(10, 1, width=12)
        models[f"specialized_{index}"] = ImageCNN(10, 1, width=12)
    models.update(
        {
            "direct_task": OrbitTaskCNN(1, width=10),
            "chart_equivariant": D4EquivariantChartCNN(1, width=10),
            "chart_ordinary_matched": ImageCNN(8, 1, width=7),
            "chart_ordinary_larger": ImageCNN(8, 1, width=14),
            "chart_ordinary_augmented": ImageCNN(8, 1, width=10),
            "adapter": LowRankContextAdapter(1, width=10),
        }
    )
    return models


def load_seed(seed: int) -> tuple[dict[str, torch.nn.Module], dict[str, object]]:
    path = TMP / "checkpoints" / "ablation" / f"seed_{seed}.pt"
    if not path.exists():
        raise FileNotFoundError(f"COST requires the executed ABLATION checkpoint: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    models = instantiate_models()
    for name, model in models.items():
        model.load_state_dict(checkpoint["models"][name])
        model.to(DEVICE).eval()
    return models, checkpoint


def read_ablation_accuracy() -> dict[tuple[int, str], float]:
    path = OUT / "ablation" / "runs.csv"
    if not path.exists():
        raise FileNotFoundError(f"COST requires the executed ABLATION ledger: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        return {(int(row["seed"]), row["method"]): float(row["task_accuracy"]) for row in csv.DictReader(handle)}


def pareto_rows(summary: list[dict[str, object]], cost_key: str) -> list[dict[str, object]]:
    result = []
    for row in summary:
        dominated = any(
            float(other["task_accuracy"]) >= float(row["task_accuracy"])
            and float(other[cost_key]) <= float(row[cost_key])
            and (
                float(other["task_accuracy"]) > float(row["task_accuracy"])
                or float(other[cost_key]) < float(row[cost_key])
            )
            for other in summary
            if other is not row
        )
        result.append({"method": row["method"], "task_accuracy": row["task_accuracy"], cost_key: row[cost_key], "pareto_optimal": not dominated})
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    seeds = [20] if arguments.smoke else list(range(20, 30))
    warmups = 1 if arguments.smoke else 10
    repeats = 2 if arguments.smoke else 100
    accuracy = read_ablation_accuracy()
    runs = []
    component_rows = []
    for seed in seeds:
        payload = prepare_seed(seed, arguments.smoke)
        models, checkpoint = load_seed(seed)
        ordinary_temperature = float(checkpoint["ordinary_temperature"])
        equivariant_temperature = float(checkpoint["equivariant_temperature"])
        threshold = float(checkpoint["abstention_threshold"])
        training_times = {str(key): float(value) for key, value in checkpoint["training_times"].items()}
        canonical = [models[f"canonical_{index}"] for index in range(4)]
        specialized = [models[f"specialized_{index}"] for index in range(4)]
        for batch_size in BATCH_SIZES:
            batch = payload["test_images"][:batch_size]
            charts = payload["test_charts"][:batch_size]
            ordinary_probability = chart_probabilities(models["chart_ordinary_larger"], batch, ordinary_temperature)
            equivariant_probability = chart_probabilities(models["chart_equivariant"], batch, equivariant_temperature)
            canonical_branches = task_branches(batch, canonical)
            specialized_branches = specialized_expert_logits(batch, specialized)
            feature_branches = task_feature_branches(batch, canonical)
            average_weight = torch.stack([model.head.weight.detach().cpu() for model in canonical]).mean(0)
            average_bias = torch.stack([model.head.bias.detach().cpu() for model in canonical]).mean(0)
            components = {
                "transformation": measure_actual(lambda: torch.stack([inverse_d4(batch, chart) for chart in range(8)]), warmups, repeats),
                "equivariant_chart_model": measure_actual(lambda: model_logits(models["chart_equivariant"], batch), warmups, repeats),
                "ordinary_chart_model": measure_actual(lambda: model_logits(models["chart_ordinary_larger"], batch), warmups, repeats),
                "canonical_branch_evaluation": measure_actual(lambda: task_branches(batch, canonical), warmups, repeats),
                "specialized_branch_evaluation": measure_actual(lambda: specialized_expert_logits(batch, specialized), warmups, repeats),
                "soft_routing": measure_actual(lambda: torch.einsum("nb,nbc->nc", equivariant_probability, canonical_branches), warmups, repeats),
                "hard_routing": measure_actual(lambda: canonical_branches[torch.arange(batch_size), equivariant_probability.argmax(1)], warmups, repeats),
                "feature_retransport_pool": measure_actual(lambda: torch.einsum("nb,nbf->nf", equivariant_probability, feature_branches) @ average_weight.T + average_bias, warmups, repeats),
            }
            for component, timing in components.items():
                component_rows.append(
                    {
                        "seed": seed,
                        "batch_size": batch_size,
                        "component": component,
                        "cold_start_latency_ms": timing["cold_start_latency_ms"],
                        "warm_start_latency_ms": timing["warm_start_latency_ms"],
                        "latency_q1_ms": timing["latency_q1_ms"],
                        "latency_q3_ms": timing["latency_q3_ms"],
                        "warmups": warmups,
                        "timed_repetitions": repeats,
                    }
                )
            for public_method, ablation_method in METHOD_MAP.items():
                timing = measure_actual(
                    lambda selected=ablation_method: infer_method(
                        selected,
                        batch,
                        charts,
                        models,
                        ordinary_temperature,
                        equivariant_temperature,
                        threshold,
                        251_000_000 + seed,
                    ),
                    warmups,
                    repeats,
                )
                names = method_model_names(ablation_method)
                runs.append(
                    {
                        "setting_id": f"FashionMNIST_cost_seed{seed}",
                        "seed": seed,
                        "method": public_method,
                        "ablation_method": ablation_method,
                        "batch_size": batch_size,
                        "task_accuracy": accuracy[seed, ablation_method],
                        "cold_start_latency_ms": timing["cold_start_latency_ms"],
                        "complete_path_latency_ms": timing["warm_start_latency_ms"],
                        "latency_q1_ms": timing["latency_q1_ms"],
                        "latency_q3_ms": timing["latency_q3_ms"],
                        "peak_process_rss_mb": timing["peak_process_memory_mb"],
                        "peak_mps_memory_mb": timing["peak_accelerator_memory_mb"],
                        "stored_bytes": sum(model_bytes(models[name]) for name in names),
                        "parameters": sum(parameter_count(models[name]) for name in names),
                        "training_time_seconds_energy_proxy": sum(training_times.get(name, 0.0) for name in names),
                        "chart_training_examples": 0 if public_method in {"single_ordinary_cnn", "direct_d4_equivariant_task_cnn", "d4_test_time_augmentation", "ensemble", "supplied_chart_oracle_diagnostic"} else len(payload["chart_images"]),
                        "warmups": warmups,
                        "timed_repetitions": repeats,
                        **provenance(SCRIPT, COMMAND + (" --smoke" if arguments.smoke else ""), seed),
                    }
                )
    summary = []
    for method in METHOD_MAP:
        block = [row for row in runs if row["method"] == method and int(row["batch_size"]) == 128]
        summary.append(
            {
                "method": method,
                "seeds": len(block),
                "task_accuracy": float(np.mean([float(row["task_accuracy"]) for row in block])),
                "complete_path_latency_ms_batch128": float(np.median([float(row["complete_path_latency_ms"]) for row in block])),
                "latency_q1_ms_batch128": float(np.median([float(row["latency_q1_ms"]) for row in block])),
                "latency_q3_ms_batch128": float(np.median([float(row["latency_q3_ms"]) for row in block])),
                "stored_bytes": int(np.mean([int(row["stored_bytes"]) for row in block])),
                "parameters": int(np.mean([int(row["parameters"]) for row in block])),
                "chart_training_examples": int(np.mean([int(row["chart_training_examples"]) for row in block])),
            }
        )
    latency_pareto = pareto_rows(summary, "complete_path_latency_ms_batch128")
    storage_pareto = pareto_rows(summary, "stored_bytes")
    chart_data_pareto = pareto_rows(summary, "chart_training_examples")
    single = next(row for row in summary if row["method"] == "single_ordinary_cnn")
    structured = next(row for row in summary if row["method"] == "canonicalize_pool_retransport")
    claims = [
        {"claim": "structured_complete_path_faster_than_single", "value": float(structured["complete_path_latency_ms_batch128"]) < float(single["complete_path_latency_ms_batch128"])},
        {"claim": "structured_storage_lower_than_single", "value": int(structured["stored_bytes"]) < int(single["stored_bytes"])},
        {"claim": "structured_on_accuracy_latency_pareto_frontier", "value": next(bool(row["pareto_optimal"]) for row in latency_pareto if row["method"] == "canonicalize_pool_retransport")},
        {"claim": "structured_on_accuracy_storage_pareto_frontier", "value": next(bool(row["pareto_optimal"]) for row in storage_pareto if row["method"] == "canonicalize_pool_retransport")},
        {"claim": "all_complete_paths_used_preregistered_timing_repetitions", "value": all(int(row["timed_repetitions"]) == repeats for row in runs)},
    ]
    write_csv(DEST / "runs.csv", runs)
    write_csv(DEST / "components.csv", component_rows)
    write_csv(DEST / "summary.csv", summary)
    write_csv(DEST / "pareto_latency.csv", latency_pareto)
    write_csv(DEST / "pareto_storage.csv", storage_pareto)
    write_csv(DEST / "pareto_chart_data.csv", chart_data_pareto)
    write_csv(DEST / "claims.csv", claims)
    latex_table(DEST / "tables" / "fashion_cost.tex", ["method", "task_accuracy", "complete_path_latency_ms_batch128", "stored_bytes", "parameters"], summary, "Complete Fashion-MNIST inference cost")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(6.2, 4.6))
    axis.scatter([row["complete_path_latency_ms_batch128"] for row in summary], [row["task_accuracy"] for row in summary])
    for row in summary:
        axis.annotate(str(row["method"]), (row["complete_path_latency_ms_batch128"], row["task_accuracy"]), fontsize=6)
    axis.set(xlabel="Complete-path latency, batch 128 (ms)", ylabel="Task accuracy")
    figure.tight_layout(); figure.savefig(DEST / "plots" / "accuracy_latency.pdf"); plt.close(figure)
    figure, axis = plt.subplots(figsize=(6.2, 4.6))
    axis.scatter([row["stored_bytes"] for row in summary], [row["task_accuracy"] for row in summary])
    for row in summary:
        axis.annotate(str(row["method"]), (row["stored_bytes"], row["task_accuracy"]), fontsize=6)
    axis.set(xlabel="Stored bytes", ylabel="Task accuracy")
    figure.tight_layout(); figure.savefig(DEST / "plots" / "accuracy_storage.pdf"); plt.close(figure)
    factual_report(
        DEST / "report.md",
        "Complete Fashion-MNIST end-to-end cost audit",
        [
            f"Execution commit: `{provenance(SCRIPT, COMMAND, 'aggregate')['execution_commit']}`. Ten ABLATION checkpoints were timed at batch sizes {list(BATCH_SIZES)} with 10 warm-ups and 100 synchronized repetitions per complete method path.",
            "Complete paths include transformations, chart inference, expert evaluation, routing, feature pooling or retransport where applicable, abstention, and final prediction. Component timings are separately recorded in `components.csv`.",
            "Speed, storage, and Pareto claims are reported independently in `claims.csv`; no component-only timing is used as a complete-path claim.",
        ],
    )


if __name__ == "__main__":
    main()
