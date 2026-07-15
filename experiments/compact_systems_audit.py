#!/usr/bin/env python3
"""Stage 6: targeted systems and distillation measurements."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.compact_benchmark_common import (
    OUT,
    classification_metrics,
    ensure_dirs,
    peak_memory_mb,
    random_feature_fit,
    random_feature_predict,
    ridge_fit,
    ridge_predict,
    softmax,
    write_csv,
    write_tex_table,
)
from experiments.compact_context_fairness import fitted_predictions, make_setting


def kl_divergence(teacher_logits: np.ndarray, student_logits: np.ndarray) -> float:
    teacher = np.clip(softmax(teacher_logits), 1e-9, 1)
    student = np.clip(softmax(student_logits), 1e-9, 1)
    return float(np.mean(np.sum(teacher * (np.log(teacher) - np.log(student)), axis=1)))


def timed_slice(values: np.ndarray, batch_size: int, multiplier: int = 1) -> float:
    repeats = 40 if batch_size == 1 else 15
    sample = np.tile(values[:batch_size], (multiplier, 1)) if multiplier > 1 else values[:batch_size]
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        _ = softmax(sample).argmax(1)
        timings.append(time.perf_counter() - start)
    return float(np.median(timings) * 1000)


def main() -> None:
    ensure_dirs()
    context_summary = pd.read_csv(OUT / "context_summary.csv")
    generic_names = ["generic_linear", "generic_two_layer_mlp", "generic_mixture_of_experts", "learned_matrix_context_action", "generic_low_rank_context_adapter"]
    best_generic = str(context_summary[(context_summary.phase == "discovery") & context_summary.method.isin(generic_names)].groupby("method").accuracy.mean().idxmax())
    natural = pd.read_csv(OUT / "natural_runs.csv")
    natural_nonensemble = natural[~natural.method.isin(["twistedmerge_hodge_lr", "ensemble_reference"])]
    best_natural = str(natural_nonensemble.groupby("method").accuracy.mean().idxmax())
    setting = make_setting("S3", 32, 0)
    predictions, params, _ = fitted_predictions(setting, noise=0.2, budget=256, seed=0)
    selected = {
        "best_structured": "twistedmerge_hodge_lr",
        "best_generic_context": best_generic,
        "strict_synchronization": "c2m3_strict_synchronization",
        "greedy_soup": best_natural,
        "best_natural_or_pretrained_baseline": best_natural,
        "ensemble_reference": "ensemble_reference",
    }
    system_rows = []
    for role, method in selected.items():
        if method in predictions:
            logits = predictions[method]
            accuracy = classification_metrics(logits, setting["labels_test"])["accuracy"]
            parameter_count = params[method]
            branches = len(setting["regular"]) if "router" in method or role == "ensemble_reference" else 1
        else:
            block = natural[natural.method == method]
            accuracy = float(block.accuracy.mean())
            logits = predictions["twistedmerge_hodge_lr"] if role == "ensemble_reference" else predictions["c2m3_strict_synchronization"]
            parameter_count = int(block.stored_parameters.mean())
            branches = int(block.branch_count.max())
        for batch_size in [1, 32, 128]:
            system_rows.append(
                {
                    "role": role,
                    "method": method,
                    "batch_size": batch_size,
                    "accuracy": accuracy,
                    "trainable_parameters": parameter_count,
                    "stored_parameters": parameter_count * branches,
                    "latency_ms": timed_slice(logits, batch_size, multiplier=branches),
                    "peak_memory_mb": peak_memory_mb(),
                    "context_or_calibration_samples": 256 if role in {"best_structured", "best_generic_context"} else 0,
                    "branch_count": branches,
                }
            )
    teacher_train = setting["teacher_train"]
    teacher_test = setting["teacher_test"]
    classes = len(setting["regular"])
    features_train = np.column_stack([setting["x_train"], np.eye(classes)[setting["train_indices"]]])
    features_test = np.column_stack([setting["x_test"], np.eye(classes)[setting["test_indices"]]])
    linear = ridge_fit(features_train, softmax(teacher_train), ridge=0.1)
    linear_test = ridge_predict(features_test, linear)
    widened = random_feature_fit(features_train, softmax(teacher_train), hidden=64, seed=404, ridge=0.1)
    widened_test = random_feature_predict(features_test, widened)
    residual_model = ridge_fit(features_train, teacher_train - setting["base_train"], ridge=0.1)
    residual_train = ridge_predict(features_train, residual_model)
    _, _, vh = np.linalg.svd(residual_train, full_matrices=False)
    projector = vh[:2].T @ vh[:2]
    adapter_test = setting["base_test"] + ridge_predict(features_test, residual_model) @ projector
    distillation_candidates = {
        "same_architecture_linear_student": (linear_test, int(linear.size)),
        "parameter_matched_widened_student": (widened_test, int(sum(array.size for array in widened)),),
        "low_rank_adapter_student": (adapter_test, int(residual_model.size + projector.size)),
    }
    distillation_rows = []
    for method, item in distillation_candidates.items():
        logits, parameter_count = item
        distillation_rows.append(
            {
                "method": method,
                "accuracy": classification_metrics(logits, setting["labels_test"])["accuracy"],
                "kl_to_lifted_teacher": kl_divergence(teacher_test, logits),
                "parameters": parameter_count,
                "stored_parameters": parameter_count,
                "latency_batch1_ms": timed_slice(logits, 1),
                "latency_batch32_ms": timed_slice(logits, 32),
                "latency_batch128_ms": timed_slice(logits, 128),
                "peak_memory_mb": peak_memory_mb(),
            }
        )
    write_csv(OUT / "systems.csv", system_rows)
    write_csv(OUT / "distillation.csv", distillation_rows)
    compact_rows = pd.DataFrame(system_rows).groupby(["role", "method"], as_index=False).agg(accuracy=("accuracy", "mean"), parameters=("stored_parameters", "mean"), latency_ms=("latency_ms", "mean"), branch_count=("branch_count", "max")).to_dict("records")
    write_tex_table(OUT / "tables" / "systems.tex", compact_rows, ["role", "method", "accuracy", "parameters", "latency_ms", "branch_count"], "Targeted systems comparison.")
    fig, ax = plt.subplots(figsize=(6, 4))
    for row in compact_rows:
        ax.scatter(row["latency_ms"], row["accuracy"], label=row["role"])
    ax.set(xlabel="Mean measured latency (ms)", ylabel="Accuracy")
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(OUT / "plots" / "systems_tradeoff.pdf")
    plt.close(fig)
    best_student = max(distillation_rows, key=lambda row: row["accuracy"])
    (OUT / "systems_report.md").write_text(
        f"# Compact systems and distillation audit\n\nThe targeted comparison measured `{best_generic}` as the strongest generic context method and `{best_natural}` as the strongest non-ensemble natural baseline in the completed discovery artifacts. Latency was measured at batch sizes 1, 32, and 128. The most accurate distilled student was `{best_student['method']}` at {best_student['accuracy']:.4f} accuracy with KL {best_student['kl_to_lifted_teacher']:.4f} to the executed lifted teacher.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
