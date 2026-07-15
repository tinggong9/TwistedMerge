#!/usr/bin/env python3
"""Stage 12 cross-benchmark capacity, latency, robustness, and distillation audit."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "overnight_program"


SOURCES = [
    ("practical_selector", "practical_selector_runs.csv", "accuracy", "method"),
    ("two_loop_context", "two_loop_context_runs.csv", "accuracy", "method"),
    ("quaternion_pose", "quaternion_pose_runs.csv", "pose_accuracy_under_10deg", "method"),
    ("pretrained_vision", "pretrained_vision_runs.csv", "average_accuracy", "method"),
    ("lora_holonomy", "lora_holonomy_runs.csv", "task_accuracy", "method"),
    ("federated_frame", "federated_frame_runs.csv", "accuracy", "method"),
    ("transformer", "transformer_runs.csv", "average_task_score", "method"),
]


def normalized_source(benchmark: str, filename: str, score_column: str, method_column: str) -> pd.DataFrame:
    frame = pd.read_csv(OUT / filename)
    result = pd.DataFrame({"benchmark": benchmark, "method": frame[method_column], "mean_accuracy": pd.to_numeric(frame[score_column], errors="coerce")})
    for target, candidates in {
        "actual_trainable_parameters": ["actual_trainable_parameters", "parameter_count"],
        "stored_parameters": ["stored_parameters", "parameter_count"],
        "parameter_multiplier": ["parameter_multiplier"],
        "branch_count": ["branch_count"],
        "measured_inference_time_seconds": ["measured_inference_time_seconds", "measured_inference_time_seconds_512"],
        "inference_multiplier": ["inference_multiplier"],
        "peak_memory_mb": ["peak_memory_mb"],
    }.items():
        result[target] = np.nan
        for candidate in candidates:
            if candidate in frame:
                result[target] = pd.to_numeric(frame[candidate], errors="coerce")
                break
    return result


def main() -> None:
    frames = [normalized_source(*source) for source in SOURCES]
    all_rows = pd.concat(frames, ignore_index=True)
    grouped = all_rows.groupby(["benchmark", "method"], as_index=False).mean(numeric_only=True)
    grouped["best_benchmark_accuracy"] = grouped.groupby("benchmark")["mean_accuracy"].transform("max")
    grouped["worst_case_regret"] = grouped["best_benchmark_accuracy"] - grouped["mean_accuracy"]
    grouped["inference_overhead"] = np.maximum(grouped["inference_multiplier"].fillna(1.0) - 1.0, 0.0)
    score_rows = []
    for row in grouped.to_dict("records"):
        for lambda_regret in (0.0, 0.1, 0.25):
            for lambda_inference in (0.0, 0.01, 0.05):
                score_rows.append({**row, "lambda_regret": lambda_regret, "lambda_inference": lambda_inference, "practical_score": row["mean_accuracy"] - lambda_regret * row["worst_case_regret"] - lambda_inference * row["inference_overhead"]})
    capacity = pd.DataFrame(score_rows)

    two_loop = pd.read_csv(OUT / "two_loop_context_summary.csv").set_index("method")
    quaternion = pd.read_csv(OUT / "quaternion_pose_summary.csv").set_index("method")
    robustness = pd.DataFrame([
        {"audit": "wrong_group_action", "benchmark": "two_loop_context", "reference": two_loop.loc["supplied_context_oracle", "mean_accuracy"], "control": two_loop.loc["wrong_group_action_control", "mean_accuracy"]},
        {"audit": "wrong_generator", "benchmark": "two_loop_context", "reference": two_loop.loc["supplied_context_oracle", "mean_accuracy"], "control": two_loop.loc["wrong_generator_control", "mean_accuracy"]},
        {"audit": "wrong_context_order", "benchmark": "two_loop_context", "reference": two_loop.loc["supplied_context_oracle", "mean_accuracy"], "control": two_loop.loc["wrong_order_control", "mean_accuracy"]},
        {"audit": "random_same_branch_count", "benchmark": "two_loop_context", "reference": two_loop.loc["supplied_context_oracle", "mean_accuracy"], "control": two_loop.loc["random_same_branch_count_control", "mean_accuracy"]},
        {"audit": "wrong_quaternion_sign", "benchmark": "quaternion_pose", "reference": quaternion.loc["two_branch_q_minus_q_lift", "mean_pose_accuracy_under_10deg"], "control": quaternion.loc["wrong_sign_control", "mean_pose_accuracy_under_10deg"]},
        {"audit": "random_quaternion_branch", "benchmark": "quaternion_pose", "reference": quaternion.loc["two_branch_q_minus_q_lift", "mean_pose_accuracy_under_10deg"], "control": quaternion.loc["random_two_branch_control", "mean_pose_accuracy_under_10deg"]},
    ])
    robustness["control_gap"] = robustness.reference - robustness.control

    hodge = json.loads((OUT / "hodge_lr_smoke.json").read_text())
    central = pd.read_csv(OUT / "central_mu2_summary.csv")
    central_means = central.groupby("method").mean(numeric_only=True)
    distillation = pd.DataFrame([
        {"benchmark": "hodge_lr_component", "teacher_metric": "KL_to_teacher", "teacher_value": 0.0, "distilled_value": hodge["distillation_final_kl"], "distillation_gap": hodge["distillation_final_kl"]},
        {"benchmark": "controlled_mu2", "teacher_metric": "accuracy", "teacher_value": central_means.loc["supplied_context_q2_branch_predictor", "mean_test_accuracy"], "distilled_value": central_means.loc["distilled_single_model_control", "mean_test_accuracy"], "distillation_gap": central_means.loc["supplied_context_q2_branch_predictor", "mean_test_accuracy"] - central_means.loc["distilled_single_model_control", "mean_test_accuracy"]},
    ])
    lift_configs = {
        "hodge_component_gate": bool(hodge["confidence_gate_activate"]),
        "natural_twist_promoted": bool(pd.read_csv(OUT / "natural_twist_claims.csv").query("gate == 'natural_twist_promoted'").passed.iloc[0]),
        "pretrained_branch_activated": bool(pd.read_csv(OUT / "pretrained_vision_runs.csv").branch_candidate_activated.any()),
        "transformer_certificate": bool(pd.read_csv(OUT / "transformer_runs.csv").certificate_passed.any()),
    }
    false_positive_rate = float(np.mean(list(lift_configs.values())))

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tables").mkdir(exist_ok=True)
    (OUT / "plots").mkdir(exist_ok=True)
    capacity.to_csv(OUT / "capacity_latency.csv", index=False)
    robustness.to_csv(OUT / "robustness_controls.csv", index=False)
    distillation.to_csv(OUT / "distillation_summary.csv", index=False)
    table = capacity[(capacity.lambda_regret == 0.1) & (capacity.lambda_inference == 0.01)].sort_values(["benchmark", "practical_score"], ascending=[True, False])
    table[["benchmark", "method", "mean_accuracy", "worst_case_regret", "inference_overhead", "practical_score"]].to_latex(OUT / "tables" / "capacity_latency.tex", index=False, float_format="%.4f")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(grouped.inference_overhead.fillna(0), grouped.mean_accuracy, alpha=0.65)
    ax.set_xlabel("Measured inference overhead")
    ax.set_ylabel("Benchmark-native accuracy/score")
    ax.set_title("Capacity/latency audit across heterogeneous smokes")
    fig.tight_layout()
    fig.savefig(OUT / "plots" / "capacity_latency.pdf")
    plt.close(fig)
    report = f"""# Stage 12: capacity, latency, and robustness audit

The audit normalizes {len(grouped)} benchmark-method summaries and reports the preregistered practical score for all 3×3 combinations of regret penalty {{0, 0.1, 0.25}} and inference penalty {{0, 0.01, 0.05}}. It does not select a lambda after seeing results. Wrong-action, wrong-generator, wrong-order/context, random-branch, and quaternion controls are retained in `robustness_controls.csv`.

Observed false-positive activation rate across four negative/certificate gates is {false_positive_rate:.4f}. False-negative rate is not identifiable without verified positive natural examples. Missing FLOPs, peak memory outside Stage 1, batch-size sensitivity, alignment-noise sweeps, and branch-count scaling are left missing rather than estimated; the available measurements do not support a full systems conclusion. Cross-benchmark native scores are not pooled into one headline accuracy.
"""
    (OUT / "capacity_latency_report.md").write_text(report, encoding="utf-8")
    distill_report = f"""# Distillation audit

The component smoke reduced teacher/student KL from {hodge['distillation_initial_kl']:.6g} to {hodge['distillation_final_kl']:.6g}. In controlled mu2, the supplied-context teacher and distilled single-model accuracies are recorded in `distillation_summary.csv`; the distilled model is not relabeled as a successful lift. No pretrained vision or language branch teacher had sufficient evidence for a full-scale distillation conclusion.
"""
    (OUT / "distillation_report.md").write_text(distill_report, encoding="utf-8")
    (OUT / "capacity_latency_config.json").write_text(json.dumps({"stage": 12, "sources": [source[0] for source in SOURCES], "lambda_regret": [0.0, 0.1, 0.25], "lambda_inference": [0.0, 0.01, 0.05], "false_positive_rate": false_positive_rate, "missing_not_imputed": ["FLOPs", "most_peak_memory", "batch_size_sensitivity", "false_negative_rate"]}, indent=2), encoding="utf-8")
    print(json.dumps({"method_summaries": len(grouped), "score_rows": len(capacity), "false_positive_rate": false_positive_rate}, indent=2))


if __name__ == "__main__":
    main()
