#!/usr/bin/env python3
"""E2: validation-only calibration audit for the controlled confirmation."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.compact_benchmark_common import classification_metrics, softmax
from experiments.emergency_level2_confirmation import variants
from experiments.future_benchmark_common import OUT, bootstrap, label_independence_record, stage_result, write_csv

DEST = OUT / "emergency"


def nll(logits: np.ndarray, labels: np.ndarray) -> float:
    probs = softmax(logits)
    return float(-np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-12, 1)).mean())


def temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    grid = np.geomspace(0.2, 5.0, 81)
    return float(min(grid, key=lambda value: nll(logits / value, labels)))


def vector_scale(logits: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    classes = logits.shape[1]
    def objective(values: np.ndarray) -> float:
        scales = np.exp(values[:classes])
        bias = values[classes:]
        return nll(logits / scales + bias, labels)
    result = minimize(objective, np.zeros(classes * 2), method="L-BFGS-B", options={"maxiter": 80})
    return np.exp(result.x[:classes]), result.x[classes:]


def metrics(logits: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    basic = classification_metrics(logits, labels)
    probs = softmax(logits)
    target = np.eye(logits.shape[1])[labels]
    brier = float(np.mean(np.sum((probs - target) ** 2, axis=1)))
    class_ece = []
    for cls in range(logits.shape[1]):
        confidence = probs[:, cls]
        truth = labels == cls
        bins = np.linspace(0, 1, 11)
        value = 0.0
        for left, right in zip(bins[:-1], bins[1:], strict=True):
            mask = (confidence >= left) & (confidence < right if right < 1 else confidence <= right)
            if mask.any():
                value += float(mask.mean() * abs(truth[mask].mean() - confidence[mask].mean()))
        class_ece.append(value)
    return {"accuracy": basic["accuracy"], "nll": basic["loss"], "brier": brier, "ece": basic["ece"], "classwise_ece": float(np.mean(class_ece))}


def main() -> None:
    rows = []
    reliability = []
    base_methods = {
        "twistedmerge_hodge_lr": "twistedmerge_hodge_lr",
        "generic_mixture_of_experts": "generic_mixture_of_experts",
        "learned_unconstrained_matrix_context_action": "learned_unconstrained_matrix_context_action",
    }
    for group in ["S3", "D4"]:
        for seed in range(20, 30):
            for noise in [0.2, 0.5, 1.0]:
                candidates, _, setting = variants(group, seed, noise, 64)
                labels = setting["labels_test"]
                validation = np.arange(0, 400)
                test = np.arange(400, len(labels))
                calibrated = {}
                for name, source in base_methods.items():
                    raw = candidates[source]
                    temp = temperature(raw[validation], labels[validation])
                    calibrated[f"{name}_uncalibrated"] = raw
                    calibrated[f"{name}_temperature"] = raw / temp
                    if name == "twistedmerge_hodge_lr":
                        scales, bias = vector_scale(raw[validation], labels[validation])
                        calibrated[f"{name}_vector"] = raw / scales + bias
                record = label_independence_record(f"E2_{group}_{seed}_{noise}", calibrated, labels[test], seed + 2200)
                if not record["label_permutation_hash_passed"]:
                    raise RuntimeError("calibration saved-logit regression failed")
                for method, logits in calibrated.items():
                    result = metrics(logits[test], labels[test])
                    rows.append({"setting_id": f"{group}_s{seed}_n{noise}", "group": group, "seed": seed, "noise": noise, "validation_samples": len(validation), "test_samples": len(test), "method": method, **result, "leakage_hash_passed": True, "logits_sha256": record["logits_sha256"]})
                    probs = softmax(logits[test]); confidence = probs.max(1); correct = logits[test].argmax(1) == labels[test]
                    for left in np.linspace(0, 0.9, 10):
                        mask = (confidence >= left) & (confidence < left + 0.1 if left < 0.9 else confidence <= 1)
                        if mask.any(): reliability.append({"setting_id": f"{group}_s{seed}_n{noise}", "method": method, "bin_left": left, "mean_confidence": float(confidence[mask].mean()), "accuracy": float(correct[mask].mean()), "count": int(mask.sum())})
    frame = pd.DataFrame(rows)
    summary = frame.groupby(["group", "noise", "method"], as_index=False).agg(accuracy=("accuracy", "mean"), nll=("nll", "mean"), brier=("brier", "mean"), ece=("ece", "mean"), classwise_ece=("classwise_ece", "mean"))
    raw = frame[frame.method == "twistedmerge_hodge_lr_uncalibrated"].set_index("setting_id")
    scaled = frame[frame.method == "twistedmerge_hodge_lr_temperature"].set_index("setting_id")
    calibrated_delta = bootstrap(raw.nll - scaled.nll, seed=202)
    moderate = summary[summary.noise.isin([0.2, 0.5])]
    structured = moderate[moderate.method == "twistedmerge_hodge_lr_temperature"]
    generic = moderate[moderate.method.str.contains("generic_mixture_of_experts_temperature")]
    retains_accuracy = float(structured.accuracy.mean()) + 1e-12 >= float(generic.accuracy.mean())
    not_materially_worse = float(structured.nll.mean()) <= float(generic.nll.mean()) + 0.02 and float(structured.ece.mean()) <= float(generic.ece.mean()) + 0.02
    gate = bool(retains_accuracy and not_materially_worse)
    write_csv(DEST / "calibration_runs.csv", rows)
    write_csv(DEST / "calibration_summary.csv", summary.to_dict("records"))
    summary.to_latex(DEST / "tables" / "calibration.tex", index=False, float_format="%.5f")
    rel = pd.DataFrame(reliability)
    fig, ax = plt.subplots(figsize=(5, 5))
    for method in ["twistedmerge_hodge_lr_uncalibrated", "twistedmerge_hodge_lr_temperature", "generic_mixture_of_experts_temperature"]:
        block = rel[rel.method == method].groupby("bin_left").agg(mean_confidence=("mean_confidence", "mean"), accuracy=("accuracy", "mean"))
        ax.plot(block.mean_confidence, block.accuracy, marker="o", label=method)
    ax.plot([0, 1], [0, 1], "k--"); ax.set(xlabel="Confidence", ylabel="Accuracy", xlim=(0, 1), ylim=(0, 1)); ax.legend(fontsize=6); fig.tight_layout(); fig.savefig(DEST / "plots" / "reliability.pdf"); plt.close(fig)
    (DEST / "calibration_report.md").write_text(f"# Calibration and uncertainty audit\n\nAll temperature and vector parameters were fitted on a fixed 400-example validation split, with the remaining 800 examples used only for evaluation. Temperature scaling changed structured NLL by {calibrated_delta[0]:+.5f} on average. The joint accuracy/calibration gate was **{'passed' if gate else 'not passed'}**; noise 1.0 remains a negative-boundary condition.\n", encoding="utf-8")
    stage_result("E2", "completed" if gate else "negative", f"calibration gate {'passed' if gate else 'did not pass'}", gate_passed=gate, nll_delta=calibrated_delta)


if __name__ == "__main__":
    main()
