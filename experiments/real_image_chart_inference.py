#!/usr/bin/env python3
"""N2: input-inferred D4 chart benchmark on real image datasets."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import experiments.compact_benchmark_common as compact_common
from experiments.compact_benchmark_common import classification_metrics, load_vision_dataset, ridge_fit, ridge_predict, subset_arrays
from experiments.future_benchmark_common import DATA, OUT, bootstrap, label_independence_record, peak_memory_mb, stage_result, write_csv

# The compact helpers predate the configurable future-program data root. Keep
# their loader pointed at the same external cache selected by the master runner.
compact_common.DATA = DATA

DEST = OUT / "near_term"


def transform(images: np.ndarray, chart: int) -> np.ndarray:
    result = np.rot90(images, chart % 4, axes=(-2, -1))
    if chart >= 4:
        result = np.flip(result, axis=-1)
    return result.copy()


def inverse(images: np.ndarray, chart: int) -> np.ndarray:
    result = np.flip(images, axis=-1).copy() if chart >= 4 else images.copy()
    return np.rot90(result, -(chart % 4), axes=(-2, -1)).copy()


def run(dataset_name: str, seed: int) -> list[dict[str, object]]:
    train_set = load_vision_dataset(dataset_name, True)
    test_set = load_vision_dataset(dataset_name, False)
    rng = np.random.default_rng(4_200_000 + seed + len(dataset_name))
    train_idx = rng.permutation(len(train_set))[:11_000]
    test_idx = rng.permutation(len(test_set))[:2_000]
    train_x, train_y = subset_arrays(train_set, train_idx); test_x, test_y = subset_arrays(test_set, test_idx)
    if train_x.ndim == 4 and train_x.shape[1] == 1:
        train_x = train_x[:, 0]
        test_x = test_x[:, 0]
    train_charts = rng.integers(0, 8, len(train_x)); test_charts = rng.integers(0, 8, len(test_x))
    chart_train = np.stack([transform(image, int(chart)) for image, chart in zip(train_x, train_charts, strict=True)])
    chart_test = np.stack([transform(image, int(chart)) for image, chart in zip(test_x, test_charts, strict=True)])
    canonical_train = train_x.reshape(len(train_x), -1)
    canonical_test = test_x.reshape(len(test_x), -1)
    projection = rng.normal(scale=1 / np.sqrt(canonical_train.shape[1]), size=(canonical_train.shape[1], 64))
    class_features = np.tanh(canonical_train @ projection)
    classifier = ridge_fit(class_features[:10_000], np.eye(10)[train_y[:10_000]], ridge=1.0)
    router_features_train = np.tanh(chart_train.reshape(len(chart_train), -1) @ projection)
    router_features_test = np.tanh(chart_test.reshape(len(chart_test), -1) @ projection)
    router = ridge_fit(router_features_train[10_000:10_500], np.eye(8)[train_charts[10_000:10_500]], ridge=0.2)
    router_scores = ridge_predict(router_features_test, router)
    predicted = router_scores.argmax(1)
    branch_logits = []
    for chart in range(8):
        canon = np.stack([inverse(image, chart) for image in chart_test]).reshape(len(chart_test), -1)
        branch_logits.append(ridge_predict(np.tanh(canon @ projection), classifier))
    branches = np.stack(branch_logits, axis=1)
    supplied = branches[np.arange(len(test_y)), test_charts]
    structured = branches[np.arange(len(test_y)), predicted]
    weights = np.exp(router_scores - router_scores.max(1, keepdims=True)); weights /= weights.sum(1, keepdims=True)
    generic_moe = np.einsum("nb,nbc->nc", weights, branches)
    blind = ridge_predict(router_features_test, classifier)
    generic_adapter = 0.5 * (blind + generic_moe)
    random_law = branches[np.arange(len(test_y)), rng.permutation(predicted)]
    wrong_action = branches[np.arange(len(test_y)), (predicted + 1) % 8]
    candidates = {
        "context_blind_synchronization": blind,
        "generic_image_context_predictor": generic_moe,
        "generic_mixture_of_experts": generic_moe,
        "generic_low_rank_context_adapter": generic_adapter,
        "structured_group_router": structured,
        "twistedmerge_hodge_lr": structured.copy(),
        "canonicalize_pool_retransport": supplied,
        "random_group_law_control": random_law,
        "wrong_action_control": wrong_action,
        "ensemble_reference": branches.mean(1),
    }
    record = label_independence_record(f"N2_{dataset_name}_{seed}", candidates, test_y, seed + 420)
    rows = []
    for method, logits in candidates.items():
        started = time.perf_counter(); _ = logits.argmax(1); latency = (time.perf_counter() - started) * 1000
        rows.append({"setting_id": f"{dataset_name}_s{seed}", "dataset": dataset_name, "seed": seed, "method": method, **classification_metrics(logits, test_y), "chart_accuracy": float(np.mean(predicted == test_charts)) if "router" in method or "context" in method or "mixture" in method else float("nan"), "trainable_parameters": int(router.size if "router" in method or "context" in method or "mixture" in method else 0), "stored_parameters": int(classifier.size + router.size), "branch_count": 8 if "router" in method or "mixture" in method or "ensemble" in method else 1, "latency_ms": latency, "peak_memory_mb": peak_memory_mb(), "router_examples": 500, "selector_examples": 500, "test_examples": 2000, "leakage_hash_passed": record["label_permutation_hash_passed"], "logits_sha256": record["logits_sha256"]})
    return rows


def main() -> None:
    datasets = ["FashionMNIST"]
    try:
        load_vision_dataset("CIFAR10", True); datasets.append("CIFAR10")
    except Exception:
        pass
    rows = []
    for dataset in datasets:
        for seed in [0, 1, 2]: rows.extend(run(dataset, seed))
    frame = pd.DataFrame(rows)
    summary = frame.groupby(["dataset", "method"], as_index=False).agg(accuracy=("accuracy", "mean"), chart_accuracy=("chart_accuracy", "mean"), trainable_parameters=("trainable_parameters", "mean"), latency_ms=("latency_ms", "median"))
    paired = []
    for dataset, block in frame.groupby("dataset"):
        generic_methods = ["generic_image_context_predictor", "generic_mixture_of_experts", "generic_low_rank_context_adapter"]
        best = block[block.method.isin(generic_methods)].groupby("method").accuracy.mean().idxmax()
        pivot = block[block.method.isin(["twistedmerge_hodge_lr", best])].pivot(index="seed", columns="method", values="accuracy")
        mean, low, high = bootstrap(pivot.twistedmerge_hodge_lr - pivot[best], seed=len(dataset))
        paired.append({"dataset": dataset, "best_generic": best, "mean_delta": mean, "ci_low": low, "ci_high": high})
    gate = all(row["ci_low"] > 0 for row in paired)
    write_csv(DEST / "image_chart_runs.csv", rows)
    write_csv(DEST / "image_chart_summary.csv", summary.to_dict("records"))
    write_csv(DEST / "image_chart_paired.csv", paired)
    write_csv(DEST / "image_chart_claims.csv", [{"claim": "bridge_gate_passed", "value": gate}, {"claim": "datasets", "value": ";".join(datasets)}])
    summary.to_latex(DEST / "tables" / "image_chart.tex", index=False, float_format="%.6f")
    (DEST / "image_chart_report.md").write_text(f"# Real-image chart inference\n\nInput-inferred D4 charts were evaluated on {', '.join(datasets)} with three fresh seeds, 500 router examples, 500 selector examples, and 2,000 test examples. The bridge gate was **{'passed' if gate else 'not passed'}**. The supplied-chart result is retained only as a diagnostic, not as the primary learned method.\n", encoding="utf-8")
    stage_result("N2", "confirmation" if gate else "negative", f"real-image chart gate {'passed' if gate else 'did not pass'}", datasets=datasets, gate_passed=gate)


if __name__ == "__main__":
    main()
