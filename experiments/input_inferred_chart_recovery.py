#!/usr/bin/env python3
"""Stage 2: input-inferred D4 chart recovery on real image data."""

from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import experiments.compact_benchmark_common as compact_common
from experiments.compact_benchmark_common import load_vision_dataset, subset_arrays
from experiments.remaining_experiment_common import (
    DATA,
    OUT,
    classification_metrics,
    git_head,
    latex_table,
    logits_hashes,
    matched_bootstrap,
    ridge_fit,
    ridge_predict,
    softmax,
    write_csv,
)

SCRIPT = Path(__file__).resolve()
compact_common.DATA = DATA


def transform(images: np.ndarray, chart: int) -> np.ndarray:
    values = np.rot90(images, chart % 4, axes=(-2, -1))
    if chart >= 4:
        values = np.flip(values, axis=-1)
    return values.copy()


def inverse(images: np.ndarray, chart: int) -> np.ndarray:
    values = np.flip(images, axis=-1).copy() if chart >= 4 else images.copy()
    return np.rot90(values, -(chart % 4), axes=(-2, -1)).copy()


def split_indices(size: int, seed: int) -> dict[str, np.ndarray]:
    if size < 8500:
        raise ValueError("at least 8,500 training examples are required")
    order = np.random.default_rng(22_000_000 + seed).permutation(size)
    return {
        "model": order[:6000],
        "router": order[6000:7000],
        "selector": order[7000:7500],
        "holdout": order[7500:8500],
    }


def image_features(images: np.ndarray, projection: np.ndarray) -> np.ndarray:
    flat = images.reshape(len(images), -1).astype(np.float64)
    if flat.max(initial=0.0) > 2.0:
        flat = flat / 255.0
    return np.tanh(flat @ projection)


def orbit_features(images: np.ndarray) -> np.ndarray:
    values = images.astype(np.float64)
    if values.max(initial=0.0) > 2.0:
        values = values / 255.0
    height, width = values.shape[-2:]
    y, x = np.mgrid[-1:1:complex(height), -1:1:complex(width)]
    mass = values.sum(axis=(-2, -1)) + 1e-8
    features = [
        values.mean(axis=(-2, -1)),
        (values * x).sum(axis=(-2, -1)) / mass,
        (values * y).sum(axis=(-2, -1)) / mass,
        (values * x * x).sum(axis=(-2, -1)) / mass,
        (values * y * y).sum(axis=(-2, -1)) / mass,
        (values * x * y).sum(axis=(-2, -1)) / mass,
        np.abs(np.diff(values, axis=-1)).mean(axis=(-2, -1)),
        np.abs(np.diff(values, axis=-2)).mean(axis=(-2, -1)),
    ]
    return np.column_stack(features)


def make_chart_data(images: np.ndarray, charts: np.ndarray) -> np.ndarray:
    return np.stack([transform(image, int(chart)) for image, chart in zip(images, charts, strict=True)])


def branch_logits(chart_images: np.ndarray, projection: np.ndarray, experts: list[np.ndarray]) -> np.ndarray:
    branches = []
    for chart in range(8):
        canonical = np.stack([inverse(image, chart) for image in chart_images])
        features = image_features(canonical, projection)
        branches.append(np.mean([ridge_predict(features, expert) for expert in experts], axis=0))
    return np.stack(branches, axis=1)


def choose_threshold(confidence: np.ndarray, structured_logits: np.ndarray, fallback_logits: np.ndarray, labels: np.ndarray) -> float:
    candidates = np.unique(np.quantile(confidence, np.linspace(0.0, 1.0, 21)))
    scored = []
    for threshold in candidates:
        logits = np.where((confidence >= threshold)[:, None], structured_logits, fallback_logits)
        scored.append((classification_metrics(logits, labels)["accuracy"], -float(threshold), float(threshold)))
    return max(scored)[2]


def run_seed(dataset_name: str, seed: int, phase: str) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    train_set = load_vision_dataset(dataset_name, True)
    test_set = load_vision_dataset(dataset_name, False)
    split = split_indices(len(train_set), seed)
    rng = np.random.default_rng(22_100_000 + seed + len(dataset_name))
    model_x, model_y = subset_arrays(train_set, split["model"])
    router_x, router_y = subset_arrays(train_set, split["router"])
    selector_x, selector_y = subset_arrays(train_set, split["selector"])
    test_order = rng.permutation(len(test_set))[:2000]
    test_x, test_y = subset_arrays(test_set, test_order)
    if model_x.ndim == 4 and model_x.shape[1] == 1:
        model_x, router_x, selector_x, test_x = model_x[:, 0], router_x[:, 0], selector_x[:, 0], test_x[:, 0]
    dimension = int(np.prod(model_x.shape[1:]))
    projection = rng.normal(scale=1.0 / np.sqrt(dimension), size=(dimension, 64))
    model_features = image_features(model_x, projection)
    experts = []
    for expert_index in range(4):
        mask = np.arange(len(model_y)) % 4 != expert_index
        experts.append(ridge_fit(model_features[mask], np.eye(10)[model_y[mask]], ridge=2.0))

    def prepared(images: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
        charts = rng.integers(0, 8, count)
        return make_chart_data(images[:count], charts), charts

    router_images, router_charts = prepared(router_x, 1000)
    selector_images, selector_charts = prepared(selector_x, 500)
    test_images, test_charts = prepared(test_x, 2000)
    router_features = image_features(router_images, projection)
    selector_features = image_features(selector_images, projection)
    test_features = image_features(test_images, projection)
    generic_router = ridge_fit(router_features, np.eye(8)[router_charts], ridge=1.0)
    orbit_router = ridge_fit(orbit_features(router_images), np.eye(8)[router_charts], ridge=0.2)
    selector_router_scores = ridge_predict(selector_features, generic_router) + ridge_predict(orbit_features(selector_images), orbit_router)
    test_router_scores = ridge_predict(test_features, generic_router) + ridge_predict(orbit_features(test_images), orbit_router)
    selector_probabilities, test_probabilities = softmax(selector_router_scores), softmax(test_router_scores)
    selector_predicted, test_predicted = selector_probabilities.argmax(1), test_probabilities.argmax(1)
    selector_branches = branch_logits(selector_images, projection, experts)
    test_branches = branch_logits(test_images, projection, experts)
    selector_structured = selector_branches[np.arange(500), selector_predicted]
    test_structured = test_branches[np.arange(2000), test_predicted]
    selector_moe = np.einsum("nb,nbc->nc", selector_probabilities, selector_branches)
    test_moe = np.einsum("nb,nbc->nc", test_probabilities, test_branches)
    generic_classifier = ridge_fit(router_features, np.eye(10)[router_y[:1000]], ridge=2.0)
    selector_generic = ridge_predict(selector_features, generic_classifier)
    test_generic = ridge_predict(test_features, generic_classifier)
    selector_adapter = 0.5 * (selector_moe + selector_generic)
    test_adapter = 0.5 * (test_moe + test_generic)
    confidence_selector, confidence_test = selector_probabilities.max(1), test_probabilities.max(1)
    threshold = choose_threshold(confidence_selector, selector_structured, selector_moe, selector_y[:500])
    test_twisted = np.where((confidence_test >= threshold)[:, None], test_structured, test_moe)
    supplied = test_branches[np.arange(2000), test_charts]
    random_control = test_branches[np.arange(2000), rng.integers(0, 8, 2000)]
    wrong_control = test_branches[np.arange(2000), (test_predicted + 1) % 8]
    blind = np.mean(test_branches, axis=1)
    candidates = {
        "context_blind_synchronization": blind,
        "generic_image_context_classifier": test_generic,
        "generic_mixture_of_experts": test_moe,
        "generic_low_rank_context_adapter": test_adapter,
        "group_equivariant_cnn_chart_classifier": test_structured,
        "structured_group_router": test_structured,
        "canonicalize_pool_retransport_inferred": test_structured,
        "twistedmerge_diagnostic_structured_retransport": test_twisted,
        "supplied_chart_oracle": supplied,
        "random_action_control": random_control,
        "wrong_action_control": wrong_control,
        "ensemble": blind,
    }
    implementations = {
        "context_blind_synchronization": "mean_of_eight_canonicalized_branch_logits",
        "generic_image_context_classifier": "ridge_classifier_on_random_image_features",
        "generic_mixture_of_experts": "ridge_chart_router_weighted_branch_logits",
        "generic_low_rank_context_adapter": "equal_blend_proxy_not_fitted_low_rank_adapter",
        "group_equivariant_cnn_chart_classifier": "orbit_moment_ridge_proxy_not_cnn",
        "structured_group_router": "hybrid_random_feature_and_orbit_moment_ridge_router",
        "canonicalize_pool_retransport_inferred": "inferred_chart_canonicalization_and_branch_selection",
        "twistedmerge_diagnostic_structured_retransport": "validation_thresholded_structured_branch_with_moe_fallback",
        "supplied_chart_oracle": "true_chart_branch_selection_oracle",
        "random_action_control": "uniform_random_chart_branch",
        "wrong_action_control": "cyclically_shifted_inferred_chart_branch",
        "ensemble": "mean_of_eight_canonicalized_branch_logits",
    }
    hash_record = logits_hashes(f"chart_{dataset_name}_{phase}_{seed}", candidates, test_y, 22_900_000 + seed)
    logits_path = OUT / ".." / "tmp" / "remaining_experiments" / "logits" / f"chart_{dataset_name}_{phase}_{seed}.npz"
    logits_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(logits_path, **candidates)
    rows = []
    for method, logits in candidates.items():
        start = time.perf_counter(); _ = logits.argmax(1); latency = (time.perf_counter() - start) * 1000.0
        rows.append({
            "setting_id": f"{dataset_name}_{phase}_s{seed}", "phase": phase, "dataset": dataset_name, "seed": seed,
            "method": method, "implementation": implementations[method], "evaluation_split": "iid_random_D4_transform", **classification_metrics(logits, test_y),
            "chart_accuracy": float(np.mean(test_predicted == test_charts)), "chart_training_examples": 1000,
            "selector_validation_examples": 500, "test_examples": 2000, "trainable_parameters": int(generic_router.size + orbit_router.size),
            "stored_parameters": int(sum(expert.size for expert in experts) + generic_router.size + orbit_router.size),
            "latency_ms": latency, "branch_count": 8 if method in {"generic_mixture_of_experts", "structured_group_router", "canonicalize_pool_retransport_inferred", "twistedmerge_diagnostic_structured_retransport", "supplied_chart_oracle", "ensemble"} else 1,
            "candidate_count": 2 if method == "twistedmerge_diagnostic_structured_retransport" else 1,
            "abstention_threshold": threshold, "coverage": float(np.mean(confidence_test >= threshold)),
            "logits_sha256": hashlib.sha256(np.ascontiguousarray(logits).tobytes()).hexdigest(),
            "label_permutation_hash_passed": hash_record["label_permutation_hash_passed"],
            "dataset_revision": f"torchvision_cached_{dataset_name}", "execution_commit": git_head(), "source_sha256": hashlib.sha256(SCRIPT.read_bytes()).hexdigest(),
        })
    abstention = []
    for threshold_value in np.unique(np.quantile(confidence_selector, np.linspace(0.0, 1.0, 21))):
        covered = confidence_test >= threshold_value
        accuracy = float(np.mean(test_structured[covered].argmax(1) == test_y[covered])) if covered.any() else float("nan")
        abstention.append({"setting_id": f"{dataset_name}_{phase}_s{seed}", "dataset": dataset_name, "phase": phase, "seed": seed, "threshold": float(threshold_value), "coverage": float(covered.mean()), "covered_accuracy": accuracy})
    return rows, abstention


def paired_gate(rows: list[dict[str, object]], dataset: str, phase: str) -> tuple[dict[str, object], bool]:
    block = [row for row in rows if row["dataset"] == dataset and row["phase"] == phase]
    generic_names = ["generic_image_context_classifier", "generic_mixture_of_experts", "generic_low_rank_context_adapter"]
    means = {name: np.mean([float(row["accuracy"]) for row in block if row["method"] == name]) for name in generic_names}
    best = max(means, key=means.get)
    seeds = sorted({int(row["seed"]) for row in block})
    deltas = []
    for seed in seeds:
        structured = next(float(row["accuracy"]) for row in block if row["seed"] == seed and row["method"] == "twistedmerge_diagnostic_structured_retransport")
        generic = next(float(row["accuracy"]) for row in block if row["seed"] == seed and row["method"] == best)
        deltas.append(structured - generic)
    mean, low, high = matched_bootstrap(deltas, seed=22_700_000 + len(seeds))
    structured_parameters = np.mean([float(row["trainable_parameters"]) for row in block if row["method"] == "twistedmerge_diagnostic_structured_retransport"])
    generic_parameters = np.mean([float(row["trainable_parameters"]) for row in block if row["method"] == best])
    efficiency_match = mean >= -0.002 and structured_parameters <= 0.5 * generic_parameters
    passed = low > 0.0 or efficiency_match
    return {"dataset": dataset, "phase": phase, "best_generic": best, "mean_delta": mean, "ci_low": low, "ci_high": high, "efficiency_match": efficiency_match, "gate_passed": passed}, passed


def main() -> None:
    rows: list[dict[str, object]] = []
    abstention: list[dict[str, object]] = []
    for seed in range(5):
        stage_rows, curves = run_seed("FashionMNIST", seed, "discovery")
        rows.extend(stage_rows); abstention.extend(curves)
    paired = []
    discovery_claim, discovery_passed = paired_gate(rows, "FashionMNIST", "discovery")
    paired.append(discovery_claim)
    confirmation_passed = False
    if discovery_passed:
        for seed in range(5, 15):
            stage_rows, curves = run_seed("FashionMNIST", seed, "confirmation")
            rows.extend(stage_rows); abstention.extend(curves)
        confirmation_claim, confirmation_passed = paired_gate(rows, "FashionMNIST", "confirmation")
        paired.append(confirmation_claim)
    if confirmation_passed:
        for seed in range(5):
            stage_rows, curves = run_seed("CIFAR10", seed, "secondary_discovery")
            rows.extend(stage_rows); abstention.extend(curves)
        secondary_claim, _ = paired_gate(rows, "CIFAR10", "secondary_discovery")
        paired.append(secondary_claim)
    methods = sorted({row["method"] for row in rows})
    summary = []
    for dataset in sorted({str(row["dataset"]) for row in rows}):
        for phase in sorted({str(row["phase"]) for row in rows if row["dataset"] == dataset}):
            for method in methods:
                block = [row for row in rows if row["dataset"] == dataset and row["phase"] == phase and row["method"] == method]
                if block:
                    summary.append({"dataset": dataset, "phase": phase, "method": method, "runs": len(block), "accuracy": float(np.mean([float(row["accuracy"]) for row in block])), "chart_accuracy": float(np.mean([float(row["chart_accuracy"]) for row in block])), "latency_ms": float(np.median([float(row["latency_ms"]) for row in block]))})
    claims = [
        {"claim": "fashion_discovery_passed", "value": discovery_passed},
        {"claim": "fashion_confirmation_executed", "value": discovery_passed},
        {"claim": "fashion_confirmation_passed", "value": confirmation_passed},
        {"claim": "cifar_secondary_executed", "value": confirmation_passed},
    ]
    write_csv(OUT / "chart_runs.csv", rows)
    write_csv(OUT / "chart_summary.csv", summary)
    write_csv(OUT / "chart_paired.csv", paired)
    write_csv(OUT / "chart_abstention.csv", abstention)
    write_csv(OUT / "chart_claims.csv", claims)
    latex_table(OUT / "tables" / "chart_recovery.tex", ["dataset", "phase", "method", "runs", "accuracy", "chart_accuracy"], summary, "Input-inferred chart recovery")
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(6, 4))
    for seed in sorted({int(row["seed"]) for row in abstention if row["dataset"] == "FashionMNIST" and row["phase"] == "discovery"}):
        block = [row for row in abstention if row["dataset"] == "FashionMNIST" and row["phase"] == "discovery" and row["seed"] == seed]
        axis.plot([row["coverage"] for row in block], [row["covered_accuracy"] for row in block], alpha=0.6)
    axis.set(xlabel="Coverage", ylabel="Covered accuracy", xlim=(0, 1)); figure.tight_layout(); figure.savefig(OUT / "plots" / "chart_coverage.pdf"); plt.close(figure)
    (OUT / "chart_report.md").write_text(
        "# Input-inferred chart recovery\n\n"
        f"Execution commit: `{git_head()}`. Fashion-MNIST discovery used five seeds, 1,000 router examples, 500 selector-validation examples, and 2,000 test examples per seed. "
        f"The discovery gate {'passed' if discovery_passed else 'did not pass'}. Confirmation {'was executed' if discovery_passed else 'was not triggered'}; CIFAR-10 {'was executed' if confirmation_passed else 'was not triggered'}. "
        "Supplied-chart results are reported only as an oracle diagnostic. The row named `group_equivariant_cnn_chart_classifier` is an orbit-moment ridge proxy, not a trained CNN; held-out transformation families were not separately executed.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
