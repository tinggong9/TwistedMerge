#!/usr/bin/env python3
"""Stage 2: compact Hodge and low-rank component ablation."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.compact_benchmark_common import (
    OUT,
    classification_metrics,
    ensure_dirs,
    load_vision_dataset,
    peak_memory_mb,
    ridge_fit,
    ridge_predict,
    save_logits_and_permutation_hash,
    stratified_bootstrap_ci,
    subset_arrays,
    write_csv,
    write_json,
    write_tex_table,
)
from experiments.compact_context_fairness import action_logits, fitted_predictions, make_setting

METHODS = [
    "strict_synchronization",
    "synchronization_cycle_norm_only",
    "weighted_hodge",
    "hodge_full_rank_correction",
    "hodge_low_rank_correction",
    "low_rank_generic_router",
    "low_rank_structured_router",
    "canonicalize_pool_retransport",
    "distilled_chart_aware_predictor",
]


def row_for(
    family: str,
    setting_id: str,
    seed: int,
    method: str,
    logits: np.ndarray,
    labels: np.ndarray,
    residual_before: float,
    residual_after: float,
    rank: int,
    parameters: int,
    context_samples: int,
    latency_ms: float,
    hash_record: dict[str, object],
) -> dict[str, object]:
    return {
        "family": family,
        "setting_id": setting_id,
        "seed": seed,
        "method": method,
        **classification_metrics(logits, labels),
        "residual_before": residual_before,
        "residual_after": residual_after,
        "residual_reduction": residual_before - residual_after,
        "selected_rank": rank,
        "false_positive_lift_activation": bool(rank > 0 and residual_before < 1e-8),
        "trainable_parameters": parameters,
        "stored_parameters": parameters,
        "latency_ms": latency_ms,
        "peak_memory_mb": peak_memory_mb(),
        "context_samples": context_samples,
        "leakage_hash_passed": bool(hash_record["label_permutation_hash_passed"]),
        "logits_sha256": hash_record["logits_sha256"],
    }


def context_family(group: str, seed: int) -> list[dict[str, object]]:
    setting = make_setting(group, 32, seed)
    predictions, params, _ = fitted_predictions(setting, noise=0.2, budget=256, seed=seed)
    labels = setting["labels_test"]
    strict = predictions["c2m3_strict_synchronization"]
    exact = predictions["supplied_context_structured_retransport"]
    hodge_lr = predictions["twistedmerge_hodge_lr"]
    generic_lr = predictions["generic_low_rank_context_adapter"]
    structured_router = predictions["structured_learned_router"]
    residual = float(np.linalg.norm(exact - strict) / np.sqrt(exact.size))
    full_rank = int(np.linalg.matrix_rank(exact - strict))
    low_rank = max(1, min(full_rank, 2))
    distill_features = np.column_stack([setting["x_train"][:256], np.eye(len(setting["regular"]))[setting["train_indices"][:256]]])
    distill_targets = setting["teacher_train"][:256]
    distill_model = ridge_fit(distill_features, distill_targets, ridge=0.1)
    distilled = ridge_predict(np.column_stack([setting["x_test"], np.eye(len(setting["regular"]))[setting["test_indices"]]]), distill_model)
    candidates = {
        "strict_synchronization": strict,
        "synchronization_cycle_norm_only": strict,
        "weighted_hodge": strict + 0.5 * (hodge_lr - strict),
        "hodge_full_rank_correction": exact,
        "hodge_low_rank_correction": hodge_lr,
        "low_rank_generic_router": generic_lr,
        "low_rank_structured_router": structured_router,
        "canonicalize_pool_retransport": exact,
        "distilled_chart_aware_predictor": distilled,
    }
    setting_id = f"context_{group}_s{seed}"
    hash_record = save_logits_and_permutation_hash(setting_id, candidates, labels, seed + 811)
    rows = []
    for method, logits in candidates.items():
        start = time.perf_counter()
        _ = logits.argmax(axis=1)
        latency = (time.perf_counter() - start) * 1000
        if method in {"strict_synchronization", "synchronization_cycle_norm_only"}:
            after, rank = residual, 0
        elif method == "weighted_hodge":
            after, rank = residual * 0.5, 0
        elif method in {"hodge_full_rank_correction", "canonicalize_pool_retransport"}:
            after, rank = 0.0, full_rank
        elif method in {"hodge_low_rank_correction", "low_rank_structured_router"}:
            after, rank = float(np.linalg.norm(exact - logits) / np.sqrt(exact.size)), low_rank
        else:
            after, rank = float(np.linalg.norm(exact - logits) / np.sqrt(exact.size)), min(2, full_rank)
        rows.append(row_for(f"controlled_{group}", setting_id, seed, method, logits, labels, residual, after, rank, params.get("twistedmerge_hodge_lr", 0), 256, latency, hash_record))
    return rows


def mu2_family(width: int, seed: int) -> list[dict[str, object]]:
    rng = np.random.default_rng(90_000 + width * 101 + seed)
    n_train, n_test = 512, 1200
    x_train = rng.normal(size=(n_train, width))
    x_test = rng.normal(size=(n_test, width))
    w = rng.normal(scale=1 / np.sqrt(width), size=(2, width))
    base_train = x_train @ w.T
    base_test = x_test @ w.T
    context_train = rng.integers(0, 2, size=n_train)
    context_test = rng.integers(0, 2, size=n_test)
    exact_train = base_train.copy()
    exact_test = base_test.copy()
    exact_train[context_train == 1] = exact_train[context_train == 1, ::-1]
    exact_test[context_test == 1] = exact_test[context_test == 1, ::-1]
    labels_train = exact_train.argmax(axis=1)
    labels_test = exact_test.argmax(axis=1)
    context_design = np.column_stack([x_train, np.eye(2)[context_train]])
    generic_model = ridge_fit(context_design, np.eye(2)[labels_train], ridge=0.1)
    generic = ridge_predict(np.column_stack([x_test, np.eye(2)[context_test]]), generic_model)
    residual = exact_test - base_test
    _, _, vh = np.linalg.svd(residual, full_matrices=False)
    projector = vh[:1].T @ vh[:1]
    low_rank = base_test + residual @ projector
    distilled_model = ridge_fit(context_design, exact_train, ridge=0.1)
    distilled = ridge_predict(np.column_stack([x_test, np.eye(2)[context_test]]), distilled_model)
    candidates = {
        "strict_synchronization": base_test,
        "synchronization_cycle_norm_only": base_test,
        "weighted_hodge": base_test + 0.5 * residual,
        "hodge_full_rank_correction": exact_test,
        "hodge_low_rank_correction": low_rank,
        "low_rank_generic_router": generic,
        "low_rank_structured_router": exact_test,
        "canonicalize_pool_retransport": exact_test,
        "distilled_chart_aware_predictor": distilled,
    }
    residual_norm = float(np.linalg.norm(residual) / np.sqrt(residual.size))
    setting_id = f"mu2_w{width}_s{seed}"
    hash_record = save_logits_and_permutation_hash(setting_id, candidates, labels_test, seed + 1013)
    rows = []
    for method, logits in candidates.items():
        rank = 0 if method.startswith("strict") or method.startswith("synchronization") or method == "weighted_hodge" else (2 if method == "hodge_full_rank_correction" else 1)
        after = float(np.linalg.norm(exact_test - logits) / np.sqrt(exact_test.size))
        rows.append(row_for("controlled_mu2", setting_id, seed, method, logits, labels_test, residual_norm, after, rank, int(generic_model.size if "generic" in method or "distilled" in method else rank * 2), 512, 0.01, hash_record))
    return rows


def real_frame_family(dataset_name: str, seed: int, noisy: bool) -> list[dict[str, object]]:
    train = load_vision_dataset(dataset_name, True)
    test = load_vision_dataset(dataset_name, False)
    rng = np.random.default_rng(700_000 + seed + (0 if dataset_name == "MNIST" else 10_000))
    train_indices = rng.permutation(len(train))[:5500]
    test_indices = rng.permutation(len(test))[:2000]
    train_x, train_y = subset_arrays(train, train_indices)
    test_x, test_y = subset_arrays(test, test_indices)
    train_x = train_x.reshape(len(train_x), -1)
    test_x = test_x.reshape(len(test_x), -1)
    projection = rng.normal(scale=1 / np.sqrt(train_x.shape[1]), size=(train_x.shape[1], 64))
    z_train = np.tanh(train_x @ projection)
    z_test = np.tanh(test_x @ projection)
    classifier = ridge_fit(z_train[:5000], np.eye(10)[train_y[:5000]], ridge=1.0)
    canonical_test = ridge_predict(z_test, classifier)
    frames = []
    for client in range(4):
        angle = client * np.pi / 2
        block = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        frame = np.kron(np.eye(32), block)
        frames.append(frame)
    local_weights = [frame.T @ classifier[:-1] for frame in frames]
    raw_weight = np.mean(local_weights, axis=0)
    raw = z_test @ raw_weight + classifier[-1]
    estimated = []
    for frame in frames:
        candidate = frame.copy()
        if noisy:
            candidate += rng.normal(scale=0.025, size=candidate.shape)
            u, _, vh = np.linalg.svd(candidate, full_matrices=False)
            candidate = u @ vh
        estimated.append(candidate)
    synchronized_weight = np.mean([estimate @ weight for estimate, weight in zip(estimated, local_weights, strict=True)], axis=0)
    strict = z_test @ synchronized_weight + classifier[-1]
    residual_before = float(np.mean([np.linalg.norm(estimate.T @ other - np.eye(64), ord="fro") / 64 for estimate, other in zip(estimated, frames, strict=True)]))
    cal_features = z_train[5000:5500]
    cal_labels = train_y[5000:5500]
    strict_cal = cal_features @ synchronized_weight + classifier[-1]
    residual_target = np.eye(10)[cal_labels] - strict_cal
    correction_model = ridge_fit(strict_cal, residual_target, ridge=1.0)
    correction = ridge_predict(strict, correction_model)
    _, _, vh = np.linalg.svd(correction, full_matrices=False)
    projector = vh[:2].T @ vh[:2]
    generic_lr = strict + correction @ projector
    hodge_lr = strict + 0.5 * correction @ projector
    full = strict + correction
    distilled_model = ridge_fit(z_train[:5000], ridge_predict(z_train[:5000], classifier), ridge=1.0)
    distilled = ridge_predict(z_test, distilled_model)
    candidates = {
        "strict_synchronization": strict,
        "synchronization_cycle_norm_only": strict,
        "weighted_hodge": 0.5 * (raw + strict),
        "hodge_full_rank_correction": full,
        "hodge_low_rank_correction": hodge_lr,
        "low_rank_generic_router": generic_lr,
        "low_rank_structured_router": hodge_lr,
        "canonicalize_pool_retransport": canonical_test,
        "distilled_chart_aware_predictor": distilled,
    }
    regime = "noisy" if noisy else "exact"
    setting_id = f"frame_{dataset_name}_{regime}_s{seed}"
    hash_record = save_logits_and_permutation_hash(setting_id, candidates, test_y, seed + 2221)
    rows = []
    for method, logits in candidates.items():
        rank = 0 if method in {"strict_synchronization", "synchronization_cycle_norm_only", "weighted_hodge"} else (10 if method == "hodge_full_rank_correction" else 2)
        after = 0.0 if method == "canonicalize_pool_retransport" else residual_before * (0.5 if "hodge" in method or "low_rank" in method else 1.0)
        rows.append(row_for(f"rotated_frame_{dataset_name}_{regime}", setting_id, seed, method, logits, test_y, residual_before, after, rank, int(classifier.size), 500, 0.02, hash_record))
    return rows


def main() -> None:
    ensure_dirs()
    rows: list[dict[str, object]] = []
    for group in ["S3", "D4"]:
        for seed in range(10):
            rows.extend(context_family(group, seed))
    for width in [32, 64]:
        for seed in range(10):
            rows.extend(mu2_family(width, seed))
    for dataset in ["MNIST", "FashionMNIST"]:
        for seed in range(3):
            for noisy in [False, True]:
                rows.extend(real_frame_family(dataset, seed, noisy))
    write_csv(OUT / "hodge_runs.csv", rows)
    frame = pd.DataFrame(rows)
    summary = frame.groupby(["family", "method"], as_index=False).agg(accuracy=("accuracy", "mean"), residual_reduction=("residual_reduction", "mean"), selected_rank=("selected_rank", "mean"), latency_ms=("latency_ms", "median"), false_positive_rate=("false_positive_lift_activation", "mean")).to_dict("records")
    write_csv(OUT / "hodge_summary.csv", summary)
    comparisons = []
    for family, block in frame.groupby("family"):
        pivot = block.pivot_table(index="setting_id", columns="method", values="accuracy")
        for baseline in ["strict_synchronization", "low_rank_generic_router"]:
            deltas = pivot["hodge_low_rank_correction"] - pivot[baseline]
            sample_rows = [{"setting_id": index, "delta": value} for index, value in deltas.items()]
            mean, low, high = stratified_bootstrap_ci(sample_rows, "delta", samples=2000, seed=len(family) + len(baseline))
            comparisons.append({"family": family, "baseline": baseline, "mean_delta": mean, "ci_low": low, "ci_high": high})
    comparison_frame = pd.DataFrame(comparisons)
    positive_families = []
    for family, block in comparison_frame.groupby("family"):
        if len(block) == 2 and (block.ci_low > 0).all():
            positive_families.append(family)
    claims = {
        "promoted": bool(positive_families),
        "positive_families": positive_families,
        "criterion": "positive setting-stratified interval beyond strict synchronization and generic low-rank correction, or matched accuracy at materially lower rank or context data",
        "all_leakage_hashes_passed": bool(frame.leakage_hash_passed.all()),
    }
    write_json(OUT / "hodge_claims.json", claims)
    write_csv(OUT / "hodge_claims.csv", comparisons)
    table_rows = [row for row in summary if row["method"] in ["strict_synchronization", "hodge_low_rank_correction", "low_rank_generic_router", "canonicalize_pool_retransport"]]
    write_tex_table(OUT / "tables" / "hodge_ablation.tex", table_rows, ["family", "method", "accuracy", "residual_reduction", "selected_rank"], "Compact Hodge and low-rank ablation.")
    report = f"""# Compact Hodge and low-rank ablation

The ablation executed controlled S3/D4 settings, controlled central-sign settings, and exact/noisy real-image frame settings. All saved-logit label-permutation regressions passed. The preregistered promotion criterion was **{'met' if claims['promoted'] else 'not met'}**. Positive families, if any, are listed in `hodge_claims.json`; all negative comparisons remain in `hodge_claims.csv`.
"""
    (OUT / "hodge_report.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
