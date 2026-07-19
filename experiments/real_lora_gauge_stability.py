#!/usr/bin/env python3
"""Gauge-stability evaluation on the trained holonomy LoRA adapter corpus."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torchvision.datasets import CIFAR10

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.lora_gauge_alignment import effective_delta, gauge_transform, sample_gauge
from src.lora_gauge_practical import merge_trained_factors

REPORT_ROOT = ROOT / "reports" / "practical_twistedmerge" / "lora_practical_followup"
OUTPUT_ROOT = REPORT_ROOT / "real_adapter_gauge"
REUSE_MANIFEST = REPORT_ROOT / "reuse_manifest.csv"
METHODS = (
    "naive_factor_average",
    "full_delta_svd",
    "canonical_svd_factor_average",
    "pairwise_reference_alignment",
    "global_synchronization",
    "cycle_aware_alignment",
    "oracle_alignment",
)
PRIMARY_FAMILIES = ("orthogonal", "positive_diagonal", "dense")
ALL_FAMILIES = (*PRIMARY_FAMILIES, "ill_conditioned")
CONDITION_LIMITS = {
    "orthogonal": 1.0,
    "positive_diagonal": 8.0,
    "dense": 30.0,
    "ill_conditioned": 1e8,
}
PILOT_GROUPS = (0, 1, 2)
CONFIRMATION_GROUPS = (3, 4)
PRESERVATION_DELTA_TOLERANCE = 1e-8
PRESERVATION_LOGIT_TOLERANCE = 1e-7
STABILITY_TOLERANCE = 1e-6
NAIVE_DEPENDENCE_FLOOR = 1e-4
VALIDATION_NONINFERIORITY_MARGIN = 0.01
BOOTSTRAP_RESAMPLES = 4000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def git_dirty() -> bool:
    return bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def classification_metrics(logits: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    shifted = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    predictions = logits.argmax(axis=1)
    accuracy = float(np.mean(predictions == labels))
    nll = float(-np.mean(np.log(np.clip(probabilities[np.arange(len(labels)), labels], 1e-15, 1.0))))
    one_hot = np.eye(logits.shape[1])[labels]
    brier = float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))
    confidence = probabilities.max(axis=1)
    ece = 0.0
    for lower in np.linspace(0.0, 0.9, 10):
        mask = (confidence >= lower) & (confidence < lower + 0.1)
        if mask.any():
            ece += float(mask.mean() * abs(confidence[mask].mean() - (predictions[mask] == labels[mask]).mean()))
    return {"accuracy": accuracy, "nll": nll, "brier": brier, "ece": ece}


def relative_frobenius(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.linalg.norm(left - right, ord="fro") / max(np.linalg.norm(right, ord="fro"), 1e-15))


def read_reuse_rows() -> list[dict[str, str]]:
    with REUSE_MANIFEST.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 5:
        raise RuntimeError("reuse manifest must contain exactly five independent groups")
    for row in rows:
        for path_key, hash_key in (
            ("checkpoint_path", "checkpoint_sha256"),
            ("test_logits_path", "test_logits_sha256"),
            ("feature_cache_path", "feature_cache_sha256"),
        ):
            path = Path(row[path_key])
            if not path.is_file() or sha256_file(path) != row[hash_key]:
                raise RuntimeError(f"reused artifact failed integrity verification: {path}")
    split_path = ROOT / rows[0]["split_manifest_path"]
    if sha256_file(split_path) != rows[0]["split_manifest_sha256"]:
        raise RuntimeError("split manifest failed integrity verification")
    return rows


def load_corpus(data_dir: Path, *, include_test_labels: bool) -> dict[str, object]:
    rows = read_reuse_rows()
    feature_path = Path(rows[0]["feature_cache_path"])
    payload = torch.load(feature_path, map_location="cpu", weights_only=False)
    features = {
        name: value.detach().cpu().numpy().astype(np.float64)
        for name, value in payload["features"].items()
    }
    train_dataset = CIFAR10(data_dir, train=True, download=False)
    splits = {
        name: value.detach().cpu().numpy().astype(np.int64)
        for name, value in payload["splits"].items()
    }
    train_targets = np.asarray(train_dataset.targets, dtype=np.int64)
    labels: dict[str, np.ndarray] = {
        "validation": train_targets[splits["validation"]],
        "overlap_validation": train_targets[splits["overlap_validation"]],
    }
    if include_test_labels:
        test_dataset = CIFAR10(data_dir, train=False, download=False)
        labels["test"] = np.asarray(test_dataset.targets, dtype=np.int64)[splits["test"]]
    return {"rows": rows, "features": features, "labels": labels, "splits": splits}


def load_group(row: dict[str, str]) -> dict[str, object]:
    payload = torch.load(Path(row["checkpoint_path"]), map_location="cpu", weights_only=False)
    factors = []
    heads = []
    for chart in range(8):
        state = payload["states"][str(chart)]
        b = state["up.weight"].detach().cpu().numpy().astype(np.float64)
        a = state["down.weight"].detach().cpu().numpy().astype(np.float64)
        head_weight = state["head.weight"].detach().cpu().numpy().astype(np.float64)
        head_bias = state["head.bias"].detach().cpu().numpy().astype(np.float64)
        expected = payload["effective_adapters"][str(chart)].detach().cpu().numpy().astype(np.float64)
        observed = np.eye(b.shape[0]) + b @ a
        if relative_frobenius(observed, expected) > 1e-7:
            raise RuntimeError(f"checkpoint effective adapter mismatch in chart {chart}")
        factors.append((b, a))
        heads.append((head_weight, head_bias))
    return {"factors": factors, "heads": heads, "seed": int(row["group_seed"])}


def adapter_logits(
    factor: tuple[np.ndarray, np.ndarray],
    head: tuple[np.ndarray, np.ndarray],
    features: np.ndarray,
) -> np.ndarray:
    delta = effective_delta(factor)
    hidden = features @ (np.eye(delta.shape[0]) + delta).T
    return hidden @ head[0].T + head[1]


def merged_logits(
    delta: np.ndarray,
    heads: list[tuple[np.ndarray, np.ndarray]],
    features: np.ndarray,
) -> tuple[np.ndarray, list[np.ndarray]]:
    head_weight = np.mean([head[0] for head in heads], axis=0)
    head_bias = np.mean([head[1] for head in heads], axis=0)
    by_chart = []
    for chart in range(features.shape[0]):
        hidden = features[chart] @ (np.eye(delta.shape[0]) + delta).T
        by_chart.append(hidden @ head_weight.T + head_bias)
    return np.concatenate(by_chart), by_chart


def evaluate_merged(
    delta: np.ndarray,
    heads: list[tuple[np.ndarray, np.ndarray]],
    features: np.ndarray,
    labels: np.ndarray,
) -> tuple[dict[str, float], np.ndarray]:
    logits, by_chart = merged_logits(delta, heads, features)
    repeated_labels = np.tile(labels, features.shape[0])
    metrics = classification_metrics(logits, repeated_labels)
    metrics["worst_chart_accuracy"] = min(
        classification_metrics(chart_logits, labels)["accuracy"] for chart_logits in by_chart
    )
    return metrics, logits


def gauges_for(group: int, family: str, scramble: int, adapter_count: int, rank: int) -> list[np.ndarray]:
    family_index = ALL_FAMILIES.index(family)
    rng = np.random.default_rng(2026071900 + group * 100000 + family_index * 1000 + scramble)
    return [sample_gauge(rng, rank, family, CONDITION_LIMITS[family]) for _ in range(adapter_count)]


def preservation_metrics(
    original: list[tuple[np.ndarray, np.ndarray]],
    scrambled: list[tuple[np.ndarray, np.ndarray]],
    heads: list[tuple[np.ndarray, np.ndarray]],
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
) -> dict[str, float | bool]:
    delta_errors = []
    logit_errors = []
    disagreements = []
    accuracy_changes = []
    for chart, (before, after) in enumerate(zip(original, scrambled)):
        delta_errors.append(relative_frobenius(effective_delta(after), effective_delta(before)))
        logits_before = adapter_logits(before, heads[chart], validation_features[chart])
        logits_after = adapter_logits(after, heads[chart], validation_features[chart])
        logit_errors.append(float(np.max(np.abs(logits_after - logits_before))))
        disagreements.append(float(np.mean(logits_after.argmax(1) != logits_before.argmax(1))))
        accuracy_changes.append(
            abs(
                classification_metrics(logits_after, validation_labels)["accuracy"]
                - classification_metrics(logits_before, validation_labels)["accuracy"]
            )
        )
    maximum_delta = max(delta_errors)
    maximum_logit = max(logit_errors)
    maximum_disagreement = max(disagreements)
    maximum_accuracy_change = max(accuracy_changes)
    accepted = (
        maximum_delta <= PRESERVATION_DELTA_TOLERANCE
        and maximum_logit <= PRESERVATION_LOGIT_TOLERANCE
        and maximum_disagreement == 0.0
        and maximum_accuracy_change == 0.0
    )
    return {
        "maximum_individual_relative_delta_error": maximum_delta,
        "maximum_individual_logit_error": maximum_logit,
        "maximum_individual_prediction_disagreement": maximum_disagreement,
        "maximum_individual_validation_accuracy_change": maximum_accuracy_change,
        "accepted_scramble": accepted,
    }


def run_smoke(data_dir: Path, command: str) -> None:
    corpus = load_corpus(data_dir, include_test_labels=False)
    rows_by_seed = {int(row["group_seed"]): row for row in corpus["rows"]}
    group = load_group(rows_by_seed[0])
    preservation_rows = []
    failures = []
    for family in ALL_FAMILIES:
        gauges = gauges_for(0, family, -17, 8, 4)
        scrambled = [gauge_transform(*factor, gauge) for factor, gauge in zip(group["factors"], gauges)]
        metrics = preservation_metrics(
            group["factors"],
            scrambled,
            group["heads"],
            corpus["features"]["validation"],
            corpus["labels"]["validation"],
        )
        preservation_rows.append({"stage": "smoke", "group_seed": 0, "family": family, **metrics})
        if family in PRIMARY_FAMILIES and not metrics["accepted_scramble"]:
            failures.append({"stage": "smoke", "group_seed": 0, "family": family, "error": "primary preservation gate failed"})
    destination = OUTPUT_ROOT / "smoke"
    destination.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(preservation_rows).to_csv(destination / "preservation.csv", index=False)
    pd.DataFrame(failures, columns=("stage", "group_seed", "family", "error")).to_csv(destination / "failure_log.csv", index=False)
    passed = not failures
    write_json(
        destination / "config.json",
        {
            "stage": "smoke",
            "command": command,
            "execution_commit": git_head(),
            "worktree_dirty_during_execution": git_dirty(),
            "source_corpus_commit": "9c91bc707d1f44beb36fe0fdce43af9ce1be79ed",
            "primary_preservation_passed": passed,
            "test_labels_accessed": False,
        },
    )
    (destination / "report.md").write_text(
        "# Trained-adapter loading and preservation smoke\n\n"
        f"All five source bundles and their shared feature cache passed SHA-256 verification. "
        f"One non-evidentiary gauge per family was applied to all eight seed-0 adapters. "
        f"The primary preservation gate **{'passed' if passed else 'failed'}**. No adapter was trained and no test label was loaded.\n",
        encoding="utf-8",
    )
    if not passed:
        raise RuntimeError("trained-adapter preservation smoke failed")


def run_groups(
    group_ids: Iterable[int],
    corpus: dict[str, object],
    scrambles: int,
    stage: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows_by_seed = {int(row["group_seed"]): row for row in corpus["rows"]}
    run_rows: list[dict[str, object]] = []
    preservation_rows: list[dict[str, object]] = []
    failure_rows: list[dict[str, object]] = []
    for group_id in group_ids:
        group = load_group(rows_by_seed[group_id])
        factors = group["factors"]
        heads = group["heads"]
        identity_gauges = [np.eye(4) for _ in factors]
        original: dict[str, dict[str, object]] = {}
        for method in METHODS:
            result = merge_trained_factors(
                factors,
                method,
                planted_gauges=identity_gauges if method == "oracle_alignment" else None,
            )
            delta = effective_delta(result.factors)
            validation_metrics, validation_logits = evaluate_merged(
                delta, heads, corpus["features"]["validation"], corpus["labels"]["validation"]
            )
            test_metrics, test_logits = evaluate_merged(
                delta, heads, corpus["features"]["test"], corpus["labels"]["test"]
            )
            original[method] = {
                "delta": delta,
                "validation_metrics": validation_metrics,
                "validation_logits": validation_logits,
                "test_metrics": test_metrics,
                "test_logits": test_logits,
            }
        for family in ALL_FAMILIES:
            for scramble in range(scrambles):
                gauges = gauges_for(group_id, family, scramble, len(factors), 4)
                scrambled = [gauge_transform(*factor, gauge) for factor, gauge in zip(factors, gauges)]
                preservation = preservation_metrics(
                    factors,
                    scrambled,
                    heads,
                    corpus["features"]["validation"],
                    corpus["labels"]["validation"],
                )
                preservation_rows.append(
                    {
                        "stage": stage,
                        "group_seed": group_id,
                        "family": family,
                        "scramble": scramble,
                        "condition_limit": CONDITION_LIMITS[family],
                        "claim_scope": "primary_well_conditioned" if family in PRIMARY_FAMILIES else "ill_conditioned_boundary",
                        **preservation,
                    }
                )
                if not preservation["accepted_scramble"]:
                    failure_rows.append(
                        {
                            "stage": stage,
                            "group_seed": group_id,
                            "family": family,
                            "scramble": scramble,
                            "method": "individual_adapter_preservation",
                            "error_type": "PreservationGateFailure",
                            "message": "scramble rejected before merge evaluation",
                        }
                    )
                    continue
                oracle_result = merge_trained_factors(scrambled, "oracle_alignment", planted_gauges=gauges)
                oracle_delta = effective_delta(oracle_result.factors)
                for method in METHODS:
                    started = time.perf_counter()
                    try:
                        result = merge_trained_factors(
                            scrambled,
                            method,
                            planted_gauges=gauges if method == "oracle_alignment" else None,
                        )
                        delta = effective_delta(result.factors)
                        merge_seconds = time.perf_counter() - started
                        validation_metrics, validation_logits = evaluate_merged(
                            delta,
                            heads,
                            corpus["features"]["validation"],
                            corpus["labels"]["validation"],
                        )
                        test_metrics, test_logits = evaluate_merged(
                            delta,
                            heads,
                            corpus["features"]["test"],
                            corpus["labels"]["test"],
                        )
                        base = original[method]
                        run_rows.append(
                            {
                                "stage": stage,
                                "group_seed": group_id,
                                "family": family,
                                "claim_scope": "primary_well_conditioned" if family in PRIMARY_FAMILIES else "ill_conditioned_boundary",
                                "scramble": scramble,
                                "method": method,
                                "status": "success",
                                "decision": result.decision,
                                "accepted_scramble": True,
                                **preservation,
                                "relative_delta_change_from_unscrambled": relative_frobenius(delta, base["delta"]),
                                "relative_delta_distance_from_oracle": relative_frobenius(delta, oracle_delta),
                                "maximum_validation_logit_change": float(np.max(np.abs(validation_logits - base["validation_logits"]))),
                                "maximum_test_logit_change": float(np.max(np.abs(test_logits - base["test_logits"]))),
                                "test_prediction_disagreement": float(np.mean(test_logits.argmax(1) != base["test_logits"].argmax(1))),
                                "validation_accuracy": validation_metrics["accuracy"],
                                "validation_accuracy_change": validation_metrics["accuracy"] - base["validation_metrics"]["accuracy"],
                                "validation_nll": validation_metrics["nll"],
                                "validation_brier": validation_metrics["brier"],
                                "validation_ece": validation_metrics["ece"],
                                "validation_worst_chart_accuracy": validation_metrics["worst_chart_accuracy"],
                                "test_accuracy": test_metrics["accuracy"],
                                "test_accuracy_change": test_metrics["accuracy"] - base["test_metrics"]["accuracy"],
                                "test_nll": test_metrics["nll"],
                                "test_brier": test_metrics["brier"],
                                "test_ece": test_metrics["ece"],
                                "test_worst_chart_accuracy": test_metrics["worst_chart_accuracy"],
                                "unscrambled_validation_accuracy": base["validation_metrics"]["accuracy"],
                                "unscrambled_test_accuracy": base["test_metrics"]["accuracy"],
                                "merge_seconds": merge_seconds,
                                "dense_allocation_count": result.dense_allocation_count,
                                "temporary_dense_bytes": result.temporary_dense_bytes,
                                "stored_factor_bytes": sum(value.nbytes for value in result.factors),
                                "output_rank": int(np.linalg.matrix_rank(delta)),
                                "output_rank_cap": result.output_rank_cap,
                                "max_cycle_frobenius_defect": result.max_cycle_frobenius_defect,
                                "max_cycle_spectral_defect": result.max_cycle_spectral_defect,
                            }
                        )
                    except Exception as error:
                        failure_rows.append(
                            {
                                "stage": stage,
                                "group_seed": group_id,
                                "family": family,
                                "scramble": scramble,
                                "method": method,
                                "error_type": type(error).__name__,
                                "message": str(error),
                            }
                        )
    return pd.DataFrame(run_rows), pd.DataFrame(preservation_rows), pd.DataFrame(
        failure_rows,
        columns=("stage", "group_seed", "family", "scramble", "method", "error_type", "message"),
    )


def bootstrap_interval(values: np.ndarray, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = np.empty(BOOTSTRAP_RESAMPLES)
    for index in range(BOOTSTRAP_RESAMPLES):
        means[index] = rng.choice(values, size=len(values), replace=True).mean()
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def aggregate_runs(runs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    layer_rows = []
    model_rows = []
    capacity_rows = []
    keys = ["claim_scope", "family", "group_seed", "method"]
    for key, frame in runs.groupby(keys, sort=True):
        scope, family, group_seed, method = key
        common = {
            "claim_scope": scope,
            "family": family,
            "group_seed": group_seed,
            "method": method,
            "dependent_scramble_count": len(frame),
            "independent_training_group_count": 1,
        }
        layer_rows.append(
            {
                **common,
                "mean_relative_delta_change": frame.relative_delta_change_from_unscrambled.mean(),
                "max_relative_delta_change": frame.relative_delta_change_from_unscrambled.max(),
                "mean_distance_from_oracle": frame.relative_delta_distance_from_oracle.mean(),
                "max_distance_from_oracle": frame.relative_delta_distance_from_oracle.max(),
                "minimum_output_rank": frame.output_rank.min(),
                "maximum_output_rank": frame.output_rank.max(),
                "output_rank_cap": frame.output_rank_cap.max(),
            }
        )
        model_rows.append(
            {
                **common,
                "mean_test_accuracy": frame.test_accuracy.mean(),
                "test_accuracy_std": frame.test_accuracy.std(ddof=0),
                "test_accuracy_range": frame.test_accuracy.max() - frame.test_accuracy.min(),
                "worst_scramble_test_accuracy": frame.test_accuracy.min(),
                "mean_test_accuracy_change": frame.test_accuracy_change.mean(),
                "mean_maximum_test_logit_change": frame.maximum_test_logit_change.mean(),
                "max_test_logit_change": frame.maximum_test_logit_change.max(),
                "mean_prediction_disagreement": frame.test_prediction_disagreement.mean(),
                "max_prediction_disagreement": frame.test_prediction_disagreement.max(),
                "mean_test_nll": frame.test_nll.mean(),
                "mean_test_brier": frame.test_brier.mean(),
                "mean_test_ece": frame.test_ece.mean(),
                "worst_chart_accuracy": frame.test_worst_chart_accuracy.min(),
            }
        )
        capacity_rows.append(
            {
                **common,
                "median_merge_seconds": frame.merge_seconds.median(),
                "p25_merge_seconds": frame.merge_seconds.quantile(0.25),
                "p75_merge_seconds": frame.merge_seconds.quantile(0.75),
                "minimum_merge_seconds": frame.merge_seconds.min(),
                "maximum_merge_seconds": frame.merge_seconds.max(),
                "maximum_dense_allocation_count": frame.dense_allocation_count.max(),
                "maximum_temporary_dense_bytes": frame.temporary_dense_bytes.max(),
                "stored_factor_bytes": frame.stored_factor_bytes.max(),
                "output_rank_cap": frame.output_rank_cap.max(),
                "failure_rate": 0.0,
            }
        )
    paired_rows = []
    metrics = (
        "relative_delta_change_from_unscrambled",
        "maximum_test_logit_change",
        "test_prediction_disagreement",
        "test_accuracy_change",
    )
    primary = runs[runs.family.isin(PRIMARY_FAMILIES)]
    for family in (*PRIMARY_FAMILIES, "all_primary"):
        selected = primary if family == "all_primary" else primary[primary.family == family]
        for method in METHODS:
            for metric in metrics:
                grouped = selected.groupby(["group_seed", "method"])[metric].mean().unstack()
                if method not in grouped or "naive_factor_average" not in grouped:
                    continue
                differences = (grouped[method] - grouped["naive_factor_average"]).dropna().to_numpy()
                low, high = bootstrap_interval(differences, 7100 + METHODS.index(method) * 31 + metrics.index(metric))
                paired_rows.append(
                    {
                        "family": family,
                        "metric": metric,
                        "method": method,
                        "baseline": "naive_factor_average",
                        "independent_training_group_count": len(differences),
                        "paired_mean_delta": float(differences.mean()),
                        "bootstrap_ci_low": low,
                        "bootstrap_ci_high": high,
                        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
                        "statistical_unit": "independent_adapter_training_group",
                    }
                )
    return pd.DataFrame(layer_rows), pd.DataFrame(model_rows), pd.DataFrame(paired_rows), pd.DataFrame(capacity_rows)


def phase_a_gates(runs: pd.DataFrame, preservation: pd.DataFrame, expected_groups: Iterable[int]) -> dict[str, bool]:
    expected = set(expected_groups)
    primary_preservation = preservation[preservation.family.isin(PRIMARY_FAMILIES)]
    preservation_passed = bool(primary_preservation.accepted_scramble.all())
    primary = runs[runs.family.isin(PRIMARY_FAMILIES)]
    naive = primary[primary.method == "naive_factor_average"].groupby("group_seed").relative_delta_change_from_unscrambled.max()
    global_rows = primary[primary.method == "global_synchronization"]
    global_max = global_rows.groupby("group_seed").relative_delta_change_from_unscrambled.max()
    naive_mean = primary[primary.method == "naive_factor_average"].groupby("group_seed").relative_delta_change_from_unscrambled.mean()
    global_mean = global_rows.groupby("group_seed").relative_delta_change_from_unscrambled.mean()
    validation_min = global_rows.groupby("group_seed").validation_accuracy_change.min()
    ranks = primary.output_rank_cap.max() if len(primary) else np.inf
    naive_dependent = expected.issubset(naive.index) and bool((naive.loc[list(expected)] > NAIVE_DEPENDENCE_FLOOR).all())
    global_stable = expected.issubset(global_max.index) and bool((global_max.loc[list(expected)] <= STABILITY_TOLERANCE).all())
    global_better = expected.issubset(global_mean.index) and bool((global_mean.loc[list(expected)] <= 0.1 * naive_mean.loc[list(expected)]).all())
    validation_safe = expected.issubset(validation_min.index) and bool((validation_min.loc[list(expected)] >= -VALIDATION_NONINFERIORITY_MARGIN).all())
    return {
        "every_primary_scramble_preserves_individual_adapters": preservation_passed,
        "naive_factor_average_representation_dependent_in_every_group": naive_dependent,
        "global_synchronization_invariant_in_every_group": global_stable,
        "global_synchronization_substantially_more_stable_than_naive": global_better,
        "global_synchronization_validation_noninferior_to_unscrambled": validation_safe,
        "same_rank_cap_for_compared_methods": bool(ranks == 4),
        "at_least_three_independent_groups": len(expected) >= 3,
    }


def write_plots(runs: pd.DataFrame, destination: Path) -> None:
    plots = destination / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    primary = runs[runs.family.isin(PRIMARY_FAMILIES)]
    order = list(METHODS)
    values = [primary[primary.method == method].relative_delta_change_from_unscrambled.clip(lower=1e-18) for method in order]
    fig, axis = plt.subplots(figsize=(10, 4.8))
    axis.boxplot(values, tick_labels=[name.replace("_", "\n") for name in order], showfliers=False)
    axis.set_yscale("log")
    axis.set_ylabel("Relative merged-delta change")
    axis.set_title("Trained-adapter gauge stability")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots / "gauge_stability.pdf")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 4.8))
    summarized = primary.groupby("method").test_accuracy_change.agg(["mean", "min", "max"]).reindex(order)
    axis.errorbar(
        np.arange(len(order)),
        summarized["mean"],
        yerr=[summarized["mean"] - summarized["min"], summarized["max"] - summarized["mean"]],
        fmt="o",
        capsize=3,
    )
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xticks(np.arange(len(order)), [name.replace("_", "\n") for name in order])
    axis.set_ylabel("Test accuracy change from unscrambled")
    axis.set_title("End-to-end stability across equivalent gauges")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots / "model_stability.pdf")
    plt.close(fig)


def artifact_manifest(destination: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(destination.rglob("*")):
        if path.is_file() and path.name != "artifact_manifest.csv":
            rows.append(
                {
                    "path": str(path.relative_to(ROOT)),
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                }
            )
    return pd.DataFrame(rows)


def write_evidence_bundle(
    runs: pd.DataFrame,
    preservation: pd.DataFrame,
    failures: pd.DataFrame,
    gates: dict[str, bool],
    stage: str,
    command: str,
    scrambles: int,
) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    layerwise, model, paired, capacity = aggregate_runs(runs)
    runs.to_csv(OUTPUT_ROOT / "runs.csv", index=False)
    preservation.to_csv(OUTPUT_ROOT / "preservation.csv", index=False)
    layerwise.to_csv(OUTPUT_ROOT / "layerwise_stability.csv", index=False)
    model.to_csv(OUTPUT_ROOT / "model_stability.csv", index=False)
    paired.to_csv(OUTPUT_ROOT / "paired_group_stats.csv", index=False)
    failures.to_csv(OUTPUT_ROOT / "failure_log.csv", index=False)
    capacity.to_csv(OUTPUT_ROOT / "capacity_cost.csv", index=False)
    write_plots(runs, OUTPUT_ROOT)
    tables = OUTPUT_ROOT / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    table = model[
        (model.family == "dense")
        & model.method.isin(("naive_factor_average", "full_delta_svd", "global_synchronization", "cycle_aware_alignment"))
    ][["method", "mean_test_accuracy", "test_accuracy_range", "max_test_logit_change"]]
    table.groupby("method").mean(numeric_only=True).reset_index().to_latex(
        tables / "real_adapter_gauge.tex", index=False, float_format="%.6g"
    )
    source_paths = (
        ROOT / "experiments" / "real_lora_gauge_stability.py",
        ROOT / "src" / "lora_gauge_practical.py",
        ROOT / "src" / "lora_gauge_alignment.py",
        ROOT / "tests" / "test_real_lora_gauge_stability.py",
    )
    config = {
        "stage": stage,
        "command": command,
        "execution_commit": git_head(),
        "worktree_dirty_during_execution": git_dirty(),
        "source_corpus_commit": "9c91bc707d1f44beb36fe0fdce43af9ce1be79ed",
        "training_commit": "4f0a08c9b7b4b0ead2e1450a9fdf57b8149d41b2",
        "independent_training_groups": sorted(int(value) for value in runs.group_seed.unique()),
        "adapters_per_group": 8,
        "adapter_rank": 4,
        "scrambles_per_family": scrambles,
        "gauge_families": list(ALL_FAMILIES),
        "condition_limits": CONDITION_LIMITS,
        "methods": list(METHODS),
        "preservation_delta_tolerance": PRESERVATION_DELTA_TOLERANCE,
        "preservation_logit_tolerance": PRESERVATION_LOGIT_TOLERANCE,
        "stability_tolerance": STABILITY_TOLERANCE,
        "validation_noninferiority_margin": VALIDATION_NONINFERIORITY_MARGIN,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "statistical_unit": "independent_adapter_training_group; scrambles averaged within group",
        "test_labels_used_for_selection": False,
        "gates": gates,
        "all_phase_a_gates_passed": all(gates.values()),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "source_hashes": {str(path.relative_to(ROOT)): sha256_file(path) for path in source_paths if path.is_file()},
    }
    write_json(OUTPUT_ROOT / "config.json", config)
    primary = runs[runs.family.isin(PRIMARY_FAMILIES)]
    method_summary = primary.groupby("method").agg(
        max_delta=("relative_delta_change_from_unscrambled", "max"),
        max_logit=("maximum_test_logit_change", "max"),
        accuracy_range=("test_accuracy", lambda values: float(values.max() - values.min())),
        failure_count=("status", lambda values: int((values != "success").sum())),
    )
    report_lines = [
        "# Real-adapter gauge stability",
        "",
        f"Evidence stage: **{stage}**. Independent training groups: {sorted(int(value) for value in runs.group_seed.unique())}. The trained holonomy factors were reused without modification; no adapter was retrained.",
        "",
        "## Gate decision",
        "",
    ]
    for name, passed in gates.items():
        report_lines.append(f"- `{name}`: `{passed}`")
    report_lines.extend(["", "## Primary method summary", ""])
    for method, row in method_summary.iterrows():
        report_lines.append(
            f"- `{method}`: maximum relative merged-delta change `{row.max_delta:.3e}`, maximum test-logit change `{row.max_logit:.3e}`, pooled test-accuracy range `{row.accuracy_range:.6f}`."
        )
    report_lines.extend(
        [
            "",
            "## Boundaries",
            "",
            "Gauge scrambles are dependent representations and are never bootstrapped as independent observations. Paired intervals resample training groups after scramble-level metrics are averaged within group. Full-delta SVD is an invariant baseline. The ill-conditioned family is excluded from the primary claim. Accuracy is reported as a preservation boundary, not a superiority claim. Holonomy, Brauer, period-index, invariant-pooling, linter, and broad baseline experiments were not rerun.",
        ]
    )
    (OUTPUT_ROOT / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    artifact_manifest(OUTPUT_ROOT).to_csv(OUTPUT_ROOT / "artifact_manifest.csv", index=False)


def copy_stage_evidence(stage: str) -> None:
    destination = OUTPUT_ROOT / ("pilot" if stage == "pilot" else "confirmation")
    destination.mkdir(parents=True, exist_ok=True)
    for name in (
        "runs.csv",
        "preservation.csv",
        "layerwise_stability.csv",
        "model_stability.csv",
        "paired_group_stats.csv",
        "failure_log.csv",
        "capacity_cost.csv",
        "config.json",
        "report.md",
    ):
        source = OUTPUT_ROOT / name
        (destination / name).write_bytes(source.read_bytes())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("smoke", "pilot", "confirmatory"), required=True)
    parser.add_argument("--scrambles", type=int, default=20)
    parser.add_argument("--data-dir", type=Path, default=Path("/Users/tinggong/Documents/GitHub/TwistedMerge/data"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.scrambles < 1:
        raise ValueError("scrambles must be positive")
    command = " ".join([sys.executable, *sys.argv])
    if args.stage == "smoke":
        run_smoke(args.data_dir, command)
        print(json.dumps({"stage": "smoke", "passed": True}, indent=2))
        return
    corpus = load_corpus(args.data_dir, include_test_labels=True)
    if args.stage == "pilot":
        runs, preservation, failures = run_groups(PILOT_GROUPS, corpus, args.scrambles, "pilot")
        gates = phase_a_gates(runs, preservation, PILOT_GROUPS)
        write_evidence_bundle(runs, preservation, failures, gates, "pilot", command, args.scrambles)
        copy_stage_evidence("pilot")
        artifact_manifest(OUTPUT_ROOT).to_csv(OUTPUT_ROOT / "artifact_manifest.csv", index=False)
    else:
        pilot_config_path = OUTPUT_ROOT / "pilot" / "config.json"
        if not pilot_config_path.is_file():
            raise RuntimeError("confirmatory extension requires completed pilot evidence")
        pilot_config = json.loads(pilot_config_path.read_text(encoding="utf-8"))
        if not pilot_config["all_phase_a_gates_passed"]:
            raise RuntimeError("pilot gate failed; confirmatory extension is forbidden")
        pilot_runs = pd.read_csv(OUTPUT_ROOT / "pilot" / "runs.csv")
        pilot_preservation = pd.read_csv(OUTPUT_ROOT / "pilot" / "preservation.csv")
        pilot_failures = pd.read_csv(OUTPUT_ROOT / "pilot" / "failure_log.csv")
        extension_runs, extension_preservation, extension_failures = run_groups(
            CONFIRMATION_GROUPS, corpus, args.scrambles, "confirmatory_extension"
        )
        runs = pd.concat([pilot_runs, extension_runs], ignore_index=True)
        preservation = pd.concat([pilot_preservation, extension_preservation], ignore_index=True)
        failures = pd.concat([pilot_failures, extension_failures], ignore_index=True)
        gates = phase_a_gates(runs, preservation, (*PILOT_GROUPS, *CONFIRMATION_GROUPS))
        write_evidence_bundle(runs, preservation, failures, gates, "confirmatory", command, args.scrambles)
        copy_stage_evidence("confirmatory")
        artifact_manifest(OUTPUT_ROOT).to_csv(OUTPUT_ROOT / "artifact_manifest.csv", index=False)
    print(
        json.dumps(
            {
                "stage": args.stage,
                "groups": sorted(int(value) for value in runs.group_seed.unique()),
                "run_rows": len(runs),
                "failures": len(failures),
                "gates": gates,
                "all_gates_passed": all(gates.values()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
