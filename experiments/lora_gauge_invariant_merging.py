#!/usr/bin/env python3
"""Controlled many-scramble smoke for gauge-invariant LoRA merging.

This program holds four effective rank-three updates fixed and changes only
their LoRA rank-space coordinates.  It is an algebraic and predictive smoke,
not a trained-adapter benchmark.  Gauge scrambles are dependent
representations of one adapter group and are never bootstrapped as independent
training groups.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.lora_cycle_diagnostics import cycle_aware_merge, triangle_cycle_metrics  # noqa: E402
from src.lora_gauge_alignment import (  # noqa: E402
    Array,
    Factor,
    align_factor,
    canonical_svd_factors,
    effective_delta,
    estimate_pairwise_transitions,
    factor_average,
    gauge_transform,
    global_align,
    mean_effective_delta,
    merged_factor_delta,
    reference_align,
    sample_gauge,
    truncated_svd,
)

DEFAULT_OUTPUT = ROOT / "reports" / "practical_twistedmerge" / "lora_gauge"
WELL_CONDITIONED_FAMILIES = ("orthogonal", "positive_diagonal", "dense")
ALL_FAMILIES = (*WELL_CONDITIONED_FAMILIES, "ill_conditioned")
GAUGE_CONDITIONS = {
    "orthogonal": 1.0,
    "positive_diagonal": 8.0,
    "dense": 30.0,
    "ill_conditioned": 1e10,
}
INVARIANCE_TOLERANCE = 1e-8
PRESERVATION_TOLERANCE = 1e-10


@dataclass(frozen=True)
class Fixture:
    base: Array
    factors: list[Factor]
    validation_x: Array
    validation_domains: Array
    validation_labels: Array
    test_x: Array
    test_domains: Array
    test_labels: Array


@dataclass(frozen=True)
class MergeOutput:
    name: str
    delta: Array | None
    validation_logits: Array
    test_logits: Array
    decision: str
    output_rank: int
    branch_count: int
    capacity_label: str
    implementation: str
    provenance: str
    merge_seconds: float
    failed: bool = False
    failure_reason: str = ""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def relative_fro(left: Array, right: Array) -> float:
    return float(np.linalg.norm(left - right, ord="fro") / max(np.linalg.norm(right, ord="fro"), 1e-15))


def build_fixture(seed: int) -> Fixture:
    """Create one fixed shared-subspace adapter group and frozen data splits."""

    rng = np.random.default_rng(seed)
    output_dim, input_dim, rank, adapter_count = 6, 12, 3, 4
    base = rng.normal(scale=0.16, size=(output_dim, input_dim))
    raw_b = rng.normal(size=(output_dim, rank))
    shared_b, _ = np.linalg.qr(raw_b)
    shared_b *= np.array([0.52, 0.40, 0.31])
    common_a = rng.normal(scale=0.20, size=(rank, input_dim))
    factors: list[Factor] = []
    for adapter in range(adapter_count):
        task_direction = rng.normal(scale=0.18, size=(rank, input_dim))
        task_direction += 0.07 * (adapter - 1.5)
        factors.append((shared_b.copy(), common_a + task_direction))

    def split(size: int) -> tuple[Array, Array, Array]:
        x = rng.normal(size=(size, input_dim))
        domains = rng.integers(0, adapter_count, size=size)
        task_deltas = [effective_delta(factor) for factor in factors]
        logits = np.stack(
            [x[index] @ (base + task_deltas[int(domain)]).T for index, domain in enumerate(domains)]
        )
        labels = logits.argmax(axis=1)
        return x, domains, labels

    validation_x, validation_domains, validation_labels = split(512)
    test_x, test_domains, test_labels = split(1024)
    return Fixture(
        base=base,
        factors=factors,
        validation_x=validation_x,
        validation_domains=validation_domains,
        validation_labels=validation_labels,
        test_x=test_x,
        test_domains=test_domains,
        test_labels=test_labels,
    )


def cross_entropy(logits: Array, labels: Array) -> float:
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_normalizer = np.log(np.exp(shifted).sum(axis=1))
    return float(np.mean(log_normalizer - shifted[np.arange(len(labels)), labels]))


def calibration_error(logits: Array, labels: Array, bins: int = 10) -> float:
    shifted = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    confidence = probabilities.max(axis=1)
    correct = logits.argmax(axis=1) == labels
    error = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for lower, upper in zip(edges[:-1], edges[1:]):
        selected = (confidence >= lower) & (confidence < upper if upper < 1.0 else confidence <= upper)
        if selected.any():
            error += float(selected.mean()) * abs(float(correct[selected].mean()) - float(confidence[selected].mean()))
    return error


def prediction_metrics(logits: Array, labels: Array, domains: Array) -> dict[str, float]:
    predictions = logits.argmax(axis=1)
    per_domain = [float((predictions[domains == index] == labels[domains == index]).mean()) for index in range(4)]
    return {
        "accuracy": float((predictions == labels).mean()),
        "cross_entropy": cross_entropy(logits, labels),
        "calibration_error": calibration_error(logits, labels),
        "worst_task_accuracy": min(per_domain),
        "mean_task_accuracy": float(np.mean(per_domain)),
    }


def internal_ties(deltas: Sequence[Array], keep_fraction: float = 0.5) -> Array:
    """Small deterministic TIES-style control; not an official implementation."""

    stack = np.stack(deltas)
    kept = np.zeros_like(stack)
    count = max(1, int(math.ceil(stack.shape[1] * stack.shape[2] * keep_fraction)))
    for index, delta in enumerate(stack):
        flat = np.abs(delta).reshape(-1)
        threshold = np.partition(flat, len(flat) - count)[len(flat) - count]
        kept[index] = np.where(np.abs(delta) >= threshold, delta, 0.0)
    elected_sign = np.sign(kept.sum(axis=0))
    agreement = (np.sign(kept) == elected_sign) & (kept != 0.0)
    numerator = np.where(agreement, kept, 0.0).sum(axis=0)
    denominator = np.maximum(agreement.sum(axis=0), 1)
    return numerator / denominator


def internal_dare(deltas: Sequence[Array], drop_rate: float = 0.5, seed: int = 9901) -> Array:
    """Deterministic DARE-style random drop/rescale control; not official."""

    rng = np.random.default_rng(seed)
    stack = np.stack(deltas)
    mask = rng.random(stack.shape) >= drop_rate
    return np.mean(stack * mask / (1.0 - drop_rate), axis=0)


def _delta_output(
    name: str,
    delta_builder: Callable[[], Array],
    fixture: Fixture,
    *,
    decision: str,
    branch_count: int = 1,
    capacity_label: str = "one fixed-rank adapter",
    implementation: str = "TwistedMerge",
    provenance: str = "controlled implementation",
    failed: bool = False,
    failure_reason: str = "",
    merge_seconds_override: float | None = None,
) -> MergeOutput:
    started = time.perf_counter()
    delta = delta_builder()
    measured_seconds = time.perf_counter() - started
    merge_seconds = measured_seconds if merge_seconds_override is None else merge_seconds_override
    return MergeOutput(
        name=name,
        delta=delta,
        validation_logits=fixture.validation_x @ (fixture.base + delta).T,
        test_logits=fixture.test_x @ (fixture.base + delta).T,
        decision=decision,
        output_rank=int(np.linalg.matrix_rank(delta)),
        branch_count=branch_count,
        capacity_label=capacity_label,
        implementation=implementation,
        provenance=provenance,
        merge_seconds=merge_seconds,
        failed=failed,
        failure_reason=failure_reason,
    )


def _safe_alignment_output(
    name: str,
    delta_builder: Callable[[], Array],
    fallback_delta: Array,
    fixture: Fixture,
    *,
    decision: str,
    capacity_label: str = "one fixed-rank adapter",
    implementation: str = "TwistedMerge",
    provenance: str = "controlled implementation",
) -> MergeOutput:
    """Record a numerical alignment failure and use the declared safe fallback."""

    try:
        return _delta_output(
            name,
            delta_builder,
            fixture,
            decision=decision,
            capacity_label=capacity_label,
            implementation=implementation,
            provenance=provenance,
        )
    except (ValueError, np.linalg.LinAlgError) as error:
        reason = f"{type(error).__name__}:{error}"
        return _delta_output(
            name,
            lambda: fallback_delta.copy(),
            fixture,
            decision="alignment_numerical_failure:fallback_full_delta_svd",
            capacity_label=capacity_label,
            implementation=implementation,
            provenance=provenance,
            failed=True,
            failure_reason=reason,
        )


def merge_methods(fixture: Fixture, factors: Sequence[Factor], planted_gauges: Sequence[Array]) -> dict[str, MergeOutput]:
    """Execute every controlled method on one representation of the adapters."""

    rank = factors[0][0].shape[1]
    deltas = [effective_delta(factor) for factor in factors]
    delta_mean = np.mean(deltas, axis=0)
    safe_fallback = truncated_svd(delta_mean, rank)
    outputs: dict[str, MergeOutput] = {}

    outputs["naive_factor_average"] = _delta_output(
        "naive_factor_average",
        lambda: merged_factor_delta(factors),
        fixture,
        decision="factorwise_mean_without_alignment",
        implementation="ordinary baseline",
        provenance="internal exact formula",
    )
    outputs["full_delta_svd"] = _delta_output(
        "full_delta_svd",
        lambda: truncated_svd(delta_mean, rank),
        fixture,
        decision="effective_delta_mean_then_truncated_svd",
        implementation="ordinary baseline",
        provenance="internal exact formula",
    )
    outputs["task_arithmetic"] = _delta_output(
        "task_arithmetic",
        lambda: delta_mean.copy(),
        fixture,
        decision="mean_effective_task_delta",
        capacity_label="one full-model delta",
        implementation="internal Task Arithmetic-style",
        provenance="internal; not official",
    )
    outputs["ties"] = _delta_output(
        "ties",
        lambda: internal_ties(deltas),
        fixture,
        decision="trim_elect_and_disjoint_mean",
        capacity_label="one full-model delta",
        implementation="internal TIES-style",
        provenance="internal; not official",
    )
    outputs["dare"] = _delta_output(
        "dare",
        lambda: internal_dare(deltas),
        fixture,
        decision="fixed_random_drop_and_rescale",
        capacity_label="one full-model delta",
        implementation="internal DARE-style",
        provenance="internal; not official",
    )
    outputs["compress_then_merge"] = _delta_output(
        "compress_then_merge",
        lambda: truncated_svd(np.mean([truncated_svd(delta, rank) for delta in deltas], axis=0), rank),
        fixture,
        decision="per_adapter_rank_compression_then_delta_merge",
        implementation="ordinary baseline",
        provenance="internal exact formula",
    )

    def pairwise_delta() -> Array:
        aligned, _, _ = reference_align(factors, mode="b")
        return merged_factor_delta(aligned)

    outputs["pairwise_reference_alignment"] = _safe_alignment_output(
        "pairwise_reference_alignment",
        pairwise_delta,
        safe_fallback,
        fixture,
        decision="B_subspace_maps_to_adapter_zero",
    )

    def global_delta() -> Array:
        aligned, _, _ = global_align(factors, mode="b")
        return merged_factor_delta(aligned)

    outputs["global_synchronization"] = _safe_alignment_output(
        "global_synchronization",
        global_delta,
        safe_fallback,
        fixture,
        decision="complete_graph_GL_rank_sync",
    )

    cycle_started = time.perf_counter()
    cycle_result = cycle_aware_merge(
        factors,
        rank=rank,
        cycle_tolerance=INVARIANCE_TOLERANCE,
        transition_condition_limit=1e8,
    )
    cycle_seconds = time.perf_counter() - cycle_started
    outputs["cycle_aware_alignment"] = _delta_output(
        "cycle_aware_alignment",
        lambda: cycle_result.delta.copy(),
        fixture,
        decision=f"{cycle_result.decision}:{cycle_result.reason}",
        merge_seconds_override=cycle_seconds,
    )

    def canonical_delta() -> Array:
        canonical = [canonical_svd_factors(delta, rank) for delta in deltas]
        b_mean, a_mean = factor_average(canonical)
        return b_mean @ a_mean

    outputs["canonical_svd_factor_average"] = _delta_output(
        "canonical_svd_factor_average",
        canonical_delta,
        fixture,
        decision="deterministic_effective_delta_factorization",
        implementation="ordinary gauge-invariant baseline",
        provenance="internal exact formula",
    )

    def oracle_delta() -> Array:
        identity = np.eye(rank)
        aligned = [
            align_factor(factor, np.linalg.solve(gauge, identity))
            for factor, gauge in zip(factors, planted_gauges)
        ]
        return merged_factor_delta(aligned)

    outputs["oracle_alignment"] = _safe_alignment_output(
        "oracle_alignment",
        oracle_delta,
        safe_fallback,
        fixture,
        decision="planted_Q_inverse",
        capacity_label="oracle fixed-rank adapter",
        implementation="oracle",
        provenance="controlled planted gauges",
    )

    started = time.perf_counter()
    validation_branch_logits = np.stack(
        [fixture.validation_x @ (fixture.base + delta).T for delta in deltas], axis=1
    )
    test_branch_logits = np.stack([fixture.test_x @ (fixture.base + delta).T for delta in deltas], axis=1)
    outputs["prediction_ensemble"] = MergeOutput(
        name="prediction_ensemble",
        delta=None,
        validation_logits=validation_branch_logits.mean(axis=1),
        test_logits=test_branch_logits.mean(axis=1),
        decision="mean_branch_predictions",
        output_rank=-1,
        branch_count=len(factors),
        capacity_label="ensemble",
        implementation="upper-bound control",
        provenance="all four effective adapters",
        merge_seconds=time.perf_counter() - started,
        failed=False,
        failure_reason="",
    )
    outputs["separate_adapters"] = MergeOutput(
        name="separate_adapters",
        delta=None,
        validation_logits=validation_branch_logits[
            np.arange(len(fixture.validation_domains)), fixture.validation_domains
        ],
        test_logits=test_branch_logits[np.arange(len(fixture.test_domains)), fixture.test_domains],
        decision="oracle_domain_routing_no_merge",
        output_rank=-1,
        branch_count=len(factors),
        capacity_label="separate adapters with oracle routing",
        implementation="upper-bound control",
        provenance="uses planted domain identity",
        merge_seconds=0.0,
        failed=False,
        failure_reason="",
    )
    return outputs


def scramble_group(
    fixture: Fixture, family: str, seed: int
) -> tuple[list[Factor], list[Array], dict[str, float | bool]]:
    rng = np.random.default_rng(seed)
    gauges = [
        sample_gauge(rng, factor[0].shape[1], family, GAUGE_CONDITIONS[family])
        for factor in fixture.factors
    ]
    factors = [gauge_transform(*factor, gauge) for factor, gauge in zip(fixture.factors, gauges)]
    maximum_delta_error = 0.0
    maximum_relative_delta_error = 0.0
    maximum_logit_error = 0.0
    disagreements = []
    for original, transformed in zip(fixture.factors, factors):
        original_delta = effective_delta(original)
        transformed_delta = effective_delta(transformed)
        maximum_delta_error = max(maximum_delta_error, float(np.max(np.abs(original_delta - transformed_delta))))
        maximum_relative_delta_error = max(maximum_relative_delta_error, relative_fro(transformed_delta, original_delta))
        original_logits = fixture.test_x @ (fixture.base + original_delta).T
        transformed_logits = fixture.test_x @ (fixture.base + transformed_delta).T
        maximum_logit_error = max(maximum_logit_error, float(np.max(np.abs(original_logits - transformed_logits))))
        disagreements.append(float(np.mean(original_logits.argmax(axis=1) != transformed_logits.argmax(axis=1))))
    accepted = family in WELL_CONDITIONED_FAMILIES and maximum_relative_delta_error <= PRESERVATION_TOLERANCE
    return factors, gauges, {
        "maximum_delta_error": maximum_delta_error,
        "maximum_relative_delta_error": maximum_relative_delta_error,
        "maximum_logit_error": maximum_logit_error,
        "maximum_prediction_disagreement": max(disagreements),
        "maximum_gauge_condition_number": max(float(np.linalg.cond(gauge)) for gauge in gauges),
        "accepted_for_primary_claim": accepted,
        "numerically_unstable": maximum_relative_delta_error > (
            PRESERVATION_TOLERANCE if family in WELL_CONDITIONED_FAMILIES else 1e-5
        ),
    }


def result_row(
    family: str,
    scramble_index: int,
    seed: int,
    output: MergeOutput,
    reference: MergeOutput,
    oracle: MergeOutput,
    fixture: Fixture,
    gauge_check: dict[str, float | bool],
    max_cycle_frobenius: float,
    max_cycle_spectral: float,
    max_transition_condition: float,
) -> dict[str, object]:
    validation = prediction_metrics(output.validation_logits, fixture.validation_labels, fixture.validation_domains)
    test = prediction_metrics(output.test_logits, fixture.test_labels, fixture.test_domains)
    delta_distance = (
        relative_fro(output.delta, reference.delta)
        if output.delta is not None and reference.delta is not None
        else float("nan")
    )
    delta_oracle_distance = (
        relative_fro(output.delta, oracle.delta)
        if output.delta is not None and oracle.delta is not None
        else float("nan")
    )
    logit_difference = output.test_logits - reference.test_logits
    oracle_logit_difference = output.test_logits - oracle.test_logits
    return {
        "family": family,
        "claim_scope": "primary_well_conditioned" if family in WELL_CONDITIONED_FAMILIES else "numerical_boundary_only",
        "scramble_index": scramble_index,
        "scramble_seed": seed,
        "method": output.name,
        "implementation": output.implementation,
        "provenance": output.provenance,
        "decision": output.decision,
        "validation_accuracy": validation["accuracy"],
        "test_accuracy": test["accuracy"],
        "test_cross_entropy": test["cross_entropy"],
        "test_calibration_error": test["calibration_error"],
        "worst_task_accuracy": test["worst_task_accuracy"],
        "mean_task_accuracy": test["mean_task_accuracy"],
        "relative_delta_distance_from_original_representation": delta_distance,
        "relative_delta_distance_from_oracle_alignment": delta_oracle_distance,
        "max_logit_distance_from_original_representation": float(np.max(np.abs(logit_difference))),
        "relative_logit_distance_from_original_representation": relative_fro(output.test_logits, reference.test_logits),
        "prediction_disagreement_from_original_representation": float(
            np.mean(output.test_logits.argmax(axis=1) != reference.test_logits.argmax(axis=1))
        ),
        "max_logit_distance_from_oracle_alignment": float(np.max(np.abs(oracle_logit_difference))),
        "output_rank": output.output_rank,
        "capacity_label": output.capacity_label,
        "branch_count": output.branch_count,
        "merge_seconds": output.merge_seconds,
        "method_failed": output.failed,
        "failure_reason": output.failure_reason,
        "validation_evaluation_count": 0,
        "test_labels_used_for_selection": False,
        "test_evaluation_count": 1,
        "max_cycle_frobenius_defect": max_cycle_frobenius,
        "max_cycle_spectral_defect": max_cycle_spectral,
        "max_transition_condition_number": max_transition_condition,
        **gauge_check,
    }


def aggregate_stability(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (family, method), group in runs.groupby(["family", "method"], sort=True):
        accuracy = group["test_accuracy"].to_numpy()
        rows.append(
            {
                "family": family,
                "claim_scope": group["claim_scope"].iloc[0],
                "method": method,
                "dependent_scramble_count": len(group),
                "independent_training_group_count": 1,
                "mean_test_accuracy": float(np.mean(accuracy)),
                "median_test_accuracy": float(np.median(accuracy)),
                "test_accuracy_std": float(np.std(accuracy, ddof=1)) if len(accuracy) > 1 else 0.0,
                "test_accuracy_range": float(np.max(accuracy) - np.min(accuracy)),
                "worst_setting_accuracy": float(np.min(accuracy)),
                "max_relative_delta_distance_from_original": float(
                    group["relative_delta_distance_from_original_representation"].max()
                ),
                "max_logit_distance_from_original": float(
                    group["max_logit_distance_from_original_representation"].max()
                ),
                "max_prediction_disagreement_from_original": float(
                    group["prediction_disagreement_from_original_representation"].max()
                ),
                "mean_relative_delta_distance_from_oracle": float(
                    group["relative_delta_distance_from_oracle_alignment"].mean()
                ),
                "mean_max_logit_distance_from_oracle": float(
                    group["max_logit_distance_from_oracle_alignment"].mean()
                ),
                "method_failure_count": int(group["method_failed"].sum()),
                "numerical_gauge_failure_count": int(group["numerically_unstable"].sum()),
                "paired_bootstrap_ci_low": float("nan"),
                "paired_bootstrap_ci_high": float("nan"),
                "ci_status": "not_computed_dependent_scrambles_single_training_group",
            }
        )
    return pd.DataFrame(rows)


def paired_comparisons(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for family in ALL_FAMILIES:
        family_rows = runs[runs.family == family]
        pivot = family_rows.pivot(index="scramble_index", columns="method", values="test_accuracy")
        for baseline in ("naive_factor_average", "full_delta_svd"):
            for method in pivot.columns:
                if method == baseline:
                    continue
                differences = (pivot[method] - pivot[baseline]).to_numpy()
                tolerance = 1e-12
                rows.append(
                    {
                        "family": family,
                        "method": method,
                        "baseline": baseline,
                        "dependent_pair_count": len(differences),
                        "mean_paired_accuracy_delta": float(np.mean(differences)),
                        "median_paired_accuracy_delta": float(np.median(differences)),
                        "paired_delta_std": float(np.std(differences, ddof=1)),
                        "wins": int(np.sum(differences > tolerance)),
                        "ties": int(np.sum(np.abs(differences) <= tolerance)),
                        "losses": int(np.sum(differences < -tolerance)),
                        "paired_bootstrap_ci_low": float("nan"),
                        "paired_bootstrap_ci_high": float("nan"),
                        "ci_status": "not_computed_dependent_scrambles_single_training_group",
                    }
                )
    return pd.DataFrame(rows)


def capacity_cost_table(runs: pd.DataFrame, fixture: Fixture) -> pd.DataFrame:
    rank = fixture.factors[0][0].shape[1]
    output_dim, input_dim = fixture.base.shape
    base_parameters = output_dim * input_dim
    factor_parameters = rank * (output_dim + input_dim)
    delta_parameters = output_dim * input_dim
    rows = []
    for method, group in runs.groupby("method", sort=True):
        sample = group.iloc[0]
        full_delta = sample.capacity_label == "one full-model delta"
        ensemble = sample.capacity_label in {"ensemble", "separate adapters with oracle routing"}
        branches = int(sample.branch_count)
        if full_delta:
            output_parameters = delta_parameters
        elif ensemble:
            output_parameters = branches * factor_parameters
        else:
            output_parameters = factor_parameters
        inference_multiplier = branches if method == "prediction_ensemble" else 1
        input_factor_bytes = 4 * factor_parameters * 8
        output_bytes = output_parameters * 8
        rows.append(
            {
                "method": method,
                "implementation": sample.implementation,
                "provenance": sample.provenance,
                "capacity_label": sample.capacity_label,
                "base_parameter_count": base_parameters,
                "output_parameter_count": output_parameters,
                "trainable_parameter_count": 0,
                "logical_adapter_trainable_parameter_count": factor_parameters,
                "adapter_rank": rank,
                "final_output_rank": int(group.output_rank.max()),
                "stored_bytes_float64": output_bytes,
                "inference_multiplier": inference_multiplier,
                "branch_count": branches,
                "mean_merge_compute_seconds": float(group.merge_seconds.mean()),
                "training_compute_seconds": 0.0,
                "estimated_peak_working_bytes": input_factor_bytes + output_bytes,
                "peak_memory_measurement_scope": "analytical factor and output arrays; excludes interpreter and BLAS workspace",
                "mean_inference_latency_seconds": float("nan"),
                "inference_latency_scope": "reported separately by repeated frozen test-forward timing",
                "validation_evaluation_count": 0,
                "recorded_method_failure_count": int(group.method_failed.sum()),
            }
        )
    return pd.DataFrame(rows)


def fill_latency(
    capacity: pd.DataFrame,
    reference_outputs: dict[str, MergeOutput],
    fixture: Fixture,
    repeats: int = 100,
) -> pd.DataFrame:
    values = capacity.copy()
    latencies = {}
    deltas = [effective_delta(factor) for factor in fixture.factors]
    for name, output in reference_outputs.items():
        started = time.perf_counter()
        for _ in range(repeats):
            if output.delta is not None:
                _ = fixture.test_x @ (fixture.base + output.delta).T
            elif name == "prediction_ensemble":
                _ = np.stack(
                    [fixture.test_x @ (fixture.base + delta).T for delta in deltas], axis=1
                ).mean(axis=1)
            else:
                routed = np.empty_like(output.test_logits)
                for domain, delta in enumerate(deltas):
                    selected = fixture.test_domains == domain
                    routed[selected] = fixture.test_x[selected] @ (fixture.base + delta).T
                _ = routed
        latencies[name] = (time.perf_counter() - started) / repeats
    values["mean_inference_latency_seconds"] = values.method.map(latencies)
    reference_latency = latencies["full_delta_svd"]
    values["measured_inference_latency_multiplier"] = (
        values.mean_inference_latency_seconds / max(reference_latency, 1e-15)
    )
    values["inference_latency_scope"] = (
        "100 repeated end-to-end NumPy forwards on frozen 1024-example test features"
    )
    return values


def write_plot(stability: pd.DataFrame, output_dir: Path) -> list[Path]:
    selected = [
        "naive_factor_average",
        "full_delta_svd",
        "pairwise_reference_alignment",
        "global_synchronization",
        "cycle_aware_alignment",
        "oracle_alignment",
    ]
    plot_data = stability[stability.method.isin(selected)].copy()
    family_order = list(ALL_FAMILIES)
    method_order = selected
    fig, axis = plt.subplots(figsize=(12, 5.5))
    width = 0.12
    x = np.arange(len(family_order))
    for index, method in enumerate(method_order):
        subset = plot_data[plot_data.method == method].set_index("family")
        values = [max(float(subset.loc[family, "max_logit_distance_from_original"]), 1e-18) for family in family_order]
        axis.bar(x + (index - 2.5) * width, values, width=width, label=method.replace("_", " "))
    axis.set_yscale("log")
    axis.set_ylabel("maximum absolute logit change vs original representation")
    axis.set_xticks(x, [family.replace("_", "\n") for family in family_order])
    axis.set_title("One fixed adapter group across dependent gauge scrambles")
    axis.legend(fontsize=8, ncol=2)
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    png = output_dir / "gauge_stability.png"
    pdf = output_dir / "gauge_stability.pdf"
    fig.savefig(png, dpi=180)
    fig.savefig(pdf)
    plt.close(fig)
    return [png, pdf]


def write_latex_table(stability: pd.DataFrame) -> Path:
    target_dir = ROOT / "reports" / "tables"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / "practical_lora_gauge_stability.tex"
    table = stability[
        (stability.family == "dense")
        & stability.method.isin(
            [
                "naive_factor_average",
                "full_delta_svd",
                "pairwise_reference_alignment",
                "global_synchronization",
                "cycle_aware_alignment",
                "oracle_alignment",
            ]
        )
    ][
        [
            "method",
            "dependent_scramble_count",
            "test_accuracy_range",
            "max_relative_delta_distance_from_original",
            "max_logit_distance_from_original",
        ]
    ].copy()
    table.to_latex(target, index=False, float_format="%.3e", escape=True)
    return target


def write_report(
    output_dir: Path,
    stability: pd.DataFrame,
    gauge_checks: pd.DataFrame,
    boundary: dict[str, object],
    gates: dict[str, bool],
    scrambles: int,
    alignment_failure_count: int,
    primary_alignment_failure_count: int,
) -> Path:
    dense = stability[stability.family == "dense"].set_index("method")
    naive = dense.loc["naive_factor_average"]
    global_sync = dense.loc["global_synchronization"]
    full_delta = dense.loc["full_delta_svd"]
    max_primary_preservation = gauge_checks[
        gauge_checks.family.isin(WELL_CONDITIONED_FAMILIES)
    ].maximum_relative_delta_error.max()
    report = f"""# Controlled LoRA gauge-invariance smoke

## Scope

This is one fixed synthetic four-adapter group with rank 3, not four trained adapters and not four independent seeds. Each well-conditioned gauge family has `{scrambles}` dependent scrambles of the same effective updates. No model or hyperparameter was selected, no validation metric was consulted, and test labels were used only after the protocol was fixed. A paired bootstrap CI is intentionally absent because the scramble rows are not independent training groups.

## Gauge preservation

The maximum relative effective-delta error over the orthogonal, positive-diagonal, and dense families is `{max_primary_preservation:.3e}`. The ill-conditioned family is reported separately as a numerical boundary and is excluded from the primary claim.

Alignment solvers recorded `{alignment_failure_count}` numerical failures across all method/representation rows, of which `{primary_alignment_failure_count}` occurred in the three well-conditioned primary families. Each is retained in `per_run.csv`; the affected method used the declared full-delta SVD safety fallback rather than silently dropping the setting.

## Primary representation-stability result

For moderately conditioned dense gauges:

- naive factor averaging has test-accuracy range `{naive.test_accuracy_range:.6f}`, maximum relative merged-delta change `{naive.max_relative_delta_distance_from_original:.3e}`, and maximum absolute logit change `{naive.max_logit_distance_from_original:.3e}`;
- global synchronization has test-accuracy range `{global_sync.test_accuracy_range:.6f}`, maximum relative merged-delta change `{global_sync.max_relative_delta_distance_from_original:.3e}`, and maximum absolute logit change `{global_sync.max_logit_distance_from_original:.3e}`;
- full-delta SVD has maximum relative merged-delta change `{full_delta.max_relative_delta_distance_from_original:.3e}` and maximum absolute logit change `{full_delta.max_logit_distance_from_original:.3e}`.

The controlled result supports gauge stability only for the planted shared-B rank space and well-conditioned transforms. Full-delta SVD, Task Arithmetic-style effective-delta averaging, internal TIES-style and DARE-style controls, fixed-rank compression, canonical SVD factors, the ensemble, and separate-adapter controls are gauge-invariant baselines because they operate on effective updates or predictions. TwistedMerge is not the only invariant method.

## Cycle-aware boundary

A separately labeled injected transition inconsistency has maximum normalized cycle defect `{float(boundary['max_cycle_frobenius_defect']):.3e}`. The cycle-aware method chose `{boundary['decision']}` and returned the gauge-invariant full-delta SVD fallback. This is a diagnostic-only controlled inconsistency, not a natural Brauer or period-index class.

## Gate decisions

| Gate | Passed |
|---|---:|
{chr(10).join(f'| `{name}` | `{passed}` |' for name, passed in gates.items())}

## Negative boundaries and next step

- No real adapter was trained or evaluated.
- No method is claimed to improve ordinary merging accuracy or beat TIES, DARE, soups, Task Arithmetic, or SVD broadly.
- The single adapter group cannot support confidence intervals or a performance-generalization claim.
- Ill-conditioned `GL(r)` transforms are a numerical boundary rather than part of the successful invariance scope.
- The real-adapter pilot remains blocked until dataset licensing is resolved and a frozen independent-training-group protocol is written.
"""
    path = output_dir / "report.md"
    path.write_text(report, encoding="utf-8")
    return path


def git_metadata() -> tuple[str, bool]:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    return commit, dirty


def run_smoke(seed: int, scrambles: int, output_dir: Path) -> dict[str, object]:
    if scrambles < 20:
        raise ValueError("the controlled smoke requires at least 20 scrambles per gauge family")
    output_dir.mkdir(parents=True, exist_ok=True)
    fixture = build_fixture(seed)
    rank = fixture.factors[0][0].shape[1]
    identity_gauges = [np.eye(rank) for _ in fixture.factors]
    reference_outputs = merge_methods(fixture, fixture.factors, identity_gauges)
    run_rows: list[dict[str, object]] = []
    gauge_rows: list[dict[str, object]] = []
    cycle_rows: list[dict[str, object]] = []

    for family_index, family in enumerate(ALL_FAMILIES):
        for scramble_index in range(scrambles):
            scramble_seed = seed + 100_000 * (family_index + 1) + scramble_index
            factors, gauges, gauge_check = scramble_group(fixture, family, scramble_seed)
            transitions = estimate_pairwise_transitions(factors, mode="b")
            cycles = triangle_cycle_metrics(transitions, len(factors))
            max_cycle_frobenius = max(metric.normalized_frobenius_defect for metric in cycles)
            max_cycle_spectral = max(metric.spectral_defect for metric in cycles)
            max_transition_condition = max(value.condition_number for value in transitions.values())
            gauge_rows.append(
                {
                    "family": family,
                    "claim_scope": "primary_well_conditioned"
                    if family in WELL_CONDITIONED_FAMILIES
                    else "numerical_boundary_only",
                    "scramble_index": scramble_index,
                    "scramble_seed": scramble_seed,
                    **gauge_check,
                }
            )
            for metric in cycles:
                cycle_rows.append(
                    {
                        "family": family,
                        "scramble_index": scramble_index,
                        "scramble_seed": scramble_seed,
                        "i": metric.i,
                        "j": metric.j,
                        "k": metric.k,
                        "normalized_frobenius_defect": metric.normalized_frobenius_defect,
                        "spectral_defect": metric.spectral_defect,
                        "holonomy_condition_number": metric.holonomy_condition_number,
                    }
                )
            outputs = merge_methods(fixture, factors, gauges)
            oracle = outputs["oracle_alignment"]
            for method, output in outputs.items():
                run_rows.append(
                    result_row(
                        family,
                        scramble_index,
                        scramble_seed,
                        output,
                        reference_outputs[method],
                        oracle,
                        fixture,
                        gauge_check,
                        max_cycle_frobenius,
                        max_cycle_spectral,
                        max_transition_condition,
                    )
                )

    runs = pd.DataFrame(run_rows)
    gauge_checks = pd.DataFrame(gauge_rows)
    cycles = pd.DataFrame(cycle_rows)
    stability = aggregate_stability(runs)
    comparisons = paired_comparisons(runs)
    capacity = fill_latency(capacity_cost_table(runs, fixture), reference_outputs, fixture)

    # A deliberately inconsistent transition edge tests abstention/fallback.
    boundary_transitions = {
        key: value.matrix.copy()
        for key, value in estimate_pairwise_transitions(fixture.factors, mode="b").items()
    }
    boundary_transitions[(0, 1)] = boundary_transitions[(0, 1)] @ np.diag([1.35, 1.0, 1.0])
    boundary_result = cycle_aware_merge(
        fixture.factors,
        transitions=boundary_transitions,
        rank=rank,
        cycle_tolerance=INVARIANCE_TOLERANCE,
    )
    boundary = {
        key: value
        for key, value in asdict(boundary_result).items()
        if key != "delta"
    }
    boundary["relative_delta_error_vs_full_delta_svd"] = relative_fro(
        boundary_result.delta, truncated_svd(mean_effective_delta(fixture.factors), rank)
    )
    boundary_table = pd.DataFrame([boundary])

    primary_runs = runs[runs.family.isin(WELL_CONDITIONED_FAMILIES)]
    primary_stability = stability[stability.family.isin(WELL_CONDITIONED_FAMILIES)]
    twisted_methods = {
        "pairwise_reference_alignment",
        "global_synchronization",
        "cycle_aware_alignment",
    }
    invariant_baselines = {
        "full_delta_svd",
        "task_arithmetic",
        "ties",
        "dare",
        "compress_then_merge",
        "canonical_svd_factor_average",
        "prediction_ensemble",
        "separate_adapters",
        "oracle_alignment",
    }
    dense_naive = stability[
        (stability.family == "dense") & (stability.method == "naive_factor_average")
    ].iloc[0]
    gates = {
        "twenty_scrambles_per_family": all(gauge_checks.groupby("family").size() >= 20),
        "well_conditioned_gauges_preserve_effective_updates": bool(
            gauge_checks[gauge_checks.family.isin(WELL_CONDITIONED_FAMILIES)].maximum_relative_delta_error.max()
            <= PRESERVATION_TOLERANCE
        ),
        "well_conditioned_gauges_preserve_predictions": bool(
            gauge_checks[gauge_checks.family.isin(WELL_CONDITIONED_FAMILIES)].maximum_prediction_disagreement.max()
            == 0.0
        ),
        "twistedmerge_methods_are_representation_stable": bool(
            primary_stability[primary_stability.method.isin(twisted_methods)].max_logit_distance_from_original.max()
            <= INVARIANCE_TOLERANCE
            and primary_stability[primary_stability.method.isin(twisted_methods)].method_failure_count.sum() == 0
        ),
        "effective_delta_baselines_are_representation_stable": bool(
            primary_stability[
                primary_stability.method.isin(invariant_baselines)
            ].max_logit_distance_from_original.max()
            <= INVARIANCE_TOLERANCE
        ),
        "naive_factor_average_is_representation_dependent": bool(
            dense_naive.max_logit_distance_from_original > 1e-3
            and dense_naive.max_relative_delta_distance_from_original > 1e-3
        ),
        "exact_transition_cycles_close": bool(
            primary_runs.max_cycle_frobenius_defect.max() <= INVARIANCE_TOLERANCE
            and primary_runs.max_cycle_spectral_defect.max() <= INVARIANCE_TOLERANCE
        ),
        "cycle_aware_method_abstains_on_injected_inconsistency": bool(
            boundary_result.decision == "fallback_full_delta_svd"
            and boundary["relative_delta_error_vs_full_delta_svd"] <= INVARIANCE_TOLERANCE
        ),
    }

    paths = {
        "per_run": output_dir / "per_run.csv",
        "gauge_checks": output_dir / "gauge_checks.csv",
        "cycle_diagnostics": output_dir / "cycle_diagnostics.csv",
        "scramble_stability": output_dir / "scramble_stability.csv",
        "paired_comparisons": output_dir / "paired_comparisons.csv",
        "capacity_cost": output_dir / "capacity_cost.csv",
        "cycle_boundary": output_dir / "cycle_boundary.csv",
        "claim_status": output_dir / "claim_status.csv",
    }
    runs.to_csv(paths["per_run"], index=False)
    gauge_checks.to_csv(paths["gauge_checks"], index=False)
    cycles.to_csv(paths["cycle_diagnostics"], index=False)
    stability.to_csv(paths["scramble_stability"], index=False)
    comparisons.to_csv(paths["paired_comparisons"], index=False)
    capacity.to_csv(paths["capacity_cost"], index=False)
    boundary_table.to_csv(paths["cycle_boundary"], index=False)
    claim_rows = [
        {
            "claim": "controlled_well_conditioned_twistedmerge_lora_invariance",
            "supported": all(
                gates[name]
                for name in (
                    "twenty_scrambles_per_family",
                    "well_conditioned_gauges_preserve_effective_updates",
                    "well_conditioned_gauges_preserve_predictions",
                    "twistedmerge_methods_are_representation_stable",
                    "exact_transition_cycles_close",
                )
            ),
            "scope": "one planted shared-B rank-3 adapter group; dependent scrambles",
        },
        {
            "claim": "naive_factor_merge_representation_dependence",
            "supported": gates["naive_factor_average_is_representation_dependent"],
            "scope": "controlled non-orthogonal gauges in one synthetic adapter group",
        },
        {
            "claim": "real_adapter_performance_gain",
            "supported": False,
            "scope": "not tested; prior real-adapter gate remains negative",
        },
        {
            "claim": "natural_brauer_or_period_index_obstruction",
            "supported": False,
            "scope": "forbidden extrapolation from controlled cycle diagnostics",
        },
    ]
    pd.DataFrame(claim_rows).to_csv(paths["claim_status"], index=False)

    plot_paths = write_plot(stability, output_dir)
    latex_path = write_latex_table(stability)
    report_path = write_report(
        output_dir,
        stability,
        gauge_checks,
        boundary,
        gates,
        scrambles,
        int(runs.method_failed.sum()),
        int(runs[runs.family.isin(WELL_CONDITIONED_FAMILIES)].method_failed.sum()),
    )
    execution_commit, dirty_state = git_metadata()
    config = {
        "experiment_id": "PTM-A2",
        "stage": "controlled_smoke",
        "command": f"{sys.executable} {Path(__file__).relative_to(ROOT)} --mode smoke --seed {seed} --scrambles {scrambles}",
        "execution_commit": execution_commit,
        "worktree_dirty_during_execution": dirty_state,
        "model_identifier": "synthetic_linear_base",
        "model_revision": "generated_by_seed",
        "dataset_identifier": "synthetic_gaussian_features_with_planted_task_teachers",
        "dataset_revision": "generated_by_seed",
        "external_license_dependency": False,
        "seed": seed,
        "adapter_provenance": "four planted shared-B rank-3 effective updates; no training",
        "adapter_count": len(fixture.factors),
        "adapter_rank": rank,
        "gauge_families": list(ALL_FAMILIES),
        "condition_number_targets": GAUGE_CONDITIONS,
        "scrambles_per_family": scrambles,
        "invariance_tolerance": INVARIANCE_TOLERANCE,
        "preservation_tolerance": PRESERVATION_TOLERANCE,
        "validation_evaluations_per_method": 0,
        "test_labels_used_for_selection": False,
        "independent_training_groups": 1,
        "statistical_boundary": "no bootstrap CI: scrambles are dependent representations of one fixed group",
        "device": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "pandas", "matplotlib", "pytest")
        },
        "fixture_hashes": {
            "base": sha256_bytes(fixture.base.tobytes()),
            "adapter_deltas": sha256_bytes(
                b"".join(effective_delta(factor).tobytes() for factor in fixture.factors)
            ),
            "validation_features": sha256_bytes(fixture.validation_x.tobytes()),
            "validation_labels": sha256_bytes(fixture.validation_labels.tobytes()),
            "test_features": sha256_bytes(fixture.test_x.tobytes()),
            "test_labels": sha256_bytes(fixture.test_labels.tobytes()),
        },
        "source_hashes": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in (
                Path(__file__),
                ROOT / "src" / "lora_gauge_alignment.py",
                ROOT / "src" / "lora_cycle_diagnostics.py",
                ROOT / "tests" / "test_lora_gauge_alignment.py",
                ROOT / "tests" / "test_lora_gauge_invariance.py",
            )
        },
        "gates": gates,
        "all_smoke_gates_passed": all(gates.values()),
        "failures": [] if all(gates.values()) else [name for name, passed in gates.items() if not passed],
        "recorded_boundary_alignment_failure_count": int(runs.method_failed.sum()),
        "recorded_primary_alignment_failure_count": int(
            runs[runs.family.isin(WELL_CONDITIONED_FAMILIES)].method_failed.sum()
        ),
        "recorded_boundary_alignment_failures": sorted(
            set(runs.loc[runs.method_failed, "failure_reason"].astype(str))
        ),
    }
    config_path = output_dir / "config.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")

    hash_targets = [*paths.values(), *plot_paths, latex_path, report_path, config_path]
    hash_path = output_dir / "artifact_hashes.csv"
    with hash_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["path", "sha256", "bytes"],
            lineterminator="\n",
        )
        writer.writeheader()
        for path in sorted(hash_targets):
            writer.writerow(
                {
                    "path": str(path.relative_to(ROOT)),
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                }
            )
    return {
        "run_rows": len(runs),
        "scrambles_per_family": scrambles,
        "all_smoke_gates_passed": all(gates.values()),
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "report": str(report_path.relative_to(ROOT)),
        "artifact_hashes": str(hash_path.relative_to(ROOT)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "pilot", "confirmatory"), default="smoke")
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--scrambles", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.mode != "smoke":
        raise RuntimeError(
            "pilot and confirmatory modes are gated: resolve dataset licenses and freeze an independent-training-group protocol"
        )
    result = run_smoke(args.seed, args.scrambles, args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
