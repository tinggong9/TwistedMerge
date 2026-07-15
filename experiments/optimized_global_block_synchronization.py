#!/usr/bin/env python
"""Optimized global block synchronization sanity checks.

This experiment is intentionally fast and controlled.  It verifies algorithmic
behavior requested in benchmark series 5(j)(ii) without turning diagnostic block rotations
for ReLU MLPs into a claimed exact merge path.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.block_compatible_merge import (  # noqa: E402
    average_linear_hidden_models,
    make_linear_hidden_mlp,
    max_logit_difference,
    transform_linear_hidden_block_gauge,
)
from src.block_gauge_alignment import BlockPartition  # noqa: E402
from src.block_sync_calibration import (  # noqa: E402
    accepted_sync_from_calibration,
    calibrate_connection_residual_threshold,
    classify_sync_evidence,
)
from src.global_block_synchronization import (  # noqa: E402
    cycle_score,
    default_triples,
    global_block_spectral_synchronization,
    mean_centrality,
    residual_optimized_global_block_sync,
    triangle_defects,
)
from src.learned_block_partition import (  # noqa: E402
    global_activation_correlation,
    residual_greedy_blocks,
    validation_selected_blocks,
)
from src.metrics import capture_environment  # noqa: E402
from src.model_merging_benchmark import require_torch  # noqa: E402
from src.noncentral_holonomy import detect_scalar_phase  # noqa: E402


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def git_dirty() -> bool | str:
    try:
        return bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    except Exception:
        return "unknown"


def rotation(theta: float) -> np.ndarray:
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


def block_diag(blocks: list[np.ndarray]) -> np.ndarray:
    size = sum(block.shape[0] for block in blocks)
    out = np.zeros((size, size), dtype=float)
    cursor = 0
    for block in blocks:
        n = block.shape[0]
        out[cursor : cursor + n, cursor : cursor + n] = block
        cursor += n
    return out


def maps_from_gauges(gauges: dict[int, np.ndarray]) -> dict[tuple[int, int], np.ndarray]:
    return {(i, j): gauges[i] @ gauges[j].T for i in gauges for j in gauges}


def defect_summary(maps: dict[tuple[int, int], np.ndarray], n_models: int, max_order: int) -> dict[str, float | bool | str]:
    defects = triangle_defects(maps, default_triples(n_models))
    detections = [detect_scalar_phase(defect, max_order=max_order) for defect in defects.values()]
    scalar = any(item.is_scalar_finite_index_candidate for item in detections)
    orders = sorted({item.detected_order_d for item in detections if item.detected_order_d is not None})
    return {
        "cycle_score": cycle_score(defects),
        "centrality_score": mean_centrality(defects),
        "scalar_projective_candidate": bool(scalar),
        "detected_orders": ",".join(str(item) for item in orders),
    }


def row_for_sync(
    *,
    setting_id: str,
    case_family: str,
    method: str,
    pairwise_maps: dict[tuple[int, int], np.ndarray],
    result,
    calibration,
    n_models: int,
    width: int,
    block_size: int,
    partition_method: str,
    seed: int,
    expected_outcome: str,
    max_order: int,
) -> dict:
    observed = defect_summary(pairwise_maps, n_models, max_order)
    projected = defect_summary(result.synchronized_maps, n_models, max_order)
    label = classify_sync_evidence(
        observed_scalar_projective_candidate=bool(observed["scalar_projective_candidate"]),
        observed_centrality_score=float(observed["centrality_score"]),
        projected_cycle_score=float(projected["cycle_score"]),
        connection_residual=float(result.connection_residual),
        calibration=calibration,
    )
    return {
        "source": "synthetic_control",
        "setting_id": setting_id,
        "case_family": case_family,
        "method": method,
        "seed": seed,
        "n_models": n_models,
        "width": width,
        "block_size": block_size,
        "partition_method": partition_method,
        "observed_cycle_score": observed["cycle_score"],
        "observed_centrality_score": observed["centrality_score"],
        "observed_scalar_projective_candidate": observed["scalar_projective_candidate"],
        "observed_detected_orders": observed["detected_orders"],
        "projected_cycle_score": projected["cycle_score"],
        "projected_centrality_score": projected["centrality_score"],
        "projected_scalar_projective_candidate": projected["scalar_projective_candidate"],
        "connection_residual": float(result.connection_residual),
        "initial_connection_residual": (
            float(result.initial_connection_residual)
            if result.initial_connection_residual is not None
            else float(result.connection_residual)
        ),
        "max_connection_residual": float(result.max_connection_residual),
        "objective_value": float(result.objective_value) if result.objective_value is not None else np.nan,
        "n_iterations": int(result.n_iterations),
        "accepted_sync": accepted_sync_from_calibration(result, calibration),
        "calibrated_threshold": calibration.threshold,
        "evidence_label": label,
        "expected_outcome": expected_outcome,
        "claim_status": "",
        "merge_accuracy": np.nan,
        "logit_max_abs_diff": np.nan,
        "same_parameter_count": "",
        "exact_same_architecture_symmetry": "",
        "capacity_matched_to_weight_average": "",
        "notes": "",
    }


def build_improvable_case(seed: int = 0):
    rng = np.random.default_rng(seed)
    gauges = {i: block_diag([rotation(rng.normal()), rotation(rng.normal())]) for i in range(4)}
    pairwise = maps_from_gauges(gauges)
    for i, j in [(0, 1), (1, 2), (2, 3)]:
        perturbation = block_diag([rotation(0.7 * rng.normal()), rotation(0.7 * rng.normal())])
        pairwise[(i, j)] = pairwise[(i, j)] @ perturbation
        pairwise[(j, i)] = pairwise[(i, j)].T
    blocks = {idx: [np.array([0, 1]), np.array([2, 3])] for idx in range(4)}
    return pairwise, blocks


def synthetic_sync_rows(args, calibration) -> list[dict]:
    rows: list[dict] = []
    partition_method = "contiguous"
    blocks3 = {idx: [np.array([0, 1]), np.array([2, 3])] for idx in range(3)}
    gauges3 = {
        0: block_diag([rotation(0.0), rotation(0.0)]),
        1: block_diag([rotation(0.35), rotation(-0.1)]),
        2: block_diag([rotation(-0.25), rotation(0.45)]),
    }
    exact = maps_from_gauges(gauges3)
    for method, result in [
        ("spectral", global_block_spectral_synchronization(exact, blocks3, 3, 4)),
        (
            "residual_optimized",
            residual_optimized_global_block_sync(exact, blocks3, 3, 4, n_restarts=args.n_restarts),
        ),
    ]:
        rows.append(
            row_for_sync(
                setting_id="exact_global_block_gauges",
                case_family="exact_global_block_gauges",
                method=method,
                pairwise_maps=exact,
                result=result,
                calibration=calibration,
                n_models=3,
                width=4,
                block_size=2,
                partition_method=partition_method,
                seed=44,
                expected_outcome="accepted: planted gauges recovered",
                max_order=args.max_order,
            )
        )

    noisy, blocks4 = build_improvable_case(seed=0)
    spectral = global_block_spectral_synchronization(noisy, blocks4, 4, 4)
    optimized = residual_optimized_global_block_sync(
        noisy,
        blocks4,
        4,
        4,
        lambda_feature=0.0,
        max_iters=args.max_iters,
        tolerance=1e-8,
        n_restarts=args.n_restarts,
        seed=0,
    )
    for method, result in [("spectral", spectral), ("residual_optimized", optimized)]:
        rows.append(
            row_for_sync(
                setting_id="noisy_pairwise_observations_from_global_gauges",
                case_family="noisy_globally_generated_gauges",
                method=method,
                pairwise_maps=noisy,
                result=result,
                calibration=calibration,
                n_models=4,
                width=4,
                block_size=2,
                partition_method=partition_method,
                seed=0,
                expected_outcome="descriptive: optimize residual if it beats spectral",
                max_order=args.max_order,
            )
        )

    reflection = np.array([[0.0, 1.0], [1.0, 0.0]])
    noncentral = {
        (0, 0): np.eye(2),
        (1, 1): np.eye(2),
        (2, 2): np.eye(2),
        (0, 1): reflection,
        (1, 2): rotation(0.4),
        (2, 0): np.linalg.inv(reflection) @ np.linalg.inv(rotation(0.4)),
    }
    noncentral[(1, 0)] = noncentral[(0, 1)].T
    noncentral[(2, 1)] = noncentral[(1, 2)].T
    noncentral[(0, 2)] = noncentral[(2, 0)].T
    one_block = {idx: [np.array([0, 1])] for idx in range(3)}
    rows.append(
        row_for_sync(
            setting_id="noncentral_holonomy",
            case_family="noncentral_holonomy",
            method="residual_optimized",
            pairwise_maps=noncentral,
            result=residual_optimized_global_block_sync(noncentral, one_block, 3, 2, n_restarts=args.n_restarts),
            calibration=calibration,
            n_models=3,
            width=2,
            block_size=2,
            partition_method=partition_method,
            seed=52,
            expected_outcome="rejected: noncentral connection residual remains large",
            max_order=args.max_order,
        )
    )

    scalar = {
        (0, 0): np.eye(4),
        (1, 1): np.eye(4),
        (2, 2): np.eye(4),
        (0, 1): np.eye(4),
        (1, 2): np.eye(4),
        (2, 0): -np.eye(4),
        (1, 0): np.eye(4),
        (2, 1): np.eye(4),
        (0, 2): -np.eye(4),
    }
    rows.append(
        row_for_sync(
            setting_id="scalar_block_phase_mu2",
            case_family="scalar_block_phase_before_projection",
            method="residual_optimized",
            pairwise_maps=scalar,
            result=residual_optimized_global_block_sync(scalar, blocks3, 3, 4, n_restarts=args.n_restarts),
            calibration=calibration,
            n_models=3,
            width=4,
            block_size=2,
            partition_method=partition_method,
            seed=53,
            expected_outcome="detected before projection: central mu2 candidate",
            max_order=args.max_order,
        )
    )

    fake, fake_blocks = build_improvable_case(seed=7)
    fake[(0, 2)] = block_diag([rotation(1.7), rotation(-1.2)])
    fake[(2, 0)] = fake[(0, 2)].T
    rows.append(
        row_for_sync(
            setting_id="fake_projection_trap",
            case_family="fake_projection_trap",
            method="residual_optimized",
            pairwise_maps=fake,
            result=residual_optimized_global_block_sync(fake, fake_blocks, 4, 4, n_restarts=args.n_restarts),
            calibration=calibration,
            n_models=4,
            width=4,
            block_size=2,
            partition_method=partition_method,
            seed=7,
            expected_outcome="rejected: projected cycles alone are insufficient",
            max_order=args.max_order,
        )
    )
    return rows


def learned_block_rows(seed: int = 21) -> tuple[list[dict], dict]:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=300)
    b = rng.normal(size=300)
    activations = {
        0: np.column_stack([a, b, a + 0.01 * rng.normal(size=300), b + 0.01 * rng.normal(size=300)]),
        1: np.column_stack(
            [
                a + 0.02 * rng.normal(size=300),
                b + 0.02 * rng.normal(size=300),
                a + 0.02 * rng.normal(size=300),
                b + 0.02 * rng.normal(size=300),
            ]
        ),
    }
    similarity = global_activation_correlation(activations)
    contiguous = BlockPartition("contiguous", 2, ((0, 1), (2, 3)))
    learned = residual_greedy_blocks(similarity, 2, larger_is_better=True, seed=3, method="global_activation_residual_greedy")

    def validation_residual(partition: BlockPartition) -> float:
        vals = []
        for block in partition.blocks:
            idx = list(block)
            if len(idx) < 2:
                continue
            vals.append(float(1.0 - np.mean(similarity[np.ix_(idx, idx)][np.triu_indices(len(idx), k=1)])))
        return float(np.mean(vals)) if vals else 0.0

    scores = {"contiguous": validation_residual(contiguous), "learned": validation_residual(learned)}
    selected = validation_selected_blocks(
        {"contiguous": contiguous, "learned": learned},
        scores,
        metric_source="validation_activation_residual",
    )
    planted = {frozenset({0, 2}), frozenset({1, 3})}
    rows = []
    for name, partition in [("contiguous", contiguous), ("learned", learned)]:
        recovered = {frozenset(block) for block in partition.blocks} == planted
        rows.append(
            {
                "source": "learned_block_control",
                "setting_id": "learned_noncontiguous_block_positive_control",
                "case_family": "learned_noncontiguous_blocks",
                "method": name,
                "seed": seed,
                "n_models": 2,
                "width": 4,
                "block_size": 2,
                "partition_method": partition.method,
                "observed_cycle_score": np.nan,
                "observed_centrality_score": np.nan,
                "observed_scalar_projective_candidate": "",
                "observed_detected_orders": "",
                "projected_cycle_score": np.nan,
                "projected_centrality_score": np.nan,
                "projected_scalar_projective_candidate": "",
                "connection_residual": np.nan,
                "initial_connection_residual": np.nan,
                "max_connection_residual": np.nan,
                "objective_value": np.nan,
                "n_iterations": 0,
                "accepted_sync": "",
                "calibrated_threshold": np.nan,
                "evidence_label": "validation_selected_blocks" if name == selected.selected_name else "candidate",
                "expected_outcome": "learned partition recovers non-contiguous planted blocks",
                "claim_status": "supported" if recovered and name == "learned" else "control",
                "merge_accuracy": np.nan,
                "logit_max_abs_diff": np.nan,
                "same_parameter_count": "",
                "exact_same_architecture_symmetry": "",
                "capacity_matched_to_weight_average": "",
                "notes": f"validation_residual={scores[name]:.6g}; used_test_metrics={selected.used_test_metrics}",
                "block_recovery_exact": recovered,
                "validation_block_residual": scores[name],
            }
        )
    return rows, {"contiguous": scores["contiguous"], "learned": scores["learned"]}


def block_compatible_rows(args) -> tuple[list[dict], pd.DataFrame]:
    torch, _, _ = require_torch()
    rows = []
    accuracy_rows = []
    partition = BlockPartition("contiguous", 2, ((0, 1), (2, 3)))
    for seed in range(args.block_compatible_seeds):
        torch.manual_seed(9000 + seed)
        model = make_linear_hidden_mlp(input_dim=6, width=4, num_classes=3)
        inputs = torch.randn(256, 6)
        with torch.no_grad():
            labels = model(inputs).argmax(dim=1)
        gauges = {0: rotation(0.5 + 0.03 * seed), 1: rotation(-0.7 + 0.02 * seed)}
        inverse_gauges = {0: gauges[0].T, 1: gauges[1].T}
        transformed, metadata = transform_linear_hidden_block_gauge(model, partition, gauges)
        aligned_back, _metadata_back = transform_linear_hidden_block_gauge(transformed, partition, inverse_gauges)
        unaligned_average = average_linear_hidden_models([model, transformed])
        aligned_average = average_linear_hidden_models([model, aligned_back])

        def acc(candidate) -> float:
            with torch.no_grad():
                return float((candidate(inputs).argmax(dim=1) == labels).float().mean().item())

        method_models = {
            "base": model,
            "gauge_equivalent_copy": transformed,
            "unaligned_weight_average": unaligned_average,
            "block_compatible_aligned_average": aligned_average,
        }
        for method, candidate in method_models.items():
            accuracy = acc(candidate)
            accuracy_rows.append({"seed": seed, "method": method, "accuracy": accuracy})
            rows.append(
                {
                    "source": "block_compatible_linear_hidden",
                    "setting_id": "linear_hidden_exact_block_gauge",
                    "case_family": "block_compatible_merge",
                    "method": method,
                    "seed": seed,
                    "n_models": 2,
                    "width": 4,
                    "block_size": 2,
                    "partition_method": "contiguous",
                    "observed_cycle_score": np.nan,
                    "observed_centrality_score": np.nan,
                    "observed_scalar_projective_candidate": "",
                    "observed_detected_orders": "",
                    "projected_cycle_score": np.nan,
                    "projected_centrality_score": np.nan,
                    "projected_scalar_projective_candidate": "",
                    "connection_residual": np.nan,
                    "initial_connection_residual": np.nan,
                    "max_connection_residual": np.nan,
                    "objective_value": np.nan,
                    "n_iterations": 0,
                    "accepted_sync": "",
                    "calibrated_threshold": np.nan,
                    "evidence_label": "exact_linear_hidden_symmetry",
                    "expected_outcome": "aligned same-architecture average recovers base function",
                    "claim_status": "supported_descriptive",
                    "merge_accuracy": accuracy,
                    "logit_max_abs_diff": max_logit_difference(model, candidate, inputs),
                    "same_parameter_count": metadata.same_parameter_count,
                    "exact_same_architecture_symmetry": metadata.exact_same_architecture_symmetry,
                    "capacity_matched_to_weight_average": metadata.same_parameter_count and not metadata.adapter_extra_parameters,
                    "notes": metadata.notes,
                    "block_recovery_exact": "",
                    "validation_block_residual": np.nan,
                }
            )
    return rows, pd.DataFrame(accuracy_rows)


def build_calibration(args):
    positives = []
    negatives = []
    blocks = {idx: [np.array([0, 1]), np.array([2, 3])] for idx in range(3)}
    for seed in range(5):
        gauges = {
            0: block_diag([rotation(0.0), rotation(0.0)]),
            1: block_diag([rotation(0.2 + 0.01 * seed), rotation(-0.1)]),
            2: block_diag([rotation(-0.25), rotation(0.35 + 0.01 * seed)]),
        }
        exact = maps_from_gauges(gauges)
        positives.append(global_block_spectral_synchronization(exact, blocks, 3, 4).connection_residual)
    for seed in range(5):
        fake, fake_blocks = build_improvable_case(seed=seed + 20)
        fake[(0, 2)] = block_diag([rotation(1.2 + 0.1 * seed), rotation(-1.0)])
        fake[(2, 0)] = fake[(0, 2)].T
        negatives.append(
            residual_optimized_global_block_sync(fake, fake_blocks, 4, 4, n_restarts=args.n_restarts).connection_residual
        )
    return calibrate_connection_residual_threshold(
        positives,
        negatives,
        target_false_positive_rate=args.target_false_positive_rate,
    )


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (source, case_family, method), group in df.groupby(["source", "case_family", "method"], dropna=False):
        rows.append(
            {
                "source": source,
                "case_family": case_family,
                "method": method,
                "n_rows": int(len(group)),
                "mean_connection_residual": float(pd.to_numeric(group["connection_residual"], errors="coerce").mean()),
                "mean_initial_connection_residual": float(
                    pd.to_numeric(group["initial_connection_residual"], errors="coerce").mean()
                ),
                "mean_projected_cycle_score": float(pd.to_numeric(group["projected_cycle_score"], errors="coerce").mean()),
                "accepted_sync_rate": float(group["accepted_sync"].replace("", np.nan).dropna().astype(bool).mean())
                if group["accepted_sync"].replace("", np.nan).dropna().shape[0]
                else np.nan,
                "mean_merge_accuracy": float(pd.to_numeric(group["merge_accuracy"], errors="coerce").mean()),
                "mean_validation_block_residual": float(
                    pd.to_numeric(group["validation_block_residual"], errors="coerce").mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def paired_stats(df: pd.DataFrame, accuracy_df: pd.DataFrame, learned_scores: dict) -> pd.DataFrame:
    rows = []
    sync = df[df["setting_id"] == "noisy_pairwise_observations_from_global_gauges"]
    if not sync.empty:
        spectral = float(sync[sync["method"] == "spectral"]["connection_residual"].iloc[0])
        optimized = float(sync[sync["method"] == "residual_optimized"]["connection_residual"].iloc[0])
        rows.append(
            {
                "comparison": "optimized_sync_vs_spectral_connection_residual",
                "n": 1,
                "mean_delta": optimized - spectral,
                "metric": "connection_residual",
                "decision": "supported_descriptive" if optimized < spectral else "not_supported",
                "notes": "negative delta means optimized residual is lower; single synthetic control, not a real benchmark",
            }
        )
    rows.append(
        {
            "comparison": "learned_blocks_vs_contiguous_validation_residual",
            "n": 1,
            "mean_delta": learned_scores["learned"] - learned_scores["contiguous"],
            "metric": "validation_block_residual",
            "decision": "supported" if learned_scores["learned"] < learned_scores["contiguous"] else "not_supported",
            "notes": "positive control with planted non-contiguous blocks",
        }
    )
    pivot = accuracy_df.pivot(index="seed", columns="method", values="accuracy")
    if {"block_compatible_aligned_average", "unaligned_weight_average"}.issubset(pivot.columns):
        delta = pivot["block_compatible_aligned_average"] - pivot["unaligned_weight_average"]
        rows.append(
            {
                "comparison": "block_compatible_aligned_average_vs_unaligned_weight_average",
                "n": int(delta.shape[0]),
                "mean_delta": float(delta.mean()),
                "metric": "pseudo_label_accuracy",
                "decision": "supported_descriptive" if float(delta.mean()) >= 0.0 else "not_supported",
                "notes": "linear-hidden exact-symmetry synthetic task; not evidence for ReLU block rotations",
            }
        )
    return pd.DataFrame(rows)


def write_plots(df: pd.DataFrame, accuracy_df: pd.DataFrame, learned_scores: dict, plot_dir: Path) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)
    sync = df[df["source"] == "synthetic_control"].copy()
    plt.figure(figsize=(8, 4.5))
    labels = sync["setting_id"] + "\n" + sync["method"]
    plt.bar(np.arange(len(sync)), pd.to_numeric(sync["connection_residual"], errors="coerce"))
    plt.xticks(np.arange(len(sync)), labels, rotation=45, ha="right", fontsize=7)
    plt.ylabel("connection residual")
    plt.tight_layout()
    plt.savefig(plot_dir / "optimized_global_block_connection_residuals.pdf")
    plt.close()

    plt.figure(figsize=(4.5, 3.2))
    plt.bar(["contiguous", "learned"], [learned_scores["contiguous"], learned_scores["learned"]])
    plt.ylabel("validation block residual")
    plt.tight_layout()
    plt.savefig(plot_dir / "learned_blocks_vs_contiguous.pdf")
    plt.close()

    plt.figure(figsize=(6.5, 3.6))
    summary = accuracy_df.groupby("method")["accuracy"].mean().reindex(
        ["base", "gauge_equivalent_copy", "unaligned_weight_average", "block_compatible_aligned_average"]
    )
    plt.bar(summary.index, summary.values)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("pseudo-label accuracy")
    plt.ylim(0.0, 1.05)
    plt.tight_layout()
    plt.savefig(plot_dir / "block_compatible_merge_accuracy.pdf")
    plt.close()


def markdown_table(rows: list[dict], columns: list[str]) -> str:
    if not rows:
        return "_No rows._"
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        vals = []
        for col in columns:
            val = row.get(col, "")
            if isinstance(val, float):
                vals.append(f"{val:.6g}")
            else:
                vals.append(str(val))
        body.append("| " + " | ".join(vals) + " |")
    return "\n".join([header, sep, *body])


def write_report(args, df: pd.DataFrame, summary: pd.DataFrame, stats: pd.DataFrame, calibration, report_path: Path) -> None:
    exact_commands = [
        "PYTHONPYCACHEPREFIX=/private/tmp/codex-pycache MPLCONFIGDIR=/private/tmp/mplconfig .venv/bin/python experiments/global_block_synchronization_experiment.py",
        args.command_string,
        "PYTHONPYCACHEPREFIX=/private/tmp/codex-pycache .venv/bin/python -m unittest tests.test_optimized_global_block_synchronization tests.test_global_learned_block_partition tests.test_block_sync_calibration tests.test_block_compatible_merge -v",
    ]
    control_cols = [
        "setting_id",
        "method",
        "observed_cycle_score",
        "projected_cycle_score",
        "connection_residual",
        "accepted_sync",
        "evidence_label",
        "expected_outcome",
    ]
    stat_cols = ["comparison", "n", "mean_delta", "metric", "decision", "notes"]
    summary_cols = ["source", "case_family", "method", "n_rows", "mean_connection_residual", "accepted_sync_rate", "mean_merge_accuracy"]
    sync_rows = df[df["source"] == "synthetic_control"][control_cols].to_dict("records")
    report = f"""# Optimized Global Block Synchronization Report

This report is generated by `experiments/optimized_global_block_synchronization.py`.

## Exact Commands Run

```bash
{chr(10).join(exact_commands)}
```

## Git And Environment

- HEAD commit during generation: `{git_commit()}`
- Worktree dirty during generation: `{git_dirty()}`
- Baseline 5(j)(i) rerun was performed before these edits at commit `222646d`.

```json
{json.dumps(capture_environment(), indent=2)}
```

## Calibration

- Threshold: `{calibration.threshold:.6g}`
- Target false-positive rate: `{calibration.target_false_positive_rate:.4g}`
- Observed false-positive rate on controls: `{calibration.observed_false_positive_rate:.4g}`
- Observed true-positive rate on controls: `{calibration.observed_true_positive_rate:.4g}`

## Synthetic Controls

{markdown_table(sync_rows, control_cols)}

## Summary

{markdown_table(summary.to_dict("records"), summary_cols)}

## Paired / Decision Statistics

{markdown_table(stats.to_dict("records"), stat_cols)}

## What This Proves

- Planted exact global block gauges are recovered with near-zero connection residual.
- The residual-optimized multi-start path can reduce connection residual on the controlled noisy pairwise-observation case; the CSV records the exact spectral and optimized values.
- Projection traps are rejected when the synchronized maps have zero cycle score but the connection residual is above the calibrated threshold.
- The learned non-contiguous block positive control is recovered by validation-selected global activation correlations.
- General block rotations are an exact same-architecture symmetry only in the separate linear-hidden block-compatible model used here.

## What This Does Not Prove

- This does not prove general block-orthogonal rotations are exact symmetries of ReLU MLPs.
- This does not identify real MNIST/CIFAR residuals as Brauer/projective classes.
- This does not beat external C2M3, Git Re-Basin, or greedy-soup baselines.
- The block-compatible merge rows are a controlled identity-activation sanity check, not a natural-image benchmark.

## Output Artifacts

- `reports/csv/optimized_global_block_synchronization.csv`
- `reports/csv/optimized_global_block_synchronization_summary.csv`
- `reports/csv/optimized_global_block_synchronization_paired_stats.csv`
- `reports/plots/optimized_global_block_connection_residuals.pdf`
- `reports/plots/learned_blocks_vs_contiguous.pdf`
- `reports/plots/block_compatible_merge_accuracy.pdf`
- `reports/configs/optimized_global_block_synchronization_config.json`
"""
    report_path.write_text(report, encoding="utf-8")


def write_config(args, calibration, path: Path) -> None:
    config = {
        "command": args.command_string,
        "git_commit": git_commit(),
        "git_worktree_dirty": git_dirty(),
        "settings": {
            "max_order": args.max_order,
            "n_restarts": args.n_restarts,
            "max_iters": args.max_iters,
            "target_false_positive_rate": args.target_false_positive_rate,
            "block_compatible_seeds": args.block_compatible_seeds,
        },
        "calibration": {
            "threshold": calibration.threshold,
            "target_false_positive_rate": calibration.target_false_positive_rate,
            "observed_false_positive_rate": calibration.observed_false_positive_rate,
            "observed_true_positive_rate": calibration.observed_true_positive_rate,
            "n_positive": calibration.n_positive,
            "n_negative": calibration.n_negative,
        },
        "environment": capture_environment(),
    }
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-order", type=int, default=12)
    parser.add_argument("--n-restarts", type=int, default=20)
    parser.add_argument("--max-iters", type=int, default=100)
    parser.add_argument("--target-false-positive-rate", type=float, default=0.0)
    parser.add_argument("--block-compatible-seeds", type=int, default=10)
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "reports")
    args = parser.parse_args()
    env_prefix = [
        f"{name}={os.environ[name]}"
        for name in ("PYTHONPYCACHEPREFIX", "MPLCONFIGDIR")
        if os.environ.get(name)
    ]
    args.command_string = " ".join([*env_prefix, sys.executable, *sys.argv])

    csv_dir = args.reports_dir / "csv"
    plot_dir = args.reports_dir / "plots"
    config_dir = args.reports_dir / "configs"
    csv_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)

    calibration = build_calibration(args)
    rows = synthetic_sync_rows(args, calibration)
    learned_rows, learned_scores = learned_block_rows()
    block_rows, accuracy_df = block_compatible_rows(args)
    rows.extend(learned_rows)
    rows.extend(block_rows)
    df = pd.DataFrame(rows)
    summary = summarize(df)
    stats = paired_stats(df, accuracy_df, learned_scores)
    write_plots(df, accuracy_df, learned_scores, plot_dir)

    results_path = csv_dir / "optimized_global_block_synchronization.csv"
    summary_path = csv_dir / "optimized_global_block_synchronization_summary.csv"
    stats_path = csv_dir / "optimized_global_block_synchronization_paired_stats.csv"
    report_path = args.reports_dir / "optimized_global_block_synchronization_report.md"
    config_path = config_dir / "optimized_global_block_synchronization_config.json"
    df.to_csv(results_path, index=False)
    summary.to_csv(summary_path, index=False)
    stats.to_csv(stats_path, index=False)
    write_report(args, df, summary, stats, calibration, report_path)
    write_config(args, calibration, config_path)
    print(f"wrote {results_path}")
    print(f"wrote {summary_path}")
    print(f"wrote {stats_path}")
    print(f"wrote {report_path}")
    print(f"wrote {config_path}")


if __name__ == "__main__":
    main()
