#!/usr/bin/env python
"""Global and learned-block synchronization diagnostics."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.block_gauge_alignment import (  # noqa: E402
    BlockPartition,
    estimate_block_orthogonal_alignments_for_model_blocks,
    induce_model_blocks,
    summarize_block_alignment_stats,
)
from src.global_block_synchronization import (  # noqa: E402
    cycle_score,
    default_triples,
    feature_alignment_residual_for_maps,
    global_block_spectral_synchronization,
    global_sync_accepted,
    mean_centrality,
    triangle_defects,
)
from src.ladder_merge_methods import estimate_signs_and_positive_scales, transform_mlp_positive_scale  # noqa: E402
from src.learned_block_partition import make_block_partition  # noqa: E402
from src.metrics import capture_environment  # noqa: E402
from src.model_merging_benchmark import (  # noqa: E402
    average_models,
    collect_features,
    device_from_arg,
    evaluate_ensemble,
    evaluate_model,
    format_markdown_table,
    greedy_soup,
    load_dataset,
    make_loader,
    make_model,
    permute_model_to_reference,
    require_torch,
    set_seed,
    synchronize_permutations,
    train_model,
)
from src.noncentral_holonomy import detect_scalar_phase  # noqa: E402
from src.structure_group_ladder import (  # noqa: E402
    estimate_gl_alignments_from_activations,
    estimate_monomial_alignments_from_activations,
    estimate_pairwise_permutations_from_activations,
)


def parse_csv(text: str, cast=str):
    return [cast(item.strip()) for item in text.split(",") if item.strip()]


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def git_worktree_dirty() -> bool | str:
    try:
        status = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
        return bool(status.strip())
    except Exception:
        return "unknown"


def split_train_val(dataset, val_fraction: float, seed: int):
    torch, _, _ = require_torch()
    n_val = max(1, int(len(dataset) * val_fraction))
    n_train = len(dataset) - n_val
    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.utils.data.random_split(dataset, [n_train, n_val], generator=generator)


def rotation(theta: float) -> np.ndarray:
    return np.array(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ]
    )


def block_diag(blocks: list[np.ndarray]) -> np.ndarray:
    size = sum(block.shape[0] for block in blocks)
    out = np.zeros((size, size), dtype=float)
    cursor = 0
    for block in blocks:
        n = block.shape[0]
        out[cursor : cursor + n, cursor : cursor + n] = block
        cursor += n
    return out


def cycle_value(matrix: np.ndarray) -> float:
    eye = np.eye(matrix.shape[0], dtype=complex)
    return float(np.linalg.norm(matrix - eye, ord="fro") / max(float(np.linalg.norm(eye, ord="fro")), 1e-12))


def diagnose_defect(matrix: np.ndarray, max_order: int) -> dict[str, object]:
    detection = detect_scalar_phase(matrix, max_order=max_order)
    candidate = bool(detection.is_scalar_finite_index_candidate)
    if candidate:
        residual_type = "central_projective_candidate"
    elif cycle_value(matrix) <= 1e-8:
        residual_type = "gauge_trivial"
    else:
        residual_type = "noncentral_holonomy"
    return {
        "cycle_score": cycle_value(matrix),
        "centrality_score": detection.centrality_score,
        "phase_residual": detection.phase_residual,
        "detected_order_d": detection.detected_order_d,
        "scalar_projective_candidate": candidate,
        "residual_type": residual_type,
    }


def diag_rows_for_maps(
    *,
    rows: list[dict],
    source: str,
    setting_id: str,
    diagnostic_level: str,
    maps: dict[tuple[int, int], np.ndarray],
    n_models: int,
    width: int,
    seed: int,
    block_size: int,
    partition: BlockPartition | None,
    model_blocks,
    activations,
    max_order: int,
    pairwise_feature_residual: float | None,
    global_feature_residual: float | None,
    global_sync_residual: float | None,
    max_global_sync_residual: float | None,
    accepted_global_sync: bool | None,
    merge_metrics: dict[str, float | bool | str],
    expected_outcome: str = "",
    permutation_centrality: float | None = None,
    pairwise_block_centrality: float | None = None,
    global_block_centrality: float | None = None,
) -> None:
    defects = triangle_defects(maps, default_triples(n_models))
    for triangle, defect in defects.items():
        diag = diagnose_defect(defect, max_order)
        pairwise_improvement = (
            permutation_centrality - pairwise_block_centrality
            if permutation_centrality is not None and pairwise_block_centrality is not None
            else float("nan")
        )
        global_improvement = (
            permutation_centrality - global_block_centrality
            if permutation_centrality is not None and global_block_centrality is not None
            else float("nan")
        )
        block_to_global = (
            pairwise_block_centrality - global_block_centrality
            if pairwise_block_centrality is not None and global_block_centrality is not None
            else float("nan")
        )
        rows.append(
            {
                "source": source,
                "setting_id": setting_id,
                "diagnostic_level": diagnostic_level,
                "n_models": n_models,
                "width": width,
                "seed": seed,
                "block_size": block_size,
                "partition_method": partition.method if partition is not None else "none",
                "block_assignment": partition.assignment_string() if partition is not None else "",
                "triangle": "-".join(str(item) for item in triangle),
                **diag,
                "pairwise_feature_alignment_residual": pairwise_feature_residual if pairwise_feature_residual is not None else float("nan"),
                "global_feature_alignment_residual": global_feature_residual if global_feature_residual is not None else float("nan"),
                "global_sync_residual": global_sync_residual if global_sync_residual is not None else float("nan"),
                "max_global_sync_residual": max_global_sync_residual if max_global_sync_residual is not None else float("nan"),
                "accepted_global_sync": accepted_global_sync if accepted_global_sync is not None else False,
                "permutation_centrality": permutation_centrality if permutation_centrality is not None else float("nan"),
                "pairwise_block_centrality": pairwise_block_centrality if pairwise_block_centrality is not None else float("nan"),
                "global_block_centrality": global_block_centrality if global_block_centrality is not None else float("nan"),
                "improvement_permutation_to_pairwise_block": pairwise_improvement,
                "improvement_permutation_to_global_block": global_improvement,
                "improvement_pairwise_block_to_global_block": block_to_global,
                "remains_noncentral": not bool(diag["scalar_projective_candidate"]),
                "expected_outcome": expected_outcome,
                "merge_evaluated": False,
                "block_merge_notes": "not evaluated: block-orthogonal rotations are feature-space diagnostics for ReLU MLPs",
                **merge_metrics,
            }
        )


def controlled_rows(max_order: int) -> list[dict]:
    rows: list[dict] = []
    blocks = {idx: [np.array([0, 1]), np.array([2, 3])] for idx in range(3)}
    partition = BlockPartition("contiguous", 2, ((0, 1), (2, 3)))
    rng = np.random.default_rng(44)
    base = rng.normal(size=(400, 4))
    gauges = {
        0: np.eye(4),
        1: block_diag([rotation(0.35), rotation(-0.1)]),
        2: block_diag([rotation(-0.25), rotation(0.45)]),
    }
    activations = {idx: base @ gauge for idx, gauge in gauges.items()}
    true_maps = {(i, j): gauges[i] @ gauges[j].T for i in range(3) for j in range(3)}
    sync = global_block_spectral_synchronization(true_maps, blocks, 3, 4, activations=activations)
    diag_rows_for_maps(
        rows=rows,
        source="synthetic",
        setting_id="planted_recoverable_block_rotations",
        diagnostic_level="global_block_synchronization",
        maps=sync.synchronized_maps,
        n_models=3,
        width=4,
        seed=-1,
        block_size=2,
        partition=partition,
        model_blocks=blocks,
        activations=activations,
        max_order=max_order,
        pairwise_feature_residual=feature_alignment_residual_for_maps(true_maps, blocks, activations, 3),
        global_feature_residual=sync.feature_alignment_residual,
        global_sync_residual=sync.connection_residual,
        max_global_sync_residual=sync.max_connection_residual,
        accepted_global_sync=global_sync_accepted(sync, 1e-8),
        merge_metrics={},
        expected_outcome="global gauges recovered",
    )

    noisy = dict(true_maps)
    noisy[(0, 1)] = noisy[(0, 1)] @ block_diag([rotation(0.18), rotation(-0.12)])
    noisy[(1, 0)] = noisy[(0, 1)].T
    sync_noisy = global_block_spectral_synchronization(noisy, blocks, 3, 4, activations=activations)
    diag_rows_for_maps(
        rows=rows,
        source="synthetic",
        setting_id="planted_pairwise_noisy_global_projection",
        diagnostic_level="global_block_synchronization",
        maps=sync_noisy.synchronized_maps,
        n_models=3,
        width=4,
        seed=-1,
        block_size=2,
        partition=partition,
        model_blocks=blocks,
        activations=activations,
        max_order=max_order,
        pairwise_feature_residual=feature_alignment_residual_for_maps(noisy, blocks, activations, 3),
        global_feature_residual=sync_noisy.feature_alignment_residual,
        global_sync_residual=sync_noisy.connection_residual,
        max_global_sync_residual=sync_noisy.max_connection_residual,
        accepted_global_sync=global_sync_accepted(sync_noisy, 0.15),
        merge_metrics={},
        expected_outcome="cycle-consistent projection with nonzero connection residual",
    )

    noncentral = {
        (0, 0): np.eye(2),
        (1, 1): np.eye(2),
        (2, 2): np.eye(2),
        (0, 1): np.array([[0.0, 1.0], [1.0, 0.0]]),
        (1, 2): rotation(0.4),
        (2, 0): np.linalg.inv(np.array([[0.0, 1.0], [1.0, 0.0]])) @ np.linalg.inv(rotation(0.4)),
    }
    noncentral[(1, 0)] = noncentral[(0, 1)].T
    noncentral[(2, 1)] = noncentral[(1, 2)].T
    noncentral[(0, 2)] = noncentral[(2, 0)].T
    one_block = {idx: [np.array([0, 1])] for idx in range(3)}
    one_partition = BlockPartition("contiguous", 2, ((0, 1),))
    sync_noncentral = global_block_spectral_synchronization(noncentral, one_block, 3, 2)
    diag_rows_for_maps(
        rows=rows,
        source="synthetic",
        setting_id="planted_noncentral_block_holonomy",
        diagnostic_level="pairwise_block_procrustes",
        maps=noncentral,
        n_models=3,
        width=2,
        seed=-1,
        block_size=2,
        partition=one_partition,
        model_blocks=one_block,
        activations=None,
        max_order=max_order,
        pairwise_feature_residual=None,
        global_feature_residual=None,
        global_sync_residual=sync_noncentral.connection_residual,
        max_global_sync_residual=sync_noncentral.max_connection_residual,
        accepted_global_sync=global_sync_accepted(sync_noncentral, 0.15),
        merge_metrics={},
        expected_outcome="noncentral holonomy rejected",
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
    sync_scalar = global_block_spectral_synchronization(scalar, blocks, 3, 4)
    diag_rows_for_maps(
        rows=rows,
        source="synthetic",
        setting_id="planted_scalar_block_phase_mu2",
        diagnostic_level="pairwise_block_procrustes",
        maps=scalar,
        n_models=3,
        width=4,
        seed=-1,
        block_size=2,
        partition=partition,
        model_blocks=blocks,
        activations=None,
        max_order=max_order,
        pairwise_feature_residual=None,
        global_feature_residual=None,
        global_sync_residual=sync_scalar.connection_residual,
        max_global_sync_residual=sync_scalar.max_connection_residual,
        accepted_global_sync=global_sync_accepted(sync_scalar, 0.15),
        merge_metrics={},
        expected_outcome="central mu2 phase detected before projection",
    )
    return rows


def evaluate_relu_compatible_merges(args, spec, models, synced, features, ref, val_subset, test_data, device, width):
    val_loader = make_loader(val_subset, args.batch_size, shuffle=False, seed=args.dataset_seed + 700)
    test_loader = make_loader(test_data, args.batch_size, shuffle=False, seed=args.dataset_seed + 999)
    aligned_c2m3 = [permute_model_to_reference(model, "mlp", spec, width, synced[idx]) for idx, model in enumerate(models)]
    c2m3_model = average_models(aligned_c2m3, "mlp", spec, width)

    scaled_models = []
    for idx, model in enumerate(models):
        perm = synced[idx]
        if idx == ref:
            scales = np.ones(width, dtype=float)
        else:
            _signs, scales = estimate_signs_and_positive_scales(features[ref], features[idx], perm)
        scaled_models.append(transform_mlp_positive_scale(model, spec, width, perm, scales))
    monomial_model = average_models(scaled_models, "mlp", spec, width)
    soup, _indices, soup_metrics = greedy_soup(models, val_loader, test_loader, device, "mlp", spec, width)
    _ = soup
    return {
        "c2m3_accuracy": evaluate_model(c2m3_model, test_loader, device)["accuracy"],
        "monomial_accuracy": evaluate_model(monomial_model, test_loader, device)["accuracy"],
        "greedy_soup_accuracy": soup_metrics["accuracy"],
        "ensemble_accuracy": evaluate_ensemble(models, test_loader, device)["accuracy"],
    }


def run_real_setting(args, spec, train_data, test_data, seed: int, n_models: int, width: int) -> list[dict]:
    device = device_from_arg(args.device)
    train_subset, val_subset = split_train_val(train_data, args.val_fraction, seed + 77)
    models = []
    for model_idx in range(n_models):
        model_seed = seed + 1000 * model_idx + 17 * width + n_models
        set_seed(model_seed)
        model = make_model("mlp", spec, width)
        train_loader = make_loader(train_subset, args.batch_size, shuffle=True, seed=model_seed + 11)
        train_model(model, train_loader, args.epochs, args.lr, device)
        model.to("cpu")
        models.append(model)
    match_loader = make_loader(train_subset, args.batch_size, shuffle=False, seed=seed + 501)
    activations = {idx: collect_features(model, match_loader, device) for idx, model in enumerate(models)}
    pairwise_perms = estimate_pairwise_permutations_from_activations(activations, n_models, width)
    ref, synced, _sync_disagreement = synchronize_permutations(pairwise_perms, n_models)
    merge_metrics = evaluate_relu_compatible_merges(args, spec, models, synced, activations, ref, val_subset, test_data, device, width)
    permutation_maps = {
        pair: np.eye(width)[np.asarray(perm, dtype=int)]
        for pair, perm in pairwise_perms.items()
    }
    monomial_maps = estimate_monomial_alignments_from_activations(pairwise_perms, activations, n_models, width)
    gl_maps = estimate_gl_alignments_from_activations(activations, n_models, width)
    permutation_centrality = mean_centrality(triangle_defects(permutation_maps, default_triples(n_models)))

    rows: list[dict] = []
    setting_id = f"mnist_mlp_N{n_models}_W{width}_S{seed}"
    diag_rows_for_maps(
        rows=rows,
        source="real_mnist",
        setting_id=setting_id,
        diagnostic_level="permutation",
        maps=permutation_maps,
        n_models=n_models,
        width=width,
        seed=seed,
        block_size=0,
        partition=None,
        model_blocks=None,
        activations=activations,
        max_order=args.max_order,
        pairwise_feature_residual=None,
        global_feature_residual=None,
        global_sync_residual=None,
        max_global_sync_residual=None,
        accepted_global_sync=None,
        merge_metrics=merge_metrics,
        permutation_centrality=permutation_centrality,
    )
    diag_rows_for_maps(
        rows=rows,
        source="real_mnist",
        setting_id=setting_id,
        diagnostic_level="monomial_positive_scale",
        maps=monomial_maps,
        n_models=n_models,
        width=width,
        seed=seed,
        block_size=0,
        partition=None,
        model_blocks=None,
        activations=activations,
        max_order=args.max_order,
        pairwise_feature_residual=None,
        global_feature_residual=None,
        global_sync_residual=None,
        max_global_sync_residual=None,
        accepted_global_sync=None,
        merge_metrics=merge_metrics,
        permutation_centrality=permutation_centrality,
    )
    diag_rows_for_maps(
        rows=rows,
        source="real_mnist",
        setting_id=setting_id,
        diagnostic_level="low_rank_GL",
        maps=gl_maps,
        n_models=n_models,
        width=width,
        seed=seed,
        block_size=0,
        partition=None,
        model_blocks=None,
        activations=activations,
        max_order=args.max_order,
        pairwise_feature_residual=None,
        global_feature_residual=None,
        global_sync_residual=None,
        max_global_sync_residual=None,
        accepted_global_sync=None,
        merge_metrics=merge_metrics,
        permutation_centrality=permutation_centrality,
    )

    output_weights = models[ref].classifier.weight.detach().cpu().numpy()
    for block_size in parse_csv(args.block_sizes, int):
        if block_size > width:
            continue
        for partition_method in parse_csv(args.partition_methods):
            partition = make_block_partition(
                partition_method,
                width,
                block_size,
                activations=activations[ref],
                output_weights=output_weights,
                seed=seed + 13 * block_size,
                allow_remainder=args.allow_remainder_block,
            )
            model_blocks = induce_model_blocks(partition.as_arrays(), synced)
            pairwise_block_maps, pairwise_stats = estimate_block_orthogonal_alignments_for_model_blocks(
                model_blocks,
                activations,
                n_models,
                width,
                block_size,
            )
            pairwise_feature_residual = summarize_block_alignment_stats(pairwise_stats)["mean_pairwise_block_residual"]
            global_sync = global_block_spectral_synchronization(
                pairwise_block_maps,
                model_blocks,
                n_models,
                width,
                activations=activations,
            )
            pairwise_block_centrality = mean_centrality(triangle_defects(pairwise_block_maps, default_triples(n_models)))
            global_block_centrality = mean_centrality(triangle_defects(global_sync.synchronized_maps, default_triples(n_models)))
            for level_name, maps in (
                ("pairwise_block_procrustes", pairwise_block_maps),
                ("global_block_synchronization", global_sync.synchronized_maps),
            ):
                diag_rows_for_maps(
                    rows=rows,
                    source="real_mnist",
                    setting_id=setting_id,
                    diagnostic_level=level_name,
                    maps=maps,
                    n_models=n_models,
                    width=width,
                    seed=seed,
                    block_size=block_size,
                    partition=partition,
                    model_blocks=model_blocks,
                    activations=activations,
                    max_order=args.max_order,
                    pairwise_feature_residual=pairwise_feature_residual,
                    global_feature_residual=global_sync.feature_alignment_residual,
                    global_sync_residual=global_sync.connection_residual,
                    max_global_sync_residual=global_sync.max_connection_residual,
                    accepted_global_sync=global_sync_accepted(global_sync, args.global_sync_acceptance_tolerance),
                    merge_metrics=merge_metrics,
                    permutation_centrality=permutation_centrality,
                    pairwise_block_centrality=pairwise_block_centrality,
                    global_block_centrality=global_block_centrality,
                )
    return rows


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (source, level, partition_method, block_size), group in df.groupby(
        ["source", "diagnostic_level", "partition_method", "block_size"], dropna=False
    ):
        central = group["scalar_projective_candidate"].fillna(False).astype(bool)
        rows.append(
            {
                "source": source,
                "diagnostic_level": level,
                "partition_method": partition_method,
                "block_size": int(block_size),
                "n_rows": int(len(group)),
                "mean_cycle_score": float(pd.to_numeric(group["cycle_score"], errors="coerce").mean()),
                "mean_centrality_score": float(pd.to_numeric(group["centrality_score"], errors="coerce").mean()),
                "fraction_central_projective_candidates": float(central.mean()),
                "mean_pairwise_feature_alignment_residual": float(pd.to_numeric(group["pairwise_feature_alignment_residual"], errors="coerce").mean()),
                "mean_global_feature_alignment_residual": float(pd.to_numeric(group["global_feature_alignment_residual"], errors="coerce").mean()),
                "mean_global_sync_residual": float(pd.to_numeric(group["global_sync_residual"], errors="coerce").mean()),
                "fraction_accepted_global_sync": float(group["accepted_global_sync"].fillna(False).astype(bool).mean()),
                "mean_improvement_permutation_to_pairwise_block": float(pd.to_numeric(group["improvement_permutation_to_pairwise_block"], errors="coerce").mean()),
                "mean_improvement_permutation_to_global_block": float(pd.to_numeric(group["improvement_permutation_to_global_block"], errors="coerce").mean()),
                "mean_improvement_pairwise_block_to_global_block": float(pd.to_numeric(group["improvement_pairwise_block_to_global_block"], errors="coerce").mean()),
                "mean_c2m3_accuracy": float(pd.to_numeric(group["c2m3_accuracy"], errors="coerce").mean()),
                "mean_monomial_accuracy": float(pd.to_numeric(group["monomial_accuracy"], errors="coerce").mean()),
                "mean_greedy_soup_accuracy": float(pd.to_numeric(group["greedy_soup_accuracy"], errors="coerce").mean()),
                "mean_ensemble_accuracy": float(pd.to_numeric(group["ensemble_accuracy"], errors="coerce").mean()),
            }
        )
    return pd.DataFrame(rows)


def table(rows: list[dict], columns: list[str]) -> str:
    if not rows:
        return "_No rows._"
    return format_markdown_table([{col: row.get(col, "") for col in columns} for row in rows], columns)


def write_config(args, path: Path) -> None:
    config = {
        "command": args.command_string,
        "git_commit": git_commit(),
        "git_worktree_dirty": git_worktree_dirty(),
        "settings": {
            "dataset": "mnist",
            "model_counts": args.model_counts,
            "widths": args.widths,
            "seeds": args.seeds,
            "block_sizes": args.block_sizes,
            "partition_methods": args.partition_methods,
            "epochs": args.epochs,
            "max_train_samples": args.max_train_samples,
            "max_test_samples": args.max_test_samples,
            "val_fraction": args.val_fraction,
            "global_sync_acceptance_tolerance": args.global_sync_acceptance_tolerance,
        },
        "environment": capture_environment(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")


def write_report(args, df: pd.DataFrame, summary: pd.DataFrame, path: Path) -> None:
    synthetic = df[df["source"] == "synthetic"].to_dict("records")
    real = summary[summary["source"] == "real_mnist"].to_dict("records")
    real_blocks = summary[
        (summary["source"] == "real_mnist")
        & (summary["diagnostic_level"].isin(["pairwise_block_procrustes", "global_block_synchronization"]))
    ].to_dict("records")
    real_projective_fraction = float(df[df["source"] == "real_mnist"]["scalar_projective_candidate"].fillna(False).astype(bool).mean())
    global_real = summary[(summary["source"] == "real_mnist") & (summary["diagnostic_level"] == "global_block_synchronization")]
    mean_global_over_pairwise = float(pd.to_numeric(global_real["mean_improvement_pairwise_block_to_global_block"], errors="coerce").mean())
    pairwise_real = summary[(summary["source"] == "real_mnist") & (summary["diagnostic_level"] == "pairwise_block_procrustes")]
    learned_pairwise = pairwise_real[pairwise_real["partition_method"] != "contiguous"]
    contiguous_pairwise = pairwise_real[pairwise_real["partition_method"] == "contiguous"]
    learned_better_text = "not evaluated"
    if not learned_pairwise.empty and not contiguous_pairwise.empty:
        learned_mean = float(pd.to_numeric(learned_pairwise["mean_centrality_score"], errors="coerce").mean())
        contiguous_mean = float(pd.to_numeric(contiguous_pairwise["mean_centrality_score"], errors="coerce").mean())
        learned_better_text = (
            "learned partitions reduce mean observed pairwise-block centrality versus contiguous"
            if learned_mean < contiguous_mean
            else "learned partitions do not reduce mean observed pairwise-block centrality versus contiguous"
        )
    accepted_fraction = float(pd.to_numeric(global_real["fraction_accepted_global_sync"], errors="coerce").mean()) if not global_real.empty else float("nan")
    if real_projective_fraction > 0:
        interpretation = "Some real rows passed scalar/projective checks; these remain descriptive and require follow-up."
    elif mean_global_over_pairwise > 0:
        interpretation = (
            "Global block synchronization projects maps to cycle-consistent gauges and lowers cycle/centrality by construction. "
            "The observed real pairwise block defects still produce no scalar/projective candidates; the connection residual and accepted-sync fraction are the relevant honesty checks."
        )
    else:
        interpretation = "Global/learned block synchronization does not improve mean centrality over pairwise block Procrustes, and real residuals remain noncentral."

    synthetic_cols = [
        "setting_id",
        "diagnostic_level",
        "expected_outcome",
        "cycle_score",
        "centrality_score",
        "scalar_projective_candidate",
        "global_sync_residual",
        "accepted_global_sync",
    ]
    summary_cols = [
        "diagnostic_level",
        "partition_method",
        "block_size",
        "n_rows",
        "mean_cycle_score",
        "mean_centrality_score",
        "fraction_central_projective_candidates",
        "mean_pairwise_feature_alignment_residual",
        "mean_global_feature_alignment_residual",
        "mean_global_sync_residual",
        "fraction_accepted_global_sync",
        "mean_improvement_permutation_to_pairwise_block",
        "mean_improvement_pairwise_block_to_global_block",
    ]
    merge_cols = [
        "diagnostic_level",
        "partition_method",
        "block_size",
        "mean_c2m3_accuracy",
        "mean_monomial_accuracy",
        "mean_greedy_soup_accuracy",
        "mean_ensemble_accuracy",
    ]
    report = f"""# Global Block Synchronization Report

This report is generated by `experiments/global_block_synchronization_experiment.py`.

## Exact Command

```bash
{args.command_string}
```

## Git State At Report Generation

- HEAD commit: `{git_commit()}`
- Worktree dirty: `{git_worktree_dirty()}`

## Settings

- Dataset: MNIST
- Architecture: one-hidden-layer ReLU MLP
- Model counts: `{args.model_counts}`
- Widths: `{args.widths}`
- Seeds: `{args.seeds}`
- Epochs: `{args.epochs}`
- Train samples: `{args.max_train_samples}`
- Test samples: `{args.max_test_samples}`
- Block sizes: `{args.block_sizes}`
- Partition methods: `{args.partition_methods}`
- Matching: activation

## Synthetic Controls

{table(synthetic, synthetic_cols)}

## Real MNIST Diagnostic Summary

{table(real, summary_cols)}

## Learned-Block Comparison

{table(real_blocks, summary_cols)}

Interpretation of learned blocks: {learned_better_text}.

## Comparison To C2M3, Monomial Scaling, Greedy Soup, And Ensemble

{table(real, merge_cols)}

The block-orthogonal rows are diagnostics only.  ReLU-compatible merge accuracy
is reported for C2M3 permutation, positive monomial scaling, greedy soup, and
the ensemble upper bound.  No block-orthogonal same-architecture ReLU merge is
evaluated here.

## Central/Projective Candidates

Real MNIST central/projective candidate fraction: `{real_projective_fraction:.4f}`.
Mean fraction of real global block rows accepted by the connection-residual
threshold: `{accepted_fraction:.4f}`.

{interpretation}

## Negative Boundaries

- Global synchronization projects pairwise maps to a cycle-consistent connection, but this is not a proof that the observed pairwise connection is globally solved unless the connection residual is small.
- General block-orthogonal rotations are feature-space diagnostics for ReLU MLPs, not exact same-architecture ReLU symmetries.
- This does not prove real neural defects are Brauer/projective classes.
- This does not prove TwistedMerge++ beats C2M3; the validated monomial benchmark remains the actionable ReLU-compatible direction.
- No capacity-matched block-gauge merge is claimed.

## Environment

```json
{json.dumps(capture_environment(), indent=2)}
```
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in range(1700, 1710)))
    parser.add_argument("--model-counts", default="3,4")
    parser.add_argument("--widths", default="16,32")
    parser.add_argument("--block-sizes", default="2,4,8")
    parser.add_argument("--partition-methods", default="contiguous,activation_correlation,output_weight_similarity")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-train-samples", type=int, default=2000)
    parser.add_argument("--max-test-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dataset-seed", type=int, default=8128)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--max-order", type=int, default=12)
    parser.add_argument("--global-sync-acceptance-tolerance", type=float, default=0.15)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "reports")
    parser.add_argument("--skip-real", action="store_true")
    parser.add_argument("--allow-remainder-block", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    env_prefix = [
        f"{name}={os.environ[name]}"
        for name in ("PYTHONPYCACHEPREFIX", "MPLCONFIGDIR")
        if os.environ.get(name)
    ]
    args.command_string = " ".join([*env_prefix, sys.executable, *sys.argv])

    rows = controlled_rows(args.max_order)
    if not args.skip_real:
        spec, train_data, test_data = load_dataset(
            "mnist",
            args.data_dir,
            args.max_train_samples,
            args.max_test_samples,
            args.dataset_seed,
        )
        for seed in parse_csv(args.seeds, int):
            for n_models in parse_csv(args.model_counts, int):
                for width in parse_csv(args.widths, int):
                    rows.extend(run_real_setting(args, spec, train_data, test_data, seed, n_models, width))

    df = pd.DataFrame(rows)
    summary = summarize(df)
    csv_dir = args.reports_dir / "csv"
    config_dir = args.reports_dir / "configs"
    csv_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    results_path = csv_dir / "global_block_synchronization.csv"
    summary_path = csv_dir / "global_block_synchronization_summary.csv"
    report_path = args.reports_dir / "global_block_synchronization_report.md"
    config_path = config_dir / "global_block_synchronization_config.json"
    df.to_csv(results_path, index=False)
    summary.to_csv(summary_path, index=False)
    write_config(args, config_path)
    write_report(args, df, summary, report_path)
    print(f"wrote {results_path}")
    print(f"wrote {summary_path}")
    print(f"wrote {report_path}")
    print(f"wrote {config_path}")


if __name__ == "__main__":
    main()
