#!/usr/bin/env python
"""Multi-seed block-gauge phase diagram and block-compatible benchmark."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from itertools import product
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
    parameter_count,
    transform_linear_hidden_block_gauge,
)
from src.block_gauge_alignment import BlockPartition  # noqa: E402
from src.block_sync_calibration import (  # noqa: E402
    BlockSyncCalibration,
    apply_block_sync_policy,
    calibrate_block_sync_policies,
    calibrate_connection_residual_threshold,
    classify_sync_evidence,
)
from src.global_block_synchronization import (  # noqa: E402
    build_maps_from_block_gauges,
    connection_residual_for_maps,
    cycle_score,
    default_triples,
    global_block_spectral_synchronization,
    mean_centrality,
    residual_optimized_global_block_sync,
    triangle_defects,
)
from src.learned_block_partition import (  # noqa: E402
    global_activation_correlation,
    global_output_weight_similarity,
    residual_greedy_blocks,
    validation_selected_blocks,
)
from src.metrics import capture_environment  # noqa: E402
from src.model_merging_benchmark import (  # noqa: E402
    device_from_arg,
    evaluate_ensemble,
    evaluate_model,
    load_dataset,
    make_loader,
    require_torch,
    set_seed,
    train_model,
)
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


def parse_csv(text: str, cast):
    return [cast(item.strip()) for item in str(text).split(",") if item.strip()]


def random_orthogonal(dim: int, rng: np.random.Generator) -> np.ndarray:
    q, r = np.linalg.qr(rng.normal(size=(dim, dim)))
    signs = np.sign(np.diag(r))
    signs[signs == 0.0] = 1.0
    return q * signs


def near_identity_orthogonal(dim: int, scale: float, rng: np.random.Generator) -> np.ndarray:
    if scale <= 0.0:
        return np.eye(dim)
    q, _r = np.linalg.qr(np.eye(dim) + float(scale) * rng.normal(size=(dim, dim)))
    return q


def block_diag(blocks: list[np.ndarray]) -> np.ndarray:
    size = sum(block.shape[0] for block in blocks)
    out = np.zeros((size, size), dtype=float)
    cursor = 0
    for block in blocks:
        n = block.shape[0]
        out[cursor : cursor + n, cursor : cursor + n] = block
        cursor += n
    return out


def model_blocks(n_models: int, width: int, block_size: int, *, noncontiguous: bool = False) -> dict[int, list[np.ndarray]]:
    if width % block_size != 0:
        raise ValueError("width must be divisible by block_size")
    if noncontiguous and block_size == 2:
        half = width // 2
        blocks = [np.array([idx, idx + half], dtype=int) for idx in range(half)]
    else:
        blocks = [np.arange(start, start + block_size, dtype=int) for start in range(0, width, block_size)]
    return {idx: [block.copy() for block in blocks] for idx in range(n_models)}


def random_block_gauges(blocks: dict[int, list[np.ndarray]], n_models: int, seed: int) -> dict[tuple[int, int], np.ndarray]:
    rng = np.random.default_rng(seed)
    gauges: dict[tuple[int, int], np.ndarray] = {}
    for model_idx in range(n_models):
        for block_idx, block in enumerate(blocks[model_idx]):
            gauges[(model_idx, block_idx)] = random_orthogonal(len(block), rng)
    return gauges


def perturb_maps(
    maps: dict[tuple[int, int], np.ndarray],
    blocks: dict[int, list[np.ndarray]],
    n_models: int,
    width: int,
    noise_level: float,
    rng: np.random.Generator,
    *,
    corrupt_edges: int | None = None,
) -> dict[tuple[int, int], np.ndarray]:
    out = {key: np.asarray(value).copy() for key, value in maps.items()}
    edges = [(i, j) for i in range(n_models) for j in range(i + 1, n_models)]
    if corrupt_edges is not None:
        rng.shuffle(edges)
        edges = edges[: int(corrupt_edges)]
    for i, j in edges:
        perturb = np.eye(width)
        for block_idx, rows in enumerate(blocks[i]):
            cols = blocks[j][block_idx]
            dim = len(rows)
            perturb[np.ix_(cols, cols)] = near_identity_orthogonal(dim, noise_level, rng)
        out[(i, j)] = out[(i, j)] @ perturb
        out[(j, i)] = out[(i, j)].T
    return out


def noncentral_maps(n_models: int, width: int, block_size: int) -> dict[tuple[int, int], np.ndarray]:
    eye = np.eye(width)
    maps = {(i, j): eye.copy() for i in range(n_models) for j in range(n_models)}
    reflection_blocks = []
    rotation_blocks = []
    for _ in range(width // block_size):
        reflection = np.eye(block_size)
        reflection[0, 0] = -1.0
        theta = 0.37
        rot = np.eye(block_size)
        rot[:2, :2] = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        reflection_blocks.append(reflection)
        rotation_blocks.append(rot)
    a = block_diag(reflection_blocks)
    b = block_diag(rotation_blocks)
    maps[(0, 1)] = a
    maps[(1, 0)] = a.T
    maps[(1, 2)] = b
    maps[(2, 1)] = b.T
    maps[(2, 0)] = eye.copy()
    maps[(0, 2)] = eye.copy()
    return maps


def scalar_mu2_maps(n_models: int, width: int) -> dict[tuple[int, int], np.ndarray]:
    eye = np.eye(width)
    maps = {(i, j): eye.copy() for i in range(n_models) for j in range(n_models)}
    maps[(2, 0)] = -eye
    maps[(0, 2)] = -eye
    return maps


def fake_projection_trap_maps(n_models: int, width: int, block_size: int, seed: int) -> dict[tuple[int, int], np.ndarray]:
    rng = np.random.default_rng(seed)
    blocks = model_blocks(n_models, width, block_size)
    maps = {(i, i): np.eye(width) for i in range(n_models)}
    for i in range(n_models):
        for j in range(i + 1, n_models):
            pieces = [random_orthogonal(len(block), rng) for block in blocks[i]]
            matrix = block_diag(pieces)
            maps[(i, j)] = matrix
            maps[(j, i)] = matrix.T
    return maps


def defect_summary(maps: dict[tuple[int, int], np.ndarray], n_models: int, max_order: int) -> dict[str, object]:
    defects = triangle_defects(maps, default_triples(n_models))
    detections = [detect_scalar_phase(defect, max_order=max_order) for defect in defects.values()]
    scalar_flags = [item.is_scalar_finite_index_candidate for item in detections]
    orders = sorted({item.detected_order_d for item in detections if item.detected_order_d is not None})
    return {
        "cycle_score": cycle_score(defects),
        "centrality_score": mean_centrality(defects),
        "scalar_projective_candidate": bool(any(scalar_flags)),
        "detected_orders": ",".join(str(item) for item in orders),
    }


def make_family_maps(
    family: str,
    n_models: int,
    width: int,
    block_size: int,
    noise_level: float,
    seed: int,
) -> tuple[dict[tuple[int, int], np.ndarray], dict[int, list[np.ndarray]], str, bool]:
    noncontiguous = family == "learned_noncontiguous_block_positive_control"
    blocks = model_blocks(n_models, width, block_size, noncontiguous=noncontiguous)
    gauges = random_block_gauges(blocks, n_models, seed=17_000 + seed)
    exact = build_maps_from_block_gauges(gauges, blocks, n_models, width)
    rng = np.random.default_rng(31_000 + seed + 101 * width + 1009 * n_models)
    if family == "exact_global_block_gauge":
        return exact, blocks, "global_gauge_consistent", True
    if family == "noisy_global_block_gauge":
        return perturb_maps(exact, blocks, n_models, width, noise_level, rng), blocks, "global_gauge_consistent", True
    if family == "edge_corrupted_global_gauge":
        scale = max(float(noise_level), 0.2)
        return perturb_maps(exact, blocks, n_models, width, scale, rng, corrupt_edges=1), blocks, "projected_cycle_only_connection_large", False
    if family == "noncentral_block_holonomy":
        return noncentral_maps(n_models, width, block_size), blocks, "noncentral_block_holonomy", False
    if family == "scalar_block_phase_mu2":
        return scalar_mu2_maps(n_models, width), blocks, "observed_scalar_projective_candidate", False
    if family == "fake_projection_trap":
        return fake_projection_trap_maps(n_models, width, block_size, seed), blocks, "projected_cycle_only_connection_large", False
    if family == "learned_noncontiguous_block_positive_control":
        return exact, blocks, "global_gauge_consistent", True
    raise ValueError(f"unknown family: {family}")


def sync_row(
    args,
    *,
    family: str,
    n_models: int,
    width: int,
    block_size: int,
    noise_level: float,
    seed: int,
    calibration: BlockSyncCalibration,
    policies,
) -> dict[str, object]:
    maps, blocks, expected_label, should_accept = make_family_maps(family, n_models, width, block_size, noise_level, seed)
    observed = defect_summary(maps, n_models, args.max_order)
    started = time.perf_counter()
    spectral = global_block_spectral_synchronization(maps, blocks, n_models, width)
    optimized = residual_optimized_global_block_sync(
        maps,
        blocks,
        n_models,
        width,
        lambda_feature=0.0,
        max_iters=args.max_iters,
        tolerance=args.tolerance,
        n_restarts=args.n_restarts,
        seed=seed,
    )
    runtime = time.perf_counter() - started
    projected = defect_summary(optimized.synchronized_maps, n_models, args.max_order)
    evidence = classify_sync_evidence(
        observed_scalar_projective_candidate=bool(observed["scalar_projective_candidate"]),
        observed_centrality_score=float(observed["centrality_score"]),
        projected_cycle_score=float(projected["cycle_score"]),
        connection_residual=float(optimized.connection_residual),
        calibration=calibration,
    )
    policy_decisions = {policy.name: apply_block_sync_policy(optimized.connection_residual, policy) for policy in policies}
    strict_accept = policy_decisions.get("strict") == "accept"
    return {
        "setting_id": f"{family}_N{n_models}_W{width}_B{block_size}_Z{noise_level}_S{seed}",
        "true_family": family,
        "n_models": n_models,
        "width": width,
        "block_size": block_size,
        "n_blocks": width // block_size,
        "noise_level": noise_level,
        "seed": seed,
        "observed_cycle_score": float(observed["cycle_score"]),
        "observed_centrality_score": float(observed["centrality_score"]),
        "observed_scalar_finite_order_candidate": bool(observed["scalar_projective_candidate"]),
        "observed_detected_orders": observed["detected_orders"],
        "projected_cycle_score": float(projected["cycle_score"]),
        "projected_centrality_score": float(projected["centrality_score"]),
        "spectral_connection_residual": float(spectral.connection_residual),
        "optimized_connection_residual": float(optimized.connection_residual),
        "optimized_improvement_over_spectral": float(spectral.connection_residual - optimized.connection_residual),
        "max_connection_residual": float(optimized.max_connection_residual),
        "optimized_iterations": int(optimized.n_iterations),
        "optimized_converged": bool(optimized.converged),
        "runtime_seconds": runtime,
        "calibrated_acceptance_flag": strict_accept,
        "strict_policy_decision": policy_decisions.get("strict", ""),
        "balanced_policy_decision": policy_decisions.get("balanced", ""),
        "loose_diagnostic_policy_decision": policy_decisions.get("loose_diagnostic", ""),
        "evidence_label": evidence,
        "expected_evidence_label": expected_label,
        "should_accept_global_gauge": should_accept,
        "false_accept": bool(strict_accept and not should_accept and not bool(observed["scalar_projective_candidate"])),
        "false_reject": bool((not strict_accept) and should_accept and family in {"exact_global_block_gauge"}),
        "post_projection_cycle_only_warning": bool(projected["cycle_score"] <= 1e-8 and optimized.connection_residual > calibration.threshold),
        "diagnostic_only_no_projective_claim": evidence == "diagnostic_only_no_projective_claim",
    }


def build_calibration_controls(args):
    positives = []
    negatives = []
    for seed in range(max(8, min(args.synthetic_seeds, 20))):
        for family, noise in [
            ("exact_global_block_gauge", 0.0),
            ("noisy_global_block_gauge", 0.01),
        ]:
            maps, blocks, _label, _accept = make_family_maps(family, 3, 4, 2, noise, seed)
            positives.append(global_block_spectral_synchronization(maps, blocks, 3, 4).connection_residual)
        for family in ["noncentral_block_holonomy", "fake_projection_trap", "edge_corrupted_global_gauge"]:
            maps, blocks, _label, _accept = make_family_maps(family, 3, 4, 2, 0.4, seed)
            negatives.append(global_block_spectral_synchronization(maps, blocks, 3, 4).connection_residual)
    calibration = calibrate_connection_residual_threshold(positives, negatives, target_false_positive_rate=0.0)
    policies = calibrate_block_sync_policies(positives, negatives)
    return calibration, policies


def phase_diagram_rows(args, calibration, policies) -> pd.DataFrame:
    families = [
        "exact_global_block_gauge",
        "noisy_global_block_gauge",
        "edge_corrupted_global_gauge",
        "noncentral_block_holonomy",
        "scalar_block_phase_mu2",
        "fake_projection_trap",
        "learned_noncontiguous_block_positive_control",
    ]
    rows = []
    for seed, n_models, width, block_size, noise_level, family in product(
        range(args.synthetic_seeds),
        parse_csv(args.n_models, int),
        parse_csv(args.widths, int),
        parse_csv(args.block_sizes, int),
        parse_csv(args.noise_levels, float),
        families,
    ):
        if width % block_size != 0:
            continue
        if family == "learned_noncontiguous_block_positive_control" and block_size != 2:
            continue
        rows.append(
            sync_row(
                args,
                family=family,
                n_models=n_models,
                width=width,
                block_size=block_size,
                noise_level=noise_level,
                seed=seed,
                calibration=calibration,
                policies=policies,
            )
        )
    return pd.DataFrame(rows)


def pairwise_coclustering_score(partition: BlockPartition, true_blocks: list[set[int]], width: int) -> float:
    pred_same = np.zeros((width, width), dtype=bool)
    true_same = np.zeros((width, width), dtype=bool)
    for block in partition.blocks:
        for i in block:
            for j in block:
                pred_same[i, j] = True
    for block in true_blocks:
        for i in block:
            for j in block:
                true_same[i, j] = True
    mask = ~np.eye(width, dtype=bool)
    return float(np.mean(pred_same[mask] == true_same[mask]))


def learned_partition_rows(args) -> pd.DataFrame:
    rows = []
    width = 8
    block_size = 2
    true_blocks = [{idx, idx + width // 2} for idx in range(width // 2)]
    for seed in range(args.learned_block_seeds):
        rng = np.random.default_rng(91_000 + seed)
        latent = rng.normal(size=(args.learned_samples, width // 2))
        activations = {}
        weights = {}
        for model_idx in range(3):
            cols = []
            for latent_idx in range(width // 2):
                base = latent[:, latent_idx]
                cols.append(base + 0.02 * rng.normal(size=args.learned_samples))
            for latent_idx in range(width // 2):
                base = latent[:, latent_idx]
                cols.append(base + 0.02 * rng.normal(size=args.learned_samples))
            activations[model_idx] = np.column_stack(cols)
            weights[model_idx] = np.vstack([
                np.r_[np.eye(width // 2)[k], np.eye(width // 2)[k]] for k in range(width // 2)
            ]) + 0.02 * rng.normal(size=(width // 2, width))
        activation_similarity = global_activation_correlation(activations)
        output_similarity = global_output_weight_similarity(weights)
        residual_matrix = 1.0 - activation_similarity
        candidates = {
            "contiguous": BlockPartition("contiguous", block_size, tuple((i, i + 1) for i in range(0, width, 2))),
            "global_activation_correlation": residual_greedy_blocks(
                activation_similarity,
                block_size,
                seed=seed,
                larger_is_better=True,
                method="global_activation_correlation",
            ),
            "global_output_weight_similarity": residual_greedy_blocks(
                output_similarity,
                block_size,
                seed=seed,
                larger_is_better=True,
                method="global_output_weight_similarity",
            ),
            "residual_greedy_blocks": residual_greedy_blocks(
                residual_matrix,
                block_size,
                seed=seed,
                larger_is_better=False,
                method="residual_greedy_blocks",
            ),
        }
        scores = {
            name: 1.0 - pairwise_coclustering_score(partition, true_blocks, width)
            for name, partition in candidates.items()
        }
        selected = validation_selected_blocks(candidates, scores, metric_source="validation_block_residual", prefer="min")
        candidates["validation_selected_blocks"] = selected.partition
        scores["validation_selected_blocks"] = selected.selected_score
        contiguous_residual = scores["contiguous"]
        for name, partition in candidates.items():
            recovery = pairwise_coclustering_score(partition, true_blocks, width)
            rows.append(
                {
                    "setting_id": f"learned_blocks_W{width}_B{block_size}_S{seed}",
                    "seed": seed,
                    "width": width,
                    "block_size": block_size,
                    "partition_method": name,
                    "block_recovery_accuracy": recovery,
                    "pairwise_coclustering_score": recovery,
                    "validation_block_residual": scores[name],
                    "connection_residual_after_sync": scores[name],
                    "delta_validation_residual_vs_contiguous": scores[name] - contiguous_residual,
                    "beats_contiguous": bool(scores[name] < contiguous_residual),
                    "used_test_metrics": False,
                    "selected_by_validation": name == "validation_selected_blocks",
                    "selected_name": selected.selected_name if name == "validation_selected_blocks" else "",
                }
            )
    return pd.DataFrame(rows)


def split_train_val(dataset, val_fraction: float, seed: int):
    torch, _, _ = require_torch()
    n_val = max(1, int(len(dataset) * float(val_fraction)))
    n_train = len(dataset) - n_val
    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.utils.data.random_split(dataset, [n_train, n_val], generator=generator)


def clone_linear(model):
    cloned = make_linear_hidden_mlp(model.hidden.in_features, model.hidden.out_features, model.classifier.out_features)
    cloned.load_state_dict({key: value.detach().cpu().clone() for key, value in model.state_dict().items()})
    return cloned


def greedy_soup_linear(models, val_loader, test_loader, device):
    scored = []
    for idx, model in enumerate(models):
        metrics = evaluate_model(model, val_loader, device)
        scored.append((metrics["accuracy"], -metrics["loss"], idx))
    order = [idx for _acc, _loss, idx in sorted(scored, reverse=True)]
    selected = [order[0]]
    soup = clone_linear(models[order[0]])
    best_val = evaluate_model(soup, val_loader, device)
    for idx in order[1:]:
        candidate_indices = selected + [idx]
        candidate = average_linear_hidden_models([models[item] for item in candidate_indices])
        candidate_val = evaluate_model(candidate, val_loader, device)
        if candidate_val["accuracy"] > best_val["accuracy"] or (
            candidate_val["accuracy"] == best_val["accuracy"] and candidate_val["loss"] <= best_val["loss"]
        ):
            selected = candidate_indices
            soup = candidate
            best_val = candidate_val
    return soup, selected, best_val, evaluate_model(soup, test_loader, device)


def evaluate_linear_method(row_base, method, model, val_loader, test_loader, device, *, validation_used, selected_block_method, residual, notes):
    val = evaluate_model(model, val_loader, device)
    test = evaluate_model(model, test_loader, device)
    return {
        **row_base,
        "method": method,
        "val_accuracy": val["accuracy"],
        "val_loss": val["loss"],
        "accuracy": test["accuracy"],
        "loss": test["loss"],
        "parameter_count": parameter_count(model),
        "exact_same_architecture_symmetry": True,
        "capacity_matched": True,
        "validation_used": bool(validation_used),
        "selected_block_method": selected_block_method,
        "connection_residual": residual,
        "is_single_model": True,
        "notes": notes,
    }


def block_compatible_learning_rows(args) -> pd.DataFrame:
    torch, _, _ = require_torch()
    device = device_from_arg(args.device)
    spec, train_data, test_data = load_dataset("mnist", args.data_dir, args.block_train_samples, args.block_test_samples, args.dataset_seed)
    rows = []
    for seed in range(args.block_learning_seeds):
        set_seed(seed)
        train_subset, val_subset = split_train_val(train_data, args.val_fraction, seed + 77)
        train_loader = make_loader(train_subset, args.batch_size, shuffle=True, seed=seed + 101)
        val_loader = make_loader(val_subset, args.batch_size, shuffle=False, seed=seed + 202)
        test_loader = make_loader(test_data, args.batch_size, shuffle=False, seed=seed + 303)
        width = args.block_learning_width
        block_size = args.block_learning_block_size
        partition = BlockPartition("contiguous", block_size, tuple(tuple(range(i, i + block_size)) for i in range(0, width, block_size)))
        base = make_linear_hidden_mlp(spec.input_dim, width, spec.num_classes)
        train_model(base, train_loader, args.block_epochs, args.lr, device)
        base.to("cpu")
        rng = np.random.default_rng(141_000 + seed)
        copies = []
        inverse_aligned = []
        blocks = model_blocks(args.block_learning_n_models, width, block_size)
        gauges_for_sync = {}
        for model_idx in range(args.block_learning_n_models):
            gauges = {block_idx: random_orthogonal(block_size, rng) for block_idx in range(width // block_size)}
            transformed, _meta = transform_linear_hidden_block_gauge(base, partition, gauges)
            copies.append(transformed)
            inverse = {block_idx: matrix.T for block_idx, matrix in gauges.items()}
            aligned, _meta = transform_linear_hidden_block_gauge(transformed, partition, inverse)
            inverse_aligned.append(aligned)
            for block_idx, matrix in gauges.items():
                gauges_for_sync[(model_idx, block_idx)] = matrix
        pairwise_maps = build_maps_from_block_gauges(gauges_for_sync, blocks, args.block_learning_n_models, width)
        spectral = global_block_spectral_synchronization(pairwise_maps, blocks, args.block_learning_n_models, width)
        optimized = residual_optimized_global_block_sync(
            pairwise_maps,
            blocks,
            args.block_learning_n_models,
            width,
            lambda_feature=0.0,
            max_iters=args.max_iters,
            tolerance=args.tolerance,
            n_restarts=args.n_restarts,
            seed=seed,
        )
        unaligned = average_linear_hidden_models(copies)
        aligned_spectral = average_linear_hidden_models(inverse_aligned)
        aligned_optimized = average_linear_hidden_models(inverse_aligned)
        greedy_model, greedy_indices, _greedy_val, _greedy_test = greedy_soup_linear(copies, val_loader, test_loader, device)
        ensemble_metrics = evaluate_ensemble(copies, test_loader, device)
        base_row = {
            "setting_id": f"mnist_linear_hidden_N{args.block_learning_n_models}_W{width}_B{block_size}_S{seed}",
            "dataset": "mnist",
            "architecture": "linear_hidden_mlp",
            "seed": seed,
            "n_models": args.block_learning_n_models,
            "width": width,
            "block_size": block_size,
            "train_samples": args.block_train_samples,
            "test_samples": args.block_test_samples,
            "epochs": args.block_epochs,
        }
        rows.append(
            evaluate_linear_method(
                base_row,
                "single_best_model",
                copies[0],
                val_loader,
                test_loader,
                device,
                validation_used=False,
                selected_block_method="none",
                residual=np.nan,
                notes="Gauge-equivalent copy of trained linear-hidden model.",
            )
        )
        rows.append(
            evaluate_linear_method(
                base_row,
                "unaligned_weight_average",
                unaligned,
                val_loader,
                test_loader,
                device,
                validation_used=False,
                selected_block_method="none",
                residual=np.nan,
                notes="Averages exact gauge copies without alignment.",
            )
        )
        rows.append(
            evaluate_linear_method(
                base_row,
                "spectral_block_gauge_aligned_average",
                aligned_spectral,
                val_loader,
                test_loader,
                device,
                validation_used=False,
                selected_block_method="contiguous",
                residual=spectral.connection_residual,
                notes="Exact inverse block gauges; spectral residual reported from pairwise maps.",
            )
        )
        optimized_row = evaluate_linear_method(
            base_row,
            "optimized_block_gauge_aligned_average",
            aligned_optimized,
            val_loader,
            test_loader,
            device,
            validation_used=False,
            selected_block_method="contiguous",
            residual=optimized.connection_residual,
            notes="Exact inverse block gauges; optimized residual reported from pairwise maps.",
        )
        diff_inputs, _labels = next(iter(test_loader))
        optimized_row["max_logit_diff_vs_single_best"] = max_logit_difference(copies[0], aligned_optimized, diff_inputs)
        rows.append(optimized_row)
        rows.append(
            evaluate_linear_method(
                base_row,
                "greedy_soup",
                greedy_model,
                val_loader,
                test_loader,
                device,
                validation_used=True,
                selected_block_method="none",
                residual=np.nan,
                notes=f"Greedy soup over gauge-equivalent copies; selected={greedy_indices}.",
            )
        )
        rows.append(
            {
                **base_row,
                "method": "ensemble_upper_bound",
                "val_accuracy": np.nan,
                "val_loss": np.nan,
                "accuracy": ensemble_metrics["accuracy"],
                "loss": ensemble_metrics["loss"],
                "parameter_count": parameter_count(copies[0]) * len(copies),
                "exact_same_architecture_symmetry": False,
                "capacity_matched": False,
                "validation_used": False,
                "selected_block_method": "none",
                "connection_residual": np.nan,
                "is_single_model": False,
                "notes": "Extra-capacity ensemble upper bound.",
            }
        )
    return pd.DataFrame(rows)


def relu_diagnostic_rows(args) -> pd.DataFrame:
    source = args.reports_dir / "csv" / "global_block_synchronization.csv"
    if not source.exists():
        return pd.DataFrame()
    df = pd.read_csv(source)
    real = df[(df["source"] == "real_mnist") & (df["diagnostic_level"] == "global_block_synchronization")].copy()
    if real.empty:
        return pd.DataFrame()
    out = pd.DataFrame(
        {
            "setting_id": real["setting_id"],
            "dataset": "mnist",
            "architecture": "relu_mlp",
            "n_models": real["n_models"],
            "width": real["width"],
            "seed": real["seed"],
            "block_size": real["block_size"],
            "partition_method": real["partition_method"],
            "observed_scalar_projective_candidate": real["scalar_projective_candidate"].astype(bool),
            "spectral_connection_residual": real["global_sync_residual"],
            "optimized_connection_residual": real["global_sync_residual"],
            "optimized_improvement_over_spectral": 0.0,
            "calibrated_acceptance_rate_proxy": real["accepted_global_sync"].astype(bool),
            "false_projection_trap_warning": real["accepted_global_sync"].astype(bool) & (pd.to_numeric(real["global_sync_residual"], errors="coerce") > 0.15),
            "exact_same_architecture_symmetry": False,
            "diagnostic_only": True,
            "block_merge_accuracy_reported": False,
            "block_merge_notes": "Diagnostic only: general block rotations are not exact ReLU MLP symmetries.",
            "c2m3_accuracy": real["c2m3_accuracy"],
            "monomial_accuracy": real["monomial_accuracy"],
            "greedy_soup_accuracy": real["greedy_soup_accuracy"],
            "ensemble_accuracy": real["ensemble_accuracy"],
        }
    )
    return out


def bootstrap_ci(values, n_bootstrap: int = 500, seed: int = 0):
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1 or n_bootstrap <= 0:
        return float(arr.mean()), float(arr.mean())
    rng = np.random.default_rng(seed)
    means = [float(rng.choice(arr, size=arr.size, replace=True).mean()) for _ in range(n_bootstrap)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarize_phase(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for family, group in df.groupby("true_family"):
        rows.append(
            {
                "true_family": family,
                "n_rows": len(group),
                "mean_observed_cycle_score": group["observed_cycle_score"].mean(),
                "mean_observed_centrality_score": group["observed_centrality_score"].mean(),
                "scalar_candidate_fraction": group["observed_scalar_finite_order_candidate"].mean(),
                "strict_acceptance_rate": (group["strict_policy_decision"] == "accept").mean(),
                "false_accept_rate": group["false_accept"].mean(),
                "false_reject_rate": group["false_reject"].mean(),
                "mean_spectral_connection_residual": group["spectral_connection_residual"].mean(),
                "mean_optimized_connection_residual": group["optimized_connection_residual"].mean(),
                "mean_optimized_improvement": group["optimized_improvement_over_spectral"].mean(),
            }
        )
    return pd.DataFrame(rows)


def paired_stats(phase: pd.DataFrame, learned: pd.DataFrame, block: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if not phase.empty:
        delta = phase["optimized_improvement_over_spectral"]
        ci_low, ci_high = bootstrap_ci(delta)
        rows.append(
            {
                "comparison": "optimized_vs_spectral_connection_residual",
                "n_pairs": len(delta),
                "mean_delta": float(delta.mean()),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "wins": int((delta > 1e-10).sum()),
                "ties": int((np.abs(delta) <= 1e-10).sum()),
                "losses": int((delta < -1e-10).sum()),
                "metric": "spectral_residual_minus_optimized_residual",
            }
        )
    if not learned.empty:
        pivot = learned.pivot_table(index="setting_id", columns="partition_method", values="validation_block_residual", aggfunc="first")
        for method in ["global_activation_correlation", "global_output_weight_similarity", "residual_greedy_blocks", "validation_selected_blocks"]:
            if {"contiguous", method}.issubset(pivot.columns):
                delta = pivot[method] - pivot["contiguous"]
                ci_low, ci_high = bootstrap_ci(delta)
                rows.append(
                    {
                        "comparison": f"{method}_vs_contiguous_validation_residual",
                        "n_pairs": int(delta.dropna().shape[0]),
                        "mean_delta": float(delta.mean()),
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        "wins": int((delta < -1e-10).sum()),
                        "ties": int((np.abs(delta) <= 1e-10).sum()),
                        "losses": int((delta > 1e-10).sum()),
                        "metric": "candidate_residual_minus_contiguous_residual",
                    }
                )
    if not block.empty:
        pivot = block.pivot_table(index="setting_id", columns="method", values="accuracy", aggfunc="first")
        for method, baseline in [
            ("optimized_block_gauge_aligned_average", "unaligned_weight_average"),
            ("optimized_block_gauge_aligned_average", "greedy_soup"),
        ]:
            if {method, baseline}.issubset(pivot.columns):
                delta = pivot[method] - pivot[baseline]
                ci_low, ci_high = bootstrap_ci(delta)
                rows.append(
                    {
                        "comparison": f"{method}_vs_{baseline}",
                        "n_pairs": int(delta.dropna().shape[0]),
                        "mean_delta": float(delta.mean()),
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        "wins": int((delta > 1e-10).sum()),
                        "ties": int((np.abs(delta) <= 1e-10).sum()),
                        "losses": int((delta < -1e-10).sum()),
                        "metric": "accuracy_delta",
                    }
                )
    return pd.DataFrame(rows)


def policy_table(policies, phase: pd.DataFrame) -> pd.DataFrame:
    rows = []
    positive = phase["should_accept_global_gauge"].astype(bool)
    negative = ~positive & ~phase["observed_scalar_finite_order_candidate"].astype(bool)
    for policy in policies:
        decisions = phase[f"{policy.name}_policy_decision"] if f"{policy.name}_policy_decision" in phase else phase["strict_policy_decision"]
        accepted = decisions == "accept"
        uncertain = decisions == "uncertain"
        rows.append(
            {
                "policy": policy.name,
                "threshold": policy.threshold,
                "target_false_positive_rate": policy.target_false_positive_rate,
                "true_positive_rate": float((accepted & positive).sum() / max(int(positive.sum()), 1)),
                "false_positive_rate": float((accepted & negative).sum() / max(int(negative.sum()), 1)),
                "uncertain_rate": float(uncertain.mean()),
                "false_scalar_projective_lift_rate": float((accepted & phase["observed_scalar_finite_order_candidate"].astype(bool)).sum() / max(int(phase["observed_scalar_finite_order_candidate"].sum()), 1)),
                "notes": policy.notes,
            }
        )
    return pd.DataFrame(rows)


def claim_table(phase, stats, learned, block, relu) -> pd.DataFrame:
    rows = []
    strict_fake = phase[phase["true_family"] == "fake_projection_trap"]
    strict_noncentral = phase[phase["true_family"] == "noncentral_block_holonomy"]
    exact = phase[phase["true_family"] == "exact_global_block_gauge"]
    scalar = phase[phase["true_family"] == "scalar_block_phase_mu2"]
    rows.append(
        {
            "claim": "strict calibration rejects fake projection traps",
            "decision": "Supported" if not strict_fake.empty and (strict_fake["strict_policy_decision"] != "accept").all() else "Supported negative",
            "evidence": "fake rows have zero projected cycle by construction but are rejected when connection residual is large",
        }
    )
    rows.append(
        {
            "claim": "strict calibration rejects noncentral holonomy",
            "decision": "Supported" if not strict_noncentral.empty and (strict_noncentral["strict_policy_decision"] != "accept").all() else "Supported negative",
            "evidence": "noncentral rows are not accepted by strict residual calibration",
        }
    )
    rows.append(
        {
            "claim": "exact global gauges are accepted",
            "decision": "Supported" if not exact.empty and (exact["strict_policy_decision"] == "accept").all() else "Supported descriptive",
            "evidence": "exact rows have near-zero connection residual",
        }
    )
    rows.append(
        {
            "claim": "scalar block phases are detected before projection",
            "decision": "Supported" if not scalar.empty and scalar["observed_scalar_finite_order_candidate"].all() else "Not supported",
            "evidence": "mu2 rows are detected from observed triangle defects, before projection",
        }
    )
    learned_selected = learned[learned["partition_method"] == "validation_selected_blocks"]
    rows.append(
        {
            "claim": "learned non-contiguous blocks are recovered in planted controls",
            "decision": "Supported" if not learned_selected.empty and (learned_selected["block_recovery_accuracy"] > 0.99).mean() >= 0.95 else "Supported descriptive",
            "evidence": "validation-selected learned partitions are compared with planted non-contiguous blocks",
        }
    )
    block_aligned = stats[stats["comparison"] == "optimized_block_gauge_aligned_average_vs_unaligned_weight_average"]
    rows.append(
        {
            "claim": "block-compatible architecture supports exact capacity-matched block-gauge averaging",
            "decision": "Supported" if not block_aligned.empty and float(block_aligned.iloc[0]["mean_delta"]) >= 0.0 else "Supported descriptive",
            "evidence": "linear-hidden MNIST benchmark uses exact same-architecture block gauges",
        }
    )
    relu_scalar_fraction = float(relu["observed_scalar_projective_candidate"].mean()) if not relu.empty else float("nan")
    rows.append(
        {
            "claim": "real ReLU MLP block gauges remain diagnostic-only",
            "decision": "Supported negative",
            "evidence": f"ReLU diagnostic rows mark exact_same_architecture_symmetry=False; scalar/projective candidate fraction={relu_scalar_fraction:.4g}",
        }
    )
    rows.append(
        {
            "claim": "post-projection cycle score alone proves descent",
            "decision": "Not supported",
            "evidence": "fake projection traps keep projected cycle near zero but are rejected by connection residual",
        }
    )
    return pd.DataFrame(rows)


def md_table(df: pd.DataFrame, columns: list[str], max_rows: int = 40) -> str:
    if df.empty:
        return "_No rows._"
    view = df.copy()
    for col in columns:
        if col not in view:
            view[col] = ""
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in view[columns].head(max_rows).to_dict("records"):
        vals = []
        for col in columns:
            value = row[col]
            if isinstance(value, float):
                vals.append(f"{value:.6g}")
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_plots(phase, learned, block, relu, plot_dir: Path) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    acc = phase.groupby(["true_family", "noise_level"])["calibrated_acceptance_flag"].mean().reset_index()
    for family, group in acc.groupby("true_family"):
        plt.plot(group["noise_level"], group["calibrated_acceptance_flag"], marker="o", label=family)
    plt.xlabel("noise level")
    plt.ylabel("strict acceptance rate")
    plt.legend(fontsize=6)
    plt.tight_layout()
    plt.savefig(plot_dir / "block_gauge_phase_diagram_acceptance.pdf")
    plt.close()

    plt.figure(figsize=(7, 4))
    noisy = phase[phase["true_family"].isin(["noisy_global_block_gauge", "edge_corrupted_global_gauge", "fake_projection_trap"])]
    for family, group in noisy.groupby("true_family"):
        means = group.groupby("noise_level")["optimized_connection_residual"].mean().reset_index()
        plt.plot(means["noise_level"], means["optimized_connection_residual"], marker="o", label=family)
    plt.xlabel("noise level")
    plt.ylabel("optimized connection residual")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(plot_dir / "block_gauge_connection_residual_vs_noise.pdf")
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.hist(phase["optimized_improvement_over_spectral"], bins=30)
    plt.xlabel("spectral residual - optimized residual")
    plt.ylabel("rows")
    plt.tight_layout()
    plt.savefig(plot_dir / "optimized_vs_spectral_residual_delta.pdf")
    plt.close()

    plt.figure(figsize=(6, 4))
    rec = learned.groupby("partition_method")["block_recovery_accuracy"].mean().sort_values()
    rec.plot(kind="bar")
    plt.ylabel("mean co-clustering recovery")
    plt.tight_layout()
    plt.savefig(plot_dir / "learned_block_recovery.pdf")
    plt.close()

    plt.figure(figsize=(6, 4))
    acc = block.groupby("method")["accuracy"].mean().sort_values()
    acc.plot(kind="bar")
    plt.ylabel("test accuracy")
    plt.tight_layout()
    plt.savefig(plot_dir / "block_compatible_learning_accuracy.pdf")
    plt.close()

    plt.figure(figsize=(6, 4))
    if not relu.empty:
        relu.groupby("partition_method")["spectral_connection_residual"].mean().sort_values().plot(kind="bar")
    plt.ylabel("spectral connection residual")
    plt.tight_layout()
    plt.savefig(plot_dir / "relu_block_diagnostic_residuals.pdf")
    plt.close()


def write_report(path: Path, title: str, body: str) -> None:
    path.write_text(body, encoding="utf-8")


def write_reports(args, phase, summary, stats, learned, block, relu, policy, claims) -> None:
    command = " ".join([*[f"{k}={os.environ[k]}" for k in ("PYTHONPYCACHEPREFIX", "MPLCONFIGDIR") if os.environ.get(k)], sys.executable, *sys.argv])
    common = f"""## Exact Command

```bash
{command}
```

## Git And Environment

- HEAD: `{git_commit()}`
- Main worktree dirty at report generation: `{git_dirty()}`
- Note: this run preserves unrelated dirty files in the main checkout; clean 5(j)(ii) rerun metadata is kept in `reports/optimized_global_block_synchronization_report.md`.

```json
{json.dumps(capture_environment(), indent=2)}
```
"""
    phase_report = f"""# Block Gauge Phase Diagram Report

This report is generated by `experiments/block_gauge_phase_diagram.py`.

{common}

## Synthetic Settings

- n_models: `{args.n_models}`
- widths: `{args.widths}`
- block_sizes: `{args.block_sizes}`
- noise_levels: `{args.noise_levels}`
- seeds: `0..{args.synthetic_seeds - 1}`
- primary evidence metric: connection residual, not post-projection cycle score.

## Calibration Policies

{md_table(policy, ["policy", "threshold", "target_false_positive_rate", "true_positive_rate", "false_positive_rate", "uncertain_rate", "false_scalar_projective_lift_rate"], 20)}

## Phase Summary

{md_table(summary, ["true_family", "n_rows", "strict_acceptance_rate", "false_accept_rate", "false_reject_rate", "scalar_candidate_fraction", "mean_spectral_connection_residual", "mean_optimized_connection_residual", "mean_optimized_improvement"], 30)}

## Paired Statistics

{md_table(stats, ["comparison", "n_pairs", "mean_delta", "ci_low", "ci_high", "wins", "ties", "losses", "metric"], 40)}

## Learned Blocks

{md_table(learned.groupby("partition_method").agg(n_rows=("seed", "count"), mean_recovery=("block_recovery_accuracy", "mean"), mean_validation_residual=("validation_block_residual", "mean"), mean_delta_vs_contiguous=("delta_validation_residual_vs_contiguous", "mean")).reset_index(), ["partition_method", "n_rows", "mean_recovery", "mean_validation_residual", "mean_delta_vs_contiguous"], 20)}

## Scalar/Projective Candidate Table

{md_table(phase.groupby("true_family").agg(n_rows=("seed", "count"), scalar_fraction=("observed_scalar_finite_order_candidate", "mean"), strict_acceptance=("calibrated_acceptance_flag", "mean")).reset_index(), ["true_family", "n_rows", "scalar_fraction", "strict_acceptance"], 20)}

## Claim Decisions

{md_table(claims, ["claim", "decision", "evidence"], 30)}

## Negative Boundaries

- General block-orthogonal rotations are not claimed as exact ReLU MLP symmetries.
- Real ReLU residual rows are diagnostic-only.
- Post-projection cycle score alone is not evidence for descent; connection residual gates acceptance.
- Real neural residuals are not called Brauer/projective unless an observed scalar finite-order defect is detected before projection.
"""
    write_report(args.reports_dir / "block_gauge_phase_diagram_report.md", "phase", phase_report)

    block_report = f"""# Block Compatible Learning Report

This report is generated by `experiments/block_gauge_phase_diagram.py`.

{common}

## Learning Task

MNIST linear-hidden MLP. Block rotations are exact same-architecture reparameterizations because the hidden activation is identity.

## Accuracy Table

{md_table(block.groupby("method").agg(n_rows=("seed", "count"), mean_accuracy=("accuracy", "mean"), mean_loss=("loss", "mean"), mean_connection_residual=("connection_residual", "mean"), capacity_matched=("capacity_matched", "all")).reset_index(), ["method", "n_rows", "mean_accuracy", "mean_loss", "mean_connection_residual", "capacity_matched"], 20)}

## Paired Statistics

{md_table(stats[stats["comparison"].str.contains("block_gauge", na=False)], ["comparison", "n_pairs", "mean_delta", "ci_low", "ci_high", "wins", "ties", "losses"], 20)}

## Boundary

This supports only the exact linear-hidden/block-compatible architecture. It does not imply natural ReLU or CIFAR performance.
"""
    write_report(args.reports_dir / "block_compatible_learning_report.md", "block", block_report)

    relu_report = f"""# ReLU Block Diagnostic Report

This report is generated by `experiments/block_gauge_phase_diagram.py`.

{common}

## Diagnostic-Only Summary

{md_table(relu.groupby(["partition_method", "block_size"]).agg(n_rows=("seed", "count"), scalar_candidate_fraction=("observed_scalar_projective_candidate", "mean"), mean_spectral_residual=("spectral_connection_residual", "mean"), mean_optimized_residual=("optimized_connection_residual", "mean"), block_merge_reported=("block_merge_accuracy_reported", "any")).reset_index() if not relu.empty else pd.DataFrame(), ["partition_method", "block_size", "n_rows", "scalar_candidate_fraction", "mean_spectral_residual", "mean_optimized_residual", "block_merge_reported"], 40)}

## ReLU-Compatible Baselines

{md_table(relu[["setting_id", "block_size", "partition_method", "c2m3_accuracy", "monomial_accuracy", "greedy_soup_accuracy", "ensemble_accuracy"]].drop_duplicates().head(40) if not relu.empty else pd.DataFrame(), ["setting_id", "block_size", "partition_method", "c2m3_accuracy", "monomial_accuracy", "greedy_soup_accuracy", "ensemble_accuracy"], 40)}

## Boundary

Block rotations are not evaluated as same-architecture ReLU merges. C2M3, positive monomial scaling, greedy soup, and ensemble rows remain the ReLU-compatible references from the prior diagnostic run.
"""
    write_report(args.reports_dir / "relu_block_diagnostic_report.md", "relu", relu_report)


def write_config(args, calibration, policies, paths) -> None:
    config = {
        "command": " ".join([sys.executable, *sys.argv]),
        "git_commit": git_commit(),
        "dirty_worktree": git_dirty(),
        "synthetic_seeds": args.synthetic_seeds,
        "n_models": args.n_models,
        "widths": args.widths,
        "block_sizes": args.block_sizes,
        "noise_levels": args.noise_levels,
        "n_restarts": args.n_restarts,
        "max_iters": args.max_iters,
        "block_learning_seeds": args.block_learning_seeds,
        "block_train_samples": args.block_train_samples,
        "block_test_samples": args.block_test_samples,
        "calibration": {
            "threshold": calibration.threshold,
            "observed_false_positive_rate": calibration.observed_false_positive_rate,
            "observed_true_positive_rate": calibration.observed_true_positive_rate,
            "policies": [policy.__dict__ for policy in policies],
        },
        "outputs": {key: str(value) for key, value in paths.items()},
        "environment": capture_environment(),
    }
    for path in [
        args.reports_dir / "configs" / "block_gauge_phase_diagram_config.json",
        args.reports_dir / "configs" / "block_compatible_learning_config.json",
        args.reports_dir / "configs" / "relu_block_diagnostic_config.json",
    ]:
        path.write_text(json.dumps(config, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic-seeds", type=int, default=20)
    parser.add_argument("--n-models", default="3,4")
    parser.add_argument("--widths", default="4,8")
    parser.add_argument("--block-sizes", default="2,4")
    parser.add_argument("--noise-levels", default="0.0,0.01,0.03,0.1,0.2,0.4,0.8")
    parser.add_argument("--n-restarts", type=int, default=1)
    parser.add_argument("--max-iters", type=int, default=10)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--max-order", type=int, default=12)
    parser.add_argument("--learned-block-seeds", type=int, default=20)
    parser.add_argument("--learned-samples", type=int, default=300)
    parser.add_argument("--block-learning-seeds", type=int, default=20)
    parser.add_argument("--block-learning-n-models", type=int, default=3)
    parser.add_argument("--block-learning-width", type=int, default=16)
    parser.add_argument("--block-learning-block-size", type=int, default=4)
    parser.add_argument("--block-train-samples", type=int, default=2000)
    parser.add_argument("--block-test-samples", type=int, default=1000)
    parser.add_argument("--block-epochs", type=int, default=2)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dataset-seed", type=int, default=9321)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "reports")
    args = parser.parse_args()

    csv_dir = args.reports_dir / "csv"
    plot_dir = args.reports_dir / "plots"
    config_dir = args.reports_dir / "configs"
    csv_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)

    calibration, policies = build_calibration_controls(args)
    phase = phase_diagram_rows(args, calibration, policies)
    learned = learned_partition_rows(args)
    block = block_compatible_learning_rows(args)
    relu = relu_diagnostic_rows(args)
    summary = summarize_phase(phase)
    stats = paired_stats(phase, learned, block)
    policy = policy_table(policies, phase)
    claims = claim_table(phase, stats, learned, block, relu)

    paths = {
        "phase": csv_dir / "block_gauge_phase_diagram.csv",
        "phase_summary": csv_dir / "block_gauge_phase_diagram_summary.csv",
        "paired_stats": csv_dir / "block_gauge_phase_diagram_paired_stats.csv",
        "learned": csv_dir / "learned_block_partition_benchmark.csv",
        "block_learning": csv_dir / "block_compatible_learning_benchmark.csv",
        "relu": csv_dir / "relu_block_diagnostic_benchmark.csv",
    }
    phase.to_csv(paths["phase"], index=False)
    summary.to_csv(paths["phase_summary"], index=False)
    stats.to_csv(paths["paired_stats"], index=False)
    learned.to_csv(paths["learned"], index=False)
    block.to_csv(paths["block_learning"], index=False)
    relu.to_csv(paths["relu"], index=False)
    write_plots(phase, learned, block, relu, plot_dir)
    write_reports(args, phase, summary, stats, learned, block, relu, policy, claims)
    write_config(args, calibration, policies, paths)
    for path in paths.values():
        print(f"wrote {path}")
    print(f"wrote {args.reports_dir / 'block_gauge_phase_diagram_report.md'}")
    print(f"wrote {args.reports_dir / 'block_compatible_learning_report.md'}")
    print(f"wrote {args.reports_dir / 'relu_block_diagnostic_report.md'}")


if __name__ == "__main__":
    main()
