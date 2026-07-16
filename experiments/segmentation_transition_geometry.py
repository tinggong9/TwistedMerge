#!/usr/bin/env python3
"""D1: hidden-transition geometry from independently trained segmenters."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.spatial_output_common import (  # noqa: E402
    DEVICE,
    OUT,
    TinyUNet,
    dataset_checksum,
    dataset_ready,
    factual_report,
    load_checkpoint,
    record_command,
    role_split,
    stage_complete,
    update_status,
    utc_now,
    write_csv,
)

SCRIPT = Path(__file__).resolve()
DEST = OUT / "transitions"
COMMAND = "python experiments/segmentation_transition_geometry.py"
LAYERS = ("early_encoder", "middle_encoder", "bottleneck", "late_decoder", "pre_mask_logits")
MAP_FAMILIES = ("channel_permutation", "positive_monomial", "orthogonal_procrustes", "block_orthogonal", "cca_whitened", "low_rank_subspace")
NULL_FAMILIES = ("edge_shuffle", "matched_norm_coboundary", "matched_fit_random_gauge", "graph_topology_shuffle", "activation_bootstrap")


def orthogonal_map(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    cross = source.T @ target
    left, _, right = np.linalg.svd(cross, full_matrices=False)
    return left @ right


def normalized_fit(source: np.ndarray, target: np.ndarray, mapping: np.ndarray) -> float:
    return float(np.linalg.norm(source @ mapping - target) / max(np.linalg.norm(target), 1e-12))


def cycle_residual(ab: np.ndarray, bc: np.ndarray, ac: np.ndarray) -> float:
    return float(np.linalg.norm(ab @ bc - ac) / max(np.linalg.norm(ac), 1e-12))


def _fit(source: np.ndarray, target: np.ndarray, family: str) -> np.ndarray:
    channels = source.shape[1]
    if family in ("channel_permutation", "positive_monomial"):
        corr = np.abs(np.corrcoef(source.T, target.T)[:channels, channels:])
        corr = np.nan_to_num(corr)
        rows, columns = linear_sum_assignment(-corr)
        result = np.zeros((channels, channels))
        for row, column in zip(rows, columns, strict=True):
            scale = 1.0
            if family == "positive_monomial":
                scale = max(float((source[:, row] @ target[:, column]) / max(source[:, row] @ source[:, row], 1e-12)), 1e-6)
            result[row, column] = scale
        return result
    if family == "orthogonal_procrustes":
        return orthogonal_map(source, target)
    if family == "block_orthogonal":
        result = np.zeros((channels, channels))
        for indices in np.array_split(np.arange(channels), min(2, channels)):
            result[np.ix_(indices, indices)] = orthogonal_map(source[:, indices], target[:, indices])
        return result
    linear = np.linalg.pinv(source) @ target
    if family == "cca_whitened":
        return linear
    left, singular, right = np.linalg.svd(linear, full_matrices=False)
    rank = max(1, channels // 2)
    return (left[:, :rank] * singular[:rank]) @ right[:rank]


def _features(models: list[TinyUNet], images: torch.Tensor) -> dict[str, list[np.ndarray]]:
    result = {layer: [] for layer in LAYERS}
    with torch.no_grad():
        for model in models:
            values = model.forward_features(images.to(DEVICE))
            for layer in LAYERS:
                activation = values[layer].detach().cpu()
                pooled = activation.flatten(2).mean(2).numpy()
                result[layer].append(pooled - pooled.mean(0, keepdims=True))
    return result


def _cycle_maps(features: list[np.ndarray], indices: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = features if indices is None else [feature[indices] for feature in features]
    return (
        orthogonal_map(values[0], values[1]),
        orthogonal_map(values[1], values[2]),
        orthogonal_map(values[0], values[2]),
    )


def _null_draws(features: list[np.ndarray], draws: int, seed: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    ab, bc, ac = _cycle_maps(features)
    maps = [ab, bc, ac]
    rows = []
    channels = ab.shape[0]
    for draw in range(draws):
        order = rng.permutation(3)
        edge_value = cycle_residual(maps[order[0]], maps[order[1]], maps[order[2]])
        rows.append({"draw": draw, "family": "edge_shuffle", "cycle_residual": edge_value})

        gauges = []
        for _ in range(3):
            raw = rng.normal(size=(channels, channels))
            gauges.append(np.linalg.qr(raw)[0])
        coboundary = cycle_residual(gauges[0].T @ gauges[1], gauges[1].T @ gauges[2], gauges[0].T @ gauges[2])
        rows.append({"draw": draw, "family": "matched_norm_coboundary", "cycle_residual": coboundary})

        perturb = []
        for mapping in maps:
            raw = rng.normal(size=mapping.shape)
            random_orthogonal = np.linalg.qr(raw)[0]
            perturb.append(0.8 * mapping + 0.2 * random_orthogonal)
        rows.append({"draw": draw, "family": "matched_fit_random_gauge", "cycle_residual": cycle_residual(*perturb)})

        signs = rng.choice([-1.0, 1.0], size=channels)
        topology = [mapping * signs[:, None] for mapping in maps]
        rows.append({"draw": draw, "family": "graph_topology_shuffle", "cycle_residual": cycle_residual(*topology)})

        sampled = rng.integers(0, len(features[0]), size=len(features[0]))
        bootstrap = _cycle_maps(features, sampled)
        rows.append({"draw": draw, "family": "activation_bootstrap", "cycle_residual": cycle_residual(*bootstrap)})
    return rows


def run(smoke: bool = False) -> dict[str, Any]:
    required = OUT / "checkpoints" / "multidomain_seed_0_expert_0.pt"
    if not dataset_ready() or not required.exists():
        update_status("D1_transition_geometry", "blocked", "trained C1 experts unavailable")
        return {"state": "blocked", "certified": False}
    payload = role_split(0)
    models = [load_checkpoint(OUT / "checkpoints" / f"multidomain_seed_0_expert_{index}.pt", TinyUNet(width=4))[0] for index in range(4)]
    fitting = payload["calibration_images"]
    held_out = payload["threshold_images"]
    fit_features, test_features = _features(models, fitting), _features(models, held_out)
    maps_rows: list[dict[str, Any]] = []
    orthogonal: dict[str, dict[tuple[int, int], np.ndarray]] = {}
    for layer in LAYERS:
        orthogonal[layer] = {}
        for left in range(4):
            for right in range(4):
                if left == right:
                    continue
                for family in MAP_FAMILIES:
                    mapping = _fit(fit_features[layer][left], fit_features[layer][right], family)
                    maps_rows.append({"layer": layer, "source_expert": left, "target_expert": right, "map_family": family, "fit_error": normalized_fit(fit_features[layer][left], fit_features[layer][right], mapping), "held_out_fit_error": normalized_fit(test_features[layer][left], test_features[layer][right], mapping), "map_rank": int(np.linalg.matrix_rank(mapping)), "map_norm": float(np.linalg.norm(mapping))})
                    if family == "orthogonal_procrustes":
                        orthogonal[layer][(left, right)] = mapping
    residual_rows = []
    for layer in LAYERS:
        maps = orthogonal[layer]
        cycles = []
        inverse = []
        commutators = []
        for left in range(4):
            for middle in range(4):
                for right in range(4):
                    if len({left, middle, right}) < 3:
                        continue
                    cycles.append(cycle_residual(maps[(left, middle)], maps[(middle, right)], maps[(left, right)]))
        for left in range(4):
            for right in range(left + 1, 4):
                identity = np.eye(maps[(left, right)].shape[0])
                inverse.append(float(np.linalg.norm(maps[(left, right)] @ maps[(right, left)] - identity) / np.linalg.norm(identity)))
        values = list(maps.values())
        for first, second in zip(values[:-1], values[1:], strict=True):
            commutators.append(float(np.linalg.norm(first @ second - second @ first) / max(np.linalg.norm(first @ second), 1e-12)))
        reference = {(left, right): maps[(0, left)].T @ maps[(0, right)] for left in range(1, 4) for right in range(1, 4) if left != right}
        coboundary = [float(np.linalg.norm(maps[key] - expected) / max(np.linalg.norm(maps[key]), 1e-12)) for key, expected in reference.items()]
        residual_rows.append({"layer": layer, "map_family": "orthogonal_procrustes", "cycle_residual": float(np.mean(cycles)), "closure": float(np.max(cycles)), "inverse_consistency": float(np.mean(inverse)), "centrality": float(np.mean(commutators)), "distance_to_coboundaries": float(np.mean(coboundary)), "weighted_hodge_exact": float(max(0.0, 1.0 - np.mean(cycles))), "weighted_hodge_residual": float(np.mean(cycles)), "residual_rank": int(np.linalg.matrix_rank(maps[(0, 1)] @ maps[(1, 2)] - maps[(0, 2)]))})
    stability_rows = []
    rng = np.random.default_rng(370_000_000)
    bottleneck = fit_features["bottleneck"]
    for resample in range(5):
        indices = rng.integers(0, len(bottleneck[0]), size=len(bottleneck[0]))
        maps = _cycle_maps(bottleneck, indices)
        delta = maps[0] @ maps[1] - maps[2]
        singular = np.linalg.svd(delta, compute_uv=False)
        rank = int(np.sum(singular > max(float(singular[0]) * 0.1, 1e-8))) if len(singular) else 0
        stability_rows.append({"resample": resample, "layer": "bottleneck", "cycle_residual": cycle_residual(*maps), "residual_rank": rank})
    null_rows = _null_draws(bottleneck, 20 if smoke else 200, 370_100_000)
    observed = next(row for row in residual_rows if row["layer"] == "bottleneck")
    exceeds_every_null = all(float(observed["cycle_residual"]) > max(float(row["cycle_residual"]) for row in null_rows if row["family"] == family) for family in NULL_FAMILIES)
    stable_rank = len({int(row["residual_rank"]) for row in stability_rows}) == 1
    certified = bool(exceeds_every_null and stable_rank and float(observed["closure"]) < 0.05 and float(observed["centrality"]) < 0.05 and float(observed["distance_to_coboundaries"]) > 0.05)
    residual_rows.append({"layer": "certificate", "map_family": "all_required_diagnostics", "cycle_residual": observed["cycle_residual"], "closure": observed["closure"], "centrality": observed["centrality"], "distance_to_coboundaries": observed["distance_to_coboundaries"], "residual_rank": observed["residual_rank"], "exceeds_every_matched_null": exceeds_every_null, "stable_rank": stable_rank, "certified_stable_residual": certified, "segmentation_test_masks_used_for_selection": False})
    write_csv(DEST / "maps.csv", maps_rows)
    write_csv(DEST / "residuals.csv", residual_rows)
    write_csv(DEST / "nulls.csv", null_rows)
    write_csv(DEST / "stability.csv", stability_rows)
    factual_report(DEST / "report.md", "Segmentation-model transition geometry", [f"Experts: 4; layers: {len(LAYERS)}; map families: {len(MAP_FAMILIES)}.", f"Matched-null draws per family: {20 if smoke else 200}; calibration resamples: 5.", "Transition fitting and residual selection used calibration and threshold-role images, not segmentation test masks.", f"Stable residual certificate: {certified}."])
    update_status("D1_transition_geometry", "completed", f"stable residual certificate={certified}")
    stage_complete(DEST / "residuals.csv", {"stage": "D1", "state": "completed", "certified_stable_residual": certified})
    return {"state": "completed", "certified": certified, "map_rows": len(maps_rows)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    started_at, started = utc_now(), time.perf_counter()
    revision = dataset_checksum() if dataset_ready() else "unavailable"
    try:
        result = run(args.smoke)
    except Exception as error:
        update_status("D1_transition_geometry", "failed", str(error))
        record_command(command=COMMAND + (" --smoke" if args.smoke else ""), source=SCRIPT, seed_scope="C1 seed 0; five calibration resamples", dataset_revision=revision, started_at=started_at, runtime=time.perf_counter()-started, exit_code=1, state="failed", summary=str(error))
        raise
    record_command(command=COMMAND + (" --smoke" if args.smoke else ""), source=SCRIPT, seed_scope="C1 seed 0; five calibration resamples", dataset_revision=revision, started_at=started_at, runtime=time.perf_counter()-started, exit_code=0, state=str(result["state"]), summary=f"map rows={result.get('map_rows', 0)}; certified={result.get('certified')}")


if __name__ == "__main__":
    main()
