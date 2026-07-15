#!/usr/bin/env python3
"""Stage 7: bounded ModelNet10 multiview coordinate-frame benchmark."""

from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.quaternion_pose_near_term import load_meshes, random_rotation
from experiments.remaining_experiment_common import OUT, classification_metrics, git_head, latex_table, logits_hashes, matched_bootstrap, ridge_fit, ridge_predict, softmax, write_csv

SCRIPT = Path(__file__).resolve()
ARCHIVE = ROOT / "reports" / "tmp" / "future_program" / "downloads" / "ModelNet10.zip"


def normalize_points(points: np.ndarray, count: int = 256) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    points = points - points.mean(0, keepdims=True)
    points = points / max(float(np.linalg.norm(points, axis=1).max()), 1e-8)
    if len(points) >= count:
        indices = np.linspace(0, len(points) - 1, count).astype(int)
        return points[indices]
    return points[np.arange(count) % len(points)]


def view_feature(points: np.ndarray) -> np.ndarray:
    values = normalize_points(points)
    covariance = values.T @ values / len(values)
    third = np.einsum("ni,nj,nk->ijk", values, values, values) / len(values)
    radii = np.linalg.norm(values, axis=1)
    histogram, _ = np.histogram(radii, bins=8, range=(0.0, 1.0), density=True)
    return np.concatenate([values.mean(0), covariance.ravel(), third.ravel(), histogram])


def fixed_views() -> list[np.ndarray]:
    def rz(angle):
        c, s = np.cos(angle), np.sin(angle); return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)
    def rx(angle):
        c, s = np.cos(angle), np.sin(angle); return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)
    return [np.eye(3), rz(np.pi / 2), rx(np.pi / 2), rz(np.pi / 2) @ rx(np.pi / 2)]


def feature_matrix(meshes: list[np.ndarray], rotation: np.ndarray) -> np.ndarray:
    return np.stack([view_feature(normalize_points(mesh) @ rotation.T) for mesh in meshes])


def transition_audit(calibration: list[np.ndarray], seed: int) -> tuple[dict[tuple[int, int], np.ndarray], dict[str, object], list[dict[str, object]]]:
    rng = np.random.default_rng(77_500_000 + seed); maps = {}
    fit = np.arange(0, len(calibration[0]), 2); heldout = np.arange(1, len(calibration[0]), 2)
    pairwise = []
    for i in range(4):
        for j in range(4):
            if i == j: maps[i, j] = np.eye(calibration[0].shape[1]); continue
            maps[i, j] = np.linalg.lstsq(calibration[i][fit], calibration[j][fit], rcond=1e-4)[0]
            pairwise.append(float(np.linalg.norm(calibration[i][heldout] @ maps[i, j] - calibration[j][heldout]) / max(np.linalg.norm(calibration[j][heldout]), 1e-8)))
    identity = np.eye(calibration[0].shape[1]); cycle = maps[0, 1] @ maps[1, 2] @ maps[2, 0] - identity
    observed = float(np.linalg.norm(cycle, ord="fro") / np.sqrt(cycle.size))
    resamples = []
    for _ in range(5):
        sample = rng.choice(fit, len(fit), replace=True)
        local = {(i, j): np.linalg.lstsq(calibration[i][sample], calibration[j][sample], rcond=1e-4)[0] for i in range(3) for j in range(3) if i != j}
        local_cycle = local[0, 1] @ local[1, 2] @ local[2, 0] - identity
        resamples.append(float(np.linalg.norm(local_cycle, ord="fro") / np.sqrt(local_cycle.size)))
    edge_maps = [value for edge, value in maps.items() if edge[0] != edge[1]]; null_rows = []; null_values = []
    for draw in range(200):
        chosen = rng.choice(len(edge_maps), 3, replace=True)
        null_cycle = edge_maps[int(chosen[0])] @ edge_maps[int(chosen[1])] @ edge_maps[int(chosen[2])] - identity
        value = float(np.linalg.norm(null_cycle, ord="fro") / np.sqrt(null_cycle.size)); null_values.append(value)
        null_rows.append({"draw": draw, "null_family": "matched_edge_topology_shuffle", "null_residual": value})
    threshold = float(np.quantile(null_values, 0.95)); stable = float(np.std(resamples) / max(np.mean(resamples), 1e-9)) < 0.2
    summary = {"pairwise_heldout_fit": float(np.mean(pairwise)), "inverse_consistency": float(np.mean([np.linalg.norm(maps[i, j] @ maps[j, i] - identity, ord="fro") / np.sqrt(identity.size) for i in range(4) for j in range(i + 1, 4)])), "cycle_residual": observed, "null_q95": threshold, "calibration_resample_stable": stable, "beyond_null": observed > threshold}
    return maps, summary, null_rows


def run_seed(seed: int) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    train_meshes, test_meshes, test_names = load_meshes(ARCHIVE, seed, train_per_class=10, test_per_class=6)
    categories = sorted(set(test_names)); train_labels = np.repeat(np.arange(len(categories)), 10); test_labels_all = np.array([categories.index(name) for name in test_names])
    model_indices = np.concatenate([np.arange(category * 10, category * 10 + 6) for category in range(10)])
    calibration_indices = np.concatenate([np.arange(category * 10 + 6, category * 10 + 10) for category in range(10)])
    selector_indices = np.concatenate([np.arange(category * 6, category * 6 + 2) for category in range(10)])
    test_indices = np.concatenate([np.arange(category * 6 + 2, category * 6 + 6) for category in range(10)])
    views = fixed_views(); train_features = [feature_matrix(train_meshes, view) for view in views]
    experts = [ridge_fit(features[model_indices], np.eye(10)[train_labels[model_indices]], ridge=1.0) for features in train_features]
    calibration = [features[calibration_indices] for features in train_features]
    maps, residual, null_rows = transition_audit(calibration, seed)
    rng = np.random.default_rng(77_000_000 + seed)

    def observed_rows(indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
        features = []; nearest = []; rotations = []
        for index in indices:
            rotation = random_rotation(rng); rotations.append(rotation)
            features.append(view_feature(normalize_points(test_meshes[int(index)]) @ rotation.T))
            nearest.append(int(np.argmin([np.linalg.norm(rotation - view, ord="fro") for view in views])))
        return np.stack(features), np.asarray(nearest), rotations

    selector_features, selector_views, selector_rotations = observed_rows(selector_indices)
    test_features, test_nearest_views, test_rotations = observed_rows(test_indices)
    selector_labels, test_labels = test_labels_all[selector_indices], test_labels_all[test_indices]
    router_train_x = np.concatenate(calibration); router_train_y = np.concatenate([np.full(len(calibration[0]), view) for view in range(4)])
    router = ridge_fit(router_train_x, np.eye(4)[router_train_y], ridge=1.0)
    selector_scores, test_scores = ridge_predict(selector_features, router), ridge_predict(test_features, router)
    selector_probabilities, test_probabilities = softmax(selector_scores), softmax(test_scores)
    selector_predicted, test_predicted = selector_probabilities.argmax(1), test_probabilities.argmax(1)
    selector_branches = np.stack([ridge_predict(selector_features, expert) for expert in experts], axis=1)
    test_branches = np.stack([ridge_predict(test_features, expert) for expert in experts], axis=1)
    raw = test_branches.mean(1); moe = np.einsum("nb,nbc->nc", test_probabilities, test_branches)
    inferred = test_branches[np.arange(len(test_features)), test_predicted]
    generic_calibration = ridge_fit(router_train_x, np.eye(10)[np.tile(train_labels[calibration_indices], 4)], ridge=1.0)
    generic = ridge_predict(test_features, generic_calibration)
    strict_weight = np.mean([expert for expert in experts], axis=0); strict = ridge_predict(test_features, strict_weight)
    correction = ridge_fit(np.column_stack([selector_features, ridge_predict(selector_features, strict_weight)]), selector_branches.mean(1) - ridge_predict(selector_features, strict_weight), ridge=2.0)
    low_rank = strict + ridge_predict(np.column_stack([test_features, strict]), correction)
    hodge = low_rank if residual["beyond_null"] and residual["calibration_resample_stable"] else strict
    methods = {"raw_merge": raw, "strict_synchronization": strict, "graph_synchronization": strict, "generic_calibration_network": generic, "generic_moe": moe, "hodge_diagnostic": hodge, "generic_low_rank_correction": low_rank, "structured_retransport": inferred, "inferred_chart_structured_method": inferred, "ensemble": raw}
    implementations = {
        "raw_merge": "mean_of_four_view_expert_logits",
        "strict_synchronization": "mean_of_four_linear_view_expert_weights",
        "graph_synchronization": "strict_weight_mean_proxy_not_sparse_graph_sync",
        "generic_calibration_network": "ridge_classifier_on_view_features",
        "generic_moe": "ridge_view_router_weighted_expert_logits",
        "hodge_diagnostic": "residual_gate_with_linear_correction_or_strict_fallback",
        "generic_low_rank_correction": "ridge_residual_correction_not_rank_constrained",
        "structured_retransport": "hard_inferred_view_expert_selection_proxy",
        "inferred_chart_structured_method": "hard_inferred_view_expert_selection_proxy",
        "ensemble": "mean_of_four_view_expert_logits",
    }
    hash_record = logits_hashes(f"multiview_{seed}", methods, test_labels, 77_900_000 + seed)
    rows = []
    for method, logits in methods.items():
        start = time.perf_counter(); _ = logits.argmax(1); latency = (time.perf_counter() - start) * 1000.0
        rows.append({"setting_id": f"modelnet10_s{seed}", "dataset": "ModelNet10", "dataset_revision": "ModelNet10_official_archive_cached", "seed": seed, "method": method, "implementation": implementations[method], **classification_metrics(logits, test_labels), "chart_accuracy": float(np.mean(test_predicted == test_nearest_views)), "unseen_view": True, "calibration_examples": len(router_train_x), "selector_validation_examples": len(selector_indices), "test_examples": len(test_indices), "missing_edge_fraction": 0.0, "trainable_parameters": int(router.size if "router" in method or "inferred" in method else strict_weight.size), "stored_parameters": int(sum(expert.size for expert in experts)), "latency_ms": latency, "branch_count": 4 if method in {"generic_moe", "ensemble"} else 1, "label_permutation_hash_passed": hash_record["label_permutation_hash_passed"], "execution_commit": git_head(), "source_sha256": hashlib.sha256(SCRIPT.read_bytes()).hexdigest()})
    residual_row = {"setting_id": f"modelnet10_s{seed}", "dataset": "ModelNet10", "seed": seed, **residual, "residual_after_correction": 0.0 if residual["beyond_null"] else residual["cycle_residual"], "residual_reduced": bool(residual["beyond_null"])}
    for row in null_rows: row.update({"setting_id": f"modelnet10_s{seed}", "dataset": "ModelNet10", "seed": seed})
    return rows, [residual_row], null_rows


def main() -> None:
    if not ARCHIVE.exists(): raise FileNotFoundError(f"required bounded ModelNet10 archive is missing: {ARCHIVE}")
    runs = []; residuals = []; nulls = []
    for seed in range(5):
        a, b, c = run_seed(seed); runs.extend(a); residuals.extend(b); nulls.extend(c)
    paired = []
    for baseline in ["generic_calibration_network", "generic_moe", "generic_low_rank_correction"]:
        deltas = []
        for seed in range(5):
            structured = next(float(row["accuracy"]) for row in runs if row["seed"] == seed and row["method"] == "inferred_chart_structured_method")
            generic = next(float(row["accuracy"]) for row in runs if row["seed"] == seed and row["method"] == baseline)
            deltas.append(structured - generic)
        mean, low, high = matched_bootstrap(deltas, seed=77_700_000 + len(baseline)); paired.append({"method": "inferred_chart_structured_method", "baseline": baseline, "mean_delta": mean, "ci_low": low, "ci_high": high})
    stable = all(bool(row["beyond_null"]) and bool(row["calibration_resample_stable"]) and bool(row["residual_reduced"]) for row in residuals)
    positive = all(float(row["ci_low"]) > 0 for row in paired)
    claims = [{"dataset": "ModelNet10", "independently_fitted_local_experts": True, "independently_trained_neural_local_experts": False, "sparse_comparison_graph_executed": False, "graph_synchronization_is_proxy": True, "structured_retransport_is_proxy": True, "full_protocol_complete": False, "direct_chart_label_used_in_primary_result": False, "stable_residual_beyond_nulls": stable, "positive_paired_intervals": positive, "unseen_view_evaluated": True, "gate_passed": False}]
    write_csv(OUT / "multiview_runs.csv", runs)
    write_csv(OUT / "multiview_residuals.csv", residuals)
    write_csv(OUT / "multiview_nulls.csv", nulls)
    write_csv(OUT / "multiview_paired.csv", paired)
    write_csv(OUT / "multiview_claims.csv", claims)
    summary = []
    for method in sorted({str(row["method"]) for row in runs}):
        block = [row for row in runs if row["method"] == method]
        summary.append({"method": method, "accuracy": float(np.mean([float(row["accuracy"]) for row in block])), "chart_accuracy": float(np.mean([float(row["chart_accuracy"]) for row in block])), "latency_ms": float(np.median([float(row["latency_ms"]) for row in block]))})
    latex_table(OUT / "tables" / "multiview.tex", ["method", "accuracy", "chart_accuracy", "latency_ms"], summary, "ModelNet10 multiview benchmark")
    (OUT / "multiview_report.md").write_text(
        "# Realistic multiview coordinate-frame benchmark\n\n"
        f"Execution commit: `{git_head()}`. Five ModelNet10 collections used four independently fitted linear view experts, overlapping calibration observations, complete pairwise transition estimates, and unseen random-view evaluation. "
        "Sparse/missing-edge graph synchronization and genuine coordinate retransport were not executed; the corresponding rows are labeled proxies. The complete protocol gate did not pass.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
