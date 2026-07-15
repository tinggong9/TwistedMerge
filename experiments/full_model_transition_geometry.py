#!/usr/bin/env python3
"""Stage 3: hidden-representation transition geometry on pretrained vision features."""

from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.remaining_experiment_common import (
    OUT,
    TMP,
    classification_metrics,
    git_head,
    latex_table,
    logits_hashes,
    matched_bootstrap,
    ridge_fit,
    ridge_predict,
    write_csv,
)

SCRIPT = Path(__file__).resolve()
FUTURE_FEATURES = ROOT / "reports" / "tmp" / "future_program" / "features"


def orthogonal_map(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    left, _, right = np.linalg.svd(source.T @ target, full_matrices=False)
    return left @ right


def whitened_map(source: np.ndarray, target: np.ndarray, ridge: float = 1e-4) -> np.ndarray:
    source_cov = source.T @ source / len(source) + np.eye(source.shape[1]) * ridge
    target_cov = target.T @ target / len(target) + np.eye(target.shape[1]) * ridge
    source_values, source_vectors = np.linalg.eigh(source_cov)
    target_values, target_vectors = np.linalg.eigh(target_cov)
    source_white = source_vectors @ np.diag(1.0 / np.sqrt(np.maximum(source_values, ridge))) @ source_vectors.T
    target_color = target_vectors @ np.diag(np.sqrt(np.maximum(target_values, ridge))) @ target_vectors.T
    return source_white @ target_color


def permutation_map(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    from scipy.optimize import linear_sum_assignment

    source_centered = source - source.mean(0); target_centered = target - target.mean(0)
    correlation = source_centered.T @ target_centered
    rows, columns = linear_sum_assignment(-np.abs(correlation))
    matrix = np.zeros((source.shape[1], target.shape[1]))
    signs = np.sign(correlation[rows, columns]); signs[signs == 0] = 1
    matrix[rows, columns] = signs
    return matrix


def low_rank_map(source: np.ndarray, target: np.ndarray, rank: int = 8) -> np.ndarray:
    full = np.linalg.lstsq(source, target, rcond=1e-5)[0]
    left, values, right = np.linalg.svd(full - np.eye(full.shape[0]), full_matrices=False)
    return np.eye(full.shape[0]) + (left[:, :rank] * values[:rank]) @ right[:rank]


def block_orthogonal_map(source: np.ndarray, target: np.ndarray, blocks: int = 4) -> np.ndarray:
    dimension = source.shape[1]; result = np.zeros((dimension, dimension))
    for indices in np.array_split(np.arange(dimension), blocks):
        result[np.ix_(indices, indices)] = orthogonal_map(source[:, indices], target[:, indices])
    return result


def fit_transition(source: np.ndarray, target: np.ndarray, family: str) -> np.ndarray:
    if family == "permutation_channel_matching": return permutation_map(source, target)
    if family == "orthogonal_procrustes": return orthogonal_map(source, target)
    if family == "cca_whitened": return whitened_map(source, target)
    if family == "block_orthogonal": return block_orthogonal_map(source, target)
    if family == "low_rank_subspace": return low_rank_map(source, target)
    raise ValueError(family)


def load_feature_cache(architecture: str) -> dict[str, np.ndarray]:
    path = FUTURE_FEATURES / f"x1_{architecture}_CIFAR10.npz"
    if not path.exists():
        from experiments.broader_vision_extended import cached_features
        values, _ = cached_features(architecture, "CIFAR10")
        return values
    data = np.load(path, allow_pickle=False)
    return {key: data[key] for key in data.files}


def compress_features(values: dict[str, np.ndarray], dimension: int = 64) -> dict[str, np.ndarray]:
    train = values["train_x"].astype(np.float64)
    mean = train.mean(0)
    _, _, right = np.linalg.svd(train[: min(2000, len(train))] - mean, full_matrices=False)
    projection = right[: min(dimension, right.shape[0])].T
    result = {}
    for key, array in values.items():
        result[key] = (array - mean) @ projection if key.endswith("_x") or key.endswith("_shift") else array
    return result


def local_representations(features: np.ndarray, labels: np.ndarray, collection: int, architecture: str) -> tuple[list[np.ndarray], list[np.ndarray]]:
    rng = np.random.default_rng(33_000_000 + collection + (0 if architecture == "resnet18" else 1000))
    dimension = features.shape[1]
    representations, maps = [], []
    for specialist in range(4):
        q, _ = np.linalg.qr(rng.normal(size=(dimension, dimension)))
        classes = np.array([(specialist + 2 * collection + offset) % 10 for offset in range(5)])
        class_mean = features[np.isin(labels, classes)].mean(0) - features.mean(0)
        direction = class_mean / max(np.linalg.norm(class_mean), 1e-8)
        adapter = q @ np.diag(1.0 + 0.12 * rng.normal(size=dimension))
        adapter += 0.15 * np.outer(direction, np.roll(direction, specialist + 1))
        maps.append(adapter)
        representations.append(np.tanh(features @ adapter))
    return representations, maps


def transition_diagnostics(representations: list[np.ndarray], collection: int, layer: str) -> tuple[list[dict[str, object]], dict[str, object], dict[tuple[int, int], np.ndarray]]:
    rng = np.random.default_rng(33_500_000 + collection + len(layer))
    families = ["permutation_channel_matching", "orthogonal_procrustes", "cca_whitened", "block_orthogonal", "low_rank_subspace"]
    rows = []; family_maps: dict[str, dict[tuple[int, int], np.ndarray]] = {}; errors = {}
    fit = np.arange(300); heldout = np.arange(300, 600)
    for family in families:
        maps = {}; family_error = []
        for i in range(4):
            for j in range(4):
                if i == j: maps[i, j] = np.eye(representations[0].shape[1]); continue
                transition = fit_transition(representations[i][fit], representations[j][fit], family)
                maps[i, j] = transition
                family_error.append(float(np.linalg.norm(representations[i][heldout] @ transition - representations[j][heldout]) / max(np.linalg.norm(representations[j][heldout]), 1e-8)))
        cycle = maps[0, 1] @ maps[1, 2] @ maps[2, 0] - np.eye(representations[0].shape[1])
        inverse = np.mean([np.linalg.norm(maps[i, j] @ maps[j, i] - np.eye(cycle.shape[0]), ord="fro") / np.sqrt(cycle.size) for i in range(4) for j in range(i + 1, 4)])
        residual = float(np.linalg.norm(cycle, ord="fro") / np.sqrt(cycle.size))
        singular = np.linalg.svd(cycle, compute_uv=False); rank = int(np.sum(singular > max(0.05 * singular[0], 1e-7))) if singular[0] > 0 else 0
        errors[family] = float(np.mean(family_error)); family_maps[family] = maps
        rows.append({"collection": collection, "layer": layer, "alignment_family": family, "heldout_pairwise_fit": errors[family], "inverse_consistency": float(inverse), "cycle_residual": residual, "residual_rank": rank})
    selected = min(errors, key=errors.get); maps = family_maps[selected]
    resamples = []
    for _ in range(5):
        sample = rng.choice(300, 300, replace=True)
        local_maps = {(i, j): fit_transition(representations[i][sample], representations[j][sample], selected) for i in range(4) for j in range(4) if i != j}
        cycle = local_maps[0, 1] @ local_maps[1, 2] @ local_maps[2, 0] - np.eye(representations[0].shape[1])
        resamples.append(float(np.linalg.norm(cycle, ord="fro") / np.sqrt(cycle.size)))
    observed = next(row["cycle_residual"] for row in rows if row["alignment_family"] == selected)
    nulls = []
    edge_maps = [value for edge, value in maps.items() if edge[0] != edge[1]]
    for _ in range(200):
        chosen = rng.choice(len(edge_maps), 3, replace=True)
        cycle = edge_maps[int(chosen[0])] @ edge_maps[int(chosen[1])] @ edge_maps[int(chosen[2])] - np.eye(representations[0].shape[1])
        nulls.append(float(np.linalg.norm(cycle, ord="fro") / np.sqrt(cycle.size)))
    threshold = float(np.quantile(nulls, 0.95))
    stable = float(np.std(resamples) / max(np.mean(resamples), 1e-9)) < 0.2
    return rows, {"selected_family": selected, "observed_residual": observed, "null_threshold": threshold, "calibration_resample_stable": stable, "beyond_null": observed > threshold}, maps


def run_collection(architecture: str, collection: int) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    cache_path = FUTURE_FEATURES / f"x1_{architecture}_CIFAR10.npz"
    cache_sha256 = hashlib.sha256(cache_path.read_bytes()).hexdigest()
    values = compress_features(load_feature_cache(architecture))
    train_x, train_y = values["train_x"], values["train_y"].astype(int)
    validation_x, validation_y = values["validation_x"], values["validation_y"].astype(int)
    test_x, test_y = values["test_x"], values["test_y"].astype(int)
    train_repr, _ = local_representations(train_x, train_y, collection, architecture)
    validation_repr, _ = local_representations(validation_x, validation_y, collection, architecture)
    test_repr, _ = local_representations(test_x, test_y, collection, architecture)
    heads = []
    for specialist in range(4):
        classes = np.array([(specialist + 2 * collection + offset) % 10 for offset in range(5)])
        mask = np.isin(train_y, classes)
        heads.append(ridge_fit(train_repr[specialist][mask], np.eye(10)[train_y[mask]], ridge=2.0))
    transition_rows = []
    diagnostic = None
    selected_maps = None
    layer_dimensions = {"block2_feature_subspace": 16, "block3_feature_subspace": 32, "block4_feature_subspace": 48, "penultimate_features": validation_repr[0].shape[1]}
    for layer, dimension in layer_dimensions.items():
        layer_repr = [values[:, : min(dimension, values.shape[1])] for values in validation_repr]
        current_rows, current_diagnostic, current_maps = transition_diagnostics(layer_repr, collection, layer)
        transition_rows.extend(current_rows)
        if layer == "penultimate_features": diagnostic = current_diagnostic; selected_maps = current_maps
    assert diagnostic is not None and selected_maps is not None
    aligned_heads = []
    for specialist in range(4):
        transition = selected_maps[0, specialist]
        aligned_heads.append(np.linalg.pinv(transition) @ heads[specialist][:-1])
    strict_weight = np.mean(aligned_heads, axis=0)
    reference_bias = np.mean([head[-1] for head in heads], axis=0)
    strict_logits = test_repr[0] @ strict_weight + reference_bias
    validation_strict = validation_repr[0] @ strict_weight + reference_bias
    branch_test = np.stack([ridge_predict(test_repr[i], heads[i]) for i in range(4)], axis=1)
    branch_validation = np.stack([ridge_predict(validation_repr[i], heads[i]) for i in range(4)], axis=1)
    ensemble = branch_test.mean(1); validation_ensemble = branch_validation.mean(1)
    router_weights = ridge_fit(validation_x[:300], np.eye(4)[np.argmax(np.max(branch_validation[:300], axis=2), axis=1)], ridge=1.0)
    router_scores = ridge_predict(test_x, router_weights); router_probabilities = np.exp(router_scores - router_scores.max(1, keepdims=True)); router_probabilities /= router_probabilities.sum(1, keepdims=True)
    router_logits = np.einsum("nb,nbc->nc", router_probabilities, branch_test)
    average_head = np.mean([head[:-1] for head in heads], axis=0); weight_average = np.mean(test_repr, axis=0) @ average_head + reference_bias
    greedy_candidates = [strict_logits, weight_average, ensemble]
    validation_candidates = [validation_strict, np.mean(validation_repr, axis=0) @ average_head + reference_bias, validation_ensemble]
    greedy_index = int(np.argmax([classification_metrics(logits, validation_y)["accuracy"] for logits in validation_candidates]))
    greedy = greedy_candidates[greedy_index]
    correction = ridge_fit(validation_strict[:300], validation_ensemble[:300] - validation_strict[:300], ridge=5.0)
    generic_low_rank = strict_logits + ridge_predict(strict_logits, correction)
    rng = np.random.default_rng(33_900_000 + collection)
    ties_weight = np.where(np.sign(np.stack(aligned_heads)).sum(0) == 0, 0.0, strict_weight)
    ties = test_repr[0] @ ties_weight + reference_bias
    mask = rng.random(strict_weight.shape) > 0.2; dare = test_repr[0] @ (strict_weight * mask / 0.8) + reference_bias
    methods = {
        "weight_average": weight_average, "greedy_soup": greedy, "git_rebasin": strict_logits,
        "c2m3": strict_logits, "representation_alignment": strict_logits, "regmean": strict_logits,
        "ties": ties, "dare": dare, "generic_low_rank_merge": generic_low_rank,
        "generic_router": router_logits, "strict_synchronization": strict_logits,
        "hodge_diagnostic_ordinary_fallback": generic_low_rank if diagnostic["beyond_null"] and diagnostic["calibration_resample_stable"] else strict_logits,
        "structured_retransport_certified_only": strict_logits, "ensemble": ensemble,
    }
    implementations = {
        "weight_average": "mean_aligned_feature_head_proxy",
        "greedy_soup": "validation_selected_feature_logit_candidate_proxy",
        "git_rebasin": "permutation_aligned_feature_head_proxy_not_full_model_git_rebasin",
        "c2m3": "strict_feature_synchronization_proxy_not_full_model_c2m3",
        "representation_alignment": "selected_feature_transition_head_merge",
        "regmean": "strict_feature_synchronization_proxy_not_regmean",
        "ties": "elementwise_head_sign_consensus_proxy",
        "dare": "randomly_masked_head_average_proxy",
        "generic_low_rank_merge": "ridge_logit_correction",
        "generic_router": "ridge_router_over_four_feature_heads",
        "strict_synchronization": "selected_feature_transition_head_merge",
        "hodge_diagnostic_ordinary_fallback": "residual_gate_with_ridge_or_strict_fallback",
        "structured_retransport_certified_only": "strict_fallback_no_certified_chart",
        "ensemble": "mean_of_four_feature_head_logits",
    }
    hash_record = logits_hashes(f"full_model_{architecture}_{collection}", methods, test_y, 33_800_000 + collection)
    np.savez_compressed(TMP / "logits" / f"full_model_{architecture}_{collection}.npz", **methods)
    run_rows = []
    for method, logits in methods.items():
        start = time.perf_counter(); _ = logits.argmax(1); latency = (time.perf_counter() - start) * 1000.0
        run_rows.append({"setting_id": f"{architecture}_collection{collection}", "architecture": architecture, "collection": collection, "fine_tune_scope": "synthetic_feature_adapter_over_cached_pretrained_features", "method": method, "implementation": implementations[method], **classification_metrics(logits, test_y), "latency_ms": latency, "trainable_parameters": int(strict_weight.size if method not in {"generic_router", "ensemble"} else router_weights.size), "stored_parameters": int(sum(head.size for head in heads)), "branch_count": 4 if method in {"generic_router", "ensemble"} else 1, "candidate_count": len(greedy_candidates) if method == "greedy_soup" else 1, "feature_cache_sha256": cache_sha256, "label_permutation_hash_passed": hash_record["label_permutation_hash_passed"], "execution_commit": git_head(), "source_sha256": hashlib.sha256(SCRIPT.read_bytes()).hexdigest()})
    null_rows = [{"setting_id": f"{architecture}_collection{collection}", "architecture": architecture, "collection": collection, "null_family": "edge_map_shuffle", "observed_residual": diagnostic["observed_residual"], "null_q95": diagnostic["null_threshold"], "beyond_null": diagnostic["beyond_null"]}]
    stability_rows = [{"setting_id": f"{architecture}_collection{collection}", "architecture": architecture, "collection": collection, **diagnostic}]
    for row in transition_rows: row.update({"architecture": architecture, "setting_id": f"{architecture}_collection{collection}"})
    return run_rows, transition_rows, null_rows, stability_rows


def main() -> None:
    runs = []; transitions = []; nulls = []; stability = []
    architectures = [name for name in ["resnet18", "deit_tiny"] if (FUTURE_FEATURES / f"x1_{name}_CIFAR10.npz").exists()]
    for architecture in architectures:
        for collection in range(5):
            a, b, c, d = run_collection(architecture, collection)
            runs.extend(a); transitions.extend(b); nulls.extend(c); stability.extend(d)
    paired = []; claims = []
    for architecture in architectures:
        block = [row for row in runs if row["architecture"] == architecture]
        structured = "hodge_diagnostic_ordinary_fallback"; baselines = ["strict_synchronization", "generic_low_rank_merge"]
        for baseline in baselines:
            deltas = []
            for collection in range(5):
                left = next(float(row["accuracy"]) for row in block if row["collection"] == collection and row["method"] == structured)
                right = next(float(row["accuracy"]) for row in block if row["collection"] == collection and row["method"] == baseline)
                deltas.append(left - right)
            mean, low, high = matched_bootstrap(deltas, seed=33_700_000 + len(baseline))
            paired.append({"architecture": architecture, "method": structured, "baseline": baseline, "mean_delta": mean, "ci_low": low, "ci_high": high})
        stable_beyond = all(bool(row["beyond_null"]) and bool(row["calibration_resample_stable"]) for row in stability if row["architecture"] == architecture)
        positive_interval = all(float(row["ci_low"]) > 0 for row in paired if row["architecture"] == architecture)
        passed = stable_beyond and positive_interval
        claims.append({"architecture": architecture, "collections": 5, "residual_blocks_finetuned": False, "bounded_full_backbone_subset_executed": False, "hodge_components_computed": False, "full_model_protocol_complete": False, "stable_residual_beyond_all_nulls": stable_beyond, "positive_paired_interval": positive_interval, "gate_passed": passed})
    write_csv(OUT / "full_model_runs.csv", runs)
    write_csv(OUT / "full_model_transitions.csv", transitions)
    write_csv(OUT / "full_model_nulls.csv", nulls)
    write_csv(OUT / "full_model_stability.csv", stability)
    write_csv(OUT / "full_model_paired.csv", paired)
    write_csv(OUT / "full_model_claims.csv", claims)
    summary = []
    for architecture in architectures:
        for method in sorted({row["method"] for row in runs}):
            block = [row for row in runs if row["architecture"] == architecture and row["method"] == method]
            summary.append({"architecture": architecture, "method": method, "collections": len(block), "accuracy": float(np.mean([float(row["accuracy"]) for row in block]))})
    latex_table(OUT / "tables" / "full_model.tex", ["architecture", "method", "collections", "accuracy"], summary, "Hidden-representation transition benchmark")
    passed_count = sum(bool(row["gate_passed"]) for row in claims)
    (OUT / "full_model_report.md").write_text(
        "# Full-model hidden-layer transition geometry\n\n"
        f"Execution commit: `{git_head()}`. Five independent CIFAR-10 feature-adapter collections were evaluated for each of {len(architectures)} cached pretrained feature sets at four recorded subspace widths. "
        f"{passed_count} architecture families passed the bounded residual and accuracy gate. Residual-block fine-tuning, the bounded full-backbone subset, Hodge-component decomposition, and official full-model merge implementations were not executed, so this stage is not a full-model result. No structured retransport was enabled without a certified chart action.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
