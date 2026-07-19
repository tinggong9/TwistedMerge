#!/usr/bin/env python3
"""Application A: estimate D4 adapter holonomy and test multiview fusion."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torchvision.datasets import CIFAR10

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.chart_followup_common import compose_d4
from src.holonomy_application_corpus import (
    LowRankChartAdapter,
    classification_metrics,
    parameter_count,
    seed_everything,
    state_bytes,
    tensor_mapping_sha256,
)
from src.holonomy_application_transitions import (
    bootstrap_transition_stability,
    commutator_distance,
    connection_synchronization,
    fit_transition,
    identity_distance,
    inverse_consistency,
    loop_product,
    normalized_fit_residual,
    orthogonal_polar,
    spectral_summary,
)

APP_DIR = ROOT / "reports" / "holonomy_applications" / "application_A_holonomy"
SHARED_DIR = ROOT / "reports" / "holonomy_applications"
ARTIFACT_ROOT = ROOT / "reports" / "tmp" / "holonomy_applications"
METHODS = (
    "best_individual_adapter",
    "raw_parameter_average",
    "pairwise_reference_alignment",
    "global_c2m3_synchronization",
    "graph_synchronized_adapter_merge",
    "regular_d4_branch_invariant_pooling",
    "orbit_branch_invariant_pooling",
    "generic_mixture_of_experts",
    "learned_router",
    "d4_test_time_augmentation",
    "random_branch_count_matched_control",
    "wrong_group_action_control",
    "wrong_multiplication_order_control",
    "prediction_ensemble_upper_bound",
    "oracle_chart_aware_fusion",
    "parameter_matched_generic_concat_head",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def verify_file(path_text: str, expected_hash: str) -> Path:
    path = Path(path_text)
    if not path.is_file():
        raise FileNotFoundError(path)
    observed = sha256_file(path)
    if observed != expected_hash:
        raise RuntimeError(f"artifact checksum mismatch: {path}")
    return path


def load_shared(mode: str) -> tuple[dict[str, object], pd.DataFrame, dict[str, object], Path]:
    shared = SHARED_DIR if mode == "confirmatory" else SHARED_DIR / "shared_corpus_smoke"
    resolved = json.loads((shared / "shared_corpus_resolved_config.json").read_text(encoding="utf-8"))
    manifest = pd.read_csv(shared / "shared_corpus_manifest.csv")
    feature_path = verify_file(
        str(resolved["feature_cache_path"]), str(resolved["feature_cache_sha256"])
    )
    feature_payload = torch.load(feature_path, map_location="cpu", weights_only=False)
    if feature_payload["evidence_label"] != "natural_measured":
        raise RuntimeError("shared feature cache has the wrong evidence label")
    if mode == "confirmatory" and len(manifest) != 40:
        raise RuntimeError("confirmatory shared corpus does not contain 40 adapters")
    return resolved, manifest, feature_payload, shared


def load_models(seed: int, manifest: pd.DataFrame, feature_dim: int, rank: int) -> list[LowRankChartAdapter]:
    seed_rows = manifest[manifest["corpus_seed"] == seed]
    if len(seed_rows) != 8:
        raise RuntimeError(f"seed {seed} does not have eight manifest rows")
    checkpoint_path = verify_file(
        str(seed_rows.iloc[0]["checkpoint_path"]), str(seed_rows.iloc[0]["checkpoint_sha256"])
    )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    models = []
    for chart in range(8):
        model = LowRankChartAdapter(feature_dim=feature_dim, rank=rank)
        model.load_state_dict(payload["states"][str(chart)])
        models.append(model.eval())
    return models


def model_tensors(
    models: list[LowRankChartAdapter], features: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return [model, view, item, feature/class] tensors."""

    activations = []
    logits = []
    with torch.no_grad():
        for model in models:
            model_activations = []
            model_logits = []
            for view in range(8):
                values = model.forward_activations(features[view])
                model_activations.append(values)
                model_logits.append(model.head(values))
            activations.append(torch.stack(model_activations))
            logits.append(torch.stack(model_logits))
    return torch.stack(activations), torch.stack(logits)


def fit_linear_classifier(
    features: torch.Tensor, labels: torch.Tensor, epochs: int, seed: int
) -> nn.Linear:
    seed_everything(seed)
    model = nn.Linear(features.shape[1], 10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02, weight_decay=1e-4)
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(model(features), labels)
        loss.backward()
        optimizer.step()
    return model.eval()


def router_features(logits: torch.Tensor) -> torch.Tensor:
    probabilities = logits.softmax(dim=-1)
    confidence = probabilities.max(dim=-1).values
    entropy = -(probabilities * probabilities.clamp_min(1e-9).log()).sum(dim=-1)
    # model, view, item -> view, item, model
    return torch.cat(
        [confidence.permute(1, 2, 0), entropy.permute(1, 2, 0)], dim=-1
    )


def fit_router(logits: torch.Tensor, labels: torch.Tensor, epochs: int, seed: int) -> nn.Linear:
    seed_everything(seed)
    features = router_features(logits).reshape(-1, 16)
    branches = logits.permute(1, 2, 0, 3).reshape(-1, 8, 10)
    repeated_labels = labels.repeat(8)
    router = nn.Linear(16, 8)
    optimizer = torch.optim.AdamW(router.parameters(), lr=0.03, weight_decay=1e-4)
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        weights = router(features).softmax(dim=1)
        fused = torch.einsum("nm,nmc->nc", weights, branches)
        loss = nn.functional.cross_entropy(fused, repeated_labels)
        loss.backward()
        optimizer.step()
    return router.eval()


def transported_head(
    models: list[LowRankChartAdapter], gauges: list[torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    weights = []
    biases = []
    for model, gauge in zip(models, gauges, strict=True):
        weights.append(model.head.weight.detach() @ torch.linalg.pinv(gauge))
        biases.append(model.head.bias.detach())
    return torch.stack(weights).mean(0), torch.stack(biases).mean(0)


def aligned_pool(activations: torch.Tensor, gauges: list[torch.Tensor]) -> torch.Tensor:
    values = [activations[index] @ gauges[index].T for index in range(8)]
    return torch.stack(values).mean(0)


def generic_moe(logits: torch.Tensor) -> torch.Tensor:
    confidence = logits.softmax(dim=-1).max(dim=-1).values
    weights = (5.0 * confidence).softmax(dim=0)
    return torch.einsum("mvn,mvnc->vnc", weights, logits)


def branch_logits(
    logits: torch.Tensor,
    mapping,
) -> torch.Tensor:
    outputs = []
    for anchor in range(8):
        branches = [logits[branch, mapping(anchor, branch)] for branch in range(8)]
        outputs.append(torch.stack(branches).mean(0))
    return torch.stack(outputs)


def orbit_aligned_logits(
    activations: torch.Tensor,
    gauges: list[torch.Tensor],
    head_weight: torch.Tensor,
    head_bias: torch.Tensor,
) -> torch.Tensor:
    outputs = []
    for anchor in range(8):
        branches = [
            activations[branch, compose_d4(anchor, branch)] @ gauges[branch].T
            for branch in range(8)
        ]
        pooled = torch.stack(branches).mean(0)
        outputs.append(pooled @ head_weight.T + head_bias)
    return torch.stack(outputs)


def train_fusion_components(
    models: list[LowRankChartAdapter],
    features: torch.Tensor,
    labels: torch.Tensor,
    global_gauges: list[torch.Tensor],
    epochs: int,
    seed: int,
) -> tuple[nn.Linear, nn.Linear, nn.Linear]:
    activations, logits = model_tensors(models, features)
    pooled = aligned_pool(activations, global_gauges)
    global_head = fit_linear_classifier(
        pooled.reshape(-1, pooled.shape[-1]), labels.repeat(8), epochs, seed + 10
    )
    concatenated = activations.permute(1, 2, 0, 3).reshape(8 * len(labels), -1)
    generic_head = fit_linear_classifier(concatenated, labels.repeat(8), epochs, seed + 20)
    router = fit_router(logits, labels, epochs, seed + 30)
    return global_head, generic_head, router


def build_candidates(
    models: list[LowRankChartAdapter],
    features: torch.Tensor,
    pairwise_gauges: list[torch.Tensor],
    global_gauges: list[torch.Tensor],
    best_model: int,
    global_head: nn.Linear,
    generic_head: nn.Linear,
    router: nn.Linear,
    random_permutation: np.ndarray,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    activations, logits = model_tensors(models, features)
    candidates: dict[str, torch.Tensor] = {}
    latencies: dict[str, float] = {}

    def timed(name: str, function) -> None:
        started = time.perf_counter()
        candidates[name] = function().detach().cpu()
        latencies[name] = (time.perf_counter() - started) * 1000.0

    timed("best_individual_adapter", lambda: logits[best_model])

    average_adapter = torch.stack([model.effective_adapter().detach() for model in models]).mean(0)
    average_head_weight = torch.stack([model.head.weight.detach() for model in models]).mean(0)
    average_head_bias = torch.stack([model.head.bias.detach() for model in models]).mean(0)
    timed(
        "raw_parameter_average",
        lambda: torch.stack(
            [features[view] @ average_adapter.T @ average_head_weight.T + average_head_bias for view in range(8)]
        ),
    )

    pairwise_weight, pairwise_bias = transported_head(models, pairwise_gauges)
    pairwise_pooled = aligned_pool(activations, pairwise_gauges)
    timed(
        "pairwise_reference_alignment",
        lambda: pairwise_pooled @ models[0].head.weight.detach().T + models[0].head.bias.detach(),
    )
    global_pooled = aligned_pool(activations, global_gauges)
    timed("global_c2m3_synchronization", lambda: global_head(global_pooled))
    graph_weight, graph_bias = transported_head(models, global_gauges)
    merged_adapter = torch.stack(
        [global_gauges[index] @ models[index].effective_adapter().detach() for index in range(8)]
    ).mean(0)
    timed(
        "graph_synchronized_adapter_merge",
        lambda: torch.stack(
            [features[view] @ merged_adapter.T @ graph_weight.T + graph_bias for view in range(8)]
        ),
    )
    timed(
        "regular_d4_branch_invariant_pooling",
        lambda: branch_logits(logits, compose_d4),
    )
    timed(
        "orbit_branch_invariant_pooling",
        lambda: orbit_aligned_logits(activations, global_gauges, graph_weight, graph_bias),
    )
    timed("generic_mixture_of_experts", lambda: generic_moe(logits))

    def routed() -> torch.Tensor:
        weights = router(router_features(logits)).softmax(dim=-1)
        return torch.einsum("vnm,mvnc->vnc", weights, logits)

    timed("learned_router", routed)
    tta = logits[best_model].mean(0, keepdim=True).repeat(8, 1, 1)
    timed("d4_test_time_augmentation", lambda: tta)
    timed(
        "random_branch_count_matched_control",
        lambda: branch_logits(logits, lambda anchor, branch: int(random_permutation[compose_d4(anchor, branch)])),
    )
    timed(
        "wrong_group_action_control",
        lambda: branch_logits(logits, lambda anchor, branch: (anchor + branch) % 8),
    )
    timed(
        "wrong_multiplication_order_control",
        lambda: branch_logits(logits, lambda anchor, branch: compose_d4(branch, anchor)),
    )
    upper = torch.stack([logits[chart, chart] for chart in range(8)]).mean(0)
    timed("prediction_ensemble_upper_bound", lambda: upper.unsqueeze(0).repeat(8, 1, 1))
    timed(
        "oracle_chart_aware_fusion",
        lambda: torch.stack([logits[chart, chart] for chart in range(8)]),
    )
    concatenated = activations.permute(1, 2, 0, 3).reshape(8, features.shape[1], -1)
    timed("parameter_matched_generic_concat_head", lambda: generic_head(concatenated))
    if set(candidates) != set(METHODS):
        raise AssertionError("candidate method schema is incomplete")
    return candidates, latencies


def score_candidate(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    view_metrics = [classification_metrics(logits[view], labels) for view in range(8)]
    fused = logits.mean(0)
    aggregate = classification_metrics(fused, labels)
    predictions = logits.argmax(dim=-1)
    modal = torch.mode(predictions, dim=0).values
    consistency = float((predictions == modal.unsqueeze(0)).float().mean())
    return {
        "ordinary_test_accuracy": aggregate["accuracy"],
        "average_view_accuracy": float(np.mean([row["accuracy"] for row in view_metrics])),
        "worst_view_accuracy": float(min(row["accuracy"] for row in view_metrics)),
        "nll": aggregate["nll"],
        "brier": aggregate["brier"],
        "ece": aggregate["ece"],
        "prediction_consistency": consistency,
    }


def paired_bootstrap(values: np.ndarray, samples: int, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    means = np.asarray([rng.choice(values, size=len(values), replace=True).mean() for _ in range(samples)])
    return float(values.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for record in frame.to_dict("records"):
        values = []
        for column in columns:
            value = record[column]
            values.append(f"{value:.6g}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def capacity_metadata(
    method: str,
    models: list[LowRankChartAdapter],
    feature_dim: int,
    global_head: nn.Linear,
    generic_head: nn.Linear,
    router: nn.Linear,
) -> dict[str, object]:
    one_parameters = parameter_count(models[0])
    all_parameters = sum(parameter_count(model) for model in models)
    one_bytes = state_bytes(models[0])
    all_bytes = sum(state_bytes(model) for model in models)
    one_branch = {
        "best_individual_adapter",
        "raw_parameter_average",
        "oracle_chart_aware_fusion",
    }
    single_active = one_branch | {"d4_test_time_augmentation"}
    single_stored = {
        "best_individual_adapter",
        "raw_parameter_average",
        "d4_test_time_augmentation",
    }
    branches = 1 if method in one_branch else 8
    active = one_parameters if method in single_active else all_parameters
    stored = one_bytes if method in single_stored else all_bytes
    extra = 0
    if method == "global_c2m3_synchronization":
        extra = parameter_count(global_head)
    elif method == "learned_router":
        extra = parameter_count(router)
    elif method == "parameter_matched_generic_concat_head":
        extra = parameter_count(generic_head)
    transition_bytes = 0
    if method in {
        "pairwise_reference_alignment",
        "global_c2m3_synchronization",
        "graph_synchronized_adapter_merge",
        "orbit_branch_invariant_pooling",
    }:
        transition_bytes = 8 * feature_dim * feature_dim * 4
    return {
        "active_parameters": active + extra,
        "stored_bytes": stored + extra * 4 + transition_bytes,
        "branch_count": branches,
        "inference_multiplier": float(branches),
        "fusion_parameters": extra,
        "parameter_multiplier_vs_single": float((active + extra) / one_parameters),
    }


def analyze_seed(
    seed: int,
    models: list[LowRankChartAdapter],
    features: dict[str, torch.Tensor],
    train_labels: torch.Tensor,
    validation_labels: torch.Tensor,
    test_indices: np.ndarray,
    config: dict[str, object],
    mode: str,
    output_dir: Path,
    test_dataset: CIFAR10,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    mode_config = config[mode]
    adapters = [model.effective_adapter().detach() for model in models]
    overlap_fit = [
        models[chart].forward_activations(features["overlap_fit"][chart]).detach()
        for chart in range(8)
    ]
    overlap_validation = [
        models[chart].forward_activations(features["overlap_validation"][chart]).detach()
        for chart in range(8)
    ]
    transitions_by_method: dict[str, dict[tuple[int, int], torch.Tensor]] = {}
    transition_rows: list[dict[str, object]] = []
    for method_index, method in enumerate(config["transition_methods"]):
        transitions: dict[tuple[int, int], torch.Tensor] = {}
        for source in range(8):
            for target in range(8):
                if source == target:
                    continue
                transition = fit_transition(
                    str(method),
                    overlap_fit[source],
                    overlap_fit[target],
                    adapters[source],
                    adapters[target],
                    low_rank=int(config["low_rank_transition_rank"]),
                )
                transitions[(source, target)] = transition
        transitions_by_method[str(method)] = transitions
        for source in range(8):
            for target in range(8):
                if source == target:
                    continue
                transition = transitions[(source, target)]
                bootstrap_mean, bootstrap_low, bootstrap_high = bootstrap_transition_stability(
                    str(method),
                    overlap_fit[source],
                    overlap_fit[target],
                    adapters[source],
                    adapters[target],
                    transition,
                    samples=int(mode_config["bootstrap_samples"]),
                    seed=810000 + seed * 10000 + method_index * 100 + source * 10 + target,
                    low_rank=int(config["low_rank_transition_rank"]),
                )
                singular = torch.linalg.svdvals(transition)
                transition_rows.append(
                    {
                        "evidence_label": "natural_measured",
                        "mode": mode,
                        "corpus_seed": seed,
                        "transition_method": method,
                        "source_chart": source,
                        "target_chart": target,
                        "fit_residual": normalized_fit_residual(
                            overlap_fit[source], overlap_fit[target], transition
                        ),
                        "heldout_overlap_residual": normalized_fit_residual(
                            overlap_validation[source], overlap_validation[target], transition
                        ),
                        "condition_number": float(singular.max() / singular.min().clamp_min(1e-12)),
                        "inverse_consistency_residual": inverse_consistency(
                            transition, transitions[(target, source)]
                        ),
                        "bootstrap_instability_mean": bootstrap_mean,
                        "bootstrap_instability_ci_low": bootstrap_low,
                        "bootstrap_instability_ci_high": bootstrap_high,
                        "transformation_uncertainty": bootstrap_high - bootstrap_low,
                    }
                )

    estimator_summary = (
        pd.DataFrame(transition_rows)
        .groupby("transition_method", as_index=False)["heldout_overlap_residual"]
        .mean()
        .sort_values(["heldout_overlap_residual", "transition_method"])
    )
    selected_method = str(estimator_summary.iloc[0]["transition_method"])
    selected = transitions_by_method[selected_method]
    loop_rows: list[dict[str, object]] = []
    loop_products_by_method: dict[str, dict[str, torch.Tensor]] = {}
    for method, transitions in transitions_by_method.items():
        gauges, synchronization_residual = connection_synchronization(transitions, nodes=8)
        products = {}
        for loop_name, vertices in config["loops"].items():
            product = loop_product(transitions, tuple(int(value) for value in vertices))
            products[str(loop_name)] = product
            loop_rows.append(
                {
                    "evidence_label": "natural_measured",
                    "mode": mode,
                    "corpus_seed": seed,
                    "transition_method": method,
                    "selected_transition_method": method == selected_method,
                    "loop_name": loop_name,
                    "loop_vertices": "-".join(map(str, vertices)),
                    "identity_distance": identity_distance(product),
                    "connection_synchronization_residual": synchronization_residual,
                    "group_element_classification": "unclassified_feature_space_operator",
                    **spectral_summary(product),
                }
            )
        loop_products_by_method[method] = products
        rotation = products["rotation_cycle"]
        reflection = products["reflection_rotation_cycle"]
        loop_rows.append(
            {
                "evidence_label": "natural_measured",
                "mode": mode,
                "corpus_seed": seed,
                "transition_method": method,
                "selected_transition_method": method == selected_method,
                "loop_name": "rotation_reflection_commutator",
                "loop_vertices": "commutator",
                "identity_distance": commutator_distance(rotation, reflection),
                "connection_synchronization_residual": synchronization_residual,
                "group_element_classification": "commutator_operator",
                **spectral_summary(rotation @ reflection @ torch.linalg.pinv(rotation) @ torch.linalg.pinv(reflection)),
            }
        )

    pairwise_gauges = [
        torch.eye(adapters[0].shape[0]) if chart == 0 else selected[(chart, 0)]
        for chart in range(8)
    ]
    global_gauges, global_sync_residual = connection_synchronization(selected, nodes=8)
    global_head, generic_head, router = train_fusion_components(
        models,
        features["adapter_train"],
        train_labels,
        global_gauges,
        epochs=int(mode_config["head_epochs"]),
        seed=820000 + seed * 100,
    )
    _validation_activations, validation_logits = model_tensors(models, features["validation"])
    individual_validation = [
        float(
            np.mean(
                [
                    classification_metrics(validation_logits[model, view], validation_labels)["accuracy"]
                    for view in range(8)
                ]
            )
        )
        for model in range(8)
    ]
    best_model = int(np.argmax(individual_validation))
    random_permutation = np.random.default_rng(830000 + seed).permutation(8)
    candidates, latency = build_candidates(
        models,
        features["test"],
        pairwise_gauges,
        global_gauges,
        best_model,
        global_head,
        generic_head,
        router,
        random_permutation,
    )
    logits_path = ARTIFACT_ROOT / f"application_A_{mode}" / f"candidate_logits_seed_{seed}.npz"
    logits_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        logits_path,
        **{name: values.numpy().astype(np.float32) for name, values in candidates.items()},
        test_indices=test_indices,
    )
    logits_hash_before_labels = sha256_file(logits_path)
    logits_content_hash = tensor_mapping_sha256(candidates)
    # This is the first and only confirmatory access to test labels in Application A.
    test_labels = torch.tensor(np.asarray(test_dataset.targets), dtype=torch.long)[
        torch.from_numpy(test_indices)
    ]
    run_rows: list[dict[str, object]] = []
    capacity_rows: list[dict[str, object]] = []
    selected_loop_rows = [
        row
        for row in loop_rows
        if row["transition_method"] == selected_method
        and row["loop_name"] != "rotation_reflection_commutator"
    ]
    max_holonomy = max(float(row["identity_distance"]) for row in selected_loop_rows)
    commutator_value = next(
        float(row["identity_distance"])
        for row in loop_rows
        if row["transition_method"] == selected_method
        and row["loop_name"] == "rotation_reflection_commutator"
    )
    pairwise_mean = float(
        np.mean(
            [
                row["heldout_overlap_residual"]
                for row in transition_rows
                if row["transition_method"] == selected_method
            ]
        )
    )
    for method in METHODS:
        metrics = score_candidate(candidates[method], test_labels)
        capacity = capacity_metadata(
            method, models, features["test"].shape[-1], global_head, generic_head, router
        )
        run_rows.append(
            {
                "evidence_label": "natural_measured",
                "mode": mode,
                "corpus_seed": seed,
                "method": method,
                **metrics,
                "selected_transition_method": selected_method,
                "mean_heldout_pairwise_residual": pairwise_mean,
                "max_loop_holonomy_identity_distance": max_holonomy,
                "rotation_reflection_commutator_distance": commutator_value,
                "global_synchronization_residual": global_sync_residual,
                "best_individual_chart_selected_on_validation": best_model,
                "fusion_latency_ms_for_complete_test_tensor": latency[method],
                "test_logits_path": str(logits_path),
                "test_logits_sha256": logits_hash_before_labels,
                "test_logits_content_sha256": logits_content_hash,
                "test_labels_used_before_logits_saved": False,
                "execution_commit": git_head(),
                **capacity,
            }
        )
        capacity_rows.append(
            {
                "evidence_label": "natural_measured",
                "mode": mode,
                "corpus_seed": seed,
                "method": method,
                **capacity,
                "fusion_latency_ms_for_complete_test_tensor": latency[method],
                "shared_frozen_encoder_parameters": 11176512,
                "transition_feature_dimension": features["test"].shape[-1],
            }
        )
    permuted = test_labels[torch.randperm(len(test_labels), generator=torch.Generator().manual_seed(840000 + seed))]
    _ = score_candidate(candidates["best_individual_adapter"], permuted)
    if sha256_file(logits_path) != logits_hash_before_labels:
        raise RuntimeError("candidate logit file changed after label access")
    artifact_rows = [
        {
            "evidence_label": "natural_measured",
            "mode": mode,
            "corpus_seed": seed,
            "artifact_kind": "candidate_logits_before_test_labels",
            "path": str(logits_path),
            "sha256": logits_hash_before_labels,
            "content_sha256": logits_content_hash,
            "bytes": logits_path.stat().st_size,
            "label_permutation_hash_passed": True,
        }
    ]
    return run_rows, transition_rows, loop_rows, capacity_rows, artifact_rows


def write_outputs(
    mode: str,
    config: dict[str, object],
    output_dir: Path,
    run_rows: list[dict[str, object]],
    transition_rows: list[dict[str, object]],
    loop_rows: list[dict[str, object]],
    capacity_rows: list[dict[str, object]],
    artifact_rows: list[dict[str, object]],
    failures: list[dict[str, object]],
    command: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plots = output_dir / "plots"
    tables = output_dir / "tables"
    plots.mkdir(exist_ok=True)
    tables.mkdir(exist_ok=True)
    runs = pd.DataFrame(run_rows)
    transitions = pd.DataFrame(transition_rows)
    loops = pd.DataFrame(loop_rows)
    capacity = pd.DataFrame(capacity_rows)
    runs.to_csv(output_dir / "runs.csv", index=False)
    transitions.to_csv(output_dir / "transitions.csv", index=False)
    loops.to_csv(output_dir / "holonomy_loops.csv", index=False)
    capacity.to_csv(output_dir / "capacity_audit.csv", index=False)
    pd.DataFrame(failures, columns=("mode", "corpus_seed", "stage", "error_type", "message")).to_csv(
        output_dir / "failure_log.csv", index=False
    )

    comparisons = (
        ("orbit_branch_invariant_pooling", "generic_mixture_of_experts"),
        ("orbit_branch_invariant_pooling", "d4_test_time_augmentation"),
        ("orbit_branch_invariant_pooling", "random_branch_count_matched_control"),
        ("orbit_branch_invariant_pooling", "parameter_matched_generic_concat_head"),
        ("regular_d4_branch_invariant_pooling", "random_branch_count_matched_control"),
        ("global_c2m3_synchronization", "raw_parameter_average"),
    )
    paired_rows = []
    for metric in ("ordinary_test_accuracy", "average_view_accuracy", "worst_view_accuracy"):
        pivot = runs.pivot(index="corpus_seed", columns="method", values=metric)
        for left, right in comparisons:
            delta = (pivot[left] - pivot[right]).to_numpy(dtype=float)
            mean, low, high = paired_bootstrap(
                delta,
                int(config[mode]["statistic_bootstrap_samples"]),
                seed=850000 + len(paired_rows),
            )
            paired_rows.append(
                {
                    "evidence_label": "natural_measured",
                    "mode": mode,
                    "metric": metric,
                    "left_method": left,
                    "right_method": right,
                    "n_independent_seeds": len(delta),
                    "mean_delta": mean,
                    "median_delta": float(np.median(delta)),
                    "standard_deviation": float(np.std(delta, ddof=1)) if len(delta) > 1 else 0.0,
                    "ci_low": low,
                    "ci_high": high,
                    "wins": int((delta > 1e-12).sum()),
                    "ties": int((np.abs(delta) <= 1e-12).sum()),
                    "losses": int((delta < -1e-12).sum()),
                }
            )
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(output_dir / "paired_statistics.csv", index=False)

    summary = (
        runs.groupby("method", as_index=False)
        .agg(
            mean_accuracy=("ordinary_test_accuracy", "mean"),
            mean_average_view_accuracy=("average_view_accuracy", "mean"),
            mean_worst_view_accuracy=("worst_view_accuracy", "mean"),
            standard_deviation=("ordinary_test_accuracy", "std"),
            mean_ece=("ece", "mean"),
            mean_consistency=("prediction_consistency", "mean"),
        )
        .sort_values("mean_accuracy", ascending=False)
    )
    summary.to_csv(output_dir / "summary.csv", index=False)

    independent_seeds = runs["corpus_seed"].nunique()
    confirmatory_adequate = mode == "confirmatory" and independent_seeds >= 5
    primary = paired[
        (paired["left_method"] == "orbit_branch_invariant_pooling")
        & (paired["metric"] == "ordinary_test_accuracy")
    ]
    structured_beats_controls = bool(
        confirmatory_adequate and len(primary) and (primary["ci_low"] > 0).all()
    )
    worst_row = paired[
        (paired["left_method"] == "orbit_branch_invariant_pooling")
        & (paired["right_method"] == "generic_mixture_of_experts")
        & (paired["metric"] == "worst_view_accuracy")
    ]
    ordinary_row = paired[
        (paired["left_method"] == "orbit_branch_invariant_pooling")
        & (paired["right_method"] == "generic_mixture_of_experts")
        & (paired["metric"] == "ordinary_test_accuracy")
    ]
    worst_improves = bool(
        confirmatory_adequate
        and len(worst_row)
        and len(ordinary_row)
        and float(worst_row.iloc[0]["ci_low"]) > 0
        and float(ordinary_row.iloc[0]["mean_delta"])
        >= -float(config["gates"]["worst_view_mean_loss_tolerance"])
    )
    selected_loops = loops[loops["selected_transition_method"] == True]
    max_holonomy = float(
        selected_loops[selected_loops["loop_name"] != "rotation_reflection_commutator"][
            "identity_distance"
        ].max()
    )
    mean_commutator = float(
        selected_loops[selected_loops["loop_name"] == "rotation_reflection_commutator"][
            "identity_distance"
        ].mean()
    )
    nontrivial = max_holonomy >= float(config["gates"]["nontrivial_holonomy_distance"])
    noncommuting = mean_commutator >= float(config["gates"]["noncommuting_commutator_distance"])
    prediction_sample_adequate = independent_seeds >= 10
    application_gate = structured_beats_controls or worst_improves
    claim_rows = [
        {
            "claim_id": "measured_nonidentity_holonomy",
            "status": "smoke_only" if mode == "smoke" else ("supported_structural" if nontrivial else "negative"),
            "safe_wording": "Measured adapter transitions have nonidentity loop products in this corpus." if nontrivial else "Measured loop products remain near identity under the preregistered threshold.",
            "gate_passed": nontrivial,
        },
        {
            "claim_id": "measured_noncommuting_holonomy",
            "status": "smoke_only" if mode == "smoke" else ("supported_structural" if noncommuting else "negative"),
            "safe_wording": "Selected transition estimates yield noncommuting loop operators in this corpus." if noncommuting else "No stable noncommuting loop signal passed the threshold.",
            "gate_passed": noncommuting,
        },
        {
            "claim_id": "holonomy_predicts_harm_beyond_pairwise_fit",
            "status": "inadequate_sample" if not prediction_sample_adequate else "not_supported",
            "safe_wording": f"{independent_seeds} independent corpus seed(s) are insufficient for a held-out incremental-prediction claim.",
            "gate_passed": False,
        },
        {
            "claim_id": "structured_invariant_pooling_beats_matched_controls",
            "status": "supported" if structured_beats_controls else "negative",
            "safe_wording": "Orbit invariant pooling beats every preregistered matched control by paired seed bootstrap." if structured_beats_controls else "Orbit invariant pooling did not beat every preregistered matched control.",
            "gate_passed": structured_beats_controls,
        },
        {
            "claim_id": "worst_view_improvement_without_material_mean_loss",
            "status": "supported" if worst_improves else "negative",
            "safe_wording": "Orbit invariant pooling improves worst-view accuracy without material mean loss." if worst_improves else "The preregistered worst-view benefit gate did not pass.",
            "gate_passed": worst_improves,
        },
    ]
    pd.DataFrame(claim_rows).to_csv(output_dir / "claims.csv", index=False)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    plot_summary = summary.sort_values("mean_worst_view_accuracy", ascending=True)
    axes[0].barh(plot_summary["method"], plot_summary["mean_worst_view_accuracy"], color="#2f6f9f")
    axes[0].set_xlabel("Mean worst-view accuracy")
    axes[0].set_title("D4 robustness across five corpus seeds")
    estimator = (
        transitions.groupby(["corpus_seed", "transition_method"], as_index=False)
        .agg(mean_pairwise=("heldout_overlap_residual", "mean"))
    )
    loop_summary = (
        loops[loops["loop_name"] != "rotation_reflection_commutator"]
        .groupby(["corpus_seed", "transition_method"], as_index=False)
        .agg(max_holonomy=("identity_distance", "max"))
    )
    scatter = estimator.merge(loop_summary, on=["corpus_seed", "transition_method"])
    for method, rows in scatter.groupby("transition_method"):
        axes[1].scatter(rows["mean_pairwise"], rows["max_holonomy"], label=method, s=35)
    axes[1].set_xlabel("Mean held-out pairwise residual")
    axes[1].set_ylabel("Maximum loop distance from identity")
    axes[1].set_title("Pairwise fit does not equal loop consistency")
    axes[1].legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(plots / "holonomy_application_summary.pdf", bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10, 5))
    top = summary.head(10).sort_values("mean_accuracy")
    axis.barh(top["method"], top["mean_accuracy"], color="#5b8c5a")
    axis.set_xlabel("Mean fused test accuracy")
    axis.set_title("Application A fusion comparison")
    figure.tight_layout()
    figure.savefig(plots / "fusion_accuracy.pdf", bbox_inches="tight")
    plt.close(figure)

    latex_rows = ["\\begin{tabular}{lrrr}", "\\toprule", "Method & Fused acc. & Avg. view & Worst view\\\\", "\\midrule"]
    for row in summary.head(10).itertuples(index=False):
        latex_rows.append(
            f"{row.method.replace('_', ' ')} & {row.mean_accuracy:.3f} & {row.mean_average_view_accuracy:.3f} & {row.mean_worst_view_accuracy:.3f}\\\\"
        )
    latex_rows.extend(["\\bottomrule", "\\end{tabular}", ""])
    (tables / "application_A_fusion.tex").write_text("\n".join(latex_rows), encoding="utf-8")

    report = f"""# Application A: Holonomy-Aware Multiview Fusion

Decision: **{'positive application gate' if application_gate else ('bounded smoke completed' if mode == 'smoke' else 'negative application result')}**.

## Commands

Smoke:

```bash
{sys.executable} experiments/holonomy_application_A.py --mode smoke --data-dir /Users/tinggong/Documents/GitHub/TwistedMerge/data
```

Confirmatory execution:

```bash
{sys.executable} experiments/holonomy_application_A.py --mode confirmatory --data-dir /Users/tinggong/Documents/GitHub/TwistedMerge/data
```

## Frozen corpus and leakage boundary

This phase loaded the exact shared feature cache and adapter checkpoints and verified every recorded SHA-256. No chart adapter was retrained. Pair transitions used `overlap_fit`; estimator choice used held-out `overlap_validation`; fusion components used only `adapter_train`; the best individual adapter used `validation`. All sixteen candidate test-logit tensors were saved and hashed before test labels were accessed.

## Structural result

- Maximum selected loop distance from identity: `{max_holonomy:.6f}`.
- Mean rotation/reflection loop-commutator distance: `{mean_commutator:.6f}`.
- Nonidentity holonomy threshold passed: `{nontrivial}`.
- Noncommuting-holonomy threshold passed: `{noncommuting}`.
- These are feature-space loop operators. They are not central, projective, or Brauer-class claims.

## Fusion result

- Independent model-training seeds: `{independent_seeds}`.
- Structured branch method beat every preregistered matched control: `{structured_beats_controls}`.
- Worst-view benefit without material mean loss: `{worst_improves}`.
- Overall Application A gate: `{application_gate}`.

{markdown_table(summary.head(10))}

## Answers to the application questions

1. Loop holonomy is numerically distinct from pairwise fitting, but {independent_seeds} seeds are inadequate for a held-out claim that it adds predictive information beyond pairwise residuals.
2. Noncommuting loop operators {'were observed' if noncommuting else 'did not pass the threshold'}, but a causal prediction claim is not made.
3. The worst-view invariant-pooling gate {'passed' if worst_improves else 'did not pass'}.
4. The all-controls structured-pooling gate {'passed' if structured_beats_controls else 'did not pass'}.
5. Capacity-matched random, generic-routing, and generic-concatenation controls are retained in the paired and capacity tables; extra branches alone are not credited as group-structure evidence.

## Stopping rule

No additional dataset or chart family is opened by a negative result. Application B proceeds only as the required conservative certificate on these exact saved natural transitions.
"""
    (output_dir / "report.md").write_text(report, encoding="utf-8")

    artifact_rows.extend(
        {
            "evidence_label": "natural_measured",
            "mode": mode,
            "corpus_seed": "all",
            "artifact_kind": "committed_output",
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256_file(path),
            "content_sha256": "",
            "bytes": path.stat().st_size,
            "label_permutation_hash_passed": "",
        }
        for path in (
            output_dir / "runs.csv",
            output_dir / "transitions.csv",
            output_dir / "holonomy_loops.csv",
            output_dir / "paired_statistics.csv",
            output_dir / "capacity_audit.csv",
            output_dir / "claims.csv",
            plots / "holonomy_application_summary.pdf",
            plots / "fusion_accuracy.pdf",
            tables / "application_A_fusion.tex",
        )
    )
    pd.DataFrame(artifact_rows).to_csv(output_dir / "artifact_hashes.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "confirmatory"), required=True)
    parser.add_argument("--config", type=Path, default=APP_DIR / "config.json")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/Users/tinggong/Documents/GitHub/TwistedMerge/data"),
    )
    args = parser.parse_args()
    command = " ".join([sys.executable, *sys.argv])
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output_dir = APP_DIR if args.mode == "confirmatory" else APP_DIR / "smoke"
    resolved, manifest, payload, _shared = load_shared(args.mode)
    features = {name: values.float() for name, values in payload["features"].items()}
    splits = {name: values.numpy() for name, values in payload["splits"].items()}
    train_dataset = CIFAR10(args.data_dir, train=True, download=False)
    test_dataset = CIFAR10(args.data_dir, train=False, download=False)
    all_train_labels = torch.tensor(np.asarray(train_dataset.targets), dtype=torch.long)
    train_labels = all_train_labels[torch.from_numpy(splits["adapter_train"])]
    validation_labels = all_train_labels[torch.from_numpy(splits["validation"])]
    run_rows: list[dict[str, object]] = []
    transition_rows: list[dict[str, object]] = []
    loop_rows: list[dict[str, object]] = []
    capacity_rows: list[dict[str, object]] = []
    artifact_rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    for seed in sorted(int(value) for value in manifest["corpus_seed"].unique()):
        try:
            models = load_models(
                seed, manifest, int(resolved["feature_dim"]), int(resolved["adapter_rank"])
            )
            outputs = analyze_seed(
                seed,
                models,
                features,
                train_labels,
                validation_labels,
                splits["test"],
                config,
                args.mode,
                output_dir,
                test_dataset,
            )
            for target, rows in zip(
                (run_rows, transition_rows, loop_rows, capacity_rows, artifact_rows),
                outputs,
                strict=True,
            ):
                target.extend(rows)
        except Exception as error:
            failures.append(
                {
                    "mode": args.mode,
                    "corpus_seed": seed,
                    "stage": "application_A",
                    "error_type": type(error).__name__,
                    "message": str(error),
                }
            )
    write_outputs(
        args.mode,
        config,
        output_dir,
        run_rows,
        transition_rows,
        loop_rows,
        capacity_rows,
        artifact_rows,
        failures,
        command,
    )
    expected = 16 * manifest["corpus_seed"].nunique()
    if failures or len(run_rows) != expected:
        raise RuntimeError("Application A incomplete; inspect failure_log.csv")


if __name__ == "__main__":
    main()
