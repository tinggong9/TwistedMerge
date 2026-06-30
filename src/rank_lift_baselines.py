"""Capacity-matched branch baselines for rank-lift experiments.

These utilities intentionally avoid triangle defects and obstruction
residuals for the non-obstruction controls.  They match the branch count and
inference multiplier of the rank-lift branch ensemble, not the capacity of a
single weight-averaged model.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from src.model_merging_benchmark import (
    average_models,
    clone_model,
    evaluate_model,
    permutation_disagreement,
)


CAPACITY_METADATA_FIELDS = (
    "method_note",
    "is_single_model",
    "branch_count",
    "parameter_count",
    "parameter_multiplier",
    "inference_multiplier",
    "capacity_matched_to_weight_average",
    "capacity_matched_to_rank_lift",
    "uses_obstruction_residual",
    "uses_validation_data",
    "uses_distillation",
)


def _effective_branch_count(n_models: int, n_branches: int) -> int:
    if n_models <= 0:
        raise ValueError("at least one model is required")
    return max(1, min(int(n_branches), n_models))


def random_branch_ensemble(aligned_models, n_branches, architecture, spec, width, seed):
    """Randomly partition aligned models and average each partition.

    This baseline matches the requested branch count when possible and uses no
    validation metrics, pairwise cycle information, or obstruction residuals.
    """

    branch_count = _effective_branch_count(len(aligned_models), n_branches)
    rng = np.random.default_rng(seed)
    indices = np.arange(len(aligned_models), dtype=int)
    rng.shuffle(indices)
    branches = []
    for group in np.array_split(indices, branch_count):
        if len(group) == 0:
            continue
        branches.append(average_models([aligned_models[int(idx)] for idx in group], architecture, spec, width))
    return branches


def validation_branch_ensemble(models, val_loader, test_loader, n_branches, architecture, spec, width, device):
    """Select the top validation-accuracy individual models as branches.

    ``test_loader`` is accepted to keep the experiment call signature explicit,
    but selection uses validation accuracy only.
    """

    del test_loader
    branch_count = _effective_branch_count(len(models), n_branches)
    scored = []
    for idx, model in enumerate(models):
        metrics = evaluate_model(model, val_loader, device)
        scored.append((float(metrics["accuracy"]), idx))
    selected = [idx for _acc, idx in sorted(scored, reverse=True)[:branch_count]]
    return [clone_model(models[idx], architecture, spec, width) for idx in selected]


def _pairwise_distance(pairwise, i: int, j: int, width: int) -> float:
    if i == j:
        return 0.0
    identity = np.arange(width)
    if (i, j) in pairwise:
        return permutation_disagreement(pairwise[(i, j)], identity)
    if (j, i) in pairwise:
        return permutation_disagreement(pairwise[(j, i)], identity)
    return 0.0


def c2m3_cluster_branch_ensemble(aligned_synced, pairwise, n_branches, architecture, spec, width):
    """Cluster post-C2M3 aligned models using pairwise permutation distances.

    This baseline may use C2M3/permutation-distance information, but it does
    not use triangle defects, obstruction residuals, or branch labels derived
    from a rank-lift obstruction signal.
    """

    branch_count = _effective_branch_count(len(aligned_synced), n_branches)
    if branch_count == 1:
        return [average_models(aligned_synced, architecture, spec, width)]

    n_models = len(aligned_synced)
    seeds = [0]
    while len(seeds) < branch_count:
        remaining = [idx for idx in range(n_models) if idx not in seeds]
        next_seed = max(
            remaining,
            key=lambda idx: (min(_pairwise_distance(pairwise, idx, seed, width) for seed in seeds), -idx),
        )
        seeds.append(next_seed)

    clusters = {seed: [seed] for seed in seeds}
    for idx in range(n_models):
        if idx in clusters:
            continue
        seed = min(seeds, key=lambda candidate: (_pairwise_distance(pairwise, idx, candidate, width), candidate))
        clusters[seed].append(idx)

    return [
        average_models([aligned_synced[int(idx)] for idx in indices], architecture, spec, width)
        for indices in clusters.values()
        if indices
    ]


def count_parameters(model) -> int:
    """Return the number of trainable and non-trainable tensor parameters."""

    return int(sum(parameter.numel() for parameter in model.parameters()))


def _as_model_sequence(models_or_branches) -> list:
    if isinstance(models_or_branches, Sequence) and not hasattr(models_or_branches, "parameters"):
        return list(models_or_branches)
    return [models_or_branches]


def _default_method_note(method_name: str, is_single_model: bool, branch_count: int) -> str:
    name = method_name.lower()
    if name.startswith("twisted_rank_lift") or name.startswith("rank_lift_branch"):
        return "rank-lift branch ensemble; extra capacity, not a single merged model"
    if name.startswith("random_branch_ensemble"):
        return "random branch ensemble matched to rank-lift branch count; non-obstruction control"
    if name.startswith("validation_branch_ensemble"):
        return "validation-selected branch ensemble matched to rank-lift branch count; non-obstruction control"
    if name.startswith("c2m3_cluster_branch_ensemble"):
        return "C2M3-distance branch ensemble matched to rank-lift branch count; no obstruction residual used"
    if name.startswith("ensemble"):
        return "extra-capacity ensemble upper bound over local models"
    if "distilled" in name:
        return "distillation experiment; single-model capacity only after parameter and inference checks"
    if is_single_model:
        return "single-model merge or selection baseline"
    return f"{branch_count}-branch ensemble baseline"


def method_capacity_metadata(method_name, models_or_branches, base_model) -> dict:
    """Build standardized capacity and signal-use metadata for a method row."""

    models = _as_model_sequence(models_or_branches)
    branch_count = len(models)
    base_count = max(count_parameters(base_model), 1)
    parameter_count = int(sum(count_parameters(model) for model in models))
    is_single_model = branch_count == 1
    name = str(method_name).lower()
    uses_distillation = "distilled" in name or "distillation" in name
    rank_lift_matched_prefixes = (
        "twisted_rank_lift",
        "rank_lift_branch",
        "random_branch_ensemble",
        "validation_branch_ensemble",
        "c2m3_cluster_branch_ensemble",
    )
    uses_obstruction = name.startswith("twisted_rank_lift") or name.startswith("rank_lift_branch")
    uses_validation = "validation" in name or "greedy_soup" in name
    capacity_matched_to_rank_lift = name.startswith(rank_lift_matched_prefixes)
    return {
        "method_note": _default_method_note(str(method_name), is_single_model, branch_count),
        "is_single_model": bool(is_single_model),
        "branch_count": int(branch_count),
        "parameter_count": int(parameter_count),
        "parameter_multiplier": float(parameter_count / base_count),
        "inference_multiplier": float(branch_count),
        "capacity_matched_to_weight_average": bool(is_single_model and parameter_count == base_count),
        "capacity_matched_to_rank_lift": bool(capacity_matched_to_rank_lift),
        "uses_obstruction_residual": bool(uses_obstruction),
        "uses_validation_data": bool(uses_validation),
        "uses_distillation": bool(uses_distillation),
    }
