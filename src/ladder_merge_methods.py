"""Actionable merge helpers for the structure-group ladder benchmark."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model_merging_benchmark import DatasetSpec, clone_model, make_model, require_torch


@dataclass(frozen=True)
class MethodMetadata:
    method: str
    symmetry_status: str
    is_single_model: bool
    capacity_matched_to_weight_average: bool
    notes: str


METHOD_METADATA: dict[str, MethodMetadata] = {
    "weight_average": MethodMetadata(
        method="weight_average",
        symmetry_status="ordinary_unaligned_single_model",
        is_single_model=True,
        capacity_matched_to_weight_average=True,
        notes="Plain parameter average without alignment.",
    ),
    "greedy_soup": MethodMetadata(
        method="greedy_soup",
        symmetry_status="ordinary_single_model_soup",
        is_single_model=True,
        capacity_matched_to_weight_average=True,
        notes="Greedy model soup over original models using validation accuracy.",
    ),
    "c2m3_permutation": MethodMetadata(
        method="c2m3_permutation",
        symmetry_status="exact_relu_permutation_symmetry",
        is_single_model=True,
        capacity_matched_to_weight_average=True,
        notes="Cycle-consistent permutation alignment followed by weight averaging.",
    ),
    "signed_permutation": MethodMetadata(
        method="signed_permutation",
        symmetry_status="heuristic_relu_not_exact",
        is_single_model=True,
        capacity_matched_to_weight_average=True,
        notes="Sign flips are not exact symmetries for ReLU hidden units; this is a heuristic control.",
    ),
    "monomial_scale": MethodMetadata(
        method="monomial_scale",
        symmetry_status="exact_relu_positive_scale_symmetry",
        is_single_model=True,
        capacity_matched_to_weight_average=True,
        notes="Positive hidden-unit rescaling is exact for ReLU when outgoing weights are inversely adjusted.",
    ),
    "low_rank_GL_diagnostic": MethodMetadata(
        method="low_rank_GL_diagnostic",
        symmetry_status="diagnostic_not_single_model_for_relu",
        is_single_model=False,
        capacity_matched_to_weight_average=False,
        notes="Full GL hidden-basis changes are not exact same-architecture ReLU model symmetries here.",
    ),
    "ensemble_upper_bound": MethodMetadata(
        method="ensemble_upper_bound",
        symmetry_status="extra_capacity_ensemble",
        is_single_model=False,
        capacity_matched_to_weight_average=False,
        notes="Prediction ensemble upper bound; not a single merged model.",
    ),
}


def estimate_signs_and_positive_scales(
    reference_features: np.ndarray,
    model_features: np.ndarray,
    ref_to_model_perm: np.ndarray,
    min_scale: float = 1e-3,
    max_scale: float = 1e3,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate signs and positive scales after permutation alignment.

    For each reference hidden unit r, `ref_to_model_perm[r]` is the matching
    hidden unit in the target model.  The positive scale estimates
    `model_hidden ~= scale * reference_hidden` by least squares on raw ReLU
    activations, then clips to avoid numerical explosions.
    """

    ref = np.asarray(reference_features, dtype=float)
    other = np.asarray(model_features, dtype=float)
    perm = np.asarray(ref_to_model_perm, dtype=int)
    signs = np.ones(len(perm), dtype=float)
    scales = np.ones(len(perm), dtype=float)
    centered_ref = ref - ref.mean(axis=0, keepdims=True)
    centered_other = other - other.mean(axis=0, keepdims=True)
    for unit, target in enumerate(perm):
        a_raw = ref[:, unit]
        b_raw = other[:, target]
        a_centered = centered_ref[:, unit]
        b_centered = centered_other[:, target]
        dot_centered = float(np.dot(a_centered, b_centered))
        signs[unit] = 1.0 if dot_centered >= 0.0 else -1.0
        denom = max(float(np.dot(a_raw, a_raw)), 1e-12)
        scale = float(np.dot(a_raw, b_raw) / denom)
        if not np.isfinite(scale) or scale <= 0.0:
            scale = 1.0
        scales[unit] = float(np.clip(scale, min_scale, max_scale))
    return signs, scales


def transform_mlp_hidden_units(
    model,
    spec: DatasetSpec,
    width: int,
    ref_to_model_perm: np.ndarray,
    signs: np.ndarray | None = None,
    positive_scales: np.ndarray | None = None,
):
    """Return an MLP aligned to reference order with optional signs/scales.

    Positive scaling is an exact ReLU reparameterization:

        hidden weights/bias divided by s, classifier columns multiplied by s.

    Sign flips use the analogous operation but are not exact for ReLU when the
    sign is negative; callers must label that path as heuristic.
    """

    torch, _, _ = require_torch()
    aligned = make_model("mlp", spec, width)
    perm = np.asarray(ref_to_model_perm, dtype=int)
    sign_vec = np.ones(width, dtype=float) if signs is None else np.asarray(signs, dtype=float)
    scale_vec = np.ones(width, dtype=float) if positive_scales is None else np.asarray(positive_scales, dtype=float)
    if sign_vec.shape[0] != width or scale_vec.shape[0] != width:
        raise ValueError("signs and positive_scales must have length width")
    if np.any(scale_vec <= 0.0):
        raise ValueError("positive scales must be strictly positive")
    incoming_factor = sign_vec / scale_vec
    outgoing_factor = sign_vec * scale_vec
    with torch.no_grad():
        source_weight = model.hidden.weight.detach().cpu()[perm, :]
        source_bias = model.hidden.bias.detach().cpu()[perm]
        source_out = model.classifier.weight.detach().cpu()[:, perm]
        aligned.hidden.weight.copy_(source_weight * torch.tensor(incoming_factor, dtype=source_weight.dtype).unsqueeze(1))
        aligned.hidden.bias.copy_(source_bias * torch.tensor(incoming_factor, dtype=source_bias.dtype))
        aligned.classifier.weight.copy_(source_out * torch.tensor(outgoing_factor, dtype=source_out.dtype).unsqueeze(0))
        aligned.classifier.bias.copy_(model.classifier.bias.detach().cpu())
    return aligned


def transform_mlp_positive_scale(
    model,
    spec: DatasetSpec,
    width: int,
    ref_to_model_perm: np.ndarray,
    positive_scales: np.ndarray,
):
    return transform_mlp_hidden_units(
        model,
        spec,
        width,
        ref_to_model_perm,
        signs=np.ones(width, dtype=float),
        positive_scales=positive_scales,
    )


def transform_mlp_signed(
    model,
    spec: DatasetSpec,
    width: int,
    ref_to_model_perm: np.ndarray,
    signs: np.ndarray,
):
    return transform_mlp_hidden_units(
        model,
        spec,
        width,
        ref_to_model_perm,
        signs=signs,
        positive_scales=np.ones(width, dtype=float),
    )


def clone_mlp(model, spec: DatasetSpec, width: int):
    return clone_model(model, "mlp", spec, width)
