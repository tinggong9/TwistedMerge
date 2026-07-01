"""ReLU-compatible positive monomial gauge alignment for ReLU MLPs.

The hidden-unit symmetry used here is exact for ReLU MLPs: each hidden layer
may be permuted and positively rescaled, provided adjacent incoming/outgoing
weights are adjusted consistently.  The implementation supports the original
one-hidden-layer ``mlp`` path and the two-hidden-layer ``mlp2`` path.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
from typing import Iterable, Mapping

import numpy as np

from src.ladder_merge_methods import transform_mlp_positive_scale
from src.model_merging_benchmark import (
    activation_permutation,
    collect_features,
    make_model,
    permutation_matrix,
    require_torch,
    weight_permutation,
)


@dataclass(frozen=True)
class MonomialAlignment:
    """Alignment from a reference model's hidden basis to a target model.

    ``permutation[r]`` is the target hidden unit matched to reference unit
    ``r``.  ``positive_scales[r]`` estimates
    ``target_feature[:, permutation[r]] ~= scale[r] * reference_feature[:, r]``.
    For multi-layer models, these two fields refer to the primary layer
    (``hidden2`` for ``mlp2``), while ``layer_permutations`` and
    ``layer_positive_scales`` store every hidden layer.
    """

    permutation: np.ndarray
    positive_scales: np.ndarray
    matching: str
    scale_source: str
    scale_method: str = "raw"
    architecture: str = "mlp"
    primary_layer: str = "hidden"
    layer_permutations: Mapping[str, np.ndarray] | None = None
    layer_positive_scales: Mapping[str, np.ndarray] | None = None
    assignment_similarity_mean: float = float("nan")
    assignment_similarity_min: float = float("nan")
    low_similarity_fraction: float = float("nan")

    def __post_init__(self) -> None:
        perm = np.asarray(self.permutation, dtype=int)
        scales = np.asarray(self.positive_scales, dtype=float)
        _validate_perm_and_scales(perm, scales, "primary")
        layer_perms = (
            {str(self.primary_layer): perm.copy()}
            if self.layer_permutations is None
            else {str(layer): np.asarray(value, dtype=int).copy() for layer, value in self.layer_permutations.items()}
        )
        layer_scales = (
            {str(self.primary_layer): scales.copy()}
            if self.layer_positive_scales is None
            else {
                str(layer): np.asarray(value, dtype=float).copy()
                for layer, value in self.layer_positive_scales.items()
            }
        )
        if set(layer_perms) != set(layer_scales):
            raise ValueError("layer_permutations and layer_positive_scales must have the same layer keys")
        if str(self.primary_layer) not in layer_perms:
            raise ValueError("primary_layer must be present in layer_permutations")
        for layer in sorted(layer_perms):
            _validate_perm_and_scales(layer_perms[layer], layer_scales[layer], layer)
        object.__setattr__(self, "permutation", perm.copy())
        object.__setattr__(self, "positive_scales", scales.copy())
        object.__setattr__(self, "architecture", str(self.architecture))
        object.__setattr__(self, "primary_layer", str(self.primary_layer))
        object.__setattr__(self, "layer_permutations", layer_perms)
        object.__setattr__(self, "layer_positive_scales", layer_scales)

    def permutation_for(self, layer: str) -> np.ndarray:
        return np.asarray(self.layer_permutations[str(layer)], dtype=int).copy()

    def positive_scales_for(self, layer: str) -> np.ndarray:
        return np.asarray(self.layer_positive_scales[str(layer)], dtype=float).copy()

    def layers(self) -> tuple[str, ...]:
        return tuple(sorted(self.layer_permutations))


def _validate_perm_and_scales(perm: np.ndarray, scales: np.ndarray, label: str) -> None:
    if perm.ndim != 1 or scales.ndim != 1 or perm.shape[0] != scales.shape[0]:
        raise ValueError(f"{label} permutation and positive scales must be one-dimensional arrays of the same length")
    if sorted(perm.tolist()) != list(range(len(perm))):
        raise ValueError(f"{label} permutation must be a valid permutation")
    if np.any(scales <= 0.0) or not np.all(np.isfinite(scales)):
        raise ValueError(f"{label} positive scales must be finite and strictly positive")


def _infer_architecture(model) -> str:
    if hasattr(model, "hidden1") and hasattr(model, "hidden2"):
        return "mlp2"
    if hasattr(model, "hidden"):
        return "mlp"
    raise ValueError("monomial gauges currently support only mlp and mlp2 models")


def _layers_for_architecture(architecture: str) -> tuple[str, ...]:
    if architecture == "mlp":
        return ("hidden",)
    if architecture == "mlp2":
        return ("hidden1", "hidden2")
    raise ValueError(f"monomial gauges currently support only mlp and mlp2, got {architecture}")


def _primary_layer_for_architecture(architecture: str) -> str:
    return _layers_for_architecture(architecture)[-1]


def _width_for_layer(model, layer: str) -> int:
    if layer == "hidden":
        return int(model.hidden.out_features)
    if layer == "hidden1":
        return int(model.hidden1.out_features)
    if layer == "hidden2":
        return int(model.hidden2.out_features)
    raise ValueError(f"unknown monomial layer: {layer}")


def _canonical_matching(matching: str) -> str:
    name = str(matching).strip().lower()
    if name in {"activation", "monomial_activation"}:
        return "monomial_activation"
    if name in {"weight", "monomial_weight"}:
        return "monomial_weight"
    if name in {"monomial_activation_mlp2", "activation_mlp2"}:
        return "monomial_activation_mlp2"
    if name in {"monomial_weight_mlp2", "weight_mlp2"}:
        return "monomial_weight_mlp2"
    if name in {"monomial_shrinkage_mlp2", "shrinkage_mlp2"}:
        return "monomial_shrinkage_mlp2"
    if name in {"monomial_global_ls_mlp2", "monomial_global_mlp2", "global_ls_mlp2"}:
        return "monomial_global_ls_mlp2"
    raise ValueError(f"unknown monomial matching protocol: {matching}")


def _matching_family(matching: str) -> str:
    protocol = _canonical_matching(matching)
    if protocol in {"monomial_weight", "monomial_weight_mlp2"}:
        return "weight"
    return "activation"


def _scale_method_for_matching(matching: str, scale_method: str) -> str:
    protocol = _canonical_matching(matching)
    requested = _canonical_scale_method(scale_method)
    if requested != "raw":
        return requested
    if protocol == "monomial_shrinkage_mlp2":
        return "shrinkage"
    if protocol == "monomial_global_ls_mlp2":
        return "global_synchronized"
    return requested


def _clip_positive(values: Iterable[float], min_scale: float, max_scale: float) -> np.ndarray:
    out = np.asarray(list(values), dtype=float)
    out[~np.isfinite(out)] = 1.0
    out[out <= 0.0] = 1.0
    return np.clip(out, min_scale, max_scale)


def _canonical_scale_method(scale_method: str) -> str:
    name = str(scale_method).strip().lower()
    aliases = {
        "raw": "raw",
        "clipped": "clipped",
        "clip": "clipped",
        "shrinkage": "shrinkage",
        "shrunk": "shrinkage",
        "global": "global_synchronized",
        "global_synchronized": "global_synchronized",
        "global-synchronized": "global_synchronized",
    }
    if name not in aliases:
        raise ValueError(f"unknown monomial scale method: {scale_method}")
    return aliases[name]


def _center_normalize_columns(features: np.ndarray) -> np.ndarray:
    centered = np.asarray(features, dtype=float) - np.asarray(features, dtype=float).mean(axis=0, keepdims=True)
    return centered / np.maximum(np.linalg.norm(centered, axis=0, keepdims=True), 1e-12)


def activation_assignment_similarities(
    reference_features: np.ndarray,
    target_features: np.ndarray,
    permutation: np.ndarray,
) -> np.ndarray:
    """Return centered cosine similarities for assigned hidden activations."""

    ref = _center_normalize_columns(reference_features)
    tgt = _center_normalize_columns(target_features)
    perm = np.asarray(permutation, dtype=int)
    similarity = ref.T @ tgt
    return np.asarray(similarity[np.arange(len(perm)), perm], dtype=float)


def _regularized_positive_scales(
    raw_scales: Iterable[float],
    *,
    scale_method: str,
    min_scale: float,
    max_scale: float,
    log_scale_clip: float,
    shrinkage: float,
    assignment_similarity: np.ndarray | None,
    activation_similarity_threshold: float,
) -> tuple[np.ndarray, float, float, float]:
    """Sanitize and regularize positive scales in log space."""

    method = _canonical_scale_method(scale_method)
    scales = _clip_positive(raw_scales, min_scale, max_scale)
    logs = np.log(np.maximum(scales, 1e-300))
    if method in {"clipped", "shrinkage", "global_synchronized"}:
        clip = abs(float(log_scale_clip))
        logs = np.clip(logs, -clip, clip)
    if method in {"shrinkage", "global_synchronized"}:
        shrink = min(1.0, max(0.0, float(shrinkage)))
        logs = (1.0 - shrink) * logs

    similarities = None if assignment_similarity is None else np.asarray(assignment_similarity, dtype=float)
    if similarities is not None and len(similarities) == len(logs):
        valid = np.isfinite(similarities)
        low = (~valid) | (similarities < float(activation_similarity_threshold))
        if np.any(low):
            if method == "raw":
                logs[low] = 0.0
            else:
                logs[low] *= 1.0 - min(1.0, max(0.0, float(shrinkage)))
        low_fraction = float(np.mean(low))
        sim_mean = float(np.mean(similarities[valid])) if np.any(valid) else float("nan")
        sim_min = float(np.min(similarities[valid])) if np.any(valid) else float("nan")
    else:
        low_fraction = float("nan")
        sim_mean = float("nan")
        sim_min = float("nan")
    return np.exp(logs), sim_mean, sim_min, low_fraction


def _activation_positive_scales(
    reference_features: np.ndarray,
    target_features: np.ndarray,
    permutation: np.ndarray,
    min_scale: float,
    max_scale: float,
) -> np.ndarray:
    ref = np.asarray(reference_features, dtype=float)
    tgt = np.asarray(target_features, dtype=float)
    scales = []
    for unit, target_unit in enumerate(np.asarray(permutation, dtype=int)):
        a = ref[:, unit]
        b = tgt[:, target_unit]
        denom = max(float(np.dot(a, a)), 1e-12)
        scale = float(np.dot(a, b) / denom)
        scales.append(scale)
    return _clip_positive(scales, min_scale, max_scale)


def _weight_positive_scales_mlp(reference_model, target_model, permutation: np.ndarray, min_scale: float, max_scale: float) -> np.ndarray:
    ref_in = reference_model.hidden.weight.detach().cpu().numpy()
    tgt_in = target_model.hidden.weight.detach().cpu().numpy()
    ref_bias = reference_model.hidden.bias.detach().cpu().numpy()
    tgt_bias = target_model.hidden.bias.detach().cpu().numpy()
    ref_out = reference_model.classifier.weight.detach().cpu().numpy()
    tgt_out = target_model.classifier.weight.detach().cpu().numpy()

    ref_in_norm = np.sqrt(np.sum(ref_in * ref_in, axis=1) + ref_bias * ref_bias)
    tgt_in_norm = np.sqrt(np.sum(tgt_in * tgt_in, axis=1) + tgt_bias * tgt_bias)
    ref_out_norm = np.linalg.norm(ref_out, axis=0)
    tgt_out_norm = np.linalg.norm(tgt_out, axis=0)

    scales = []
    for unit, target_unit in enumerate(np.asarray(permutation, dtype=int)):
        candidates = []
        if ref_in_norm[unit] > 1e-12:
            candidates.append(float(tgt_in_norm[target_unit] / ref_in_norm[unit]))
        if tgt_out_norm[target_unit] > 1e-12:
            candidates.append(float(ref_out_norm[unit] / tgt_out_norm[target_unit]))
        if candidates:
            positive = [value for value in candidates if np.isfinite(value) and value > 0.0]
            scale = float(np.exp(np.mean(np.log(positive)))) if positive else 1.0
        else:
            scale = 1.0
        scales.append(scale)
    return _clip_positive(scales, min_scale, max_scale)


def _weight_positive_scales_mlp2(
    reference_model,
    target_model,
    layer: str,
    permutation: np.ndarray,
    hidden1_permutation: np.ndarray,
    hidden2_permutation: np.ndarray,
    min_scale: float,
    max_scale: float,
) -> np.ndarray:
    if layer == "hidden1":
        ref_in = reference_model.hidden1.weight.detach().cpu().numpy()
        tgt_in = target_model.hidden1.weight.detach().cpu().numpy()
        ref_bias = reference_model.hidden1.bias.detach().cpu().numpy()
        tgt_bias = target_model.hidden1.bias.detach().cpu().numpy()
        ref_out = reference_model.hidden2.weight.detach().cpu().numpy()
        tgt_out = target_model.hidden2.weight.detach().cpu().numpy()[hidden2_permutation, :]
        ref_in_norm = np.sqrt(np.sum(ref_in * ref_in, axis=1) + ref_bias * ref_bias)
        tgt_in_norm = np.sqrt(np.sum(tgt_in * tgt_in, axis=1) + tgt_bias * tgt_bias)
        ref_out_norm = np.linalg.norm(ref_out, axis=0)
        tgt_out_norm = np.linalg.norm(tgt_out, axis=0)
    elif layer == "hidden2":
        ref_in = reference_model.hidden2.weight.detach().cpu().numpy()
        tgt_in = target_model.hidden2.weight.detach().cpu().numpy()
        ref_bias = reference_model.hidden2.bias.detach().cpu().numpy()
        tgt_bias = target_model.hidden2.bias.detach().cpu().numpy()
        ref_out = reference_model.classifier.weight.detach().cpu().numpy()
        tgt_out = target_model.classifier.weight.detach().cpu().numpy()
        # Row norms are computed after aligning hidden1 columns so the row score
        # is not dominated by an unrelated first-layer permutation.
        ref_in_norm = np.sqrt(np.sum(ref_in * ref_in, axis=1) + ref_bias * ref_bias)
        tgt_in_aligned = tgt_in[:, hidden1_permutation]
        tgt_in_norm = np.sqrt(np.sum(tgt_in_aligned * tgt_in_aligned, axis=1) + tgt_bias * tgt_bias)
        ref_out_norm = np.linalg.norm(ref_out, axis=0)
        tgt_out_norm = np.linalg.norm(tgt_out, axis=0)
    else:
        raise ValueError(f"unknown mlp2 monomial layer: {layer}")

    scales = []
    for unit, target_unit in enumerate(np.asarray(permutation, dtype=int)):
        candidates = []
        if ref_in_norm[unit] > 1e-12:
            candidates.append(float(tgt_in_norm[target_unit] / ref_in_norm[unit]))
        if tgt_out_norm[target_unit] > 1e-12:
            candidates.append(float(ref_out_norm[unit] / tgt_out_norm[target_unit]))
        positive = [value for value in candidates if np.isfinite(value) and value > 0.0]
        scales.append(float(np.exp(np.mean(np.log(positive)))) if positive else 1.0)
    return _clip_positive(scales, min_scale, max_scale)


def transform_mlp2_positive_scale(
    model,
    spec,
    width: int,
    hidden1_perm: np.ndarray,
    hidden1_scales: np.ndarray,
    hidden2_perm: np.ndarray,
    hidden2_scales: np.ndarray,
):
    """Return an exact positive monomial rewrite of a two-hidden-layer ReLU MLP.

    ``hidden*_perm`` follows the existing reference-to-model convention:
    reference unit ``r`` corresponds to source/model unit ``perm[r]``.
    ``hidden*_scales[r]`` is the source feature scale relative to the new
    reference-coordinate feature.  All scales must be strictly positive.
    """

    torch, _, _ = require_torch()
    p1 = np.asarray(hidden1_perm, dtype=int)
    p2 = np.asarray(hidden2_perm, dtype=int)
    s1_np = np.asarray(hidden1_scales, dtype=float)
    s2_np = np.asarray(hidden2_scales, dtype=float)
    _validate_perm_and_scales(p1, s1_np, "hidden1")
    _validate_perm_and_scales(p2, s2_np, "hidden2")
    if len(p1) != width or len(p2) != width:
        raise ValueError("mlp2 monomial gauges require hidden1 and hidden2 widths to match width")

    aligned = make_model("mlp2", spec, width)
    with torch.no_grad():
        w1 = model.hidden1.weight.detach().cpu()
        b1 = model.hidden1.bias.detach().cpu()
        w2 = model.hidden2.weight.detach().cpu()
        b2 = model.hidden2.bias.detach().cpu()
        cout = model.classifier.weight.detach().cpu()
        bout = model.classifier.bias.detach().cpu()
        s1 = torch.tensor(s1_np, dtype=w1.dtype)
        s2 = torch.tensor(s2_np, dtype=w2.dtype)

        aligned.hidden1.weight.copy_(w1[p1, :] / s1.unsqueeze(1))
        aligned.hidden1.bias.copy_(b1[p1] / s1)
        # h1_source[p1[r]] = s1[r] * h1_aligned[r].  Then divide the selected
        # hidden2 preactivation by s2 so ReLU(z) / s2 = ReLU(z / s2).
        aligned.hidden2.weight.copy_(w2[p2, :][:, p1] * s1.unsqueeze(0) / s2.unsqueeze(1))
        aligned.hidden2.bias.copy_(b2[p2] / s2)
        aligned.classifier.weight.copy_(cout[:, p2] * s2.unsqueeze(0))
        aligned.classifier.bias.copy_(bout)
    return aligned


def estimate_monomial_alignment(
    reference_model,
    target_model,
    loader=None,
    device="cpu",
    *,
    matching: str = "monomial_activation",
    max_batches: int = 8,
    min_scale: float = 1e-3,
    max_scale: float = 1e3,
    scale_method: str = "raw",
    log_scale_clip: float = 2.0,
    shrinkage: float = 0.5,
    activation_similarity_threshold: float = 0.2,
) -> MonomialAlignment:
    """Estimate a permutation-plus-positive-scale alignment for MLP hidden units."""

    architecture = _infer_architecture(reference_model)
    target_architecture = _infer_architecture(target_model)
    if target_architecture != architecture:
        raise ValueError("reference_model and target_model must have the same monomial-supported architecture")
    protocol = _canonical_matching(matching)
    method = _scale_method_for_matching(protocol, scale_method)
    family = _matching_family(protocol)
    if architecture == "mlp2" and protocol in {"monomial_activation", "monomial_weight"}:
        # Keep the old names mlp-only so accidental mlp2 runs are explicit in
        # the report/config.  The new *_mlp2 modes use the same estimators.
        raise ValueError(f"{protocol} is the one-hidden-layer mode; use {protocol}_mlp2 for mlp2")
    if architecture == "mlp" and protocol.endswith("_mlp2"):
        raise ValueError(f"{protocol} requires architecture mlp2")

    layer_perms: dict[str, np.ndarray] = {}
    raw_scales_by_layer: dict[str, np.ndarray] = {}
    similarities_by_layer: dict[str, np.ndarray | None] = {}
    layers = _layers_for_architecture(architecture)

    if family == "activation":
        if loader is None:
            raise ValueError("activation monomial alignment requires a feature loader")
        for layer in layers:
            reference_features = collect_features(reference_model, loader, device, max_batches=max_batches, layer=layer)
            target_features = collect_features(target_model, loader, device, max_batches=max_batches, layer=layer)
            permutation = activation_permutation(reference_features, target_features)
            layer_perms[layer] = permutation
            raw_scales_by_layer[layer] = _activation_positive_scales(
                reference_features,
                target_features,
                permutation,
                min_scale,
                max_scale,
            )
            similarities_by_layer[layer] = activation_assignment_similarities(reference_features, target_features, permutation)
        scale_source = "activation_least_squares"
    else:
        if architecture == "mlp":
            permutation = weight_permutation(reference_model, target_model, "mlp", layer="hidden")
            layer_perms["hidden"] = permutation
            raw_scales_by_layer["hidden"] = _weight_positive_scales_mlp(
                reference_model,
                target_model,
                permutation,
                min_scale,
                max_scale,
            )
            similarities_by_layer["hidden"] = None
        else:
            p1 = weight_permutation(reference_model, target_model, "mlp2", layer="hidden1")
            p2 = weight_permutation(reference_model, target_model, "mlp2", layer="hidden2")
            layer_perms["hidden1"] = p1
            layer_perms["hidden2"] = p2
            raw_scales_by_layer["hidden1"] = _weight_positive_scales_mlp2(
                reference_model,
                target_model,
                "hidden1",
                p1,
                p1,
                p2,
                min_scale,
                max_scale,
            )
            raw_scales_by_layer["hidden2"] = _weight_positive_scales_mlp2(
                reference_model,
                target_model,
                "hidden2",
                p2,
                p1,
                p2,
                min_scale,
                max_scale,
            )
            similarities_by_layer["hidden1"] = None
            similarities_by_layer["hidden2"] = None
        scale_source = "weight_norm_geometric_mean"

    layer_scales: dict[str, np.ndarray] = {}
    sim_means = []
    sim_mins = []
    low_fractions = []
    for layer in layers:
        scales, sim_mean, sim_min, low_fraction = _regularized_positive_scales(
            raw_scales_by_layer[layer],
            scale_method=method,
            min_scale=min_scale,
            max_scale=max_scale,
            log_scale_clip=log_scale_clip,
            shrinkage=shrinkage,
            assignment_similarity=similarities_by_layer[layer],
            activation_similarity_threshold=activation_similarity_threshold,
        )
        layer_scales[layer] = scales
        sim_means.append(sim_mean)
        sim_mins.append(sim_min)
        low_fractions.append(low_fraction)

    primary_layer = _primary_layer_for_architecture(architecture)
    return MonomialAlignment(
        layer_perms[primary_layer],
        layer_scales[primary_layer],
        matching=protocol,
        scale_source=scale_source,
        scale_method=method,
        architecture=architecture,
        primary_layer=primary_layer,
        layer_permutations=layer_perms,
        layer_positive_scales=layer_scales,
        assignment_similarity_mean=float(np.nanmean(sim_means)) if np.isfinite(sim_means).any() else float("nan"),
        assignment_similarity_min=float(np.nanmin(sim_mins)) if np.isfinite(sim_mins).any() else float("nan"),
        low_similarity_fraction=float(np.nanmean(low_fractions)) if np.isfinite(low_fractions).any() else float("nan"),
    )


def synchronize_monomial_log_scales(
    alignments: dict[tuple[int, int], MonomialAlignment],
    n_models: int,
) -> dict[tuple[int, int], MonomialAlignment]:
    """Project pairwise log-scales to globally synchronized hidden gauges."""

    if not alignments:
        return {}
    template = next(iter(alignments.values()))
    layers = template.layers()
    layer_synchronized_scales: dict[str, dict[tuple[int, int], np.ndarray]] = {}
    for layer in layers:
        width = len(template.positive_scales_for(layer))
        n_variables = n_models * width
        equations = []
        rhs = []
        for (i, j), alignment in alignments.items():
            if i == j:
                continue
            perm = alignment.permutation_for(layer)
            logs = np.log(np.maximum(alignment.positive_scales_for(layer), 1e-300))
            for unit, target_unit in enumerate(perm):
                row = np.zeros(n_variables, dtype=float)
                row[j * width + int(target_unit)] = 1.0
                row[i * width + int(unit)] = -1.0
                equations.append(row)
                rhs.append(float(logs[unit]))
        for unit in range(width):
            row = np.zeros(n_variables, dtype=float)
            row[unit] = 1.0
            equations.append(row)
            rhs.append(0.0)
        if not equations:
            layer_synchronized_scales[layer] = {
                pair: alignment.positive_scales_for(layer) for pair, alignment in alignments.items()
            }
            continue
        design = np.vstack(equations)
        target = np.asarray(rhs, dtype=float)
        gauges, *_ = np.linalg.lstsq(design, target, rcond=None)
        synchronized_for_layer = {}
        for pair, alignment in alignments.items():
            i, j = pair
            if i == j:
                scales = np.ones(width, dtype=float)
            else:
                perm = alignment.permutation_for(layer)
                logs = np.empty(width, dtype=float)
                for unit, target_unit in enumerate(perm):
                    logs[unit] = gauges[j * width + int(target_unit)] - gauges[i * width + int(unit)]
                scales = np.exp(logs)
            synchronized_for_layer[pair] = scales
        layer_synchronized_scales[layer] = synchronized_for_layer

    out: dict[tuple[int, int], MonomialAlignment] = {}
    for pair, alignment in alignments.items():
        layer_scales = {layer: layer_synchronized_scales[layer][pair] for layer in layers}
        primary = alignment.primary_layer
        out[pair] = MonomialAlignment(
            alignment.permutation_for(primary),
            layer_scales[primary],
            matching=alignment.matching,
            scale_source="global_synchronized_log_least_squares",
            scale_method="global_synchronized",
            architecture=alignment.architecture,
            primary_layer=primary,
            layer_permutations={layer: alignment.permutation_for(layer) for layer in layers},
            layer_positive_scales=layer_scales,
            assignment_similarity_mean=alignment.assignment_similarity_mean,
            assignment_similarity_min=alignment.assignment_similarity_min,
            low_similarity_fraction=alignment.low_similarity_fraction,
        )
    return out


def estimate_pairwise_monomial_alignments(
    models: list,
    loader,
    device,
    *,
    matching: str = "monomial_activation",
    max_batches: int = 8,
    min_scale: float = 1e-3,
    max_scale: float = 1e3,
    scale_method: str = "raw",
    log_scale_clip: float = 2.0,
    shrinkage: float = 0.5,
    activation_similarity_threshold: float = 0.2,
) -> dict[tuple[int, int], MonomialAlignment]:
    """Estimate directed monomial alignments for all ordered model pairs."""

    architecture = _infer_architecture(models[0])
    layers = _layers_for_architecture(architecture)
    primary = _primary_layer_for_architecture(architecture)
    protocol = _canonical_matching(matching)
    method = _scale_method_for_matching(protocol, scale_method)
    identity_perms = {layer: np.arange(_width_for_layer(models[0], layer), dtype=int) for layer in layers}
    identity_scales = {layer: np.ones(_width_for_layer(models[0], layer), dtype=float) for layer in layers}
    out: dict[tuple[int, int], MonomialAlignment] = {}
    for i, j in product(range(len(models)), repeat=2):
        if i == j:
            out[(i, j)] = MonomialAlignment(
                identity_perms[primary],
                identity_scales[primary],
                matching=protocol,
                scale_source="identity",
                scale_method=method,
                architecture=architecture,
                primary_layer=primary,
                layer_permutations=identity_perms,
                layer_positive_scales=identity_scales,
                assignment_similarity_mean=1.0,
                assignment_similarity_min=1.0,
                low_similarity_fraction=0.0,
            )
        else:
            out[(i, j)] = estimate_monomial_alignment(
                models[i],
                models[j],
                loader,
                device,
                matching=matching,
                max_batches=max_batches,
                min_scale=min_scale,
                max_scale=max_scale,
                scale_method="shrinkage" if method == "global_synchronized" else method,
                log_scale_clip=log_scale_clip,
                shrinkage=shrinkage,
                activation_similarity_threshold=activation_similarity_threshold,
            )
    if method == "global_synchronized":
        out = synchronize_monomial_log_scales(out, len(models))
    return out


def apply_monomial_alignment_to_reference(model, spec, width: int, alignment: MonomialAlignment):
    """Return ``model`` rewritten in the reference hidden coordinate system."""

    architecture = alignment.architecture if alignment.architecture else _infer_architecture(model)
    if architecture == "mlp":
        return transform_mlp_positive_scale(model, spec, width, alignment.permutation, alignment.positive_scales)
    if architecture == "mlp2":
        return transform_mlp2_positive_scale(
            model,
            spec,
            width,
            alignment.permutation_for("hidden1"),
            alignment.positive_scales_for("hidden1"),
            alignment.permutation_for("hidden2"),
            alignment.positive_scales_for("hidden2"),
        )
    raise ValueError(f"unsupported monomial architecture: {architecture}")


def monomial_permutations_by_layer(
    alignments: dict[tuple[int, int], MonomialAlignment],
) -> dict[str, dict[tuple[int, int], np.ndarray]]:
    """Extract layerwise permutation observations from pairwise monomial maps."""

    if not alignments:
        return {}
    layers = next(iter(alignments.values())).layers()
    return {
        layer: {pair: alignment.permutation_for(layer) for pair, alignment in alignments.items()}
        for layer in layers
    }


def invert_monomial_alignment(alignment: MonomialAlignment) -> MonomialAlignment:
    """Return the inverse directed monomial alignment."""

    layer_perms = {}
    layer_scales = {}
    for layer in alignment.layers():
        perm = alignment.permutation_for(layer)
        inv = np.empty_like(perm)
        inv[perm] = np.arange(len(perm))
        scales = alignment.positive_scales_for(layer)
        layer_perms[layer] = inv
        layer_scales[layer] = 1.0 / scales[inv]
    primary = alignment.primary_layer
    return MonomialAlignment(
        layer_perms[primary],
        layer_scales[primary],
        matching=alignment.matching,
        scale_source=f"inverse({alignment.scale_source})",
        scale_method=alignment.scale_method,
        architecture=alignment.architecture,
        primary_layer=primary,
        layer_permutations=layer_perms,
        layer_positive_scales=layer_scales,
        assignment_similarity_mean=alignment.assignment_similarity_mean,
        assignment_similarity_min=alignment.assignment_similarity_min,
        low_similarity_fraction=alignment.low_similarity_fraction,
    )


def compose_monomial_alignments(first: MonomialAlignment, second: MonomialAlignment) -> MonomialAlignment:
    """Compose two directed monomial maps represented in reference-to-target form."""

    if first.architecture != second.architecture or first.layers() != second.layers():
        raise ValueError("cannot compose monomial alignments with different layer structures")
    layer_perms = {}
    layer_scales = {}
    for layer in first.layers():
        p_ab = first.permutation_for(layer)
        p_bc = second.permutation_for(layer)
        s_ab = first.positive_scales_for(layer)
        s_bc = second.positive_scales_for(layer)
        layer_perms[layer] = p_bc[p_ab]
        layer_scales[layer] = s_ab * s_bc[p_ab]
    primary = first.primary_layer
    return MonomialAlignment(
        layer_perms[primary],
        layer_scales[primary],
        matching=first.matching,
        scale_source=f"compose({first.scale_source},{second.scale_source})",
        scale_method=first.scale_method,
        architecture=first.architecture,
        primary_layer=primary,
        layer_permutations=layer_perms,
        layer_positive_scales=layer_scales,
        assignment_similarity_mean=float("nan"),
        assignment_similarity_min=float("nan"),
        low_similarity_fraction=float("nan"),
    )


def monomial_matrix(alignment: MonomialAlignment) -> np.ndarray:
    """Return the row-scaled permutation matrix for a pairwise monomial map."""

    return np.diag(alignment.positive_scales) @ permutation_matrix(alignment.permutation)


def monomial_cycle_defect(
    alignments: dict[tuple[int, int], MonomialAlignment],
    i: int,
    j: int,
    k: int,
) -> np.ndarray:
    """Return the monomial triangle holonomy ``M_ij M_jk M_ki``."""

    return monomial_matrix(alignments[(i, j)]) @ monomial_matrix(alignments[(j, k)]) @ monomial_matrix(alignments[(k, i)])


def monomial_defect_score(cycle_map: np.ndarray) -> float:
    """Frobenius distance from identity, normalized by ``||I||_F``."""

    matrix = np.asarray(cycle_map, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("cycle_map must be a square matrix")
    width = int(matrix.shape[0])
    return float(np.linalg.norm(matrix - np.eye(width), ord="fro") / max(np.sqrt(width), 1e-12))


def average_monomial_defect_score(
    alignments: dict[tuple[int, int], MonomialAlignment],
    n_models: int,
) -> tuple[float, list[dict[str, float]]]:
    """Compute the mean monomial triangle score and per-triangle rows."""

    rows = []
    scores = []
    for i, j, k in combinations(range(n_models), 3):
        cycle_map = monomial_cycle_defect(alignments, i, j, k)
        score = monomial_defect_score(cycle_map)
        scores.append(score)
        rows.append(
            {
                "i": i,
                "j": j,
                "k": k,
                "monomial_defect_score": score,
                "monomial_cycle_trace": float(np.trace(cycle_map)),
                "monomial_cycle_determinant": float(np.linalg.det(cycle_map)),
            }
        )
    return (float(np.mean(scores)) if scores else 0.0), rows


def monomial_scaling_statistics(alignments: dict[tuple[int, int], MonomialAlignment]) -> dict[str, float]:
    """Summarize positive-scale magnitudes across non-identity pairwise maps."""

    scale_blocks = []
    for (i, j), alignment in alignments.items():
        if i == j:
            continue
        for layer in alignment.layers():
            scales = alignment.positive_scales_for(layer)
            if len(scales) > 0:
                scale_blocks.append(scales)
    nonidentity = [alignment for (i, j), alignment in alignments.items() if i != j]
    if not scale_blocks:
        return {
            "monomial_scale_min": float("nan"),
            "monomial_scale_max": float("nan"),
            "monomial_scale_mean": float("nan"),
            "monomial_scale_std": float("nan"),
            "monomial_mean_abs_log_scale": float("nan"),
            "monomial_max_abs_log_scale": float("nan"),
            "monomial_log_scale_variance": float("nan"),
            "monomial_scale_assignment_similarity_mean": float("nan"),
            "monomial_scale_assignment_similarity_min": float("nan"),
            "monomial_scale_low_similarity_fraction": float("nan"),
        }
    scales = np.concatenate(scale_blocks).astype(float)
    logs = np.log(np.maximum(scales, 1e-300))
    sim_means = [alignment.assignment_similarity_mean for alignment in nonidentity]
    sim_mins = [alignment.assignment_similarity_min for alignment in nonidentity]
    low_fractions = [alignment.low_similarity_fraction for alignment in nonidentity]
    return {
        "monomial_scale_min": float(np.min(scales)),
        "monomial_scale_max": float(np.max(scales)),
        "monomial_scale_mean": float(np.mean(scales)),
        "monomial_scale_std": float(np.std(scales)),
        "monomial_mean_abs_log_scale": float(np.mean(np.abs(logs))),
        "monomial_max_abs_log_scale": float(np.max(np.abs(logs))),
        "monomial_log_scale_variance": float(np.var(logs)),
        "monomial_scale_assignment_similarity_mean": float(np.nanmean(sim_means)) if np.isfinite(sim_means).any() else float("nan"),
        "monomial_scale_assignment_similarity_min": float(np.nanmin(sim_mins)) if np.isfinite(sim_mins).any() else float("nan"),
        "monomial_scale_low_similarity_fraction": float(np.nanmean(low_fractions)) if np.isfinite(low_fractions).any() else float("nan"),
    }


def compare_function_before_after_alignment(
    original_model,
    aligned_model,
    loader,
    device,
    *,
    max_batches: int = 8,
) -> dict[str, float]:
    """Compare logits before and after an exact gauge rewrite."""

    torch, _, _ = require_torch()
    original_model.to(device)
    aligned_model.to(device)
    original_model.eval()
    aligned_model.eval()
    max_abs = 0.0
    sum_abs = 0.0
    sum_sq = 0.0
    n_values = 0
    n_examples = 0
    n_disagree = 0
    with torch.no_grad():
        for batch_idx, (x, _y) in enumerate(loader):
            if batch_idx >= max_batches:
                break
            x = x.to(device)
            original_logits = original_model(x)
            aligned_logits = aligned_model(x)
            diff = (original_logits - aligned_logits).detach().cpu()
            max_abs = max(max_abs, float(diff.abs().max().item()))
            sum_abs += float(diff.abs().sum().item())
            sum_sq += float((diff * diff).sum().item())
            n_values += int(diff.numel())
            n_examples += int(diff.shape[0])
            n_disagree += int((original_logits.argmax(dim=1) != aligned_logits.argmax(dim=1)).sum().item())
    denom = max(n_values, 1)
    return {
        "functional_preservation_error": float(max_abs),
        "functional_preservation_mean_abs_error": float(sum_abs / denom),
        "functional_preservation_rms_error": float(np.sqrt(sum_sq / denom)),
        "functional_preservation_prediction_disagreement": float(n_disagree / max(n_examples, 1)),
        "functional_preservation_examples": float(n_examples),
    }
