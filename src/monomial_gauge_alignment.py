"""ReLU-compatible monomial gauge alignment for one-hidden-layer MLPs.

The hidden-unit symmetry used here is exact for one-hidden-layer ReLU MLPs:
permuting hidden units and applying a positive diagonal scale to incoming
hidden weights, with the inverse effect absorbed by outgoing classifier
columns, preserves the represented function.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
from typing import Iterable

import numpy as np

from src.ladder_merge_methods import transform_mlp_positive_scale
from src.model_merging_benchmark import (
    activation_permutation,
    collect_features,
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
    """

    permutation: np.ndarray
    positive_scales: np.ndarray
    matching: str
    scale_source: str

    def __post_init__(self) -> None:
        perm = np.asarray(self.permutation, dtype=int)
        scales = np.asarray(self.positive_scales, dtype=float)
        if perm.ndim != 1 or scales.ndim != 1 or perm.shape[0] != scales.shape[0]:
            raise ValueError("permutation and positive_scales must be one-dimensional arrays of the same length")
        if sorted(perm.tolist()) != list(range(len(perm))):
            raise ValueError("permutation must be a valid permutation")
        if np.any(scales <= 0.0) or not np.all(np.isfinite(scales)):
            raise ValueError("positive_scales must be finite and strictly positive")
        object.__setattr__(self, "permutation", perm.copy())
        object.__setattr__(self, "positive_scales", scales.copy())


def _canonical_matching(matching: str) -> str:
    name = str(matching).strip().lower()
    if name in {"activation", "monomial_activation"}:
        return "monomial_activation"
    if name in {"weight", "monomial_weight"}:
        return "monomial_weight"
    raise ValueError(f"unknown monomial matching protocol: {matching}")


def _clip_positive(values: Iterable[float], min_scale: float, max_scale: float) -> np.ndarray:
    out = np.asarray(list(values), dtype=float)
    out[~np.isfinite(out)] = 1.0
    out[out <= 0.0] = 1.0
    return np.clip(out, min_scale, max_scale)


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


def _weight_positive_scales(reference_model, target_model, permutation: np.ndarray, min_scale: float, max_scale: float) -> np.ndarray:
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
) -> MonomialAlignment:
    """Estimate a permutation-plus-positive-scale alignment for MLP hidden units."""

    protocol = _canonical_matching(matching)
    if protocol == "monomial_activation":
        if loader is None:
            raise ValueError("activation monomial alignment requires a feature loader")
        reference_features = collect_features(reference_model, loader, device, max_batches=max_batches)
        target_features = collect_features(target_model, loader, device, max_batches=max_batches)
        permutation = activation_permutation(reference_features, target_features)
        scales = _activation_positive_scales(reference_features, target_features, permutation, min_scale, max_scale)
        return MonomialAlignment(permutation, scales, matching=protocol, scale_source="activation_least_squares")

    permutation = weight_permutation(reference_model, target_model, "mlp", layer="hidden")
    scales = _weight_positive_scales(reference_model, target_model, permutation, min_scale, max_scale)
    return MonomialAlignment(permutation, scales, matching=protocol, scale_source="weight_norm_geometric_mean")


def estimate_pairwise_monomial_alignments(
    models: list,
    loader,
    device,
    *,
    matching: str = "monomial_activation",
    max_batches: int = 8,
    min_scale: float = 1e-3,
    max_scale: float = 1e3,
) -> dict[tuple[int, int], MonomialAlignment]:
    """Estimate directed monomial alignments for all ordered model pairs."""

    width = int(models[0].hidden.out_features)
    identity = MonomialAlignment(
        np.arange(width, dtype=int),
        np.ones(width, dtype=float),
        matching=_canonical_matching(matching),
        scale_source="identity",
    )
    out: dict[tuple[int, int], MonomialAlignment] = {}
    for i, j in product(range(len(models)), repeat=2):
        if i == j:
            out[(i, j)] = identity
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
            )
    return out


def apply_monomial_alignment_to_reference(model, spec, width: int, alignment: MonomialAlignment):
    """Return ``model`` rewritten in the reference hidden coordinate system."""

    return transform_mlp_positive_scale(model, spec, width, alignment.permutation, alignment.positive_scales)


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

    scale_blocks = [
        alignment.positive_scales
        for (i, j), alignment in alignments.items()
        if i != j and len(alignment.positive_scales) > 0
    ]
    if not scale_blocks:
        return {
            "monomial_scale_min": float("nan"),
            "monomial_scale_max": float("nan"),
            "monomial_scale_mean": float("nan"),
            "monomial_scale_std": float("nan"),
            "monomial_mean_abs_log_scale": float("nan"),
            "monomial_max_abs_log_scale": float("nan"),
            "monomial_log_scale_variance": float("nan"),
        }
    scales = np.concatenate(scale_blocks).astype(float)
    logs = np.log(np.maximum(scales, 1e-300))
    return {
        "monomial_scale_min": float(np.min(scales)),
        "monomial_scale_max": float(np.max(scales)),
        "monomial_scale_mean": float(np.mean(scales)),
        "monomial_scale_std": float(np.std(scales)),
        "monomial_mean_abs_log_scale": float(np.mean(np.abs(logs))),
        "monomial_max_abs_log_scale": float(np.max(np.abs(logs))),
        "monomial_log_scale_variance": float(np.var(logs)),
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
