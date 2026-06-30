"""Structure-group ladder diagnostics for model-merging residuals.

This module extends the TwistedMerge++ residual classifier with a conservative
ladder over larger structure groups.  It is diagnostic only: central/projective
language is used only when the observed triangle defects are actually close to
scalar roots of unity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations, product
from typing import Mapping

import numpy as np

from .finite_index_twists import determinant_obstruction_allows
from .model_merging_benchmark import (
    activation_permutation,
    compose_perm,
    invert_perm,
    permutation_disagreement,
    permutation_matrix,
    synchronize_permutations,
)
from .noncentral_holonomy import detect_scalar_phase


IndexPair = tuple[int, int]
Triple = tuple[int, int, int]
Alignment = np.ndarray


@dataclass(frozen=True)
class StructureGroupLevel:
    name: str
    description: str


@dataclass
class LadderDiagnostics:
    level: str
    residual_type: str
    centrality_score: float
    phase_residual: float | None
    detected_order_d: int | None
    cycle_score: float
    rank_allowed: bool | None
    selected_resolution: str
    notes: list[str] = field(default_factory=list)
    centrality_improvement_from_previous_level: float | None = None
    supports_brauer_projective_interpretation: bool = False
    is_finite_index_candidate: bool = False


@dataclass(frozen=True)
class LadderResult:
    final_decision: str
    selected_level: str | None
    diagnostics: list[LadderDiagnostics]
    notes: tuple[str, ...] = ()


LEVELS = (
    StructureGroupLevel("permutation", "Permutation matrices in S_h."),
    StructureGroupLevel("signed_permutation", "Signed permutation matrices."),
    StructureGroupLevel("monomial_phase_or_scale", "Diagonal sign/phase/scale times permutation."),
    StructureGroupLevel("block_orthogonal", "Synthetic block-orthogonal gauges."),
    StructureGroupLevel("low_rank_GL", "Activation least-squares or exact GL gauges for diagnostics."),
)


def default_triples(n_models: int) -> list[Triple]:
    return list(combinations(range(n_models), 3))


def _is_pair_keyed(mapping: Mapping) -> bool:
    return all(isinstance(key, tuple) and len(key) == 2 for key in mapping)


def _infer_width(pairwise: Mapping[IndexPair, Alignment], width: int | None) -> int:
    if width is not None:
        return int(width)
    first = np.asarray(next(iter(pairwise.values())))
    return int(first.shape[0])


def _is_permutation_payload(pairwise: Mapping[IndexPair, Alignment]) -> bool:
    return all(np.asarray(value).ndim == 1 for value in pairwise.values())


def complete_permutations(
    pairwise: Mapping[IndexPair, Alignment],
    n_models: int,
    width: int,
) -> dict[IndexPair, np.ndarray]:
    completed: dict[IndexPair, np.ndarray] = {}
    for i in range(n_models):
        completed[(i, i)] = np.arange(width, dtype=int)
    for pair, perm in pairwise.items():
        i, j = pair
        value = np.asarray(perm, dtype=int)
        if value.ndim != 1:
            raise ValueError("permutation alignments must be rank-1 arrays")
        completed[(i, j)] = value
        if (j, i) not in completed:
            completed[(j, i)] = invert_perm(value)
    missing = [(i, j) for i, j in product(range(n_models), repeat=2) if (i, j) not in completed]
    if missing:
        raise ValueError(f"missing pairwise permutation alignments: {missing[:5]}")
    return completed


def complete_matrices(
    pairwise: Mapping[IndexPair, Alignment],
    n_models: int,
    width: int,
) -> dict[IndexPair, np.ndarray]:
    completed: dict[IndexPair, np.ndarray] = {}
    eye = np.eye(width, dtype=complex)
    for i in range(n_models):
        completed[(i, i)] = eye
    for pair, value in pairwise.items():
        i, j = pair
        arr = np.asarray(value)
        matrix = permutation_matrix(arr.astype(int)).astype(complex) if arr.ndim == 1 else arr.astype(complex)
        completed[(i, j)] = matrix
        if (j, i) not in completed:
            completed[(j, i)] = np.linalg.pinv(matrix)
    missing = [(i, j) for i, j in product(range(n_models), repeat=2) if (i, j) not in completed]
    if missing:
        raise ValueError(f"missing pairwise matrix alignments: {missing[:5]}")
    return completed


def triangle_defects(
    matrices: Mapping[IndexPair, np.ndarray],
    triples: list[Triple],
) -> dict[Triple, np.ndarray]:
    return {
        triple: matrices[(triple[0], triple[1])] @ matrices[(triple[1], triple[2])] @ matrices[(triple[2], triple[0])]
        for triple in triples
    }


def _cycle_score(defects: Mapping[Triple, np.ndarray]) -> float:
    values = []
    for matrix in defects.values():
        eye = np.eye(matrix.shape[0], dtype=complex)
        values.append(float(np.linalg.norm(matrix - eye, ord="fro") / max(np.linalg.norm(eye, ord="fro"), 1e-12)))
    return float(np.mean(values)) if values else 0.0


def _permutation_sync_residual(perms: Mapping[IndexPair, np.ndarray], n_models: int) -> float:
    _ref, gauges, residual = synchronize_permutations(dict(perms), n_models)
    disagreements = []
    for i, j in product(range(n_models), repeat=2):
        implied = compose_perm(invert_perm(gauges[i]), gauges[j])
        disagreements.append(permutation_disagreement(perms[(i, j)], implied))
    return min(float(residual), float(np.mean(disagreements)) if disagreements else 0.0)


def _aggregate_scalar_detection(
    defects: Mapping[Triple, np.ndarray],
    max_order: int,
    centrality_tolerance: float,
    phase_tolerance: float,
) -> tuple[float, float | None, int | None, bool]:
    detections = [
        detect_scalar_phase(
            matrix,
            max_order=max_order,
            centrality_tolerance=centrality_tolerance,
            phase_tolerance=phase_tolerance,
        )
        for matrix in defects.values()
    ]
    if not detections:
        return 0.0, 0.0, 1, False
    centrality = float(max(d.centrality_score for d in detections))
    finite_phase_residuals = [d.phase_residual for d in detections if d.phase_residual is not None]
    phase_residual = float(max(finite_phase_residuals)) if finite_phase_residuals else None
    orders = {d.detected_order_d for d in detections if d.detected_order_d is not None}
    detected_order = next(iter(orders)) if len(orders) == 1 else None
    finite_candidate = all(d.is_scalar_finite_index_candidate for d in detections)
    return centrality, phase_residual, detected_order, finite_candidate


def _rank_allowed(order: int | None, candidate_lift_rank: int | None) -> bool | None:
    if order is None or order <= 1 or candidate_lift_rank is None:
        return None
    return determinant_obstruction_allows(order, candidate_lift_rank)


def _diag(
    *,
    level: str,
    residual_type: str,
    centrality_score: float,
    phase_residual: float | None,
    detected_order_d: int | None,
    cycle_score: float,
    rank_allowed: bool | None,
    selected_resolution: str,
    notes: list[str],
    previous: LadderDiagnostics | None,
    supports_brauer_projective_interpretation: bool = False,
    is_finite_index_candidate: bool = False,
) -> LadderDiagnostics:
    improvement = None
    if previous is not None and np.isfinite(previous.centrality_score) and np.isfinite(centrality_score):
        improvement = float(previous.centrality_score - centrality_score)
    return LadderDiagnostics(
        level=level,
        residual_type=residual_type,
        centrality_score=centrality_score,
        phase_residual=phase_residual,
        detected_order_d=detected_order_d,
        cycle_score=cycle_score,
        rank_allowed=rank_allowed,
        selected_resolution=selected_resolution,
        notes=notes,
        centrality_improvement_from_previous_level=improvement,
        supports_brauer_projective_interpretation=supports_brauer_projective_interpretation,
        is_finite_index_candidate=is_finite_index_candidate,
    )


def _not_evaluated(level: str, reason: str, previous: LadderDiagnostics | None) -> LadderDiagnostics:
    return _diag(
        level=level,
        residual_type="not_evaluated",
        centrality_score=float("nan"),
        phase_residual=None,
        detected_order_d=None,
        cycle_score=float("nan"),
        rank_allowed=None,
        selected_resolution="not_evaluated",
        notes=[reason],
        previous=previous,
    )


def estimate_pairwise_permutations_from_activations(
    activations: Mapping[int, np.ndarray],
    n_models: int,
    width: int,
) -> dict[IndexPair, np.ndarray]:
    pairwise: dict[IndexPair, np.ndarray] = {}
    for i, j in product(range(n_models), repeat=2):
        pairwise[(i, j)] = np.arange(width, dtype=int) if i == j else activation_permutation(activations[i], activations[j])
    return pairwise


def estimate_monomial_alignments_from_activations(
    pairwise_permutations: Mapping[IndexPair, np.ndarray],
    activations: Mapping[int, np.ndarray],
    n_models: int,
    width: int,
    mode: str = "signed_scale",
) -> dict[IndexPair, np.ndarray]:
    """Estimate signed or scaled monomial maps from activation correlations.

    The convention is row-vector composition: features_i @ G_ij approximates
    features_j after the permutation match.
    """

    matrices: dict[IndexPair, np.ndarray] = {}
    centered = {
        idx: np.asarray(value, dtype=float) - np.asarray(value, dtype=float).mean(axis=0, keepdims=True)
        for idx, value in activations.items()
    }
    for i, j in product(range(n_models), repeat=2):
        if i == j:
            matrices[(i, j)] = np.eye(width, dtype=complex)
            continue
        perm = np.asarray(pairwise_permutations[(i, j)], dtype=int)
        xi = centered[i]
        xj = centered[j]
        mat = np.zeros((width, width), dtype=complex)
        for source, target in enumerate(perm):
            a = xi[:, source]
            b = xj[:, target]
            dot = float(np.dot(a, b))
            if mode == "signed":
                value = 1.0 if dot >= 0.0 else -1.0
            elif mode == "positive_scale":
                value = abs(dot) / max(float(np.dot(a, a)), 1e-12)
            elif mode == "signed_scale":
                value = dot / max(float(np.dot(a, a)), 1e-12)
            else:
                raise ValueError(f"unknown monomial mode: {mode}")
            mat[source, target] = value
        matrices[(i, j)] = mat
    return matrices


def estimate_gl_alignments_from_activations(
    activations: Mapping[int, np.ndarray],
    n_models: int,
    width: int,
    ridge: float = 1e-4,
) -> dict[IndexPair, np.ndarray]:
    matrices: dict[IndexPair, np.ndarray] = {}
    centered = {
        idx: np.asarray(value, dtype=float) - np.asarray(value, dtype=float).mean(axis=0, keepdims=True)
        for idx, value in activations.items()
    }
    eye = np.eye(width)
    for i, j in product(range(n_models), repeat=2):
        if i == j:
            matrices[(i, j)] = eye.astype(complex)
            continue
        xi = centered[i]
        xj = centered[j]
        lhs = xi.T @ xi + ridge * eye
        rhs = xi.T @ xj
        matrices[(i, j)] = np.linalg.solve(lhs, rhs).astype(complex)
    return matrices


def stored_permutation_row_diagnostic(
    row: Mapping[str, object],
    centrality_tolerance: float = 1e-6,
    phase_tolerance: float = 1e-6,
) -> LadderDiagnostics:
    centrality = float(row.get("centrality_score", float("nan")))
    phase_raw = row.get("phase_residual", None)
    try:
        phase = float(phase_raw) if phase_raw not in {None, ""} else None
    except (TypeError, ValueError):
        phase = None
    order_raw = row.get("detected_order_d", None)
    try:
        order = int(float(order_raw)) if order_raw not in {None, ""} else None
    except (TypeError, ValueError):
        order = None
    cycle = float(row.get("cycle_score", float("nan")))
    finite = (
        np.isfinite(centrality)
        and phase is not None
        and centrality <= centrality_tolerance
        and phase <= phase_tolerance
        and order is not None
        and order > 1
    )
    residual_type = "central_finite_index_projective" if finite else "noncentral_permutation_holonomy"
    return LadderDiagnostics(
        level="permutation",
        residual_type=residual_type,
        centrality_score=centrality,
        phase_residual=phase,
        detected_order_d=order,
        cycle_score=cycle,
        rank_allowed=None,
        selected_resolution="central_projective_lift" if finite else "c2m3_synchronization",
        notes=["diagnostic reconstructed from stored finite-index residual mining row"],
        supports_brauer_projective_interpretation=finite,
        is_finite_index_candidate=finite,
    )


class StructureGroupLadderMerge:
    def __init__(
        self,
        max_order: int = 12,
        centrality_tolerance: float = 1e-6,
        phase_tolerance: float = 1e-6,
        c2m3_tolerance: float = 1e-8,
    ):
        self.max_order = int(max_order)
        self.centrality_tolerance = float(centrality_tolerance)
        self.phase_tolerance = float(phase_tolerance)
        self.c2m3_tolerance = float(c2m3_tolerance)

    def run(
        self,
        pairwise_alignments,
        n_models: int,
        width: int,
        activations: Mapping[int, np.ndarray] | None = None,
        candidate_lift_rank: int | None = None,
        triples: list[Triple] | None = None,
    ) -> LadderResult:
        level_inputs = self._prepare_level_inputs(pairwise_alignments, n_models, width, activations)
        actual_triples = triples or default_triples(n_models)
        actual_rank = width if candidate_lift_rank is None else int(candidate_lift_rank)
        diagnostics: list[LadderDiagnostics] = []
        previous: LadderDiagnostics | None = None
        for level in LEVELS:
            if level.name not in level_inputs:
                diag = _not_evaluated(level.name, "no transition maps or activation estimator available for this level", previous)
            else:
                diag = self._diagnose_level(
                    level.name,
                    level_inputs[level.name],
                    n_models,
                    width,
                    actual_rank,
                    actual_triples,
                    previous,
                )
            diagnostics.append(diag)
            if diag.residual_type != "not_evaluated":
                previous = diag
        final_decision, selected_level, notes = self._select(diagnostics)
        return LadderResult(
            final_decision=final_decision,
            selected_level=selected_level,
            diagnostics=diagnostics,
            notes=tuple(notes),
        )

    def _prepare_level_inputs(
        self,
        pairwise_alignments,
        n_models: int,
        width: int,
        activations: Mapping[int, np.ndarray] | None,
    ) -> dict[str, Mapping[IndexPair, Alignment]]:
        if isinstance(pairwise_alignments, Mapping) and pairwise_alignments and _is_pair_keyed(pairwise_alignments):
            if _is_permutation_payload(pairwise_alignments):
                level_inputs: dict[str, Mapping[IndexPair, Alignment]] = {"permutation": pairwise_alignments}
            else:
                level_inputs = {"low_rank_GL": pairwise_alignments}
        else:
            level_inputs = dict(pairwise_alignments or {})
        if activations is not None:
            if "permutation" not in level_inputs:
                level_inputs["permutation"] = estimate_pairwise_permutations_from_activations(activations, n_models, width)
            perms = complete_permutations(level_inputs["permutation"], n_models, width)
            level_inputs.setdefault(
                "signed_permutation",
                estimate_monomial_alignments_from_activations(perms, activations, n_models, width, mode="signed"),
            )
            level_inputs.setdefault(
                "monomial_phase_or_scale",
                estimate_monomial_alignments_from_activations(perms, activations, n_models, width, mode="signed_scale"),
            )
            level_inputs.setdefault(
                "low_rank_GL",
                estimate_gl_alignments_from_activations(activations, n_models, width),
            )
        return level_inputs

    def _diagnose_level(
        self,
        level: str,
        pairwise: Mapping[IndexPair, Alignment],
        n_models: int,
        width: int,
        candidate_lift_rank: int,
        triples: list[Triple],
        previous: LadderDiagnostics | None,
    ) -> LadderDiagnostics:
        matrices = complete_matrices(pairwise, n_models, width)
        defects = triangle_defects(matrices, triples)
        cycle_score = _cycle_score(defects)
        centrality, phase_residual, detected_order, finite_candidate = _aggregate_scalar_detection(
            defects,
            self.max_order,
            self.centrality_tolerance,
            self.phase_tolerance,
        )
        allowed = _rank_allowed(detected_order, candidate_lift_rank)
        notes: list[str] = []

        if level == "permutation":
            perms = complete_permutations(pairwise, n_models, width) if _is_permutation_payload(pairwise) else None
            sync_residual = _permutation_sync_residual(perms, n_models) if perms is not None else float("inf")
            if cycle_score <= self.c2m3_tolerance or sync_residual <= self.c2m3_tolerance:
                residual_type = "gauge_trivial"
                resolution = "c2m3_synchronization"
                notes.append("permutation-level residual is already explained by C2M3-style synchronization")
                supports = False
            else:
                residual_type = "noncentral_permutation_holonomy"
                resolution = "nonabelian_synchronization_or_report_holonomy"
                notes.append("permutation defect is noncentral; not a Brauer/projective scalar claim")
                supports = False
            return _diag(
                level=level,
                residual_type=residual_type,
                centrality_score=centrality,
                phase_residual=phase_residual,
                detected_order_d=detected_order,
                cycle_score=cycle_score,
                rank_allowed=allowed,
                selected_resolution=resolution,
                notes=notes,
                previous=previous,
                supports_brauer_projective_interpretation=supports,
                is_finite_index_candidate=False,
            )

        if level == "signed_permutation":
            if cycle_score <= self.c2m3_tolerance:
                residual_type = "signed_gauge_trivial"
                resolution = "signed_c2m3_synchronization"
                supports = False
            elif centrality <= self.centrality_tolerance and detected_order == 2 and phase_residual is not None and phase_residual <= self.phase_tolerance:
                residual_type = "central_mu2_candidate"
                resolution = "central_mu2_lift_diagnostic"
                supports = True
                notes.append("signed extension reveals a central mu_2 residual")
            else:
                residual_type = "signed_noncentral_holonomy"
                resolution = "report_noncentral_holonomy"
                supports = False
                notes.append("signed extension did not make the residual scalar/central")
            return _diag(
                level=level,
                residual_type=residual_type,
                centrality_score=centrality,
                phase_residual=phase_residual,
                detected_order_d=detected_order,
                cycle_score=cycle_score,
                rank_allowed=allowed,
                selected_resolution=resolution,
                notes=notes,
                previous=previous,
                supports_brauer_projective_interpretation=supports,
                is_finite_index_candidate=residual_type == "central_mu2_candidate",
            )

        if level == "monomial_phase_or_scale":
            if finite_candidate and allowed:
                residual_type = "finite_index_projective_lift"
                resolution = "finite_index_projective_lift"
                supports = True
            elif finite_candidate and allowed is False:
                residual_type = "finite_index_projective_obstructed"
                resolution = "rank_obstructed_no_lift"
                supports = True
            elif centrality <= self.centrality_tolerance and detected_order is not None and detected_order > 1:
                residual_type = "central_root_of_unity_candidate"
                resolution = "central_projective_diagnostic"
                supports = True
            elif cycle_score <= self.c2m3_tolerance:
                residual_type = "monomial_gauge_trivial"
                resolution = "monomial_synchronization"
                supports = False
            else:
                residual_type = "noncentral_holonomy"
                resolution = "report_noncentral_holonomy"
                supports = False
                notes.append("monomial extension did not produce a scalar finite-index residual")
            return _diag(
                level=level,
                residual_type=residual_type,
                centrality_score=centrality,
                phase_residual=phase_residual,
                detected_order_d=detected_order,
                cycle_score=cycle_score,
                rank_allowed=allowed,
                selected_resolution=resolution,
                notes=notes,
                previous=previous,
                supports_brauer_projective_interpretation=supports,
                is_finite_index_candidate=finite_candidate,
            )

        if level == "block_orthogonal":
            improved = previous is not None and np.isfinite(previous.centrality_score) and centrality < previous.centrality_score
            if cycle_score <= self.c2m3_tolerance or improved:
                residual_type = "block_gauge_reduces_residual"
                resolution = "block_orthogonal_diagnostic"
                notes.append("synthetic block-orthogonal diagnostic only; no real-model merge improvement is claimed")
            else:
                residual_type = "block_noncentral_holonomy"
                resolution = "report_noncentral_holonomy"
            return _diag(
                level=level,
                residual_type=residual_type,
                centrality_score=centrality,
                phase_residual=phase_residual,
                detected_order_d=detected_order,
                cycle_score=cycle_score,
                rank_allowed=allowed,
                selected_resolution=resolution,
                notes=notes,
                previous=previous,
                supports_brauer_projective_interpretation=False,
                is_finite_index_candidate=False,
            )

        if level == "low_rank_GL":
            if finite_candidate and allowed:
                residual_type = "central_projective_after_GL"
                resolution = "finite_index_projective_lift"
                supports = True
            elif finite_candidate and allowed is False:
                residual_type = "finite_index_projective_obstructed"
                resolution = "rank_obstructed_no_lift"
                supports = True
            elif previous is not None and np.isfinite(previous.centrality_score) and centrality < previous.centrality_score:
                residual_type = "gl_reduces_residual"
                resolution = "gl_diagnostic_only"
                supports = False
                notes.append("GL alignment is diagnostic; no practical merge improvement is claimed")
            else:
                residual_type = "gl_noncentral_holonomy"
                resolution = "report_noncentral_holonomy"
                supports = False
            return _diag(
                level=level,
                residual_type=residual_type,
                centrality_score=centrality,
                phase_residual=phase_residual,
                detected_order_d=detected_order,
                cycle_score=cycle_score,
                rank_allowed=allowed,
                selected_resolution=resolution,
                notes=notes,
                previous=previous,
                supports_brauer_projective_interpretation=supports,
                is_finite_index_candidate=finite_candidate,
            )

        return _not_evaluated(level, f"unknown structure group level: {level}", previous)

    def _select(self, diagnostics: list[LadderDiagnostics]) -> tuple[str, str | None, list[str]]:
        permutation = next((diag for diag in diagnostics if diag.level == "permutation"), None)
        if permutation is not None and permutation.residual_type == "gauge_trivial":
            return (
                "c2m3_synchronization",
                "permutation",
                ["C2M3/permutation synchronization has priority when it already explains the residual."],
            )
        for diag in diagnostics:
            if diag.residual_type in {"finite_index_projective_lift", "central_projective_after_GL"}:
                return diag.selected_resolution, diag.level, ["central finite-index residual passed the rank threshold"]
        for diag in diagnostics:
            if diag.residual_type == "finite_index_projective_obstructed":
                return diag.selected_resolution, diag.level, ["central finite-index residual was detected, but the candidate rank is obstructed"]
        for diag in diagnostics:
            if diag.residual_type in {"central_mu2_candidate", "central_root_of_unity_candidate"}:
                return diag.selected_resolution, diag.level, ["central residual detected, but this is diagnostic unless a merge is evaluated"]
        for diag in diagnostics:
            if diag.residual_type in {"block_gauge_reduces_residual", "gl_reduces_residual"}:
                return diag.selected_resolution, diag.level, ["larger gauge reduced residual but remains diagnostic"]
        return "report_noncentral_holonomy", None, ["no evaluated level produced a valid central/projective residual"]
