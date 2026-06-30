"""TwistedMerge++ residual classifier and merge selector.

This module is an obstruction-aware wrapper around the C2M3-style
permutation synchronization already used in the model-merging benchmark.  It
does not claim that every failed synchronization is a twist: it first checks
whether C2M3 explains the observations, then separates edge noise,
central-coboundary mu_2 residuals, central non-coboundary candidates, and
random/noncentral residuals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations, product
from typing import Mapping

import numpy as np

from .model_merging_benchmark import (
    compose_perm,
    cycle_score as permutation_cycle_score,
    invert_perm,
    permutation_disagreement,
    permutation_matrix,
    synchronize_permutations,
)
from .simplicial_mu2 import Face, canonical_face, is_coboundary_mu2, tetrahedral_sphere
from .twisted_merge_algorithm import (
    lift_mu2_transition,
    lookup_twist,
    solve_mu2_edge_cochain,
)


IndexPair = tuple[int, int]
Triple = tuple[int, int, int]
Alignment = np.ndarray


@dataclass(frozen=True)
class TwistedMergePlusConfig:
    """Numerical knobs for the prototype classifier."""

    c2m3_tolerance: float = 1e-8
    central_tolerance: float = 1e-8
    alpha_tolerance: float = 1e-8
    edge_outlier_tolerance: float = 0.25
    edge_outlier_fraction: float = 0.25
    rank_lift_q: int = 2
    allow_branch_lift: bool = True


@dataclass(frozen=True)
class ResidualDiagnostics:
    cycle_score: float
    c2m3_residual: float
    c2m3_max_residual: float
    c2m3_reference: int | None
    max_triangle_residual: float
    centrality_score: float
    alpha_residual: float | None
    outlier_edge_score: float
    bad_edge_fraction: float
    classification: str
    is_central: bool
    is_coboundary: bool | None
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class TwistedMergePlusResult:
    status: str
    selected_method: str
    reason: str
    diagnostics: ResidualDiagnostics
    synced_alignments: dict[IndexPair, Alignment]
    lifted_transition_maps: dict[IndexPair, Alignment]
    edge_central_signs: dict[IndexPair, int] | None
    metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    notes: tuple[str, ...] = ()


def default_triples(n_models: int) -> list[Triple]:
    return list(combinations(range(n_models), 3))


def _is_permutation_payload(pairwise: Mapping[IndexPair, Alignment]) -> bool:
    return all(np.asarray(value).ndim == 1 for value in pairwise.values())


def _infer_width(pairwise: Mapping[IndexPair, Alignment], width: int | None) -> int:
    if width is not None:
        return int(width)
    first = np.asarray(next(iter(pairwise.values())))
    return int(first.shape[0])


def _complete_permutations(
    pairwise: Mapping[IndexPair, Alignment],
    n_models: int,
    width: int,
) -> dict[IndexPair, np.ndarray]:
    completed: dict[IndexPair, np.ndarray] = {}
    for i in range(n_models):
        completed[(i, i)] = np.arange(width)
    for pair, perm in pairwise.items():
        i, j = pair
        arr = np.asarray(perm, dtype=int)
        if arr.ndim != 1:
            raise ValueError("permutation alignments must be rank-1 arrays")
        completed[(i, j)] = arr
        if (j, i) not in completed:
            completed[(j, i)] = invert_perm(arr)
    missing = [(i, j) for i, j in product(range(n_models), repeat=2) if (i, j) not in completed]
    if missing:
        raise ValueError(f"missing pairwise permutation alignments: {missing[:5]}")
    return completed


def _complete_matrices(
    pairwise: Mapping[IndexPair, Alignment],
    n_models: int,
    width: int,
) -> dict[IndexPair, np.ndarray]:
    completed: dict[IndexPair, np.ndarray] = {}
    eye = np.eye(width)
    for i in range(n_models):
        completed[(i, i)] = eye
    for pair, matrix in pairwise.items():
        i, j = pair
        arr = np.asarray(matrix, dtype=float)
        if arr.ndim == 1:
            arr = permutation_matrix(arr.astype(int))
        completed[(i, j)] = arr
        if (j, i) not in completed:
            completed[(j, i)] = np.linalg.pinv(arr)
    missing = [(i, j) for i, j in product(range(n_models), repeat=2) if (i, j) not in completed]
    if missing:
        raise ValueError(f"missing pairwise matrix alignments: {missing[:5]}")
    return completed


def _matrix_triangle_defects(
    matrices: Mapping[IndexPair, np.ndarray],
    triples: list[Triple],
) -> dict[Triple, np.ndarray]:
    return {
        tuple(triple): matrices[(triple[0], triple[1])] @ matrices[(triple[1], triple[2])] @ matrices[(triple[2], triple[0])]
        for triple in triples
    }


def compute_triangle_defects(
    pairwise_alignments: Mapping[IndexPair, Alignment],
    n_models: int,
    width: int | None = None,
    triples: list[Triple] | None = None,
) -> dict[Triple, np.ndarray]:
    """Compute matrix-valued triangle defects for permutations or matrices."""

    actual_width = _infer_width(pairwise_alignments, width)
    actual_triples = triples or default_triples(n_models)
    matrices = _complete_matrices(pairwise_alignments, n_models, actual_width)
    return _matrix_triangle_defects(matrices, actual_triples)


def _matrix_sync(
    matrices: Mapping[IndexPair, np.ndarray],
    n_models: int,
    tolerance: float,
) -> tuple[int, dict[int, np.ndarray], float, float, dict[IndexPair, np.ndarray]]:
    width = next(iter(matrices.values())).shape[0]
    best_ref = 0
    best_gauges: dict[int, np.ndarray] = {0: np.eye(width)}
    best_residual = float("inf")
    best_max = float("inf")
    best_synced: dict[IndexPair, np.ndarray] = {}
    for ref in range(n_models):
        gauges = {i: matrices[(ref, i)] if i != ref else np.eye(width) for i in range(n_models)}
        residuals = []
        synced: dict[IndexPair, np.ndarray] = {}
        for i, j in product(range(n_models), repeat=2):
            implied = gauges[j] @ np.linalg.pinv(gauges[i])
            synced[(i, j)] = implied
            denom = max(np.linalg.norm(matrices[(i, j)], ord="fro"), 1e-12)
            residuals.append(float(np.linalg.norm(matrices[(i, j)] - implied, ord="fro") / denom))
        residual = float(np.mean(residuals)) if residuals else 0.0
        max_residual = float(np.max(residuals)) if residuals else 0.0
        if max_residual <= tolerance:
            return ref, gauges, residual, max_residual, synced
        if residual < best_residual:
            best_ref = ref
            best_gauges = gauges
            best_residual = residual
            best_max = max_residual
            best_synced = synced
    return best_ref, best_gauges, best_residual, best_max, best_synced


def _central_distance(matrix: np.ndarray) -> tuple[float, int]:
    width = matrix.shape[0]
    eye = np.eye(width)
    dist_pos = float(np.linalg.norm(matrix - eye, ord="fro") / max(np.linalg.norm(eye, ord="fro"), 1e-12))
    dist_neg = float(np.linalg.norm(matrix + eye, ord="fro") / max(np.linalg.norm(eye, ord="fro"), 1e-12))
    return (dist_pos, 1) if dist_pos <= dist_neg else (dist_neg, -1)


def _defect_signs(defects: Mapping[Triple, np.ndarray]) -> dict[Triple, int]:
    return {triple: _central_distance(matrix)[1] for triple, matrix in defects.items()}


def _centrality(defects: Mapping[Triple, np.ndarray]) -> tuple[float, float]:
    if not defects:
        return 0.0, 0.0
    distances = [_central_distance(matrix)[0] for matrix in defects.values()]
    return float(np.mean(distances)), float(np.max(distances))


def _cycle_score_from_defects(defects: Mapping[Triple, np.ndarray]) -> float:
    if not defects:
        return 0.0
    values = []
    for matrix in defects.values():
        eye = np.eye(matrix.shape[0])
        values.append(float(np.linalg.norm(matrix - eye, ord="fro") / max(np.linalg.norm(eye, ord="fro"), 1e-12)))
    return float(np.mean(values))


def _alpha_residual(
    defects: Mapping[Triple, np.ndarray],
    alpha: Mapping[Triple, int] | Mapping[Face, int] | None,
) -> float | None:
    if alpha is None:
        return None
    values = []
    for triple, matrix in defects.items():
        sign = lookup_twist(alpha, triple)
        target = sign * np.eye(matrix.shape[0])
        values.append(float(np.linalg.norm(matrix - target, ord="fro") / max(np.linalg.norm(target, ord="fro"), 1e-12)))
    return float(np.mean(values)) if values else 0.0


def _alpha_is_coboundary(
    alpha: Mapping[Triple, int] | Mapping[Face, int],
    n_models: int,
    triples: list[Triple],
) -> bool:
    if n_models == 4 and {canonical_face(triple) for triple in triples} == set(tetrahedral_sphere().faces):
        return is_coboundary_mu2({canonical_face(face): int(sign) for face, sign in alpha.items()}, tetrahedral_sphere())
    return solve_mu2_edge_cochain(alpha, n_models, triples) is not None


def _permutation_sync_diagnostics(
    perms: Mapping[IndexPair, np.ndarray],
    n_models: int,
    width: int,
) -> tuple[float, float, int, dict[IndexPair, np.ndarray], float, float]:
    ref, gauges, residual = synchronize_permutations(perms, n_models)
    edge_residuals = []
    synced: dict[IndexPair, np.ndarray] = {}
    for i, j in product(range(n_models), repeat=2):
        implied = compose_perm(invert_perm(gauges[i]), gauges[j])
        synced[(i, j)] = implied
        edge_residuals.append(permutation_disagreement(perms[(i, j)], implied))
    arr = np.asarray(edge_residuals, dtype=float)
    outlier_score = float(np.max(arr) - np.median(arr)) if arr.size else 0.0
    bad_fraction = float(np.mean(arr > 0.0)) if arr.size else 0.0
    score, _rows = permutation_cycle_score(perms, n_models, width)
    return score, residual, ref, synced, outlier_score, bad_fraction


class TwistedMergePlus:
    """Obstruction-aware selector that reduces to C2M3 when residuals vanish."""

    def __init__(self, config: TwistedMergePlusConfig | None = None):
        self.config = config or TwistedMergePlusConfig()

    def run(
        self,
        pairwise_alignments: Mapping[IndexPair, Alignment],
        n_models: int,
        width: int | None = None,
        known_alpha: Mapping[Triple, int] | Mapping[Face, int] | None = None,
        triples: list[Triple] | None = None,
        method_metrics: Mapping[str, Mapping[str, float]] | None = None,
    ) -> TwistedMergePlusResult:
        actual_width = _infer_width(pairwise_alignments, width)
        actual_triples = triples or default_triples(n_models)
        is_permutation = _is_permutation_payload(pairwise_alignments)
        notes: list[str] = []

        matrices = _complete_matrices(pairwise_alignments, n_models, actual_width)
        defects = _matrix_triangle_defects(matrices, actual_triples)
        signs = _defect_signs(defects)
        centrality_score, max_triangle_residual = _centrality(defects)
        alpha_residual = _alpha_residual(defects, known_alpha)
        matrix_ref, _matrix_gauges, matrix_residual, matrix_max, synced_matrices = _matrix_sync(
            matrices,
            n_models,
            self.config.c2m3_tolerance,
        )
        cycle_score = _cycle_score_from_defects(defects)
        c2m3_residual = matrix_residual
        c2m3_max = matrix_max
        c2m3_ref: int | None = matrix_ref
        outlier_edge_score = 0.0
        bad_edge_fraction = 0.0
        synced_alignments = synced_matrices

        if is_permutation:
            perms = _complete_permutations(pairwise_alignments, n_models, actual_width)
            cycle_score, c2m3_residual, c2m3_ref, synced_perms, outlier_edge_score, bad_edge_fraction = (
                _permutation_sync_diagnostics(perms, n_models, actual_width)
            )
            c2m3_max = c2m3_residual
            synced_alignments = {
                pair: permutation_matrix(perm)
                for pair, perm in synced_perms.items()
            }

        classification = "unknown"
        is_central = centrality_score <= self.config.central_tolerance
        is_coboundary: bool | None = None

        if known_alpha is not None:
            alpha_coboundary = _alpha_is_coboundary(known_alpha, n_models, actual_triples)
            is_coboundary = alpha_coboundary
            if not alpha_coboundary:
                classification = "central_non_coboundary_candidate"
                notes.append(
                    "Supplied alpha is non-coboundary; this is not an ordinary untwisted descent claim."
                )
                if alpha_residual is not None and alpha_residual > self.config.alpha_tolerance:
                    notes.append(
                        "Observed pairwise defects do not realize the supplied non-coboundary alpha at transition level."
                    )
            elif c2m3_residual <= self.config.c2m3_tolerance and alpha_residual is not None and alpha_residual <= self.config.alpha_tolerance:
                classification = "gauge_trivial"
            elif alpha_residual is not None and alpha_residual <= self.config.alpha_tolerance and is_central:
                classification = "central_coboundary"
            else:
                classification = "unknown"
                notes.append("Supplied alpha is coboundary/trivial but does not match the observed defects.")
        elif c2m3_residual <= self.config.c2m3_tolerance:
            classification = "gauge_trivial"
            is_coboundary = True
        elif (
            is_permutation
            and outlier_edge_score >= self.config.edge_outlier_tolerance
            and bad_edge_fraction <= self.config.edge_outlier_fraction
        ):
            classification = "edge_outlier_or_noise"
            is_coboundary = None
        elif is_central:
            is_coboundary = solve_mu2_edge_cochain(signs, n_models, actual_triples) is not None
            classification = "central_coboundary" if is_coboundary else "central_non_coboundary_candidate"
        elif not is_permutation and max_triangle_residual > self.config.central_tolerance:
            classification = "random_noncentral"
        else:
            classification = "random_noncentral"

        lifted_transition_maps: dict[IndexPair, np.ndarray] = {}
        edge_central_signs: dict[IndexPair, int] | None = None
        if classification == "central_coboundary" and self.config.rank_lift_q >= 2:
            alpha_for_lift: Mapping[Triple, int] | Mapping[Face, int] = known_alpha if known_alpha is not None else signs
            edge_central_signs = solve_mu2_edge_cochain(alpha_for_lift, n_models, actual_triples)
            if edge_central_signs is not None:
                lifted_transition_maps = {
                    pair: lift_mu2_transition(matrix, edge_central_signs[pair])
                    for pair, matrix in matrices.items()
                }
                notes.append("Built lifted transition maps using rho(beta_ij) tensor G_ij.")

        status, selected_method, reason = self._select(classification, lifted_transition_maps, alpha_residual)
        diagnostics = ResidualDiagnostics(
            cycle_score=cycle_score,
            c2m3_residual=c2m3_residual,
            c2m3_max_residual=c2m3_max,
            c2m3_reference=c2m3_ref,
            max_triangle_residual=max_triangle_residual,
            centrality_score=centrality_score,
            alpha_residual=alpha_residual,
            outlier_edge_score=outlier_edge_score,
            bad_edge_fraction=bad_edge_fraction,
            classification=classification,
            is_central=is_central,
            is_coboundary=is_coboundary,
            notes=tuple(notes),
        )
        return TwistedMergePlusResult(
            status=status,
            selected_method=selected_method,
            reason=reason,
            diagnostics=diagnostics,
            synced_alignments=synced_alignments,
            lifted_transition_maps=lifted_transition_maps,
            edge_central_signs=edge_central_signs,
            metrics={key: dict(value) for key, value in (method_metrics or {}).items()},
            notes=tuple(notes),
        )

    def _select(
        self,
        classification: str,
        lifted_transition_maps: Mapping[IndexPair, np.ndarray],
        alpha_residual: float | None,
    ) -> tuple[str, str, str]:
        if classification in {"gauge_trivial", "edge_outlier_or_noise"}:
            return (
                "untwisted_c2m3",
                "c2m3_cycle_consistent",
                "C2M3-style synchronization explains the residual; no twist language is needed.",
            )
        if classification == "central_coboundary":
            if self.config.rank_lift_q < 2:
                return (
                    "failed",
                    "none",
                    "Central coboundary residual found, but rank_lift_q < 2 so no mu_2 lift is available.",
                )
            if lifted_transition_maps:
                return (
                    "central_coboundary_lift",
                    "lifted_transition_merge",
                    "Finite central coboundary residual is absorbed by explicit rho(beta_ij) lifted maps.",
                )
            return (
                "failed",
                "none",
                "Central coboundary was detected but no edge cochain/lifted maps were constructed.",
            )
        if classification == "central_non_coboundary_candidate":
            if self.config.allow_branch_lift and self.config.rank_lift_q >= 2:
                residual_text = "unknown" if alpha_residual is None else f"{alpha_residual:.4g}"
                return (
                    "twisted_branch_lift",
                    "branch_lift_extra_capacity",
                    "Non-coboundary central candidate; only branch/extra-capacity prediction is allowed "
                    f"(alpha residual {residual_text}), not ordinary untwisted descent.",
                )
            return (
                "failed",
                "none",
                "Non-coboundary central candidate detected, but branch lift is disabled or q < 2.",
            )
        if classification == "random_noncentral":
            return (
                "rejected_noncentral",
                "best_validation_baseline",
                "Residual is noncentral/random; TwistedMerge++ refuses to call it a central twist.",
            )
        return (
            "failed",
            "none",
            "Residual could not be classified without inventing twist structure.",
        )


def pseudocode() -> str:
    return """TwistedMerge++(M_i, observed g_ij, optional alpha_ijk, q):
  1. Compute/receive pairwise transition maps g_ij.
  2. Run C2M3-style synchronization. If residual is small, return untwisted_c2m3.
  3. Compute triangle defects c_ijk = g_ij g_jk g_ki.
  4. Classify the residual: gauge_trivial, edge_outlier_or_noise,
     central_coboundary, central_non_coboundary_candidate,
     random_noncentral, or unknown.
  5. For random/noncentral residuals, refuse twist language.
  6. For a finite central coboundary, solve beta with delta beta = alpha and
     build lifted maps rho(beta_ij) tensor G_ij.
  7. For a central non-coboundary candidate, allow only a branch/rank-lift
     prediction prototype and label it extra capacity.
  8. Report ordinary, C2M3, lifted, branch, and ensemble metrics without
     claiming a win unless validation supports it."""
