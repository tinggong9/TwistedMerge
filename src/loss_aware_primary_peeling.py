"""Loss-aware representative selection for primary residual peeling.

The v2 primary peeling smoke test solves quotient edge cochains and then picks a
single representative per label.  This module keeps the quotient solve, but
searches over valid representatives with the same quotient labels and scores
candidate corrections by residual, displacement, inverse consistency, and an
optional alignment proxy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Iterable, Mapping

import numpy as np

from src.primary_residual_peeling import (
    apply_edge_label_corrections,
    compose_perm,
    invert_perm,
    is_valid_permutation,
    oriented_edge_value,
    permutation_disagreement,
    permutation_power,
    quotient_label_from_fit,
    triangle_defects_from_pairwise,
)


@dataclass(frozen=True)
class RepresentativeCandidate:
    label: int
    perm: np.ndarray
    source: str
    fit_label: int | None
    fit_label_verified: bool | None
    disagreement_from_identity: float
    order: int
    is_valid_permutation: bool


@dataclass(frozen=True)
class LossAwareObjectiveWeights:
    quotient_residual: float = 1.0
    permutation_cycle_residual: float = 1.0
    representative_displacement: float = 0.1
    inverse_consistency: float = 0.1
    alignment_cost_proxy: float = 0.0


@dataclass(frozen=True)
class LossAwareCorrectionResult:
    corrections: dict[tuple[int, int], np.ndarray]
    corrected: dict[tuple[int, int], np.ndarray]
    selected_candidates: dict[tuple[int, int], RepresentativeCandidate]
    edge_labels: dict[tuple[int, int], int]
    quotient_residual_after: float
    permutation_cycle_residual_before: float
    permutation_cycle_residual_after: float
    representative_displacement_mean: float
    representative_displacement_max: float
    inverse_consistency_violation: float
    alignment_cost_proxy: float
    objective_value: float
    implemented: bool
    status: str
    candidate_role: str = "loss_aware"
    diagnostics: dict[str, float] = field(default_factory=dict)


def permutation_order(perm: Iterable[int]) -> int:
    arr = np.asarray(tuple(int(item) for item in perm), dtype=int)
    visited = np.zeros(len(arr), dtype=bool)
    order = 1
    for start in range(len(arr)):
        if visited[start]:
            continue
        cur = start
        length = 0
        while not visited[cur]:
            visited[cur] = True
            length += 1
            cur = int(arr[cur])
        if length:
            order = int(order * length // np.gcd(order, length))
    return int(order)


def _identity(width: int) -> np.ndarray:
    return np.arange(int(width), dtype=int)


def _candidate_key(perm: Iterable[int]) -> tuple[int, ...]:
    return tuple(int(item) for item in perm)


def _fit_label(fit, perm: Iterable[int], p: int) -> int | None:
    assignment = getattr(fit, "assignment", {}) if fit is not None else {}
    key = _candidate_key(perm)
    if key not in assignment:
        return None
    return int(assignment[key]) % int(p)


def _candidate_from_perm(label: int, perm: Iterable[int], source: str, fit, p: int) -> RepresentativeCandidate | None:
    arr = np.asarray(tuple(int(item) for item in perm), dtype=int)
    valid = is_valid_permutation(arr)
    if not valid:
        return None
    label = int(label) % int(p)
    fit_label = _fit_label(fit, arr, p)
    verified = None if fit_label is None else bool(fit_label == label)
    if verified is False:
        return None
    identity = _identity(len(arr))
    return RepresentativeCandidate(
        label=label,
        perm=arr.copy(),
        source=source,
        fit_label=fit_label,
        fit_label_verified=verified,
        disagreement_from_identity=permutation_disagreement(arr, identity),
        order=permutation_order(arr),
        is_valid_permutation=True,
    )


def _add_candidate(bank: dict[int, list[RepresentativeCandidate]], candidate: RepresentativeCandidate | None) -> None:
    if candidate is None:
        return
    label = int(candidate.label)
    existing = {_candidate_key(item.perm) for item in bank.setdefault(label, [])}
    if _candidate_key(candidate.perm) in existing:
        return
    bank[label].append(candidate)


def build_representative_bank(
    fit,
    observed_pairwise: Mapping[tuple[int, int], Iterable[int]],
    observed_holonomies: Iterable[Iterable[int]],
    width: int,
    p: int,
    max_candidates_per_label: int = 16,
) -> dict[int, list[RepresentativeCandidate]]:
    """Build valid permutation representatives grouped by quotient label."""

    p = int(p)
    width = int(width)
    bank: dict[int, list[RepresentativeCandidate]] = {label: [] for label in range(p)}
    identity = _identity(width)
    _add_candidate(bank, _candidate_from_perm(0, identity, "identity", fit, p))

    generators: list[tuple[np.ndarray, str, int | None]] = []
    for idx, holonomy in enumerate(observed_holonomies):
        arr = np.asarray(tuple(int(item) for item in holonomy), dtype=int)
        if is_valid_permutation(arr):
            generators.append((arr, f"observed_holonomy_{idx}", _fit_label(fit, arr, p)))
    for edge, perm in sorted(observed_pairwise.items()):
        arr = np.asarray(tuple(int(item) for item in perm), dtype=int)
        if is_valid_permutation(arr):
            generators.append((arr, f"observed_pairwise_{edge[0]}_{edge[1]}", _fit_label(fit, arr, p)))

    for generator, source, gen_label in generators:
        if gen_label is not None:
            _add_candidate(bank, _candidate_from_perm(gen_label, generator, source, fit, p))
        for exponent in range(1, max(1, p)):
            powered = permutation_power(generator, exponent)
            if gen_label is None:
                label = _fit_label(fit, powered, p)
                if label is None:
                    continue
            else:
                label = (int(gen_label) * int(exponent)) % p
            _add_candidate(bank, _candidate_from_perm(label, powered, f"{source}_power_{exponent}", fit, p))
            _add_candidate(bank, _candidate_from_perm((-label) % p, invert_perm(powered), f"{source}_power_{exponent}_inverse", fit, p))

    for label, candidates in list(bank.items()):
        candidates = [candidate for candidate in candidates if candidate.is_valid_permutation]
        candidates.sort(key=lambda item: (item.disagreement_from_identity, item.order, item.source))
        bank[label] = candidates[: int(max_candidates_per_label)]
    return bank


def directed_edge_labels(edge_labels: Mapping[tuple[int, int], int], n_models: int, p: int) -> dict[tuple[int, int], int]:
    out = {}
    for i, j in product(range(int(n_models)), repeat=2):
        out[(i, j)] = oriented_edge_value(dict(edge_labels), i, j, int(p)) if i != j else 0
    return out


def quotient_residual_after_from_edge_labels(edge_labels: Mapping[tuple[int, int], int], n_models: int, p: int) -> float:
    vals = []
    labels = dict(edge_labels)
    for i in range(int(n_models)):
        for j in range(i + 1, int(n_models)):
            for k in range(j + 1, int(n_models)):
                vals.append((labels[(i, j)] + labels[(j, k)] + labels[(k, i)]) % int(p) != 0)
    return float(np.mean(vals)) if vals else 0.0


def permutation_cycle_residual(pairwise: Mapping[tuple[int, int], Iterable[int]], n_models: int) -> float:
    if not pairwise:
        return float("nan")
    width = len(next(iter(pairwise.values())))
    identity = _identity(width)
    residuals = []
    for i in range(int(n_models)):
        for j in range(i + 1, int(n_models)):
            for k in range(j + 1, int(n_models)):
                defect = compose_perm(compose_perm(pairwise[(i, j)], pairwise[(j, k)]), pairwise[(k, i)])
                residuals.append(permutation_disagreement(defect, identity))
    return float(np.mean(residuals)) if residuals else 0.0


def inverse_consistency_violation(corrections: Mapping[tuple[int, int], Iterable[int]], n_models: int) -> float:
    vals = []
    for i, j in product(range(int(n_models)), repeat=2):
        if i >= j:
            continue
        vals.append(permutation_disagreement(corrections[(j, i)], invert_perm(corrections[(i, j)])))
    return float(np.mean(vals)) if vals else 0.0


def score_assignment(
    pairwise: Mapping[tuple[int, int], np.ndarray],
    corrections: Mapping[tuple[int, int], np.ndarray],
    selected: Mapping[tuple[int, int], RepresentativeCandidate],
    edge_labels: Mapping[tuple[int, int], int],
    n_models: int,
    p: int,
    weights: LossAwareObjectiveWeights,
    alignment_cost_proxy: float = 0.0,
) -> LossAwareCorrectionResult:
    corrected = apply_edge_label_corrections(dict(pairwise), dict(corrections))
    q_after = quotient_residual_after_from_edge_labels(edge_labels, n_models, int(p))
    p_before = permutation_cycle_residual(pairwise, n_models)
    p_after = permutation_cycle_residual(corrected, n_models)
    displacements = [candidate.disagreement_from_identity for edge, candidate in selected.items() if edge[0] != edge[1]]
    disp_mean = float(np.mean(displacements)) if displacements else 0.0
    disp_max = float(np.max(displacements)) if displacements else 0.0
    inv = inverse_consistency_violation(corrections, n_models)
    objective = (
        weights.quotient_residual * q_after
        + weights.permutation_cycle_residual * p_after
        + weights.representative_displacement * disp_mean
        + weights.inverse_consistency * inv
        + weights.alignment_cost_proxy * float(alignment_cost_proxy)
    )
    return LossAwareCorrectionResult(
        corrections=dict(corrections),
        corrected=corrected,
        selected_candidates=dict(selected),
        edge_labels=dict(edge_labels),
        quotient_residual_after=float(q_after),
        permutation_cycle_residual_before=float(p_before),
        permutation_cycle_residual_after=float(p_after),
        representative_displacement_mean=disp_mean,
        representative_displacement_max=disp_max,
        inverse_consistency_violation=float(inv),
        alignment_cost_proxy=float(alignment_cost_proxy),
        objective_value=float(objective),
        implemented=True,
        status="loss_aware_representative_selection_available",
    )


def assemble_loss_aware_corrections(
    pairwise: Mapping[tuple[int, int], np.ndarray],
    edge_labels: Mapping[tuple[int, int], int],
    representative_bank: Mapping[int, list[RepresentativeCandidate]],
    n_models: int,
    p: int,
    objective_weights: LossAwareObjectiveWeights | Mapping[str, float] | None = None,
    max_beam_size: int = 64,
    permutation_residual_tolerance: float | None = None,
    candidate_role: str = "combined_objective",
    alignment_cost_proxy: float = 0.0,
) -> LossAwareCorrectionResult:
    """Beam-search representative corrections for all directed edge labels."""

    weights = objective_weights if isinstance(objective_weights, LossAwareObjectiveWeights) else LossAwareObjectiveWeights(**(objective_weights or {}))
    p = int(p)
    labels = {edge: int(value) % p for edge, value in edge_labels.items()}
    width = len(next(iter(pairwise.values())))
    identity_candidate = RepresentativeCandidate(0, _identity(width), "identity", 0, True, 0.0, 1, True)
    directed = [(i, j) for i, j in product(range(int(n_models)), repeat=2) if i != j]
    base_corrections = {(idx, idx): _identity(width) for idx in range(int(n_models))}
    base_selected = {(idx, idx): identity_candidate for idx in range(int(n_models))}
    beams: list[tuple[float, dict[tuple[int, int], np.ndarray], dict[tuple[int, int], RepresentativeCandidate]]] = [(0.0, base_corrections, base_selected)]

    for edge in directed:
        label = int(labels.get(edge, 0))
        candidates = list(representative_bank.get(label, []))
        if not candidates:
            return LossAwareCorrectionResult(
                corrections=dict(base_corrections),
                corrected={edge: np.asarray(value, dtype=int).copy() for edge, value in pairwise.items()},
                selected_candidates=dict(base_selected),
                edge_labels=labels,
                quotient_residual_after=float("nan"),
                permutation_cycle_residual_before=permutation_cycle_residual(pairwise, n_models),
                permutation_cycle_residual_after=float("nan"),
                representative_displacement_mean=float("nan"),
                representative_displacement_max=float("nan"),
                inverse_consistency_violation=float("nan"),
                alignment_cost_proxy=float(alignment_cost_proxy),
                objective_value=float("inf"),
                implemented=False,
                status=f"missing_representative_for_label_{label}",
                candidate_role=candidate_role,
            )
        next_beams = []
        for partial_score, corrections, selected in beams:
            for candidate in candidates:
                new_corrections = dict(corrections)
                new_selected = dict(selected)
                new_corrections[edge] = candidate.perm.copy()
                new_selected[edge] = candidate
                partial = partial_score + weights.representative_displacement * candidate.disagreement_from_identity
                reverse = (edge[1], edge[0])
                if reverse in new_corrections:
                    partial += weights.inverse_consistency * permutation_disagreement(new_corrections[reverse], invert_perm(candidate.perm))
                next_beams.append((float(partial), new_corrections, new_selected))
        next_beams.sort(key=lambda item: item[0])
        beams = next_beams[: int(max_beam_size)]

    scored = []
    for _partial, corrections, selected in beams:
        corrected = apply_edge_label_corrections(dict(pairwise), dict(corrections))
        p_before = permutation_cycle_residual(pairwise, n_models)
        p_after = permutation_cycle_residual(corrected, n_models)
        if permutation_residual_tolerance is not None and p_after > p_before + float(permutation_residual_tolerance):
            continue
        displacements = [candidate.disagreement_from_identity for edge, candidate in selected.items() if edge[0] != edge[1]]
        disp_mean = float(np.mean(displacements)) if displacements else 0.0
        disp_max = float(np.max(displacements)) if displacements else 0.0
        inv = inverse_consistency_violation(corrections, n_models)
        q_after = quotient_residual_after_from_edge_labels(labels, n_models, p)
        objective = (
            weights.quotient_residual * q_after
            + weights.permutation_cycle_residual * p_after
            + weights.representative_displacement * disp_mean
            + weights.inverse_consistency * inv
            + weights.alignment_cost_proxy * float(alignment_cost_proxy)
        )
        scored.append(
            LossAwareCorrectionResult(
                corrections=dict(corrections),
                corrected=corrected,
                selected_candidates=dict(selected),
                edge_labels=labels,
                quotient_residual_after=float(q_after),
                permutation_cycle_residual_before=float(p_before),
                permutation_cycle_residual_after=float(p_after),
                representative_displacement_mean=disp_mean,
                representative_displacement_max=disp_max,
                inverse_consistency_violation=float(inv),
                alignment_cost_proxy=float(alignment_cost_proxy),
                objective_value=float(objective),
                implemented=True,
                status="loss_aware_representative_selection_available",
                candidate_role=candidate_role,
            )
        )
    if not scored:
        return LossAwareCorrectionResult(
            corrections=dict(base_corrections),
            corrected={edge: np.asarray(value, dtype=int).copy() for edge, value in pairwise.items()},
            selected_candidates=dict(base_selected),
            edge_labels=labels,
            quotient_residual_after=float("nan"),
            permutation_cycle_residual_before=permutation_cycle_residual(pairwise, n_models),
            permutation_cycle_residual_after=float("nan"),
            representative_displacement_mean=float("nan"),
            representative_displacement_max=float("nan"),
            inverse_consistency_violation=float("nan"),
            alignment_cost_proxy=float(alignment_cost_proxy),
            objective_value=float("inf"),
            implemented=False,
            status="permutation_residual_tolerance_blocked_all_candidates",
            candidate_role=candidate_role,
        )
    scored.sort(key=lambda item: item.objective_value)
    return scored[0]


def correction_result_with_q_residual(result: LossAwareCorrectionResult, quotient_residual_after: float) -> LossAwareCorrectionResult:
    return LossAwareCorrectionResult(
        corrections=result.corrections,
        corrected=result.corrected,
        selected_candidates=result.selected_candidates,
        edge_labels=result.edge_labels,
        quotient_residual_after=float(quotient_residual_after),
        permutation_cycle_residual_before=result.permutation_cycle_residual_before,
        permutation_cycle_residual_after=result.permutation_cycle_residual_after,
        representative_displacement_mean=result.representative_displacement_mean,
        representative_displacement_max=result.representative_displacement_max,
        inverse_consistency_violation=result.inverse_consistency_violation,
        alignment_cost_proxy=result.alignment_cost_proxy,
        objective_value=result.objective_value,
        implemented=result.implemented,
        status=result.status,
        candidate_role=result.candidate_role,
        diagnostics=result.diagnostics,
    )


def validation_selection_decision(row: Mapping[str, object]) -> tuple[bool, str]:
    if bool(row.get("uses_test_for_selection", False)):
        return False, "blocked_test_metric_selection_forbidden"
    if not bool(row.get("implemented_corrected_merge", False)):
        return False, str(row.get("na_reason") or "corrected_merge_not_implemented")
    q_before = float(row.get("quotient_residual_before", float("nan")))
    q_after = float(row.get("quotient_residual_after", float("nan")))
    if not np.isfinite(q_before) or not np.isfinite(q_after) or q_after >= q_before:
        return False, "metric_produced_but_not_claimable"
    p_before = float(row.get("permutation_cycle_residual_before", float("nan")))
    p_after = float(row.get("permutation_cycle_residual_after", float("nan")))
    if np.isfinite(p_before) and np.isfinite(p_after) and p_after > p_before + 1e-12:
        return False, "quotient_peel_not_permutation_safe"
    val = float(row.get("validation_accuracy", float("nan")))
    baseline = float(row.get("baseline_validation_accuracy", float("nan")))
    if not np.isfinite(val):
        return False, "missing_corrected_validation_metric"
    if not np.isfinite(baseline) or val <= baseline:
        return False, "not_selected_fails_unpeeled_baseline_gate"
    for key, label in [
        ("wrong_prime_control_validation_accuracy", "wrong_prime_control"),
        ("shuffled_control_validation_accuracy", "shuffled_control"),
        ("random_control_validation_accuracy", "random_control"),
        ("no_quotient_control_validation_accuracy", "no_quotient_control"),
    ]:
        control = float(row.get(key, float("nan")))
        if not np.isfinite(control):
            return False, f"not_selected_missing_{label}"
        if val <= control:
            return False, f"not_selected_fails_{label}"
    return True, "loss_aware_real_positive_validation_selected"


def no_lift_capacity_metadata() -> dict[str, float]:
    return {"capacity_multiplier": 1.0, "inference_multiplier": 1.0}


def cumulative_update_allowed_loss_aware(
    quotient_reduces: bool,
    permutation_safe: bool,
    validation_improves_over_current: bool,
) -> bool:
    return bool(quotient_reduces and permutation_safe and validation_improves_over_current)
