"""Conservative nearest finite-Heisenberg projection for learned chart maps."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Mapping

import numpy as np

from .period_index_detector import RobustPeriodIndexDetection, robust_detect_commutator_matrix_period_index
from .period_index_mining import project_to_nearest_unitary
from .time_frequency_benchmark import time_frequency_generator_dict, time_frequency_generator_chart_names
from .time_frequency_learned_charts import (
    CALIBRATED_CONFIDENCE_MARGIN,
    CALIBRATED_TOLERANCE,
    relative_residual,
)


HEISENBERG_PROJECTION_METHOD = "heisenberg_projection_projective_morita_lift"
PROJECTION_RESIDUAL_THRESHOLDS = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2)


@dataclass(frozen=True)
class PhaseCommutatorFit:
    generator_names: tuple[str, ...]
    d: int
    exponent_matrix: list[list[int]] | None
    expected_exponent_matrix: list[list[int]] | None
    exponent_matrix_matches: bool
    exponent_matrix_residual: float
    commutator_residual: float
    max_centrality_score: float
    max_phase_residual: float
    min_root_margin: float
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class HeisenbergProjectionResult:
    expected_d: int
    expected_k: int
    candidate_rank: int
    projection_residual_threshold: float
    projection_accepted: bool
    projected_generators: dict[str, np.ndarray]
    projection_residual: float
    commutator_residual_before: float
    commutator_residual_after: float
    exponent_matrix_residual: float
    detector_before_projection: RobustPeriodIndexDetection
    detector_after_projection: RobustPeriodIndexDetection
    decision: str
    selected_method: str
    phase_fit: PhaseCommutatorFit
    notes: tuple[str, ...] = ()


def _root(d: int, exponent: int) -> complex:
    return complex(np.exp(2j * np.pi * (int(exponent) % int(d)) / int(d)))


def _commutator(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return left @ right @ np.linalg.inv(left) @ np.linalg.inv(right)


def _scalar_centrality(matrix: np.ndarray) -> tuple[float, complex | None]:
    arr = np.asarray(matrix, dtype=complex)
    width = arr.shape[0]
    scalar = complex(np.trace(arr) / max(width, 1))
    if abs(scalar) <= 1e-12:
        return relative_residual(arr, np.eye(width, dtype=complex)), None
    target = scalar * np.eye(width, dtype=complex)
    return relative_residual(arr, target), scalar / abs(scalar)


def _nearest_expected_root(matrix: np.ndarray, d: int) -> tuple[int, float, float]:
    width = matrix.shape[0]
    residuals = []
    for exponent in range(d):
        residual = relative_residual(matrix, _root(d, exponent) * np.eye(width, dtype=complex))
        residuals.append((float(residual), int(exponent)))
    residuals.sort(key=lambda item: (item[0], item[1]))
    best_residual, best_exponent = residuals[0]
    second_residual = residuals[1][0] if len(residuals) > 1 else float("inf")
    margin = second_residual - best_residual if isfinite(second_residual) else float("inf")
    return best_exponent, best_residual, float(margin)


def _exponent_matrix_from_generators(
    generators: Mapping[str, np.ndarray],
    *,
    d: int,
) -> tuple[list[list[int]] | None, float, float, float, tuple[str, ...]]:
    names = tuple(generators)
    matrices = [np.asarray(generators[name], dtype=complex) for name in names]
    if len(matrices) < 2:
        return None, float("inf"), float("inf"), 0.0, names
    first_shape = matrices[0].shape
    if len(first_shape) != 2 or first_shape[0] != first_shape[1]:
        return None, float("inf"), float("inf"), 0.0, names
    if any(matrix.shape != first_shape for matrix in matrices):
        return None, float("inf"), float("inf"), 0.0, names

    exponent_matrix = [[0 for _ in names] for _ in names]
    centralities = []
    phase_residuals = []
    root_margins = []
    try:
        for row, left in enumerate(matrices):
            for col in range(row + 1, len(matrices)):
                commutator = _commutator(left, matrices[col])
                centrality, _scalar = _scalar_centrality(commutator)
                exponent, phase_residual, margin = _nearest_expected_root(commutator, d)
                exponent_matrix[row][col] = exponent % d
                exponent_matrix[col][row] = (-exponent) % d
                centralities.append(centrality)
                phase_residuals.append(phase_residual)
                root_margins.append(margin)
    except np.linalg.LinAlgError:
        return None, float("inf"), float("inf"), 0.0, names

    max_centrality = float(max(centralities, default=0.0))
    max_phase = float(max(phase_residuals, default=0.0))
    min_margin = float(min(root_margins, default=float("inf")))
    commutator_residual = float(max(max_centrality, max_phase))
    return exponent_matrix, commutator_residual, max_centrality, max_phase, names


def canonical_heisenberg_generators(
    d: int,
    k: int,
    *,
    generator_names: tuple[str, ...] | None = None,
) -> dict[str, np.ndarray]:
    """Return canonical finite time-frequency Heisenberg generators."""

    names = generator_names or time_frequency_generator_chart_names(k)
    canonical = time_frequency_generator_dict(d, k)
    missing = [name for name in names if name not in canonical]
    if missing:
        raise KeyError(f"unknown finite-Heisenberg generator names: {missing}")
    return {name: canonical[name] for name in names}


def phase_commutator_fit(
    generators: Mapping[str, np.ndarray],
    *,
    expected_d: int,
    expected_k: int,
    generator_names: tuple[str, ...] | None = None,
) -> PhaseCommutatorFit:
    """Fit the nearest commutator exponent matrix over ``Z / d Z``."""

    if expected_d <= 1 or expected_k <= 0:
        raise ValueError("expected_d must exceed 1 and expected_k must be positive")
    names = generator_names or tuple(generators)
    selected = {name: np.asarray(generators[name], dtype=complex) for name in names if name in generators}
    notes: list[str] = []
    if tuple(selected) != tuple(names):
        missing = [name for name in names if name not in selected]
        notes.append(f"missing generators for commutator fit: {missing}")
        return PhaseCommutatorFit(
            generator_names=tuple(selected),
            d=expected_d,
            exponent_matrix=None,
            expected_exponent_matrix=None,
            exponent_matrix_matches=False,
            exponent_matrix_residual=float("inf"),
            commutator_residual=float("inf"),
            max_centrality_score=float("inf"),
            max_phase_residual=float("inf"),
            min_root_margin=0.0,
            notes=tuple(notes),
        )

    exponent_matrix, commutator_residual, max_centrality, max_phase, actual_names = _exponent_matrix_from_generators(
        selected,
        d=expected_d,
    )
    canonical = canonical_heisenberg_generators(expected_d, expected_k, generator_names=actual_names)
    expected_matrix, _expected_residual, _expected_centrality, _expected_phase, _ = _exponent_matrix_from_generators(
        canonical,
        d=expected_d,
    )
    if exponent_matrix is None or expected_matrix is None:
        notes.append("could not fit a square invertible commutator exponent matrix")
        matches = False
        exponent_residual = float("inf")
    else:
        mismatches = 0
        total = 0
        for row in range(len(exponent_matrix)):
            for col in range(row + 1, len(exponent_matrix)):
                total += 1
                if exponent_matrix[row][col] % expected_d != expected_matrix[row][col] % expected_d:
                    mismatches += 1
        matches = mismatches == 0
        exponent_residual = float(mismatches / max(total, 1))
        if not matches:
            notes.append(f"fitted exponent matrix differs from expected finite-Heisenberg form on {mismatches} pairs")

    margins = []
    try:
        matrices = [np.asarray(selected[name], dtype=complex) for name in actual_names]
        for row, left in enumerate(matrices):
            for col in range(row + 1, len(matrices)):
                _exponent, _residual, margin = _nearest_expected_root(_commutator(left, matrices[col]), expected_d)
                margins.append(margin)
    except np.linalg.LinAlgError:
        margins.append(0.0)

    return PhaseCommutatorFit(
        generator_names=actual_names,
        d=expected_d,
        exponent_matrix=exponent_matrix,
        expected_exponent_matrix=expected_matrix,
        exponent_matrix_matches=matches,
        exponent_matrix_residual=exponent_residual,
        commutator_residual=commutator_residual,
        max_centrality_score=max_centrality,
        max_phase_residual=max_phase,
        min_root_margin=float(min(margins, default=0.0)),
        notes=tuple(notes),
    )


def nearest_unitary_preprocess(generators: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Project every learned generator to the nearest unitary before fitting."""

    return {name: project_to_nearest_unitary(matrix) for name, matrix in generators.items()}


def _mean_generator_residual(
    learned: Mapping[str, np.ndarray],
    projected: Mapping[str, np.ndarray],
) -> float:
    residuals = [relative_residual(np.asarray(learned[name]), np.asarray(projected[name])) for name in projected]
    return float(max(residuals, default=float("inf")))


def _detection(
    generators: Mapping[str, np.ndarray],
    *,
    candidate_rank: int,
) -> RobustPeriodIndexDetection:
    return robust_detect_commutator_matrix_period_index(
        generators,
        candidate_rank=candidate_rank,
        centrality_tol_grid=(CALIBRATED_TOLERANCE,),
        phase_tol_grid=(CALIBRATED_TOLERANCE,),
        confidence_margin=CALIBRATED_CONFIDENCE_MARGIN,
    )


def _decision(
    *,
    accepted: bool,
    detection: RobustPeriodIndexDetection,
) -> tuple[str, str]:
    if not accepted:
        return "heisenberg_projection_rejected", "none"
    if detection.status != "certified":
        return "heisenberg_projection_uncertain", "none"
    if detection.decision == "period_index_lift_success":
        return "heisenberg_projection_lift_success", HEISENBERG_PROJECTION_METHOD
    if detection.decision in {"period_divisible_index_obstructed", "rank_obstructed"}:
        return "heisenberg_projection_index_obstructed", "none"
    return "heisenberg_projection_uncertain", "none"


def project_to_nearest_finite_heisenberg(
    generators: Mapping[str, np.ndarray],
    *,
    expected_d: int,
    expected_k: int,
    candidate_rank: int,
    projection_residual_threshold: float = 1e-2,
    preprocess_unitary: bool = False,
    generator_names: tuple[str, ...] | None = None,
) -> HeisenbergProjectionResult:
    """Project learned generators to a nearby finite-Heisenberg form if safe.

    This routine intentionally implements the conservative "commutator-form
    projection" path.  It only accepts canonical finite-Heisenberg replacement
    when the learned commutator exponent matrix already matches the expected
    form and the learned-to-canonical operator residual is below the configured
    threshold.
    """

    names = generator_names or time_frequency_generator_chart_names(expected_k)
    selected = {name: np.asarray(generators[name], dtype=complex) for name in names if name in generators}
    notes: list[str] = []
    if tuple(selected) != tuple(names):
        missing = [name for name in names if name not in selected]
        notes.append(f"missing generators: {missing}")
        before = _detection(selected or {"missing": np.eye(1, dtype=complex)}, candidate_rank=candidate_rank)
        phase_fit = PhaseCommutatorFit(
            generator_names=tuple(selected),
            d=expected_d,
            exponent_matrix=None,
            expected_exponent_matrix=None,
            exponent_matrix_matches=False,
            exponent_matrix_residual=float("inf"),
            commutator_residual=float("inf"),
            max_centrality_score=float("inf"),
            max_phase_residual=float("inf"),
            min_root_margin=0.0,
            notes=tuple(notes),
        )
        decision, selected_method = _decision(accepted=False, detection=before)
        return HeisenbergProjectionResult(
            expected_d=expected_d,
            expected_k=expected_k,
            candidate_rank=candidate_rank,
            projection_residual_threshold=float(projection_residual_threshold),
            projection_accepted=False,
            projected_generators=selected,
            projection_residual=float("inf"),
            commutator_residual_before=float("inf"),
            commutator_residual_after=before.max_phase_residual,
            exponent_matrix_residual=float("inf"),
            detector_before_projection=before,
            detector_after_projection=before,
            decision=decision,
            selected_method=selected_method,
            phase_fit=phase_fit,
            notes=tuple(notes),
        )

    learned = nearest_unitary_preprocess(selected) if preprocess_unitary else selected
    before = _detection(learned, candidate_rank=candidate_rank)

    expected_dimension = expected_d**expected_k
    actual_dimensions = {matrix.shape[0] for matrix in learned.values() if matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1]}
    if actual_dimensions != {expected_dimension}:
        notes.append(
            f"generator dimension {sorted(actual_dimensions)} does not match expected finite-Heisenberg dimension {expected_dimension}"
        )
        phase_fit = phase_commutator_fit(
            learned,
            expected_d=expected_d,
            expected_k=expected_k,
            generator_names=names,
        ) if actual_dimensions else PhaseCommutatorFit(
            generator_names=tuple(learned),
            d=expected_d,
            exponent_matrix=None,
            expected_exponent_matrix=None,
            exponent_matrix_matches=False,
            exponent_matrix_residual=float("inf"),
            commutator_residual=float("inf"),
            max_centrality_score=float("inf"),
            max_phase_residual=float("inf"),
            min_root_margin=0.0,
            notes=tuple(notes),
        )
        decision, selected_method = _decision(accepted=False, detection=before)
        return HeisenbergProjectionResult(
            expected_d=expected_d,
            expected_k=expected_k,
            candidate_rank=candidate_rank,
            projection_residual_threshold=float(projection_residual_threshold),
            projection_accepted=False,
            projected_generators=learned,
            projection_residual=float("inf"),
            commutator_residual_before=phase_fit.commutator_residual,
            commutator_residual_after=before.max_phase_residual,
            exponent_matrix_residual=phase_fit.exponent_matrix_residual,
            detector_before_projection=before,
            detector_after_projection=before,
            decision=decision,
            selected_method=selected_method,
            phase_fit=phase_fit,
            notes=tuple(notes + list(phase_fit.notes)),
        )

    phase_fit = phase_commutator_fit(
        learned,
        expected_d=expected_d,
        expected_k=expected_k,
        generator_names=names,
    )
    projected = canonical_heisenberg_generators(expected_d, expected_k, generator_names=names)
    after = _detection(projected, candidate_rank=candidate_rank)
    projection_residual = _mean_generator_residual(learned, projected)
    after_fit = phase_commutator_fit(
        projected,
        expected_d=expected_d,
        expected_k=expected_k,
        generator_names=names,
    )
    form_tolerance = max(
        CALIBRATED_TOLERANCE,
        5.0 * float(projection_residual_threshold),
    )
    accepted = (
        phase_fit.exponent_matrix_matches
        and phase_fit.commutator_residual <= form_tolerance
        and projection_residual <= float(projection_residual_threshold)
        and after.status == "certified"
        and after.period == expected_d
        and after.index == expected_d**expected_k
    )
    if not phase_fit.exponent_matrix_matches:
        notes.append("projection rejected: learned commutator form does not match expected finite-Heisenberg form")
    if phase_fit.commutator_residual > form_tolerance:
        notes.append(
            f"projection rejected: commutator residual {phase_fit.commutator_residual:.6g} exceeds form tolerance {form_tolerance:.6g}"
        )
    if projection_residual > float(projection_residual_threshold):
        notes.append(
            f"projection rejected: projection residual {projection_residual:.6g} exceeds threshold {float(projection_residual_threshold):.6g}"
        )
    if after.status != "certified" or after.period != expected_d or after.index != expected_d**expected_k:
        notes.append("projection rejected: projected generators did not certify the expected period/index")
    if accepted:
        notes.append("commutator-form projection accepted below residual threshold")

    decision, selected_method = _decision(accepted=accepted, detection=after)
    return HeisenbergProjectionResult(
        expected_d=expected_d,
        expected_k=expected_k,
        candidate_rank=int(candidate_rank),
        projection_residual_threshold=float(projection_residual_threshold),
        projection_accepted=bool(accepted),
        projected_generators=projected,
        projection_residual=float(projection_residual),
        commutator_residual_before=phase_fit.commutator_residual,
        commutator_residual_after=after_fit.commutator_residual,
        exponent_matrix_residual=phase_fit.exponent_matrix_residual,
        detector_before_projection=before,
        detector_after_projection=after,
        decision=decision,
        selected_method=selected_method,
        phase_fit=phase_fit,
        notes=tuple(notes + list(phase_fit.notes)),
    )
