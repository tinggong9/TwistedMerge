"""Small-order central/projective torsion diagnostics.

The functions here are intentionally conservative.  They can identify exact
scalar root-of-unity residuals, but they classify ordinary noncentral
holonomies as rejections even when those holonomies have finite order.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from typing import Iterable, Sequence

import numpy as np


DEFAULT_ORDERS = (2, 3, 4, 5, 6, 8)
EPS = 1e-12


@dataclass(frozen=True)
class TorsionThresholdPolicy:
    name: str
    centrality_threshold: float
    scalar_threshold: float
    order_threshold: float
    target_false_positive_rate: float
    activates_lift: bool


DEFAULT_POLICIES = (
    TorsionThresholdPolicy("strict_fpr_0", 1e-3, 1e-3, 1e-3, 0.0, True),
    TorsionThresholdPolicy("strict_fpr_001", 1e-3, 1e-3, 1e-3, 0.01, True),
    TorsionThresholdPolicy("loose_1e-2", 1e-2, 1e-2, 1e-2, 0.01, False),
    TorsionThresholdPolicy("diagnostic_5e-2", 5e-2, 5e-2, 5e-2, 0.01, False),
)


def root_of_unity(order: int, exponent: int) -> complex:
    return complex(np.exp(2j * np.pi * (int(exponent) % int(order)) / int(order)))


def reduced_root_order(order: int, exponent: int) -> int:
    value = int(exponent) % int(order)
    if value == 0:
        return 1
    return int(order) // gcd(int(order), value)


def relative_frobenius(left: np.ndarray, right: np.ndarray) -> float:
    denom = max(float(np.linalg.norm(left, ord="fro")), EPS)
    return float(np.linalg.norm(left - right, ord="fro") / denom)


def relative_frobenius_to_reference(left: np.ndarray, right: np.ndarray) -> float:
    denom = max(float(np.linalg.norm(right, ord="fro")), EPS)
    return float(np.linalg.norm(left - right, ord="fro") / denom)


def permutation_matrix(perm: Sequence[int]) -> np.ndarray:
    arr = np.asarray(perm, dtype=int)
    matrix = np.zeros((len(arr), len(arr)), dtype=complex)
    matrix[np.arange(len(arr)), arr] = 1.0
    return matrix


def compose_permutations(*perms: Sequence[int]) -> np.ndarray:
    if not perms:
        raise ValueError("at least one permutation is required")
    out = np.asarray(perms[0], dtype=int)
    for perm in perms[1:]:
        out = np.asarray(perm, dtype=int)[out]
    return out


def _permutation_power(perm: np.ndarray, order: int) -> np.ndarray:
    out = np.arange(len(perm), dtype=int)
    step = np.asarray(perm, dtype=int)
    for _ in range(int(order)):
        out = step[out]
    return out


def permutation_cycle_lengths(perm: Sequence[int]) -> list[int]:
    arr = np.asarray(perm, dtype=int)
    visited = np.zeros(len(arr), dtype=bool)
    lengths: list[int] = []
    for start in range(len(arr)):
        if visited[start]:
            continue
        cur = int(start)
        length = 0
        while not visited[cur]:
            visited[cur] = True
            length += 1
            cur = int(arr[cur])
        lengths.append(length)
    return lengths


def analyze_permutation_residual(
    perm: Sequence[int],
    orders: Iterable[int] = DEFAULT_ORDERS,
) -> dict[str, float | int | bool | str]:
    """Fast scalar/finite-order diagnostics for a permutation residual."""

    arr = np.asarray(perm, dtype=int)
    width = int(len(arr))
    if width <= 0:
        raise ValueError("permutation must be nonempty")
    fixed = int(np.sum(arr == np.arange(width)))
    fixed_fraction = float(fixed / width)
    centrality = float(np.sqrt(max(0.0, 1.0 - fixed_fraction**2)))
    order_rows = []
    for order in orders:
        best = None
        for exponent in range(int(order)):
            root = root_of_unity(order, exponent)
            residual = float(np.sqrt(max(0.0, 2.0 - 2.0 * fixed_fraction * root.real)))
            root_order = reduced_root_order(order, exponent)
            key = (residual, root_order, exponent)
            if best is None or key < best[0]:
                best = (key, root, root_order)
        assert best is not None
        (scalar_residual, root_order, exponent), root, _root_order = best
        powered = _permutation_power(arr, int(order))
        fixed_power = float(np.mean(powered == np.arange(width)))
        finite_order_residual = float(np.sqrt(max(0.0, 2.0 - 2.0 * fixed_power)))
        order_rows.append(
            {
                "d": int(order),
                "scalar_residual": float(scalar_residual),
                "finite_order_residual": finite_order_residual,
                "root_exponent": int(exponent),
                "root_order": int(root_order),
                "root_real": float(root.real),
                "root_imag": float(root.imag),
                "phase_angle": float(np.angle(root)),
            }
        )
    best_row = min(
        order_rows,
        key=lambda row: (
            float(row["scalar_residual"]),
            int(row["root_order"]),
            int(row["d"]),
        ),
    )
    lengths = permutation_cycle_lengths(arr)
    noncentral_explanation = bool(centrality > 1e-3 and min(row["finite_order_residual"] for row in order_rows) <= 1e-3)
    return {
        "matrix_dim": width,
        "centrality_residual": centrality,
        "detected_order": int(best_row["root_order"]),
        "detected_phase": f"{best_row['root_real']:.8g}{best_row['root_imag']:+.8g}j",
        "phase_angle": float(best_row["phase_angle"]),
        "best_root_exponent": int(best_row["root_exponent"]),
        "best_root_search_order": int(best_row["d"]),
        "scalar_residual_best": float(best_row["scalar_residual"]),
        "finite_order_residual_best_search_order": float(best_row["finite_order_residual"]),
        "finite_order_residual_min": float(min(row["finite_order_residual"] for row in order_rows)),
        "eigenvalue_spread": float(np.std(lengths)) if lengths else 0.0,
        "condition_number": 1.0,
        "is_nontrivial_root": bool(int(best_row["root_order"]) > 1),
        "explained_as_noncentral_holonomy": noncentral_explanation,
        "fixed_point_fraction": fixed_fraction,
        "permutation_num_cycles": int(len(lengths)),
        "permutation_max_cycle_length": int(max(lengths)) if lengths else 0,
        **{
            f"scalar_residual_d{row['d']}": float(row["scalar_residual"])
            for row in order_rows
        },
        **{
            f"finite_order_residual_d{row['d']}": float(row["finite_order_residual"])
            for row in order_rows
        },
    }


def _best_scalar_root(matrix: np.ndarray, order: int) -> dict[str, float | int]:
    width = matrix.shape[0]
    identity = np.eye(width, dtype=complex)
    best = None
    for exponent in range(int(order)):
        root = root_of_unity(order, exponent)
        residual = relative_frobenius(matrix, root * identity)
        root_order = reduced_root_order(order, exponent)
        key = (residual, root_order, exponent)
        if best is None or key < best[0]:
            best = (key, root, root_order)
    assert best is not None
    (_residual, _root_order, exponent), root, root_order = best
    return {
        "scalar_residual": float(_residual),
        "root_exponent": int(exponent),
        "root_order": int(root_order),
        "root_real": float(root.real),
        "root_imag": float(root.imag),
        "phase_angle": float(np.angle(root)),
    }


def _eigenvalue_spread(matrix: np.ndarray) -> float:
    try:
        eigvals = np.linalg.eigvals(matrix)
    except np.linalg.LinAlgError:
        return float("inf")
    if eigvals.size == 0:
        return float("nan")
    radii = np.abs(eigvals)
    angles = np.angle(eigvals)
    return float(np.std(angles) + np.std(radii))


def analyze_residual_matrix(
    matrix: np.ndarray,
    orders: Iterable[int] = DEFAULT_ORDERS,
) -> dict[str, float | int | bool | str]:
    """Return scalar, centrality, and finite-order diagnostics for one residual."""

    arr = np.asarray(matrix, dtype=complex)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("residual matrix must be square")
    width = int(arr.shape[0])
    identity = np.eye(width, dtype=complex)
    scalar = complex(np.trace(arr) / max(width, 1))
    scalar_target = scalar * identity
    centrality = relative_frobenius(arr, scalar_target)

    order_rows = []
    for order in orders:
        root_fit = _best_scalar_root(arr, int(order))
        finite_order_residual = relative_frobenius_to_reference(
            np.linalg.matrix_power(arr, int(order)),
            identity,
        )
        order_rows.append(
            {
                "d": int(order),
                "finite_order_residual": float(finite_order_residual),
                **root_fit,
            }
        )

    best = min(
        order_rows,
        key=lambda row: (
            float(row["scalar_residual"]),
            int(row["root_order"]),
            int(row["d"]),
        ),
    )
    noncentral_explanation = bool(centrality > 1e-3 and min(float(row["finite_order_residual"]) for row in order_rows) <= 1e-3)
    try:
        condition_number = float(np.linalg.cond(arr))
    except np.linalg.LinAlgError:
        condition_number = float("inf")
    return {
        "matrix_dim": width,
        "centrality_residual": float(centrality),
        "detected_order": int(best["root_order"]),
        "detected_phase": f"{best['root_real']:.8g}{best['root_imag']:+.8g}j",
        "phase_angle": float(best["phase_angle"]),
        "best_root_exponent": int(best["root_exponent"]),
        "best_root_search_order": int(best["d"]),
        "scalar_residual_best": float(best["scalar_residual"]),
        "finite_order_residual_best_search_order": float(best["finite_order_residual"]),
        "finite_order_residual_min": float(min(row["finite_order_residual"] for row in order_rows)),
        "eigenvalue_spread": _eigenvalue_spread(arr),
        "condition_number": condition_number,
        "is_nontrivial_root": bool(int(best["root_order"]) > 1),
        "explained_as_noncentral_holonomy": noncentral_explanation,
        **{
            f"scalar_residual_d{row['d']}": float(row["scalar_residual"])
            for row in order_rows
        },
        **{
            f"finite_order_residual_d{row['d']}": float(row["finite_order_residual"])
            for row in order_rows
        },
    }


def policy_passes(metrics: dict, policy: TorsionThresholdPolicy) -> bool:
    return bool(
        metrics.get("is_nontrivial_root", False)
        and not metrics.get("explained_as_noncentral_holonomy", False)
        and float(metrics.get("centrality_residual", float("inf"))) <= policy.centrality_threshold
        and float(metrics.get("scalar_residual_best", float("inf"))) <= policy.scalar_threshold
        and float(metrics.get("finite_order_residual_min", float("inf"))) <= policy.order_threshold
    )


def policy_label(
    metrics: dict,
    policy: TorsionThresholdPolicy,
    false_positive_rate: float,
    bootstrap_detection_rate: float | None = None,
    bootstrap_order_agreement_rate: float | None = None,
) -> str:
    if not policy_passes(metrics, policy):
        if float(metrics.get("finite_order_residual_min", float("inf"))) <= policy.order_threshold:
            return "finite_order_noncentral_rejected"
        return "rejected"
    if false_positive_rate > policy.target_false_positive_rate:
        return "rejected_null_fpr"
    if bootstrap_detection_rate is not None and bootstrap_detection_rate < 0.8:
        return "rejected_bootstrap_unstable"
    if bootstrap_order_agreement_rate is not None and bootstrap_order_agreement_rate < 0.8:
        return "rejected_bootstrap_order_unstable"
    return "certified_torsion" if policy.activates_lift else "central_projective_candidate_uncertain"


def bootstrap_stability(
    matrix: np.ndarray,
    policy: TorsionThresholdPolicy,
    orders: Iterable[int] = DEFAULT_ORDERS,
    n_bootstrap: int = 200,
    seed: int = 0,
    min_fraction: float = 0.5,
) -> dict[str, float | int | str]:
    """Coordinate-resampling stability check for artifact-backed residuals."""

    arr = np.asarray(matrix, dtype=complex)
    width = arr.shape[0]
    if width <= 1 or n_bootstrap <= 0:
        metrics = analyze_residual_matrix(arr, orders)
        detected = policy_passes(metrics, policy)
        return {
            "bootstrap_mode": "identity_single_sample",
            "bootstrap_samples": int(max(n_bootstrap, 1)),
            "bootstrap_detection_rate": float(detected),
            "bootstrap_order_agreement_rate": float(detected),
            "bootstrap_phase_std": 0.0,
            "bootstrap_residual_mean": float(metrics["scalar_residual_best"]),
            "bootstrap_residual_std": 0.0,
        }

    base = analyze_residual_matrix(arr, orders)
    base_order = int(base["detected_order"])
    sample_size = max(1, int(round(float(min_fraction) * width)))
    rng = np.random.default_rng(seed)
    detections = []
    order_matches = []
    phases = []
    residuals = []
    for _ in range(int(n_bootstrap)):
        idx = rng.integers(0, width, size=sample_size)
        sub = arr[np.ix_(idx, idx)]
        metrics = analyze_residual_matrix(sub, orders)
        detections.append(policy_passes(metrics, policy))
        order_matches.append(int(metrics["detected_order"]) == base_order)
        phases.append(float(metrics["phase_angle"]))
        residuals.append(float(metrics["scalar_residual_best"]))
    return {
        "bootstrap_mode": "coordinate_resample_residual_submatrix",
        "bootstrap_samples": int(n_bootstrap),
        "bootstrap_detection_rate": float(np.mean(detections)),
        "bootstrap_order_agreement_rate": float(np.mean(order_matches)),
        "bootstrap_phase_std": float(np.nanstd(phases)),
        "bootstrap_residual_mean": float(np.nanmean(residuals)),
        "bootstrap_residual_std": float(np.nanstd(residuals)),
    }


def random_orthogonal_matrix(width: int, rng: np.random.Generator) -> np.ndarray:
    raw = rng.normal(size=(width, width))
    q, r = np.linalg.qr(raw)
    signs = np.sign(np.diag(r))
    signs[signs == 0.0] = 1.0
    return (q * signs).astype(complex)


def random_permutation(width: int, rng: np.random.Generator) -> np.ndarray:
    return rng.permutation(width).astype(int)


def permutation_commutator(width: int, rng: np.random.Generator) -> np.ndarray:
    p = random_permutation(width, rng)
    q = random_permutation(width, rng)
    p_mat = permutation_matrix(p)
    q_mat = permutation_matrix(q)
    return p_mat @ q_mat @ p_mat.T @ q_mat.T


def noncentral_s3_control(width: int) -> np.ndarray:
    if width < 3:
        width = 3
    p = np.arange(width)
    q = np.arange(width)
    p[[0, 1]] = p[[1, 0]]
    q[[1, 2]] = q[[2, 1]]
    return permutation_matrix(compose_permutations(p, q, p, q))


def noisy_fake_scalar(width: int, order: int, rng: np.random.Generator, noise_scale: float = 0.01) -> np.ndarray:
    root = root_of_unity(order, 1)
    diagonal_noise = rng.normal(loc=0.0, scale=noise_scale, size=width)
    return np.diag(root + diagonal_noise).astype(complex)
