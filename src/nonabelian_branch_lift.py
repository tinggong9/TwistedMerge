"""Branch-lift helpers for controlled nonabelian holonomy experiments."""

from __future__ import annotations

import numpy as np

from src.finite_group_cohomology import FinitePermutationGroup, Permutation
from src.nonabelian_invariant_pooling import invariant_pool, regular_action_permutation


def gamma_branch_lift(hidden: np.ndarray, group: FinitePermutationGroup) -> np.ndarray:
    """Repeat hidden features over Gamma-indexed branches."""

    arr = np.asarray(hidden, dtype=float)
    repeats = [arr for _ in group.elements]
    return np.stack(repeats, axis=-2)


def apply_branch_action(branch_tensor: np.ndarray, group: FinitePermutationGroup, element: Permutation) -> np.ndarray:
    """Apply left-regular branch motion to a branch tensor."""

    perm = regular_action_permutation(group, element, side="left")
    arr = np.asarray(branch_tensor, dtype=float)
    inverse = np.argsort(np.asarray(perm, dtype=int))
    return arr[..., inverse, :]


def branch_lift_with_invariant_pooling(hidden: np.ndarray, group: FinitePermutationGroup, element: Permutation) -> np.ndarray:
    """Lift, move branches by element, and pool back to hidden coordinates."""

    lifted = gamma_branch_lift(hidden, group)
    moved = apply_branch_action(lifted, group, element)
    return invariant_pool(moved)


def oracle_true_branch_lift_logits(logits: np.ndarray) -> np.ndarray:
    return np.asarray(logits, dtype=float).copy()


def random_same_branch_count_control_logits(
    logits: np.ndarray,
    branch_count: int,
    rng: np.random.Generator,
    noise_scale: float = 1.0,
) -> np.ndarray:
    """Same branch-count control with randomized branch action/noise."""

    arr = np.asarray(logits, dtype=float)
    noise = rng.normal(scale=float(noise_scale) / max(1.0, np.sqrt(max(1, int(branch_count)))), size=arr.shape)
    return 0.7 * arr + noise
