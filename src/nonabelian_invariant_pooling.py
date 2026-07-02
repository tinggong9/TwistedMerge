"""Invariant pooling maps for Gamma-indexed branch spaces."""

from __future__ import annotations

import numpy as np

from src.finite_group_cohomology import FinitePermutationGroup, Permutation


def branch_permutation_matrix(perm: tuple[int, ...], feature_dim: int) -> np.ndarray:
    """Return the block permutation matrix acting on flattened branch features."""

    branches = len(perm)
    matrix = np.zeros((branches * feature_dim, branches * feature_dim), dtype=float)
    for src, dst in enumerate(perm):
        row = int(dst) * feature_dim
        col = int(src) * feature_dim
        matrix[row : row + feature_dim, col : col + feature_dim] = np.eye(feature_dim)
    return matrix


def invariant_pooling_matrix(group_order: int, feature_dim: int) -> np.ndarray:
    """Average over group-indexed branches and keep the feature coordinate."""

    matrix = np.zeros((feature_dim, int(group_order) * int(feature_dim)), dtype=float)
    for branch in range(int(group_order)):
        start = branch * int(feature_dim)
        matrix[:, start : start + int(feature_dim)] = np.eye(int(feature_dim)) / float(group_order)
    return matrix


def invariant_pool(branch_tensor: np.ndarray) -> np.ndarray:
    """Average a tensor of shape (..., branches, features) over branches."""

    arr = np.asarray(branch_tensor, dtype=float)
    return arr.mean(axis=-2)


def pooling_residual(branch_perm: tuple[int, ...], feature_dim: int) -> float:
    """Compute ||P rho(h)-P|| / ||P|| for branch permutation rho(h)."""

    pooling = invariant_pooling_matrix(len(branch_perm), int(feature_dim))
    action = branch_permutation_matrix(branch_perm, int(feature_dim))
    denom = max(float(np.linalg.norm(pooling, ord="fro")), 1e-12)
    return float(np.linalg.norm(pooling @ action - pooling, ord="fro") / denom)


def naive_representation_residual(branch_perm: tuple[int, ...], feature_dim: int = 1) -> float:
    """Compute ||rho(h)-I|| / ||I|| for a branch permutation representation."""

    action = branch_permutation_matrix(branch_perm, int(feature_dim))
    identity = np.eye(action.shape[0])
    denom = max(float(np.linalg.norm(identity, ord="fro")), 1e-12)
    return float(np.linalg.norm(action - identity, ord="fro") / denom)


def regular_action_permutation(group: FinitePermutationGroup, element: Permutation, side: str = "left") -> tuple[int, ...]:
    """Permutation of group-indexed branches induced by left or right action."""

    index = {item: idx for idx, item in enumerate(group.elements)}
    out = []
    inv_element = group.inverse(element)
    for branch in group.elements:
        if side == "left":
            image = group.multiply(element, branch)
        elif side == "right":
            image = group.multiply(branch, inv_element)
        else:
            raise ValueError("side must be 'left' or 'right'")
        out.append(index[image])
    return tuple(out)
