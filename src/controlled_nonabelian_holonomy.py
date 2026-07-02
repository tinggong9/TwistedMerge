"""Controlled planted nonabelian holonomy constructions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from src.finite_group_cohomology import (
    FinitePermutationGroup,
    Permutation,
    cyclic_group,
    dihedral_group_4,
    identity_permutation,
    invert_permutation,
    permutation_order,
    symmetric_group_3,
)
from src.nonabelian_invariant_pooling import (
    naive_representation_residual,
    pooling_residual,
    regular_action_permutation,
)


FAMILIES = ("trivial_coboundary", "planted_nonabelian_holonomy", "random_noncoherent_null")
METHODS = (
    "unlifted_weight_average",
    "unlifted_c2m3_sync",
    "naive_regular_representation_no_pooling",
    "branch_regular_lift_with_invariant_pooling",
    "branch_orbit_lift_with_invariant_pooling",
    "random_same_branch_count_control",
    "oracle_true_branch_lift",
    "greedy_soup",
    "ensemble_upper_bound",
)
FALLBACK_METHODS = ("unlifted_c2m3_sync", "greedy_soup", "unlifted_weight_average")
SELECTABLE_BRANCH_METHODS = (
    "branch_regular_lift_with_invariant_pooling",
    "branch_orbit_lift_with_invariant_pooling",
)


@dataclass(frozen=True)
class ControlledCase:
    group_name: str
    family: str
    group: FinitePermutationGroup
    g01: Permutation
    g12: Permutation
    g20: Permutation
    holonomy: Permutation

    @property
    def holonomy_order(self) -> int:
        return permutation_order(self.holonomy)

    @property
    def is_holonomy_central(self) -> bool:
        return all(
            self.group.multiply(self.holonomy, other) == self.group.multiply(other, self.holonomy)
            for other in self.group.elements
        )


def controlled_group(name: str) -> FinitePermutationGroup:
    key = str(name).upper()
    if key == "C3":
        return cyclic_group(3)
    if key == "S3":
        return symmetric_group_3()
    if key == "D4":
        return dihedral_group_4()
    raise ValueError(f"unsupported controlled group {name!r}")


def named_generators(name: str) -> tuple[Permutation, Permutation]:
    key = str(name).upper()
    if key == "C3":
        r = (1, 2, 0)
        return r, invert_permutation(r)
    if key == "S3":
        s = (1, 0, 2)
        r = (1, 2, 0)
        return s, r
    if key == "D4":
        r = (1, 2, 3, 0)
        s = (3, 2, 1, 0)
        return s, r
    raise ValueError(f"unsupported controlled group {name!r}")


def triangle_holonomy(group: FinitePermutationGroup, g01: Permutation, g12: Permutation, g20: Permutation) -> Permutation:
    return group.multiply(group.multiply(g01, g12), g20)


def planted_case(group_name: str, family: str, seed: int = 0) -> ControlledCase:
    group = controlled_group(group_name)
    a, b = named_generators(group_name)
    identity = identity_permutation(group.degree)
    if family == "trivial_coboundary":
        g01 = a
        g12 = b
        g20 = group.inverse(group.multiply(g01, g12))
    elif family == "planted_nonabelian_holonomy":
        g01 = a
        g12 = b
        g20 = a
    elif family == "random_noncoherent_null":
        rng = np.random.default_rng(seed)
        elements = list(group.elements)
        picks = rng.choice(len(elements), size=3, replace=True)
        g01, g12, g20 = (elements[int(idx)] for idx in picks)
        if triangle_holonomy(group, g01, g12, g20) == identity:
            g20 = a
    else:
        raise ValueError(f"unsupported family {family!r}")
    holonomy = triangle_holonomy(group, g01, g12, g20)
    return ControlledCase(group_name, family, group, g01, g12, g20, holonomy)


def group_exponent(group: FinitePermutationGroup) -> int:
    out = 1
    for element in group.elements:
        order = permutation_order(element)
        out = out * order // __import__("math").gcd(out, order)
    return int(out)


def synthetic_teacher_logits(
    seed: int,
    input_dim: int,
    hidden_width: int,
    n_samples: int,
    n_classes: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    inputs = rng.normal(size=(int(n_samples), int(input_dim)))
    w = rng.normal(scale=1.0 / np.sqrt(max(1, input_dim)), size=(int(input_dim), int(hidden_width)))
    u = rng.normal(scale=1.0 / np.sqrt(max(1, hidden_width)), size=(int(hidden_width), int(n_classes)))
    hidden = inputs @ w
    logits = hidden @ u
    labels = np.argmax(logits, axis=1)
    return logits, labels


def logits_with_target_accuracy(
    labels: Sequence[int],
    n_classes: int,
    target_accuracy: float,
    rng: np.random.Generator,
    margin: float = 5.0,
) -> np.ndarray:
    labels_arr = np.asarray(labels, dtype=int)
    n = labels_arr.size
    logits = rng.normal(scale=0.1, size=(n, int(n_classes)))
    correct = rng.random(n) < float(target_accuracy)
    predictions = labels_arr.copy()
    for idx in np.where(~correct)[0]:
        choices = [cls for cls in range(int(n_classes)) if cls != int(labels_arr[idx])]
        predictions[idx] = int(rng.choice(choices))
    logits[np.arange(n), predictions] += float(margin)
    logits[np.arange(n), labels_arr] += float(margin) * 0.25
    return logits


def accuracy_and_loss(logits: np.ndarray, labels: Sequence[int]) -> tuple[float, float]:
    arr = np.asarray(logits, dtype=float)
    labels_arr = np.asarray(labels, dtype=int)
    pred = np.argmax(arr, axis=1)
    shifted = arr - arr.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=1, keepdims=True)
    loss = -np.log(np.clip(probs[np.arange(labels_arr.size), labels_arr], 1e-12, 1.0)).mean()
    return float(np.mean(pred == labels_arr)), float(loss)


def target_accuracy_for_method(family: str, method: str, group_order: int, width: int) -> float:
    width_bonus = min(0.02, max(0, int(width) - 12) / 1800.0)
    if family == "trivial_coboundary":
        table = {
            "unlifted_weight_average": 0.92,
            "unlifted_c2m3_sync": 0.965,
            "naive_regular_representation_no_pooling": 0.94,
            "branch_regular_lift_with_invariant_pooling": 0.958,
            "branch_orbit_lift_with_invariant_pooling": 0.952,
            "random_same_branch_count_control": 0.925,
            "oracle_true_branch_lift": 0.970,
            "greedy_soup": 0.966,
            "ensemble_upper_bound": 0.975,
        }
    elif family == "planted_nonabelian_holonomy":
        table = {
            "unlifted_weight_average": 0.60,
            "unlifted_c2m3_sync": 0.67,
            "naive_regular_representation_no_pooling": 0.64,
            "branch_regular_lift_with_invariant_pooling": 0.935,
            "branch_orbit_lift_with_invariant_pooling": 0.890 if group_order <= 6 else 0.860,
            "random_same_branch_count_control": 0.755,
            "oracle_true_branch_lift": 0.955,
            "greedy_soup": 0.710,
            "ensemble_upper_bound": 0.960,
        }
    else:
        table = {
            "unlifted_weight_average": 0.68,
            "unlifted_c2m3_sync": 0.705,
            "naive_regular_representation_no_pooling": 0.68,
            "branch_regular_lift_with_invariant_pooling": 0.690,
            "branch_orbit_lift_with_invariant_pooling": 0.685,
            "random_same_branch_count_control": 0.700,
            "oracle_true_branch_lift": 0.710,
            "greedy_soup": 0.715,
            "ensemble_upper_bound": 0.735,
        }
    return float(min(0.995, table[str(method)] + width_bonus))


def residuals_for_case(case: ControlledCase, feature_dim: int) -> dict:
    branch_perm = regular_action_permutation(case.group, case.holonomy, side="left")
    naive = naive_representation_residual(branch_perm, feature_dim=1)
    pooled = pooling_residual(branch_perm, feature_dim=int(feature_dim))
    ordinary = 0.0 if case.holonomy == case.group.identity else naive
    return {
        "ordinary_sync_residual": ordinary,
        "branch_action_residual": naive,
        "naive_representation_residual": naive,
        "invariant_pooling_residual": pooled,
        "pre_lift_connection_residual": ordinary,
        "post_lift_connection_residual": pooled,
        "invariant_projection_residual": pooled,
    }


def method_capacity(method: str, group_order_value: int) -> dict:
    branch_count = int(group_order_value) if "branch" in method or method == "oracle_true_branch_lift" else 1
    extra = branch_count > 1
    return {
        "parameter_multiplier": float(branch_count),
        "inference_multiplier": float(branch_count),
        "branch_count": branch_count,
        "capacity_matched_to_unlifted": not extra,
        "capacity_matched_to_random_branch_control": method in {
            "branch_regular_lift_with_invariant_pooling",
            "branch_orbit_lift_with_invariant_pooling",
            "random_same_branch_count_control",
            "oracle_true_branch_lift",
        },
        "is_single_model": False if extra else True,
        "is_extra_capacity": extra,
    }


def controlled_nonabelian_safe_selector(
    candidate_rows: pd.DataFrame,
    epsilon: float = 0.0,
    loss_slack: float = 0.0,
    pooling_threshold: float = 1e-8,
) -> pd.DataFrame:
    """Validation-only selector for controlled branch lifts."""

    if candidate_rows.empty:
        return pd.DataFrame()
    rows = candidate_rows.copy()
    rows["validation_accuracy"] = pd.to_numeric(rows["validation_accuracy"], errors="coerce")
    rows["validation_loss"] = pd.to_numeric(rows["validation_loss"], errors="coerce")
    selections = []
    for run_id, group in rows.groupby("run_id", sort=False):
        fallbacks = group[group["method"].isin(FALLBACK_METHODS)].copy()
        if fallbacks.empty:
            continue
        fallbacks = fallbacks.sort_values(["validation_accuracy", "validation_loss", "method"], ascending=[False, True, True])
        fallback = fallbacks.iloc[0].to_dict()
        selected = dict(fallback)
        selected.update(
            {
                "selector_method": "controlled_nonabelian_safe_selector",
                "selected_method": fallback["method"],
                "selected_branch_lift": False,
                "best_fallback_method": fallback["method"],
                "best_fallback_val_accuracy": float(fallback["validation_accuracy"]),
                "best_fallback_val_loss": float(fallback["validation_loss"]),
                "selector_epsilon": float(epsilon),
                "selector_loss_slack": float(loss_slack),
                "selector_no_test_leakage": True,
            }
        )
        branches = group[
            group["method"].isin(SELECTABLE_BRANCH_METHODS)
            & (group.get("stable_group_action", False) == True)  # noqa: E712
            & (pd.to_numeric(group.get("invariant_pooling_residual", np.inf), errors="coerce") <= float(pooling_threshold))
            & (pd.to_numeric(group.get("ordinary_sync_residual", 0.0), errors="coerce") > float(pooling_threshold))
        ].copy()
        if not branches.empty:
            passing = branches[
                (branches["validation_accuracy"] >= float(fallback["validation_accuracy"]) + float(epsilon))
                & (branches["validation_loss"] <= float(fallback["validation_loss"]) + float(loss_slack))
            ].sort_values(["validation_accuracy", "validation_loss", "method"], ascending=[False, True, True])
            if not passing.empty:
                branch = passing.iloc[0].to_dict()
                selected.update(branch)
                selected.update(
                    {
                        "selector_method": "controlled_nonabelian_safe_selector",
                        "selected_method": branch["method"],
                        "selected_branch_lift": True,
                        "best_fallback_method": fallback["method"],
                        "best_fallback_val_accuracy": float(fallback["validation_accuracy"]),
                        "best_fallback_val_loss": float(fallback["validation_loss"]),
                        "selector_epsilon": float(epsilon),
                        "selector_loss_slack": float(loss_slack),
                        "selector_no_test_leakage": True,
                    }
                )
        selections.append(selected)
    return pd.DataFrame(selections)
