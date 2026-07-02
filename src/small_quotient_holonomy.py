"""Small-quotient holonomy fitting for real permutation residuals.

The routines in this module deliberately fit quotients of observed permutation
triangle data only.  They do not certify central Brauer or projective
period-index structure, and they do not implement a model-level lift.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log
from typing import Iterable, Sequence

import numpy as np

from src.finite_group_cohomology import (
    FinitePermutationGroup,
    Permutation,
    cyclic_group,
    dihedral_group_4,
    identity_permutation,
    klein_four_group,
    permutation_order,
    symmetric_group_3,
)


QUOTIENT_NAMES = ("C2", "C3", "C4", "V4", "S3", "D4")


@dataclass(frozen=True)
class TriangleRelation:
    """A source relation p_ij p_jk p_ki = h_ijk."""

    first: Permutation
    second: Permutation
    third: Permutation
    holonomy: Permutation


@dataclass(frozen=True)
class QuotientFit:
    """Best heuristic map from observed source elements to a small group Q."""

    Q_name: str
    Q_group: FinitePermutationGroup
    assignment: dict[Permutation, Permutation]
    relations: tuple[TriangleRelation, ...]
    relation_violation_rate: float
    quotient_holonomy_nontrivial_rate: float
    quotient_holonomy_entropy: float
    quotient_holonomies: tuple[Permutation, ...]
    quotient_fit_status: str
    assignment_strategy: str
    score: float

    @property
    def Q_order(self) -> int:
        return int(self.Q_group.order)


def quotient_group(name: str) -> FinitePermutationGroup:
    key = str(name).upper()
    if key == "C2":
        return cyclic_group(2)
    if key == "C3":
        return cyclic_group(3)
    if key == "C4":
        return cyclic_group(4)
    if key == "V4":
        return klein_four_group()
    if key == "S3":
        return symmetric_group_3()
    if key == "D4":
        return dihedral_group_4()
    raise ValueError(f"unsupported quotient group {name!r}")


def candidate_quotients(names: Iterable[str] | None = None) -> dict[str, FinitePermutationGroup]:
    selected = QUOTIENT_NAMES if names is None else tuple(str(name).upper() for name in names)
    return {name: quotient_group(name) for name in selected}


def permutation_parity(perm: Sequence[int]) -> int:
    """Return 0 for even and 1 for odd permutations."""

    arr = tuple(int(value) for value in perm)
    seen = [False] * len(arr)
    cycles = 0
    for start in range(len(arr)):
        if seen[start]:
            continue
        cycles += 1
        cur = start
        while not seen[cur]:
            seen[cur] = True
            cur = arr[cur]
    return (len(arr) - cycles) % 2


def source_elements(relations: Iterable[TriangleRelation]) -> tuple[Permutation, ...]:
    seen: dict[Permutation, None] = {}
    for relation in relations:
        seen.setdefault(relation.first, None)
        seen.setdefault(relation.second, None)
        seen.setdefault(relation.third, None)
        seen.setdefault(relation.holonomy, None)
    return tuple(seen)


def element_order_table(group: FinitePermutationGroup) -> dict[Permutation, int]:
    return {element: permutation_order(element) for element in group.elements}


def compatible_quotient_images(source: Permutation, group: FinitePermutationGroup) -> tuple[Permutation, ...]:
    source_order = permutation_order(source)
    orders = element_order_table(group)
    out = [
        element
        for element, order in orders.items()
        if order == 1 or (source_order > 0 and source_order % int(order) == 0)
    ]
    return tuple(out) if out else (group.identity,)


def _distribution(elements: Sequence[Permutation], group: FinitePermutationGroup) -> np.ndarray:
    counts = np.zeros(group.order, dtype=float)
    index = {element: idx for idx, element in enumerate(group.elements)}
    for element in elements:
        counts[index[element]] += 1.0
    if counts.sum() <= 0:
        return counts
    return counts / counts.sum()


def _entropy(elements: Sequence[Permutation], group: FinitePermutationGroup) -> float:
    probs = _distribution(elements, group)
    probs = probs[probs > 0]
    if probs.size == 0 or group.order <= 1:
        return 0.0
    return float(-np.sum(probs * np.log(probs)) / log(group.order))


def evaluate_assignment(
    relations: Iterable[TriangleRelation],
    group: FinitePermutationGroup,
    assignment: dict[Permutation, Permutation],
) -> dict:
    rels = tuple(relations)
    violations = 0
    holonomies = []
    for relation in rels:
        q_first = assignment.get(relation.first, group.identity)
        q_second = assignment.get(relation.second, group.identity)
        q_third = assignment.get(relation.third, group.identity)
        q_holonomy = assignment.get(relation.holonomy, group.identity)
        product = group.multiply(group.multiply(q_first, q_second), q_third)
        holonomies.append(product)
        if product != q_holonomy:
            violations += 1
    nontrivial = [value != group.identity for value in holonomies]
    relation_count = max(1, len(rels))
    return {
        "relation_violation_rate": float(violations / relation_count),
        "quotient_holonomy_nontrivial_rate": float(np.mean(nontrivial)) if nontrivial else 0.0,
        "quotient_holonomy_entropy": _entropy(holonomies, group),
        "quotient_holonomies": tuple(holonomies),
    }


def _fit_score(evaluation: dict) -> float:
    return float(
        10.0 * evaluation["relation_violation_rate"]
        - evaluation["quotient_holonomy_nontrivial_rate"]
        - 0.25 * evaluation["quotient_holonomy_entropy"]
    )


def _identity_assignment(elements: Sequence[Permutation], group: FinitePermutationGroup) -> dict[Permutation, Permutation]:
    return {element: group.identity for element in elements}


def _parity_assignments(elements: Sequence[Permutation], group: FinitePermutationGroup) -> list[tuple[str, dict[Permutation, Permutation]]]:
    order_two = [element for element in group.elements if permutation_order(element) == 2]
    out = []
    for idx, nontrivial in enumerate(order_two):
        assignment = {
            element: (nontrivial if permutation_parity(element) else group.identity)
            for element in elements
        }
        out.append((f"ambient_parity_to_order2_{idx}", assignment))
    return out


def _random_assignment(
    elements: Sequence[Permutation],
    group: FinitePermutationGroup,
    rng: np.random.Generator,
) -> dict[Permutation, Permutation]:
    out = {}
    for element in elements:
        choices = compatible_quotient_images(element, group)
        out[element] = choices[int(rng.integers(0, len(choices)))]
    return out


def fit_quotient_map(
    relations: Iterable[TriangleRelation],
    Q_name: str,
    seed: int = 0,
    random_restarts: int = 64,
) -> QuotientFit:
    rels = tuple(relations)
    group = quotient_group(Q_name)
    elements = source_elements(rels)
    candidates: list[tuple[str, dict[Permutation, Permutation]]] = [
        ("identity", _identity_assignment(elements, group)),
        *_parity_assignments(elements, group),
    ]
    rng = np.random.default_rng(seed)
    for restart in range(int(random_restarts)):
        candidates.append((f"random_restart_{restart}", _random_assignment(elements, group, rng)))

    best_strategy = "identity"
    best_assignment = candidates[0][1]
    best_eval = evaluate_assignment(rels, group, best_assignment)
    best_score = _fit_score(best_eval)
    for strategy, assignment in candidates[1:]:
        evaluation = evaluate_assignment(rels, group, assignment)
        score = _fit_score(evaluation)
        if score < best_score:
            best_strategy = strategy
            best_assignment = assignment
            best_eval = evaluation
            best_score = score

    status = "heuristic" if random_restarts > 0 else "deterministic"
    return QuotientFit(
        Q_name=str(Q_name).upper(),
        Q_group=group,
        assignment=best_assignment,
        relations=rels,
        relation_violation_rate=float(best_eval["relation_violation_rate"]),
        quotient_holonomy_nontrivial_rate=float(best_eval["quotient_holonomy_nontrivial_rate"]),
        quotient_holonomy_entropy=float(best_eval["quotient_holonomy_entropy"]),
        quotient_holonomies=tuple(best_eval["quotient_holonomies"]),
        quotient_fit_status=status,
        assignment_strategy=best_strategy,
        score=float(best_score),
    )


def bootstrap_quotient_fit(
    fit: QuotientFit,
    relation_threshold: float,
    nontrivial_threshold: float,
    n_bootstrap: int = 100,
    seed: int = 0,
) -> dict:
    if not fit.relations:
        return {
            "bootstrap_samples": int(n_bootstrap),
            "bootstrap_same_Q_rate": 0.0,
            "bootstrap_holonomy_distribution_stability": 0.0,
            "bootstrap_relation_violation_mean": np.nan,
            "bootstrap_nontrivial_rate_mean": np.nan,
        }
    rng = np.random.default_rng(seed)
    rels = np.asarray(fit.relations, dtype=object)
    full_distribution = _distribution(fit.quotient_holonomies, fit.Q_group)
    stable = []
    distribution_stability = []
    violations = []
    nontrivial = []
    for _ in range(int(n_bootstrap)):
        sample = tuple(rels[rng.integers(0, len(rels), size=len(rels))])
        evaluation = evaluate_assignment(sample, fit.Q_group, fit.assignment)
        violation = float(evaluation["relation_violation_rate"])
        rate = float(evaluation["quotient_holonomy_nontrivial_rate"])
        distribution = _distribution(evaluation["quotient_holonomies"], fit.Q_group)
        tv_distance = 0.5 * float(np.sum(np.abs(distribution - full_distribution)))
        stability = max(0.0, 1.0 - tv_distance)
        violations.append(violation)
        nontrivial.append(rate)
        distribution_stability.append(stability)
        stable.append(violation <= float(relation_threshold) and rate >= float(nontrivial_threshold))
    return {
        "bootstrap_samples": int(n_bootstrap),
        "bootstrap_same_Q_rate": float(np.mean(stable)) if stable else 0.0,
        "bootstrap_holonomy_distribution_stability": float(np.mean(distribution_stability))
        if distribution_stability
        else 0.0,
        "bootstrap_relation_violation_mean": float(np.mean(violations)) if violations else np.nan,
        "bootstrap_nontrivial_rate_mean": float(np.mean(nontrivial)) if nontrivial else np.nan,
    }


def quotient_certified(
    fit: QuotientFit,
    bootstrap: dict,
    relation_threshold: float,
    nontrivial_threshold: float,
    bootstrap_same_Q_threshold: float = 0.8,
    bootstrap_distribution_threshold: float = 0.8,
) -> bool:
    return bool(
        fit.relation_violation_rate <= float(relation_threshold)
        and fit.quotient_holonomy_nontrivial_rate >= float(nontrivial_threshold)
        and float(bootstrap.get("bootstrap_same_Q_rate", 0.0)) >= float(bootstrap_same_Q_threshold)
        and float(bootstrap.get("bootstrap_holonomy_distribution_stability", 0.0))
        >= float(bootstrap_distribution_threshold)
    )


def fit_summary_row(
    fit: QuotientFit,
    relation_threshold: float,
    nontrivial_threshold: float,
    bootstrap: dict,
    base: dict | None = None,
) -> dict:
    certified = quotient_certified(fit, bootstrap, relation_threshold, nontrivial_threshold)
    index = {element: idx for idx, element in enumerate(fit.Q_group.elements)}
    holonomy_counts = {}
    for element in fit.quotient_holonomies:
        key = str(index[element])
        holonomy_counts[key] = holonomy_counts.get(key, 0) + 1
    image_counts = {}
    for image in fit.assignment.values():
        key = str(index[image])
        image_counts[key] = image_counts.get(key, 0) + 1
    return {
        **(base or {}),
        "Q_name": fit.Q_name,
        "Q_order": fit.Q_order,
        "quotient_fit_status": fit.quotient_fit_status,
        "assignment_strategy": fit.assignment_strategy,
        "relation_threshold": float(relation_threshold),
        "nontrivial_threshold": float(nontrivial_threshold),
        "relation_violation_rate": fit.relation_violation_rate,
        "quotient_holonomy_nontrivial_rate": fit.quotient_holonomy_nontrivial_rate,
        "quotient_holonomy_entropy": fit.quotient_holonomy_entropy,
        "quotient_score": fit.score,
        "source_element_count": int(len(fit.assignment)),
        "triangle_relation_count": int(len(fit.relations)),
        "quotient_certified": certified,
        "quotient_status": "certified_candidate" if certified else "unstable_no_lift",
        "assignment_image_counts_json": image_counts,
        "quotient_holonomy_counts_json": holonomy_counts,
        **bootstrap,
    }


def null_random_assignment_rate(
    relations: Iterable[TriangleRelation],
    Q_name: str,
    relation_threshold: float,
    nontrivial_threshold: float,
    n_null: int,
    seed: int,
) -> dict:
    rels = tuple(relations)
    group = quotient_group(Q_name)
    elements = source_elements(rels)
    rng = np.random.default_rng(seed)
    pass_flags = []
    pool_flags = []
    nontrivial_rates = []
    violations = []
    for _ in range(int(n_null)):
        assignment = _random_assignment(elements, group, rng)
        evaluation = evaluate_assignment(rels, group, assignment)
        violation = float(evaluation["relation_violation_rate"])
        nontrivial = float(evaluation["quotient_holonomy_nontrivial_rate"])
        pass_flags.append(violation <= float(relation_threshold) and nontrivial >= float(nontrivial_threshold))
        pool_flags.append(bool(evaluation["quotient_holonomies"]))
        violations.append(violation)
        nontrivial_rates.append(nontrivial)
    return {
        "false_quotient_certification_rate": float(np.mean(pass_flags)) if pass_flags else 0.0,
        "false_pooling_pass_rate": float(np.mean(pool_flags)) if pool_flags else 0.0,
        "null_relation_violation_mean": float(np.mean(violations)) if violations else np.nan,
        "null_nontrivial_rate_mean": float(np.mean(nontrivial_rates)) if nontrivial_rates else np.nan,
    }


def triangle_relation_from_perms(
    first: Sequence[int],
    second: Sequence[int],
    third: Sequence[int],
    holonomy: Sequence[int] | None = None,
) -> TriangleRelation:
    first_p = tuple(int(value) for value in first)
    second_p = tuple(int(value) for value in second)
    third_p = tuple(int(value) for value in third)
    if holonomy is None:
        hol = identity_permutation(len(first_p))
    else:
        hol = tuple(int(value) for value in holonomy)
    return TriangleRelation(first_p, second_p, third_p, hol)
