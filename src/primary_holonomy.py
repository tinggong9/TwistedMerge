"""Primary-factor holonomy fitting utilities.

These helpers estimate cyclic 2-primary, 3-primary, and small mixed factors of
observed permutation holonomy.  They are nonabelian-descent diagnostics, not
central Brauer or period-index computations.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd, log
from typing import Iterable, Sequence

import numpy as np

from src.finite_group_cohomology import Permutation, identity_permutation, permutation_order
from src.small_quotient_holonomy import TriangleRelation


PRIMARY_Q_2 = (2, 4, 8, 16, 32)
PRIMARY_Q_3 = (3, 9, 27)
MIXED_Q = (6, 12, 18, 36)
DEFAULT_Q_ORDERS = (*PRIMARY_Q_2, *PRIMARY_Q_3, *MIXED_Q)


@dataclass(frozen=True)
class PrimaryFit:
    q_order: int
    primary_type: str
    primary_depth: int
    assignment: dict[Permutation, int]
    relations: tuple[TriangleRelation, ...]
    relation_violation_rate: float
    quotient_holonomy_nontrivial_rate: float
    quotient_holonomy_entropy: float
    quotient_assignment_confidence: float
    quotient_holonomy_residues: tuple[int, ...]
    quotient_fit_status: str
    assignment_strategy: str


def p_adic_valuation(value: int | float | None, prime: int) -> int:
    if value is None or not np.isfinite(float(value)):
        return 0
    n = abs(int(value))
    if n <= 0:
        return 0
    out = 0
    while n % int(prime) == 0:
        out += 1
        n //= int(prime)
    return int(out)


def primary_type_and_depth(q_order: int) -> tuple[str, int]:
    q = int(q_order)
    if q in PRIMARY_Q_2:
        return "2-primary", p_adic_valuation(q, 2)
    if q in PRIMARY_Q_3:
        return "3-primary", p_adic_valuation(q, 3)
    return "mixed", max(p_adic_valuation(q, 2), p_adic_valuation(q, 3))


def prime_axis_metadata(q_order: int, observed_order: int) -> dict:
    """Classify q as a headline prime axis, depth control, or mixed control."""

    q = int(q_order)
    v2_q = p_adic_valuation(q, 2)
    v3_q = p_adic_valuation(q, 3)
    v2_obs = p_adic_valuation(observed_order, 2)
    v3_obs = p_adic_valuation(observed_order, 3)
    is_mixed = bool(v2_q > 0 and v3_q > 0)
    is_prime_headline = q in {2, 3}
    is_depth_control = bool((q in PRIMARY_Q_2 and q != 2) or (q in PRIMARY_Q_3 and q != 3))
    is_mixed_control = bool(is_mixed)
    if v2_q > 0 and v3_q == 0:
        prime_axis = "C2"
        observed_depth = v2_obs
        eligible = v2_obs >= v2_q
    elif v3_q > 0 and v2_q == 0:
        prime_axis = "C3"
        observed_depth = v3_obs
        eligible = v3_obs >= v3_q
    elif is_mixed:
        prime_axis = "C2_then_C3"
        observed_depth = min(v2_obs, v3_obs)
        eligible = bool(v2_obs >= v2_q and v3_obs >= v3_q)
    else:
        prime_axis = "none"
        observed_depth = 0
        eligible = False
    return {
        "prime_axis": prime_axis,
        "is_prime_headline": bool(is_prime_headline),
        "is_depth_control": bool(is_depth_control),
        "is_mixed_control": bool(is_mixed_control),
        "v_p_observed_order": int(observed_depth),
        "eligible_by_observed_primary_depth": bool(eligible),
    }


def lcm(left: int, right: int) -> int:
    if left == 0 or right == 0:
        return 0
    return abs(int(left) * int(right)) // gcd(int(left), int(right))


def lcm_many(values: Iterable[int]) -> int:
    out = 1
    for value in values:
        out = lcm(out, int(value))
    return int(out)


def observed_holonomy_order_lcm(relations: Iterable[TriangleRelation]) -> int:
    orders = []
    for relation in relations:
        try:
            orders.append(permutation_order(relation.holonomy))
        except Exception:
            continue
    return lcm_many(orders) if orders else 1


def source_elements(relations: Iterable[TriangleRelation]) -> tuple[Permutation, ...]:
    seen: dict[Permutation, None] = {}
    for relation in relations:
        seen.setdefault(relation.first, None)
        seen.setdefault(relation.second, None)
        seen.setdefault(relation.third, None)
        seen.setdefault(relation.holonomy, None)
    return tuple(seen)


def relation_count_status(relation_count: int, min_relation_count: int = 4) -> str:
    return "sufficient" if int(relation_count) >= int(min_relation_count) else "underconstrained"


def q_divides_primary_source(q_order: int, observed_order: int, group_exponent: int | None = None) -> bool:
    q = int(q_order)
    if q <= 0:
        return False
    if int(observed_order) % q == 0:
        return True
    if group_exponent is not None and int(group_exponent) > 0 and int(group_exponent) % q == 0:
        return True
    return False


def candidate_q_orders_for_source(observed_order: int, group_exponent: int | None = None) -> list[dict]:
    rows = []
    for q_order in DEFAULT_Q_ORDERS:
        p_type, depth = primary_type_and_depth(q_order)
        axis = prime_axis_metadata(q_order, observed_order)
        divides = q_divides_primary_source(q_order, observed_order, group_exponent)
        if divides and axis["eligible_by_observed_primary_depth"] and axis["is_prime_headline"]:
            role = "prime_headline"
        elif divides and axis["eligible_by_observed_primary_depth"] and axis["is_depth_control"]:
            role = "depth_control"
        elif divides and axis["eligible_by_observed_primary_depth"] and axis["is_mixed_control"]:
            role = "mixed_control"
        else:
            role = "wrong_prime_or_depth_control"
        rows.append(
            {
                "q_order": int(q_order),
                "q_name": f"C{int(q_order)}",
                "primary_type": p_type,
                "primary_depth": int(depth),
                **axis,
                "divides_primary_source": bool(divides),
                "candidate_role": role,
            }
        )
    return rows


def _residue_entropy(residues: Sequence[int], q_order: int) -> float:
    q = int(q_order)
    if not residues or q <= 1:
        return 0.0
    counts = np.bincount(np.asarray(residues, dtype=int) % q, minlength=q).astype(float)
    probs = counts / max(1.0, counts.sum())
    probs = probs[probs > 0]
    if probs.size <= 1:
        return 0.0
    return float(-np.sum(probs * np.log(probs)) / log(q))


def _primary_residue_from_order(order: int, q_order: int) -> int:
    q = int(q_order)
    common = gcd(max(1, int(order)), q)
    if common <= 1:
        return 0
    return int(q // common) % q


def _complete_assignment(relations: tuple[TriangleRelation, ...], q_order: int) -> tuple[dict[Permutation, int], int]:
    assignment: dict[Permutation, int] = {}
    conflicts = 0
    for relation in relations:
        for element in (relation.first, relation.second, relation.third, relation.holonomy):
            if element == identity_permutation(len(element)):
                assignment[element] = 0
    for relation in relations:
        h_residue = _primary_residue_from_order(permutation_order(relation.holonomy), q_order)
        old = assignment.get(relation.holonomy)
        if old is not None and old != h_residue:
            conflicts += 1
        assignment[relation.holonomy] = h_residue

        edge_values = [assignment.get(relation.first), assignment.get(relation.second), assignment.get(relation.third)]
        missing = [idx for idx, value in enumerate(edge_values) if value is None]
        if len(missing) == 3:
            assignment[relation.first] = 0
            assignment[relation.second] = 0
            assignment[relation.third] = h_residue
        elif missing:
            current = sum(value for value in edge_values if value is not None) % int(q_order)
            fill = (h_residue - current) % int(q_order)
            target = (relation.first, relation.second, relation.third)[missing[0]]
            assignment[target] = fill
            for idx in missing[1:]:
                assignment[(relation.first, relation.second, relation.third)[idx]] = 0
        else:
            lhs = sum(edge_values) % int(q_order)
            if lhs != h_residue:
                conflicts += 1
    return assignment, conflicts


def evaluate_primary_assignment(
    relations: Iterable[TriangleRelation],
    q_order: int,
    assignment: dict[Permutation, int],
) -> dict:
    q = int(q_order)
    rels = tuple(relations)
    violations = 0
    residues = []
    for relation in rels:
        lhs = (
            int(assignment.get(relation.first, 0))
            + int(assignment.get(relation.second, 0))
            + int(assignment.get(relation.third, 0))
        ) % q
        h_residue = int(assignment.get(relation.holonomy, 0)) % q
        residues.append(h_residue)
        if lhs != h_residue:
            violations += 1
    n_rel = max(1, len(rels))
    nontrivial = [residue % q != 0 for residue in residues]
    return {
        "relation_violation_rate": float(violations / n_rel),
        "quotient_holonomy_nontrivial_rate": float(np.mean(nontrivial)) if nontrivial else 0.0,
        "quotient_holonomy_entropy": _residue_entropy(residues, q),
        "quotient_holonomy_residues": tuple(int(residue % q) for residue in residues),
    }


def fit_primary_quotient(
    relations: Iterable[TriangleRelation],
    q_order: int,
    random_restarts: int = 0,
    seed: int = 0,
) -> PrimaryFit:
    rels = tuple(relations)
    q = int(q_order)
    p_type, depth = primary_type_and_depth(q)
    assignment, conflicts = _complete_assignment(rels, q)
    best_eval = evaluate_primary_assignment(rels, q, assignment)
    best_assignment = assignment
    best_score = 10.0 * best_eval["relation_violation_rate"] - best_eval["quotient_holonomy_nontrivial_rate"]
    rng = np.random.default_rng(seed)
    elements = source_elements(rels)
    for _ in range(int(random_restarts)):
        trial = dict(best_assignment)
        for element in elements:
            if element == identity_permutation(len(element)):
                trial[element] = 0
                continue
            if rng.random() < 0.1:
                trial[element] = int(rng.integers(0, q))
        evaluation = evaluate_primary_assignment(rels, q, trial)
        score = 10.0 * evaluation["relation_violation_rate"] - evaluation["quotient_holonomy_nontrivial_rate"]
        if score < best_score:
            best_assignment = trial
            best_eval = evaluation
            best_score = score
    source_count = max(1, len(elements))
    confidence = max(0.0, min(1.0, 1.0 - conflicts / max(1, len(rels)) + min(0.25, len(rels) / source_count) - 0.25))
    return PrimaryFit(
        q_order=q,
        primary_type=p_type,
        primary_depth=depth,
        assignment=best_assignment,
        relations=rels,
        relation_violation_rate=float(best_eval["relation_violation_rate"]),
        quotient_holonomy_nontrivial_rate=float(best_eval["quotient_holonomy_nontrivial_rate"]),
        quotient_holonomy_entropy=float(best_eval["quotient_holonomy_entropy"]),
        quotient_assignment_confidence=float(confidence),
        quotient_holonomy_residues=tuple(best_eval["quotient_holonomy_residues"]),
        quotient_fit_status="heuristic_primary_residue_completion",
        assignment_strategy="holonomy_primary_residue_edge_completion",
    )


def bootstrap_primary_fit(
    fit: PrimaryFit,
    relation_threshold: float = 0.01,
    nontrivial_threshold: float = 0.10,
    entropy_threshold: float = 0.10,
    n_bootstrap: int = 100,
    seed: int = 0,
) -> dict:
    rels = tuple(fit.relations)
    if not rels:
        return {
            "bootstrap_samples": int(n_bootstrap),
            "bootstrap_meaningful": False,
            "bootstrap_same_q_rate": 0.0,
            "bootstrap_same_primary_depth_rate": 0.0,
            "bootstrap_relation_violation_mean": np.nan,
            "bootstrap_relation_violation_std": np.nan,
            "bootstrap_holonomy_distribution_stability": 0.0,
            "bootstrap_nontrivial_rate_mean": np.nan,
            "bootstrap_nontrivial_rate_std": np.nan,
        }
    rng = np.random.default_rng(seed)
    full_counts = np.bincount(np.asarray(fit.quotient_holonomy_residues, dtype=int) % fit.q_order, minlength=fit.q_order)
    full_dist = full_counts / max(1.0, float(full_counts.sum()))
    pass_flags = []
    depth_flags = []
    violations = []
    nontrivial_rates = []
    stabilities = []
    for _ in range(int(n_bootstrap)):
        sample = tuple(rels[idx] for idx in rng.integers(0, len(rels), size=len(rels)))
        evaluation = evaluate_primary_assignment(sample, fit.q_order, fit.assignment)
        residues = np.asarray(evaluation["quotient_holonomy_residues"], dtype=int) % fit.q_order
        counts = np.bincount(residues, minlength=fit.q_order)
        dist = counts / max(1.0, float(counts.sum()))
        tv_distance = 0.5 * float(np.sum(np.abs(dist - full_dist)))
        stability = max(0.0, 1.0 - tv_distance)
        violation = float(evaluation["relation_violation_rate"])
        nontrivial = float(evaluation["quotient_holonomy_nontrivial_rate"])
        entropy = float(evaluation["quotient_holonomy_entropy"])
        pass_flag = (
            violation <= float(relation_threshold)
            and nontrivial >= float(nontrivial_threshold)
            and entropy >= float(entropy_threshold)
        )
        pass_flags.append(pass_flag)
        depth_flags.append(pass_flag and any(int(residue) % fit.q_order for residue in residues))
        violations.append(violation)
        nontrivial_rates.append(nontrivial)
        stabilities.append(stability)
    return {
        "bootstrap_samples": int(n_bootstrap),
        "bootstrap_meaningful": len(rels) >= 4,
        "bootstrap_same_q_rate": float(np.mean(pass_flags)) if pass_flags else 0.0,
        "bootstrap_same_primary_depth_rate": float(np.mean(depth_flags)) if depth_flags else 0.0,
        "bootstrap_relation_violation_mean": float(np.mean(violations)) if violations else np.nan,
        "bootstrap_relation_violation_std": float(np.std(violations, ddof=1)) if len(violations) > 1 else 0.0,
        "bootstrap_holonomy_distribution_stability": float(np.mean(stabilities)) if stabilities else 0.0,
        "bootstrap_nontrivial_rate_mean": float(np.mean(nontrivial_rates)) if nontrivial_rates else np.nan,
        "bootstrap_nontrivial_rate_std": float(np.std(nontrivial_rates, ddof=1)) if len(nontrivial_rates) > 1 else 0.0,
    }


def q_branch_permutation(q_order: int, residue: int) -> Permutation:
    q = int(q_order)
    r = int(residue) % q
    return tuple((idx + r) % q for idx in range(q))


def primary_pooling_residuals(fit: PrimaryFit, feature_dim: int = 1) -> dict:
    from src.nonabelian_invariant_pooling import naive_representation_residual, pooling_residual

    naive = []
    pooled = []
    for residue in fit.quotient_holonomy_residues:
        perm = q_branch_permutation(fit.q_order, residue)
        naive.append(naive_representation_residual(perm, feature_dim=1))
        pooled.append(pooling_residual(perm, feature_dim=int(feature_dim)))
    return {
        "naive_residual_q": float(np.mean(naive)) if naive else np.nan,
        "naive_residual_q_max": float(np.max(naive)) if naive else np.nan,
        "pooling_residual_q": float(np.mean(pooled)) if pooled else np.nan,
        "pooling_residual_q_max": float(np.max(pooled)) if pooled else np.nan,
    }


def primary_fit_certified(
    fit: PrimaryFit,
    bootstrap: dict,
    relation_count: int,
    relation_threshold: float = 0.01,
    nontrivial_threshold: float = 0.10,
    entropy_threshold: float = 0.10,
    min_relation_count: int = 4,
) -> bool:
    return bool(
        int(relation_count) >= int(min_relation_count)
        and fit.relation_violation_rate <= float(relation_threshold)
        and fit.quotient_holonomy_nontrivial_rate >= float(nontrivial_threshold)
        and fit.quotient_holonomy_entropy >= float(entropy_threshold)
        and float(bootstrap.get("bootstrap_same_q_rate", 0.0)) >= 0.8
        and float(bootstrap.get("bootstrap_same_primary_depth_rate", 0.0)) >= 0.8
        and float(bootstrap.get("bootstrap_holonomy_distribution_stability", 0.0)) >= 0.8
    )


def triangle_relation_from_perms(
    first: Sequence[int],
    second: Sequence[int],
    third: Sequence[int],
    holonomy: Sequence[int] | None = None,
) -> TriangleRelation:
    first_p = tuple(int(value) for value in first)
    second_p = tuple(int(value) for value in second)
    third_p = tuple(int(value) for value in third)
    hol = identity_permutation(len(first_p)) if holonomy is None else tuple(int(value) for value in holonomy)
    return TriangleRelation(first_p, second_p, third_p, hol)
