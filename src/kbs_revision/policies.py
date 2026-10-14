"""Frozen action rules. Algebraic gate controls are not neural risk observations."""
from __future__ import annotations

from dataclasses import dataclass
import math

GATES = ("inverse_consistency", "coefficient_identification", "projection_fidelity", "centrality",
         "closure", "uncertainty_margin", "topology")


@dataclass(frozen=True)
class Candidate:
    method_id: str
    validation_loss: float
    structural_candidate: bool = False


@dataclass(frozen=True)
class Evidence:
    coefficient_model: str = "inapplicable"
    gates: tuple[tuple[str, bool], ...] = ()
    upper_defect: float | None = None
    lower_defect: float | None = None
    exact_trivial: bool | None = None
    pairwise_loss: float | None = None
    hodge_residual: float | None = None


def decide(policy, candidates, fallback, evidence, *, tolerance=0.05, separation_threshold=0.1, pairwise_threshold=0.05,
           removed_gates=()):
    if not 0 <= tolerance < separation_threshold or not math.isfinite(separation_threshold):
        raise ValueError("thresholds must satisfy finite 0 <= a < b")
    for bound in (evidence.upper_defect, evidence.lower_defect, evidence.pairwise_loss, evidence.hodge_residual):
        if bound is not None and (not math.isfinite(bound) or bound < 0):
            raise ValueError("bounds and diagnostic losses must be finite and nonnegative")
    if evidence.lower_defect is not None and evidence.upper_defect is not None and evidence.lower_defect > evidence.upper_defect:
        raise ValueError("lower defect cannot exceed upper defect")
    by_id = {candidate.method_id: candidate for candidate in candidates}
    if len(by_id) != len(candidates) or fallback not in by_id:
        raise ValueError("unique candidates and a constructed fallback are required")
    if any(not math.isfinite(c.validation_loss) or not 0 <= c.validation_loss <= 1 for c in candidates):
        raise ValueError("validation losses must be finite and bounded")
    if policy == "ORACLE":
        raise ValueError("ORACLE is evaluation-only; it cannot choose deployable actions")
    gates = dict(evidence.gates)
    valid = all(gates.get(gate, False) for gate in GATES if gate not in removed_gates)
    if policy == "P0":
        admitted = [by_id[fallback]]
    elif policy == "P1":
        admitted = list(candidates)
    elif policy == "P2":
        admitted = list(candidates) if evidence.pairwise_loss is not None and evidence.pairwise_loss <= pairwise_threshold else [by_id[fallback]]
    elif policy in ("P3", "P3G", "P4"):
        admitted = [candidate for candidate in candidates if not candidate.structural_candidate]
        if policy == "P3":
            allow = evidence.hodge_residual is not None and evidence.hodge_residual <= tolerance
        elif policy == "P3G" or evidence.coefficient_model == "vector":
            allow = (valid and evidence.coefficient_model == "vector" and evidence.upper_defect is not None
                     and evidence.upper_defect <= tolerance)
        else:
            allow = (valid and evidence.coefficient_model == "mu2" and evidence.exact_trivial is True)
        if allow:
            admitted = list(candidates)
    else:
        raise ValueError(f"unknown policy: {policy}")
    if by_id[fallback] not in admitted:
        admitted.append(by_id[fallback])
    selected = min(admitted, key=lambda c: (c.validation_loss, c.method_id))
    return {"policy_id": policy, "selected_action": selected.method_id, "fallback_action": fallback,
            "admitted_candidate_count": len(admitted), "coefficient_model": evidence.coefficient_model,
            "input_exact_trivial": evidence.exact_trivial,
            "exact_trivial": evidence.exact_trivial if valid and evidence.coefficient_model == "mu2" else None,
            "repairable_to_tolerance": evidence.upper_defect <= tolerance if valid and evidence.upper_defect is not None else None,
            "separated_from_coboundaries": evidence.lower_defect >= separation_threshold if valid and evidence.lower_defect is not None else None,
            "a": tolerance, "b": separation_threshold,
            "structural_status": "inapplicable" if evidence.coefficient_model == "inapplicable" else ("gates_passed" if valid else "uncertified")}


def gate_controls():
    """Decision fixtures only, with an explicit populated K3 closure control."""
    from .finite_certificates import full_simplex
    import numpy as np
    tetrahedron = full_simplex(4)
    input_cochain = np.array([1, 0, 0, 0])
    closure = tetrahedron.delta(2) @ input_cochain
    candidates = [Candidate("compact_qr_svd", 0.2), Candidate("structural_candidate", 0.1, True)]
    rows = []
    controls = {"independently_fitted_reverse": "inverse_consistency", "wrong_coefficient": "coefficient_identification",
                "projection_trap": "projection_fidelity", "noncentral_products": "centrality",
                "filled_tetrahedron_input_cochain_error": "closure", "uncertain_estimate": "uncertainty_margin",
                "incoherent_context": "topology"}
    for control, failing in controls.items():
        evidence = Evidence("vector", tuple((gate, gate != failing) for gate in GATES), 0.0, 0.0, None, 0.0, 0.0)
        for removed in ((), (failing,), GATES):
            for policy in ("P3G", "P4"):
                result = decide(policy, candidates, "compact_qr_svd", evidence, removed_gates=removed)
                result.update(control_id=control, removed_gates=",".join(removed) or "none", phase="controlled_algebraic",
                              construction_type="input_cochain_error" if failing == "closure" else "decision_rule_input",
                              populated_K3=failing == "closure", task_metric_measured=False,
                              fixture_validation_losses=True,
                              raw_closure_residual=closure.tolist() if failing == "closure" else None)
                rows.append(result)
    return rows
