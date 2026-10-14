from __future__ import annotations

import json

import numpy as np
import pytest

from src.kbs_revision.finite_certificates import (
    CertificateVerificationError,
    SimplicialComplex,
    build_torus,
    certificate_decision,
    face_cochain,
    full_simplex,
    gate_ablation_controls,
    gf2_rank,
    gf2_solve,
    tetrahedral_boundary,
    vector_hodge,
    verify_all,
    verify_simply_connected_no_go,
    verify_torus,
)


@pytest.mark.parametrize("size", [3, 4])
def test_torus_is_edge_realizable_and_exactly_nontrivial(size):
    result = verify_torus(size)
    assert len(result["construction"]["native_edges"]) == 3 * size**2
    assert result["negative_faces"] == 1
    assert result["rank_delta1_f2"] == 2 * size**2 - 1
    assert result["rank_augmented_f2"] == 2 * size**2
    assert result["betti_f2"] == [1, 2, 1]
    assert result["exact_trivial"] is False
    assert all(row["cochain_law_verified"] and row["gauge_invariant"] for row in result["perturbations"])
    complex_, edges, _ = build_torus(size)
    # Independently multiply exactly the data returned by the builder.
    negative = 0
    for i, j, k in complex_.level(2):
        residual = edges[(k, i)] @ edges[(j, k)] @ edges[(i, j)]
        assert np.array_equal(residual, np.eye(2)) or np.array_equal(residual, -np.eye(2))
        negative += np.array_equal(residual, -np.eye(2))
    assert negative == 1


def test_identity_vertical_seam_control_is_coboundary():
    result = verify_torus(3, trivial_control=True)
    assert result["negative_faces"] == 0
    assert result["exact_trivial"] is True
    assert result["rank_augmented_f2"] == result["rank_delta1_f2"]
    witness = np.array(result["coboundary_witness"])
    delta1 = np.array(result["construction"]["signed_coboundaries"][1])
    assert not np.any(delta1 @ witness % 2)


def test_noncentral_stored_edge_change_is_rejected():
    complex_, edges, _ = build_torus(3)
    edge = complex_.level(1)[0]
    edges[edge] = np.array([[1, 1], [0, 1]]) @ edges[edge]
    with pytest.raises(CertificateVerificationError, match="mu_2"):
        face_cochain(complex_, edges)


def test_exact_binary_solver_handles_rank_deficiency_and_inconsistency():
    lhs = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]])
    assert gf2_rank(lhs) == 1
    witness = gf2_solve(lhs, np.array([1, 1, 0]))
    assert np.array_equal(lhs @ witness % 2, [1, 1, 0])
    assert gf2_solve(lhs, np.array([1, 0, 0])) is None
    assert gf2_rank(np.empty((0, 5), dtype=int)) == 0
    with pytest.raises(ValueError, match="exact integers"):
        gf2_rank(np.eye(2))


def test_simply_connected_certificate_uses_construction_and_explicit_witnesses():
    results = verify_simply_connected_no_go()
    boundary = results[0]
    assert boundary["betti_f2"] == [1, 0, 1]  # H2 need not vanish in the no-go.
    oracle = boundary["oracle_face_assignment_control"]
    assert oracle["closed_exact_f2"] and not oracle["exact_trivial"]
    assert oracle["rank_delta1_f2"] == 3 and oracle["rank_augmented_f2"] == 4
    assert oracle["edge_realizability_under_stated_model"] == "ruled_out_by_simply_connected_no_go"
    for row in results:
        assert row["certificate_basis"] == "construction"
        assert row["neural_merge_safety_claim"] is False
        assert row["decision"] == "nontrivial_central_class_ruled_out_under_model"
        delta1 = np.array(row["construction"]["signed_coboundaries"][1])
        for trial in row["trials"]:
            assert np.array_equal(delta1 @ np.array(trial["coboundary_witness"]) % 2, trial["face_cochain"])
            assert all(trial["checks"].values())


def test_tetrahedral_degree_two_arbitrary_hodge_includes_degree_three_coexact_energy():
    complex_ = full_simplex(4)
    # A degree-two cochain that fails closure; its coexact component is essential.
    values = np.array([1., 0., 0., 0.])
    result = vector_hodge(complex_, 2, values)
    assert result["closure_norm"] == 1
    assert result["coexact_energy"] > 0.2
    assert result["distance_squared"] == pytest.approx(result["harmonic_energy"] + result["coexact_energy"])
    assert result["distance_squared"] != pytest.approx(result["harmonic_energy"])


def test_tetrahedral_boundary_closed_hodge_has_nonzero_harmonic_component():
    complex_ = tetrahedral_boundary()
    values = np.array([-1., 1., -1., 1.])
    result = vector_hodge(complex_, 2, values, {2: [1., 2., 3., 4.]})
    assert result["closure_norm"] == 0
    assert result["harmonic_energy"] > 0
    assert result["distance_squared"] == pytest.approx(result["harmonic_energy"])


def test_hodge_degree_three_and_weighted_vector_coefficients():
    complex_ = full_simplex(5)
    values = np.arange(15, dtype=float).reshape(5, 3)
    result = vector_hodge(complex_, 3, values, {3: [1., 2., 3., 4., 5.]})
    assert result["coexact_energy"] > 0
    assert result["distance_squared"] == pytest.approx(result["harmonic_energy"] + result["coexact_energy"])
    assert abs(result["gradient_coexact_inner"]) < 1e-10
    with pytest.raises(ValueError, match="positive"):
        vector_hodge(complex_, 3, values, {3: [1., 2., -3., 4., 5.]})


def test_exact_and_approximate_decisions_are_separate():
    complex_, edges, _ = build_torus()
    c, _ = face_cochain(complex_, edges)
    d1, d2 = complex_.delta(1), complex_.delta(2)
    exact = certificate_decision(d1, d2, c, exact_model_verified=True)
    assert exact["exact_trivial"] is False
    approximate = certificate_decision(d1, d2, c, repair_upper_bound=0.01, metric_bound_prerequisites=True)
    assert approximate["repairable_to_tolerance"] is True
    assert approximate["exact_trivial"] is None
    unqualified = certificate_decision(d1, d2, c, repair_upper_bound=0.01)
    assert unqualified["uncertified"] and unqualified["repairable_to_tolerance"] is None
    failed = certificate_decision(d1, d2, c, exact_model_verified=True, confidence_prerequisites=False)
    assert failed["uncertified"] and failed["exact_trivial"] is None
    assert certificate_decision(d1, d2, c, coefficient_theory="mu4")["inapplicable"]
    assert certificate_decision(d1, d2, c, common_coordinate=False)["inapplicable"]


def test_rounding_requires_latent_model_and_strict_half_separation_margin():
    complex_, edges, _ = build_torus()
    c, _ = face_cochain(complex_, edges)
    d1, d2 = complex_.delta(1), complex_.delta(2)
    uncertified = certificate_decision(d1, d2, c, rounded=True, error_radius=0.01, coefficient_separation=2.)
    assert uncertified["exact_rounded_trivial"] is False
    assert uncertified["exact_trivial"] is None and uncertified["uncertified"]
    recovered = certificate_decision(d1, d2, c, rounded=True, exact_model_verified=True, error_radius=0.99, coefficient_separation=2.)
    assert recovered["exact_trivial"] is False and not recovered["uncertified"]
    boundary = certificate_decision(d1, d2, c, rounded=True, exact_model_verified=True, error_radius=1., coefficient_separation=2.)
    assert boundary["exact_trivial"] is None and boundary["uncertified"]


def test_unclosed_oracle_tetrahedral_assignment_is_uncertified():
    complex_ = full_simplex(4)
    c = np.array([1, 0, 0, 0])
    result = certificate_decision(complex_.delta(1), complex_.delta(2), c, exact_model_verified=True)
    assert not result["closed_exact_f2"]
    assert result["uncertified"] and result["exact_trivial"] is None


def test_complex_rejects_missing_faces_and_verifier_is_json_serializable():
    with pytest.raises(ValueError, match="downward closed"):
        SimplicialComplex((((0,), (1,), (2,)), ((0, 1),), ((0, 1, 2),)))
    with pytest.raises(ValueError, match="size"):
        build_torus(2)
    result = verify_all()
    assert result["passed"]
    assert len(result["torus"]) == 4
    assert all(row["passed"] for row in result["hodge"])
    assert any(row["degree"] == 3 for row in result["hodge"])
    assert json.loads(json.dumps(result, allow_nan=False))["status"] == "controlled_algebraic"


def test_gate_controls_measure_raw_prerequisites_and_keep_edge_scope_explicit():
    rows = {row["control_id"]: row for row in gate_ablation_controls()}
    assert len(rows) == 7
    inverse = rows["independently_fitted_reverse"]
    assert inverse["inverse_residual"] > 0
    assert inverse["raw_centrality_residual"] == 0
    assert inverse["exact_projected_cochain_trivial"] is True
    assert inverse["exact_underlying_transition_class_trivial"] is None
    wrong = rows["wrong_coefficient_mu3"]
    assert len(wrong["mu3_power_identity_failures"]) == 1
    assert wrong["inapplicable"]
    assert wrong["exact_mu2_ground_truth_trivial"] is False
    noncentral = rows["noncentral_face_products"]
    assert noncentral["inverse_residual"] == 0
    assert noncentral["raw_centrality_residual"] > 0
    trap = rows["projection_trap"]
    assert trap["projected_centrality_residual"] == 0
    assert trap["raw_centrality_residual"] > 0
    assert trap["projection_fidelity_residual"] > 0
    assert trap["exact_underlying_transition_class_trivial"] is None
    assert rows["incoherent_coordinate_context"]["inapplicable"]
    radius = rows["uncertified_radius"]
    assert radius["uncertified"]
    assert radius["certified_error_radius"] == radius["half_separation_margin"]
    closure = rows["filled_tetrahedron_input_cochain_error"]
    assert closure["populated_K3"] and not closure["edge_realizability_asserted"]
    assert closure["closure_f2"] == [1]
    assert closure["hodge_real_harmonic_energy"] < 1e-20
    assert closure["hodge_real_coexact_energy"] == pytest.approx(0.25)
    assert all(not row["task_metric_measured"] for row in rows.values())
    assert all(not row["all_prerequisites_pass"] for row in rows.values())
    assert json.loads(json.dumps(list(rows.values()), allow_nan=False))
