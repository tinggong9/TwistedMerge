from src.finite_group_cohomology import (
    center,
    compute_h2_cyclic_coefficients,
    cyclic_group,
    dihedral_group_4,
    element_order_histogram,
    klein_four_group,
    symmetric_group_3,
)
from src.projective_representation_index import classify_group_cohomology_rank


def test_standard_small_group_orders_and_centers():
    assert cyclic_group(2).order == 2
    assert cyclic_group(3).order == 3
    assert cyclic_group(4).order == 4
    assert klein_four_group().order == 4
    assert symmetric_group_3().order == 6
    assert dihedral_group_4().order == 8
    assert len(center(klein_four_group())) == 4
    assert len(center(symmetric_group_3())) == 1
    assert element_order_histogram(symmetric_group_3()) == {1: 1, 2: 3, 3: 2}


def test_cyclic_h2_mod_two_matches_basic_group_cohomology():
    c2 = compute_h2_cyclic_coefficients(cyclic_group(2), coefficient_modulus=2)
    c3 = compute_h2_cyclic_coefficients(cyclic_group(3), coefficient_modulus=2)
    c4 = compute_h2_cyclic_coefficients(cyclic_group(4), coefficient_modulus=2)
    assert c2.exact and c2.h2_size == 2 and c2.h2_dimension == 1
    assert c3.exact and c3.h2_size == 1 and c3.h2_dimension == 0
    assert c4.exact and c4.h2_size == 2 and c4.h2_dimension == 1


def test_noncyclic_h2_mod_two_is_exact_for_small_reference_groups():
    v4 = compute_h2_cyclic_coefficients(klein_four_group(), coefficient_modulus=2)
    s3 = compute_h2_cyclic_coefficients(symmetric_group_3(), coefficient_modulus=2)
    d4 = compute_h2_cyclic_coefficients(dihedral_group_4(), coefficient_modulus=2)
    assert v4.exact and v4.h2_size is not None and v4.h2_size >= 2
    assert s3.exact and s3.h2_size is not None and s3.h2_size >= 1
    assert d4.exact and d4.h2_size is not None and d4.h2_size >= 1


def test_period_index_rank_gate_is_conservative():
    assert classify_group_cohomology_rank("no_certified_class", None, None, "none", 8) == "no_certified_class"
    assert classify_group_cohomology_rank("coboundary", 1, 1, "coboundary", 1) == "coboundary_no_lift_needed"
    assert classify_group_cohomology_rank("nontrivial_H2_class", 2, None, "index_unknown_no_lift", 4) == "index_unknown_no_lift"
    assert classify_group_cohomology_rank("nontrivial_H2_class", 2, 4, "certified", 2) == "period_divisible_index_obstructed"
    assert classify_group_cohomology_rank("nontrivial_H2_class", 2, 4, "certified", 4) == "index_divisible_lift_allowed"
