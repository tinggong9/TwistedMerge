from src.finite_group_cohomology import symmetric_group_3
from src.nonabelian_holonomy import (
    abelianization_size,
    commutator_subgroup,
    conjugacy_classes,
    group_exponent,
    group_orbits,
    infer_holonomy_group,
    is_abelian,
)
from src.nonabelian_lift_candidates import build_lift_candidate_rows
from src.nonabelian_representation_index import representation_candidates, splitting_score


def test_s3_nonabelian_invariants_are_computed():
    group = symmetric_group_3()
    assert group.order == 6
    assert is_abelian(group) is False
    assert group_exponent(group) == 6
    assert len(conjugacy_classes(group)) == 3
    assert commutator_subgroup(group).order == 3
    assert abelianization_size(group) == 2


def test_orbit_representation_reduces_quotient_holonomy_diagnostic():
    summary = infer_holonomy_group(
        edge_transports=[(1, 0, 2), (1, 2, 0)],
        triangle_holonomies=[(1, 0, 2)],
        max_group_order=16,
        max_exact_order=16,
    )
    assert summary.group.order == 6
    assert summary.noncentral_holonomy_score > 0.0
    assert group_orbits(summary.group) == [(0, 1, 2)]
    reps = representation_candidates(summary.group, max_exact_representation_order=16)
    orbit = [rep for rep in reps if rep.representation_name == "orbit_representation"][0]
    score = splitting_score(orbit, [(1, 0, 2)], reduction_threshold=0.1)
    assert score["split_success_flag"]
    assert score["relative_holonomy_reduction"] > 0.99


def test_lift_candidates_are_diagnostic_when_no_model_lift_exists():
    summary = infer_holonomy_group(
        edge_transports=[(1, 0, 2), (1, 2, 0)],
        triangle_holonomies=[(1, 0, 2)],
        max_group_order=16,
        max_exact_order=16,
    )
    rows = []
    for rep in representation_candidates(summary.group, max_exact_representation_order=16):
        row = {
            "run_id": "r0",
            "width": 3,
            **rep.__dict__,
            **splitting_score(rep, [(1, 0, 2)], reduction_threshold=0.1),
        }
        rows.append(row)
    import pandas as pd

    lifts = build_lift_candidate_rows(pd.DataFrame(rows))
    assert "lift_implemented" in lifts.columns
    assert not lifts["lift_implemented"].any()
    assert "random_same_rank_lift_control" in set(lifts["candidate_method"])
