from experiments.nontrivial_refinement_invariance import OPERATIONS, run


def test_exact_nontrivial_classes_survive_all_refinements():
    rows, maps, combined = run()
    summary = [row for row in combined if "construction" in row]
    assert len(rows) == 3 * len(OPERATIONS)
    assert len(maps) == len(rows)
    assert all(row["closure_error"] == 0 for row in rows)
    assert all(row["nontrivial"] for row in rows)
    assert all(row["section_left_inverse_verified"] for row in rows)
    assert all(row["prediction_equivalence_error"] < 1e-12 for row in rows)
    assert all(row["gate_passed"] for row in summary)
