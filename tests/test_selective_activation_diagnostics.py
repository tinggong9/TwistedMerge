from experiments.selective_activation_diagnostics import FEATURES, build_dataset, leave_out_evaluation


def test_selective_activation_uses_leave_family_out_and_abstention():
    rows = build_dataset()
    assert len(rows) == 120
    assert all(all(feature in row for feature in FEATURES) for row in rows)
    evaluation = leave_out_evaluation(rows, "family")
    assert len(evaluation) == 6
    assert all(0 <= row["coverage"] <= 1 for row in evaluation)
