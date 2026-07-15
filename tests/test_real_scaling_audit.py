from experiments.real_scaling_audit import configurations, run


def test_scaling_executes_all_requested_axes():
    rows = run(repeats=2)
    assert {row["axis"] for row in rows} == {"models", "graph_edges", "faces", "hidden_width", "group_order", "residual_rank", "branch_count", "calibration_samples"}
    assert all(row["timed_repetitions"] == 2 for row in rows)
    assert all(row["measurement_type"].startswith("executed_") for row in rows)
