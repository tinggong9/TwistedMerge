from experiments.extended_benchmark_suite import alignment_rows, group_rows, period_rows, topology_rows


def test_extended_algebra_and_topology_checks_execute():
    groups = group_rows()
    assert {"C2", "C3", "C4", "C6", "S3", "D4", "Q8", "A4", "S4"}.issubset({row["group"] for row in groups})
    assert all(row["associativity_passed"] for row in groups)
    assert len(period_rows()) >= 20
    assert all(row["gauge_invariance_error"] < 1e-8 for row in topology_rows())
    assert len(alignment_rows()) == 6
