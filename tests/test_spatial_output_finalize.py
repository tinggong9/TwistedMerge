from experiments.spatial_output_finalize import paired_positive


def test_paired_gate_requires_every_named_positive_interval():
    rows = [
        {"comparison": "a", "ci_lower": "0.1"},
        {"comparison": "b", "ci_lower": "-0.1"},
    ]
    assert paired_positive(rows, ("a",))
    assert not paired_positive(rows, ("a", "b"))
    assert not paired_positive(rows, ("missing",))
