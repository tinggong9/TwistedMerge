from experiments.spatial_output_finalize import build_claim_ladder, paired_positive


def test_paired_gate_requires_every_named_positive_interval():
    rows = [
        {"comparison": "a", "ci_lower": "0.1"},
        {"comparison": "b", "ci_lower": "-0.1"},
    ]
    assert paired_positive(rows, ("a",))
    assert not paired_positive(rows, ("a", "b"))
    assert not paired_positive(rows, ("missing",))


def test_controlled_retransport_uses_current_mask_claim_name(monkeypatch):
    def fake_read(path):
        if path.name == "mask_claims.csv":
            return [
                {"claim": "nearest_neighbor_mask_action_exact", "passed": "True"},
                {"claim": "every_negative_control_detected", "passed": "True"},
            ]
        if path.name == "output_action_runs.csv":
            return [{"passed": "True"}]
        return []

    monkeypatch.setattr("experiments.spatial_output_finalize.read_csv", fake_read)
    monkeypatch.setattr("experiments.spatial_output_finalize.claim_value", lambda *args: False)
    ladder = build_claim_ladder()
    assert ladder[0]["passed"]
    assert ladder[1]["passed"]
