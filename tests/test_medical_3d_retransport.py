from experiments import medical_3d_retransport as stage


def test_3d_stage_closes_without_positive_2d_gate(monkeypatch):
    monkeypatch.setattr(stage, "positive_2d_result", lambda: False)
    assert stage.run(smoke=True)["state"] == "gate_closed"
