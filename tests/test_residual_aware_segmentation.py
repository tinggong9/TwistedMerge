from experiments.residual_aware_segmentation import run


def test_gate_closed_without_positive_d1_certificate(monkeypatch):
    monkeypatch.setattr("experiments.residual_aware_segmentation._certificate", lambda: False)
    assert run(smoke=True)["state"] == "gate_closed"
