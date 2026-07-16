from experiments import second_biomedical_segmentation as stage


def test_second_dataset_stage_closes_when_primary_gate_fails(monkeypatch):
    monkeypatch.setattr(stage, "b1_gate", lambda: False)
    assert stage.run(smoke=True)["state"] == "gate_closed"
