from experiments import biomedical_landmark_retransport as stage


def test_landmark_stage_does_not_derive_annotations(monkeypatch):
    monkeypatch.setattr(stage, "sanity_passed", lambda: True)
    assert stage.run(smoke=True)["state"] == "blocked"
