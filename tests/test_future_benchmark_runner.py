from pathlib import Path


def test_future_runner_exposes_required_control_states():
    text = Path("experiments/run_all_future_benchmarks.py").read_text()
    for token in ["--tier", "--resume", "--force-stage", "clean-freeze", "blocked", "confirmation"]:
        assert token in text


def test_future_runner_has_all_scientific_tiers():
    text = Path("experiments/run_all_future_benchmarks.py").read_text()
    assert '"emergency"' in text
    assert '"near-term"' in text
    assert '"extended"' in text


def test_future_runner_exposes_each_extended_stage():
    from experiments.run_all_future_benchmarks import STAGES

    identifiers = {stage.stage_id for stage in STAGES}
    assert {f"X{index}" for index in range(1, 13)} <= identifiers


def test_attempt_history_is_append_only(tmp_path, monkeypatch):
    import experiments.run_all_future_benchmarks as runner

    monkeypatch.setattr(runner, "OUT", tmp_path)
    item = {"stage_id": "E0", "state": "completed", "summary": "ok"}
    runner.append_attempt("run", item)
    runner.append_attempt("run", item)
    assert len((tmp_path / "attempt_history.csv").read_text().splitlines()) == 3
    assert b"\r\n" not in (tmp_path / "attempt_history.csv").read_bytes()
