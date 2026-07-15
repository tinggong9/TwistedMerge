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
