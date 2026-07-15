from experiments.emergency_mechanism_ablation import run_setting


def test_mechanism_ladder_contains_thirteen_executed_methods():
    rows = run_setting("S3", 0, 0.2, 16)
    assert len(rows) == 13
    assert len({row["method"] for row in rows}) == 13
    assert all(row["leakage_hash_passed"] for row in rows)
    assert all(0 <= row["accuracy"] <= 1 for row in rows)
