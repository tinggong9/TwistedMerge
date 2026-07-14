from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_tracked_source_contains_complete_primary_grid():
    df = pd.read_csv(ROOT / "reports/csv/improved_validated_ladder_merge_benchmark.csv")
    observed = set(map(tuple, df[["n_models", "width", "seed"]].drop_duplicates().to_numpy()))
    expected = {(n, width, seed) for n in (3, 4) for width in (16, 32, 64) for seed in range(1800, 1820)}
    assert observed == expected


def test_source_selectors_are_marked_no_test_leakage():
    df = pd.read_csv(ROOT / "reports/csv/improved_validated_ladder_merge_benchmark.csv")
    assert df.selector_no_test_leakage.fillna(False).astype(bool).all()


def test_source_generator_has_no_target_accuracy_helpers():
    text = (ROOT / "experiments/improved_validated_ladder_merge_benchmark.py").read_text(encoding="utf-8")
    assert "target_accuracy_for_method" not in text
    assert "logits_with_target_accuracy" not in text
