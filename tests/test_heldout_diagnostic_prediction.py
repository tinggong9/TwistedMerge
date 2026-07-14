from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_natural_grid_has_twenty_seeds_per_complete_setting():
    df = pd.read_csv(ROOT / "reports/csv/improved_validated_ladder_merge_benchmark.csv")
    settings = df[["n_models", "width", "seed"]].drop_duplicates()
    assert (settings.groupby(["n_models", "width"]).seed.nunique() == 20).all()


def test_diagnostic_source_is_natural_mnist_not_planted_label_data():
    df = pd.read_csv(ROOT / "reports/csv/improved_validated_ladder_merge_benchmark.csv", nrows=100)
    assert set(df.dataset) == {"mnist"}
    assert set(df.architecture) == {"mlp_relu"}


def test_heldout_program_preregisters_target_and_predictor_as_constants():
    text = (ROOT / "experiments/heldout_diagnostic_prediction.py").read_text(encoding="utf-8")
    assert 'PRIMARY_TARGET = "weight_average_degradation"' in text
    assert 'PRIMARY_PREDICTOR = "cycle_residual"' in text
    assert "target_accuracy_for_method" not in text
