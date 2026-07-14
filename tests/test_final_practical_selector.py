from __future__ import annotations

import pandas as pd

from experiments.final_practical_selector import METHOD_RENAME, paired_statistics, prepare_runs, summarize


def _source_rows() -> pd.DataFrame:
    rows = []
    for setting, seed in (("a", 1), ("b", 2)):
        for source_method, accuracy in (("greedy_soup", 0.7), ("improved_validated_selector", 0.8)):
            rows.append(
                {
                    "setting_id": setting,
                    "dataset": "mnist",
                    "n_models": 3,
                    "width": 16,
                    "seed": seed,
                    "method": source_method,
                    "accuracy": accuracy,
                    "loss": 1 - accuracy,
                    "val_accuracy": accuracy,
                    "val_loss": 1 - accuracy,
                    "selector_chose": "greedy_soup",
                    "selector_val_margin": 0.1,
                    "selector_validation_budget": 100,
                    "union_candidate_count": 1,
                    "label_permutation_regression_passed": True,
                    "measured_inference_time_seconds_512": 0.01,
                    "inference_multiplier": 1.0,
                    "actual_trainable_parameters": 10,
                    "stored_parameters": 10,
                    "parameter_multiplier": 1,
                    "branch_count": 1,
                }
            )
    return pd.DataFrame(rows)


def test_method_map_covers_required_selector_and_control() -> None:
    assert METHOD_RENAME["improved_validated_selector"] == "twistedmerge_exact_gauge_soup_selector"
    assert METHOD_RENAME["randomly_augmented_candidate_union"] == "randomly_augmented_candidate_union"


def test_summary_and_paired_statistics_use_matched_settings(monkeypatch) -> None:
    monkeypatch.setattr("experiments.final_practical_selector.git_output", lambda *args: "a" * 40 if "rev-parse" in args else "")
    runs = prepare_runs(_source_rows(), 1.0, 2.0)
    summary = summarize(runs)
    paired = paired_statistics(runs)
    selector = paired[paired["method"] == "twistedmerge_exact_gauge_soup_selector"].iloc[0]
    assert selector["n_pairs"] == 2
    assert abs(selector["mean_accuracy_delta"] - 0.1) < 1e-12
    assert set(summary["scope"]) == {"overall", "fixed_setting"}
    assert not runs["central_lift_activated"].any()
    assert not runs["nonabelian_lift_activated"].any()
