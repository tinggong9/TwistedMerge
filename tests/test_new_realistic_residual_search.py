from experiments.new_realistic_residual_search import selection_metrics


def test_selection_is_validation_independent_and_bounded_to_two_families():
    rows = []
    for family in ("a", "b", "c"):
        for collection in range(5):
            rows.append({"family": family, "collection": collection, "heldout_pairwise_fit": 0.1, "cycle_residual": 1.0, "null_percentile": 0.99, "calibration_resamples": 5, "residual_rank_reproducible": True, "residual_rank": 2})
    summary = selection_metrics(rows)
    assert sum(row["selected_without_test_accuracy"] for row in summary) == 2
