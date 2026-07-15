from experiments.end_to_end_controlled_cost import run_setting, summarize


def test_controlled_cost_uses_actual_models_and_label_independent_logits():
    accuracy, costs = run_setting("S3", noise=0.2, context_budget=16, seed=40, repeats=2)
    assert len(accuracy) == 7
    assert len(costs) == 7 * 4
    assert all(row["label_permutation_hash_passed"] for row in accuracy)
    assert all(row["measurement_type"] == "end_to_end_torch_cpu" for row in costs)
    assert all(row["timed_repetitions"] == 2 for row in costs)
    summary, paired, claims = summarize(accuracy, costs)
    assert len(summary) == 7
    assert len(paired) == 6
    assert any(row["claim"] == "all_costs_end_to_end_measured" and row["value"] for row in claims)
