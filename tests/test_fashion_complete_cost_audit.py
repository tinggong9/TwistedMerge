from experiments.fashion_complete_cost_audit import BATCH_SIZES, METHOD_MAP, pareto_rows


def test_cost_audit_has_preregistered_batches_and_methods():
    assert BATCH_SIZES == (1, 8, 32, 128)
    assert len(METHOD_MAP) == 12
    assert "canonicalize_pool_retransport" in METHOD_MAP


def test_pareto_frontier_uses_accuracy_and_cost():
    rows = [
        {"method": "a", "task_accuracy": 0.8, "cost": 10},
        {"method": "b", "task_accuracy": 0.9, "cost": 9},
        {"method": "c", "task_accuracy": 0.95, "cost": 12},
    ]
    result = {row["method"]: row["pareto_optimal"] for row in pareto_rows(rows, "cost")}
    assert result == {"a": False, "b": True, "c": True}
