from __future__ import annotations

import copy

import pandas as pd
import pytest

from experiments.post_iclr_selector_attribution import (
    bootstrap_group_ci,
    budget_match_names,
    diagnostic_choice,
    oracle_choice,
    validation_choice,
)


def test_deployable_selector_does_not_read_test_metrics() -> None:
    validation = {
        "ordinary": {"accuracy": 0.8, "loss": 0.5},
        "gauge": {"accuracy": 0.79, "loss": 0.4},
    }
    test = {
        "ordinary": {"accuracy": 0.1, "loss": 4.0},
        "gauge": {"accuracy": 0.99, "loss": 0.01},
    }
    choice = validation_choice(["ordinary", "gauge"], validation)
    changed_test = copy.deepcopy(test)
    changed_test["ordinary"]["accuracy"] = 1.0
    assert choice == "ordinary"
    assert validation_choice(["ordinary", "gauge"], validation) == choice
    assert oracle_choice(["ordinary", "gauge"], changed_test) == "ordinary"


def test_budget_match_is_exact_and_rejects_underfilled_pool() -> None:
    assert budget_match_names(["a", "b", "c", "c"], 2) == ["a", "b"]
    with pytest.raises(ValueError, match="needs 4"):
        budget_match_names(["a", "b"], 4)


def test_diagnostic_rule_changes_pool_not_scores() -> None:
    metrics = {
        "a0": {"accuracy": 0.8, "loss": 0.4},
        "official": {"accuracy": 0.81, "loss": 0.5},
    }
    assert diagnostic_choice(["a0"], ["a0", "official"], metrics, 0.2, 0.1) == "a0"
    assert diagnostic_choice(["a0"], ["a0", "official"], metrics, 0.05, 0.1) == "official"


def test_group_bootstrap_uses_group_means() -> None:
    frame = pd.DataFrame(
        {
            "seed": [1, 1, 2, 2],
            "delta": [0.0, 0.2, 0.2, 0.4],
        }
    )
    low, high = bootstrap_group_ci(frame, "delta", n_bootstrap=500, seed=7)
    assert low <= 0.2 <= high
    assert low >= 0.1 - 1e-12
