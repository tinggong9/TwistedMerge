from __future__ import annotations

import numpy as np

from experiments.context_dependent_two_loop_holonomy import (
    apply_actions,
    compose,
    evaluate_setting,
    generated_group,
    smoke_gates,
)


def test_regular_actions_are_noncommuting_homomorphisms() -> None:
    for name, order in (("S3", 6), ("D4", 8)):
        group = generated_group(name)
        assert len(group.elements) == order
        assert not np.allclose(group.regular[group.s] @ group.regular[group.r], group.regular[group.r] @ group.regular[group.s])
        for left in group.elements:
            for right in group.elements:
                assert np.allclose(group.regular[compose(left, right)], group.regular[left] @ group.regular[right])


def test_context_action_changes_executed_logits() -> None:
    group = generated_group("S3")
    base = np.arange(12, dtype=float).reshape(2, 6)
    transformed = apply_actions(base, group, [group.s, group.r])
    assert transformed.shape == base.shape
    assert not np.allclose(transformed, base)


def test_small_setting_passes_leakage_and_structural_gates(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("experiments.context_dependent_two_loop_holonomy.OUT", tmp_path)
    rows, residual = evaluate_setting("S3", 12, 0, 128, 128, 256)
    import pandas as pd

    runs = pd.DataFrame(rows)
    residuals = pd.DataFrame([residual])
    gates = smoke_gates(runs, residuals)
    assert all(gates.values())
    assert runs["label_permutation_regression_passed"].all()
    assert runs.loc[runs.method == "supplied_context_oracle", "accuracy"].iloc[0] == 1.0
