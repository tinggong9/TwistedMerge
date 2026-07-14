from __future__ import annotations


def test_practical_score_penalties_are_monotone() -> None:
    accuracy, regret, overhead = 0.8, 0.1, 2.0
    low_penalty = accuracy - 0.1 * regret - 0.01 * overhead
    high_penalty = accuracy - 0.25 * regret - 0.05 * overhead
    assert high_penalty < low_penalty
