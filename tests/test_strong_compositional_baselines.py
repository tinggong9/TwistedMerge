import itertools

from experiments.strong_compositional_baselines import build_group, reduce_word


def test_group_tables_are_associative_and_words_reduce():
    for name, expected_order in [("S3", 6), ("D4", 8), ("Q8", 8), ("A4", 12)]:
        group = build_group(name)
        assert group.order == expected_order
        assert all(group.multiplication[group.multiplication[a, b], c] == group.multiplication[a, group.multiplication[b, c]] for a, b, c in itertools.product(range(group.order), repeat=3))
        assert reduce_word(group, ()) == group.identity
