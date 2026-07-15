from experiments.strict_zeroshot_chart_generalization import SEEN, UNSEEN, allowed_compositions


def test_heldout_charts_are_strictly_disjoint():
    assert set(SEEN).isdisjoint(UNSEEN)
    assert set(SEEN) | set(UNSEEN) == set(range(8))


def test_expanded_consistency_uses_only_seen_products():
    pairs = allowed_compositions(SEEN, require_product_seen=True)
    assert pairs
    assert all(left in SEEN and right in SEEN and product in SEEN for left, right, product in pairs)
