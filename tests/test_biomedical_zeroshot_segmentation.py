from experiments.biomedical_zeroshot_segmentation import SEEN, UNSEEN


def test_seen_and_unseen_chart_roles_are_disjoint_and_complete():
    assert set(SEEN).isdisjoint(UNSEEN)
    assert set(SEEN) | set(UNSEEN) == set(range(8))


def test_heldout_charts_are_not_reflections_of_training_role_names():
    assert 2 in UNSEEN and 3 in UNSEEN and 5 in UNSEEN
