import numpy as np

from experiments.broader_vision_extended import merge_heads, specialist_head, task_classes


def test_extended_vision_tasks_overlap_and_cover_every_class():
    tasks = task_classes(10, 0)
    assert len(tasks) == 4
    assert set(np.concatenate(tasks)) == set(range(10))
    assert sum(3 in task for task in tasks) >= 2


def test_specialist_and_merge_shapes_are_stable():
    rng = np.random.default_rng(4)
    features = rng.normal(size=(40, 8))
    labels = np.arange(40) % 4
    head = specialist_head(features, labels, np.asarray([0, 1, 2]), 4)
    assert head.shape == (8, 4)
    for mode in ["mean", "regmean", "ties", "dare", "low_rank"]:
        assert merge_heads([head, head, head, head], mode, 0).shape == head.shape
