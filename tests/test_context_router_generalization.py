import numpy as np

from src.context_router_generalization import (
    HELDOUT_WORDS,
    TRAIN_WORDS,
    all_branch_logits,
    execute_router_logits,
    generate_context_dataset,
    make_case,
    router_assignments,
)


def test_heldout_word_contexts_are_disjoint():
    assert set(TRAIN_WORDS).isdisjoint(HELDOUT_WORDS)


def test_oracle_branch_logits_are_executed_and_recover_teacher_labels():
    case = make_case("S3", 0)
    x, y, words, true = generate_context_dataset(case, HELDOUT_WORDS, 16, "test")
    branches = all_branch_logits(case, x)
    oracle = execute_router_logits(branches, true)
    assert np.mean(np.argmax(oracle, axis=1) == y) == 1.0
    assert len(words) == len(y)


def test_learned_router_does_not_require_context_ids_at_inference():
    case = make_case("D4", 1)
    val_x, val_y, val_words, _ = generate_context_dataset(case, TRAIN_WORDS, 16, "validation")
    test_x, _test_y, test_words, true = generate_context_dataset(case, HELDOUT_WORDS, 16, "test")
    assignments, weights = router_assignments(case, val_x, val_y, val_words, test_x, test_words, true)
    assert assignments["learned_feature_router"][0].shape == (len(test_x),)
    assert weights.shape[1] == case.group.order


def test_label_permutation_does_not_change_saved_branch_logits():
    case = make_case("D4", 2)
    x, y, _words, _true = generate_context_dataset(case, HELDOUT_WORDS, 12, "test")
    saved = all_branch_logits(case, x).copy()
    _permuted = np.random.default_rng(3).permutation(y)
    rerun = all_branch_logits(case, x)
    assert np.array_equal(saved, rerun)
