import numpy as np

from experiments.emergency_level2_confirmation import GENERIC, variants


def test_confirmation_has_all_required_controls():
    candidates, counts, setting = variants("S3", 20, 0.2, 16)
    required = {
        "twistedmerge_hodge_lr",
        "generic_mixture_of_experts",
        "learned_unconstrained_matrix_context_action",
        "generic_low_rank_context_adapter",
        "group_structured_without_hodge",
        "hodge_lr_generic_retransport",
        "random_multiplication_table_control",
        "shuffled_context_control",
    }
    assert required == set(candidates)
    assert all(values.shape == setting["teacher_test"].shape for values in candidates.values())
    assert all(method in candidates for method in GENERIC)
    assert all(counts[method] >= 0 for method in candidates)


def test_random_law_control_differs_from_structured_prediction():
    candidates, _, _ = variants("D4", 21, 0.5, 64)
    assert not np.array_equal(candidates["random_multiplication_table_control"], candidates["twistedmerge_hodge_lr"])
