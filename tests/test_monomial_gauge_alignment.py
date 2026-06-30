import numpy as np

from src.ladder_merge_methods import transform_mlp_positive_scale
from src.model_merging_benchmark import DatasetSpec, make_loader, make_model, require_torch, set_seed
from src.monomial_gauge_alignment import (
    MonomialAlignment,
    apply_monomial_alignment_to_reference,
    compare_function_before_after_alignment,
    estimate_monomial_alignment,
    monomial_cycle_defect,
    monomial_defect_score,
)


def _toy_loader(n_examples: int = 64):
    torch, _, _ = require_torch()
    x = torch.randn(n_examples, 1, 4, 4)
    y = torch.zeros(n_examples, dtype=torch.long)
    return make_loader(torch.utils.data.TensorDataset(x, y), batch_size=16, shuffle=False, seed=11)


def test_known_monomial_gauge_preserves_relu_mlp_function():
    set_seed(123)
    spec = DatasetSpec("toy", (1, 4, 4), 10)
    width = 5
    model = make_model("mlp", spec, width)
    perm = np.array([2, 4, 1, 3, 0])
    scales = np.array([0.5, 1.7, 0.9, 2.3, 1.2])
    gauged = transform_mlp_positive_scale(model, spec, width, perm, scales)

    inverse_perm = np.empty_like(perm)
    inverse_perm[perm] = np.arange(width)
    inverse_scales = 1.0 / scales[inverse_perm]
    alignment = MonomialAlignment(inverse_perm, inverse_scales, "monomial_weight", "known_inverse")
    aligned = apply_monomial_alignment_to_reference(gauged, spec, width, alignment)

    metrics = compare_function_before_after_alignment(gauged, aligned, _toy_loader(), "cpu", max_batches=4)
    assert metrics["functional_preservation_error"] < 1e-6
    assert metrics["functional_preservation_prediction_disagreement"] == 0.0


def test_estimated_monomial_alignment_is_valid_and_function_preserving():
    set_seed(456)
    spec = DatasetSpec("toy", (1, 4, 4), 10)
    width = 6
    reference = make_model("mlp", spec, width)
    target = make_model("mlp", spec, width)
    loader = _toy_loader()

    for matching in ("monomial_activation", "monomial_weight"):
        alignment = estimate_monomial_alignment(reference, target, loader, "cpu", matching=matching, max_batches=4)
        assert sorted(alignment.permutation.tolist()) == list(range(width))
        assert alignment.positive_scales.shape == (width,)
        assert np.all(alignment.positive_scales > 0.0)
        aligned = apply_monomial_alignment_to_reference(target, spec, width, alignment)
        metrics = compare_function_before_after_alignment(target, aligned, loader, "cpu", max_batches=4)
        assert metrics["functional_preservation_error"] < 1e-6


def test_monomial_cycle_defect_detects_scale_inconsistency():
    width = 4
    gauges = [
        (np.array([0, 1, 2, 3]), np.array([1.0, 1.2, 0.7, 2.0])),
        (np.array([2, 0, 3, 1]), np.array([0.8, 1.5, 1.1, 0.9])),
        (np.array([1, 3, 0, 2]), np.array([1.3, 0.6, 1.7, 1.0])),
    ]

    alignments = {}
    for i, (q_i, a_i) in enumerate(gauges):
        for j, (q_j, a_j) in enumerate(gauges):
            perm = np.empty(width, dtype=int)
            scales = np.empty(width, dtype=float)
            for global_unit in range(width):
                source_unit = q_i[global_unit]
                target_unit = q_j[global_unit]
                perm[source_unit] = target_unit
                scales[source_unit] = a_j[global_unit] / a_i[global_unit]
            alignments[(i, j)] = MonomialAlignment(perm, scales, "monomial_weight", "planted")

    assert monomial_defect_score(monomial_cycle_defect(alignments, 0, 1, 2)) < 1e-10

    broken = dict(alignments)
    broken_alignment = alignments[(0, 1)]
    broken_scales = broken_alignment.positive_scales.copy()
    broken_scales[0] *= 1.4
    broken[(0, 1)] = MonomialAlignment(
        broken_alignment.permutation,
        broken_scales,
        "monomial_weight",
        "corrupted",
    )
    assert monomial_defect_score(monomial_cycle_defect(broken, 0, 1, 2)) > 0.05
