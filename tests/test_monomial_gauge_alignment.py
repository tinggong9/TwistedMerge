import numpy as np

from src.ladder_merge_methods import transform_mlp_positive_scale
from src.model_merging_benchmark import DatasetSpec, make_loader, make_model, require_torch, set_seed
from src.monomial_gauge_alignment import (
    MonomialAlignment,
    apply_monomial_alignment_to_reference,
    compare_function_before_after_alignment,
    compose_monomial_alignments,
    estimate_monomial_alignment,
    estimate_pairwise_monomial_alignments,
    invert_monomial_alignment,
    monomial_cycle_defect,
    monomial_defect_score,
    monomial_scaling_statistics,
    transform_mlp2_positive_scale,
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


def test_known_mlp2_positive_monomial_gauge_preserves_function_and_predictions():
    set_seed(321)
    spec = DatasetSpec("toy", (1, 4, 4), 10)
    width = 5
    model = make_model("mlp2", spec, width)
    p1 = np.array([2, 4, 1, 3, 0])
    p2 = np.array([1, 3, 0, 4, 2])
    s1 = np.array([0.6, 1.8, 0.9, 2.4, 1.1])
    s2 = np.array([1.7, 0.7, 2.2, 0.8, 1.3])

    gauged = transform_mlp2_positive_scale(model, spec, width, p1, s1, p2, s2)
    metrics = compare_function_before_after_alignment(model, gauged, _toy_loader(), "cpu", max_batches=4)

    assert metrics["functional_preservation_error"] < 1e-6
    assert metrics["functional_preservation_prediction_disagreement"] == 0.0


def test_mlp2_monomial_alignment_application_preserves_target_function_for_all_modes():
    set_seed(654)
    spec = DatasetSpec("toy", (1, 4, 4), 10)
    width = 6
    reference = make_model("mlp2", spec, width)
    target = make_model("mlp2", spec, width)
    loader = _toy_loader()

    for matching in (
        "monomial_activation_mlp2",
        "monomial_weight_mlp2",
        "monomial_shrinkage_mlp2",
        "monomial_global_ls_mlp2",
    ):
        alignment = estimate_monomial_alignment(reference, target, loader, "cpu", matching=matching, max_batches=4)
        assert alignment.architecture == "mlp2"
        assert alignment.primary_layer == "hidden2"
        assert alignment.layers() == ("hidden1", "hidden2")
        assert all(np.all(alignment.positive_scales_for(layer) > 0.0) for layer in alignment.layers())

        aligned = apply_monomial_alignment_to_reference(target, spec, width, alignment)
        metrics = compare_function_before_after_alignment(target, aligned, loader, "cpu", max_batches=4)
        assert metrics["functional_preservation_error"] < 1e-6
        assert metrics["functional_preservation_prediction_disagreement"] == 0.0


def test_mlp2_rejects_negative_scales_unless_explicit_exact_pair_support_exists():
    set_seed(987)
    spec = DatasetSpec("toy", (1, 4, 4), 10)
    width = 4
    model = make_model("mlp2", spec, width)
    with np.testing.assert_raises(ValueError):
        transform_mlp2_positive_scale(
            model,
            spec,
            width,
            np.arange(width),
            np.array([1.0, -1.0, 1.0, 1.0]),
            np.arange(width),
            np.ones(width),
        )
    with np.testing.assert_raises(ValueError):
        MonomialAlignment(
            np.arange(width),
            np.ones(width),
            "monomial_activation_mlp2",
            "negative_control",
            architecture="mlp2",
            primary_layer="hidden2",
            layer_permutations={"hidden1": np.arange(width), "hidden2": np.arange(width)},
            layer_positive_scales={"hidden1": np.ones(width), "hidden2": np.array([1.0, 1.0, -0.5, 1.0])},
        )


def test_mlp2_monomial_gauge_composition_and_inverse_are_identity():
    width = 5
    p1 = np.array([2, 4, 1, 3, 0])
    p2 = np.array([1, 3, 0, 4, 2])
    s1 = np.array([0.6, 1.8, 0.9, 2.4, 1.1])
    s2 = np.array([1.7, 0.7, 2.2, 0.8, 1.3])
    alignment = MonomialAlignment(
        p2,
        s2,
        "monomial_weight_mlp2",
        "known",
        architecture="mlp2",
        primary_layer="hidden2",
        layer_permutations={"hidden1": p1, "hidden2": p2},
        layer_positive_scales={"hidden1": s1, "hidden2": s2},
    )

    inverse = invert_monomial_alignment(alignment)
    identity = compose_monomial_alignments(alignment, inverse)

    for layer in ("hidden1", "hidden2"):
        assert np.array_equal(identity.permutation_for(layer), np.arange(width))
        assert np.allclose(identity.positive_scales_for(layer), np.ones(width))


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


def test_log_scale_clipping_and_shrinkage_regularize_large_weight_scales():
    set_seed(789)
    spec = DatasetSpec("toy", (1, 4, 4), 10)
    width = 5
    reference = make_model("mlp", spec, width)
    target = transform_mlp_positive_scale(
        reference,
        spec,
        width,
        np.arange(width),
        np.array([20.0, 0.05, 7.0, 0.2, 1.0]),
    )
    loader = _toy_loader()

    clipped = estimate_monomial_alignment(
        reference,
        target,
        loader,
        "cpu",
        matching="monomial_weight",
        scale_method="clipped",
        log_scale_clip=1.0,
    )
    shrinkage = estimate_monomial_alignment(
        reference,
        target,
        loader,
        "cpu",
        matching="monomial_weight",
        scale_method="shrinkage",
        log_scale_clip=1.0,
        shrinkage=0.5,
    )

    assert clipped.scale_method == "clipped"
    assert float(np.max(np.abs(np.log(clipped.positive_scales)))) <= 1.0 + 1e-9
    assert shrinkage.scale_method == "shrinkage"
    assert float(np.max(np.abs(np.log(shrinkage.positive_scales)))) <= 0.5 + 1e-9


def test_activation_low_similarity_threshold_regularizes_to_identity_scale():
    set_seed(2468)
    spec = DatasetSpec("toy", (1, 4, 4), 10)
    width = 6
    reference = make_model("mlp", spec, width)
    target = make_model("mlp", spec, width)
    loader = _toy_loader()

    alignment = estimate_monomial_alignment(
        reference,
        target,
        loader,
        "cpu",
        matching="monomial_activation",
        scale_method="raw",
        activation_similarity_threshold=2.0,
    )

    assert alignment.scale_method == "raw"
    assert np.allclose(alignment.positive_scales, np.ones(width))
    assert alignment.low_similarity_fraction == 1.0


def test_global_synchronized_scale_method_records_stable_same_capacity_metadata():
    set_seed(1357)
    spec = DatasetSpec("toy", (1, 4, 4), 10)
    width = 4
    reference = make_model("mlp", spec, width)
    models = [
        reference,
        transform_mlp_positive_scale(reference, spec, width, np.arange(width), np.array([2.0, 0.7, 1.5, 1.0])),
        transform_mlp_positive_scale(reference, spec, width, np.arange(width), np.array([0.5, 1.4, 0.8, 1.2])),
    ]
    alignments = estimate_pairwise_monomial_alignments(
        models,
        _toy_loader(),
        "cpu",
        matching="monomial_weight",
        scale_method="global_synchronized",
        log_scale_clip=2.0,
        shrinkage=0.25,
    )
    stats = monomial_scaling_statistics(alignments)

    assert all(alignment.scale_method == "global_synchronized" for alignment in alignments.values())
    assert stats["monomial_max_abs_log_scale"] <= 2.0 + 1e-9
    assert monomial_defect_score(monomial_cycle_defect(alignments, 0, 1, 2)) < 1e-8
