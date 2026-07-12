import numpy as np

from src.finite_group_cohomology import close_permutation_group
from src.sequential_quotient_lift import (
    bootstrap_chain_stability,
    build_successive_quotient_chain,
    c2_fourier_components,
    certified_cyclic_quotients,
    coset_action_representation,
    cyclic_regular_branch_permutation,
    hidden_permutation_preservation_error,
    label_permutation_logit_invariance,
    named_group,
    permutation_parity,
    uniform_pool,
)


def test_exact_c2_quotient_from_cyclic_group():
    group = named_group("C2")

    certs = certified_cyclic_quotients(group, target_orders=(2, 3))

    c2 = [cert for cert in certs if cert.quotient_order == 2]
    assert c2
    assert c2[0].certified is True
    assert c2[0].homomorphism_residual == 0.0
    assert c2[0].kernel_order == 1


def test_exact_quotient_chain_for_klein_four_is_c2_then_c2():
    group = named_group("C2xC2")

    chain = build_successive_quotient_chain(group, group.elements)

    assert [stage.quotient.quotient_order for stage in chain.stages[:2]] == [2, 2]
    assert [stage.residual_group_order for stage in chain.stages[:2]] == [2, 1]
    assert chain.stages[0].residual_after > 0.0
    assert chain.stages[-1].residual_after == 0.0
    assert all(stage.coset_action_law_residual == 0.0 for stage in chain.stages)
    assert chain.stages[-1].final_regular_representation_verified is True
    assert chain.final_kernel_order == 1


def test_s3_chain_has_sign_then_c3():
    group = named_group("S3")

    chain = build_successive_quotient_chain(group, group.elements)

    assert [stage.quotient.quotient_order for stage in chain.stages[:2]] == [2, 3]
    assert [stage.residual_group_order for stage in chain.stages[:2]] == [3, 1]
    assert all(stage.coset_action_law_residual == 0.0 for stage in chain.stages)
    assert chain.stages[-1].final_regular_representation_verified is True
    assert chain.final_kernel_order == 1


def test_d4_has_successive_certified_c2_factors():
    group = named_group("D4")

    chain = build_successive_quotient_chain(group, group.elements)

    assert [stage.quotient.quotient_order for stage in chain.stages] == [2, 2, 2]
    assert [stage.residual_group_order for stage in chain.stages] == [4, 2, 1]
    assert all(stage.coset_action_law_residual == 0.0 for stage in chain.stages)
    assert chain.stages[-1].final_regular_representation_verified is True


def test_c4_chain_retains_extension_as_c2_then_c2():
    group = named_group("C4")

    chain = build_successive_quotient_chain(group, group.elements)

    assert [stage.quotient.quotient_order for stage in chain.stages] == [2, 2]
    assert [stage.residual_group_order for stage in chain.stages] == [2, 1]
    assert chain.stages[-1].final_regular_representation_verified is True


def test_coset_action_kernel_matches_normal_subgroup():
    group = named_group("C4")
    chain = build_successive_quotient_chain(group, group.elements)
    first_kernel = chain.stages[0].quotient.kernel

    rep = coset_action_representation(group, first_kernel)

    assert len(rep.cosets) == 2
    assert rep.law_residual == 0.0
    assert rep.kernel_order == 2
    assert rep.stabilizer_matches_subgroup is True


def test_sign_character_is_not_element_order_heuristic():
    even_group = close_permutation_group([(1, 2, 0)], max_group_order=8)

    certs = certified_cyclic_quotients(even_group, target_orders=(2,))

    assert not certs
    assert all(permutation_parity(element) == 0 for element in even_group.elements)


def test_truncated_sign_certificate_does_not_recurse_into_fake_kernel():
    truncated = close_permutation_group([(1, 0, 2), (1, 2, 0)], max_group_order=3)

    certs = certified_cyclic_quotients(truncated, target_orders=(2,))
    chain = build_successive_quotient_chain(truncated, truncated.elements)

    assert truncated.truncated is True
    assert certs
    assert certs[0].certification_method == "ambient_permutation_sign_character_truncated_group"
    assert certs[0].kernel_order is None
    assert len(chain.stages) == 1
    assert chain.final_kernel_order is None
    assert chain.stopped_reason == "truncated_sign_only_no_recursive_kernel"


def test_bootstrap_chain_stability_is_resampled_not_constant():
    group = named_group("C2xC2")
    sparse_holonomies = group.generators

    stability = bootstrap_chain_stability(group, sparse_holonomies, n_bootstrap=100, seed=7)

    assert stability["bootstrap_method"] == "resample_holonomies_rebuild_group"
    assert 0.0 <= stability["bootstrap_stability"] < 1.0


def test_cyclic_regular_branch_permutation():
    assert cyclic_regular_branch_permutation(3, 1) == (1, 2, 0)
    assert cyclic_regular_branch_permutation(3, -1) == (2, 0, 1)


def test_c2_fourier_pooling_retains_minus_component():
    branches = np.asarray([[[4.0, 2.0], [2.0, -2.0]]])

    z_plus, z_minus = c2_fourier_components(branches)

    assert np.allclose(z_plus, [[3.0, 0.0]])
    assert np.allclose(z_minus, [[1.0, 2.0]])
    assert np.allclose(uniform_pool(branches), z_plus)
    assert not np.allclose(z_minus, 0.0)


def test_relu_hidden_permutation_preserves_logits():
    rng = np.random.default_rng(12)
    x = rng.normal(size=(9, 5))
    w1 = rng.normal(size=(5, 4))
    b1 = rng.normal(size=4)
    w2 = rng.normal(size=(4, 3))
    b2 = rng.normal(size=3)

    err = hidden_permutation_preservation_error(x, w1, b1, w2, b2, [2, 0, 3, 1])

    assert err < 1e-10


def test_label_permutation_does_not_change_label_independent_logits():
    rng = np.random.default_rng(123)
    x = rng.normal(size=(11, 4))
    labels = np.arange(11) % 3
    permuted = labels[::-1]
    weights = rng.normal(size=(4, 3))

    def logits_from_model(inputs, _labels):
        return inputs @ weights

    err = label_permutation_logit_invariance(logits_from_model, x, labels, permuted)

    assert err == 0.0


def test_label_permutation_guard_catches_label_injected_logits():
    x = np.zeros((6, 2))
    labels = np.arange(6) % 3
    permuted = (labels + 1) % 3

    def leaky_logits(_inputs, current_labels):
        logits = np.zeros((len(current_labels), 3))
        logits[np.arange(len(current_labels)), current_labels] = 1.0
        return logits

    err = label_permutation_logit_invariance(leaky_logits, x, labels, permuted)

    assert err > 0.0
