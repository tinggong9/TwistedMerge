import numpy as np

from src.finite_group_cohomology import close_permutation_group
from src.sequential_quotient_lift import (
    build_successive_quotient_chain,
    c2_fourier_components,
    certified_cyclic_quotients,
    cyclic_regular_branch_permutation,
    hidden_permutation_preservation_error,
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
    assert chain.final_kernel_order == 1


def test_s3_chain_has_sign_then_c3():
    group = named_group("S3")

    chain = build_successive_quotient_chain(group, group.elements)

    assert [stage.quotient.quotient_order for stage in chain.stages[:2]] == [2, 3]
    assert chain.final_kernel_order == 1


def test_d4_has_successive_certified_c2_factors():
    group = named_group("D4")

    chain = build_successive_quotient_chain(group, group.elements)

    assert len(chain.stages) >= 2
    assert all(stage.quotient.quotient_order == 2 for stage in chain.stages[:2])


def test_sign_character_is_not_element_order_heuristic():
    even_group = close_permutation_group([(1, 2, 0)], max_group_order=8)

    certs = certified_cyclic_quotients(even_group, target_orders=(2,))

    assert not certs
    assert all(permutation_parity(element) == 0 for element in even_group.elements)


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
