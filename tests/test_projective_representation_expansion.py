from experiments.projective_representation_expansion import carry_cocycle, cocycle_identity_error, cyclic_group, projective_multiplication_error, projective_regular


def test_cyclic_carry_cocycle_and_projective_regular_representation():
    group = cyclic_group(4); cocycle = carry_cocycle(group, 3)
    assert cocycle_identity_error(group, cocycle, 3) == 0
    matrices = projective_regular(group, cocycle, 3)
    assert projective_multiplication_error(group, matrices, cocycle, 3) < 1e-10
