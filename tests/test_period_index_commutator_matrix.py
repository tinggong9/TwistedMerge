import unittest

import numpy as np

from src.model_merging_benchmark import permutation_matrix
from src.period_index_central import clock_matrix, heisenberg_generators, shift_matrix
from src.period_index_detector import detect_commutator_matrix_period_index
from src.twisted_merge_plus import TwistedMergePlus


def generator_dict(d: int, k: int) -> dict[str, np.ndarray]:
    system = heisenberg_generators(d, k)
    generators: dict[str, np.ndarray] = {}
    for idx in range(k):
        generators[f"U{idx + 1}"] = system.U[idx]
        generators[f"V{idx + 1}"] = system.V[idx]
    return generators


def shuffled_generator_dict(d: int, k: int) -> dict[str, np.ndarray]:
    system = heisenberg_generators(d, k)
    if k == 2:
        return {"A": system.V[1], "B": system.U[0], "C": system.V[0], "D": system.U[1]}
    if k == 3:
        return {
            "A": system.U[1],
            "B": system.U[0],
            "C": system.V[2],
            "D": system.V[0],
            "E": system.U[2],
            "F": system.V[1],
        }
    raise ValueError("only k=2 or k=3 shuffles are defined")


def rank_deficient_generators(d: int) -> dict[str, np.ndarray]:
    return {
        "A": clock_matrix(d),
        "B": shift_matrix(d),
        "C": np.eye(d, dtype=complex),
        "D": np.eye(d, dtype=complex),
    }


def mixed_period_generators() -> dict[str, np.ndarray]:
    identity3 = np.eye(3, dtype=complex)
    identity4 = np.eye(4, dtype=complex)
    return {
        "U3": np.kron(clock_matrix(3), identity4),
        "V3": np.kron(shift_matrix(3), identity4),
        "U4": np.kron(identity3, clock_matrix(4)),
        "V4": np.kron(identity3, shift_matrix(4)),
    }


def identity_generators(width: int, count: int) -> dict[str, np.ndarray]:
    return {f"I{idx}": np.eye(width, dtype=complex) for idx in range(count)}


def s3_noncentral_generators() -> dict[str, np.ndarray]:
    return {
        "s12": permutation_matrix(np.array([1, 0, 2])),
        "s23": permutation_matrix(np.array([0, 2, 1])),
    }


def unresolved_pairwise(width: int) -> dict[tuple[int, int], np.ndarray]:
    diagonal = np.diag(np.linspace(1.0, 2.0, width)).astype(complex)
    return {
        (0, 0): np.eye(width, dtype=complex),
        (1, 1): np.eye(width, dtype=complex),
        (2, 2): np.eye(width, dtype=complex),
        (0, 1): diagonal,
        (1, 2): np.eye(width, dtype=complex),
        (2, 0): np.eye(width, dtype=complex),
    }


class PeriodIndexCommutatorMatrixTests(unittest.TestCase):
    def test_commutator_matrix_agrees_with_independent_pairs_d2_k2(self):
        obstructed = detect_commutator_matrix_period_index(generator_dict(2, 2), candidate_rank=2)
        accepted = detect_commutator_matrix_period_index(generator_dict(2, 2), candidate_rank=4)

        self.assertEqual(obstructed.detector_mode, "commutator_matrix")
        self.assertEqual(obstructed.period, 2)
        self.assertEqual(obstructed.index, 4)
        self.assertEqual(obstructed.alternating_rank, 4)
        self.assertEqual(obstructed.decision, "period_divisible_index_obstructed")
        self.assertEqual(accepted.decision, "period_index_lift_success")

    def test_commutator_matrix_agrees_with_independent_pairs_d3_k2(self):
        for rank in [3, 6]:
            result = detect_commutator_matrix_period_index(generator_dict(3, 2), candidate_rank=rank)
            self.assertEqual(result.period, 3)
            self.assertEqual(result.index, 9)
            self.assertEqual(result.alternating_rank, 4)
            self.assertEqual(result.decision, "period_divisible_index_obstructed")
        accepted = detect_commutator_matrix_period_index(generator_dict(3, 2), candidate_rank=9)
        self.assertEqual(accepted.decision, "period_index_lift_success")

    def test_rank_deficient_form_has_smaller_index(self):
        result = detect_commutator_matrix_period_index(rank_deficient_generators(3), candidate_rank=3)

        self.assertEqual(result.period, 3)
        self.assertEqual(result.alternating_rank, 2)
        self.assertEqual(result.radical_size, 9)
        self.assertEqual(result.quotient_size, 9)
        self.assertEqual(result.index, 3)
        self.assertEqual(result.decision, "period_index_lift_success")

    def test_degenerate_form_with_trivial_commutators_has_no_projective_index(self):
        result = detect_commutator_matrix_period_index(identity_generators(3, 4), candidate_rank=3)

        self.assertFalse(result.detected)
        self.assertEqual(result.decision, "not_central_projective")
        self.assertIsNone(result.index)

    def test_nonstandard_generator_ordering(self):
        result = detect_commutator_matrix_period_index(shuffled_generator_dict(2, 3), candidate_rank=8)

        self.assertEqual(result.period, 2)
        self.assertEqual(result.index, 8)
        self.assertEqual(result.alternating_rank, 6)
        self.assertEqual(result.decision, "period_index_lift_success")

    def test_composite_modulus_small_bruteforce(self):
        obstructed = detect_commutator_matrix_period_index(generator_dict(4, 1), candidate_rank=2)
        accepted = detect_commutator_matrix_period_index(generator_dict(4, 1), candidate_rank=4)

        self.assertEqual(obstructed.period, 4)
        self.assertEqual(obstructed.index, 4)
        self.assertEqual(obstructed.radical_size, 1)
        self.assertEqual(obstructed.quotient_size, 16)
        self.assertEqual(obstructed.decision, "rank_obstructed")
        self.assertEqual(accepted.decision, "period_index_lift_success")

    def test_unknown_index_not_overclaimed(self):
        result = detect_commutator_matrix_period_index(
            mixed_period_generators(),
            candidate_rank=12,
            max_root_order=4,
        )

        self.assertTrue(result.detected)
        self.assertEqual(result.period, 12)
        self.assertIsNone(result.index)
        self.assertEqual(result.decision, "central_projective_index_unknown")
        self.assertIsNone(result.index_divides_rank)

    def test_noncentral_commutator_rejected(self):
        result = detect_commutator_matrix_period_index(s3_noncentral_generators(), candidate_rank=3)

        self.assertFalse(result.detected)
        self.assertEqual(result.decision, "not_central_projective")

    def test_twisted_merge_plus_uses_commutator_matrix_detector(self):
        result = TwistedMergePlus().run(
            unresolved_pairwise(9),
            n_models=3,
            width=9,
            period_index_generators=shuffled_generator_dict(3, 2),
            candidate_lift_rank=9,
        )

        detection = result.diagnostics.period_index
        self.assertEqual(result.diagnostics.classification, "central_period_index_lift")
        self.assertEqual(result.selected_method, "period_index_projective_morita_lift")
        self.assertEqual(detection.detector_mode, "commutator_matrix")
        self.assertEqual(detection.period, 3)
        self.assertEqual(detection.index, 9)


if __name__ == "__main__":
    unittest.main()
