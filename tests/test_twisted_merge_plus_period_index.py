import unittest

import numpy as np

from src.finite_index_twists import evaluate_rank_absorption
from src.model_merging_benchmark import permutation_matrix
from src.period_index_central import heisenberg_generators, period_index_metadata
from src.period_index_detector import detect_period_index_structure
from src.twisted_merge_plus import TwistedMergePlus


def generator_dict(d: int, k: int) -> dict[str, np.ndarray]:
    system = heisenberg_generators(d, k)
    generators: dict[str, np.ndarray] = {}
    for idx in range(k):
        generators[f"U{idx + 1}"] = system.U[idx]
        generators[f"V{idx + 1}"] = system.V[idx]
    return generators


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


def run_tmpp_for_generators(d: int, k: int, rank: int):
    width = d**k
    return TwistedMergePlus().run(
        unresolved_pairwise(width),
        n_models=3,
        width=width,
        period_index_generators=generator_dict(d, k),
        candidate_lift_rank=rank,
    )


def s3_noncentral_generators() -> dict[str, np.ndarray]:
    transposition_12 = permutation_matrix(np.array([1, 0, 2]))
    transposition_23 = permutation_matrix(np.array([0, 2, 1]))
    return {"s12": transposition_12, "s23": transposition_23}


class TwistedMergePlusPeriodIndexTests(unittest.TestCase):
    def test_d2_k2_rank2_period_divisible_but_obstructed(self):
        result = run_tmpp_for_generators(2, 2, 2)
        period_index = result.diagnostics.period_index

        self.assertEqual(result.diagnostics.classification, "period_divisible_index_obstructed")
        self.assertEqual(result.status, "period_divisible_index_obstructed")
        self.assertEqual(result.selected_method, "none")
        self.assertIsNotNone(period_index)
        self.assertEqual(period_index.period, 2)
        self.assertEqual(period_index.index, 4)
        self.assertTrue(period_index.period_divides_rank)
        self.assertFalse(period_index.index_divides_rank)
        self.assertIn("period divisibility alone is insufficient", result.reason)

    def test_d2_k2_rank4_lift_success(self):
        result = run_tmpp_for_generators(2, 2, 4)
        period_index = result.diagnostics.period_index

        self.assertEqual(result.diagnostics.classification, "central_period_index_lift")
        self.assertEqual(result.status, "central_period_index_lift")
        self.assertEqual(result.selected_method, "period_index_projective_morita_lift")
        self.assertIsNotNone(result.period_index_lift)
        self.assertIsNotNone(period_index)
        self.assertEqual(period_index.period, 2)
        self.assertEqual(period_index.index, 4)
        self.assertTrue(period_index.index_divides_rank)

    def test_d3_k2_rank3_and_rank6_fail_rank9_succeeds(self):
        for rank in [3, 6]:
            result = run_tmpp_for_generators(3, 2, rank)
            period_index = result.diagnostics.period_index

            self.assertEqual(result.diagnostics.classification, "period_divisible_index_obstructed")
            self.assertIsNotNone(period_index)
            self.assertEqual(period_index.period, 3)
            self.assertEqual(period_index.index, 9)
            self.assertTrue(period_index.period_divides_rank)
            self.assertFalse(period_index.index_divides_rank)

        result = run_tmpp_for_generators(3, 2, 9)
        self.assertEqual(result.diagnostics.classification, "central_period_index_lift")
        self.assertEqual(result.selected_method, "period_index_projective_morita_lift")
        self.assertEqual(result.diagnostics.period_index.index, 9)

    def test_d2_k3_index8(self):
        for rank in [2, 4]:
            result = run_tmpp_for_generators(2, 3, rank)
            self.assertEqual(result.diagnostics.classification, "period_divisible_index_obstructed")
            self.assertEqual(result.diagnostics.period_index.period, 2)
            self.assertEqual(result.diagnostics.period_index.index, 8)
            self.assertFalse(result.diagnostics.period_index.index_divides_rank)

        result = run_tmpp_for_generators(2, 3, 8)
        self.assertEqual(result.diagnostics.classification, "central_period_index_lift")
        self.assertEqual(result.diagnostics.period_index.index, 8)
        self.assertEqual(result.selected_method, "period_index_projective_morita_lift")

    def test_k1_reduces_to_finite_index_detector(self):
        for d in [2, 3, 4]:
            for rank in [d - 1, d, 2 * d]:
                result = run_tmpp_for_generators(d, 1, rank)
                old_result = evaluate_rank_absorption(d, 1, rank)

                self.assertEqual(result.diagnostics.period_index.period, d)
                self.assertEqual(result.diagnostics.period_index.index, d)
                self.assertEqual(
                    result.diagnostics.classification == "central_period_index_lift",
                    old_result.constructed_lift_success,
                )

    def test_noncentral_generators_rejected(self):
        detection = detect_period_index_structure(s3_noncentral_generators(), candidate_rank=3)

        self.assertFalse(detection.detected)
        self.assertEqual(detection.decision, "not_central_projective")

    def test_unknown_central_projective_not_overclaimed(self):
        system = heisenberg_generators(3, 2)
        generators = {"U1": system.U[0], "V1": system.V[0], "U2": system.U[1]}
        detection = detect_period_index_structure(generators, candidate_rank=3)

        self.assertTrue(detection.detected)
        self.assertEqual(detection.period, 3)
        self.assertIsNone(detection.index)
        self.assertEqual(detection.decision, "central_projective_index_unknown")
        self.assertFalse(detection.index_divides_rank or False)

        result = TwistedMergePlus().run(
            unresolved_pairwise(system.dimension),
            n_models=3,
            width=system.dimension,
            period_index_generators=generators,
            candidate_lift_rank=3,
        )
        self.assertEqual(result.diagnostics.classification, "central_projective_index_unknown")
        self.assertEqual(result.selected_method, "none")
        self.assertIn("no lift is claimed", result.reason)

    def test_no_same_cover_trivialization_claim(self):
        metadata = period_index_metadata(3, 2)
        result = run_tmpp_for_generators(3, 2, 9)

        self.assertEqual(metadata.lift_kind, "finite_rank_projective_or_morita_lift")
        self.assertFalse(metadata.original_class_vanishes_on_same_cover)
        self.assertFalse(metadata.ordinary_untwisted_descent_on_original_rank)
        self.assertIn("projective/Morita", " ".join(result.notes))


if __name__ == "__main__":
    unittest.main()
