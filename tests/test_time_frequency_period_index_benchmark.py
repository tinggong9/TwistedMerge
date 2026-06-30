import unittest

import numpy as np

from src.period_index_detector import detect_commutator_matrix_period_index
from src.time_frequency_benchmark import (
    TIME_FREQUENCY_SCOPE_NOTE,
    check_time_frequency_relations,
    complex_to_real_block_matrix,
    generate_time_frequency_dataset,
    orbit_invariant_prototype_accuracy,
    primitive_time_frequency_root,
    real_vector_to_complex,
    time_frequency_generator_dict,
    time_frequency_generators,
    time_shift_operator,
    frequency_modulation_operator,
)


class TimeFrequencyPeriodIndexBenchmarkTests(unittest.TestCase):
    def test_time_shift_modulation_relation(self):
        for d in [2, 3, 4]:
            T = time_shift_operator(d)
            M = frequency_modulation_operator(d)
            zeta = primitive_time_frequency_root(d)
            self.assertTrue(np.allclose(M @ T, zeta * (T @ M), atol=1e-10))

        relations = check_time_frequency_relations(time_frequency_generators(3, 2))
        self.assertTrue(relations.all_relations_hold)
        self.assertLessEqual(relations.max_relation_residual, 1e-10)
        self.assertEqual(relations.convention, "M_i T_i = zeta T_i M_i")

    def test_realification_preserves_commutator_phase(self):
        d = 3
        T = time_shift_operator(d)
        M = frequency_modulation_operator(d)
        zeta = primitive_time_frequency_root(d)
        commutator = T @ M @ np.linalg.inv(T) @ np.linalg.inv(M)
        real_commutator = (
            complex_to_real_block_matrix(T)
            @ complex_to_real_block_matrix(M)
            @ np.linalg.inv(complex_to_real_block_matrix(T))
            @ np.linalg.inv(complex_to_real_block_matrix(M))
        )
        target = complex_to_real_block_matrix(np.conjugate(zeta) * np.eye(d, dtype=complex))

        self.assertTrue(np.allclose(real_commutator, target, atol=1e-10))

    def test_known_operator_chart_d2_k2(self):
        obstructed = detect_commutator_matrix_period_index(time_frequency_generator_dict(2, 2), candidate_rank=2)
        accepted = detect_commutator_matrix_period_index(time_frequency_generator_dict(2, 2), candidate_rank=4)

        self.assertEqual(obstructed.period, 2)
        self.assertEqual(obstructed.index, 4)
        self.assertEqual(obstructed.decision, "period_divisible_index_obstructed")
        self.assertEqual(accepted.decision, "period_index_lift_success")

    def test_known_operator_chart_d3_k2(self):
        accepted = detect_commutator_matrix_period_index(time_frequency_generator_dict(3, 2), candidate_rank=9)

        self.assertEqual(accepted.period, 3)
        self.assertEqual(accepted.index, 9)
        self.assertEqual(accepted.alternating_rank, 4)
        self.assertEqual(accepted.decision, "period_index_lift_success")

    def test_period_divisible_but_index_obstructed(self):
        for rank in [3, 6]:
            result = detect_commutator_matrix_period_index(time_frequency_generator_dict(3, 2), candidate_rank=rank)
            self.assertEqual(result.period, 3)
            self.assertEqual(result.index, 9)
            self.assertTrue(result.period_divides_rank)
            self.assertFalse(result.index_divides_rank)
            self.assertEqual(result.decision, "period_divisible_index_obstructed")

        accepted = detect_commutator_matrix_period_index(time_frequency_generator_dict(3, 2), candidate_rank=9)
        self.assertEqual(accepted.decision, "period_index_lift_success")

    def test_time_frequency_dataset_shapes(self):
        dataset = generate_time_frequency_dataset(
            3,
            2,
            n_classes=3,
            train_samples=11,
            validation_samples=7,
            test_samples=5,
            noise_level=0.0,
            seed=17,
        )

        self.assertEqual(dataset.train_x_complex.shape, (11, 9))
        self.assertEqual(dataset.train_x.shape, (11, 18))
        self.assertEqual(dataset.validation_x.shape, (7, 18))
        self.assertEqual(dataset.test_x.shape, (5, 18))
        self.assertTrue(np.allclose(real_vector_to_complex(dataset.train_x), dataset.train_x_complex))
        accuracy = orbit_invariant_prototype_accuracy(dataset, split="test")
        self.assertGreaterEqual(accuracy, 0.8)
        self.assertLessEqual(accuracy, 1.0)

    def test_no_mnist_claim(self):
        self.assertIn("not a MNIST/CIFAR residual claim", TIME_FREQUENCY_SCOPE_NOTE)
        self.assertIn("learned model chart transitions are not certified", TIME_FREQUENCY_SCOPE_NOTE)


if __name__ == "__main__":
    unittest.main()
