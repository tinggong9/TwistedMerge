import unittest

import numpy as np

from src.period_index_central import clock_matrix, heisenberg_generators, shift_matrix
from src.period_index_detector import robust_detect_commutator_matrix_period_index
from src.period_index_mining import (
    detect_mined_period_index,
    generate_noisy_heisenberg_generators,
    generate_noncentral_controls,
    mine_period_index_generators,
)
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


def mixed_period_generators() -> dict[str, np.ndarray]:
    identity3 = np.eye(3, dtype=complex)
    identity4 = np.eye(4, dtype=complex)
    return {
        "U3": np.kron(clock_matrix(3), identity4),
        "V3": np.kron(shift_matrix(3), identity4),
        "U4": np.kron(identity3, clock_matrix(4)),
        "V4": np.kron(identity3, shift_matrix(4)),
    }


def synthetic_transition_maps() -> tuple[dict[tuple[int, int], np.ndarray], list[tuple[int, ...]]]:
    system = heisenberg_generators(2, 2)
    hidden = [system.U[0], system.V[0], system.U[1], system.V[1]]
    loops = [
        (0, 1, 2, 0),
        (0, 2, 3, 0),
        (0, 3, 4, 0),
        (0, 4, 5, 0),
    ]
    identity = np.eye(system.dimension, dtype=complex)
    transition_maps: dict[tuple[int, int], np.ndarray] = {}
    for loop, generator in zip(loops, hidden, strict=True):
        transition_maps[(loop[0], loop[1])] = generator
        transition_maps[(loop[1], loop[2])] = identity
        transition_maps[(loop[2], loop[3])] = identity
    return transition_maps, loops


class RobustPeriodIndexDetectorTests(unittest.TestCase):
    def test_exact_cases_still_certified(self):
        d2 = robust_detect_commutator_matrix_period_index(generator_dict(2, 2), candidate_rank=4)
        d3_obstructed = robust_detect_commutator_matrix_period_index(generator_dict(3, 2), candidate_rank=3)
        d3_lift = robust_detect_commutator_matrix_period_index(generator_dict(3, 2), candidate_rank=9)

        self.assertEqual(d2.status, "certified")
        self.assertEqual(d2.period, 2)
        self.assertEqual(d2.index, 4)
        self.assertEqual(d2.decision, "period_index_lift_success")
        self.assertEqual(d3_obstructed.status, "certified")
        self.assertEqual(d3_obstructed.index, 9)
        self.assertEqual(d3_obstructed.decision, "period_divisible_index_obstructed")
        self.assertEqual(d3_lift.decision, "period_index_lift_success")

    def test_small_noise_certified(self):
        d2 = robust_detect_commutator_matrix_period_index(
            generate_noisy_heisenberg_generators(2, 2, 1e-5, "unitary_near_identity", seed=11),
            candidate_rank=4,
        )
        d3 = robust_detect_commutator_matrix_period_index(
            generate_noisy_heisenberg_generators(3, 2, 1e-6, "entrywise_projected_unitary", seed=12),
            candidate_rank=9,
        )

        self.assertEqual(d2.status, "certified")
        self.assertEqual(d2.threshold_level, "medium")
        self.assertEqual(d2.index, 4)
        self.assertEqual(d3.status, "certified")
        self.assertEqual(d3.index, 9)

    def test_medium_noise_uncertain_not_lifted(self):
        generators = generate_noisy_heisenberg_generators(2, 2, 1e-4, "unitary_near_identity", seed=21)
        detection = robust_detect_commutator_matrix_period_index(generators, candidate_rank=4)
        result = TwistedMergePlus().run(
            unresolved_pairwise(4),
            n_models=3,
            width=4,
            period_index_generators=generators,
            candidate_lift_rank=4,
            period_index_detection_mode="robust_only",
        )

        self.assertEqual(detection.status, "candidate_uncertain")
        self.assertEqual(detection.decision, "central_projective_candidate_uncertain")
        self.assertEqual(result.diagnostics.classification, "central_projective_candidate_uncertain")
        self.assertEqual(result.selected_method, "none")
        self.assertIsNone(result.period_index_lift)

    def test_large_noise_rejected(self):
        detection = robust_detect_commutator_matrix_period_index(
            generate_noisy_heisenberg_generators(2, 2, 1e-2, "unitary_near_identity", seed=31),
            candidate_rank=4,
        )

        self.assertEqual(detection.status, "rejected_noncentral")
        self.assertEqual(detection.decision, "not_central_projective")

    def test_noncentral_control_rejected_even_with_low_noise(self):
        for control_type in ["permutation", "random_gl"]:
            detection = robust_detect_commutator_matrix_period_index(
                generate_noncentral_controls(3, 1e-6, seed=41, control_type=control_type),
                candidate_rank=3,
            )
            self.assertEqual(detection.status, "rejected_noncentral")
            self.assertEqual(detection.decision, "not_central_projective")

    def test_period_divisible_index_obstructed_under_noise(self):
        generators = generate_noisy_heisenberg_generators(3, 2, 1e-6, "unitary_near_identity", seed=51)
        for rank in [3, 6]:
            detection = robust_detect_commutator_matrix_period_index(generators, candidate_rank=rank)
            self.assertEqual(detection.status, "certified")
            self.assertEqual(detection.period, 3)
            self.assertEqual(detection.index, 9)
            self.assertEqual(detection.decision, "period_divisible_index_obstructed")
            self.assertFalse(detection.index_divides_rank)

    def test_unknown_index_not_overclaimed(self):
        detection = robust_detect_commutator_matrix_period_index(
            mixed_period_generators(),
            candidate_rank=12,
            max_root_order=4,
        )
        result = TwistedMergePlus().run(
            unresolved_pairwise(12),
            n_models=3,
            width=12,
            period_index_generators=mixed_period_generators(),
            candidate_lift_rank=12,
            max_root_order=4,
        )

        self.assertEqual(detection.status, "unknown_index")
        self.assertEqual(detection.decision, "central_projective_index_unknown")
        self.assertIsNone(detection.index)
        self.assertEqual(result.selected_method, "none")

    def test_generator_mining_on_synthetic_loops(self):
        transition_maps, loops = synthetic_transition_maps()
        mining = mine_period_index_generators(transition_maps, loops=loops, max_generators=4)
        mined_detection = detect_mined_period_index(
            transition_maps,
            candidate_rank=4,
            loops=loops,
            max_generators=4,
        )

        self.assertEqual(mining.status, "mined_candidate")
        self.assertEqual(len(mining.generators), 4)
        self.assertIsNotNone(mined_detection.detection)
        self.assertEqual(mined_detection.detection.status, "certified")
        self.assertEqual(mined_detection.detection.detector_mode, "robust_commutator_matrix_mined_candidate")
        self.assertEqual(mined_detection.detection.period, 2)
        self.assertEqual(mined_detection.detection.index, 4)

    def test_twisted_merge_plus_robust_integration(self):
        noisy_generators = generate_noisy_heisenberg_generators(2, 2, 1e-5, "unitary_near_identity", seed=11)
        noisy_result = TwistedMergePlus().run(
            unresolved_pairwise(4),
            n_models=3,
            width=4,
            period_index_generators=noisy_generators,
            candidate_lift_rank=4,
        )
        transition_maps, _loops = synthetic_transition_maps()
        mined_result = TwistedMergePlus().run(
            unresolved_pairwise(4),
            n_models=3,
            width=4,
            candidate_lift_rank=4,
            candidate_transition_maps_for_mining=transition_maps,
        )

        self.assertEqual(noisy_result.diagnostics.classification, "central_period_index_lift")
        self.assertEqual(noisy_result.selected_method, "period_index_projective_morita_lift")
        self.assertEqual(noisy_result.diagnostics.period_index.status, "certified")
        self.assertEqual(mined_result.diagnostics.classification, "central_period_index_lift")
        self.assertEqual(mined_result.selected_method, "period_index_projective_morita_lift")
        self.assertEqual(mined_result.diagnostics.period_index.detector_mode, "robust_commutator_matrix_mined_candidate")


if __name__ == "__main__":
    unittest.main()
