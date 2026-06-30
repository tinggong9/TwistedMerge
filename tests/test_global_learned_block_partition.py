import unittest

import numpy as np

from src.learned_block_partition import (
    global_activation_correlation,
    global_output_weight_similarity,
    residual_greedy_blocks,
    validation_selected_blocks,
)


def assignment_sets(partition):
    return {frozenset(block) for block in partition.blocks}


class GlobalLearnedBlockPartitionTests(unittest.TestCase):
    def test_global_activation_correlation_recovers_noncontiguous_planted_blocks(self):
        rng = np.random.default_rng(21)
        a = rng.normal(size=300)
        b = rng.normal(size=300)
        activations = {
            0: np.column_stack(
                [
                    a,
                    b,
                    a + 0.01 * rng.normal(size=300),
                    b + 0.01 * rng.normal(size=300),
                ]
            ),
            1: np.column_stack(
                [
                    a + 0.02 * rng.normal(size=300),
                    b + 0.02 * rng.normal(size=300),
                    a + 0.02 * rng.normal(size=300),
                    b + 0.02 * rng.normal(size=300),
                ]
            ),
        }

        similarity = global_activation_correlation(activations)
        partition = residual_greedy_blocks(similarity, 2, larger_is_better=True, seed=3)

        self.assertEqual(assignment_sets(partition), {frozenset({0, 2}), frozenset({1, 3})})

    def test_global_output_weight_similarity_recovers_noncontiguous_planted_blocks(self):
        output_weights = {
            0: np.array([[1.0, 0.0, 0.95, 0.02], [0.0, 1.0, 0.01, 0.98]]),
            1: np.array([[0.9, 0.02, 1.0, 0.0], [0.01, 1.1, 0.0, 1.0]]),
        }

        similarity = global_output_weight_similarity(output_weights)
        partition = residual_greedy_blocks(similarity, 2, larger_is_better=True, seed=4)

        self.assertEqual(assignment_sets(partition), {frozenset({0, 2}), frozenset({1, 3})})

    def test_validation_selected_blocks_uses_validation_scores_only(self):
        candidates = {
            "contiguous": residual_greedy_blocks(np.ones((4, 4)), 2, seed=0),
            "learned": residual_greedy_blocks(
                np.array(
                    [
                        [0.0, 2.0, 0.1, 2.0],
                        [2.0, 0.0, 2.0, 0.1],
                        [0.1, 2.0, 0.0, 2.0],
                        [2.0, 0.1, 2.0, 0.0],
                    ]
                ),
                2,
                seed=0,
            ),
        }

        selected = validation_selected_blocks(
            candidates,
            {"contiguous": 0.4, "learned": 0.1},
            metric_source="validation_residual",
        )

        self.assertEqual(selected.selected_name, "learned")
        self.assertFalse(selected.used_test_metrics)
        with self.assertRaises(ValueError):
            validation_selected_blocks(candidates, {"contiguous": 0.4, "learned": 0.1}, metric_source="test_accuracy")


if __name__ == "__main__":
    unittest.main()
