import unittest

import numpy as np

from src.learned_block_partition import (
    activation_correlation_partition,
    make_block_partition,
    output_weight_similarity_partition,
)


def assignment_sets(partition):
    return {frozenset(block) for block in partition.blocks}


class LearnedBlockPartitionTests(unittest.TestCase):
    def test_contiguous_partition_assignment_string(self):
        partition = make_block_partition("contiguous", 5, 2)

        self.assertEqual(partition.blocks, ((0, 1), (2, 3), (4,)))
        self.assertEqual(partition.assignment_string(), "0,0,1,1,2")

    def test_activation_correlation_clusters_correlated_units(self):
        rng = np.random.default_rng(12)
        a = rng.normal(size=200)
        b = rng.normal(size=200)
        activations = np.column_stack(
            [
                a,
                a + 0.01 * rng.normal(size=200),
                b,
                b + 0.01 * rng.normal(size=200),
            ]
        )

        partition = activation_correlation_partition(activations, 2, seed=4)

        self.assertEqual(assignment_sets(partition), {frozenset({0, 1}), frozenset({2, 3})})
        self.assertEqual(partition.assignment_string(), activation_correlation_partition(activations, 2, seed=4).assignment_string())

    def test_output_weight_similarity_clusters_similar_columns(self):
        weights = np.array(
            [
                [1.0, 0.95, 0.0, 0.02],
                [0.0, 0.01, 1.0, 1.05],
            ]
        )

        partition = output_weight_similarity_partition(weights, 2, seed=9)

        self.assertEqual(assignment_sets(partition), {frozenset({0, 1}), frozenset({2, 3})})

    def test_missing_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            make_block_partition("activation_correlation", 4, 2)
        with self.assertRaises(ValueError):
            make_block_partition("output_weight_similarity", 4, 2)


if __name__ == "__main__":
    unittest.main()
