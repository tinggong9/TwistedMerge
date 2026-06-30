import unittest

from experiments.block_gauge_phase_diagram import learned_partition_rows


class Args:
    learned_block_seeds = 8
    learned_samples = 160


class LearnedBlockPartitionStatisticsTests(unittest.TestCase):
    def test_learned_noncontiguous_blocks_beat_contiguous_blocks(self):
        rows = learned_partition_rows(Args())
        pivot = rows.pivot_table(index="setting_id", columns="partition_method", values="validation_block_residual")

        self.assertIn("contiguous", pivot.columns)
        self.assertIn("validation_selected_blocks", pivot.columns)
        self.assertLess(pivot["validation_selected_blocks"].mean(), pivot["contiguous"].mean())
        selected = rows[rows["partition_method"] == "validation_selected_blocks"]
        self.assertGreaterEqual((selected["block_recovery_accuracy"] > 0.99).mean(), 0.9)


if __name__ == "__main__":
    unittest.main()
