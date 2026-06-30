import csv
import unittest
from pathlib import Path

import numpy as np

from src.block_gauge_alignment import (
    estimate_block_orthogonal_alignments,
    orthogonal_procrustes,
)
from src.structure_group_ladder import StructureGroupLadderMerge


ROOT = Path(__file__).resolve().parents[1]


def level(result, name):
    return next(diag for diag in result.diagnostics if diag.level == name)


def rotation(theta: float) -> np.ndarray:
    return np.array(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ]
    )


def block_diag(blocks: list[np.ndarray]) -> np.ndarray:
    size = sum(block.shape[0] for block in blocks)
    out = np.zeros((size, size))
    cursor = 0
    for block in blocks:
        n = block.shape[0]
        out[cursor : cursor + n, cursor : cursor + n] = block
        cursor += n
    return out


class BlockGaugeAlignmentTests(unittest.TestCase):
    def test_block_rotation_recovered(self):
        rng = np.random.default_rng(123)
        base = rng.normal(size=(256, 4))
        gauges = {
            0: np.eye(4),
            1: block_diag([rotation(0.35), rotation(-0.2)]),
            2: block_diag([rotation(-0.15), rotation(0.4)]),
        }
        activations = {idx: base @ gauge for idx, gauge in gauges.items()}
        perms = {(i, j): np.arange(4) for i in range(3) for j in range(3)}

        pairwise, stats = estimate_block_orthogonal_alignments(
            perms,
            activations,
            n_models=3,
            width=4,
            block_size=2,
        )

        np.testing.assert_allclose(pairwise[(0, 1)], gauges[1], atol=1e-10)
        self.assertLess(stats[(0, 1)].mean_block_residual, 1e-10)

        result = StructureGroupLadderMerge(block_size=2).run(
            {"permutation": perms, "block_orthogonal": pairwise},
            n_models=3,
            width=4,
            triples=[(0, 1, 2)],
        )
        block_diag_result = level(result, "block_orthogonal")
        self.assertEqual(block_diag_result.residual_type, "block_gauge_reduces_residual")
        self.assertLess(block_diag_result.cycle_score, 1e-10)

    def test_block_noncentral_not_brauer(self):
        reflection = np.array([[0.0, 1.0], [1.0, 0.0]])
        rot = rotation(0.4)
        pairwise = {
            (0, 0): np.eye(2),
            (1, 1): np.eye(2),
            (2, 2): np.eye(2),
            (0, 1): reflection,
            (1, 2): rot,
            (2, 0): np.linalg.inv(reflection) @ np.linalg.inv(rot),
        }

        result = StructureGroupLadderMerge().run(
            {"block_orthogonal": pairwise},
            n_models=3,
            width=2,
            triples=[(0, 1, 2)],
        )
        diag = level(result, "block_orthogonal")

        self.assertEqual(diag.residual_type, "block_noncentral_holonomy")
        self.assertFalse(diag.supports_brauer_projective_interpretation)
        self.assertGreater(diag.centrality_score, 0.1)

    def test_scalar_block_phase_detected(self):
        pairwise = {
            (0, 0): np.eye(4),
            (1, 1): np.eye(4),
            (2, 2): np.eye(4),
            (0, 1): np.eye(4),
            (1, 2): np.eye(4),
            (2, 0): -np.eye(4),
        }

        result = StructureGroupLadderMerge().run(
            {"block_orthogonal": pairwise},
            n_models=3,
            width=4,
            triples=[(0, 1, 2)],
        )
        diag = level(result, "block_orthogonal")

        self.assertEqual(diag.residual_type, "central_projective_after_block")
        self.assertTrue(diag.supports_brauer_projective_interpretation)
        self.assertEqual(diag.detected_order_d, 2)

    def test_real_block_rows_not_overclaimed(self):
        path = ROOT / "reports" / "csv" / "block_orthogonal_ladder.csv"
        if not path.exists():
            self.skipTest("block-orthogonal report CSV has not been generated")
        with path.open(newline="", encoding="utf-8") as handle:
            rows = [row for row in csv.DictReader(handle) if row.get("source") == "real_mnist" and row.get("level") == "block_orthogonal"]
        self.assertTrue(rows)
        for row in rows:
            supports = row["supports_brauer_projective_interpretation"] == "True"
            if supports:
                self.assertLessEqual(float(row["centrality_score"]), 1e-6)
                self.assertLessEqual(float(row["phase_residual"]), 1e-6)
                self.assertGreater(int(float(row["detected_order_d"])), 1)

    def test_procrustes_rejects_shape_mismatch(self):
        with self.assertRaises(ValueError):
            orthogonal_procrustes(np.zeros((3, 2)), np.zeros((3, 3)))


if __name__ == "__main__":
    unittest.main()
