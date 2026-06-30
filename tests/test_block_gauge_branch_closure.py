import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from experiments.block_gauge_phase_diagram import relu_diagnostic_rows


ROOT = Path(__file__).resolve().parents[1]


class ExistingReportArgs:
    reports_dir = ROOT / "reports"


class BlockGaugeBranchClosureTests(unittest.TestCase):
    def test_smoke_closure_outputs_and_threshold_columns(self):
        with tempfile.TemporaryDirectory(prefix="block_gauge_closure_") as tmp:
            reports_dir = Path(tmp)
            cmd = [
                sys.executable,
                "experiments/block_gauge_phase_diagram.py",
                "--synthetic-seeds",
                "1",
                "--n-models",
                "3",
                "--widths",
                "4",
                "--block-sizes",
                "2",
                "--noise-levels",
                "0.0,0.4",
                "--learned-block-seeds",
                "1",
                "--block-learning-seeds",
                "1",
                "--block-train-samples",
                "128",
                "--block-test-samples",
                "64",
                "--block-epochs",
                "1",
                "--reports-dir",
                str(reports_dir),
                "--calibration-floor",
                "1e-12",
            ]
            env = dict(os.environ)
            env.setdefault("PYTHONPYCACHEPREFIX", "/private/tmp/codex-pycache")
            env.setdefault("MPLCONFIGDIR", "/private/tmp/mplconfig")
            subprocess.run(cmd, cwd=ROOT, check=True, stdout=subprocess.DEVNULL, env=env)

            phase = pd.read_csv(reports_dir / "csv" / "block_gauge_phase_diagram.csv")
            by_noise = pd.read_csv(reports_dir / "csv" / "block_gauge_acceptance_by_noise.csv")
            block = pd.read_csv(reports_dir / "csv" / "block_compatible_learning_benchmark.csv")
            config_text = (reports_dir / "configs" / "block_gauge_branch_closure_config.json").read_text()
            closure_text = (reports_dir / "block_gauge_branch_closure_report.md").read_text()

            self.assertIn("raw_calibrated_threshold", phase.columns)
            self.assertIn("effective_threshold", phase.columns)
            self.assertTrue((phase["effective_threshold"] >= phase["numerical_floor"]).all())
            self.assertIn("raw_calibrated_threshold", by_noise.columns)
            self.assertIn("effective_threshold", by_noise.columns)
            self.assertIn("block-gauge diagnostics", closure_text)
            self.assertIn("raw_calibrated_threshold", config_text)

            aligned = block[block["method"] == "optimized_block_gauge_aligned_average"]
            self.assertFalse(aligned.empty)
            self.assertTrue(aligned["capacity_matched"].astype(bool).all())

    def test_existing_relu_diagnostic_rows_remain_diagnostic_only(self):
        rows = relu_diagnostic_rows(ExistingReportArgs())
        if rows.empty:
            self.skipTest("prior ReLU diagnostic CSV is not available")

        self.assertFalse(rows["block_merge_accuracy_reported"].any())
        self.assertFalse(rows["exact_same_architecture_symmetry"].any())
        self.assertEqual(float(rows["observed_scalar_projective_candidate"].mean()), 0.0)


if __name__ == "__main__":
    unittest.main()
