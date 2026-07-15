#!/usr/bin/env python3
"""N1: fresh natural discovery plus targeted confirmation of the two closest families."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import experiments.compact_natural_twist as natural
from experiments.future_benchmark_common import OUT, patch_compact_paths, stage_result, write_csv

DEST = OUT / "near_term"


def main() -> None:
    patch_compact_paths(natural, DEST)
    natural.main()
    discovery_runs = pd.read_csv(DEST / "natural_runs.csv")
    discovery_residuals = pd.read_csv(DEST / "natural_residuals.csv")
    discovery_nulls = pd.read_csv(DEST / "natural_nulls.csv")
    family_cols = ["dataset", "architecture", "model_count", "relation"]
    scored = discovery_residuals.groupby(family_cols, as_index=False).agg(cycle_residual=("cycle_residual", "mean"), pairwise_error=("pairwise_heldout_alignment_error", "mean"), stable=("calibration_resample_stable", "mean"), persistent_rank=("persistent_rank", "median"))
    validation_gain = discovery_runs.pivot_table(index=[*family_cols, "setting_id"], columns="method", values="accuracy").reset_index()
    validation_gain["validation_only_gain"] = validation_gain["twistedmerge_hodge_lr"] - validation_gain["strict_synchronization"]
    gain = validation_gain.groupby(family_cols, as_index=False).validation_only_gain.mean()
    scored = scored.merge(gain, on=family_cols)
    scored["selection_score"] = scored.cycle_residual / scored.pairwise_error.clip(lower=1e-8) + scored.stable + scored.validation_only_gain
    selected = scored.sort_values("selection_score", ascending=False).head(2).to_dict("records")
    run_rows, residual_rows, null_rows = [], [], []
    trained = 0
    for family in selected:
        for seed in range(10, 17):
            dataset, architecture, relation = str(family["dataset"]), str(family["architecture"]), str(family["relation"])
            data = natural.prepare_data(dataset, seed)
            states, count, elapsed = natural.train_collection(dataset, architecture, relation, seed, data)
            trained += count
            rows, residual, nulls = natural.run_collection(dataset, architecture, relation, seed, 4, states, data, elapsed)
            # A second independent null call provides draws 100--199 without changing the observed test result.
            calibration, _ = natural.evaluate_states(dataset, architecture, states[:4], data.calibration_x)
            second = natural.transition_diagnostics(calibration, architecture, np.random.default_rng(9_100_000 + seed))
            for null_name, values in second["nulls"].items():
                for index, value in enumerate(values, start=100):
                    nulls.append({"setting_id": residual["setting_id"], "null": null_name, "permutation": index, "residual": value, "observed_residual": residual["cycle_residual"], "threshold_95": float(np.quantile(values, 0.95))})
            run_rows.extend(rows); residual_rows.append(residual); null_rows.extend(nulls)
    claims, positive = natural.family_claims(run_rows, residual_rows)
    corrections = []
    run_frame = pd.DataFrame(run_rows)
    for setting_id, block in run_frame.groupby("setting_id"):
        scores = block.set_index("method").accuracy
        residual = next(row for row in residual_rows if row["setting_id"] == setting_id)
        corrections.append({"setting_id": setting_id, "strict_accuracy": scores["strict_synchronization"], "generic_low_rank_accuracy": scores["generic_low_rank_correction"], "hodge_lr_accuracy": scores["twistedmerge_hodge_lr"], "delta_vs_strict": scores["twistedmerge_hodge_lr"] - scores["strict_synchronization"], "delta_vs_generic": scores["twistedmerge_hodge_lr"] - scores["generic_low_rank_correction"], "residual_before": residual["cycle_residual"], "residual_after": residual["residual_after_correction"]})
    write_csv(DEST / "realistic_runs.csv", run_rows)
    write_csv(DEST / "realistic_residuals.csv", residual_rows)
    write_csv(DEST / "realistic_nulls.csv", null_rows)
    write_csv(DEST / "realistic_stability.csv", [{**row, "five_resample_requirement": False, "available_resamples": 3} for row in residual_rows])
    write_csv(DEST / "realistic_corrections.csv", corrections)
    write_csv(DEST / "realistic_claims.csv", claims)
    pd.DataFrame(claims).to_latex(DEST / "tables" / "realistic.tex", index=False, float_format="%.6f")
    null_complete = bool(pd.DataFrame(null_rows).groupby(["setting_id", "null"]).size().min() >= 200)
    five_resamples = False
    gate = bool(positive) and null_complete and five_resamples
    (DEST / "realistic_report.md").write_text(f"# Targeted realistic confirmation\n\nThe two closest families were selected from discovery metadata only, followed by seven fresh four-model collections per family and 200 draws per matched-null family. The existing diagnostic exposes three calibration resamples rather than the required five, so the full gate is **not passed** even if partial numerical conditions are favorable. All fresh checkpoints, nulls, and negative outcomes are retained.\n", encoding="utf-8")
    stage_result("N1", "negative", f"targeted confirmation executed for two families; full gate not passed; fresh checkpoints={trained}", selected_families=selected, null_draws_complete=null_complete, five_resamples_complete=five_resamples)


if __name__ == "__main__":
    main()
