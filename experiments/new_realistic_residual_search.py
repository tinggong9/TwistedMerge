#!/usr/bin/env python3
"""B4: preregistered search over new ResNet and multiview residual families."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.next_program_common import OUT, git_head, write_csv

DEST = OUT / "iclr"


def read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def family_metrics(
    family: str,
    residual_rows: list[dict[str, object]],
    stability_rows: list[dict[str, object]],
    null_rows: list[dict[str, object]],
    fit_key: str,
    gauge_filter: tuple[str, str] | None = None,
) -> list[dict[str, object]]:
    output = []
    for collection in range(5):
        block = [row for row in residual_rows if int(row["collection"]) == collection]
        stable = [row for row in stability_rows if int(row["collection"]) == collection]
        null = [row for row in null_rows if int(row["collection"]) == collection]
        if gauge_filter:
            layer, gauge = gauge_filter
            block = [row for row in block if row.get("layer") == layer and row.get("gauge_family") == gauge]
            stable = [row for row in stable if row.get("layer") == layer and row.get("gauge_family") == gauge]
            null = [row for row in null if row.get("layer") == layer and row.get("gauge_family") == gauge]
        residual = float(np.mean([float(row["cycle_residual"]) for row in block]))
        fit = float(np.mean([float(row[fit_key]) for row in block]))
        ranks = [int(float(row["residual_rank"])) for row in stable]
        percentile = float(np.mean([float(row["statistic"]) < residual for row in null])) if null else 0.0
        output.append(
            {
                "family": family,
                "collection": collection,
                "heldout_pairwise_fit": fit,
                "cycle_residual": residual,
                "null_percentile": percentile,
                "calibration_resamples": len(stable),
                "residual_rank_reproducible": len(set(ranks)) == 1 and len(ranks) == 5,
                "residual_rank": ranks[0] if ranks and len(set(ranks)) == 1 else "",
            }
        )
    return output


def selection_metrics(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summary = []
    for family in sorted({str(row["family"]) for row in rows}):
        block = [row for row in rows if row["family"] == family]
        stable = all(bool(row["residual_rank_reproducible"]) for row in block)
        high_null = min(float(row["null_percentile"]) for row in block) >= 0.95
        fit = float(np.mean([float(row["heldout_pairwise_fit"]) for row in block]))
        selected = stable and high_null and fit <= 0.25
        summary.append({"family": family, "collections": len(block), "mean_pairwise_fit": fit, "minimum_null_percentile": min(float(row["null_percentile"]) for row in block), "rank_reproducible_all_collections": stable, "selected_without_test_accuracy": selected})
    selected = [row for row in summary if row["selected_without_test_accuracy"]]
    if len(selected) > 2:
        selected.sort(key=lambda row: (-float(row["minimum_null_percentile"]), float(row["mean_pairwise_fit"])))
        keep = {row["family"] for row in selected[:2]}
        for row in summary: row["selected_without_test_accuracy"] = row["family"] in keep
    return summary


def main() -> None:
    required = [DEST / name for name in ("full_model_hodge.csv", "full_model_transitions.csv", "full_model_stability.csv", "full_model_nulls.csv", "multiview_transitions.csv", "multiview_stability.csv", "multiview_nulls.csv")]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("B4 requires completed B1 and B3 discovery artifacts: " + ", ".join(missing))
    full_residual = read(DEST / "full_model_hodge.csv")
    full_transitions = read(DEST / "full_model_transitions.csv")
    # Attach the pairwise fit mean to the primary hodge rows.
    for row in full_residual:
        matching = [candidate for candidate in full_transitions if candidate["collection"] == row["collection"] and candidate["layer"] == row["layer"] and candidate["gauge_family"] == row["gauge_family"]]
        row["heldout_pairwise_fit"] = np.mean([float(value["heldout_pairwise_fit"]) for value in matching])
    rows = family_metrics("domain_specific_final_block_resnets", full_residual, read(DEST / "full_model_stability.csv"), read(DEST / "full_model_nulls.csv"), "heldout_pairwise_fit", ("penultimate", "orthogonal_procrustes"))
    rows.extend(family_metrics("multiview_specialists", read(DEST / "multiview_transitions.csv"), read(DEST / "multiview_stability.csv"), read(DEST / "multiview_nulls.csv"), "heldout_transition_fit"))
    summary = selection_metrics(rows)
    selected = {row["family"] for row in summary if row["selected_without_test_accuracy"]}
    natural_runs = []
    method_mappings = {
        "domain_specific_final_block_resnets": [("strict_synchronization", "strict_synchronization"), ("hodge_gated_ordinary_fallback", "ordinary_validation_fallback"), ("generic_low_rank_merge", "generic_low_rank_correction"), ("generic_router", "generic_router"), ("structured_retransport_certified_only", "structured_correction_certified_only")],
        "multiview_specialists": [("strict_graph_synchronization", "strict_synchronization"), ("raw_parameter_merge", "ordinary_validation_fallback"), ("generic_low_rank_correction", "generic_low_rank_correction"), ("generic_moe", "generic_router"), ("inferred_view_structured_retransport", "structured_correction_certified_only")],
    }
    source_files = {"domain_specific_final_block_resnets": DEST / "full_model_runs.csv", "multiview_specialists": DEST / "multiview_runs.csv"}
    for family in selected:
        source = read(source_files[family])
        for source_name, output_name in method_mappings[family]:
            for row in source:
                if row["method"] == source_name:
                    natural_runs.append({"family": family, "setting_id": row["setting_id"], "collection": row["collection"], "method": output_name, "accuracy": row["accuracy"], "selected_by_test_accuracy": False})
    claims = [
        {"claim": "families_selected", "value": len(selected)},
        {"claim": "selection_used_test_accuracy", "value": False},
        {"claim": "at_most_two_families_selected", "value": len(selected) <= 2},
        {"claim": "complete_new_family_gate_passed", "value": bool(selected)},
    ]
    write_csv(DEST / "natural_runs.csv", natural_runs, ["family", "setting_id", "collection", "method", "accuracy", "selected_by_test_accuracy"])
    write_csv(DEST / "natural_residuals.csv", rows); write_csv(DEST / "natural_nulls.csv", read(DEST / "full_model_nulls.csv") + read(DEST / "multiview_nulls.csv"))
    write_csv(DEST / "natural_stability.csv", read(DEST / "full_model_stability.csv") + read(DEST / "multiview_stability.csv")); write_csv(DEST / "natural_claims.csv", claims)
    (DEST / "natural_report.md").write_text(
        "# New realistic residual search\n\n"
        f"Execution commit: `{git_head()}`. Two new families were screened using only held-out pairwise fit, five-resample "
        f"rank reproducibility, and matched-null percentile; test accuracy was not used for selection. {len(selected)} "
        f"families passed the preregistered discovery filter. Selected families: {', '.join(sorted(selected)) if selected else 'none'}.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
