from pathlib import Path

from experiments.next_program_common import OUT
from experiments.reproducibility_release import checkpoint_manifest, stage_for


def test_table_and_plot_prefixes_map_to_stage_scripts():
    assert stage_for(Path("reports/next_program/immediate/tables/chart_main.tex"))[:2] == ("A1", "experiments/trained_chart_inference.py")
    assert stage_for(Path("reports/next_program/immediate/tables/cost.tex"))[0] == "A2"
    assert stage_for(Path("reports/next_program/immediate/tables/refinement.tex"))[0] == "A3"
    assert stage_for(Path("reports/next_program/iclr/plots/full_model_residuals.pdf"))[0] == "B1"
    assert stage_for(Path("reports/next_program/iclr/tables/composition.tex"))[0] == "B2"
    assert stage_for(Path("reports/next_program/iclr/tables/multiview.tex"))[0] == "B3"
    assert stage_for(Path("reports/next_program/iclr/tables/compression.tex"))[0] == "B5"
    assert stage_for(Path("reports/next_program/extended/scaling_runs.csv"))[0] == "C7"


def test_checkpoint_manifest_has_stable_schema_when_cache_is_empty_or_populated():
    for row in checkpoint_manifest():
        assert set(row) == {"artifact", "artifact_type", "bytes", "sha256"}
        assert len(row["sha256"]) == 64


def test_all_tier_table_and_plot_directories_exist():
    for tier in ("immediate", "iclr", "extended"):
        assert (OUT / tier / "tables").is_dir()
        assert (OUT / tier / "plots").is_dir()
