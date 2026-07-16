from experiments.run_spatial_output_program import STAGES, stages_for


def test_runner_tiers_and_explicit_stage_are_bounded():
    assert stages_for("sanity", None) == ["S1", "S2", "S3"]
    assert stages_for("confirmation", None) == ["E1", "E2"]
    assert stages_for("all", "b1") == ["B1"]
    assert stages_for("all", None)[-1] == "Z0"
    assert set(stages_for("all", None)) == set(STAGES)
