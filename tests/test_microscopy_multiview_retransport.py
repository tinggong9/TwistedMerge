from experiments.microscopy_multiview_retransport import run


def test_microscopy_stage_blocks_without_real_archive():
    assert run(smoke=True)["state"] == "blocked"
