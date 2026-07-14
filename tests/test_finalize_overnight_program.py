from __future__ import annotations

from experiments.finalize_overnight_program import manifest_entries


def test_final_manifest_covers_every_stage_once() -> None:
    entries = manifest_entries()
    assert [entry["stage"] for entry in entries] == list(range(13))
    assert all(len(entry["actual_execution_commit"]) == 40 for entry in entries)
