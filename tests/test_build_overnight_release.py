from __future__ import annotations

from experiments.build_overnight_release import scan_forbidden_eligible_artifacts


def test_forbidden_scan_ignores_ineligible_and_flags_eligible(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("experiments.build_overnight_release.ROOT", tmp_path)
    path = tmp_path / "artifact.csv"
    path.write_text("target_accuracy_for_method\n", encoding="utf-8")
    entry = {"id": "x", "paper_eligibility": False, "config": "artifact.csv", "raw_csv": None, "summary_csv": None, "latex_table": None}
    assert scan_forbidden_eligible_artifacts([entry]) == []
    entry["paper_eligibility"] = True
    assert scan_forbidden_eligible_artifacts([entry])[0]["token"] == "target_accuracy_for_method"
