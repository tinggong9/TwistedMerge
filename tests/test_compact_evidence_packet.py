from __future__ import annotations

from experiments.compact_evidence_packet import read_json


def test_missing_json_has_safe_default(tmp_path, monkeypatch) -> None:
    import experiments.compact_evidence_packet as module

    monkeypatch.setattr(module, "OUT", tmp_path)
    assert read_json("missing.json") == {}
