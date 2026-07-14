import json
from pathlib import Path

import pytest

from src.evidence_provenance import execution_command, execution_commit


def test_execution_commit_is_read_from_artifact_config(tmp_path: Path):
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps({"execution_commit": "a" * 40, "exact_command": "python run.py"}),
        encoding="utf-8",
    )
    assert execution_commit(path) == "a" * 40
    assert execution_command(path) == "python run.py"


def test_execution_commit_rejects_missing_global_fallback(tmp_path: Path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"command": "python run.py"}), encoding="utf-8")
    with pytest.raises(ValueError, match="per-artifact execution commit"):
        execution_commit(path)
