import json
from pathlib import Path

import pytest

from src.kbs_revision.protocol import ROLES, canonical_bytes, check_frozen, file_hash, freeze, readiness_errors, split_errors
from src.kbs_revision.policies import Candidate, Evidence, GATES, decide, gate_controls


def test_freeze_refuses_missing_prerequisites(tmp_path):
    with pytest.raises(ValueError, match="unresolved base model revision"):
        freeze({}, tmp_path, tmp_path / "run")
    assert not (tmp_path / "run").exists()


def test_split_guard_hash_content_and_ids(tmp_path):
    splits = {"near_duplicate_audit": {"passed": True}}
    for index, role in enumerate(ROLES):
        path = tmp_path / f"{role}.jsonl"
        path.write_text(json.dumps({"task_id": "task", "example_id": index, "normalized_text_sha256": f"{index:064x}"}) + "\n")
        splits[role] = {"path": path.name, "sha256": file_hash(path)}
    assert split_errors(splits, tmp_path) == []
    splits["D_test"] = splits["train"]
    assert any("duplicate" in error for error in split_errors(splits, tmp_path))
    (tmp_path / "train.jsonl").write_text("{}\n")
    assert any("hash-mismatched" in error for error in split_errors(splits, tmp_path))


def test_run_refuses_tampered_or_different_configuration(tmp_path):
    config = {"revision": 1}
    frozen = tmp_path / "preregistration.yaml"
    frozen.write_bytes(canonical_bytes(config))
    (tmp_path / "preregistration.sha256").write_text(file_hash(frozen))
    supplied = tmp_path / "supplied.yaml"
    supplied.write_bytes(canonical_bytes(config))
    assert check_frozen(supplied, tmp_path)[0] == config
    supplied.write_text('{"revision":2}')
    with pytest.raises(ValueError, match="hash mismatch"):
        check_frozen(supplied, tmp_path)


def test_matched_hodge_and_vector_policy_agree_and_oracle_forbidden():
    candidates = [Candidate("fallback", .3), Candidate("structural", .1, True)]
    for missing in (None, *GATES):
        evidence = Evidence("vector", tuple((g, g != missing) for g in GATES), .01, 0, None, .01, .01)
        assert decide("P3G", candidates, "fallback", evidence)["selected_action"] == decide("P4", candidates, "fallback", evidence)["selected_action"]
    with pytest.raises(ValueError, match="evaluation-only"):
        decide("ORACLE", candidates, "fallback", Evidence())


def test_approximate_zero_never_implies_exact_zero():
    evidence = Evidence("mu2", tuple((g, True) for g in GATES), .001, 0, None)
    decision = decide("P4", [Candidate("fallback", .2), Candidate("structural", .1, True)], "fallback", evidence)
    assert decision["repairable_to_tolerance"]
    assert decision["exact_trivial"] is None
    assert decision["selected_action"] == "fallback"


def test_controls_distinguish_closure_and_non_neural_metrics():
    rows = gate_controls()
    assert all(not row["task_metric_measured"] for row in rows)
    assert any(row["populated_K3"] for row in rows)
