import json

import pytest

from src.kbs_revision.neural_runner import EpisodeStore, StageProfiler, atomic_json, evaluate_summary, require_confirmatory_access


def test_atomic_publication_cannot_replace_a_receipt(tmp_path):
    path = tmp_path / "result.json"
    atomic_json(path, {"value": 1})
    with pytest.raises(FileExistsError):
        atomic_json(path, {"value": 2})
    assert json.loads(path.read_text()) == {"value": 1}


def test_resume_requires_same_sources_and_checksums(tmp_path):
    store = EpisodeStore(tmp_path, {"rank": 8}, inputs={"model_sha": "one"})
    assert store.execute("group1", lambda: {"metric": .5}) == ({"metric": .5}, False)
    assert store.execute("group1", lambda: pytest.fail("duplicate evaluation"), resume=True) == ({"metric": .5}, True)
    with pytest.raises(ValueError, match="differ"):
        EpisodeStore(tmp_path, {"rank": 8}, inputs={"model_sha": "two"})
    (tmp_path / "episodes/group1/result.json").write_text('{"metric":0.8}')
    with pytest.raises(ValueError, match="checksum"):
        store.completed("group1")


def test_failure_retained_and_can_retry_without_false_completion(tmp_path):
    store = EpisodeStore(tmp_path, {}, inputs={})
    def failed():
        raise RuntimeError("training failed")
    with pytest.raises(RuntimeError):
        store.execute("episode", failed)
    assert store.completed("episode") is None
    assert len(list((tmp_path / "failures").glob("*.json"))) == 1
    assert store.execute("episode", lambda: {"status": "completed"}, resume=True)[1] is False
    with pytest.raises(ValueError, match="safe identifier"):
        store.execute("../bad", lambda: {})


def test_stage_timer_reports_failed_stage_and_cpu_peak():
    timer = StageProfiler("cpu")
    with pytest.raises(RuntimeError):
        with timer.stage("inference"):
            raise RuntimeError("failed")
    assert timer.rows[0]["seconds"] >= 0
    assert timer.rows[0]["cpu_peak_rss_bytes"] > 0
    assert timer.rows[0]["gpu_peak_allocated_bytes"] is None
    assert timer.rows[0]["status"] == "failed"


def test_macro_error_is_equal_weight_by_task():
    rows = [{"task_id": "one", "bounded_loss": 1}] + [{"task_id": "two", "bounded_loss": 0}] * 3
    assert evaluate_summary(rows)["macro_loss"] == .5
    with pytest.raises(ValueError, match="no scored"):
        evaluate_summary([])


def test_final_labels_require_real_go_and_frozen_protocol(tmp_path):
    with pytest.raises(PermissionError, match="explicitly frozen"):
        require_confirmatory_access({"phase": "confirmatory"}, tmp_path, tmp_path)
    with pytest.raises(PermissionError, match="GO and a frozen"):
        require_confirmatory_access({"test_access": True}, tmp_path, tmp_path)


def test_dead_owner_lock_can_resume_same_episode(tmp_path):
    store = EpisodeStore(tmp_path, {}, inputs={})
    episodes = tmp_path / "episodes"
    episodes.mkdir()
    (episodes / ".case.lock").write_text(json.dumps({"pid": 999999999, "fingerprint": store.fingerprint}))
    assert store.execute("case", lambda: {"value": 1}, resume=True)[1] is False
    (tmp_path / "freeze_manifest.json").write_text(json.dumps({"pilot_decision": "GO", "software_only": True}))
    with pytest.raises(PermissionError, match="real pilot"):
        require_confirmatory_access({"test_access": True}, tmp_path, tmp_path)
