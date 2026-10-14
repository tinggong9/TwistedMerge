import pytest

from src.kbs_revision.resource_guard import BudgetGuard, preflight


def test_software_accelerator_and_missing_caps_rejected(tmp_path):
    errors = preflight({"phase": "software_only", "device": "cuda", "resource_limits": {}}, tmp_path)
    assert any("CPU" in value for value in errors)
    assert any("ceiling" in value for value in errors)


def test_guard_counts_actual_output_bytes(tmp_path):
    guard = BudgetGuard({"resource_limits": {"cpu_seconds": 300, "disk_bytes": 100}, "device": "cpu"}, tmp_path)
    (tmp_path / "oversize").write_bytes(b"x" * 200)
    with pytest.raises(OSError, match="disk ceiling"):
        guard.check()
