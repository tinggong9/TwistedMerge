from __future__ import annotations

import subprocess
import sys

from experiments.overnight_stage0_audit import run_tests


def test_stage0_uses_active_python_interpreter(monkeypatch) -> None:
    seen = {}

    def fake_run(command, **kwargs):
        seen["command"] = command
        return subprocess.CompletedProcess(command, 0, stdout="1 passed\n", stderr="")

    monkeypatch.setattr("experiments.overnight_stage0_audit.subprocess.run", fake_run)
    command, result = run_tests()
    assert seen["command"][0] == sys.executable
    assert sys.executable in command
    assert result == "1 passed"
