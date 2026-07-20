from pathlib import Path

from experiments.freeze_model_lineage_data import compute_freeze_id, sha256_file, verify_rows


def frozen_row(path: Path, *, observed_path: Path | None = None) -> dict[str, object]:
    return {
        "scope": "test",
        "artifact_kind": "fixture",
        "seed": "",
        "node_or_family": "",
        "storage_class": "git_tracked",
        "logical_path": "payload.bin",
        "observed_path": str(observed_path or path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def test_freeze_id_is_independent_of_observed_absolute_path(tmp_path: Path) -> None:
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"frozen evidence")
    left = frozen_row(payload, observed_path=Path("/first/machine/payload.bin"))
    right = frozen_row(payload, observed_path=Path("/second/machine/payload.bin"))

    assert compute_freeze_id([left]) == compute_freeze_id([right])


def test_verifier_accepts_exact_payload_and_detects_mutation(tmp_path: Path) -> None:
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"frozen evidence")
    row = frozen_row(payload)

    assert verify_rows([row], root=tmp_path) == []
    payload.write_bytes(b"changed evidence")

    failures = verify_rows([row], root=tmp_path)
    assert len(failures) == 1
    assert failures[0]["error"] in {"size_mismatch", "sha256_mismatch"}
