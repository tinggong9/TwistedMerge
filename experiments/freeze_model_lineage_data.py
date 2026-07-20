#!/usr/bin/env python3
"""Create or verify the model-lineage holonomy data freeze.

The freeze is a content-addressed ledger. Large datasets, checkpoints, logits,
and caches remain outside Git; the committed ledger binds each one to its exact
path, byte count, and SHA-256 digest.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = ROOT / "reports" / "model_lineage_holonomy"
FREEZE_ROOT = REPORT_ROOT / "data_freeze"
ARTIFACT_MANIFEST = REPORT_ROOT / "artifact_manifest.csv"

MANIFEST_FIELDS = (
    "freeze_id",
    "scope",
    "artifact_kind",
    "seed",
    "node_or_family",
    "storage_class",
    "logical_path",
    "observed_path",
    "sha256",
    "bytes",
)

CODE_INPUTS = (
    "experiments/freeze_model_lineage_data.py",
    "experiments/model_lineage_holonomy.py",
    "src/holonomy_application_corpus.py",
    "src/lineage_merge_audit.py",
    "src/lineage_transport_sync.py",
    "src/model_lineage_holonomy.py",
    "tests/test_lineage_merge_audit.py",
    "tests/test_lineage_transport_sync.py",
    "tests/test_model_lineage_holonomy.py",
    "tests/test_model_lineage_data_freeze.py",
    "requirements.txt",
    "requirements-benchmarks.txt",
    "requirements-synthetic.txt",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
    ).strip()


def is_git_tracked(path: Path) -> bool:
    try:
        relative = path.resolve().relative_to(ROOT.resolve())
    except ValueError:
        return False
    return subprocess.run(
        ["git", "ls-files", "--error-unmatch", str(relative)],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[Mapping[str, object]], fields: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _path_from_manifest(raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else ROOT / path


def _repo_logical_path(path: Path) -> str | None:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return None


def make_entry(
    path: Path,
    *,
    scope: str,
    artifact_kind: str,
    seed: str = "",
    node_or_family: str = "",
    logical_path: str | None = None,
    expected_sha256: str | None = None,
    expected_bytes: int | None = None,
) -> dict[str, object]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"freeze input is missing: {path}")
    actual_bytes = path.stat().st_size
    actual_sha256 = sha256_file(path)
    if expected_bytes is not None and actual_bytes != expected_bytes:
        raise RuntimeError(
            f"size mismatch for {path}: expected {expected_bytes}, found {actual_bytes}"
        )
    if expected_sha256 and actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"SHA-256 mismatch for {path}: expected {expected_sha256}, found {actual_sha256}"
        )
    repository_path = _repo_logical_path(path)
    storage_class = (
        "git_tracked"
        if is_git_tracked(path)
        else "ignored_local"
        if repository_path is not None
        else "external_input"
    )
    return {
        "scope": scope,
        "artifact_kind": artifact_kind,
        "seed": seed,
        "node_or_family": node_or_family,
        "storage_class": storage_class,
        "logical_path": logical_path or repository_path or path.name,
        "observed_path": str(path),
        "sha256": actual_sha256,
        "bytes": actual_bytes,
    }


def _add_unique(entries: dict[str, dict[str, object]], entry: dict[str, object]) -> None:
    logical_path = str(entry["logical_path"])
    prior = entries.get(logical_path)
    if prior is None:
        entries[logical_path] = entry
        return
    identity = (entry["sha256"], int(entry["bytes"]))
    prior_identity = (prior["sha256"], int(prior["bytes"]))
    if identity != prior_identity:
        raise RuntimeError(f"conflicting frozen identities for {logical_path}")


def collect_entries() -> list[dict[str, object]]:
    if not ARTIFACT_MANIFEST.is_file():
        raise FileNotFoundError(f"missing evidence manifest: {ARTIFACT_MANIFEST}")
    entries: dict[str, dict[str, object]] = {}

    for row in read_csv(ARTIFACT_MANIFEST):
        path = _path_from_manifest(row["path"])
        repository_path = _repo_logical_path(path)
        logical_path = repository_path or f"external/evidence/{path.name}"
        _add_unique(
            entries,
            make_entry(
                path,
                scope="experiment_evidence",
                artifact_kind=row["artifact_kind"],
                seed=row.get("seed", ""),
                node_or_family=row.get("node_or_family", ""),
                logical_path=logical_path,
                expected_sha256=row["sha256"],
                expected_bytes=int(row["bytes"]),
            ),
        )

    _add_unique(
        entries,
        make_entry(
            ARTIFACT_MANIFEST,
            scope="control_record",
            artifact_kind="artifact_manifest",
        ),
    )

    for relative in CODE_INPUTS:
        _add_unique(
            entries,
            make_entry(
                ROOT / relative,
                scope="execution_environment",
                artifact_kind="code_or_requirement",
            ),
        )

    config = json.loads((REPORT_ROOT / "config.json").read_text(encoding="utf-8"))
    dataset_archive = Path(config["dataset_archive_path"])
    dataset_root = dataset_archive.parent / "cifar-10-batches-py"
    _add_unique(
        entries,
        make_entry(
            dataset_archive,
            scope="raw_input",
            artifact_kind="dataset_archive",
            logical_path=f"external/cifar10/{dataset_archive.name}",
            expected_sha256=config["dataset_archive_sha256"],
        ),
    )
    for path in sorted(dataset_root.iterdir()):
        if path.is_file():
            _add_unique(
                entries,
                make_entry(
                    path,
                    scope="raw_input",
                    artifact_kind="dataset_extracted_file",
                    logical_path=f"external/cifar10/cifar-10-batches-py/{path.name}",
                ),
            )

    encoder_weights = Path(config["encoder_weights_path"])
    _add_unique(
        entries,
        make_entry(
            encoder_weights,
            scope="raw_input",
            artifact_kind="pretrained_encoder_weights",
            logical_path=f"external/torch-hub/{encoder_weights.name}",
            expected_sha256=config["encoder_weights_sha256"],
        ),
    )

    lineage_rows = read_csv(REPORT_ROOT / "lineage_manifest.csv")
    source_rows = {
        row["source_m0_path"]: row["source_m0_sha256"]
        for row in lineage_rows
        if row.get("source_m0_path")
    }
    if not source_rows:
        raise RuntimeError("lineage manifest contains no source M0 checkpoints")
    source_parent: Path | None = None
    for raw_path, expected_hash in sorted(source_rows.items()):
        path = Path(raw_path)
        source_parent = path.parent
        seed = path.stem.removeprefix("adapter_seed_")
        _add_unique(
            entries,
            make_entry(
                path,
                scope="source_corpus",
                artifact_kind="source_m0_checkpoint",
                seed=seed,
                node_or_family="M0",
                logical_path=f"external/holonomy-source/shared_corpus_confirmatory/{path.name}",
                expected_sha256=expected_hash,
            ),
        )
    assert source_parent is not None
    source_features = source_parent / "projected_features.pt"
    _add_unique(
        entries,
        make_entry(
            source_features,
            scope="source_corpus",
            artifact_kind="source_projection_cache",
            logical_path=(
                "external/holonomy-source/shared_corpus_confirmatory/projected_features.pt"
            ),
            expected_sha256=config["source_feature_cache_sha256"],
        ),
    )

    return sorted(entries.values(), key=lambda row: str(row["logical_path"]))


def compute_freeze_id(rows: Iterable[Mapping[str, object]]) -> str:
    identity_rows = [
        {
            key: int(row[key]) if key == "bytes" else str(row[key])
            for key in (
                "scope",
                "artifact_kind",
                "seed",
                "node_or_family",
                "storage_class",
                "logical_path",
                "sha256",
                "bytes",
            )
        }
        for row in rows
    ]
    payload = json.dumps(identity_rows, sort_keys=True, separators=(",", ":")).encode()
    return f"tm-mlh-v1-{hashlib.sha256(payload).hexdigest()[:16]}"


def resolve_frozen_path(row: Mapping[str, str], root: Path = ROOT) -> Path:
    logical_path = row["logical_path"]
    observed_path = Path(row["observed_path"])
    if row["storage_class"] in {"git_tracked", "ignored_local"}:
        candidate = root / logical_path
        if candidate.is_file():
            return candidate
    return observed_path


def verify_rows(
    rows: Iterable[Mapping[str, str]], root: Path = ROOT
) -> list[dict[str, object]]:
    failures: list[dict[str, object]] = []
    for row in rows:
        path = resolve_frozen_path(row, root)
        if not path.is_file():
            failures.append({"logical_path": row["logical_path"], "error": "missing"})
            continue
        actual_bytes = path.stat().st_size
        if actual_bytes != int(row["bytes"]):
            failures.append(
                {
                    "logical_path": row["logical_path"],
                    "error": "size_mismatch",
                    "expected": int(row["bytes"]),
                    "actual": actual_bytes,
                }
            )
            continue
        actual_hash = sha256_file(path)
        if actual_hash != row["sha256"]:
            failures.append(
                {
                    "logical_path": row["logical_path"],
                    "error": "sha256_mismatch",
                    "expected": row["sha256"],
                    "actual": actual_hash,
                }
            )
    return failures


def verify_package() -> list[dict[str, object]]:
    package_path = FREEZE_ROOT / "package_manifest.csv"
    if not package_path.is_file():
        return [{"logical_path": str(package_path), "error": "missing"}]
    failures = []
    for row in read_csv(package_path):
        path = ROOT / row["path"]
        if not path.is_file():
            failures.append({"logical_path": row["path"], "error": "missing"})
        elif path.stat().st_size != int(row["bytes"]):
            failures.append({"logical_path": row["path"], "error": "size_mismatch"})
        elif sha256_file(path) != row["sha256"]:
            failures.append({"logical_path": row["path"], "error": "sha256_mismatch"})
    return failures


def create_freeze() -> tuple[str, int, int]:
    rows = collect_entries()
    freeze_id = compute_freeze_id(rows)
    frozen_rows = [{"freeze_id": freeze_id, **row} for row in rows]
    FREEZE_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_path = FREEZE_ROOT / "freeze_manifest.csv"
    write_csv(manifest_path, frozen_rows, MANIFEST_FIELDS)

    class_counts = Counter(str(row["storage_class"]) for row in rows)
    scope_counts = Counter(str(row["scope"]) for row in rows)
    total_bytes = sum(int(row["bytes"]) for row in rows)
    metadata = {
        "schema_version": 1,
        "freeze_id": freeze_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "model_lineage_holonomy",
        "evidence_label": "natural_model_lineage",
        "source_branch": git_output("branch", "--show-current"),
        "source_snapshot_commit": git_output("rev-parse", "HEAD"),
        "execution_commit": json.loads((REPORT_ROOT / "config.json").read_text())["execution_commit"],
        "entry_count": len(rows),
        "referenced_bytes": total_bytes,
        "storage_class_counts": dict(sorted(class_counts.items())),
        "scope_counts": dict(sorted(scope_counts.items())),
        "scientific_gate_status": {"H1": False, "H2": False, "H3": False, "H4": False},
        "payload_policy": "large payloads remain outside Git and are bound by SHA-256",
    }
    metadata_path = FREEZE_ROOT / "freeze_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    readme = f"""# Model-lineage holonomy data freeze

Freeze ID: `{freeze_id}`

This directory locks the completed natural model-lineage holonomy experiment to
`{len(rows)}` exact files (`{total_bytes}` referenced bytes). Every input,
checkpoint, representation, pre-label logit bundle, transport bundle, split,
and committed result is identified by byte size and SHA-256 in
`freeze_manifest.csv`.

The freeze deliberately does not commit the large binary payloads to Git.
`git_tracked` records are stored normally; `ignored_local` and `external_input`
records remain at the paths recorded in the manifest. This is an integrity
freeze, not a remote backup of those payloads.

The scientific outcome is also frozen without reinterpretation: H1, H2, H3,
and H4 all failed their preregistered gates.

Verify the complete freeze from the repository root with:

```bash
python3 experiments/freeze_model_lineage_data.py --verify-only
```

Recreating the freeze is intentionally separate and overwrites only this
directory's generated ledgers:

```bash
python3 experiments/freeze_model_lineage_data.py --create
```
"""
    readme_path = FREEZE_ROOT / "README.md"
    readme_path.write_text(readme, encoding="utf-8")

    package_rows = []
    for path in (manifest_path, metadata_path, readme_path):
        package_rows.append(
            {
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    write_csv(FREEZE_ROOT / "package_manifest.csv", package_rows, ("path", "sha256", "bytes"))

    failures = verify_rows(read_csv(manifest_path)) + verify_package()
    verification = {
        "freeze_id": freeze_id,
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if not failures else "failed",
        "verified_entries": len(rows),
        "verified_referenced_bytes": total_bytes,
        "package_entries": len(package_rows),
        "failures": failures,
    }
    (FREEZE_ROOT / "verification.json").write_text(
        json.dumps(verification, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if failures:
        raise RuntimeError(f"freeze verification failed: {failures[:3]}")
    return freeze_id, len(rows), total_bytes


def verify_freeze() -> tuple[str, int, int]:
    manifest_path = FREEZE_ROOT / "freeze_manifest.csv"
    rows = read_csv(manifest_path)
    if not rows:
        raise RuntimeError("freeze manifest is empty")
    freeze_ids = {row["freeze_id"] for row in rows}
    if len(freeze_ids) != 1:
        raise RuntimeError(f"freeze manifest contains multiple IDs: {freeze_ids}")
    expected_id = compute_freeze_id(rows)
    freeze_id = next(iter(freeze_ids))
    failures = verify_rows(rows) + verify_package()
    if freeze_id != expected_id:
        failures.append(
            {"logical_path": "freeze_manifest.csv", "error": "freeze_id_mismatch"}
        )
    total_bytes = sum(int(row["bytes"]) for row in rows)
    if failures:
        raise RuntimeError(f"freeze verification failed: {failures[:3]}")
    return freeze_id, len(rows), total_bytes


def main() -> None:
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--create", action="store_true")
    action.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    freeze_id, entry_count, total_bytes = create_freeze() if args.create else verify_freeze()
    print(
        f"freeze {freeze_id}: verified {entry_count} entries / {total_bytes} referenced bytes"
    )


if __name__ == "__main__":
    main()
