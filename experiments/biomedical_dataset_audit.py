#!/usr/bin/env python3
"""B0: canonical source, terms, masks, metadata, and split audit."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.fetch_kvasir_subset import (  # noqa: E402
    CANONICAL_ARCHIVE_SHA256,
    CANONICAL_SOURCE,
    REVISION,
)
from experiments.spatial_output_common import (  # noqa: E402
    DATA,
    OUT,
    dataset_checksum,
    dataset_counts,
    dataset_paths,
    dataset_ready,
    ensure_dirs,
    factual_report,
    record_command,
    stage_complete,
    update_status,
    utc_now,
    write_csv,
)

SCRIPT = Path(__file__).resolve()
DEST = OUT / "data"
COMMAND = "python experiments/biomedical_dataset_audit.py"


def _resolution_summary() -> tuple[str, str]:
    widths, heights = [], []
    for split in ("train", "validation", "test"):
        for image, _ in dataset_paths(split):
            with Image.open(image) as handle:
                widths.append(handle.width)
                heights.append(handle.height)
    return f"{min(widths)}x{min(heights)}", f"{max(widths)}x{max(heights)}"


def run() -> dict[str, object]:
    ensure_dirs()
    ready = dataset_ready()
    counts = dataset_counts()
    minimum, maximum = _resolution_summary() if ready else ("", "")
    manifest_path = DATA / "download_manifest.csv"
    checksum = dataset_checksum() if ready else ""
    manifest_rows = [
        {
            "selection": "primary",
            "dataset": "Kvasir-SEG",
            "task": "gastrointestinal_polyp_binary_segmentation",
            "canonical_source": CANONICAL_SOURCE,
            "resolved_source": "https://huggingface.co/datasets/MedOtter/kvasir-seg",
            "revision_or_checksum": REVISION,
            "canonical_archive_sha256": CANONICAL_ARCHIVE_SHA256,
            "usage_terms": "research_and_educational_use; citation_required; commercial_use_requires_permission",
            "license_verified": True,
            "local_subset_checksum": checksum,
            "image_count": sum(counts.values()),
            "mask_count": sum(counts.values()),
            "resolution_min": minimum,
            "resolution_max": maximum,
            "center_site_metadata": "absent",
            "tissue_domain_metadata": "polyp_only; no per-image tissue domain labels",
            "split": f"train={counts['train']};validation={counts['validation']};test={counts['test']}",
            "patient_level_separation": "not_possible; patient identifiers absent",
            "state": "selected" if ready else "blocked",
        },
        {
            "selection": "secondary",
            "dataset": "KvasirCapsule-SEG",
            "task": "capsule_endoscopy_polyp_segmentation",
            "canonical_source": "https://datasets.simula.no/kvasir-capsule-seg/",
            "resolved_source": "canonical archive listed as kvasir-capsule-seg.zip",
            "revision_or_checksum": "not_downloaded",
            "canonical_archive_sha256": "not_resolved",
            "usage_terms": "research_and_educational_use; citation_required; commercial_use_requires_permission",
            "license_verified": True,
            "local_subset_checksum": "",
            "image_count": 55,
            "mask_count": 55,
            "resolution_min": "",
            "resolution_max": "",
            "center_site_metadata": "absent",
            "tissue_domain_metadata": "polyp_only",
            "split": "not_resolved",
            "patient_level_separation": "not_verified",
            "state": "candidate; download deferred until E1 gate",
        },
    ]
    write_csv(DEST / "dataset_manifest.csv", manifest_rows)
    facts = [
        f"Primary selection: Kvasir-SEG; local paired subset counts {counts}.",
        f"Pinned mirror revision: {REVISION}.",
        f"Local subset checksum: {checksum or 'unavailable'}.",
        "Canonical terms restrict use to research and education, require citation, and require permission for commercial use.",
        "No center, site, scanner, institution, patient, or per-image tissue-domain metadata is present; the dataset is not called multi-center.",
        "The fixed mirror split is used; patient-level separation cannot be verified because patient identifiers are absent.",
        "Secondary candidate KvasirCapsule-SEG is recorded but is not used unless the E1 gate opens and its archive is resolved.",
    ]
    factual_report(DEST / "dataset_report.md", "Biomedical dataset and terms audit", facts)
    state = "completed" if ready and manifest_path.exists() else "blocked"
    update_status("B0_dataset_audit", state, facts[0])
    stage_complete(DEST / "dataset_manifest.csv", {"stage": "B0", "state": state, "counts": counts, "checksum": checksum})
    return {"state": state, "counts": counts, "checksum": checksum}


def main() -> None:
    argparse.ArgumentParser().parse_args()
    started_at, started = utc_now(), time.perf_counter()
    try:
        result = run()
    except Exception as error:
        update_status("B0_dataset_audit", "failed", str(error))
        record_command(command=COMMAND, source=SCRIPT, seed_scope="none", dataset_revision=REVISION, started_at=started_at, runtime=time.perf_counter()-started, exit_code=1, state="failed", summary=str(error))
        raise
    record_command(command=COMMAND, source=SCRIPT, seed_scope="none", dataset_revision=REVISION, started_at=started_at, runtime=time.perf_counter()-started, exit_code=0, state=str(result["state"]), summary=f"audited primary counts {result['counts']}")


if __name__ == "__main__":
    main()
