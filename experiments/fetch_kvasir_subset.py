#!/usr/bin/env python3
"""Fetch a bounded, checksum-verified subset of the pinned Kvasir-SEG mirror."""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
import subprocess
import tempfile
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "Kvasir-SEG-subset"
REPO = "https://huggingface.co/datasets/MedOtter/kvasir-seg"
REVISION = "54a220cf383e341b4622a5d89dc1cf3b630902a7"
CANONICAL_SOURCE = "https://datasets.simula.no/kvasir-seg/"
CANONICAL_ARCHIVE_SHA256 = "03b30e21d584e04facf49397a2576738fd626815771afbbf788f74a7153478f7"


def git_output(repo: Path, *arguments: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *arguments], check=True, capture_output=True, text=True).stdout


def pointer(repo: Path, path: str) -> tuple[str, int]:
    text = git_output(repo, "show", f"{REVISION}:{path}")
    lines = dict(line.split(" ", 1) for line in text.strip().splitlines()[1:])
    return lines["oid"].removeprefix("sha256:"), int(lines["size"])


def selected(repo: Path, counts: dict[str, int]) -> list[dict[str, object]]:
    paths = git_output(repo, "ls-tree", "-r", "--name-only", REVISION).splitlines()
    rows = []
    for split, count in counts.items():
        images = sorted(path for path in paths if path.startswith(f"{split}/images/") and path.endswith(".png"))[:count]
        for image_path in images:
            mask_path = image_path.replace(f"{split}/images/", f"{split}/masks/")
            for role, path in (("image", image_path), ("mask", mask_path)):
                digest, size = pointer(repo, path)
                rows.append({"split": split, "role": role, "path": path, "sha256": digest, "bytes": size})
    return rows


def download(row: dict[str, object], force: bool) -> dict[str, object]:
    relative = Path(str(row["path"]))
    destination = DEST / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected = str(row["sha256"])
    if destination.exists() and not force:
        actual = hashlib.sha256(destination.read_bytes()).hexdigest()
        if actual == expected:
            return row
    url = f"{REPO}/resolve/{REVISION}/{urllib.parse.quote(relative.as_posix())}"
    request = urllib.request.Request(url, headers={"User-Agent": "TwistedMerge-spatial-output/1.0"})
    temporary = destination.with_suffix(destination.suffix + ".part")
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    actual = hashlib.sha256(temporary.read_bytes()).hexdigest()
    if actual != expected or temporary.stat().st_size != int(row["bytes"]):
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"checksum mismatch for {relative}: {actual} != {expected}")
    temporary.replace(destination)
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    counts = {"train": 24, "validation": 8, "test": 8} if args.smoke else {"train": 160, "validation": 40, "test": 40}
    with tempfile.TemporaryDirectory(prefix="kvasir-metadata-") as temporary:
        repo = Path(temporary) / "repo"
        subprocess.run(["git", "clone", "--depth", "1", "--no-checkout", REPO, str(repo)], check=True)
        if git_output(repo, "rev-parse", "HEAD").strip() != REVISION:
            raise RuntimeError("mirror revision moved away from the pinned commit")
        rows = selected(repo, counts)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(download, row, args.force) for row in rows]
        for index, future in enumerate(as_completed(futures), 1):
            row = future.result()
            if index % 40 == 0 or index == len(futures):
                print(f"verified {index}/{len(futures)} files; latest={row['path']}", flush=True)
    manifest = DEST / "download_manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "role", "path", "sha256", "bytes"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: str(row["path"])))
    print(f"dataset_root={DEST}")
    print(f"mirror_revision={REVISION}")
    print(f"canonical_source={CANONICAL_SOURCE}")
    print(f"canonical_archive_sha256={CANONICAL_ARCHIVE_SHA256}")


if __name__ == "__main__":
    main()
