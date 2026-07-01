#!/usr/bin/env python3
"""Compact oversized fixed-setting verification CSV outputs.

The fixed-setting verifier can emit full pairwise/layerwise permutation maps in
per-method rows. Those JSON fields are useful for audit trails but can push the
main CSVs over GitHub's 100 MB blob limit. This utility moves the bulky fields
into deterministic gzip-compressed shards and leaves compact scalar CSVs in the
original expected locations.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import shutil
from pathlib import Path


RUN_BULKY_COLUMNS = (
    "pairwise_alignment_permutations_json",
    "layerwise_alignment_permutations_json",
)
TRIANGLE_BULKY_COLUMNS = (
    "pairwise_alignment_permutations_json",
    "layerwise_alignment_permutations_json",
    "p_ij",
    "p_jk",
    "p_ki",
    "triangle_perm",
)
KEY_COLUMNS = (
    "setting_id",
    "run_id",
    "dataset",
    "architecture",
    "n_models",
    "width",
    "domain_shift",
    "matching",
    "alignment_source",
    "seed",
    "method",
    "triangle",
)


def _open_gzip_csv(path: Path, fieldnames: list[str]) -> tuple[io.TextIOWrapper, csv.DictWriter]:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = path.open("wb")
    gz = gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0)
    text = io.TextIOWrapper(gz, encoding="utf-8", newline="")
    writer = csv.DictWriter(text, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    return text, writer


def compact_csv(
    source: Path,
    shard_dir: Path,
    shard_prefix: str,
    bulky_columns: tuple[str, ...],
    rows_per_shard: int,
) -> dict[str, object]:
    temp_source = source.with_suffix(source.suffix + ".compact_tmp")
    present_bulk = []
    writers: dict[int, tuple[io.TextIOWrapper, csv.DictWriter, Path, int]] = {}
    row_count = 0

    with source.open("r", encoding="utf-8", newline="") as src:
        reader = csv.DictReader(src)
        if reader.fieldnames is None:
            raise ValueError(f"{source} has no header")
        present_bulk = [column for column in bulky_columns if column in reader.fieldnames]
        if not present_bulk:
            return {
                "source_csv": str(source),
                "compact_csv": str(source),
                "rows": 0,
                "shard_count": 0,
                "columns_moved": "",
                "max_shard_bytes": 0,
            }
        slim_fields = [field for field in reader.fieldnames if field not in present_bulk]
        slim_fields.extend(["large_field_shard", "large_field_row"])
        shard_fields = [
            field
            for field in ("large_field_row", *KEY_COLUMNS, *present_bulk)
            if field == "large_field_row" or field in reader.fieldnames or field in present_bulk
        ]

        with temp_source.open("w", encoding="utf-8", newline="") as dst:
            slim_writer = csv.DictWriter(dst, fieldnames=slim_fields, lineterminator="\n")
            slim_writer.writeheader()
            for row_count, row in enumerate(reader, start=1):
                shard_id = (row_count - 1) // rows_per_shard
                if shard_id not in writers:
                    shard_path = shard_dir / f"{shard_prefix}_part_{shard_id:03d}.csv.gz"
                    writers[shard_id] = (*_open_gzip_csv(shard_path, shard_fields), shard_path, 0)
                text, shard_writer, shard_path, shard_rows = writers[shard_id]
                shard_row = {field: row.get(field, "") for field in shard_fields if field != "large_field_row"}
                shard_row["large_field_row"] = row_count
                shard_writer.writerow(shard_row)
                writers[shard_id] = (text, shard_writer, shard_path, shard_rows + 1)

                slim_row = {field: row.get(field, "") for field in slim_fields}
                slim_row["large_field_shard"] = f"{shard_prefix}_part_{shard_id:03d}.csv.gz"
                slim_row["large_field_row"] = row_count
                slim_writer.writerow(slim_row)

    for text, _writer, _path, _rows in writers.values():
        text.close()
    temp_source.replace(source)
    shard_sizes = [path.stat().st_size for _text, _writer, path, _rows in writers.values()]
    return {
        "source_csv": str(source),
        "compact_csv": str(source),
        "rows": row_count,
        "shard_count": len(writers),
        "columns_moved": ",".join(present_bulk),
        "max_shard_bytes": max(shard_sizes) if shard_sizes else 0,
    }


def write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["source_csv", "compact_csv", "rows", "shard_count", "columns_moved", "max_shard_bytes"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-dir", type=Path, default=Path("reports/csv"))
    parser.add_argument("--rows-per-shard", type=int, default=2000)
    args = parser.parse_args()

    csv_dir = args.csv_dir
    shard_dir = csv_dir / "fixed_setting_large_artifacts"
    shard_dir.mkdir(parents=True, exist_ok=True)

    # Remove stale deterministic shards from prior compaction attempts.
    for old_shard in shard_dir.glob("fixed_setting_*_part_*.csv.gz"):
        old_shard.unlink()

    manifest_rows = []
    runs_csv = csv_dir / "fixed_setting_verification_runs.csv"
    triangles_csv = csv_dir / "fixed_setting_triangle_defects.csv"
    manifest_rows.append(
        compact_csv(runs_csv, shard_dir, "fixed_setting_runs_maps", RUN_BULKY_COLUMNS, args.rows_per_shard)
    )
    manifest_rows.append(
        compact_csv(
            triangles_csv,
            shard_dir,
            "fixed_setting_triangle_maps",
            TRIANGLE_BULKY_COLUMNS,
            args.rows_per_shard,
        )
    )

    real_runs = csv_dir / "real_obstruction_degradation.csv"
    real_triangles = csv_dir / "real_obstruction_triangle_defects.csv"
    shutil.copyfile(runs_csv, real_runs)
    shutil.copyfile(triangles_csv, real_triangles)
    manifest_rows.append({**manifest_rows[0], "source_csv": str(real_runs), "compact_csv": str(real_runs)})
    manifest_rows.append({**manifest_rows[1], "source_csv": str(real_triangles), "compact_csv": str(real_triangles)})
    write_manifest(csv_dir / "fixed_setting_large_artifacts_manifest.csv", manifest_rows)

    for row in manifest_rows:
        print(
            f"compacted {row['source_csv']} rows={row['rows']} "
            f"shards={row['shard_count']} max_shard_bytes={row['max_shard_bytes']}"
        )


if __name__ == "__main__":
    main()
