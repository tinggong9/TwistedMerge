"""Environment receipts and non-destructive artifact writes."""
from __future__ import annotations

import datetime as dt
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import psutil
import torch

from .protocol import file_hash


def git(root, *args):
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def environment(root):
    root = Path(root)
    memory = psutil.virtual_memory()
    packages = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}
    accelerators = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            accelerators.append({"kind": "cuda", "name": props.name, "memory_bytes": props.total_memory})
    if torch.backends.mps.is_available():
        accelerators.append({"kind": "mps", "name": "Apple integrated GPU", "shared_memory": True,
                             "recommended_memory_bytes": torch.mps.recommended_max_memory()})
    executed_files = [root / "experiments/kbs_revision.py"] + sorted((root / "tests").glob("test_kbs_revision_*.py"))
    executed_files += [p for directory in ("src/kbs_revision", "configs/kbs_revision")
                       for p in sorted((root / directory).rglob("*")) if p.is_file() and "__pycache__" not in str(p)]
    return {"utc": dt.datetime.now(dt.timezone.utc).isoformat(), "repository": str(root),
            "branch": git(root, "branch", "--show-current"), "source_commit": git(root, "rev-parse", "HEAD"),
            "working_tree": git(root, "status", "--porcelain"),
            "python": sys.version, "python_executable": sys.executable, "argv": [sys.executable, *sys.argv], "platform": platform.platform(),
            "torch": torch.__version__, "cuda": torch.version.cuda, "accelerator_count": len(accelerators),
            "accelerators": accelerators, "cpu": platform.processor(), "cpu_count": psutil.cpu_count(),
            "ram_total_bytes": memory.total, "ram_available_bytes": memory.available,
            "disk": dict(zip(("total", "used", "free"), shutil.disk_usage(root))), "packages": packages,
            "code_hashes": {str(p.relative_to(root)): file_hash(p) for p in executed_files},
            "network_model_downloads": False, "remote_push": False}


def write_once(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n" if not isinstance(value, str) else value
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)


def manifest(out):
    import csv
    out = Path(out)
    rows = [{"path": str(p.relative_to(out)), "bytes": p.stat().st_size, "sha256": file_hash(p)}
            for p in sorted(out.rglob("*")) if p.is_file() and p.name != "manifest.csv"]
    with (out / "manifest.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("path", "bytes", "sha256"))
        writer.writeheader()
        writer.writerows(rows)
