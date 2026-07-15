#!/usr/bin/env python3
"""Stage 0: minimal environment, test, data, and provenance record."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.compact_benchmark_common import DATA, OUT, ensure_dirs, git_branch, git_head, sha256_file, write_csv, write_json


def dataset_record(name: str, directory: Path, source: str, license_name: str) -> dict[str, object]:
    files = sorted(path for path in directory.rglob("*") if path.is_file()) if directory.exists() else []
    return {
        "resource": name,
        "status": "available" if files else "missing",
        "source": source,
        "license": license_name,
        "file_count": len(files),
        "bytes": sum(path.stat().st_size for path in files),
        "aggregate_sha256": __import__("hashlib").sha256("".join(sha256_file(path) for path in files).encode()).hexdigest() if files else "",
        "attempts": 0 if files else 1,
        "last_error": "" if files else "not present in local cache",
    }


def main() -> None:
    ensure_dirs()
    packages = {}
    for package in ["numpy", "pandas", "matplotlib", "torch", "torchvision", "scipy", "scikit-learn", "timm", "transformers", "datasets", "peft", "accelerate", "einops", "sentencepiece"]:
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    memory_bytes = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip())
    environment = {
        "branch": git_branch(),
        "head": git_head(),
        "worktree": "<repository-root>",
        "python": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "mps_available": bool(torch.backends.mps.is_available()),
        "cuda_available": bool(torch.cuda.is_available()),
        "physical_memory_bytes": memory_bytes,
        "free_disk_bytes": shutil.disk_usage(ROOT).free,
        "packages": packages,
        "environment_variables_recorded": ["PYTHONHASHSEED"],
        "pythonhashseed": os.environ.get("PYTHONHASHSEED", "unset"),
    }
    write_json(OUT / "stage0_environment.json", environment)
    test_command = [sys.executable, "-m", "pytest", "-q"]
    result = subprocess.run(test_command, cwd=ROOT, text=True, capture_output=True)
    (OUT / "stage0_tests.txt").write_text(
        "$ python -m pytest -q\n" + result.stdout + result.stderr,
        encoding="utf-8",
    )
    if result.returncode:
        raise SystemExit(result.returncode)
    downloads = [
        dataset_record("MNIST", DATA / "MNIST", "https://yann.lecun.com/exdb/mnist/", "dataset terms at source"),
        dataset_record("Fashion-MNIST", DATA / "FashionMNIST", "https://github.com/zalandoresearch/fashion-mnist", "MIT"),
        dataset_record(
            "CIFAR-10",
            DATA / "cifar-10-batches-py" if (DATA / "cifar-10-batches-py").exists() else DATA / "huggingface" / "datasets" / "uoft-cs___cifar10",
            "https://huggingface.co/datasets/uoft-cs/cifar10 revision 0b2714987fa478483af9968de7c934580d0bb9a2",
            "dataset terms at canonical source",
        ),
    ]
    write_csv(OUT / "stage0_downloads.csv", downloads)
    baselines = [
        {
            "baseline": "cycle-consistent multi-model synchronization",
            "official_repository": "https://github.com/crisostomi/cycle-consistent-model-merging",
            "revision": "ea1eca76b19c5d57ed97b1ef396368189e864eee",
            "license": "MIT",
            "execution": "faithful internal shared-base specialization; official metadata pinned",
        },
        {
            "baseline": "Task Arithmetic",
            "official_repository": "https://github.com/mlfoundations/task_vectors",
            "revision": "826a64c67082fab0f40628233287948f0f8d7fa3",
            "license": "not declared in repository metadata",
            "execution": "faithful internal implementation; official metadata pinned",
        },
        {
            "baseline": "TIES",
            "official_repository": "https://github.com/prateeky2806/ties-merging",
            "revision": "44e7891fc84f3de7e4caa52664cd864ca3715e91",
            "license": "BSD-3-Clause",
            "execution": "faithful internal implementation; official metadata pinned",
        },
        {
            "baseline": "DARE",
            "official_repository": "https://github.com/yule-BUAA/MergeLM",
            "revision": "6d49ad96fd69c92013654b837041b868aa806564",
            "license": "not declared in repository metadata",
            "execution": "faithful internal implementation; official metadata pinned",
        },
    ]
    write_csv(OUT / "stage0_baselines.csv", baselines)
    provenance = f"""# Compact benchmark provenance

This run starts from commit `{environment['head']}` on branch `{environment['branch']}`. The existing test suite completed successfully: `{result.stdout.strip().splitlines()[-1]}`.

The benchmark uses MNIST and Fashion-MNIST from their public sources. CIFAR-10 is optional until its canonical download or licensed mirror succeeds; a missing cache is never replaced with fabricated data. Downloaded data and model checkpoints remain in ignored local directories, while checksums and source records are public.

Candidate predictions are saved before test-label evaluation. Every accuracy stage performs a byte-identity regression after label permutation. Test labels are not used for model fitting, candidate construction, routing, or selection.

This repository contains scientific evidence and reproducibility artifacts only.
"""
    (OUT / "stage0_provenance.md").write_text(provenance, encoding="utf-8")


if __name__ == "__main__":
    main()
