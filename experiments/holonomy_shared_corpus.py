#!/usr/bin/env python3
"""Build the single frozen-encoder D4 adapter corpus used by Applications A-D."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torchvision import __version__ as torchvision_version
from torchvision.datasets import CIFAR10
from torchvision.models import ResNet18_Weights, resnet18

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.chart_followup_common import apply_d4, compose_d4, inverse_chart
from src.holonomy_application_corpus import (
    CorpusSplitSizes,
    classification_metrics,
    deterministic_split_indices,
    parameter_count,
    seed_everything,
    state_bytes,
    tensor_mapping_sha256,
    train_chart_adapter,
    validate_split_indices,
)

DEFAULT_REPORT_DIR = ROOT / "reports" / "holonomy_applications"
DEFAULT_ARTIFACT_ROOT = ROOT / "reports" / "tmp" / "holonomy_applications"
CHART_NAMES = (
    "rotation_0",
    "rotation_90",
    "rotation_180",
    "rotation_270",
    "reflection_rotation_0",
    "reflection_rotation_90",
    "reflection_rotation_180",
    "reflection_rotation_270",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def device_for(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def raw_images(dataset: CIFAR10, indices: np.ndarray) -> torch.Tensor:
    images = torch.from_numpy(np.asarray(dataset.data)[indices]).permute(0, 3, 1, 2)
    return images.float().div_(255.0)


def encoder_inputs(images: torch.Tensor, chart: int, weights: ResNet18_Weights) -> torch.Tensor:
    transformed = apply_d4(images, chart)
    resized = nn.functional.interpolate(transformed, size=(64, 64), mode="bilinear", align_corners=False)
    mean = torch.tensor(weights.transforms().mean).view(1, 3, 1, 1)
    std = torch.tensor(weights.transforms().std).view(1, 3, 1, 1)
    return (resized - mean) / std


def extract_raw_features(
    encoder: nn.Module,
    images: torch.Tensor,
    weights: ResNet18_Weights,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    charts: list[torch.Tensor] = []
    encoder.eval().to(device)
    with torch.no_grad():
        for chart in range(8):
            parts = []
            for batch in images.split(batch_size):
                parts.append(encoder(encoder_inputs(batch, chart, weights).to(device)).cpu())
            charts.append(torch.cat(parts))
    encoder.cpu()
    return torch.stack(charts)


def fit_projection(training_features: torch.Tensor, feature_dim: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    flattened = training_features.reshape(-1, training_features.shape[-1]).double()
    mean = flattened.mean(dim=0)
    centered = flattened - mean
    covariance = centered.T @ centered / max(len(centered) - 1, 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    order = torch.argsort(eigenvalues, descending=True)[:feature_dim]
    projection = eigenvectors[:, order]
    scale = torch.sqrt(eigenvalues[order].clamp_min(1e-10))
    return mean.float(), projection.float(), scale.float()


def project_features(
    raw: torch.Tensor, mean: torch.Tensor, projection: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    return ((raw - mean) @ projection / scale).float()


def split_payload(
    train_dataset: CIFAR10,
    test_dataset: CIFAR10,
    splits: dict[str, np.ndarray],
    encoder: nn.Module,
    weights: ResNet18_Weights,
    feature_dim: int,
    batch_size: int,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    raw_by_split: dict[str, torch.Tensor] = {}
    for name in ("adapter_train", "overlap_fit", "overlap_validation", "validation"):
        raw_by_split[name] = extract_raw_features(
            encoder, raw_images(train_dataset, splits[name]), weights, batch_size, device
        )
    raw_by_split["test"] = extract_raw_features(
        encoder, raw_images(test_dataset, splits["test"]), weights, batch_size, device
    )
    mean, projection, scale = fit_projection(raw_by_split["adapter_train"], feature_dim)
    projected = {
        name: project_features(values, mean, projection, scale)
        for name, values in raw_by_split.items()
    }
    projection_payload = {"mean": mean, "projection": projection, "scale": scale}
    return projected, projection_payload


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def chart_group_table() -> list[list[int]]:
    return [[compose_d4(left, right) for right in range(8)] for left in range(8)]


def run(args: argparse.Namespace) -> None:
    config = json.loads(args.config.read_text(encoding="utf-8"))
    mode = args.mode
    mode_config = config[mode]
    seeds = [int(value) for value in mode_config["seeds"]]
    sizes = CorpusSplitSizes(**{key: int(value) for key, value in mode_config["split_sizes"].items()})
    feature_dim = int(config["adapter"]["feature_dim"])
    rank = int(config["adapter"]["rank"])
    report_dir = args.report_dir.resolve()
    if mode == "smoke":
        report_dir = report_dir / "shared_corpus_smoke"
    artifact_dir = (args.artifact_root / f"shared_corpus_{mode}").resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    command = " ".join([sys.executable, *sys.argv])
    started = time.perf_counter()
    failures: list[dict[str, object]] = []

    weights = ResNet18_Weights.IMAGENET1K_V1
    encoder_weight_path = Path(torch.hub.get_dir()) / "checkpoints" / Path(weights.url).name
    dataset_archive_path = args.data_dir / "cifar-10-python.tar.gz"
    if not encoder_weight_path.is_file():
        raise FileNotFoundError(f"cached encoder weights are missing: {encoder_weight_path}")
    if not dataset_archive_path.is_file():
        raise FileNotFoundError(f"cached CIFAR-10 archive is missing: {dataset_archive_path}")
    observed_encoder_hash = sha256_file(encoder_weight_path)
    observed_dataset_hash = sha256_file(dataset_archive_path)
    if observed_encoder_hash != args.encoder_weights_sha256:
        raise RuntimeError("cached encoder weight checksum differs from the preregistered value")
    if observed_dataset_hash != args.dataset_archive_sha256:
        raise RuntimeError("cached CIFAR-10 archive checksum differs from the preregistered value")
    source_hash = sha256_file(Path(__file__))
    config_hash = sha256_file(args.config)

    train_dataset = CIFAR10(args.data_dir, train=True, download=False)
    test_dataset = CIFAR10(args.data_dir, train=False, download=False)
    splits = deterministic_split_indices(
        len(train_dataset), len(test_dataset), sizes, int(config["split_seed"])
    )
    validate_split_indices(splits, len(train_dataset), len(test_dataset))
    split_json = {
        name: [int(value) for value in values]
        for name, values in splits.items()
    }
    split_path = report_dir / "shared_corpus_splits.json"
    write_json(split_path, split_json)

    encoder = resnet18(weights=weights)
    encoder.fc = nn.Identity()
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    device = device_for(args.device)
    feature_started = time.perf_counter()
    projected, projection_payload = split_payload(
        train_dataset,
        test_dataset,
        splits,
        encoder,
        weights,
        feature_dim,
        int(mode_config["encoder_batch_size"]),
        device,
    )
    feature_seconds = time.perf_counter() - feature_started
    feature_path = artifact_dir / "projected_features.pt"
    torch.save(
        {
            "schema_version": 1,
            "evidence_label": "natural_measured",
            "features": projected,
            "projection": projection_payload,
            "splits": {name: torch.from_numpy(values) for name, values in splits.items()},
            "chart_names": CHART_NAMES,
            "d4_table": chart_group_table(),
            "inverse_charts": [inverse_chart(chart) for chart in range(8)],
            "encoder_weights": weights.name,
        },
        feature_path,
    )
    feature_file_hash = sha256_file(feature_path)
    feature_content_hash = tensor_mapping_sha256(projected)

    train_labels = torch.tensor(np.asarray(train_dataset.targets), dtype=torch.long)
    adapter_labels = train_labels[torch.from_numpy(splits["adapter_train"])]
    validation_labels = train_labels[torch.from_numpy(splits["validation"])]
    manifest_rows: list[dict[str, object]] = []
    validation_rows: list[dict[str, object]] = []
    artifact_rows: list[dict[str, object]] = [
        {
            "artifact_kind": "projected_feature_cache",
            "seed": "shared",
            "chart": "all",
            "path": str(feature_path),
            "sha256": feature_file_hash,
            "content_sha256": feature_content_hash,
            "bytes": feature_path.stat().st_size,
        },
        {
            "artifact_kind": "split_manifest",
            "seed": "shared",
            "chart": "all",
            "path": str(split_path.relative_to(ROOT)),
            "sha256": sha256_file(split_path),
            "content_sha256": "",
            "bytes": split_path.stat().st_size,
        },
    ]
    capacity_rows: list[dict[str, object]] = []
    validation_guard_passed = True

    for corpus_seed in seeds:
        models = {}
        seed_started = time.perf_counter()
        for chart in range(8):
            training_seed = int(config["training_seed_base"]) + corpus_seed * 100 + chart
            try:
                model, metrics = train_chart_adapter(
                    projected["adapter_train"][chart],
                    adapter_labels,
                    projected["validation"][chart],
                    validation_labels,
                    feature_dim=feature_dim,
                    rank=rank,
                    seed=training_seed,
                    epochs=int(mode_config["epochs"]),
                    batch_size=int(mode_config["adapter_batch_size"]),
                    learning_rate=float(config["adapter"]["learning_rate"]),
                    weight_decay=float(config["adapter"]["weight_decay"]),
                    patience=int(mode_config["patience"]),
                )
                models[str(chart)] = model
                validation_logits = model(projected["validation"][chart]).detach().cpu()
                before_hash = tensor_mapping_sha256({"logits": validation_logits})
                permutation = torch.randperm(
                    len(validation_labels), generator=torch.Generator().manual_seed(training_seed + 991)
                )
                _ = classification_metrics(validation_logits, validation_labels[permutation])
                after_hash = tensor_mapping_sha256({"logits": validation_logits})
                guard_passed = before_hash == after_hash
                validation_guard_passed = validation_guard_passed and guard_passed
                validation_rows.append(
                    {
                        "evidence_label": "natural_measured",
                        "mode": mode,
                        "corpus_seed": corpus_seed,
                        "chart": chart,
                        "chart_name": CHART_NAMES[chart],
                        "training_seed": training_seed,
                        **metrics,
                        "validation_logit_sha256": before_hash,
                        "validation_label_permutation_guard_passed": guard_passed,
                    }
                )
            except Exception as error:
                failures.append(
                    {
                        "mode": mode,
                        "corpus_seed": corpus_seed,
                        "chart": chart,
                        "stage": "adapter_training",
                        "error_type": type(error).__name__,
                        "message": str(error),
                    }
                )
        if len(models) != 8:
            continue
        checkpoint_path = artifact_dir / f"adapter_seed_{corpus_seed}.pt"
        torch.save(
            {
                "schema_version": 1,
                "evidence_label": "natural_measured",
                "corpus_seed": corpus_seed,
                "feature_dim": feature_dim,
                "rank": rank,
                "chart_names": CHART_NAMES,
                "states": {chart: model.state_dict() for chart, model in models.items()},
                "effective_adapters": {
                    chart: model.effective_adapter().detach().cpu() for chart, model in models.items()
                },
            },
            checkpoint_path,
        )
        checkpoint_hash = sha256_file(checkpoint_path)
        test_logits = {
            f"chart_{chart}": models[str(chart)](projected["test"][chart]).detach().cpu().numpy().astype(np.float32)
            for chart in range(8)
        }
        test_logit_content_hash = tensor_mapping_sha256(test_logits)
        logits_path = artifact_dir / f"test_logits_seed_{corpus_seed}.npz"
        np.savez_compressed(
            logits_path,
            **test_logits,
            test_indices=splits["test"],
        )
        logits_file_hash = sha256_file(logits_path)
        elapsed = time.perf_counter() - seed_started
        artifact_rows.extend(
            [
                {
                    "artifact_kind": "adapter_checkpoint_bundle",
                    "seed": corpus_seed,
                    "chart": "all",
                    "path": str(checkpoint_path),
                    "sha256": checkpoint_hash,
                    "content_sha256": "",
                    "bytes": checkpoint_path.stat().st_size,
                },
                {
                    "artifact_kind": "test_logits_before_labels",
                    "seed": corpus_seed,
                    "chart": "all",
                    "path": str(logits_path),
                    "sha256": logits_file_hash,
                    "content_sha256": test_logit_content_hash,
                    "bytes": logits_path.stat().st_size,
                },
            ]
        )
        for chart in range(8):
            model = models[str(chart)]
            metrics = next(
                row for row in validation_rows if row["corpus_seed"] == corpus_seed and row["chart"] == chart
            )
            manifest_rows.append(
                {
                    "evidence_label": "natural_measured",
                    "mode": mode,
                    "corpus_seed": corpus_seed,
                    "chart": chart,
                    "chart_name": CHART_NAMES[chart],
                    "encoder": "torchvision_resnet18",
                    "encoder_weights": weights.name,
                    "encoder_weights_sha256": observed_encoder_hash,
                    "dataset": "CIFAR10",
                    "dataset_archive_sha256": observed_dataset_hash,
                    "feature_dim": feature_dim,
                    "adapter_rank": rank,
                    "training_examples": sizes.adapter_train,
                    "validation_examples": sizes.validation,
                    "overlap_fit_examples": sizes.overlap_fit,
                    "overlap_validation_examples": sizes.overlap_validation,
                    "test_examples": sizes.test,
                    "validation_accuracy": metrics["accuracy"],
                    "validation_nll": metrics["nll"],
                    "validation_brier": metrics["brier"],
                    "validation_ece": metrics["ece"],
                    "epochs_completed": metrics["epochs_completed"],
                    "trainable_parameters": parameter_count(model),
                    "stored_adapter_bytes": state_bytes(model),
                    "checkpoint_path": str(checkpoint_path),
                    "checkpoint_sha256": checkpoint_hash,
                    "test_logits_path": str(logits_path),
                    "test_logits_sha256": logits_file_hash,
                    "test_logits_content_sha256": test_logit_content_hash,
                    "test_labels_used_for_transition_or_selection": False,
                    "validation_label_permutation_guard_passed": metrics[
                        "validation_label_permutation_guard_passed"
                    ],
                    "execution_commit": git_head(),
                    "source_sha256": source_hash,
                    "config_sha256": config_hash,
                    "command": command,
                }
            )
        capacity_rows.append(
            {
                "evidence_label": "natural_measured",
                "mode": mode,
                "corpus_seed": corpus_seed,
                "charts": 8,
                "adapter_rank": rank,
                "active_parameters_per_chart": parameter_count(models["0"]),
                "stored_bytes_all_charts": sum(state_bytes(model) for model in models.values()),
                "branch_count": 8,
                "encoder_parameter_count_frozen_shared": sum(
                    int(parameter.numel()) for parameter in encoder.parameters()
                ),
                "training_seconds_all_charts": elapsed,
                "feature_extraction_seconds_shared": feature_seconds,
            }
        )

    pd.DataFrame(manifest_rows).to_csv(report_dir / "shared_corpus_manifest.csv", index=False)
    pd.DataFrame(validation_rows).to_csv(report_dir / "shared_corpus_validation.csv", index=False)
    pd.DataFrame(artifact_rows).to_csv(report_dir / "shared_corpus_artifact_hashes.csv", index=False)
    pd.DataFrame(capacity_rows).to_csv(report_dir / "shared_corpus_capacity.csv", index=False)
    failure_columns = ("mode", "corpus_seed", "chart", "stage", "error_type", "message")
    pd.DataFrame(failures, columns=failure_columns).to_csv(
        report_dir / "shared_corpus_failure_log.csv", index=False
    )
    resolved = {
        "schema_version": 1,
        "mode": mode,
        "evidence_label": "natural_measured",
        "command": command,
        "execution_commit": git_head(),
        "device": str(device),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision_version,
        "data_dir": str(args.data_dir.resolve()),
        "artifact_dir": str(artifact_dir),
        "feature_cache_path": str(feature_path),
        "feature_cache_sha256": feature_file_hash,
        "feature_content_sha256": feature_content_hash,
        "encoder_weights": weights.name,
        "encoder_weight_path": str(encoder_weight_path),
        "encoder_weights_sha256": observed_encoder_hash,
        "dataset_archive_path": str(dataset_archive_path),
        "dataset_archive_sha256": observed_dataset_hash,
        "source_sha256": source_hash,
        "config_sha256": config_hash,
        "chart_names": CHART_NAMES,
        "d4_multiplication_table": chart_group_table(),
        "inverse_charts": [inverse_chart(chart) for chart in range(8)],
        "feature_dim": feature_dim,
        "adapter_rank": rank,
        "seeds": seeds,
        "split_sizes": mode_config["split_sizes"],
        "feature_extraction_seconds": feature_seconds,
        "total_seconds": time.perf_counter() - started,
        "successful_adapter_count": len(manifest_rows),
        "expected_adapter_count": 8 * len(seeds),
        "failure_count": len(failures),
        "validation_label_permutation_guard_passed": validation_guard_passed,
        "test_labels_used_during_corpus_construction": False,
    }
    mean_validation = float(pd.DataFrame(validation_rows)["accuracy"].mean()) if validation_rows else float("nan")
    minimum_validation = float(pd.DataFrame(validation_rows)["accuracy"].min()) if validation_rows else float("nan")
    quality_gate_passed = (
        mean_validation >= float(mode_config["minimum_mean_validation_accuracy"])
        and minimum_validation >= float(mode_config["minimum_worst_validation_accuracy"])
    )
    resolved["mean_validation_accuracy"] = mean_validation
    resolved["minimum_validation_accuracy"] = minimum_validation
    resolved["corpus_quality_gate_passed"] = quality_gate_passed
    write_json(report_dir / "shared_corpus_resolved_config.json", resolved)
    report = f"""# Shared D4 Adapter Corpus Report

Decision: **{'complete' if len(manifest_rows) == 8 * len(seeds) and not failures and validation_guard_passed and quality_gate_passed else 'incomplete'} {mode} corpus construction**.

## Exact command

```bash
{command}
```

- Evidence label: `natural_measured`
- Frozen encoder: torchvision ResNet-18 `{weights.name}`
- Dataset: CIFAR-10 from the existing local cache
- Charts: eight D4 transforms using the repository's audited action convention
- Adapter: rank-{rank} residual map in a {feature_dim}-dimensional train-only PCA feature space plus a ten-class head
- Independent model-training seeds: {seeds}
- Successful chart adapters: {len(manifest_rows)} / {8 * len(seeds)}
- Mean validation accuracy: {mean_validation:.6f}
- Worst chart/seed validation accuracy: {minimum_validation:.6f}
- Corpus quality gate: {'passed' if quality_gate_passed else 'failed'}
- Test logits: saved and hashed before any test labels are accessed
- Test labels used during corpus construction: no
- Validation label-permutation guard: {'passed' if validation_guard_passed else 'failed'}
- Failures: {len(failures)}

## Boundary

This corpus is the only trained model family authorized for Applications A-D. Later phases must load the exact feature cache, checkpoints, splits, and logits identified in the manifests; they must not retrain chart adapters. Validation accuracy is a corpus-quality diagnostic, not an application result. No test accuracy or holonomy claim is made here.

The encoder and all its parameters remain frozen. PCA fitting uses only `adapter_train` features. Transition fitting is reserved for `overlap_fit`, transition validation for `overlap_validation`, method selection for `validation`, and final application scoring for the untouched `test` identities.
"""
    (report_dir / "shared_corpus_report.md").write_text(report, encoding="utf-8")
    if (
        len(manifest_rows) != 8 * len(seeds)
        or failures
        or not validation_guard_passed
        or not quality_gate_passed
    ):
        raise RuntimeError("shared corpus integrity gate failed; inspect the committed failure log")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "confirmatory"), required=True)
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_REPORT_DIR / "shared_corpus_config.json"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/Users/tinggong/Documents/GitHub/TwistedMerge/data"),
    )
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--encoder-weights-sha256",
        default="f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec",
    )
    parser.add_argument(
        "--dataset-archive-sha256",
        default="6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
