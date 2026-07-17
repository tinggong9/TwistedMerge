#!/usr/bin/env python3
"""Post-ICLR official-source integration on exact TwistedMerge checkpoints."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.external_baseline_comparison import split_train_val  # noqa: E402
from experiments.same_base_task_vector_benchmark import (  # noqa: E402
    TASK_PRESETS,
    combined_loader,
    evaluate_across_tasks,
    make_subset,
    select_candidate,
    split_indices,
    state_vector,
    subset_by_classes,
    ties_vector,
    vector_to_model,
)
from src.model_merging_benchmark import (  # noqa: E402
    average_models,
    device_from_arg,
    evaluate_ensemble,
    evaluate_model,
    greedy_soup,
    load_dataset,
    make_loader,
    make_model,
)
from src.official_baseline_adapters import (  # noqa: E402
    average_state_dicts,
    git_rebasin_arrays_to_torch_state,
    official_c2m3_synchronized_states,
    official_ties_state,
    torch_state_to_git_rebasin_arrays,
)


REPOSITORIES = {
    "official_git_rebasin": {
        "directory": "git-re-basin",
        "url": "https://github.com/samuela/git-re-basin.git",
        "license": "MIT",
        "regime": "independent_initialization",
    },
    "official_c2m3": {
        "directory": "c2m3",
        "url": "https://github.com/crisostomi/cycle-consistent-model-merging.git",
        "license": "MIT",
        "regime": "independent_initialization",
    },
    "official_model_soups": {
        "directory": "model-soups",
        "url": "https://github.com/mlfoundations/model-soups.git",
        "license": "MIT",
        "regime": "both_separate",
    },
    "official_task_arithmetic": {
        "directory": "task-vectors",
        "url": "https://github.com/mlfoundations/task_vectors.git",
        "license": "NO_REPOSITORY_LICENSE_FILE",
        "regime": "common_base_task_vector",
    },
    "official_ties": {
        "directory": "ties-merging",
        "url": "https://github.com/prateeky2806/ties-merging.git",
        "license": "BSD-3-Clause",
        "regime": "common_base_task_vector",
    },
    "official_dare": {
        "directory": "mergelm",
        "url": "https://github.com/yule-BUAA/MergeLM.git",
        "license": "NO_REPOSITORY_LICENSE_FILE",
        "regime": "common_base_task_vector",
    },
}

RUN_FIELDS = [
    "regime",
    "baseline",
    "setting_id",
    "status",
    "implementation_kind",
    "official_repository",
    "source_commit",
    "source_dirty",
    "license",
    "adapter",
    "patches",
    "checkpoint_conversion",
    "exact_command",
    "dataset",
    "architecture",
    "n_models",
    "width",
    "seed",
    "val_accuracy",
    "val_loss",
    "test_accuracy",
    "test_loss",
    "average_test_accuracy",
    "worst_task_accuracy",
    "parameter_multiplier",
    "active_parameter_count",
    "inference_multiplier",
    "branches",
    "output_type",
    "functional_preservation_max_abs_error",
    "prediction_disagreement",
    "merge_compute_seconds",
    "training_compute",
    "selected_hyperparameters",
    "delta_vs_internal_same_method",
    "delta_vs_internal_c2m3",
    "delta_vs_internal_greedy_soup",
    "delta_vs_weight_average",
    "delta_vs_twistedmerge_gauge",
    "delta_vs_twistedmerge_selector",
    "delta_vs_prediction_ensemble",
    "failed_reason",
]


def parse_csv(text: str, cast=str):
    return [cast(item.strip()) for item in text.split(",") if item.strip()]


def git_output(path: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def source_metadata(args, baseline: str) -> dict:
    metadata = REPOSITORIES[baseline]
    source = args.official_root / metadata["directory"]
    if not source.exists():
        return {**metadata, "source": source, "commit": "", "dirty": "", "source_status": "clone_missing"}
    return {
        **metadata,
        "source": source,
        "commit": git_output(source, "rev-parse", "HEAD"),
        "dirty": bool(git_output(source, "status", "--short")),
        "source_status": "source_pinned",
    }


def row_base(args, baseline: str, *, setting_id: str, regime: str | None = None) -> dict:
    source = source_metadata(args, baseline)
    return {
        "regime": regime or source["regime"],
        "baseline": baseline,
        "setting_id": setting_id,
        "official_repository": source["url"],
        "source_commit": source["commit"],
        "source_dirty": source["dirty"],
        "license": source["license"],
        "parameter_multiplier": 1.0,
        "inference_multiplier": 1.0,
        "branches": 1,
        "output_type": "same_capacity_single_model",
        "training_compute": "checkpoint_reuse_no_training",
        "exact_command": args.command,
    }


def load_state(path: Path):
    import torch

    payload = torch.load(path, map_location="cpu")
    return payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload


def model_from_state(state, architecture, spec, width):
    model = make_model(architecture, spec, width)
    model.load_state_dict(state)
    model.to("cpu")
    return model


def preservation_metrics(original_states, aligned_states, architecture, spec, width, loader) -> tuple[float, float]:
    import torch

    maximum = 0.0
    disagreements = 0
    total = 0
    with torch.no_grad():
        for original_state, aligned_state in zip(original_states, aligned_states):
            original = model_from_state(original_state, architecture, spec, width).eval()
            aligned = model_from_state(aligned_state, architecture, spec, width).eval()
            for images, _ in loader:
                first = original(images)
                second = aligned(images)
                maximum = max(maximum, float((first - second).abs().max().item()))
                disagreements += int((first.argmax(dim=1) != second.argmax(dim=1)).sum().item())
                total += int(images.shape[0])
    return maximum, float(disagreements / max(total, 1))


def git_rebasin_align(args, reference_state, candidate_state, *, seed: int, index: int):
    reference = torch_state_to_git_rebasin_arrays(reference_state)
    candidate = torch_state_to_git_rebasin_arrays(candidate_state)
    with tempfile.TemporaryDirectory(prefix="twistedmerge-git-rebasin-") as directory:
        directory = Path(directory)
        input_path = directory / "input.npz"
        output_path = directory / "output.npz"
        metadata_path = directory / "metadata.json"
        np.savez(
            input_path,
            **{**{f"a_{key}": value for key, value in reference.items()}, **{f"b_{key}": value for key, value in candidate.items()}},
        )
        command = [
            str(args.jax_python),
            str(ROOT / "experiments" / "official_git_rebasin_worker.py"),
            "--source-root",
            str(args.official_root / "git-re-basin"),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--metadata",
            str(metadata_path),
            "--seed",
            str(seed + index),
            "--max-iter",
            str(args.git_rebasin_max_iter),
        ]
        subprocess.run(command, check=True, capture_output=True, text=True)
        payload = np.load(output_path, allow_pickle=False)
        arrays = {key: payload[key] for key in payload.files}
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return git_rebasin_arrays_to_torch_state(arrays, candidate_state), metadata


def independent_references(internal: pd.DataFrame, setting_id: str) -> dict[str, float]:
    subset = internal[internal["setting_id"].eq(setting_id)]
    return {row.method: float(row.test_accuracy) for row in subset.itertuples(index=False)}


def run_independent(args, spec, train_data, test_data, internal: pd.DataFrame) -> list[dict]:
    rows = []
    device = device_from_arg(args.device)
    for seed in args.seeds:
        for n_models in args.model_counts:
            for width in args.widths:
                setting_id = f"mnist_mlp_N{n_models}_W{width}_S{seed}"
                checkpoint_dir = args.independent_checkpoint_root / setting_id
                paths = [checkpoint_dir / f"model_{index}.pt" for index in range(n_models)]
                if not all(path.exists() for path in paths):
                    for baseline in ("official_git_rebasin", "official_c2m3"):
                        rows.append({**row_base(args, baseline, setting_id=setting_id), "status": "blocked_missing_checkpoint", "failed_reason": ";".join(str(path) for path in paths if not path.exists())})
                    continue
                states = [load_state(path) for path in paths]
                _, val_subset = split_train_val(train_data, args.val_fraction, seed + 77)
                val_loader = make_loader(val_subset, args.batch_size, shuffle=False, seed=seed + 700)
                test_loader = make_loader(test_data, args.batch_size, shuffle=False, seed=seed + 999)
                references = independent_references(internal, setting_id)
                original_models = [model_from_state(state, "mlp", spec, width) for state in states]
                references["prediction_ensemble"] = float(evaluate_ensemble(original_models, test_loader, device)["accuracy"])

                start = time.perf_counter()
                aligned = [states[0]]
                metadata = []
                try:
                    for index, state in enumerate(states[1:], start=1):
                        converted, details = git_rebasin_align(args, states[0], state, seed=seed, index=index)
                        aligned.append(converted)
                        metadata.append(details)
                    maximum, disagreement = preservation_metrics(states, aligned, "mlp", spec, width, val_loader)
                    merged_state = average_state_dicts(aligned)
                    merged = model_from_state(merged_state, "mlp", spec, width)
                    val = evaluate_model(merged, val_loader, device)
                    test = evaluate_model(merged, test_loader, device)
                    base = row_base(args, "official_git_rebasin", setting_id=setting_id)
                    rows.append({
                        **base,
                        "status": "evaluated",
                        "implementation_kind": "adapter_assisted_official_core",
                        "adapter": "PyTorch state dict to official Flax axes; official weight_matching.py; PyTorch evaluation",
                        "patches": "import-only rngmix shim; no optimizer patch",
                        "checkpoint_conversion": "hidden/classifier tensors transposed to Dense_0/Dense_1 axes and converted back",
                        "dataset": "mnist",
                        "architecture": "one_hidden_layer_relu_mlp",
                        "n_models": n_models,
                        "width": width,
                        "seed": seed,
                        "val_accuracy": float(val["accuracy"]),
                        "val_loss": float(val["loss"]),
                        "test_accuracy": float(test["accuracy"]),
                        "test_loss": float(test["loss"]),
                        "active_parameter_count": sum(value.numel() for value in merged_state.values()),
                        "functional_preservation_max_abs_error": maximum,
                        "prediction_disagreement": disagreement,
                        "merge_compute_seconds": time.perf_counter() - start,
                        "selected_hyperparameters": json.dumps({"max_iter": args.git_rebasin_max_iter, "pairwise_reference": 0, "worker_runs": metadata}, sort_keys=True),
                        "delta_vs_internal_same_method": float(test["accuracy"]) - references.get("git_rebasin_pairwise", math.nan),
                        "delta_vs_internal_c2m3": float(test["accuracy"]) - references.get("c2m3_permutation", math.nan),
                        "delta_vs_internal_greedy_soup": float(test["accuracy"]) - references.get("greedy_soup", math.nan),
                        "delta_vs_weight_average": float(test["accuracy"]) - references.get("weight_average", math.nan),
                        "delta_vs_twistedmerge_gauge": float(test["accuracy"]) - references.get("monomial_scale", math.nan),
                        "delta_vs_twistedmerge_selector": float(test["accuracy"]) - references.get("improved_validated_selector", math.nan),
                        "delta_vs_prediction_ensemble": float(test["accuracy"]) - references["prediction_ensemble"],
                    })
                except Exception as error:
                    rows.append({**row_base(args, "official_git_rebasin", setting_id=setting_id), "status": "blocked_runtime", "failed_reason": f"{type(error).__name__}: {error}", "merge_compute_seconds": time.perf_counter() - start})

                start = time.perf_counter()
                try:
                    aligned, permutations, optimization = official_c2m3_synchronized_states(
                        states,
                        args.official_root / "c2m3",
                        max_iter=args.c2m3_max_iter,
                    )
                    maximum, disagreement = preservation_metrics(states, aligned, "mlp", spec, width, val_loader)
                    merged_state = average_state_dicts(aligned)
                    merged = model_from_state(merged_state, "mlp", spec, width)
                    val = evaluate_model(merged, val_loader, device)
                    test = evaluate_model(merged, test_loader, device)
                    rows.append({
                        **row_base(args, "official_c2m3", setting_id=setting_id),
                        "status": "evaluated",
                        "implementation_kind": "adapter_assisted_official_core",
                        "adapter": "application initializer bypass; exact MLP PermutationSpec; official Frank-Wolfe synchronized matcher; PyTorch evaluation",
                        "patches": "external_baselines/patches/c2m3_cpu_device.patch",
                        "checkpoint_conversion": "native PyTorch state tensors; checkpoint keys mapped by adapter PermutationSpec",
                        "dataset": "mnist",
                        "architecture": "one_hidden_layer_relu_mlp",
                        "n_models": n_models,
                        "width": width,
                        "seed": seed,
                        "val_accuracy": float(val["accuracy"]),
                        "val_loss": float(val["loss"]),
                        "test_accuracy": float(test["accuracy"]),
                        "test_loss": float(test["loss"]),
                        "active_parameter_count": sum(value.numel() for value in merged_state.values()),
                        "functional_preservation_max_abs_error": maximum,
                        "prediction_disagreement": disagreement,
                        "merge_compute_seconds": time.perf_counter() - start,
                        "selected_hyperparameters": json.dumps({"max_iter": args.c2m3_max_iter, "initialization": "identity", "permutations": permutations, "objective_values": [float(value) for value in optimization["obj_values"]]}, sort_keys=True),
                        "delta_vs_internal_same_method": float(test["accuracy"]) - references.get("c2m3_permutation", math.nan),
                        "delta_vs_internal_c2m3": float(test["accuracy"]) - references.get("c2m3_permutation", math.nan),
                        "delta_vs_internal_greedy_soup": float(test["accuracy"]) - references.get("greedy_soup", math.nan),
                        "delta_vs_weight_average": float(test["accuracy"]) - references.get("weight_average", math.nan),
                        "delta_vs_twistedmerge_gauge": float(test["accuracy"]) - references.get("monomial_scale", math.nan),
                        "delta_vs_twistedmerge_selector": float(test["accuracy"]) - references.get("improved_validated_selector", math.nan),
                        "delta_vs_prediction_ensemble": float(test["accuracy"]) - references["prediction_ensemble"],
                    })
                except Exception as error:
                    rows.append({**row_base(args, "official_c2m3", setting_id=setting_id), "status": "blocked_runtime", "failed_reason": f"{type(error).__name__}: {error}", "merge_compute_seconds": time.perf_counter() - start})
    return rows


def common_task_loaders(args, seed: int):
    spec, train_base, test_base = load_dataset("mnist", args.data_dir, 6000, 2000, 314159, augmentation="none")
    _, val_indices = split_indices(len(train_base), 0.2, 314159 + 17 + seed)
    task_defs = TASK_PRESETS["mnist_digit_subsets"]
    val_loaders = {}
    test_loaders = {}
    for task_index, task in enumerate(task_defs):
        val_task_indices = subset_by_classes(train_base, val_indices, task.classes, 600, seed + 2000 + task_index)
        test_task_indices = subset_by_classes(test_base, list(range(len(test_base))), task.classes, 600, seed + 3000 + task_index)
        val_loaders[task.name] = make_loader(make_subset(train_base, val_task_indices), 128, shuffle=False, seed=seed + 5000 + task_index)
        test_loaders[task.name] = make_loader(make_subset(test_base, test_task_indices), 128, shuffle=False, seed=seed + 6000 + task_index)
    return spec, task_defs, val_loaders, test_loaders


def exact_common_references(args, spec, base_state, task_states, val_loaders, test_loaders, device, seed: int) -> dict:
    """Recompute internal controls on the exact official-core checkpoints.

    The repository's aggregate same-base CSV can be regenerated with a
    different seed range, so it is not a reliable paired source for the saved
    7200--7202 checkpoints.  These controls use the original benchmark's
    implementations and validation-only selection on the exact same loaders.
    """

    task_models = [model_from_state(state, "mlp2", spec, 64) for state in task_states]
    weight_model = average_models(task_models, "mlp2", spec, 64)
    combined_val = combined_loader(val_loaders, args.batch_size, seed + 7000)
    soup_model, soup_indices, _, _ = greedy_soup(
        task_models,
        combined_val,
        combined_val,
        device,
        "mlp2",
        spec,
        64,
        return_trajectory=True,
    )

    base_model = model_from_state(base_state, "mlp2", spec, 64)
    base_vector, meta = state_vector(base_model)
    task_vectors = [state_vector(model)[0] for model in task_models]
    deltas = [vector - base_vector for vector in task_vectors]
    ties_candidates = []
    for density in args.ties_densities:
        for scale in args.ties_scales:
            model = vector_to_model(ties_vector(base_vector, deltas, density, scale), meta, "mlp2", spec, 64)
            ties_candidates.append(({"density": density, "scale": scale}, model))
    internal_ties_selection, internal_ties_model, internal_ties_trace = select_candidate(
        "internal_ties_merging",
        ties_candidates,
        val_loaders,
        device,
    )

    return {
        "weight_average": evaluate_across_tasks(weight_model, test_loaders, device)["average_accuracy"],
        "greedy_soup": evaluate_across_tasks(soup_model, test_loaders, device)["average_accuracy"],
        "ties_merging": evaluate_across_tasks(internal_ties_model, test_loaders, device)["average_accuracy"],
        "greedy_soup_indices": soup_indices,
        "internal_ties_selection": internal_ties_selection,
        "internal_ties_trace": internal_ties_trace["candidates"],
    }


def run_common_base(args) -> list[dict]:
    rows = []
    device = device_from_arg(args.device)
    setting = "mnist_mnist_digit_subsets_mlp2_W64_N3"
    for seed in args.common_seeds:
        checkpoint_dir = args.common_checkpoint_root / setting / f"seed{seed}"
        task_paths = sorted(checkpoint_dir.glob("task_*.pt"))
        base_path = checkpoint_dir / "base.pt"
        if not base_path.exists() or len(task_paths) != 3:
            rows.append({**row_base(args, "official_ties", setting_id=f"{setting}_seed{seed}"), "status": "blocked_missing_checkpoint", "failed_reason": str(checkpoint_dir)})
            continue
        spec, _, val_loaders, test_loaders = common_task_loaders(args, seed)
        base_state = load_state(base_path)
        task_states = [load_state(path) for path in task_paths]
        candidates = []
        start = time.perf_counter()
        try:
            for density in args.ties_densities:
                for scale in args.ties_scales:
                    state = official_ties_state(
                        base_state,
                        task_states,
                        args.official_root / "ties-merging",
                        density=density,
                        scale=scale,
                    )
                    candidates.append(({"density": density, "scale": scale}, model_from_state(state, "mlp2", spec, 64)))
            selected, model, trace = select_candidate("official_ties", candidates, val_loaders, device)
            val = evaluate_across_tasks(model, val_loaders, device)
            test = evaluate_across_tasks(model, test_loaders, device)
            references = exact_common_references(
                args,
                spec,
                base_state,
                task_states,
                val_loaders,
                test_loaders,
                device,
                seed,
            )
            rows.append({
                **row_base(args, "official_ties", setting_id=f"{setting}_seed{seed}"),
                "status": "evaluated",
                "implementation_kind": "adapter_assisted_official_core",
                "adapter": "TwistedMerge state flatten/unflatten and validation evaluator around official merge_utils.merge_methods",
                "patches": "none",
                "checkpoint_conversion": "stable sorted state tensors flattened to official TIES delta matrix and restored",
                "dataset": "mnist",
                "architecture": "two_hidden_layer_relu_mlp",
                "n_models": 3,
                "width": 64,
                "seed": seed,
                "val_accuracy": float(val["average_accuracy"]),
                "val_loss": float(val["average_loss"]),
                "average_test_accuracy": float(test["average_accuracy"]),
                "worst_task_accuracy": float(test["worst_accuracy"]),
                "active_parameter_count": sum(value.numel() for value in base_state.values()),
                "merge_compute_seconds": time.perf_counter() - start,
                "selected_hyperparameters": json.dumps({
                    **selected,
                    "selection_trace": trace["candidates"],
                    "paired_internal_controls": {
                        "weight_average": references["weight_average"],
                        "greedy_soup": references["greedy_soup"],
                        "ties_merging": references["ties_merging"],
                        "greedy_soup_indices": references["greedy_soup_indices"],
                        "internal_ties_selection": references["internal_ties_selection"],
                        "internal_ties_trace": references["internal_ties_trace"],
                    },
                }, sort_keys=True),
                "delta_vs_internal_same_method": float(test["average_accuracy"]) - references.get("ties_merging", math.nan),
                "delta_vs_internal_greedy_soup": float(test["average_accuracy"]) - references.get("greedy_soup", math.nan),
                "delta_vs_weight_average": float(test["average_accuracy"]) - references.get("weight_average", math.nan),
            })
        except Exception as error:
            rows.append({**row_base(args, "official_ties", setting_id=f"{setting}_seed{seed}"), "status": "blocked_runtime", "failed_reason": f"{type(error).__name__}: {error}", "merge_compute_seconds": time.perf_counter() - start})
    return rows


def status_rows(args) -> list[dict]:
    reasons = {
        "official_model_soups": ("blocked_incompatible_interface", "Official main.py is inseparable from its CLIP/ImageNet model loader and evaluator; replacing those components would be a reimplementation, so the faithful internal greedy soup remains separately labeled."),
        "official_task_arithmetic": ("blocked_license", "The pinned author repository contains no LICENSE or COPYING file; author code was not used for a publishable metric."),
        "official_dare": ("blocked_license", "The pinned author MergeLM repository contains no LICENSE or COPYING file; author code was not used for a publishable metric."),
    }
    rows = []
    for baseline, (status, reason) in reasons.items():
        rows.append({
            **row_base(args, baseline, setting_id="integration_status"),
            "status": status,
            "implementation_kind": "official_source_probe_no_metric",
            "adapter": "none",
            "patches": "none",
            "checkpoint_conversion": "not_performed",
            "failed_reason": reason,
        })
    return rows


def bootstrap_ci(values, samples: int, seed: int) -> tuple[float, float]:
    array = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if len(array) == 0:
        return math.nan, math.nan
    if len(array) == 1:
        return float(array[0]), float(array[0])
    rng = np.random.default_rng(seed)
    means = [float(rng.choice(array, size=len(array), replace=True).mean()) for _ in range(samples)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarize(rows: list[dict], bootstrap_samples: int) -> list[dict]:
    all_rows = pd.DataFrame(rows)
    frame = all_rows[all_rows["status"].eq("evaluated")].copy()
    output = []
    for (regime, baseline), group in frame.groupby(["regime", "baseline"], dropna=False):
        score_column = "test_accuracy" if regime == "independent_initialization" else "average_test_accuracy"
        scores = pd.to_numeric(group[score_column], errors="coerce").dropna().to_numpy(dtype=float)
        deltas = pd.to_numeric(group["delta_vs_internal_same_method"], errors="coerce").dropna().to_numpy(dtype=float)
        low, high = bootstrap_ci(scores, bootstrap_samples, 73000 + len(output))
        delta_low, delta_high = bootstrap_ci(deltas, bootstrap_samples, 74000 + len(output))
        output.append({
            "regime": regime,
            "baseline": baseline,
            "n_rows": len(group),
            "n_unique_seeds": int(pd.to_numeric(group["seed"], errors="coerce").nunique()),
            "failed_run_count": int(
                all_rows[
                    all_rows["regime"].eq(regime)
                    & all_rows["baseline"].eq(baseline)
                    & ~all_rows["status"].eq("evaluated")
                ].shape[0]
            ),
            "mean_score": float(np.mean(scores)),
            "median_score": float(np.median(scores)),
            "std_score": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
            "score_ci_low": low,
            "score_ci_high": high,
            "mean_delta_vs_internal_same_method": float(np.mean(deltas)) if len(deltas) else math.nan,
            "delta_ci_low": delta_low,
            "delta_ci_high": delta_high,
            "wins_vs_internal_same_method": int((deltas > 1e-12).sum()),
            "ties_vs_internal_same_method": int((np.abs(deltas) <= 1e-12).sum()),
            "losses_vs_internal_same_method": int((deltas < -1e-12).sum()),
            "mean_merge_compute_seconds": float(pd.to_numeric(group["merge_compute_seconds"], errors="coerce").mean()),
            "parameter_multiplier": 1.0,
            "inference_multiplier": 1.0,
            "output_type": "same_capacity_single_model",
        })
        for field, name in [
            ("delta_vs_internal_greedy_soup", "greedy_soup"),
            ("delta_vs_twistedmerge_gauge", "twistedmerge_gauge"),
            ("delta_vs_twistedmerge_selector", "twistedmerge_selector"),
            ("delta_vs_prediction_ensemble", "prediction_ensemble"),
        ]:
            values = pd.to_numeric(group[field], errors="coerce").dropna().to_numpy(dtype=float)
            comparison_low, comparison_high = bootstrap_ci(
                values,
                bootstrap_samples,
                75000 + len(output) * 10 + len(name),
            )
            output[-1].update({
                f"mean_delta_vs_{name}": float(np.mean(values)) if len(values) else math.nan,
                f"delta_vs_{name}_ci_low": comparison_low,
                f"delta_vs_{name}_ci_high": comparison_high,
                f"wins_vs_{name}": int((values > 1e-12).sum()),
                f"ties_vs_{name}": int((np.abs(values) <= 1e-12).sum()),
                f"losses_vs_{name}": int((values < -1e-12).sum()),
            })
    return output


def write_plot(summary: list[dict], path: Path) -> None:
    import matplotlib.pyplot as plt

    evaluated = [row for row in summary if np.isfinite(row["mean_delta_vs_internal_same_method"])]
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    if evaluated:
        display_names = {
            "official_c2m3": "C2M3",
            "official_git_rebasin": "Git Re-Basin",
            "official_ties": "TIES",
        }
        labels = [display_names.get(row["baseline"], row["baseline"]) for row in evaluated]
        values = [row["mean_delta_vs_internal_same_method"] for row in evaluated]
        errors = [
            [value - row["delta_ci_low"] for value, row in zip(values, evaluated)],
            [row["delta_ci_high"] - value for value, row in zip(values, evaluated)],
        ]
        axis.bar(labels, values, color="#4C78A8")
        axis.errorbar(range(len(values)), values, yerr=errors, fmt="none", ecolor="black", capsize=4)
    axis.axhline(0.0, color="black", linewidth=1)
    axis.set_ylabel("Accuracy delta vs internal counterpart")
    axis.set_title("Adapter-assisted official-core comparisons")
    axis.tick_params(axis="x", rotation=12)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def write_latex(summary: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "Regime & Official core & $n$ & Mean score & Delta vs internal & 95\\% CI \\\\",
        "\\midrule",
    ]
    for row in summary:
        regime = row["regime"].replace("_", "\\_")
        baseline = row["baseline"].replace("official_", "").replace("_", "\\_")
        lines.append(f"{regime} & {baseline} & {row['n_rows']} & {row['mean_score']:.4f} & {row['mean_delta_vs_internal_same_method']:.4f} & [{row['delta_ci_low']:.4f}, {row['delta_ci_high']:.4f}] \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def baseline_regime_rows() -> list[dict]:
    return [
        {"method": "raw_weight_average", "regime": "independent_initialization", "official_status": "native_control", "fair_comparison": True, "capacity": "1x single model", "boundary": "unaligned control only"},
        {"method": "official_git_rebasin", "regime": "independent_initialization", "official_status": "adapter_assisted_official_core", "fair_comparison": True, "capacity": "1x single model", "boundary": "not unmodified official end-to-end pipeline"},
        {"method": "official_c2m3", "regime": "independent_initialization", "official_status": "adapter_assisted_official_core", "fair_comparison": True, "capacity": "1x single model", "boundary": "CPU portability patch and checkpoint PermutationSpec adapter"},
        {"method": "internal_c2m3_style", "regime": "independent_initialization", "official_status": "internal_faithful_style", "fair_comparison": True, "capacity": "1x single model", "boundary": "never label official"},
        {"method": "TwistedMerge_gauge_merge", "regime": "independent_initialization", "official_status": "project_method", "fair_comparison": True, "capacity": "1x single model", "boundary": "exact gauge does not imply performance gain"},
        {"method": "greedy_model_soup", "regime": "independent_initialization", "official_status": "internal_faithful_only_official_blocked", "fair_comparison": True, "capacity": "1x single model", "boundary": "official CLIP/ImageNet code did not run"},
        {"method": "prediction_ensemble", "regime": "independent_initialization", "official_status": "upper_bound", "fair_comparison": False, "capacity": "N branches Nx inference", "boundary": "upper bound not same-cost candidate"},
        {"method": "official_task_arithmetic", "regime": "common_base_task_vector", "official_status": "blocked_license", "fair_comparison": False, "capacity": "1x single model", "boundary": "author repository has no license file"},
        {"method": "official_ties", "regime": "common_base_task_vector", "official_status": "adapter_assisted_official_core", "fair_comparison": True, "capacity": "1x single model", "boundary": "common-base checkpoints only"},
        {"method": "official_dare", "regime": "common_base_task_vector", "official_status": "blocked_license", "fair_comparison": False, "capacity": "1x single model", "boundary": "author repository has no license file"},
        {"method": "descent_envelope_selector", "regime": "common_base_task_vector", "official_status": "project_method", "fair_comparison": True, "capacity": "1x selected single model", "boundary": "validation-selected exact settings only"},
    ]


def write_documentation(args, rows: list[dict], summary: list[dict]) -> None:
    evaluated = [row for row in rows if row.get("status") == "evaluated"]
    blocked = [row for row in rows if str(row.get("status", "")).startswith("blocked")]
    source_lines = []
    for baseline in REPOSITORIES:
        source = source_metadata(args, baseline)
        source_lines.append(f"| `{baseline}` | {source['url']} | `{source['commit'] or 'missing'}` | {source['license']} | {source['source_status']} | {source['dirty']} |")
    integration = f"""# Post-ICLR Official Baseline Integration

This integration was generated from the isolated `codex/post-iclr-experiments` worktree. Official sources were cloned outside the tracked repository and pinned by commit. No third-party source is vendored.

## Source ledger

| Baseline | Repository | Commit | License | Clone status | Dirty after required patch |
| --- | --- | --- | --- | --- | --- |
{os.linesep.join(source_lines)}

## Installation and adapter boundary

The official trees are external to the tracked repository. Recreate the source ledger with `git clone <repository-url> <official-root>/<directory>` followed by `git -C <official-root>/<directory> checkout <commit-from-the-table>`. The isolated Git Re-Basin worker environment was created with `python3.12 -m venv <jax-env>` and `<jax-env>/bin/python -m pip install jax==0.4.38 scipy`. The main integration uses the repository's existing Python 3.12 environment. The exact smoke and confirmatory commands are recorded in the config and run CSV.

- Git Re-Basin: a Python 3.12 environment installed `jax==0.4.38` and SciPy; the worker converts PyTorch MLP tensors to the axes expected by the official `src/weight_matching.py`, executes that file, converts back, and evaluates in TwistedMerge. The optimizer source is unmodified; only an import-only `rngmix` shim avoids pulling the unrelated Flax/W&B application stack.
- C2M3: the adapter bypasses the Hydra/Lightning application initializer, supplies an exact one-hidden-layer `PermutationSpec`, and executes the official Frank-Wolfe synchronized matcher. The tracked patch `external_baselines/patches/c2m3_cpu_device.patch` replaces one hard-coded `.cuda()` with the current permutation tensor's device.
- TIES: the adapter flattens the saved common-base task deltas, executes the official BSD-licensed `merge_utils.merge_methods` trim/elect/disjoint-mean kernel, restores the state dictionary, and chooses density/scale using validation data only. At the keep-all boundary, the adapter maps density 1.0 to the immediately preceding floating-point value because the official `topk_values_mask` requests invalid `k=0` at exactly 1.0; this preserves the intended keep-all mask without changing the official source.
- Model Soups: the MIT source is pinned, but `main.py` is inseparable from the official CLIP/ImageNet loader/evaluator. Replacing those components would no longer be an official execution, so no official metric is emitted.
- Task Arithmetic and DARE: the pinned author repositories contain no LICENSE or COPYING file. They are recorded as legal-use blockers and no publishable official metric is emitted.

## Exact checkpoint families

Independent-initialization runs use the existing MNIST one-hidden-layer MLP groups at `{args.independent_checkpoint_root}`. Common-base TIES runs use the available MNIST `mlp2` base/task groups at `{args.common_checkpoint_root}`. Independent internal comparisons are read from the exact-setting report CSV; common-base TIES, greedy-soup, and weight-average controls are recomputed on the exact saved 7200--7202 checkpoints because the aggregate same-base CSV now represents a later seed range. Completed training is not rerun.

## Outcome

Evaluated rows: `{len(evaluated)}`. Blocked integration/status rows: `{len(blocked)}`. Successful rows are labeled `adapter_assisted_official_core`, not unmodified official end-to-end runs. Failed or legally blocked methods have no metric row.

The integration supports comparison only on the exact checkpoint families and regimes in `reports/csv/post_iclr_official_baseline_runs.csv`. It does not support a broad official-baseline or SOTA claim.
"""
    (ROOT / "external_baselines" / "POST_ICLR_INTEGRATION.md").write_text(integration, encoding="utf-8")

    summary_lines = [
        "# Post-ICLR Official Baseline Report",
        "",
        f"Exact command: `{args.command}`",
        "",
        f"Execution commit: `{args.execution_commit}`; worktree dirty before execution: `{args.starting_worktree_dirty}`.",
        "",
        "## Successful adapter-assisted official cores",
        "",
        "| Regime | Baseline | Rows | Seeds | Failed | Mean score | Median | SD | 95% CI | Mean delta vs internal same method | Delta 95% CI | Wins/ties/losses |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- | --- |",
    ]
    for row in summary:
        summary_lines.append(f"| {row['regime']} | {row['baseline']} | {row['n_rows']} | {row['n_unique_seeds']} | {row['failed_run_count']} | {row['mean_score']:.4f} | {row['median_score']:.4f} | {row['std_score']:.4f} | [{row['score_ci_low']:.4f}, {row['score_ci_high']:.4f}] | {row['mean_delta_vs_internal_same_method']:.4f} | [{row['delta_ci_low']:.4f}, {row['delta_ci_high']:.4f}] | {row['wins_vs_internal_same_method']}/{row['ties_vs_internal_same_method']}/{row['losses_vs_internal_same_method']} |")
    independent_summary = [row for row in summary if row["regime"] == "independent_initialization"]
    summary_lines.extend([
        "",
        "The bootstrap unit is the exact checkpoint setting; the seed column reports unique training-group seeds. Same-method deltas compare official Git Re-Basin with the internal pairwise implementation, official C2M3 with the internal C2M3-style implementation, and official TIES with the internal TIES-style implementation.",
        "",
        "## Independent-regime comparison context",
        "",
        "| Official core | Delta vs greedy soup | Delta vs TwistedMerge gauge | Delta vs TwistedMerge selector | Delta vs prediction ensemble upper bound |",
        "| --- | --- | --- | --- | --- |",
        *[
            f"| {row['baseline']} | {row['mean_delta_vs_greedy_soup']:.4f} [{row['delta_vs_greedy_soup_ci_low']:.4f}, {row['delta_vs_greedy_soup_ci_high']:.4f}] | {row['mean_delta_vs_twistedmerge_gauge']:.4f} [{row['delta_vs_twistedmerge_gauge_ci_low']:.4f}, {row['delta_vs_twistedmerge_gauge_ci_high']:.4f}] | {row['mean_delta_vs_twistedmerge_selector']:.4f} [{row['delta_vs_twistedmerge_selector_ci_low']:.4f}, {row['delta_vs_twistedmerge_selector_ci_high']:.4f}] | {row['mean_delta_vs_prediction_ensemble']:.4f} [{row['delta_vs_prediction_ensemble_ci_low']:.4f}, {row['delta_vs_prediction_ensemble_ci_high']:.4f}] |"
            for row in independent_summary
        ],
        "",
        "Greedy soup, TwistedMerge gauge/selector, and the prediction ensemble are pre-existing internal controls evaluated on the same checkpoint settings. The ensemble uses N branches and N-times inference and is an upper bound, not a same-cost candidate.",
        "",
        "## Blocked methods and negative integrations",
        "",
        *[f"- `{row['baseline']}`: `{row['status']}` -- {row.get('failed_reason', '')}" for row in blocked],
        "",
        "## Claim decision",
        "",
        "The run establishes that official Git Re-Basin and C2M3 matching cores, and the official TIES merge core, can be connected to exact TwistedMerge checkpoint families through explicit adapters. Whether any paired delta is positive is reported numerically and is not generalized beyond these checkpoints. No official Model Soups, Task Arithmetic, or DARE performance result is claimed.",
        "",
        "All evaluated outputs are same-capacity single models with 1x inference. No lift or ensemble is included in the official-core score table.",
    ])
    (ROOT / "reports" / "post_iclr_official_baseline_report.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--jax-python", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--independent-checkpoint-root", type=Path, required=True)
    parser.add_argument("--common-checkpoint-root", type=Path, required=True)
    parser.add_argument("--seeds", default="1800,1801,1802,1803,1804")
    parser.add_argument("--model-counts", default="3,4")
    parser.add_argument("--widths", default="32,64")
    parser.add_argument("--common-seeds", default="7200,7201,7202")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--git-rebasin-max-iter", type=int, default=100)
    parser.add_argument("--c2m3-max-iter", type=int, default=30)
    parser.add_argument("--ties-densities", default="0.2,0.5,1.0")
    parser.add_argument("--ties-scales", default="0.5,1.0,1.25")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.seeds = parse_csv(args.seeds, int)
    args.model_counts = parse_csv(args.model_counts, int)
    args.widths = parse_csv(args.widths, int)
    args.common_seeds = parse_csv(args.common_seeds, int)
    args.ties_densities = parse_csv(args.ties_densities, float)
    args.ties_scales = parse_csv(args.ties_scales, float)
    if args.smoke:
        args.seeds = args.seeds[:1]
        args.model_counts = args.model_counts[:1]
        args.widths = args.widths[:1]
        args.common_seeds = args.common_seeds[:1]
        args.git_rebasin_max_iter = min(args.git_rebasin_max_iter, 5)
        args.c2m3_max_iter = min(args.c2m3_max_iter, 3)
        args.bootstrap_samples = min(args.bootstrap_samples, 200)
    args.command = " ".join([sys.executable, *sys.argv])
    args.execution_commit = git_output(ROOT, "rev-parse", "HEAD")
    args.starting_worktree_dirty = bool(git_output(ROOT, "status", "--short"))
    args.confirmatory_command = args.command.removesuffix(" --smoke")
    args.smoke_command = f"{args.confirmatory_command} --smoke"

    internal_independent = pd.read_csv(ROOT / "reports" / "csv" / "external_baseline_comparison.csv")
    spec, train_data, test_data = load_dataset("mnist", args.data_dir, 5000, 0, 8128)
    rows = [
        *run_independent(args, spec, train_data, test_data, internal_independent),
        *run_common_base(args),
        *status_rows(args),
    ]
    summary = summarize(rows, args.bootstrap_samples)

    write_csv(ROOT / "reports" / "csv" / "post_iclr_official_baseline_runs.csv", rows, RUN_FIELDS)
    write_csv(ROOT / "reports" / "csv" / "post_iclr_official_baseline_summary.csv", summary)
    write_csv(ROOT / "reports" / "csv" / "post_iclr_baseline_regime_audit.csv", baseline_regime_rows())
    write_plot(summary, ROOT / "reports" / "plots" / "post_iclr_official_baseline_deltas.pdf")
    write_latex(summary, ROOT / "reports" / "tables" / "post_iclr_official_baseline.tex")
    write_documentation(args, rows, summary)

    import torch
    import torchvision

    jax_environment = json.loads(subprocess.check_output(
        [
            str(args.jax_python),
            "-c",
            "import json,jax,jaxlib,scipy; print(json.dumps({'jax':jax.__version__,'jaxlib':jaxlib.__version__,'scipy':scipy.__version__,'devices':[str(x) for x in jax.devices()]}))",
        ],
        text=True,
    ))
    config = {
        "command": args.command,
        "smoke_command": args.smoke_command,
        "confirmatory_command": args.confirmatory_command,
        "execution_commit": args.execution_commit,
        "worktree_dirty_at_start": args.starting_worktree_dirty,
        "worktree_dirty_during_artifact_generation": bool(git_output(ROOT, "status", "--short")),
        "smoke": args.smoke,
        "official_root": str(args.official_root),
        "jax_python": str(args.jax_python),
        "data_dir": str(args.data_dir),
        "independent_checkpoint_root": str(args.independent_checkpoint_root),
        "common_checkpoint_root": str(args.common_checkpoint_root),
        "seeds": args.seeds,
        "model_counts": args.model_counts,
        "widths": args.widths,
        "common_seeds": args.common_seeds,
        "git_rebasin_max_iter": args.git_rebasin_max_iter,
        "c2m3_max_iter": args.c2m3_max_iter,
        "ties_densities": args.ties_densities,
        "ties_scales": args.ties_scales,
        "bootstrap_samples": args.bootstrap_samples,
        "protocol": {
            "independent_dataset": "torchvision MNIST; ToTensor only; local downloaded copy",
            "independent_training_provenance": "reused external_baseline_comparison checkpoints; Adam lr=0.001; 3 epochs; 5000 sampled train examples; validation fraction=0.2; full 10000-example test set; dataset seed=8128",
            "common_dataset": "torchvision MNIST; ToTensor only; 6000 sampled train examples; 2000 sampled test examples; dataset seed=314159",
            "common_training_provenance": "reused same_base_task_vector checkpoints; AdamW lr=0.001 base and 0.0005 fine-tune; cosine schedule; 3 base epochs; 2 fine-tune epochs; validation fraction=0.2",
            "selection": "validation accuracy then validation loss; test evaluated after selection",
            "bootstrap_unit": "exact checkpoint setting",
            "training_compute": "checkpoint reuse; no training performed",
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "torch_device": str(device_from_arg(args.device)),
            "git_rebasin_worker": jax_environment,
        },
        "sources": {baseline: {key: str(value) for key, value in source_metadata(args, baseline).items() if key != "source"} for baseline in REPOSITORIES},
    }
    config_path = ROOT / "reports" / "configs" / "post_iclr_official_baseline_config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    artifacts = []
    for path in [
        ROOT / "external_baselines" / "POST_ICLR_INTEGRATION.md",
        ROOT / "external_baselines" / "patches" / "c2m3_cpu_device.patch",
        ROOT / "reports" / "post_iclr_official_baseline_report.md",
        ROOT / "reports" / "csv" / "post_iclr_official_baseline_runs.csv",
        ROOT / "reports" / "csv" / "post_iclr_official_baseline_summary.csv",
        ROOT / "reports" / "csv" / "post_iclr_baseline_regime_audit.csv",
        ROOT / "reports" / "plots" / "post_iclr_official_baseline_deltas.pdf",
        ROOT / "reports" / "tables" / "post_iclr_official_baseline.tex",
        config_path,
    ]:
        artifacts.append({"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha256(path)})
    write_csv(ROOT / "reports" / "csv" / "post_iclr_official_baseline_artifacts.csv", artifacts)


if __name__ == "__main__":
    main()
