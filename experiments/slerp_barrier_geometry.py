#!/usr/bin/env python
"""SLERP and interpolation-barrier geometry diagnostics.

This script is intentionally a report/data generator, not a training benchmark.
It consumes saved same-base task-vector checkpoints and fixed-setting
independent-seed checkpoints, then compares path barriers for linear
interpolation, SLERP, and alignment-conditioned linear paths.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.model_merging_fixed_setting_verification import (  # noqa: E402
    layer_reference_perms,
    split_indices as fixed_split_indices,
    synced_layer_perms,
    synchronize_alignment_bundle,
)
from experiments.same_base_task_vector_benchmark import (  # noqa: E402
    combined_loader,
    make_subset,
    sample_indices,
    split_indices as same_base_split_indices,
    slerp as slerp_vector,
    state_vector,
    subset_by_classes,
    vector_to_model,
)
from src.model_merging_benchmark import (  # noqa: E402
    DatasetSpec,
    average_models,
    clone_model,
    compute_layerwise_pairwise_permutations,
    cycle_score,
    device_from_arg,
    evaluate_model,
    format_markdown_table,
    greedy_soup,
    load_dataset,
    make_loader,
    make_model,
    model_layer_widths,
    permute_model_to_reference,
    primary_pairwise_permutations,
    require_torch,
)
from src.monomial_gauge_alignment import (  # noqa: E402
    apply_monomial_alignment_to_reference,
    estimate_pairwise_monomial_alignments,
)


ROW_CSV = "slerp_barrier_geometry.csv"
SUMMARY_CSV = "slerp_barrier_geometry_summary.csv"
REPORT_MD = "slerp_barrier_geometry_report.md"
PLOT_PDF = "slerp_vs_linear_barriers.pdf"


@dataclass
class LoadedRun:
    regime: str
    setting_id: str
    run_id: str
    dataset: str
    architecture: str
    width: int
    n_models: int
    seed: int
    spec: DatasetSpec
    models: list
    val_loader: object
    test_loader: object
    match_loader: object
    metadata: dict


def parse_csv(text: str, cast=str) -> list:
    return [cast(item.strip()) for item in str(text).split(",") if item.strip()]


def safe_float(value) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def safe_mean(values) -> float:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if len(arr) else float("nan")


def safe_pearson(x, y) -> float:
    x_arr = pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(dtype=float)
    y_arr = pd.to_numeric(pd.Series(y), errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    if int(mask.sum()) < 3:
        return float("nan")
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]
    if float(np.std(x_arr)) <= 1e-12 or float(np.std(y_arr)) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def bootstrap_corr_ci(x, y, samples: int, seed: int) -> tuple[float, float]:
    x_arr = pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(dtype=float)
    y_arr = pd.to_numeric(pd.Series(y), errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]
    if len(x_arr) < 3:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(max(1, int(samples))):
        idx = rng.integers(0, len(x_arr), len(x_arr))
        corr = safe_pearson(x_arr[idx], y_arr[idx])
        if math.isfinite(corr):
            estimates.append(corr)
    if not estimates:
        return float("nan"), float("nan")
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def resolve_checkpoint_path(path_text: str | Path) -> Path:
    path = Path(str(path_text))
    if path.exists():
        return path
    if not path.is_absolute():
        candidate = ROOT / path
        if candidate.exists():
            return candidate
    text = str(path_text)
    for marker in [
        "reports/checkpoints/same_base_task_vector/",
        "reports/checkpoints/fixed_setting_verification/",
    ]:
        if marker in text:
            candidate = ROOT / text[text.index(marker) :]
            if candidate.exists():
                return candidate
    return path


def load_checkpoint_model(path: Path, architecture: str, spec: DatasetSpec, width: int):
    torch, _, _ = require_torch()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = make_model(architecture, spec, width)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def checkpoint_metadata(path: Path) -> dict:
    torch, _, _ = require_torch()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return dict(payload.get("metadata", {}))


def limited_eval(model, loader, device, max_batches: int) -> dict[str, float]:
    if int(max_batches) <= 0:
        return evaluate_model(model, loader, device)
    torch, _, _ = require_torch()
    model.to(device)
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(loader):
            if batch_idx >= int(max_batches):
                break
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            total_loss += float(torch.nn.functional.cross_entropy(logits, y, reduction="sum").item())
            correct += int((logits.argmax(dim=1) == y).sum().item())
            total += int(y.numel())
    model.to("cpu")
    return {"loss": total_loss / max(total, 1), "accuracy": correct / max(total, 1), "n_examples": float(total)}


def linear_interpolate_model(start_model, end_model, architecture: str, spec: DatasetSpec, width: int, t: float):
    torch, _, _ = require_torch()
    out = clone_model(start_model, architecture, spec, width)
    start_state = start_model.state_dict()
    end_state = end_model.state_dict()
    with torch.no_grad():
        state = out.state_dict()
        for key in state:
            state[key].copy_((1.0 - float(t)) * start_state[key].detach().cpu() + float(t) * end_state[key].detach().cpu())
        out.load_state_dict(state)
    out.eval()
    return out


def slerp_interpolate_model(start_model, end_model, architecture: str, spec: DatasetSpec, width: int, t: float):
    v0, meta = state_vector(start_model)
    v1, _ = state_vector(end_model)
    return vector_to_model(slerp_vector(v0, v1, float(t)), meta, architecture, spec, width)


def path_metrics_from_factory(
    model_factory: Callable[[float], object],
    loader,
    device,
    t_grid: list[float],
    max_batches: int,
) -> dict[str, float]:
    torch, _, _ = require_torch()
    models_by_t = {float(t): model_factory(float(t)) for t in t_grid}
    if 0.5 not in models_by_t:
        models_by_t[0.5] = model_factory(0.5)
    if 0.0 not in models_by_t:
        models_by_t[0.0] = model_factory(0.0)
    if 1.0 not in models_by_t:
        models_by_t[1.0] = model_factory(1.0)
    for model in models_by_t.values():
        model.to(device)
        model.eval()
    accum = {t: {"loss_sum": 0.0, "correct": 0, "total": 0} for t in models_by_t}
    disagree = 0
    disagree_total = 0
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(loader):
            if int(max_batches) > 0 and batch_idx >= int(max_batches):
                break
            x = x.to(device)
            y = y.to(device)
            logits_by_t = {}
            for t, model in models_by_t.items():
                logits = model(x)
                logits_by_t[t] = logits
                accum[t]["loss_sum"] += float(torch.nn.functional.cross_entropy(logits, y, reduction="sum").item())
                accum[t]["correct"] += int((logits.argmax(dim=1) == y).sum().item())
                accum[t]["total"] += int(y.numel())
            endpoint_logits = 0.5 * (logits_by_t[0.0] + logits_by_t[1.0])
            disagree += int((endpoint_logits.argmax(dim=1) != logits_by_t[0.5].argmax(dim=1)).sum().item())
            disagree_total += int(x.shape[0])
    rows = [
        {
            "t": t,
            "loss": values["loss_sum"] / max(values["total"], 1),
            "accuracy": values["correct"] / max(values["total"], 1),
            "n_examples": values["total"],
        }
        for t, values in sorted(accum.items())
    ]
    for model in models_by_t.values():
        model.to("cpu")
    loss_by_t = {row["t"]: float(row["loss"]) for row in rows}
    acc_by_t = {row["t"]: float(row["accuracy"]) for row in rows}
    loss0 = loss_by_t[0.0]
    loss1 = loss_by_t[1.0]
    acc0 = acc_by_t[0.0]
    acc1 = acc_by_t[1.0]
    mid_loss = loss_by_t[0.5]
    mid_acc = acc_by_t[0.5]
    max_loss_barrier = max(
        float(row["loss"]) - ((1.0 - float(row["t"])) * loss0 + float(row["t"]) * loss1)
        for row in rows
    )
    return {
        "loss_t0": float(loss0),
        "loss_t05": float(mid_loss),
        "loss_t1": float(loss1),
        "accuracy_t0": float(acc0),
        "accuracy_t05": float(mid_acc),
        "accuracy_t1": float(acc1),
        "midpoint_loss_barrier": float(mid_loss - 0.5 * (loss0 + loss1)),
        "max_loss_barrier": float(max_loss_barrier),
        "accuracy_drop_barrier_t05": float(0.5 * (acc0 + acc1) - mid_acc),
        "logit_interpolation_disagreement_t05": float(disagree / max(disagree_total, 1)),
        "n_examples": float(max(row["n_examples"] for row in rows) if rows else 0),
        "t_grid_json": json.dumps([float(t) for t in t_grid], separators=(",", ":")),
    }


def path_metrics(
    start_model,
    end_model,
    path_family: str,
    architecture: str,
    spec: DatasetSpec,
    width: int,
    loader,
    device,
    t_grid: list[float],
    max_batches: int,
) -> dict[str, float]:
    if path_family == "linear":
        factory = lambda t: linear_interpolate_model(start_model, end_model, architecture, spec, width, t)
    elif path_family == "slerp":
        factory = lambda t: slerp_interpolate_model(start_model, end_model, architecture, spec, width, t)
    else:
        raise ValueError(f"unknown path family: {path_family}")
    return path_metrics_from_factory(factory, loader, device, t_grid, max_batches)


def task_rows_from_json(payload: str) -> list[dict]:
    rows = json.loads(str(payload))
    return [
        {
            "task_name": str(row["task_name"]),
            "classes": tuple(int(item) for item in row["classes"]),
        }
        for row in rows
    ]


def same_base_loaders(args: argparse.Namespace, row: pd.Series) -> tuple[DatasetSpec, object, object, object]:
    dataset = str(row["dataset"])
    seed = int(row["seed"])
    spec, train_base, test_base = load_dataset(
        dataset,
        args.data_dir,
        int(row.get("max_train_samples", args.max_train_samples)),
        int(row.get("max_test_samples", args.max_test_samples)),
        int(args.dataset_seed),
        augmentation=str(args.augmentation),
    )
    _train_indices, val_indices = same_base_split_indices(len(train_base), float(args.val_fraction), int(args.dataset_seed) + 17 + seed)
    task_defs = task_rows_from_json(row["task_definitions_json"])
    task_val_loaders = {}
    task_test_loaders = {}
    for task_idx, task in enumerate(task_defs):
        val_task_indices = subset_by_classes(
            train_base,
            val_indices,
            task["classes"],
            int(args.max_task_val_samples),
            seed + 2000 + task_idx,
        )
        test_task_indices = subset_by_classes(
            test_base,
            list(range(len(test_base))),
            task["classes"],
            int(args.max_task_test_samples),
            seed + 3000 + task_idx,
        )
        task_val_loaders[task["task_name"]] = make_loader(
            make_subset(train_base, val_task_indices),
            int(args.batch_size),
            shuffle=False,
            seed=seed + 5000 + task_idx,
        )
        task_test_loaders[task["task_name"]] = make_loader(
            make_subset(test_base, test_task_indices),
            int(args.batch_size),
            shuffle=False,
            seed=seed + 6000 + task_idx,
        )
    val_loader = combined_loader(task_val_loaders, int(args.batch_size), seed + 7000)
    test_loader = combined_loader(task_test_loaders, int(args.batch_size), seed + 8000)
    return spec, val_loader, test_loader, val_loader


def load_same_base_runs(args: argparse.Namespace) -> list[LoadedRun]:
    path = Path(args.same_base_csv)
    if not path.exists():
        return []
    rows = pd.read_csv(path)
    rows = rows[rows["method"].astype(str).eq("base_model")].drop_duplicates("run_id").copy()
    if args.same_base_setting:
        rows = rows[rows["setting_id"].astype(str).isin(parse_csv(args.same_base_setting, str))]
    rows = rows.sort_values(["setting_id", "seed"])
    if int(args.max_same_base_runs) > 0:
        rows = rows.head(int(args.max_same_base_runs))
    out: list[LoadedRun] = []
    for row in rows.itertuples(index=False):
        row_series = pd.Series(row._asdict())
        checkpoint_paths = [resolve_checkpoint_path(item) for item in json.loads(str(row_series["task_checkpoints_json"]))]
        if not checkpoint_paths or not all(path.exists() for path in checkpoint_paths):
            continue
        spec, val_loader, test_loader, match_loader = same_base_loaders(args, row_series)
        models = [
            load_checkpoint_model(path, str(row_series["architecture"]), spec, int(row_series["width"]))
            for path in checkpoint_paths
        ]
        out.append(
            LoadedRun(
                regime="same_base_task_vector",
                setting_id=str(row_series["setting_id"]),
                run_id=str(row_series["run_id"]),
                dataset=str(row_series["dataset"]),
                architecture=str(row_series["architecture"]),
                width=int(row_series["width"]),
                n_models=len(models),
                seed=int(row_series["seed"]),
                spec=spec,
                models=models,
                val_loader=val_loader,
                test_loader=test_loader,
                match_loader=match_loader,
                metadata={
                    "task_preset": str(row_series.get("task_preset", "")),
                    "domain_shift": "same_base_tasks",
                    "base_checkpoint": str(resolve_checkpoint_path(row_series["base_checkpoint"])),
                    "task_checkpoints_json": json.dumps([str(path) for path in checkpoint_paths], separators=(",", ":")),
                    "task_vector_sign_conflict_fraction": safe_float(row_series.get("task_vector_sign_conflict_fraction")),
                    "task_vector_mean_pairwise_cosine": safe_float(row_series.get("task_vector_mean_pairwise_cosine")),
                    "input_artifact": str(path),
                },
            )
        )
    return out


def fixed_setting_loaders(args: argparse.Namespace, meta: dict) -> tuple[DatasetSpec, object, object, object]:
    torch, _, _ = require_torch()
    dataset_seed = int(meta.get("dataset_seed", args.dataset_seed))
    batch_size = int(meta.get("batch_size", args.batch_size))
    spec, train_base, test_base = load_dataset(
        str(meta["dataset"]),
        args.data_dir,
        int(meta.get("max_train_samples", args.max_train_samples)),
        int(meta.get("max_test_samples", args.max_test_samples)),
        dataset_seed,
        augmentation=str(meta.get("augmentation", args.augmentation)),
    )
    _train_indices, val_indices = fixed_split_indices(
        len(train_base),
        float(meta.get("val_fraction", args.val_fraction)),
        dataset_seed + 17,
    )
    val_subset = torch.utils.data.Subset(train_base, val_indices)
    val_loader = make_loader(val_subset, batch_size, shuffle=False, seed=dataset_seed + 100)
    test_loader = make_loader(test_base, batch_size, shuffle=False, seed=dataset_seed + 200)
    match_loader = make_loader(val_subset, batch_size, shuffle=False, seed=dataset_seed + 300)
    return spec, val_loader, test_loader, match_loader


def discover_fixed_checkpoint_groups(args: argparse.Namespace) -> list[tuple[str, int, list[Path], dict]]:
    root = Path(args.fixed_checkpoint_dir)
    if not root.exists():
        return []
    allowed_settings = set(parse_csv(args.fixed_settings, str)) if str(args.fixed_settings).strip() else set()
    groups = []
    total = 0
    for setting_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        if allowed_settings and setting_dir.name not in allowed_settings:
            continue
        by_seed: dict[int, dict[int, Path]] = defaultdict(dict)
        for path in sorted(setting_dir.glob("seed*_model*.pt")):
            match = re.match(r"seed(\d+)_model(\d+)\.pt$", path.name)
            if not match:
                continue
            by_seed[int(match.group(1))][int(match.group(2))] = path
        for seed in sorted(by_seed):
            if int(args.max_fixed_runs_per_setting) > 0 and sum(1 for item in groups if item[0] == setting_dir.name) >= int(args.max_fixed_runs_per_setting):
                break
            if int(args.max_fixed_total_runs) > 0 and total >= int(args.max_fixed_total_runs):
                break
            paths_by_model = by_seed[seed]
            first_path = paths_by_model[min(paths_by_model)]
            meta = checkpoint_metadata(first_path)
            n_models = int(meta.get("n_models", len(paths_by_model)))
            paths = [paths_by_model[idx] for idx in range(n_models) if idx in paths_by_model]
            if len(paths) != n_models:
                continue
            groups.append((setting_dir.name, seed, paths, meta))
            total += 1
        if int(args.max_fixed_total_runs) > 0 and total >= int(args.max_fixed_total_runs):
            break
    return groups


def load_fixed_runs(args: argparse.Namespace) -> list[LoadedRun]:
    out: list[LoadedRun] = []
    for setting_id, seed, paths, meta in discover_fixed_checkpoint_groups(args):
        spec, val_loader, test_loader, match_loader = fixed_setting_loaders(args, meta)
        models = [
            load_checkpoint_model(path, str(meta["architecture"]), spec, int(meta["width"]))
            for path in paths
        ]
        out.append(
            LoadedRun(
                regime="fixed_independent_seed",
                setting_id=setting_id,
                run_id=f"{setting_id}_seed{seed}",
                dataset=str(meta["dataset"]),
                architecture=str(meta["architecture"]),
                width=int(meta["width"]),
                n_models=int(meta.get("n_models", len(models))),
                seed=int(seed),
                spec=spec,
                models=models,
                val_loader=val_loader,
                test_loader=test_loader,
                match_loader=match_loader,
                metadata={
                    "domain_shift": str(meta.get("domain_shift", "none")),
                    "matching": str(meta.get("matching", "activation")),
                    "checkpoint_paths_json": json.dumps([str(path) for path in paths], separators=(",", ":")),
                    "input_artifact": str(args.fixed_checkpoint_dir),
                    "epochs": int(meta.get("epochs", -1)),
                },
            )
        )
    return out


def alignment_predictors(pairwise_by_layer: dict[str, dict[tuple[int, int], np.ndarray]], architecture: str, models: list) -> dict:
    n_models = len(models)
    widths = model_layer_widths(models[0], architecture)
    layer_scores = []
    for layer, pairwise in pairwise_by_layer.items():
        score, _rows = cycle_score(pairwise, n_models, int(widths[layer]))
        layer_scores.append(float(score))
    _sync_ref, _synced, sync_disagreement = synchronize_alignment_bundle(pairwise_by_layer, n_models)
    primary = primary_pairwise_permutations(pairwise_by_layer, architecture)
    primary_width = len(primary[(0, 0)])
    primary_cycle, _triangles = cycle_score(primary, n_models, primary_width)
    nonidentity = []
    for i, j in combinations(range(n_models), 2):
        nonidentity.append(float(np.mean(primary[(i, j)] != np.arange(primary_width))))
    return {
        "mean_cycle_score": float(np.mean(layer_scores)) if layer_scores else 0.0,
        "max_cycle_score": float(np.max(layer_scores)) if layer_scores else 0.0,
        "triangle_cycle_score": float(primary_cycle),
        "sync_disagreement": float(sync_disagreement),
        "pairwise_permutation_nonidentity_mean": float(np.mean(nonidentity)) if nonidentity else 0.0,
    }


def build_alignment_cache(run: LoadedRun, args: argparse.Namespace, device) -> dict:
    pairwise_by_layer = compute_layerwise_pairwise_permutations(
        run.models,
        run.architecture,
        run.match_loader,
        device,
        str(args.matching),
    )
    sync_ref, synced_by_layer, sync_disagreement = synchronize_alignment_bundle(pairwise_by_layer, len(run.models))
    cache = {
        "pairwise_by_layer": pairwise_by_layer,
        "sync_ref": sync_ref,
        "synced_by_layer": synced_by_layer,
        "sync_disagreement_for_alignment": sync_disagreement,
        "predictors": alignment_predictors(pairwise_by_layer, run.architecture, run.models),
    }
    if "monomial_shrinkage_aligned_linear" in args.methods and run.architecture == "mlp2":
        try:
            cache["monomial_alignments"] = estimate_pairwise_monomial_alignments(
                run.models,
                run.match_loader,
                device,
                matching="monomial_shrinkage_mlp2",
                max_batches=int(args.feature_batches),
                scale_method="shrinkage",
                log_scale_clip=float(args.monomial_log_scale_clip),
                shrinkage=float(args.monomial_shrinkage),
                activation_similarity_threshold=float(args.monomial_activation_similarity_threshold),
            )
            cache["monomial_status"] = "ok"
        except Exception as exc:
            cache["monomial_alignments"] = None
            cache["monomial_status"] = f"failed:{repr(exc)}"
    return cache


def endpoints_for_method(
    method: str,
    run: LoadedRun,
    model_i: int,
    model_j: int,
    cache: dict,
) -> tuple[object | None, object | None, str, dict]:
    details = {
        "aligned_path": False,
        "uses_validation_data": False,
        "common_base_required": method == "slerp_interpolation",
        "path_family": "linear",
    }
    if method == "linear_interpolation":
        return run.models[model_i], run.models[model_j], "linear", details
    if method == "slerp_interpolation":
        details["path_family"] = "slerp"
        return run.models[model_i], run.models[model_j], "slerp", details
    if method == "git_rebasin_aligned_linear":
        pairwise_by_layer = cache["pairwise_by_layer"]
        end = permute_model_to_reference(
            run.models[model_j],
            run.architecture,
            run.spec,
            run.width,
            layer_reference_perms(pairwise_by_layer, model_i, model_j),
        )
        details["aligned_path"] = True
        details["alignment_reference"] = f"pairwise_ref_{model_i}"
        return run.models[model_i], end, "linear", details
    if method == "c2m3_aligned_linear":
        synced_by_layer = cache["synced_by_layer"]
        start = permute_model_to_reference(
            run.models[model_i],
            run.architecture,
            run.spec,
            run.width,
            synced_layer_perms(synced_by_layer, model_i),
        )
        end = permute_model_to_reference(
            run.models[model_j],
            run.architecture,
            run.spec,
            run.width,
            synced_layer_perms(synced_by_layer, model_j),
        )
        details["aligned_path"] = True
        details["alignment_reference"] = str(cache.get("sync_ref", ""))
        details["sync_disagreement_for_alignment"] = safe_float(cache.get("sync_disagreement_for_alignment"))
        return start, end, "linear", details
    if method == "monomial_shrinkage_aligned_linear":
        alignments = cache.get("monomial_alignments")
        if alignments is None:
            details["skip_reason"] = str(cache.get("monomial_status", "monomial_alignment_unavailable"))
            return None, None, "linear", details
        end = apply_monomial_alignment_to_reference(run.models[model_j], run.spec, run.width, alignments[(model_i, model_j)])
        alignment = alignments[(model_i, model_j)]
        details["aligned_path"] = True
        details["alignment_reference"] = f"monomial_pairwise_ref_{model_i}"
        details["monomial_mean_abs_log_scale"] = float(
            np.mean(
                [
                    np.mean(np.abs(np.log(np.maximum(alignment.positive_scales_for(layer), 1e-12))))
                    for layer in alignment.layers()
                ]
            )
        )
        details["monomial_low_similarity_fraction"] = float(alignment.low_similarity_fraction)
        return run.models[model_i], end, "linear", details
    raise ValueError(f"unknown method: {method}")


def base_row(run: LoadedRun, args: argparse.Namespace, cache: dict) -> dict:
    meta = run.metadata
    return {
        "regime": run.regime,
        "setting_id": run.setting_id,
        "run_id": run.run_id,
        "dataset": run.dataset,
        "architecture": run.architecture,
        "width": run.width,
        "n_models": run.n_models,
        "seed": run.seed,
        "domain_shift": meta.get("domain_shift", ""),
        "task_preset": meta.get("task_preset", ""),
        "matching": meta.get("matching", args.matching),
        "max_eval_batches": int(args.max_eval_batches),
        "feature_batches": int(args.feature_batches),
        "input_artifact": meta.get("input_artifact", ""),
        "status": "ok",
        **cache.get("predictors", {}),
    }


def evaluate_pair_paths(run: LoadedRun, args: argparse.Namespace, device, cache: dict) -> list[dict]:
    rows = []
    t_grid = sorted({float(t) for t in parse_csv(args.t_grid, float)})
    for model_i, model_j in combinations(range(run.n_models), 2):
        pair_id = f"{model_i}-{model_j}"
        for method in args.methods:
            if method == "greedy_soup_endpoint":
                continue
            row = {
                **base_row(run, args, cache),
                "analysis_unit": "pair",
                "pair_id": pair_id,
                "model_i": model_i,
                "model_j": model_j,
                "path_method": method,
                "capacity_matched": True,
                "single_model_path": True,
            }
            try:
                start, end, path_family, details = endpoints_for_method(method, run, model_i, model_j, cache)
                row.update(details)
                row["path_family"] = path_family
                if start is None or end is None:
                    row["status"] = "skipped"
                    row["skip_reason"] = row.get("skip_reason", "endpoint_unavailable")
                    rows.append(row)
                    continue
                val_metrics = path_metrics(
                    start,
                    end,
                    path_family,
                    run.architecture,
                    run.spec,
                    run.width,
                    run.val_loader,
                    device,
                    t_grid,
                    int(args.max_eval_batches),
                )
                test_metrics = path_metrics(
                    start,
                    end,
                    path_family,
                    run.architecture,
                    run.spec,
                    run.width,
                    run.test_loader,
                    device,
                    t_grid,
                    int(args.max_eval_batches),
                )
                row.update({f"val_{key}": value for key, value in val_metrics.items()})
                row.update({f"test_{key}": value for key, value in test_metrics.items()})
            except Exception as exc:
                row["status"] = "failed"
                row["skip_reason"] = repr(exc)
            rows.append(row)
    return rows


def evaluate_greedy_endpoint(run: LoadedRun, args: argparse.Namespace, device, cache: dict) -> dict:
    row = {
        **base_row(run, args, cache),
        "analysis_unit": "set",
        "pair_id": "set",
        "model_i": "",
        "model_j": "",
        "path_method": "greedy_soup_endpoint",
        "path_family": "linear",
        "aligned_path": False,
        "uses_validation_data": True,
        "common_base_required": False,
        "capacity_matched": True,
        "single_model_path": True,
    }
    try:
        soup, indices, test_metrics, trajectory = greedy_soup(
            run.models,
            run.val_loader,
            run.test_loader,
            device,
            run.architecture,
            run.spec,
            run.width,
            return_trajectory=True,
        )
        start = run.models[indices[0]]
        t_grid = sorted({float(t) for t in parse_csv(args.t_grid, float)})
        val_path = path_metrics(
            start,
            soup,
            "linear",
            run.architecture,
            run.spec,
            run.width,
            run.val_loader,
            device,
            t_grid,
            int(args.max_eval_batches),
        )
        test_path = path_metrics(
            start,
            soup,
            "linear",
            run.architecture,
            run.spec,
            run.width,
            run.test_loader,
            device,
            t_grid,
            int(args.max_eval_batches),
        )
        final_val = limited_eval(soup, run.val_loader, device, int(args.max_eval_batches))
        final_test = limited_eval(soup, run.test_loader, device, int(args.max_eval_batches))
        row.update({f"val_{key}": value for key, value in val_path.items()})
        row.update({f"test_{key}": value for key, value in test_path.items()})
        row.update(
            {
                "greedy_soup_indices": ",".join(str(idx) for idx in indices),
                "greedy_soup_size": len(indices),
                "greedy_final_val_accuracy": final_val["accuracy"],
                "greedy_final_val_loss": final_val["loss"],
                "greedy_final_test_accuracy": final_test["accuracy"],
                "greedy_final_test_loss": final_test["loss"],
                "greedy_internal_test_accuracy_full_loader": safe_float(test_metrics.get("accuracy")),
                "greedy_trajectory_json": json.dumps(trajectory, sort_keys=True, separators=(",", ":")),
            }
        )
    except Exception as exc:
        row["status"] = "failed"
        row["skip_reason"] = repr(exc)
    return row


def add_linear_deltas(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows
    rows = rows.copy()
    key_cols = ["regime", "setting_id", "run_id", "pair_id"]
    pair_rows = rows[(rows["analysis_unit"].astype(str) == "pair") & (rows["status"].astype(str) == "ok")].copy()
    linear = pair_rows[pair_rows["path_method"].astype(str).eq("linear_interpolation")]
    metrics = [
        "val_midpoint_loss_barrier",
        "val_max_loss_barrier",
        "test_midpoint_loss_barrier",
        "test_max_loss_barrier",
        "val_accuracy_drop_barrier_t05",
        "test_accuracy_drop_barrier_t05",
    ]
    keep = key_cols + [col for col in metrics if col in linear.columns]
    baseline = linear[keep].rename(columns={col: f"linear_{col}" for col in metrics if col in linear.columns})
    rows = rows.merge(baseline, on=key_cols, how="left")
    for metric in metrics:
        linear_col = f"linear_{metric}"
        if metric in rows.columns and linear_col in rows.columns:
            rows[f"delta_{metric}_vs_linear"] = pd.to_numeric(rows[metric], errors="coerce") - pd.to_numeric(rows[linear_col], errors="coerce")
            rows[f"help_{metric}_vs_linear"] = -rows[f"delta_{metric}_vs_linear"]
    return rows


def summarize(rows: pd.DataFrame, bootstrap_samples: int) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    ok = rows[rows["status"].astype(str).eq("ok")].copy()
    records = []
    group_cols = ["regime", "dataset", "architecture", "width", "n_models", "domain_shift", "task_preset", "path_method"]
    for key, group in ok.groupby(group_cols, dropna=False, sort=True):
        meta = dict(zip(group_cols, key))
        pair_group = group[group["analysis_unit"].astype(str).eq("pair")]
        val_delta = pd.to_numeric(pair_group.get("delta_val_max_loss_barrier_vs_linear"), errors="coerce")
        test_delta = pd.to_numeric(pair_group.get("delta_test_max_loss_barrier_vs_linear"), errors="coerce")
        help_values = pd.to_numeric(pair_group.get("help_val_max_loss_barrier_vs_linear"), errors="coerce")
        x_cycle = pd.to_numeric(pair_group.get("mean_cycle_score"), errors="coerce")
        corr = safe_pearson(x_cycle, help_values)
        corr_low, corr_high = bootstrap_corr_ci(x_cycle, help_values, bootstrap_samples, 9107 + len(records) * 13)
        n_help = int(np.isfinite(help_values.to_numpy(dtype=float)).sum()) if len(help_values) else 0
        mean_delta = safe_mean(val_delta)
        if str(meta["path_method"]) == "slerp_interpolation":
            if n_help >= 3 and math.isfinite(mean_delta) and mean_delta < 0.0:
                claim = "limited_slerp_lowers_validation_barrier"
            else:
                claim = "descriptive_no_slerp_barrier_reduction_claim"
        elif str(meta["path_method"]) in {"git_rebasin_aligned_linear", "c2m3_aligned_linear", "monomial_shrinkage_aligned_linear"}:
            claim = "alignment_conditioned_path_geometry"
        elif str(meta["path_method"]) == "greedy_soup_endpoint":
            claim = "validation_selected_endpoint_diagnostic"
        else:
            claim = "baseline_path_geometry"
        records.append(
            {
                **meta,
                "n_rows": int(len(group)),
                "n_pair_rows": int(len(pair_group)),
                "n_unique_runs": int(group["run_id"].nunique()),
                "n_unique_seeds": int(group["seed"].nunique()),
                "mean_val_midpoint_loss_barrier": safe_mean(group.get("val_midpoint_loss_barrier")),
                "mean_val_max_loss_barrier": safe_mean(group.get("val_max_loss_barrier")),
                "mean_test_max_loss_barrier": safe_mean(group.get("test_max_loss_barrier")),
                "mean_val_accuracy_drop_barrier_t05": safe_mean(group.get("val_accuracy_drop_barrier_t05")),
                "mean_test_accuracy_drop_barrier_t05": safe_mean(group.get("test_accuracy_drop_barrier_t05")),
                "mean_delta_val_max_loss_barrier_vs_linear": mean_delta,
                "mean_delta_test_max_loss_barrier_vs_linear": safe_mean(test_delta),
                "fraction_lowers_val_max_loss_barrier_vs_linear": float(np.nanmean((val_delta < 0.0).astype(float))) if len(val_delta) else float("nan"),
                "mean_help_val_max_loss_barrier_vs_linear": safe_mean(help_values),
                "pearson_cycle_vs_val_barrier_help": corr,
                "pearson_cycle_vs_val_barrier_help_ci_low": corr_low,
                "pearson_cycle_vs_val_barrier_help_ci_high": corr_high,
                "mean_cycle_score": safe_mean(group.get("mean_cycle_score")),
                "mean_sync_disagreement": safe_mean(group.get("sync_disagreement")),
                "claim_decision": claim,
                "claim_boundary": (
                    "SLERP is a path geometry baseline, not a descent obstruction method"
                    if str(meta["path_method"]) == "slerp_interpolation"
                    else "descriptive geometry diagnostic"
                ),
            }
        )
    return pd.DataFrame(records)


def md_table(df: pd.DataFrame, columns: list[str], max_rows: int = 50) -> str:
    if df.empty:
        return "_No rows._"
    view = df.head(max_rows).copy()
    for col in columns:
        if col not in view.columns:
            view[col] = ""
    return format_markdown_table(view[columns].to_dict("records"), columns)


def plot_barriers(summary: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11.0, 5.8))
    if summary.empty:
        ax.text(0.5, 0.5, "No SLERP barrier rows", ha="center", va="center")
        ax.set_axis_off()
    else:
        plot_df = summary[summary["path_method"].isin(["linear_interpolation", "slerp_interpolation", "git_rebasin_aligned_linear", "c2m3_aligned_linear", "monomial_shrinkage_aligned_linear"])].copy()
        plot_df = plot_df.sort_values(["regime", "path_method"])
        labels = [f"{row.regime}\n{row.path_method.replace('_', ' ')}" for row in plot_df.itertuples(index=False)]
        x = np.arange(len(plot_df))
        y = pd.to_numeric(plot_df["mean_val_max_loss_barrier"], errors="coerce").to_numpy(dtype=float)
        colors = ["tab:orange" if method == "slerp_interpolation" else "tab:blue" for method in plot_df["path_method"]]
        ax.bar(x, y, color=colors, alpha=0.82)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_ylabel("mean validation max loss barrier")
        ax.set_title("SLERP versus linear and aligned interpolation barriers")
        ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_report(args: argparse.Namespace, rows: pd.DataFrame, summary: pd.DataFrame, runtime_seconds: float, path: Path) -> None:
    slerp_rows = summary[summary["path_method"].astype(str).eq("slerp_interpolation")].copy()
    if slerp_rows.empty:
        headline = "No completed SLERP rows were generated."
    else:
        mean_delta = safe_mean(slerp_rows["mean_delta_val_max_loss_barrier_vs_linear"])
        frac = safe_mean(slerp_rows["fraction_lowers_val_max_loss_barrier_vs_linear"])
        if math.isfinite(mean_delta) and mean_delta < 0.0:
            headline = (
                f"SLERP lowered validation max-loss barriers relative to linear interpolation in this bounded run "
                f"(mean delta {mean_delta:.4f}; fraction lower {frac:.3f})."
            )
        else:
            headline = (
                f"SLERP did not lower validation max-loss barriers on average in this bounded run "
                f"(mean delta {mean_delta:.4f}; fraction lower {frac:.3f})."
            )
    report = f"""# SLERP Barrier Geometry

Generated by `experiments/slerp_barrier_geometry.py`.

## Exact Command

```bash
{args.command_string}
```

## Scope

This artifact treats SLERP as a path-geometry baseline. It is not a descent obstruction method by itself. The benchmark compares linear interpolation, SLERP, Git-ReBasin-style aligned linear paths, C2M3-style synchronized linear paths, monomial-shrinkage aligned linear paths when available, and greedy soup endpoint diagnostics.

The run uses saved checkpoints only. It does not retrain models and does not write paper prose.

## Headline

{headline}

## Inputs And Caps

- Regimes requested: `{args.regimes}`
- Same-base CSV: `{args.same_base_csv}`
- Fixed checkpoint directory: `{args.fixed_checkpoint_dir}`
- Fixed settings: `{args.fixed_settings}`
- Max same-base runs: `{args.max_same_base_runs}`
- Max fixed runs per setting: `{args.max_fixed_runs_per_setting}`
- Max fixed total runs: `{args.max_fixed_total_runs}`
- Max eval batches per loader: `{args.max_eval_batches}` (`0` means full loader)
- Feature batches for activation/monomial matching: `{args.feature_batches}`
- Runtime seconds: `{runtime_seconds:.2f}`

## Outputs

- `reports/csv/{ROW_CSV}`
- `reports/csv/{SUMMARY_CSV}`
- `reports/{REPORT_MD}`
- `reports/plots/{PLOT_PDF}`

## Method Summary

{md_table(summary, ["regime", "dataset", "n_models", "domain_shift", "task_preset", "path_method", "n_rows", "n_unique_runs", "mean_val_max_loss_barrier", "mean_test_max_loss_barrier", "mean_delta_val_max_loss_barrier_vs_linear", "fraction_lowers_val_max_loss_barrier_vs_linear", "claim_decision"], 80)}

## SLERP Versus Linear

{md_table(slerp_rows, ["regime", "dataset", "n_models", "domain_shift", "task_preset", "n_pair_rows", "mean_val_max_loss_barrier", "mean_delta_val_max_loss_barrier_vs_linear", "fraction_lowers_val_max_loss_barrier_vs_linear", "mean_help_val_max_loss_barrier_vs_linear", "claim_boundary"], 40)}

## Obstruction Prediction Of Path Help

Positive `mean_help_val_max_loss_barrier_vs_linear` means the method lowered the validation max-loss barrier relative to ordinary linear interpolation. The correlation columns are descriptive unless a later run adds enough settings and seeds for a formal claim gate.

{md_table(summary, ["regime", "dataset", "path_method", "n_pair_rows", "mean_cycle_score", "mean_sync_disagreement", "mean_help_val_max_loss_barrier_vs_linear", "pearson_cycle_vs_val_barrier_help", "pearson_cycle_vs_val_barrier_help_ci_low", "pearson_cycle_vs_val_barrier_help_ci_high"], 80)}

## Claim Boundaries

- SLERP is a path-geometry baseline, not a descent obstruction method by itself.
- If SLERP helps here, the supported wording is only that it lowers interpolation barriers in the reported regime.
- Greedy soup rows are endpoint diagnostics selected by validation accuracy; test metrics are evaluation only.
- Alignment-conditioned paths are geometry diagnostics and should not be treated as method wins without paired performance gates.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")


def update_claims_audit(summary: pd.DataFrame, path: Path) -> None:
    if not path.exists():
        return
    slerp = summary[summary["path_method"].astype(str).eq("slerp_interpolation")]
    if slerp.empty:
        decision = "Not run"
        evidence = "`reports/slerp_barrier_geometry_report.md` did not produce completed SLERP rows."
    else:
        mean_delta = safe_mean(slerp["mean_delta_val_max_loss_barrier_vs_linear"])
        decision = "Supported descriptive"
        evidence = (
            "`reports/slerp_barrier_geometry_report.md` compares SLERP against linear and aligned paths; "
            f"mean validation max-loss barrier delta versus linear is `{mean_delta:.4f}`; "
            "SLERP remains a path-geometry baseline, not a descent obstruction method."
        )
    row = (
        "| SLERP is audited as a path-geometry baseline rather than a descent obstruction method. "
        f"| {decision} | {evidence} |"
    )
    text = path.read_text(encoding="utf-8")
    marker = "SLERP is audited as a path-geometry baseline rather than a descent obstruction method."
    lines = text.splitlines()
    replaced = False
    for idx, line in enumerate(lines):
        if marker in line:
            lines[idx] = row
            replaced = True
            break
    if replaced:
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    insert_marker = "<!-- claim_audit-claim-audit:start -->"
    if insert_marker in text:
        text = text.replace(insert_marker, row + "\n\n" + insert_marker)
    else:
        text = text.rstrip() + "\n\n" + row + "\n"
    path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--regimes", default="same_base,fixed")
    parser.add_argument("--methods", default="linear_interpolation,slerp_interpolation,git_rebasin_aligned_linear,c2m3_aligned_linear,monomial_shrinkage_aligned_linear,greedy_soup_endpoint")
    parser.add_argument("--same-base-csv", type=Path, default=ROOT / "reports" / "csv" / "same_base_task_vector_benchmark.csv")
    parser.add_argument("--same-base-setting", default="")
    parser.add_argument("--max-same-base-runs", type=int, default=3)
    parser.add_argument("--fixed-checkpoint-dir", type=Path, default=ROOT / "reports" / "checkpoints" / "fixed_setting_verification")
    parser.add_argument("--fixed-settings", default="mnist_mlp2_N3_W128_none_activation,fashion_mnist_mlp2_N3_W128_none_activation")
    parser.add_argument("--max-fixed-runs-per-setting", type=int, default=2)
    parser.add_argument("--max-fixed-total-runs", type=int, default=4)
    parser.add_argument("--matching", default="activation", choices=["activation", "weight"])
    parser.add_argument("--t-grid", default="0,0.25,0.5,0.75,1")
    parser.add_argument("--max-eval-batches", type=int, default=4)
    parser.add_argument("--feature-batches", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-train-samples", type=int, default=10000)
    parser.add_argument("--max-test-samples", type=int, default=5000)
    parser.add_argument("--max-task-val-samples", type=int, default=600)
    parser.add_argument("--max-task-test-samples", type=int, default=600)
    parser.add_argument("--dataset-seed", type=int, default=314159)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--augmentation", default="none")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "reports")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--monomial-log-scale-clip", type=float, default=2.0)
    parser.add_argument("--monomial-shrinkage", type=float, default=0.5)
    parser.add_argument("--monomial-activation-similarity-threshold", type=float, default=0.2)
    parser.add_argument("--update-claims-audit", action="store_true", default=True)
    parser.add_argument("--no-update-claims-audit", action="store_false", dest="update_claims_audit")
    args = parser.parse_args()
    args.methods = parse_csv(args.methods, str)
    args.regime_list = parse_csv(args.regimes, str)
    args.command_string = " ".join([sys.executable, *sys.argv])
    return args


def main() -> None:
    args = parse_args()
    start_time = time.time()
    device = device_from_arg(str(args.device))
    csv_dir = args.reports_dir / "csv"
    plot_dir = args.reports_dir / "plots"
    csv_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    loaded_runs: list[LoadedRun] = []
    if "same_base" in args.regime_list:
        loaded_runs.extend(load_same_base_runs(args))
    if "fixed" in args.regime_list:
        loaded_runs.extend(load_fixed_runs(args))

    rows = []
    for idx, run in enumerate(loaded_runs, start=1):
        print(f"[{idx}/{len(loaded_runs)}] {run.regime} {run.run_id}", flush=True)
        cache = build_alignment_cache(run, args, device)
        rows.extend(evaluate_pair_paths(run, args, device, cache))
        if "greedy_soup_endpoint" in args.methods:
            rows.append(evaluate_greedy_endpoint(run, args, device, cache))
        for model in run.models:
            model.to("cpu")

    row_df = add_linear_deltas(pd.DataFrame(rows))
    summary = summarize(row_df, int(args.bootstrap_samples))
    row_df.to_csv(csv_dir / ROW_CSV, index=False, lineterminator="\n")
    summary.to_csv(csv_dir / SUMMARY_CSV, index=False, lineterminator="\n")
    plot_barriers(summary, plot_dir / PLOT_PDF)
    write_report(args, row_df, summary, time.time() - start_time, args.reports_dir / REPORT_MD)
    if args.update_claims_audit:
        update_claims_audit(summary, args.reports_dir / "claims_audit.md")
    print(f"wrote {csv_dir / ROW_CSV}")
    print(f"wrote {csv_dir / SUMMARY_CSV}")
    print(f"wrote {args.reports_dir / REPORT_MD}")
    print(f"wrote {plot_dir / PLOT_PDF}")


if __name__ == "__main__":
    main()
