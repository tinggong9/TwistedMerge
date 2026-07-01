#!/usr/bin/env python
"""Compute alignment-conditioned interpolation barrier targets from checkpoints."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.model_merging_fixed_setting_verification import (  # noqa: E402
    layer_reference_perms,
    split_indices,
    synced_layer_perms,
    synchronize_alignment_bundle,
)
from src.model_merging_benchmark import (  # noqa: E402
    DomainShiftDataset,
    DatasetSpec,
    average_models,
    clone_model,
    device_from_arg,
    evaluate_model,
    greedy_soup,
    load_dataset,
    make_loader,
    make_model,
    permute_model_to_reference,
    require_torch,
)
from src.monomial_gauge_alignment import (  # noqa: E402
    apply_monomial_alignment_to_reference,
    estimate_pairwise_monomial_alignments,
)


RUNS_CSV = "fixed_setting_verification_runs.csv"
INDIVIDUALS_CSV = "fixed_setting_individual_models.csv"
BARRIER_CSV = "alignment_barrier_targets.csv"
BARRIER_STATS_CSV = "alignment_barrier_target_stats.csv"
BARRIER_REPORT = "alignment_barrier_targets_report.md"
BARRIER_PLOT = "alignment_barrier_vs_obstruction.pdf"
BARRIER_STABILITY_REPORT = "alignment_barrier_stability_report.md"
BARRIER_STABILITY_PLOT = "alignment_barrier_stability.pdf"
PREVIOUS_BARRIER_PREDICTOR_STATS_CSV = "obstruction_barrier_predictor_stats.csv"

TARGET_COLUMNS = (
    "linear_mode_connectivity_barrier",
    "c2m3_barrier_delta_vs_git_rebasin",
    "c2m3_barrier_delta_vs_weight_average",
    "monomial_barrier_delta_vs_c2m3",
)


@dataclass(frozen=True)
class LoadedSetting:
    spec: DatasetSpec
    val_loader: object
    test_loader: object
    match_loader: object


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-csv", type=Path, default=ROOT / "reports" / "csv" / RUNS_CSV)
    parser.add_argument("--individuals-csv", type=Path, default=ROOT / "reports" / "csv" / INDIVIDUALS_CSV)
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "reports")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--methods", default="weight_average,git_rebasin_pairwise_ref0,c2m3_synchronized,greedy_soup,monomial_shrinkage")
    parser.add_argument("--t-grid", default="0,0.25,0.5,0.75,1")
    parser.add_argument("--max-eval-batches", type=int, default=0, help="0 means evaluate full validation/test loaders.")
    parser.add_argument("--feature-batches", type=int, default=8)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--max-seeds-per-setting", type=int, default=0, help="0 means all seeds.")
    parser.add_argument("--batch-size", type=int, default=0, help="0 uses the saved run batch size.")
    parser.add_argument("--monomial-log-scale-clip", type=float, default=2.0)
    parser.add_argument("--monomial-shrinkage", type=float, default=0.5)
    parser.add_argument("--monomial-activation-similarity-threshold", type=float, default=0.2)
    parser.add_argument(
        "--previous-barrier-predictor-stats-csv",
        type=Path,
        default=ROOT / "reports" / "csv" / PREVIOUS_BARRIER_PREDICTOR_STATS_CSV,
        help="Prior predictor-target barrier stats used only for stability comparison.",
    )
    parser.add_argument(
        "--feasibility-note",
        default="",
        help="Optional note explaining why a nonzero evaluation cap was used.",
    )
    return parser.parse_args()


def parse_csv(text: str, cast=str) -> list:
    return [cast(item.strip()) for item in str(text).split(",") if item.strip()]


def safe_float(value) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def safe_mean(values: Iterable[float]) -> float:
    arr = np.asarray([safe_float(value) for value in values], dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")


def safe_std(values: Iterable[float]) -> float:
    arr = np.asarray([safe_float(value) for value in values], dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0


def safe_pearson(x, y) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return float("nan")
    x = x[mask]
    y = y[mask]
    if float(np.std(x)) <= 1e-12 or float(np.std(y)) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def finite_std(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return 0.0
    return float(np.std(values))


def prepare_regression_design(
    x: np.ndarray,
    y: np.ndarray,
    controls: list[np.ndarray],
) -> tuple[np.ndarray | None, np.ndarray | None, int, str]:
    arrays = [np.asarray(x, dtype=float), np.asarray(y, dtype=float)]
    arrays.extend(np.asarray(control, dtype=float) for control in controls)
    mask = np.ones(len(arrays[0]), dtype=bool)
    for array in arrays:
        mask &= np.isfinite(array)
    if int(mask.sum()) < 4:
        return None, None, int(mask.sum()), "insufficient_finite_rows"
    x_f = arrays[0][mask]
    y_f = arrays[1][mask]
    if finite_std(x_f) <= 1e-12:
        return None, None, int(mask.sum()), "predictor_missing_or_constant"
    design_parts = [np.ones_like(x_f), x_f]
    for control in arrays[2:]:
        control_f = control[mask]
        if finite_std(control_f) > 1e-12:
            design_parts.append(control_f)
    return np.column_stack(design_parts), y_f, int(mask.sum()), "ok"


def regression_beta(
    x: np.ndarray,
    y: np.ndarray,
    controls: list[np.ndarray],
) -> tuple[float, int, str]:
    design, y_f, n_finite, status = prepare_regression_design(x, y, controls)
    if design is None or y_f is None:
        return float("nan"), n_finite, status
    try:
        beta = np.linalg.lstsq(design, y_f, rcond=None)[0][1]
    except np.linalg.LinAlgError:
        return float("nan"), n_finite, "singular_design"
    return float(beta), n_finite, "ok"


def bootstrap_beta_ci(
    x: np.ndarray,
    y: np.ndarray,
    controls: list[np.ndarray],
    samples: int,
    seed: int,
) -> tuple[float, float]:
    design, y_f, n, status = prepare_regression_design(x, y, controls)
    if design is None or y_f is None or status != "ok" or int(samples) <= 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(int(samples), n))
    boot_design = design[idx]
    boot_y = y_f[idx]
    xtx = np.einsum("sni,snj->sij", boot_design, boot_design)
    xty = np.einsum("sni,sn->si", boot_design, boot_y)
    coefs = np.einsum("sij,sj->si", np.linalg.pinv(xtx, rcond=1e-12), xty)
    betas = coefs[:, 1]
    betas = betas[np.isfinite(betas)]
    if betas.size == 0:
        return float("nan"), float("nan")
    return float(np.percentile(betas, 2.5)), float(np.percentile(betas, 97.5))


def bootstrap_corr_ci(x, y, samples: int, seed: int) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = len(x)
    if n < 3:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(max(0, int(samples))):
        idx = rng.integers(0, n, size=n)
        corr = safe_pearson(x[idx], y[idx])
        if math.isfinite(corr):
            vals.append(corr)
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def load_checkpoint_model(path: Path, architecture: str, spec: DatasetSpec, width: int):
    torch, _, _ = require_torch()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = make_model(architecture, spec, width)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def parse_layerwise_perms(payload: str) -> dict[str, dict[tuple[int, int], np.ndarray]]:
    raw = json.loads(payload)
    out: dict[str, dict[tuple[int, int], np.ndarray]] = {}
    for layer, maps in raw.items():
        out[layer] = {}
        for key, perm in maps.items():
            left, right = key.split("->", 1)
            out[layer][(int(left), int(right))] = np.asarray(perm, dtype=int)
    return out


def checkpoint_path(path_text: str) -> Path:
    path = Path(str(path_text))
    if path.exists():
        return path
    if not path.is_absolute():
        candidate = ROOT / path
        if candidate.exists():
            return candidate
    # Some saved CSVs contain absolute paths from the local checkout.  Keep the
    # repo-relative suffix stable if the checkout moved.
    marker = "reports/checkpoints/fixed_setting_verification/"
    text = str(path_text)
    if marker in text:
        candidate = ROOT / text[text.index(marker) :]
        if candidate.exists():
            return candidate
    return path


def make_setting_loaders(args: argparse.Namespace, meta: dict) -> LoadedSetting:
    torch, _, _ = require_torch()
    batch_size = int(args.batch_size) if int(args.batch_size) > 0 else int(meta["batch_size"])
    spec, train_base, test_base = load_dataset(
        str(meta["dataset"]),
        args.data_dir,
        int(meta["max_train_samples"]),
        int(meta["max_test_samples"]),
        int(meta["dataset_seed"]),
        augmentation=str(meta.get("augmentation", "none")),
    )
    _train_indices, val_indices = split_indices(
        len(train_base),
        float(meta.get("val_fraction", 0.2)),
        int(meta["dataset_seed"]) + 17,
    )
    val_subset = torch.utils.data.Subset(train_base, val_indices)
    val_loader = make_loader(val_subset, batch_size, shuffle=False, seed=int(meta["dataset_seed"]) + 100)
    test_loader = make_loader(test_base, batch_size, shuffle=False, seed=int(meta["dataset_seed"]) + 200)
    match_loader = make_loader(val_subset, batch_size, shuffle=False, seed=int(meta["dataset_seed"]) + 300)
    return LoadedSetting(spec=spec, val_loader=val_loader, test_loader=test_loader, match_loader=match_loader)


def interpolate_models(start_model, end_model, architecture: str, spec: DatasetSpec, width: int, t: float):
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


def evaluate_limited(model, loader, device, max_batches: int) -> dict[str, float]:
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
    return {"loss": total_loss / max(total, 1), "accuracy": correct / max(total, 1)}


def midpoint_logit_disagreement(start_model, end_model, midpoint_model, loader, device, max_batches: int) -> float:
    torch, _, _ = require_torch()
    for model in (start_model, end_model, midpoint_model):
        model.to(device)
        model.eval()
    total = 0
    disagree = 0
    with torch.no_grad():
        for batch_idx, (x, _y) in enumerate(loader):
            if int(max_batches) > 0 and batch_idx >= int(max_batches):
                break
            x = x.to(device)
            endpoint_logits = 0.5 * (start_model(x) + end_model(x))
            midpoint_logits = midpoint_model(x)
            disagree += int((endpoint_logits.argmax(dim=1) != midpoint_logits.argmax(dim=1)).sum().item())
            total += int(x.shape[0])
    for model in (start_model, end_model, midpoint_model):
        model.to("cpu")
    return float(disagree / max(total, 1))


def path_metrics(start_model, end_model, architecture: str, spec: DatasetSpec, width: int, loader, device, t_grid: list[float], max_batches: int) -> dict[str, float]:
    torch, _, _ = require_torch()
    models_by_t = {}
    for t in t_grid:
        models_by_t[float(t)] = interpolate_models(start_model, end_model, architecture, spec, width, t)
    if 0.5 not in models_by_t:
        models_by_t[0.5] = interpolate_models(start_model, end_model, architecture, spec, width, 0.5)
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
        }
        for t, values in sorted(accum.items())
    ]
    for model in models_by_t.values():
        model.to("cpu")
    loss_by_t = {row["t"]: float(row["loss"]) for row in rows}
    acc_by_t = {row["t"]: float(row["accuracy"]) for row in rows}
    loss0 = loss_by_t.get(0.0, rows[0]["loss"])
    loss1 = loss_by_t.get(1.0, rows[-1]["loss"])
    acc0 = acc_by_t.get(0.0, rows[0]["accuracy"])
    acc1 = acc_by_t.get(1.0, rows[-1]["accuracy"])
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
        "t_grid_json": json.dumps([float(t) for t in t_grid], separators=(",", ":")),
    }


def method_models(
    method: str,
    models: list,
    best_index: int,
    architecture: str,
    spec: DatasetSpec,
    width: int,
    pairwise_by_layer: dict[str, dict[tuple[int, int], np.ndarray]],
    val_loader,
    test_loader,
    match_loader,
    device,
    args: argparse.Namespace,
) -> tuple[object | None, object | None, dict]:
    details = {"meaningful": True, "selection_indices": "", "alignment_conditioned": method not in {"weight_average", "greedy_soup"}}
    if method == "weight_average":
        return models[best_index], average_models(models, architecture, spec, width), details
    if method == "git_rebasin_pairwise_ref0":
        aligned = [
            permute_model_to_reference(model, architecture, spec, width, layer_reference_perms(pairwise_by_layer, 0, idx))
            for idx, model in enumerate(models)
        ]
        return aligned[best_index], average_models(aligned, architecture, spec, width), details
    if method == "c2m3_synchronized":
        sync_ref, synced, sync_disagreement = synchronize_alignment_bundle(pairwise_by_layer, len(models))
        aligned = [
            permute_model_to_reference(model, architecture, spec, width, synced_layer_perms(synced, idx))
            for idx, model in enumerate(models)
        ]
        details["sync_reference_model"] = sync_ref
        details["sync_disagreement_for_path"] = sync_disagreement
        return aligned[best_index], average_models(aligned, architecture, spec, width), details
    if method == "greedy_soup":
        soup, indices, _test_metrics = greedy_soup(models, val_loader, test_loader, device, architecture, spec, width)
        details["selection_indices"] = ",".join(str(idx) for idx in indices)
        details["meaningful"] = len(indices) > 1
        details["alignment_conditioned"] = False
        return models[indices[0]], soup, details
    if method == "monomial_shrinkage":
        if architecture != "mlp2":
            return None, None, {"meaningful": False, "skip_reason": "monomial_shrinkage_path_requires_mlp2"}
        alignments = estimate_pairwise_monomial_alignments(
            models,
            match_loader,
            device,
            matching="monomial_shrinkage_mlp2",
            max_batches=int(args.feature_batches),
            scale_method="shrinkage",
            log_scale_clip=float(args.monomial_log_scale_clip),
            shrinkage=float(args.monomial_shrinkage),
            activation_similarity_threshold=float(args.monomial_activation_similarity_threshold),
        )
        aligned = [models[0]]
        for idx in range(1, len(models)):
            aligned.append(apply_monomial_alignment_to_reference(models[idx], spec, width, alignments[(0, idx)]))
        details["alignment_conditioned"] = True
        details["monomial_scale_method"] = "shrinkage"
        return aligned[best_index], average_models(aligned, architecture, spec, width), details
    raise ValueError(f"unknown barrier method: {method}")


def copy_predictors(row: pd.Series) -> dict:
    predictor_cols = [
        "mean_cycle_score",
        "max_cycle_score",
        "nonidentity_triangle_fraction",
        "sync_disagreement",
        "pairwise_alignment_residual_mean",
        "activation_assignment_similarity_mean",
        "combined_obstruction_score",
        "monomial_defect_score",
        "cycle_score",
    ]
    return {col: row[col] for col in predictor_cols if col in row}


def compute_barriers(args: argparse.Namespace) -> pd.DataFrame:
    torch, _, _ = require_torch()
    device = device_from_arg(str(args.device))
    runs = pd.read_csv(args.runs_csv)
    individuals = pd.read_csv(args.individuals_csv)
    weight_rows = runs[
        (runs["method"].astype(str) == "weight_average")
        & (runs["alignment_source"].astype(str) == "observed")
        & (pd.to_numeric(runs["alignment_noise_fraction"], errors="coerce") == 0.0)
    ].copy()
    methods = parse_csv(args.methods, str)
    t_grid = sorted({float(t) for t in parse_csv(args.t_grid, float)})
    if 0.0 not in t_grid or 1.0 not in t_grid or 0.5 not in t_grid:
        raise ValueError("--t-grid must include 0, 0.5, and 1")
    rows = []
    loader_cache: dict[tuple, LoadedSetting] = {}
    group_cols = ["setting_id", "dataset", "architecture", "n_models", "width", "domain_shift", "matching"]
    for setting_key, setting_group in weight_rows.groupby(group_cols, dropna=False, sort=True):
        setting_meta = dict(zip(group_cols, setting_key))
        if int(args.max_seeds_per_setting) > 0:
            keep_seeds = sorted(setting_group["seed"].unique())[: int(args.max_seeds_per_setting)]
            setting_group = setting_group[setting_group["seed"].isin(keep_seeds)].copy()
        sample = setting_group.iloc[0].to_dict()
        loader_key = (
            sample["dataset"],
            sample["max_train_samples"],
            sample["max_test_samples"],
            sample["dataset_seed"],
            sample.get("augmentation", "none"),
            sample.get("batch_size", 128),
            sample.get("val_fraction", 0.2),
        )
        if loader_key not in loader_cache:
            loader_cache[loader_key] = make_setting_loaders(args, sample)
        loaded = loader_cache[loader_key]
        setting_individuals = individuals[individuals["setting_id"].astype(str) == str(setting_meta["setting_id"])].copy()
        for run_row in setting_group.sort_values("seed").itertuples(index=False):
            row_dict = run_row._asdict()
            seed = int(row_dict["seed"])
            run_id = str(row_dict["run_id"])
            seed_individuals = setting_individuals[setting_individuals["seed"].astype(int) == seed].sort_values("model_index")
            if seed_individuals.empty:
                continue
            models = [
                load_checkpoint_model(
                    checkpoint_path(item.checkpoint_path),
                    str(setting_meta["architecture"]),
                    loaded.spec,
                    int(setting_meta["width"]),
                )
                for item in seed_individuals.itertuples()
            ]
            best_index = int(seed_individuals.sort_values(["test_accuracy", "val_accuracy"], ascending=False).iloc[0]["model_index"])
            pairwise_by_layer = parse_layerwise_perms(str(row_dict["layerwise_alignment_permutations_json"]))
            for method in methods:
                try:
                    start, end, details = method_models(
                        method,
                        models,
                        best_index,
                        str(setting_meta["architecture"]),
                        loaded.spec,
                        int(setting_meta["width"]),
                        pairwise_by_layer,
                        loaded.val_loader,
                        loaded.test_loader,
                        loaded.match_loader,
                        device,
                        args,
                    )
                except Exception as exc:  # keep the report honest and continue.
                    rows.append({**setting_meta, "seed": seed, "run_id": run_id, "method": method, "status": "failed", "skip_reason": repr(exc)})
                    continue
                if start is None or end is None:
                    rows.append({**setting_meta, "seed": seed, "run_id": run_id, "method": method, "status": "skipped", **details})
                    continue
                val = path_metrics(
                    start,
                    end,
                    str(setting_meta["architecture"]),
                    loaded.spec,
                    int(setting_meta["width"]),
                    loaded.val_loader,
                    device,
                    t_grid,
                    int(args.max_eval_batches),
                )
                test = path_metrics(
                    start,
                    end,
                    str(setting_meta["architecture"]),
                    loaded.spec,
                    int(setting_meta["width"]),
                    loaded.test_loader,
                    device,
                    t_grid,
                    int(args.max_eval_batches),
                )
                rows.append(
                    {
                        **setting_meta,
                        "seed": seed,
                        "run_id": run_id,
                        "method": method,
                        "status": "ok",
                        "max_eval_batches": int(args.max_eval_batches),
                        "best_model_index": best_index,
                        **copy_predictors(pd.Series(row_dict)),
                        **details,
                        **{f"val_{key}": value for key, value in val.items()},
                        **{f"test_{key}": value for key, value in test.items()},
                    }
                )
            for model in models:
                model.to("cpu")
    barriers = pd.DataFrame(rows)
    return add_target_columns(barriers)


def add_target_columns(barriers: pd.DataFrame) -> pd.DataFrame:
    if barriers.empty:
        return barriers
    barriers = barriers.copy()
    key_cols = ["setting_id", "run_id", "seed"]
    ok = barriers[barriers["status"].astype(str) == "ok"].copy()
    pivot = ok.pivot_table(index=key_cols, columns="method", values="val_max_loss_barrier", aggfunc="first")
    target_rows = []
    for key, row in pivot.iterrows():
        payload = dict(zip(key_cols, key))
        weight = safe_float(row.get("weight_average"))
        git = safe_float(row.get("git_rebasin_pairwise_ref0"))
        c2m3 = safe_float(row.get("c2m3_synchronized"))
        mono = safe_float(row.get("monomial_shrinkage"))
        payload.update(
            {
                "linear_mode_connectivity_barrier": c2m3,
                "c2m3_barrier_delta_vs_git_rebasin": git - c2m3 if math.isfinite(git) and math.isfinite(c2m3) else float("nan"),
                "c2m3_barrier_delta_vs_weight_average": weight - c2m3 if math.isfinite(weight) and math.isfinite(c2m3) else float("nan"),
                "monomial_barrier_delta_vs_c2m3": c2m3 - mono if math.isfinite(c2m3) and math.isfinite(mono) else float("nan"),
            }
        )
        target_rows.append(payload)
    targets = pd.DataFrame(target_rows)
    if targets.empty:
        return barriers
    return barriers.merge(targets, on=key_cols, how="left")


def compute_stats(barriers: pd.DataFrame, bootstrap_samples: int) -> pd.DataFrame:
    if barriers.empty:
        return pd.DataFrame()
    ok = barriers[barriers["status"].astype(str) == "ok"].copy()
    rows = []
    group_cols = ["dataset", "architecture", "n_models", "width", "domain_shift", "matching", "method"]
    for key, group in ok.groupby(group_cols, dropna=False, sort=True):
        meta = dict(zip(group_cols, key))
        x = pd.to_numeric(group.get("mean_cycle_score"), errors="coerce").to_numpy()
        y = pd.to_numeric(group.get("val_max_loss_barrier"), errors="coerce").to_numpy()
        corr = safe_pearson(x, y)
        low, high = bootstrap_corr_ci(x, y, bootstrap_samples, seed=3319 + len(rows) * 97)
        rows.append(
            {
                **meta,
                "n_rows": int(len(group)),
                "n_unique_seeds": int(group["seed"].nunique()),
                "mean_val_midpoint_loss_barrier": safe_mean(group["val_midpoint_loss_barrier"]),
                "mean_val_max_loss_barrier": safe_mean(group["val_max_loss_barrier"]),
                "mean_val_accuracy_drop_barrier_t05": safe_mean(group["val_accuracy_drop_barrier_t05"]),
                "mean_test_max_loss_barrier": safe_mean(group["test_max_loss_barrier"]),
                "pearson_cycle_vs_val_max_loss_barrier": corr,
                "pearson_ci_low": low,
                "pearson_ci_high": high,
                "claim_status": "descriptive_n_below_20" if int(group["seed"].nunique()) < 20 else "descriptive_barrier_target",
                "claim_supported": False,
            }
        )
    return pd.DataFrame(rows)


def load_existing_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def compare_barrier_estimates(previous: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    if previous.empty or current.empty:
        return pd.DataFrame()
    key_cols = ["setting_id", "run_id", "seed", "method"]
    if not all(col in previous.columns and col in current.columns for col in key_cols):
        return pd.DataFrame()
    metric_cols = [
        "val_midpoint_loss_barrier",
        "val_max_loss_barrier",
        "val_accuracy_drop_barrier_t05",
        "val_logit_interpolation_disagreement_t05",
        "test_max_loss_barrier",
        "linear_mode_connectivity_barrier",
        "c2m3_barrier_delta_vs_git_rebasin",
        "c2m3_barrier_delta_vs_weight_average",
        "monomial_barrier_delta_vs_c2m3",
    ]
    merged = previous[key_cols + [col for col in metric_cols if col in previous.columns]].merge(
        current[key_cols + [col for col in metric_cols if col in current.columns]],
        on=key_cols,
        how="inner",
        suffixes=("_previous", "_current"),
    )
    rows = []
    for metric in metric_cols:
        old_col = f"{metric}_previous"
        new_col = f"{metric}_current"
        if old_col not in merged.columns or new_col not in merged.columns:
            continue
        for method, group in merged.groupby("method", dropna=False, sort=True):
            old = pd.to_numeric(group[old_col], errors="coerce").to_numpy()
            new = pd.to_numeric(group[new_col], errors="coerce").to_numpy()
            mask = np.isfinite(old) & np.isfinite(new)
            diff = new[mask] - old[mask]
            rows.append(
                {
                    "metric": metric,
                    "method": method,
                    "n_pairs": int(mask.sum()),
                    "previous_mean": float(np.mean(old[mask])) if mask.any() else float("nan"),
                    "current_mean": float(np.mean(new[mask])) if mask.any() else float("nan"),
                    "mean_delta_current_minus_previous": float(np.mean(diff)) if diff.size else float("nan"),
                    "mean_abs_difference": float(np.mean(np.abs(diff))) if diff.size else float("nan"),
                    "max_abs_difference": float(np.max(np.abs(diff))) if diff.size else float("nan"),
                    "pearson_previous_current": safe_pearson(old[mask], new[mask]) if int(mask.sum()) >= 3 else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def run_level_barrier_targets(barriers: pd.DataFrame, runs_csv: Path) -> pd.DataFrame:
    if barriers.empty or not runs_csv.exists():
        return pd.DataFrame()
    runs = pd.read_csv(runs_csv)
    base = runs[
        (runs["method"].astype(str) == "weight_average")
        & (runs["alignment_source"].astype(str) == "observed")
        & (pd.to_numeric(runs["alignment_noise_fraction"], errors="coerce") == 0.0)
    ].copy()
    key_cols = ["setting_id", "run_id", "seed"]
    target_cols = [col for col in TARGET_COLUMNS if col in barriers.columns]
    if not target_cols or not all(col in base.columns and col in barriers.columns for col in key_cols):
        return pd.DataFrame()
    targets = barriers[key_cols + target_cols].drop_duplicates(subset=key_cols, keep="first")
    return base.merge(targets, on=key_cols, how="inner")


def supported_claim_stability(
    previous_predictor_stats: pd.DataFrame,
    current_barriers: pd.DataFrame,
    runs_csv: Path,
    bootstrap_samples: int,
) -> pd.DataFrame:
    if previous_predictor_stats.empty:
        return pd.DataFrame()
    supported = previous_predictor_stats[
        (previous_predictor_stats["claim_supported"] == True)  # noqa: E712
        & (previous_predictor_stats["target"].astype(str).isin(TARGET_COLUMNS))
    ].copy()
    if supported.empty:
        return pd.DataFrame()
    run_level = run_level_barrier_targets(current_barriers, runs_csv)
    if run_level.empty:
        return pd.DataFrame()
    rows = []
    for idx, claim in supported.iterrows():
        group = run_level[
            (run_level["dataset"].astype(str) == str(claim["dataset"]))
            & (run_level["architecture"].astype(str) == str(claim["architecture"]))
            & (pd.to_numeric(run_level["n_models"], errors="coerce") == int(claim["n_models"]))
            & (pd.to_numeric(run_level["width"], errors="coerce") == int(claim["width"]))
            & (run_level["domain_shift"].astype(str) == str(claim["domain_shift"]))
            & (run_level["matching"].astype(str) == str(claim["matching"]))
        ].copy()
        target = str(claim["target"])
        predictor = str(claim["predictor"])
        controls = [
            pd.to_numeric(group[col], errors="coerce").to_numpy()
            for col in ("mean_individual_accuracy", "pairwise_alignment_residual_mean")
            if col in group.columns
        ]
        x = pd.to_numeric(group[predictor], errors="coerce").to_numpy() if predictor in group.columns else np.full(len(group), np.nan)
        y = pd.to_numeric(group[target], errors="coerce").to_numpy() if target in group.columns else np.full(len(group), np.nan)
        beta, n_finite, status = regression_beta(x, y, controls)
        low, high = (
            bootstrap_beta_ci(x, y, controls, bootstrap_samples, seed=120011 + int(idx) * 1237)
            if status == "ok"
            else (float("nan"), float("nan"))
        )
        rows.append(
            {
                "dataset": claim["dataset"],
                "architecture": claim["architecture"],
                "n_models": int(claim["n_models"]),
                "width": int(claim["width"]),
                "domain_shift": claim["domain_shift"],
                "matching": claim["matching"],
                "target": target,
                "predictor": predictor,
                "previous_claim_status": claim.get("claim_status", ""),
                "previous_beta": safe_float(claim.get("predictor_beta")),
                "previous_ci_low": safe_float(claim.get("predictor_beta_ci_low")),
                "previous_ci_high": safe_float(claim.get("predictor_beta_ci_high")),
                "current_beta": beta,
                "current_ci_low": low,
                "current_ci_high": high,
                "n_unique_seeds": int(group["seed"].nunique()) if "seed" in group else 0,
                "n_finite": n_finite,
                "fit_status": status,
                "stable_positive_gate": bool(math.isfinite(low) and low > 0.0),
            }
        )
    return pd.DataFrame(rows)


def write_stability_plot(stability: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    metric = "val_max_loss_barrier"
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    if stability.empty:
        ax.text(0.5, 0.5, "No previous barrier rows available", ha="center", va="center")
        ax.set_axis_off()
    else:
        subset = stability[stability["metric"].astype(str) == metric].copy()
        if subset.empty:
            ax.text(0.5, 0.5, "No val barrier stability rows", ha="center", va="center")
            ax.set_axis_off()
        else:
            x = pd.to_numeric(subset["previous_mean"], errors="coerce")
            y = pd.to_numeric(subset["current_mean"], errors="coerce")
            labels = subset["method"].astype(str).to_list()
            ax.scatter(x, y, s=60, color="#4c78a8")
            for x_i, y_i, label in zip(x, y, labels):
                if math.isfinite(float(x_i)) and math.isfinite(float(y_i)):
                    ax.text(float(x_i), float(y_i), label, fontsize=7, ha="left", va="bottom")
            finite = np.asarray([*x[np.isfinite(x)], *y[np.isfinite(y)]], dtype=float)
            if finite.size:
                lo = float(np.min(finite))
                hi = float(np.max(finite))
                pad = max((hi - lo) * 0.08, 1e-4)
                ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="black", linewidth=0.8)
            ax.set_xlabel("previous mean validation max loss barrier")
            ax.set_ylabel("current mean validation max loss barrier")
            ax.set_title("Barrier stability: previous debug cap vs larger evaluation")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_stability_report(
    args: argparse.Namespace,
    previous_barriers: pd.DataFrame,
    current_barriers: pd.DataFrame,
    stability: pd.DataFrame,
    claim_stability: pd.DataFrame,
    runtime_seconds: float,
    path: Path,
) -> None:
    previous_cap = (
        sorted(previous_barriers["max_eval_batches"].dropna().unique().tolist())
        if not previous_barriers.empty and "max_eval_batches" in previous_barriers.columns
        else []
    )
    current_cap = (
        sorted(current_barriers["max_eval_batches"].dropna().unique().tolist())
        if not current_barriers.empty and "max_eval_batches" in current_barriers.columns
        else []
    )
    skipped = current_barriers[current_barriers["status"].astype(str) != "ok"].copy() if not current_barriers.empty else pd.DataFrame()
    cap_label = "full loaders" if int(args.max_eval_batches) == 0 else f"cap {int(args.max_eval_batches)} batches per validation/test loader"
    if int(args.max_eval_batches) == 0:
        feasibility = "Full-loader evaluation completed and should replace the previous 2-batch debug barriers for citation."
    else:
        note = args.feasibility_note.strip() or "Full-loader evaluation was not run in this pass; the report records the bounded cap used."
        feasibility = f"Largest feasible local run in this pass used {cap_label}. {note}"
    if stability.empty or "metric" not in stability.columns:
        stability_focus = pd.DataFrame()
    else:
        stability_focus = stability[
            stability["metric"].astype(str).isin(["val_max_loss_barrier", "test_max_loss_barrier", *TARGET_COLUMNS])
        ].copy()
    report = f"""# Alignment Barrier Stability Report

Generated by `experiments/alignment_barrier_targets.py`.

## Evaluation Scale

- Previous barrier cap(s): `{previous_cap}`
- Current barrier cap(s): `{current_cap}`
- Current evaluation scale: `{cap_label}`
- Runtime: `{runtime_seconds:.2f}` seconds
- Bootstrap samples for barrier stats and claim-stability checks: `{args.bootstrap_samples}`
- Feasibility note: {feasibility}

## Acceptance Audit

- Current rows: {len(current_barriers)}
- Current skipped/failed rows: {len(skipped)}
- Previous rows available for comparison: {len(previous_barriers)}
- Methods: {", ".join(sorted(current_barriers["method"].dropna().astype(str).unique())) if not current_barriers.empty and "method" in current_barriers else ""}

## Stability Summary

{md_table(stability_focus, ["metric", "method", "n_pairs", "previous_mean", "current_mean", "mean_delta_current_minus_previous", "mean_abs_difference", "max_abs_difference", "pearson_previous_current"], 80)}

## Supported Predictor-Target Claim Stability

{md_table(claim_stability, ["dataset", "n_models", "width", "domain_shift", "target", "predictor", "previous_beta", "previous_ci_low", "previous_ci_high", "current_beta", "current_ci_low", "current_ci_high", "n_unique_seeds", "stable_positive_gate"], 80)}

## Skipped Or Failed Current Rows

{md_table(skipped, ["dataset", "architecture", "n_models", "width", "domain_shift", "matching", "seed", "method", "status", "skip_reason"], 60)}

## Claim Boundary

The paper should cite the current larger-evaluation barrier files, not the previous 2-batch debug estimates. Validation barriers remain selector/predictor diagnostics, and test barriers remain evaluation-only.
"""
    path.write_text(report, encoding="utf-8")


def md_table(df: pd.DataFrame, columns: list[str], max_rows: int = 40) -> str:
    if df.empty:
        return "_None._"
    view = df[[col for col in columns if col in df.columns]].head(max_rows).copy()
    for col in view.columns:
        if pd.api.types.is_bool_dtype(view[col]):
            view[col] = view[col].map(lambda v: "true" if bool(v) else "false")
        elif pd.api.types.is_numeric_dtype(view[col]):
            if col in {"n_rows", "n_unique_seeds", "n_models", "width", "seed", "best_model_index"}:
                view[col] = view[col].map(lambda v: "" if pd.isna(v) else str(int(round(float(v)))))
            else:
                view[col] = view[col].map(lambda v: "" if pd.isna(v) else f"{float(v):.4f}")
        else:
            view[col] = view[col].fillna("").astype(str)
    header = "| " + " | ".join(view.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(view.columns)) + " |"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in view.to_numpy()]
    suffix = f"\n\n_Showing {max_rows} of {len(df)} rows._" if len(df) > max_rows else ""
    return "\n".join([header, sep, *body]) + suffix


def write_plot(barriers: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    ok = barriers[barriers["status"].astype(str) == "ok"].copy() if not barriers.empty else pd.DataFrame()
    fig, ax = plt.subplots(figsize=(9.0, 6.2))
    if ok.empty:
        ax.text(0.5, 0.5, "No barrier rows", ha="center", va="center")
        ax.set_axis_off()
    else:
        methods = sorted(ok["method"].dropna().astype(str).unique())
        for method in methods:
            group = ok[ok["method"].astype(str) == method]
            ax.scatter(group["mean_cycle_score"], group["val_max_loss_barrier"], s=14, alpha=0.55, label=method)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_xlabel("mean cycle score")
        ax.set_ylabel("validation max loss barrier")
        ax.set_title("Alignment-conditioned barriers vs obstruction score")
        ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_report(args: argparse.Namespace, barriers: pd.DataFrame, stats: pd.DataFrame, runtime_seconds: float, path: Path) -> None:
    ok = barriers[barriers["status"].astype(str) == "ok"].copy() if not barriers.empty else pd.DataFrame()
    skipped = barriers[barriers["status"].astype(str) != "ok"].copy() if not barriers.empty else pd.DataFrame()
    target_summary = (
        ok.groupby(["dataset", "architecture", "n_models", "width", "domain_shift", "matching"], dropna=False)
        .agg(
            n_rows=("seed", "count"),
            n_unique_seeds=("seed", "nunique"),
            mean_lmc_barrier=("linear_mode_connectivity_barrier", "mean"),
            mean_c2m3_delta_vs_git=("c2m3_barrier_delta_vs_git_rebasin", "mean"),
            mean_c2m3_delta_vs_weight=("c2m3_barrier_delta_vs_weight_average", "mean"),
            mean_monomial_delta_vs_c2m3=("monomial_barrier_delta_vs_c2m3", "mean"),
        )
        .reset_index()
        if not ok.empty
        else pd.DataFrame()
    )
    report = f"""# Alignment Barrier Targets

Generated by `experiments/alignment_barrier_targets.py`.

## Exact Command

```bash
{" ".join(sys.argv)}
```

## Scope

- Inputs are the saved quality-gated fixed-setting CSVs and checkpoints.
- Validation and test barriers are computed separately. Validation barriers are selector/diagnostic targets; test barriers are evaluation-only.
- Evaluation batch cap: `{args.max_eval_batches}` (`0` means full loader).
- Runtime: `{runtime_seconds:.2f}` seconds.
- The path endpoint is the best local model in the same coordinate system as the merged target for each method.
- Midpoint loss barrier is `loss(t=0.5) - 0.5 * (loss(t=0) + loss(t=1))`.
- Max loss barrier is the maximum loss excess above the linear endpoint-loss interpolation over `t in {args.t_grid}`.
- Accuracy drop barrier is `0.5 * (acc(t=0) + acc(t=1)) - acc(t=0.5)`.
- `linear_mode_connectivity_barrier` is the validation max loss barrier for the C2M3 synchronized path.
- C2M3 barrier deltas are baseline validation max loss barrier minus C2M3 validation max loss barrier, so positive values mean C2M3 has a lower barrier.
- No performance claim is made here. Barrier prediction claims are delegated to the predictor-target diagnostics and require observed rows with `n>=20`, positive bootstrap lower bound, and secondary-setting sign consistency.

## Outputs

- `reports/csv/{BARRIER_CSV}`
- `reports/csv/{BARRIER_STATS_CSV}`
- `reports/{BARRIER_REPORT}`
- `reports/{BARRIER_STABILITY_REPORT}`
- `reports/plots/{BARRIER_PLOT}`
- `reports/plots/{BARRIER_STABILITY_PLOT}`

## Target Summary

{md_table(target_summary, ["dataset", "architecture", "n_models", "width", "domain_shift", "matching", "n_rows", "n_unique_seeds", "mean_lmc_barrier", "mean_c2m3_delta_vs_git", "mean_c2m3_delta_vs_weight", "mean_monomial_delta_vs_c2m3"], 40)}

## Method Barrier Stats

{md_table(stats, ["dataset", "architecture", "n_models", "width", "domain_shift", "matching", "method", "n_rows", "n_unique_seeds", "mean_val_midpoint_loss_barrier", "mean_val_max_loss_barrier", "mean_val_accuracy_drop_barrier_t05", "mean_test_max_loss_barrier", "pearson_cycle_vs_val_max_loss_barrier", "claim_status"], 80)}

## Skipped Or Failed Rows

{md_table(skipped, ["dataset", "architecture", "n_models", "width", "domain_shift", "matching", "seed", "method", "status", "skip_reason"], 60)}

## Claim Boundary

These targets refine the outcome side of the real-model diagnostics. They do not show a general performance win, and test-set barriers are not used for method selection. The paper should cite these current larger-evaluation barriers rather than the earlier 2-batch debug estimates.
"""
    path.write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    csv_dir = args.reports_dir / "csv"
    plot_dir = args.reports_dir / "plots"
    csv_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    previous_barriers = load_existing_csv(csv_dir / BARRIER_CSV)
    previous_predictor_stats = load_existing_csv(args.previous_barrier_predictor_stats_csv)
    start_time = time.perf_counter()
    barriers = compute_barriers(args)
    runtime_seconds = time.perf_counter() - start_time
    stats = compute_stats(barriers, args.bootstrap_samples)
    stability = compare_barrier_estimates(previous_barriers, barriers)
    claim_stability = supported_claim_stability(
        previous_predictor_stats,
        barriers,
        args.runs_csv,
        args.bootstrap_samples,
    )
    barriers.to_csv(csv_dir / BARRIER_CSV, index=False, lineterminator="\n")
    stats.to_csv(csv_dir / BARRIER_STATS_CSV, index=False, lineterminator="\n")
    write_plot(barriers, plot_dir / BARRIER_PLOT)
    write_report(args, barriers, stats, runtime_seconds, args.reports_dir / BARRIER_REPORT)
    write_stability_plot(stability, plot_dir / BARRIER_STABILITY_PLOT)
    write_stability_report(
        args,
        previous_barriers,
        barriers,
        stability,
        claim_stability,
        runtime_seconds,
        args.reports_dir / BARRIER_STABILITY_REPORT,
    )
    print(f"wrote {csv_dir / BARRIER_CSV}")
    print(f"wrote {csv_dir / BARRIER_STATS_CSV}")
    print(f"wrote {args.reports_dir / BARRIER_REPORT}")
    print(f"wrote {args.reports_dir / BARRIER_STABILITY_REPORT}")
    print(f"wrote {plot_dir / BARRIER_PLOT}")
    print(f"wrote {plot_dir / BARRIER_STABILITY_PLOT}")


if __name__ == "__main__":
    main()
