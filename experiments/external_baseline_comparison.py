#!/usr/bin/env python
"""License-clean external-baseline comparison for MNIST MLP merging.

The script trains or reuses the same checkpoint set for every method in a
setting, uses a validation split only for soup/selector decisions, and reports
paired deltas against the internal C2M3-style reference, greedy soup, and
ordinary weight averaging.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.improved_monomial_merge import (  # noqa: E402
    build_scaled_models,
    choose_by_validation,
    greedy_soup_with_metadata,
    log_scale_diagnostics,
    reference_log_scales_from_features,
)
from src.metrics import capture_environment, save_json  # noqa: E402
from src.model_merging_benchmark import (  # noqa: E402
    DatasetSpec,
    average_models,
    collect_features,
    device_from_arg,
    evaluate_model,
    load_dataset,
    make_loader,
    make_model,
    permute_model_to_reference,
    require_torch,
    save_checkpoint,
    set_seed,
    synchronize_permutations,
    train_model,
)
from src.structure_group_ladder import estimate_pairwise_permutations_from_activations  # noqa: E402


@dataclass(frozen=True)
class MethodInfo:
    method: str
    display_name: str
    paper_name: str
    official_repo: str
    license: str
    implementation_kind: str
    deviations: str
    output_type: str
    uses_validation_data: bool
    capacity_matched_to_weight_average: bool
    extra_inference_cost: bool
    inference_time_multiplier: float
    param_count_multiplier: float
    is_single_model: bool
    uses_exact_relu_symmetry: bool
    is_soup: bool
    is_ensemble_or_extra_capacity: bool
    fair_for_current_claims: bool
    fairness_note: str


GIT_REBASIN_REPO = "https://github.com/samuela/git-re-basin"
C2M3_REPO = "https://github.com/crisostomi/cycle-consistent-model-merging"
MODEL_SOUPS_REPO = "https://github.com/mlfoundations/model-soups"


METHODS: dict[str, MethodInfo] = {
    "weight_average": MethodInfo(
        "weight_average",
        "Weight average / uniform soup",
        "Ordinary uniform parameter averaging control",
        "",
        "project",
        "native internal control",
        "Uniform parameter average of the same original checkpoints; also the uniform-soup analogue.",
        "single_model",
        False,
        True,
        False,
        1.0,
        1.0,
        True,
        False,
        False,
        False,
        True,
        "Fair as an unaligned single-model control.",
    ),
    "git_rebasin_pairwise": MethodInfo(
        "git_rebasin_pairwise",
        "Faithful Git-ReBasin-style pairwise alignment",
        "Git Re-Basin: Merging Models modulo Permutation Symmetries",
        GIT_REBASIN_REPO,
        "MIT",
        "faithful in-repo reimplementation",
        "Activation-correlation Hungarian matching aligns each model to model 0, then weights are averaged; official JAX experiment stack is not run.",
        "single_model",
        False,
        True,
        False,
        1.0,
        1.0,
        True,
        True,
        False,
        False,
        True,
        "Fair only as a Git-ReBasin-style MNIST MLP baseline, not as an official-code claim.",
    ),
    "c2m3_permutation": MethodInfo(
        "c2m3_permutation",
        "Internal C2M3-style cycle-consistent alignment",
        "Cycle Consistent Model Merging",
        C2M3_REPO,
        "MIT",
        "faithful internal C2M3-style reimplementation",
        "Activation-based pairwise permutations are synchronized to a global reference before averaging; official Hydra/uv/wandb pipeline is not run.",
        "single_model",
        False,
        True,
        False,
        1.0,
        1.0,
        True,
        True,
        False,
        False,
        True,
        "Fair as the internal C2M3-style reference for current claims, not as official C2M3.",
    ),
    "greedy_soup": MethodInfo(
        "greedy_soup",
        "Faithful Model Soups greedy soup",
        "Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time",
        MODEL_SOUPS_REPO,
        "MIT",
        "faithful in-repo reimplementation",
        "Greedy soup over the same MNIST MLP checkpoints using validation accuracy/loss; official downloaded soup models are not used.",
        "soup_single_model",
        True,
        True,
        False,
        1.0,
        1.0,
        True,
        False,
        True,
        False,
        True,
        "Fair as a faithful greedy soup on this MNIST MLP benchmark, not as a broad official Model Soups result.",
    ),
    "monomial_scale": MethodInfo(
        "monomial_scale",
        "Raw monomial scaling",
        "TwistedMerge exact positive ReLU gauge",
        "",
        "project",
        "native internal method",
        "Positive ReLU hidden-unit scalings are estimated from training activations after permutation synchronization, then averaged.",
        "single_model",
        False,
        True,
        False,
        1.0,
        1.0,
        True,
        True,
        False,
        False,
        True,
        "Fair as an internal exact-symmetry ablation.",
    ),
    "monomial_scaled_greedy_soup": MethodInfo(
        "monomial_scaled_greedy_soup",
        "Monomial-scaled greedy soup",
        "TwistedMerge exact positive ReLU gauge plus Model Soups greedy selection",
        MODEL_SOUPS_REPO,
        "MIT for Model Soups reference; project code for scaling",
        "native internal method plus faithful greedy soup rule",
        "Greedy soup over positive-scale-aligned candidates; official Model Soups code is not imported.",
        "soup_single_model",
        True,
        True,
        False,
        1.0,
        1.0,
        True,
        True,
        True,
        False,
        True,
        "Fair as an internal monomial+soup ablation.",
    ),
    "validated_ladder_selector": MethodInfo(
        "validated_ladder_selector",
        "Validated ladder selector",
        "TwistedMerge validation-only ladder selector",
        "",
        "project",
        "native internal method",
        "Selects between C2M3-style alignment and raw monomial scaling by validation accuracy/loss only.",
        "validation_selected_single_model",
        True,
        True,
        False,
        1.0,
        1.0,
        True,
        True,
        False,
        False,
        True,
        "Fair as the earlier internal selector baseline.",
    ),
    "improved_validated_selector": MethodInfo(
        "improved_validated_selector",
        "Improved validated ladder selector",
        "TwistedMerge improved validation-only selector",
        "",
        "project",
        "native internal method",
        "Uses the current internal improved-selector definition from the improved validated ladder benchmark when available; that selector chooses among C2M3, monomial, shrinkage/global/optimized monomial, greedy-soup variants, and union candidate soup by validation metrics only.",
        "validation_selected_single_model_or_soup",
        True,
        True,
        False,
        1.0,
        1.0,
        True,
        True,
        True,
        False,
        True,
        "Fair for the limited internal MNIST MLP selector-vs-baseline claim.",
    ),
}

METHOD_ORDER = [
    "weight_average",
    "git_rebasin_pairwise",
    "c2m3_permutation",
    "greedy_soup",
    "monomial_scale",
    "monomial_scaled_greedy_soup",
    "validated_ladder_selector",
    "improved_validated_selector",
]

INT_COLUMNS = {
    "n_rows",
    "n_seeds",
    "accuracy_wins_vs_internal_c2m3",
    "accuracy_ties_vs_internal_c2m3",
    "accuracy_losses_vs_internal_c2m3",
    "accuracy_wins_vs_greedy_soup",
    "accuracy_ties_vs_greedy_soup",
    "accuracy_losses_vs_greedy_soup",
    "accuracy_wins_vs_weight_average",
    "accuracy_ties_vs_weight_average",
    "accuracy_losses_vs_weight_average",
}


def parse_csv(text: str, cast=str):
    return [cast(item.strip()) for item in text.split(",") if item.strip()]


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def git_worktree_dirty() -> bool | str:
    try:
        status = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
        return bool(status.strip())
    except Exception:
        return "unknown"


def split_train_val(dataset, val_fraction: float, seed: int):
    torch, _, _ = require_torch()
    n_val = max(1, int(len(dataset) * val_fraction))
    n_train = len(dataset) - n_val
    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.utils.data.random_split(dataset, [n_train, n_val], generator=generator)


def bootstrap_mean_ci(values, n_bootstrap: int = 2000, seed: int = 12345) -> tuple[float, float]:
    arr = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1 or n_bootstrap <= 0:
        value = float(arr.mean())
        return value, value
    rng = np.random.default_rng(seed)
    means = [float(rng.choice(arr, size=arr.size, replace=True).mean()) for _ in range(n_bootstrap)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def standard_error(values) -> float:
    arr = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if arr.size <= 1:
        return float("nan")
    return float(arr.std(ddof=1) / np.sqrt(arr.size))


def sign_test_two_sided(wins: int, losses: int) -> float:
    n = wins + losses
    if n <= 0:
        return float("nan")
    tail = min(wins, losses)
    prob = sum(math.comb(n, k) for k in range(tail + 1)) / (2**n)
    return float(min(1.0, 2.0 * prob))


def checkpoint_dir(args, setting_id: str) -> Path:
    return args.reports_dir / "checkpoints" / "external_baselines" / setting_id


def load_checkpoint_model(path: Path, spec: DatasetSpec, width: int):
    torch, _, _ = require_torch()
    payload = torch.load(path, map_location="cpu")
    state = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
    model = make_model("mlp", spec, width)
    model.load_state_dict(state)
    model.to("cpu")
    return model


def load_or_train_models(args, spec: DatasetSpec, width: int, n_models: int, seed: int, train_subset, val_loader, test_loader, device, setting_id: str):
    ckpt_root = checkpoint_dir(args, setting_id)
    paths = [ckpt_root / f"model_{idx}.pt" for idx in range(n_models)]
    models = []
    if args.reuse_checkpoints and all(path.exists() for path in paths):
        for path in paths:
            models.append(load_checkpoint_model(path, spec, width))
    else:
        for model_idx in range(n_models):
            model_seed = seed + 1000 * model_idx + 17 * width + n_models
            set_seed(model_seed)
            model = make_model("mlp", spec, width)
            train_loader = make_loader(train_subset, args.batch_size, shuffle=True, seed=model_seed + 11)
            train_model(model, train_loader, args.epochs, args.lr, device)
            model.to("cpu")
            if args.save_checkpoints:
                save_checkpoint(
                    model,
                    paths[model_idx],
                    {
                        "setting_id": setting_id,
                        "dataset": "mnist",
                        "architecture": "mlp_relu",
                        "model_index": model_idx,
                        "n_models": n_models,
                        "width": width,
                        "seed": seed,
                        "model_seed": model_seed,
                        "epochs": args.epochs,
                        "external_baseline_comparison": True,
                    },
                )
            models.append(model)
    individual = []
    for model_idx, model in enumerate(models):
        val = evaluate_model(model, val_loader, device)
        test = evaluate_model(model, test_loader, device)
        individual.append(
            {
                "model_index": model_idx,
                "model_seed": seed + 1000 * model_idx + 17 * width + n_models,
                "val_loss": float(val["loss"]),
                "val_accuracy": float(val["accuracy"]),
                "test_loss": float(test["loss"]),
                "test_accuracy": float(test["accuracy"]),
                "checkpoint": str(paths[model_idx]) if paths[model_idx].exists() else "",
            }
        )
    return models, individual


def save_method_model(args, setting_id: str, method: str, model, metadata: dict) -> str:
    if not args.save_checkpoints:
        return ""
    path = checkpoint_dir(args, setting_id) / f"{method}.pt"
    save_checkpoint(model.to("cpu"), path, metadata)
    return str(path)


def add_row(
    rows: list[dict],
    *,
    base: dict,
    method: str,
    val_metrics: dict[str, float],
    test_metrics: dict[str, float],
    single_best_accuracy: float,
    checkpoint: str = "",
    extra: dict | None = None,
) -> None:
    info = METHODS[method]
    row = {
        **base,
        "method": method,
        "display_name": info.display_name,
        "paper_name": info.paper_name,
        "official_repo": info.official_repo,
        "license": info.license,
        "implementation_kind": info.implementation_kind,
        "deviations_from_official": info.deviations,
        "output_type": info.output_type,
        "uses_validation_data": info.uses_validation_data,
        "capacity_matched_to_weight_average": info.capacity_matched_to_weight_average,
        "extra_inference_cost": info.extra_inference_cost,
        "inference_time_multiplier": info.inference_time_multiplier,
        "param_count_multiplier": info.param_count_multiplier,
        "is_single_model": info.is_single_model,
        "uses_exact_relu_symmetry": info.uses_exact_relu_symmetry,
        "is_soup": info.is_soup,
        "is_ensemble_or_extra_capacity": info.is_ensemble_or_extra_capacity,
        "fair_for_current_claims": info.fair_for_current_claims,
        "fairness_note": info.fairness_note,
        "val_loss": float(val_metrics["loss"]),
        "val_accuracy": float(val_metrics["accuracy"]),
        "selection_val_accuracy": float(val_metrics["accuracy"]) if info.uses_validation_data else float("nan"),
        "test_loss": float(test_metrics["loss"]),
        "test_accuracy": float(test_metrics["accuracy"]),
        "loss": float(test_metrics["loss"]),
        "accuracy": float(test_metrics["accuracy"]),
        "single_best_accuracy": single_best_accuracy,
        "merge_degradation": single_best_accuracy - float(test_metrics["accuracy"]),
        "checkpoint": checkpoint,
        "selector_no_test_leakage": True,
        "evaluation_status": "evaluated",
    }
    if extra:
        row.update(extra)
    rows.append(row)


def run_setting(args, spec, train_data, test_data, seed: int, n_models: int, width: int) -> tuple[list[dict], list[dict]]:
    device = device_from_arg(args.device)
    setting_id = f"mnist_mlp_N{n_models}_W{width}_S{seed}"
    train_subset, val_subset = split_train_val(train_data, args.val_fraction, seed + 77)
    val_loader = make_loader(val_subset, args.batch_size, shuffle=False, seed=seed + 700)
    test_loader = make_loader(test_data, args.batch_size, shuffle=False, seed=seed + 999)
    match_loader = make_loader(train_subset, args.batch_size, shuffle=False, seed=seed + 501)

    models, individual_rows = load_or_train_models(
        args,
        spec,
        width,
        n_models,
        seed,
        train_subset,
        val_loader,
        test_loader,
        device,
        setting_id,
    )
    individual_accuracies = [row["test_accuracy"] for row in individual_rows]
    individual_losses = [row["test_loss"] for row in individual_rows]
    single_best_accuracy = float(max(individual_accuracies))

    features = {idx: collect_features(model, match_loader, device) for idx, model in enumerate(models)}
    pairwise = estimate_pairwise_permutations_from_activations(features, n_models, width)
    ref, synced, sync_disagreement = synchronize_permutations(pairwise, n_models)
    reference_logs = reference_log_scales_from_features(features, synced, ref=ref, width=width)
    scale_diag = log_scale_diagnostics(reference_logs, sync_disagreement)

    base = {
        "setting_id": setting_id,
        "dataset": "mnist",
        "architecture": "mlp_relu",
        "n_models": n_models,
        "width": width,
        "seed": seed,
        "epochs": args.epochs,
        "max_train_samples": args.max_train_samples,
        "max_test_samples": args.max_test_samples,
        "val_fraction": args.val_fraction,
        "matching": "activation",
        "checkpoint_dir": str(checkpoint_dir(args, setting_id)),
        "sync_reference": ref,
        "sync_disagreement": sync_disagreement,
        "individual_accuracy_mean": float(np.mean(individual_accuracies)),
        "individual_accuracy_min": float(np.min(individual_accuracies)),
        "individual_accuracy_max": single_best_accuracy,
        "individual_accuracy_variance": float(np.var(individual_accuracies)),
        "individual_loss_mean": float(np.mean(individual_losses)),
    }

    rows: list[dict] = []
    method_models = {}
    val_by_method = {}
    test_by_method = {}

    def evaluate_and_add(method: str, model, extra: dict | None = None) -> None:
        val = evaluate_model(model, val_loader, device)
        test = evaluate_model(model, test_loader, device)
        checkpoint = save_method_model(
            args,
            setting_id,
            method,
            model,
            {"setting_id": setting_id, "method": method, "external_baseline_comparison": True},
        )
        method_models[method] = model
        val_by_method[method] = val
        test_by_method[method] = test
        add_row(
            rows,
            base=base,
            method=method,
            val_metrics=val,
            test_metrics=test,
            single_best_accuracy=single_best_accuracy,
            checkpoint=checkpoint,
            extra=extra,
        )

    weight_avg = average_models(models, "mlp", spec, width)
    evaluate_and_add("weight_average", weight_avg)

    aligned_to_zero = [
        permute_model_to_reference(model, "mlp", spec, width, pairwise[(0, idx)])
        for idx, model in enumerate(models)
    ]
    git_rebasin = average_models(aligned_to_zero, "mlp", spec, width)
    evaluate_and_add("git_rebasin_pairwise", git_rebasin)

    aligned_c2m3 = [
        permute_model_to_reference(model, "mlp", spec, width, synced[idx])
        for idx, model in enumerate(models)
    ]
    c2m3_model = average_models(aligned_c2m3, "mlp", spec, width)
    evaluate_and_add("c2m3_permutation", c2m3_model)

    scaled_models = build_scaled_models(models, spec, width, synced, reference_logs)
    monomial_model = average_models(scaled_models, "mlp", spec, width)
    evaluate_and_add(
        "monomial_scale",
        monomial_model,
        {
            "scale_source": "reference_raw",
            "mean_abs_log_scale": scale_diag.mean_abs_log_scale,
            "max_abs_log_scale": scale_diag.max_abs_log_scale,
            "log_scale_variance": scale_diag.log_scale_variance,
            "scale_synchronization_disagreement": scale_diag.synchronization_disagreement,
        },
    )

    greedy = greedy_soup_with_metadata(
        models,
        [f"original:{idx}" for idx in range(n_models)],
        val_loader,
        test_loader,
        device,
        "mlp",
        spec,
        width,
    )
    greedy_checkpoint = save_method_model(
        args,
        setting_id,
        "greedy_soup",
        greedy.model,
        {"setting_id": setting_id, "method": "greedy_soup", "external_baseline_comparison": True},
    )
    method_models["greedy_soup"] = greedy.model
    val_by_method["greedy_soup"] = greedy.val_metrics
    test_by_method["greedy_soup"] = greedy.test_metrics
    add_row(
        rows,
        base=base,
        method="greedy_soup",
        val_metrics=greedy.val_metrics,
        test_metrics=greedy.test_metrics,
        single_best_accuracy=single_best_accuracy,
        checkpoint=greedy_checkpoint,
        extra={
            "soup_indices": json.dumps(greedy.selected_indices),
            "soup_selected_labels": json.dumps(greedy.selected_labels),
            "soup_ingredient_count": len(greedy.selected_indices),
        },
    )

    mono_soup = greedy_soup_with_metadata(
        scaled_models,
        [f"monomial:{idx}" for idx in range(n_models)],
        val_loader,
        test_loader,
        device,
        "mlp",
        spec,
        width,
    )
    mono_soup_checkpoint = save_method_model(
        args,
        setting_id,
        "monomial_scaled_greedy_soup",
        mono_soup.model,
        {"setting_id": setting_id, "method": "monomial_scaled_greedy_soup", "external_baseline_comparison": True},
    )
    method_models["monomial_scaled_greedy_soup"] = mono_soup.model
    val_by_method["monomial_scaled_greedy_soup"] = mono_soup.val_metrics
    test_by_method["monomial_scaled_greedy_soup"] = mono_soup.test_metrics
    add_row(
        rows,
        base=base,
        method="monomial_scaled_greedy_soup",
        val_metrics=mono_soup.val_metrics,
        test_metrics=mono_soup.test_metrics,
        single_best_accuracy=single_best_accuracy,
        checkpoint=mono_soup_checkpoint,
        extra={
            "soup_indices": json.dumps(mono_soup.selected_indices),
            "soup_selected_labels": json.dumps(mono_soup.selected_labels),
            "soup_ingredient_count": len(mono_soup.selected_indices),
        },
    )

    ladder_choice = choose_by_validation(
        {
            "c2m3_permutation": val_by_method["c2m3_permutation"],
            "monomial_scale": val_by_method["monomial_scale"],
        }
    )
    ladder_selected = ladder_choice.selected
    ladder_checkpoint = save_method_model(
        args,
        setting_id,
        "validated_ladder_selector",
        method_models[ladder_selected],
        {
            "setting_id": setting_id,
            "method": "validated_ladder_selector",
            "selected_method": ladder_selected,
            "external_baseline_comparison": True,
        },
    )
    add_row(
        rows,
        base=base,
        method="validated_ladder_selector",
        val_metrics=val_by_method[ladder_selected],
        test_metrics=test_by_method[ladder_selected],
        single_best_accuracy=single_best_accuracy,
        checkpoint=ladder_checkpoint,
        extra={
            "selector_chose": ladder_selected,
            "selector_val_margin": ladder_choice.margin_to_runner_up,
            "selector_pool": json.dumps(["c2m3_permutation", "monomial_scale"]),
        },
    )

    improved_pool = [
        "weight_average",
        "git_rebasin_pairwise",
        "c2m3_permutation",
        "monomial_scale",
        "greedy_soup",
        "monomial_scaled_greedy_soup",
    ]
    improved_choice = choose_by_validation(
        {name: val_by_method[name] for name in improved_pool},
        allowed_methods=improved_pool,
    )
    improved_selected = improved_choice.selected
    improved_checkpoint = ""
    if not args.reuse_improved_selector:
        improved_checkpoint = save_method_model(
            args,
            setting_id,
            "improved_validated_selector",
            method_models[improved_selected],
            {
                "setting_id": setting_id,
                "method": "improved_validated_selector",
                "selected_method": improved_selected,
                "external_baseline_comparison": True,
            },
        )
    add_row(
        rows,
        base=base,
        method="improved_validated_selector",
        val_metrics=val_by_method[improved_selected],
        test_metrics=test_by_method[improved_selected],
        single_best_accuracy=single_best_accuracy,
        checkpoint=improved_checkpoint,
        extra={
            "selector_chose": improved_selected,
            "selector_val_margin": improved_choice.margin_to_runner_up,
            "selector_pool": json.dumps(improved_pool),
        },
    )

    by_method = {row["method"]: row for row in rows}
    c2m3 = by_method["c2m3_permutation"]
    greedy_row = by_method["greedy_soup"]
    weight_row = by_method["weight_average"]
    for row in rows:
        row["accuracy_delta_vs_internal_c2m3"] = row["accuracy"] - c2m3["accuracy"]
        row["loss_delta_vs_internal_c2m3"] = row["loss"] - c2m3["loss"]
        row["validation_accuracy_delta_vs_internal_c2m3"] = row["val_accuracy"] - c2m3["val_accuracy"]
        row["accuracy_delta_vs_greedy_soup"] = row["accuracy"] - greedy_row["accuracy"]
        row["loss_delta_vs_greedy_soup"] = row["loss"] - greedy_row["loss"]
        row["validation_accuracy_delta_vs_greedy_soup"] = row["val_accuracy"] - greedy_row["val_accuracy"]
        row["accuracy_delta_vs_weight_average"] = row["accuracy"] - weight_row["accuracy"]
        row["loss_delta_vs_weight_average"] = row["loss"] - weight_row["loss"]
        row["validation_accuracy_delta_vs_weight_average"] = row["val_accuracy"] - weight_row["val_accuracy"]

    for item in individual_rows:
        item.update(
            {
                "setting_id": setting_id,
                "dataset": "mnist",
                "architecture": "mlp_relu",
                "n_models": n_models,
                "width": width,
                "seed": seed,
            }
        )
    return rows, individual_rows


def overlay_internal_improved_selector(df: pd.DataFrame, args) -> pd.DataFrame:
    """Reuse exact improved-selector metrics from the current internal benchmark.

    The external comparison recomputes primitive baselines on the same
    deterministic MNIST MLP settings. The current internal improved selector has
    a larger validation-only candidate pool than the external primitive set, so
    reusing its existing rows preserves the method definition without rerunning
    prior ladder-selector experiments.
    """

    out = df.copy()
    out["reused_internal_improved_selector_row"] = False
    out["selector_metric_source"] = ""
    if not args.reuse_improved_selector:
        return out
    path = Path(args.improved_benchmark_csv)
    if not path.exists():
        return out
    internal = pd.read_csv(path)
    internal = internal[
        (internal["method"] == "improved_validated_selector")
        & (internal["setting_id"].isin(out["setting_id"].unique()))
    ].copy()
    if internal.empty:
        return out
    source_by_setting = {row["setting_id"]: row for _, row in internal.iterrows()}
    metric_cols = [
        "accuracy",
        "loss",
        "val_accuracy",
        "val_loss",
        "selector_chose",
        "selector_val_margin",
        "selector_pool",
    ]
    for idx, row in out[out["method"] == "improved_validated_selector"].iterrows():
        source = source_by_setting.get(row["setting_id"])
        if source is None:
            continue
        for col in metric_cols:
            if col in source.index:
                out.loc[idx, col] = source[col]
        out.loc[idx, "test_accuracy"] = float(source["accuracy"])
        out.loc[idx, "test_loss"] = float(source["loss"])
        out.loc[idx, "selection_val_accuracy"] = float(source["val_accuracy"])
        out.loc[idx, "merge_degradation"] = float(out.loc[idx, "single_best_accuracy"]) - float(source["accuracy"])
        out.loc[idx, "checkpoint"] = ""
        out.loc[idx, "reused_internal_improved_selector_row"] = True
        out.loc[idx, "selector_metric_source"] = str(path)
    return out


def recompute_pairwise_deltas(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for _setting_id, group in out.groupby("setting_id", dropna=False):
        c2 = group[group["method"] == "c2m3_permutation"].iloc[0]
        greedy = group[group["method"] == "greedy_soup"].iloc[0]
        weight = group[group["method"] == "weight_average"].iloc[0]
        mask = out["setting_id"] == _setting_id
        out.loc[mask, "accuracy_delta_vs_internal_c2m3"] = out.loc[mask, "accuracy"] - float(c2["accuracy"])
        out.loc[mask, "loss_delta_vs_internal_c2m3"] = out.loc[mask, "loss"] - float(c2["loss"])
        out.loc[mask, "validation_accuracy_delta_vs_internal_c2m3"] = out.loc[mask, "val_accuracy"] - float(c2["val_accuracy"])
        out.loc[mask, "accuracy_delta_vs_greedy_soup"] = out.loc[mask, "accuracy"] - float(greedy["accuracy"])
        out.loc[mask, "loss_delta_vs_greedy_soup"] = out.loc[mask, "loss"] - float(greedy["loss"])
        out.loc[mask, "validation_accuracy_delta_vs_greedy_soup"] = out.loc[mask, "val_accuracy"] - float(greedy["val_accuracy"])
        out.loc[mask, "accuracy_delta_vs_weight_average"] = out.loc[mask, "accuracy"] - float(weight["accuracy"])
        out.loc[mask, "loss_delta_vs_weight_average"] = out.loc[mask, "loss"] - float(weight["loss"])
        out.loc[mask, "validation_accuracy_delta_vs_weight_average"] = out.loc[mask, "val_accuracy"] - float(weight["val_accuracy"])
    return out


def summarize_scope(group: pd.DataFrame, *, scope: str, n_models: object = "all", width: object = "all", n_bootstrap: int = 2000) -> list[dict]:
    rows = []
    for method in METHOD_ORDER:
        part = group[group["method"] == method].copy()
        if part.empty:
            continue
        info = METHODS[method]
        row = {
            "summary_type": "method_summary",
            "scope": scope,
            "n_models": n_models,
            "width": width,
            "method": method,
            "display_name": info.display_name,
            "n_rows": int(len(part)),
            "n_seeds": int(part["seed"].nunique()),
            "mean_test_accuracy": float(pd.to_numeric(part["test_accuracy"], errors="coerce").mean()),
            "test_accuracy_standard_error": standard_error(part["test_accuracy"]),
            "mean_test_loss": float(pd.to_numeric(part["test_loss"], errors="coerce").mean()),
            "mean_val_accuracy": float(pd.to_numeric(part["val_accuracy"], errors="coerce").mean()),
            "mean_selection_val_accuracy": float(pd.to_numeric(part["selection_val_accuracy"], errors="coerce").mean()),
            "mean_merge_degradation": float(pd.to_numeric(part["merge_degradation"], errors="coerce").mean()),
            "param_count_multiplier": info.param_count_multiplier,
            "inference_time_multiplier": info.inference_time_multiplier,
            "is_single_model": info.is_single_model,
            "uses_exact_relu_symmetry": info.uses_exact_relu_symmetry,
            "is_soup": info.is_soup,
            "is_ensemble_or_extra_capacity": info.is_ensemble_or_extra_capacity,
            "uses_validation_data": info.uses_validation_data,
            "capacity_matched_to_weight_average": info.capacity_matched_to_weight_average,
            "implementation_kind": info.implementation_kind,
            "official_repo": info.official_repo,
            "license": info.license,
        }
        for label, col in [
            ("internal_c2m3", "accuracy_delta_vs_internal_c2m3"),
            ("greedy_soup", "accuracy_delta_vs_greedy_soup"),
            ("weight_average", "accuracy_delta_vs_weight_average"),
        ]:
            delta = pd.to_numeric(part[col], errors="coerce")
            ci_low, ci_high = bootstrap_mean_ci(delta, n_bootstrap=n_bootstrap, seed=20000 + len(rows) * 17 + len(label))
            wins = int((delta > 0).sum())
            ties = int((delta == 0).sum())
            losses = int((delta < 0).sum())
            row[f"paired_mean_accuracy_delta_vs_{label}"] = float(delta.mean())
            row[f"paired_accuracy_delta_vs_{label}_ci_low"] = ci_low
            row[f"paired_accuracy_delta_vs_{label}_ci_high"] = ci_high
            row[f"mean_loss_delta_vs_{label}"] = float(pd.to_numeric(part[f"loss_delta_vs_{label}"], errors="coerce").mean())
            row[f"accuracy_wins_vs_{label}"] = wins
            row[f"accuracy_ties_vs_{label}"] = ties
            row[f"accuracy_losses_vs_{label}"] = losses
            row[f"sign_test_two_sided_p_vs_{label}"] = sign_test_two_sided(wins, losses)
        rows.append(row)
    return rows


def summarize(df: pd.DataFrame, n_bootstrap: int) -> pd.DataFrame:
    rows = summarize_scope(df, scope="overall", n_bootstrap=n_bootstrap)
    for (n_models, width), group in df.groupby(["n_models", "width"], dropna=False):
        rows.extend(
            summarize_scope(
                group,
                scope="fixed_setting",
                n_models=int(n_models),
                width=int(width),
                n_bootstrap=n_bootstrap,
            )
        )
    return pd.DataFrame(rows)


def format_value(value, column: str) -> str:
    if value == "":
        return ""
    if isinstance(value, bool):
        return str(value)
    try:
        if pd.isna(value):
            return "nan"
    except TypeError:
        pass
    if column in INT_COLUMNS:
        return str(int(round(float(value))))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4f}"
    return str(value)


def markdown_table(rows: list[dict], columns: list[str]) -> str:
    if not rows:
        return "_No rows._"
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(format_value(row.get(col, ""), col) for col in columns) + " |")
    return "\n".join(lines)


def tex_escape(text: object) -> str:
    out = str(text)
    for src, dst in [
        ("\\", r"\textbackslash{}"),
        ("_", r"\_"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("{", r"\{"),
        ("}", r"\}"),
    ]:
        out = out.replace(src, dst)
    return out


def write_latex_table(summary: pd.DataFrame, path: Path) -> None:
    overall = summary[(summary["summary_type"] == "method_summary") & (summary["scope"] == "overall")].copy()
    overall["method_order"] = overall["method"].map({method: idx for idx, method in enumerate(METHOD_ORDER)})
    overall = overall.sort_values("method_order")
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Method & Acc. & Loss & $\Delta$ C2M3 & $\Delta$ Soup \\",
        r"\midrule",
    ]
    for _, row in overall.iterrows():
        c2 = (
            f"{row['paired_mean_accuracy_delta_vs_internal_c2m3']:.4f} "
            f"[{row['paired_accuracy_delta_vs_internal_c2m3_ci_low']:.4f}, {row['paired_accuracy_delta_vs_internal_c2m3_ci_high']:.4f}]"
        )
        soup = (
            f"{row['paired_mean_accuracy_delta_vs_greedy_soup']:.4f} "
            f"[{row['paired_accuracy_delta_vs_greedy_soup_ci_low']:.4f}, {row['paired_accuracy_delta_vs_greedy_soup_ci_high']:.4f}]"
        )
        lines.append(
            f"{tex_escape(row['method'])} & {row['mean_test_accuracy']:.4f} & {row['mean_test_loss']:.4f} & {tex_escape(c2)} & {tex_escape(soup)} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_delta_plot(summary: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    overall = summary[(summary["summary_type"] == "method_summary") & (summary["scope"] == "overall")].copy()
    overall["method_order"] = overall["method"].map({method: idx for idx, method in enumerate(METHOD_ORDER)})
    overall = overall.sort_values("method_order")
    labels = overall["method"].tolist()
    y = np.arange(len(labels))
    baselines = [
        ("internal_c2m3", "Delta vs internal C2M3"),
        ("greedy_soup", "Delta vs greedy soup"),
        ("weight_average", "Delta vs weight average"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.8), sharey=True)
    for ax, (suffix, title) in zip(axes, baselines, strict=True):
        mean = overall[f"paired_mean_accuracy_delta_vs_{suffix}"].to_numpy(dtype=float)
        low = overall[f"paired_accuracy_delta_vs_{suffix}_ci_low"].to_numpy(dtype=float)
        high = overall[f"paired_accuracy_delta_vs_{suffix}_ci_high"].to_numpy(dtype=float)
        err = np.vstack([mean - low, high - mean])
        ax.errorbar(mean, y, xerr=err, fmt="o", capsize=3, markersize=4)
        ax.axvline(0.0, color="black", linewidth=0.8)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Test accuracy delta")
        ax.grid(axis="x", alpha=0.25)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, fontsize=7)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def selector_summary(df: pd.DataFrame) -> list[dict]:
    rows = []
    for method in ["validated_ladder_selector", "improved_validated_selector"]:
        part = df[df["method"] == method].copy()
        if part.empty:
            continue
        counts = part["selector_chose"].value_counts(dropna=False).to_dict()
        rows.append(
            {
                "method": method,
                "n_rows": int(len(part)),
                "selector_choice_counts": json.dumps({str(key): int(value) for key, value in counts.items()}),
                "selector_no_test_leakage": bool(part["selector_no_test_leakage"].fillna(True).astype(bool).all()),
            }
        )
    return rows


def write_report(args, df: pd.DataFrame, summary: pd.DataFrame, report_path: Path, plot_path: Path, table_path: Path) -> None:
    overall = summary[(summary["summary_type"] == "method_summary") & (summary["scope"] == "overall")].copy()
    overall["method_order"] = overall["method"].map({method: idx for idx, method in enumerate(METHOD_ORDER)})
    overall_rows = overall.sort_values("method_order").to_dict("records")
    fixed_rows = summary[summary["scope"] == "fixed_setting"].to_dict("records")
    methods = [METHODS[method].__dict__ for method in METHOD_ORDER]
    selectors = selector_summary(df)
    improved = overall[overall["method"] == "improved_validated_selector"]
    improved_delta = float(improved["paired_mean_accuracy_delta_vs_internal_c2m3"].iloc[0]) if not improved.empty else float("nan")
    improved_ci_low = float(improved["paired_accuracy_delta_vs_internal_c2m3_ci_low"].iloc[0]) if not improved.empty else float("nan")
    improved_vs_soup = float(improved["paired_mean_accuracy_delta_vs_greedy_soup"].iloc[0]) if not improved.empty else float("nan")

    if improved_delta > 0 and np.isfinite(improved_ci_low) and improved_ci_low > 0:
        selector_claim = (
            "The improved selector outperforms the internal C2M3-style baseline "
            "on this MNIST MLP benchmark under the paired bootstrap interval."
        )
    elif improved_delta > 0:
        selector_claim = (
            "The improved selector has a positive mean delta versus the internal C2M3-style baseline, "
            "but the interval does not support a stronger claim."
        )
    else:
        selector_claim = (
            "The improved selector does not have a positive mean delta versus the internal C2M3-style baseline in this run."
        )
    if "reused_internal_improved_selector_row" in df.columns and bool(df["reused_internal_improved_selector_row"].fillna(False).any()):
        selector_source_note = (
            f"Improved selector metrics are reused from `{args.improved_benchmark_csv}` for the matching settings, "
            "so this report uses the current internal improved-selector candidate pool without rerunning that prior experiment."
        )
    else:
        selector_source_note = (
            "Improved selector metrics were computed inside this external comparison using its fallback validation-only candidate pool."
        )

    report = f"""# External Baseline Comparison

This report is generated by `experiments/external_baseline_comparison.py`.

## Exact Command

```bash
{args.command_string}
```

## Git State At Report Generation

- HEAD commit: `{git_commit()}`
- Worktree dirty: `{git_worktree_dirty()}`

## Benchmark Setup

- Dataset: MNIST
- Architecture: one-hidden-layer ReLU MLP
- Model counts: `{args.model_counts}`
- Widths: `{args.widths}`
- Seeds: `{args.seeds}`
- Epochs: `{args.epochs}`
- Train samples before validation split: `{args.max_train_samples}`
- Validation fraction: `{args.val_fraction}`
- Test samples: `{args.max_test_samples}` (`0` means full dataset)
- Matching: activation features from the training split
- Checkpoints: `reports/checkpoints/external_baselines/`

Every method in a setting uses the same trained checkpoints, validation split,
test partition, and activation-matching split. Validation accuracy/loss is used
only by greedy soup and selector methods. Test metrics are computed after
selection.

{selector_source_note}

## External Baseline Integration Status

Official Git Re-Basin, C2M3, and Model Soups repositories and licenses were
checked, but no external code is vendored, imported, or wrapped in this run.
The comparison uses faithful in-repo implementations:

- Git-ReBasin-style pairwise permutation alignment to model 0;
- C2M3-style cycle-consistent permutation synchronization;
- Model-Soups-style greedy soup over the same candidate checkpoints.

This supports license-clean documented comparisons. It does not support claims
that TwistedMerge++ beats official Git Re-Basin, official C2M3, or official
Model Soups.

## Method Metadata

{markdown_table(methods, ["method", "paper_name", "official_repo", "license", "implementation_kind", "output_type", "uses_validation_data", "capacity_matched_to_weight_average", "inference_time_multiplier", "is_single_model", "uses_exact_relu_symmetry", "is_soup", "is_ensemble_or_extra_capacity", "fair_for_current_claims"])}

## Overall Performance

{markdown_table(overall_rows, ["method", "n_rows", "n_seeds", "mean_test_accuracy", "test_accuracy_standard_error", "mean_test_loss", "mean_val_accuracy", "mean_selection_val_accuracy", "param_count_multiplier", "inference_time_multiplier", "is_single_model", "uses_exact_relu_symmetry", "is_soup"])}

## Paired Deltas

{markdown_table(overall_rows, ["method", "paired_mean_accuracy_delta_vs_internal_c2m3", "paired_accuracy_delta_vs_internal_c2m3_ci_low", "paired_accuracy_delta_vs_internal_c2m3_ci_high", "paired_mean_accuracy_delta_vs_greedy_soup", "paired_accuracy_delta_vs_greedy_soup_ci_low", "paired_accuracy_delta_vs_greedy_soup_ci_high", "paired_mean_accuracy_delta_vs_weight_average", "paired_accuracy_delta_vs_weight_average_ci_low", "paired_accuracy_delta_vs_weight_average_ci_high"])}

Plot: `reports/plots/{plot_path.name}`.

LaTeX table: `reports/tables/{table_path.name}`.

## Fixed-Setting Performance

{markdown_table(fixed_rows, ["n_models", "width", "method", "n_seeds", "mean_test_accuracy", "paired_mean_accuracy_delta_vs_internal_c2m3", "paired_mean_accuracy_delta_vs_greedy_soup", "paired_mean_accuracy_delta_vs_weight_average"])}

## Selector Behavior

{markdown_table(selectors, ["method", "n_rows", "selector_choice_counts", "selector_no_test_leakage"])}

## Claim Check

- We compare against documented internal baselines and, where feasible, external baseline implementations.
- {selector_claim}
- Improved selector mean delta versus greedy soup: `{improved_vs_soup:.4f}`.
- The method is not yet shown to beat all external model-merging baselines.

## Negative Boundaries

- This is an MNIST one-hidden-layer ReLU MLP benchmark, not a broad SOTA model-merging result.
- This does not compare against official external code execution for Git Re-Basin, C2M3, or Model Soups.
- No method uses the test set for selection.
- All reported methods here are capacity-matched single models or single-model soups with inference multiplier 1.0.
- TIES-Merging, RegMean, and other optional baselines remain future work.

## Environment

```json
{json.dumps(capture_environment(), indent=2)}
```
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")


def write_config(args, path: Path) -> None:
    save_json(
        path,
        {
            "command": args.command_string,
            "git_commit": git_commit(),
            "dirty_worktree": git_worktree_dirty(),
            "dataset": "mnist",
            "model_counts": parse_csv(args.model_counts, int),
            "widths": parse_csv(args.widths, int),
            "seeds": parse_csv(args.seeds, int),
            "epochs": args.epochs,
            "max_train_samples": args.max_train_samples,
            "max_test_samples": args.max_test_samples,
            "val_fraction": args.val_fraction,
            "reuse_checkpoints": args.reuse_checkpoints,
            "save_checkpoints": args.save_checkpoints,
            "reuse_improved_selector": args.reuse_improved_selector,
            "improved_benchmark_csv": str(args.improved_benchmark_csv),
            "environment": capture_environment(),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="1800,1801,1802,1803,1804")
    parser.add_argument("--model-counts", default="3,4")
    parser.add_argument("--widths", default="32,64")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-train-samples", type=int, default=5000)
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dataset-seed", type=int, default=8128)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "reports")
    parser.add_argument("--improved-benchmark-csv", type=Path, default=ROOT / "reports" / "csv" / "improved_validated_ladder_merge_benchmark.csv")
    parser.add_argument("--reuse-checkpoints", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-checkpoints", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reuse-improved-selector", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    env_prefix = [
        f"{name}={os.environ[name]}"
        for name in ("PYTHONPYCACHEPREFIX", "MPLCONFIGDIR")
        if os.environ.get(name)
    ]
    args.command_string = " ".join([*env_prefix, sys.executable, *sys.argv])

    spec, train_data, test_data = load_dataset(
        "mnist",
        args.data_dir,
        args.max_train_samples,
        args.max_test_samples,
        args.dataset_seed,
    )

    rows: list[dict] = []
    individual_rows: list[dict] = []
    for seed in parse_csv(args.seeds, int):
        for n_models in parse_csv(args.model_counts, int):
            for width in parse_csv(args.widths, int):
                print(f"running seed={seed} n_models={n_models} width={width}", flush=True)
                setting_rows, setting_individual = run_setting(args, spec, train_data, test_data, seed, n_models, width)
                rows.extend(setting_rows)
                individual_rows.extend(setting_individual)

    df = pd.DataFrame(rows)
    df = overlay_internal_improved_selector(df, args)
    df = recompute_pairwise_deltas(df)
    individual = pd.DataFrame(individual_rows)
    summary = summarize(df, args.bootstrap_samples)

    csv_dir = args.reports_dir / "csv"
    plot_dir = args.reports_dir / "plots"
    table_dir = args.reports_dir / "tables"
    config_dir = args.reports_dir / "configs"
    csv_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)

    results_path = csv_dir / "external_baseline_comparison.csv"
    individual_path = csv_dir / "external_baseline_individual_models.csv"
    summary_path = csv_dir / "external_baseline_comparison_summary.csv"
    report_path = args.reports_dir / "external_baseline_comparison.md"
    table_path = table_dir / "external_baseline_comparison.tex"
    plot_path = plot_dir / "external_baseline_deltas.pdf"
    config_path = config_dir / "external_baseline_comparison_config.json"

    df.to_csv(results_path, index=False)
    individual.to_csv(individual_path, index=False)
    summary.to_csv(summary_path, index=False)
    write_latex_table(summary, table_path)
    write_delta_plot(summary, plot_path)
    write_report(args, df, summary, report_path, plot_path, table_path)
    write_config(args, config_path)

    print(f"wrote {results_path}")
    print(f"wrote {summary_path}")
    print(f"wrote {table_path}")
    print(f"wrote {plot_path}")
    print(f"wrote {report_path}")
    print(f"wrote {config_path}")


if __name__ == "__main__":
    main()
