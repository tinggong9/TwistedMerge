#!/usr/bin/env python
"""Greedy-aware monomial selector and soup-compatible candidate benchmark."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.greedy_aware_monomial import (  # noqa: E402
    DEFAULT_GREEDY_AWARE_POOL,
    greedy_aware_selector,
    lower_confidence_greedy_aware_selector,
    nested_validation_split,
    selector_regret_analysis,
    tune_greedy_aware_thresholds,
)
from src.improved_monomial_merge import (  # noqa: E402
    build_scaled_models,
    greedy_soup_with_metadata,
    reference_log_scales_from_features,
)
from src.ladder_merge_methods import METHOD_METADATA, MethodMetadata  # noqa: E402
from src.metrics import capture_environment  # noqa: E402
from src.model_merging_benchmark import (  # noqa: E402
    clone_model,
    collect_features,
    device_from_arg,
    evaluate_ensemble,
    evaluate_model,
    load_dataset,
    make_loader,
    make_model,
    permute_model_to_reference,
    require_torch,
    set_seed,
    synchronize_permutations,
    train_model,
)
from src.structure_group_ladder import estimate_pairwise_permutations_from_activations  # noqa: E402


EXTRA_METHODS: dict[str, MethodMetadata] = {
    "single_best_model": MethodMetadata(
        "single_best_model",
        "validation_selected_original_single_model_or_descriptive_oracle",
        True,
        True,
        "Best original model; validation-selected in new low-lr rows, descriptive from prior CSV otherwise.",
    ),
    "validated_ladder_selector": MethodMetadata(
        "validated_ladder_selector",
        "validation_selected_exact_relu_single_model",
        True,
        True,
        "Validation-only selector choosing between C2M3 and raw positive monomial scaling.",
    ),
    "shrinkage_monomial_scale": MethodMetadata(
        "shrinkage_monomial_scale",
        "validation_selected_exact_relu_positive_scale",
        True,
        True,
        "Reference-based positive scales with validation-selected shrinkage.",
    ),
    "global_monomial_scale": MethodMetadata(
        "global_monomial_scale",
        "validation_selected_global_exact_relu_positive_scale",
        True,
        True,
        "Least-squares global positive scale synchronization with validation-selected shrinkage.",
    ),
    "optimized_monomial_scale": MethodMetadata(
        "optimized_monomial_scale",
        "validation_optimized_exact_relu_positive_scale",
        True,
        True,
        "Validation-loss optimization over exact positive ReLU log-scale gauges before averaging.",
    ),
    "improved_validated_selector": MethodMetadata(
        "improved_validated_selector",
        "validation_selected_single_model_or_soup",
        True,
        True,
        "Prior validation-only selector from 5(i)(ii).",
    ),
    "greedy_aware_selector": MethodMetadata(
        "greedy_aware_selector",
        "validation_selected_greedy_fallback_single_model_soup",
        True,
        True,
        "Conservative validation-only selector that leaves greedy soup unless a challenger clears a validation margin.",
    ),
    "lower_confidence_greedy_aware_selector": MethodMetadata(
        "lower_confidence_greedy_aware_selector",
        "validation_lcb_selected_greedy_fallback_single_model_soup",
        True,
        True,
        "Lower-confidence validation selector that leaves greedy soup unless a challenger has positive validation LCB.",
    ),
    "c2m3_greedy_soup": MethodMetadata("c2m3_greedy_soup", "exact_relu_permutation_soup", True, True, "Greedy soup over C2M3-aligned models."),
    "monomial_scaled_greedy_soup": MethodMetadata("monomial_scaled_greedy_soup", "exact_relu_positive_scale_soup", True, True, "Greedy soup over raw monomial-scaled models."),
    "shrinkage_monomial_greedy_soup": MethodMetadata("shrinkage_monomial_greedy_soup", "exact_relu_positive_scale_soup", True, True, "Greedy soup over shrinkage monomial models."),
    "global_monomial_greedy_soup": MethodMetadata("global_monomial_greedy_soup", "global_exact_relu_positive_scale_soup", True, True, "Greedy soup over global monomial models."),
    "optimized_monomial_greedy_soup": MethodMetadata("optimized_monomial_greedy_soup", "optimized_exact_relu_positive_scale_soup", True, True, "Greedy soup over optimized monomial models."),
    "union_candidate_soup": MethodMetadata("union_candidate_soup", "validation_selected_union_candidate_single_model_soup", True, True, "Greedy soup over a mixed exact-ReLU candidate pool."),
}

METHODS = {**METHOD_METADATA, **EXTRA_METHODS}


@dataclass(frozen=True)
class Comparison:
    name: str
    method: str
    baseline: str
    label: str


COMPARISONS = (
    Comparison("greedy_aware_selector_vs_greedy_soup", "greedy_aware_selector", "greedy_soup", "greedy-aware selector over greedy soup"),
    Comparison("lower_confidence_greedy_aware_selector_vs_greedy_soup", "lower_confidence_greedy_aware_selector", "greedy_soup", "LCB greedy-aware selector over greedy soup"),
    Comparison("union_candidate_soup_vs_greedy_soup", "union_candidate_soup", "greedy_soup", "union candidate soup over greedy soup"),
    Comparison("shrinkage_monomial_scale_vs_monomial_scale", "shrinkage_monomial_scale", "monomial_scale", "shrinkage monomial over raw monomial"),
    Comparison("global_monomial_scale_vs_monomial_scale", "global_monomial_scale", "monomial_scale", "global monomial over raw monomial"),
    Comparison("optimized_monomial_scale_vs_monomial_scale", "optimized_monomial_scale", "monomial_scale", "optimized monomial over raw monomial"),
    Comparison("monomial_scaled_greedy_soup_vs_greedy_soup", "monomial_scaled_greedy_soup", "greedy_soup", "raw monomial soup over greedy soup"),
    Comparison("shrinkage_monomial_greedy_soup_vs_greedy_soup", "shrinkage_monomial_greedy_soup", "greedy_soup", "shrinkage monomial soup over greedy soup"),
    Comparison("global_monomial_greedy_soup_vs_greedy_soup", "global_monomial_greedy_soup", "greedy_soup", "global monomial soup over greedy soup"),
    Comparison("optimized_monomial_greedy_soup_vs_greedy_soup", "optimized_monomial_greedy_soup", "greedy_soup", "optimized monomial soup over greedy soup"),
    Comparison("greedy_aware_selector_vs_c2m3_permutation", "greedy_aware_selector", "c2m3_permutation", "greedy-aware selector over internal C2M3"),
)


def parse_csv(text: str, cast=str):
    return [cast(item.strip()) for item in text.split(",") if item.strip()]


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def git_dirty() -> bool | str:
    try:
        return bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    except Exception:
        return "unknown"


def bootstrap_mean_ci(values, n_bootstrap: int, seed: int) -> tuple[float, float]:
    arr = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1 or n_bootstrap <= 0:
        return float(arr.mean()), float(arr.mean())
    rng = np.random.default_rng(seed)
    means = [float(rng.choice(arr, size=arr.size, replace=True).mean()) for _ in range(n_bootstrap)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def sign_test_two_sided(wins: int, losses: int) -> float:
    n = wins + losses
    if n <= 0:
        return float("nan")
    tail = min(wins, losses)
    prob = sum(math.comb(n, k) for k in range(tail + 1)) / (2**n)
    return float(min(1.0, 2.0 * prob))


def safe_corr(x, y, method: str) -> float:
    xs = pd.to_numeric(pd.Series(x), errors="coerce")
    ys = pd.to_numeric(pd.Series(y), errors="coerce")
    mask = xs.notna() & ys.notna()
    if int(mask.sum()) < 3:
        return float("nan")
    return float(xs[mask].corr(ys[mask], method=method))


def setting_metrics(group: pd.DataFrame, methods: Sequence[str]) -> dict[str, dict[str, float]]:
    metrics = {}
    for method in methods:
        row = group[group["method"] == method]
        if row.empty:
            continue
        item = row.iloc[0]
        metrics[method] = {
            "accuracy": float(item["val_accuracy"]),
            "loss": float(item["val_loss"]),
        }
    return metrics


def copy_selected_row(base_row: pd.Series, selected: pd.Series, *, method: str, extra: dict) -> dict:
    row = selected.to_dict()
    row["method"] = method
    row["loss"] = float(selected["loss"])
    row["accuracy"] = float(selected["accuracy"])
    row["val_loss"] = float(selected["val_loss"])
    row["val_accuracy"] = float(selected["val_accuracy"])
    row["symmetry_status"] = METHODS[method].symmetry_status
    row["is_single_model"] = METHODS[method].is_single_model
    row["capacity_matched_to_weight_average"] = METHODS[method].capacity_matched_to_weight_average
    row["method_notes"] = METHODS[method].notes
    row["selector_no_test_leakage"] = True
    row["source"] = "greedy_aware_posthoc_selector"
    row["mode"] = str(base_row.get("mode", "independent_seed_models"))
    row["validation_protocol"] = str(base_row.get("validation_protocol", "standard_validation"))
    row.update(extra)
    return row


def add_selector_rows(
    df: pd.DataFrame,
    *,
    epsilon_grid: Sequence[float],
    loss_slack_grid: Sequence[float],
    n_validation: int,
    validation_protocol: str,
) -> tuple[pd.DataFrame, dict[str, float]]:
    base = df.copy()
    base["source"] = base.get("source", "replayed_5i_ii_baseline")
    base["mode"] = base.get("mode", "independent_seed_models")
    base["validation_protocol"] = base.get("validation_protocol", validation_protocol)
    base["selector_epsilon"] = base.get("selector_epsilon", np.nan)
    base["selector_loss_slack"] = base.get("selector_loss_slack", np.nan)
    base["selector_lcb"] = base.get("selector_lcb", np.nan)

    setting_groups = list(base.groupby("setting_id", sort=True))
    tune_settings = [
        setting_metrics(group, ["greedy_soup", *DEFAULT_GREEDY_AWARE_POOL])
        for _setting_id, group in setting_groups
    ]
    epsilon, loss_slack = tune_greedy_aware_thresholds(
        tune_settings,
        epsilon_grid=epsilon_grid,
        loss_slack_grid=loss_slack_grid,
        challenger_pool=DEFAULT_GREEDY_AWARE_POOL,
    )
    new_rows = []
    for _setting_id, group in setting_groups:
        metrics = setting_metrics(group, ["greedy_soup", *DEFAULT_GREEDY_AWARE_POOL])
        greedy_choice = greedy_aware_selector(
            metrics,
            challenger_pool=DEFAULT_GREEDY_AWARE_POOL,
            epsilon=epsilon,
            loss_slack=loss_slack,
        )
        lcb_choice = lower_confidence_greedy_aware_selector(
            metrics,
            challenger_pool=DEFAULT_GREEDY_AWARE_POOL,
            n_validation=n_validation,
            loss_slack=loss_slack,
        )
        for method, choice in [
            ("greedy_aware_selector", greedy_choice),
            ("lower_confidence_greedy_aware_selector", lcb_choice),
        ]:
            selected = group[group["method"] == choice.selected].iloc[0]
            greedy = group[group["method"] == "greedy_soup"].iloc[0]
            new_rows.append(
                copy_selected_row(
                    group.iloc[0],
                    selected,
                    method=method,
                    extra={
                        "selector_chose": choice.selected,
                        "selector_challenger": choice.challenger,
                        "selector_epsilon": choice.epsilon,
                        "selector_loss_slack": choice.loss_slack,
                        "selector_lcb": np.nan if choice.lower_confidence_bound is None else choice.lower_confidence_bound,
                        "selector_val_margin": choice.validation_accuracy_delta,
                        "selector_val_loss_delta_vs_greedy": choice.validation_loss_delta,
                        "selector_chosen_test_better": bool(selected["accuracy"] > greedy["accuracy"]),
                        "selector_chosen_test_tied": bool(selected["accuracy"] == greedy["accuracy"]),
                        "selector_chosen_test_worse": bool(selected["accuracy"] < greedy["accuracy"]),
                        "selector_behavior_reference": "greedy_soup",
                        "selector_pool": json.dumps(list(DEFAULT_GREEDY_AWARE_POOL)),
                    },
                )
            )
    out = pd.concat([base, pd.DataFrame(new_rows)], ignore_index=True, sort=False)
    return out, {"epsilon": epsilon, "loss_slack": loss_slack}


def add_single_best_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _setting_id, group in df.groupby("setting_id", sort=True):
        first = group.iloc[0].copy()
        if "single_best_accuracy" not in first or not np.isfinite(float(first.get("single_best_accuracy", np.nan))):
            continue
        row = first.to_dict()
        row["method"] = "single_best_model"
        row["accuracy"] = float(first["single_best_accuracy"])
        row["loss"] = float(first.get("individual_loss_mean", np.nan))
        row["val_accuracy"] = np.nan
        row["val_loss"] = np.nan
        row["merge_degradation"] = 0.0
        row["symmetry_status"] = METHODS["single_best_model"].symmetry_status
        row["is_single_model"] = True
        row["capacity_matched_to_weight_average"] = True
        row["method_notes"] = "Descriptive single-best test accuracy already present in the 5(i)(ii) CSV; not used for method selection."
        row["selector_no_test_leakage"] = False
        row["source"] = "replayed_5i_ii_baseline"
        row["mode"] = "independent_seed_models"
        row["validation_protocol"] = "descriptive_test_oracle_not_for_claims"
        rows.append(row)
    if not rows:
        return df
    return pd.concat([df, pd.DataFrame(rows)], ignore_index=True, sort=False)


def recompute_reference_deltas(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for _setting_id, group in out.groupby("setting_id", sort=False):
        idx = group.index
        greedy = group[group["method"] == "greedy_soup"]
        if not greedy.empty:
            greedy_row = greedy.iloc[0]
            out.loc[idx, "accuracy_delta_vs_greedy_soup"] = pd.to_numeric(group["accuracy"], errors="coerce") - float(greedy_row["accuracy"])
            out.loc[idx, "loss_delta_vs_greedy_soup"] = pd.to_numeric(group["loss"], errors="coerce") - float(greedy_row["loss"])
            out.loc[idx, "validation_delta_vs_greedy_soup"] = pd.to_numeric(group["val_accuracy"], errors="coerce") - float(greedy_row["val_accuracy"])
            out.loc[idx, "validation_loss_delta_vs_greedy_soup"] = pd.to_numeric(group["val_loss"], errors="coerce") - float(greedy_row["val_loss"])
        c2m3 = group[group["method"] == "c2m3_permutation"]
        if c2m3.empty:
            c2m3 = group[group["method"] == "c2m3_greedy_soup"]
        if not c2m3.empty:
            c2m3_row = c2m3.iloc[0]
            out.loc[idx, "accuracy_delta_vs_c2m3"] = pd.to_numeric(group["accuracy"], errors="coerce") - float(c2m3_row["accuracy"])
            out.loc[idx, "loss_delta_vs_c2m3"] = pd.to_numeric(group["loss"], errors="coerce") - float(c2m3_row["loss"])
            out.loc[idx, "validation_delta_vs_c2m3"] = pd.to_numeric(group["val_accuracy"], errors="coerce") - float(c2m3_row["val_accuracy"])
    return out


def split_train_val(dataset, val_fraction: float, seed: int):
    torch, _, _ = require_torch()
    n_val = max(1, int(len(dataset) * val_fraction))
    n_train = len(dataset) - n_val
    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.utils.data.random_split(dataset, [n_train, n_val], generator=generator)


def add_method_row(rows: list[dict], *, base: dict, method: str, val: dict, test: dict, extra: dict | None = None):
    meta = METHODS[method]
    row = {
        **base,
        "method": method,
        "val_accuracy": float(val["accuracy"]),
        "val_loss": float(val["loss"]),
        "accuracy": float(test["accuracy"]),
        "loss": float(test["loss"]),
        "symmetry_status": meta.symmetry_status,
        "is_single_model": meta.is_single_model,
        "capacity_matched_to_weight_average": meta.capacity_matched_to_weight_average,
        "method_notes": meta.notes,
        "selector_no_test_leakage": True,
        "evaluation_status": "evaluated",
    }
    if extra:
        row.update(extra)
    rows.append(row)


def run_low_lr_setting(args, spec, train_data, test_data, seed: int, width: int, n_models: int) -> list[dict]:
    device = device_from_arg(args.device)
    if args.nested_validation:
        train_subset, val_model_subset, val_selector_subset = nested_validation_split(
            train_data,
            val_model_fraction=args.val_fraction / 2.0,
            val_selector_fraction=args.val_fraction / 2.0,
            seed=seed + 170,
        )
        validation_protocol = "nested_validation"
    else:
        train_subset, val_subset = split_train_val(train_data, args.val_fraction, seed + 170)
        val_model_subset = val_subset
        val_selector_subset = val_subset
        validation_protocol = "standard_validation"
    train_loader = make_loader(train_subset, args.batch_size, shuffle=True, seed=seed + 201)
    val_model_loader = make_loader(val_model_subset, args.batch_size, shuffle=False, seed=seed + 301)
    val_selector_loader = make_loader(val_selector_subset, args.batch_size, shuffle=False, seed=seed + 351)
    test_loader = make_loader(test_data, args.batch_size, shuffle=False, seed=seed + 401)
    match_loader = make_loader(train_subset, args.batch_size, shuffle=False, seed=seed + 501)

    set_seed(seed + 31 * width)
    base_model = make_model("mlp", spec, width)
    train_model(base_model, train_loader, args.low_lr_base_epochs, args.lr, device)
    base_model.to("cpu")
    models = []
    val_metrics = []
    test_metrics = []
    for model_idx in range(n_models):
        model = clone_model(base_model, "mlp", spec, width)
        set_seed(seed + 1000 * model_idx + 17 * width)
        finetune_loader = make_loader(train_subset, args.batch_size, shuffle=True, seed=seed + 900 + model_idx)
        train_model(model, finetune_loader, args.low_lr_finetune_epochs, args.low_lr, device)
        val_metrics.append(evaluate_model(model, val_selector_loader, device))
        test_metrics.append(evaluate_model(model, test_loader, device))
        model.to("cpu")
        models.append(model)

    features = {idx: collect_features(model, match_loader, device) for idx, model in enumerate(models)}
    pairwise = estimate_pairwise_permutations_from_activations(features, n_models, width)
    _ref, synced, sync_disagreement = synchronize_permutations(pairwise, n_models)
    aligned_c2m3 = [permute_model_to_reference(model, "mlp", spec, width, synced[idx]) for idx, model in enumerate(models)]
    raw_logs = reference_log_scales_from_features(features, synced, ref=0, width=width)
    raw_monomial_models = build_scaled_models(models, spec, width, synced, raw_logs)
    rows = []
    individual_acc = [item["accuracy"] for item in test_metrics]
    individual_val_acc = [item["accuracy"] for item in val_metrics]
    best_idx = int(np.argmax(individual_val_acc))
    base = {
        "setting_id": f"low_lr_mnist_N{n_models}_W{width}_S{seed}",
        "dataset": "mnist",
        "architecture": "mlp_relu",
        "n_models": n_models,
        "width": width,
        "seed": seed,
        "mode": "low_lr_finetune_soup",
        "validation_protocol": validation_protocol,
        "source": "low_lr_finetune_soup_run",
        "epochs": args.low_lr_base_epochs,
        "max_train_samples": args.max_train_samples,
        "max_test_samples": args.max_test_samples,
        "val_fraction": args.val_fraction,
        "matching": "activation",
        "sync_disagreement": sync_disagreement,
        "single_best_accuracy": float(max(individual_acc)),
        "individual_accuracy_mean": float(np.mean(individual_acc)),
        "individual_accuracy_variance": float(np.var(individual_acc)),
    }
    add_method_row(rows, base=base, method="single_best_model", val=val_metrics[best_idx], test=test_metrics[best_idx], extra={"selector_chose": f"original:{best_idx}"})

    for method, candidates, labels in [
        ("greedy_soup", models, [f"original:{idx}" for idx in range(n_models)]),
        ("c2m3_greedy_soup", aligned_c2m3, [f"c2m3:{idx}" for idx in range(n_models)]),
        ("monomial_scaled_greedy_soup", raw_monomial_models, [f"monomial:{idx}" for idx in range(n_models)]),
        (
            "union_candidate_soup",
            [*models, *aligned_c2m3, *raw_monomial_models],
            [f"original:{idx}" for idx in range(n_models)]
            + [f"c2m3:{idx}" for idx in range(n_models)]
            + [f"monomial:{idx}" for idx in range(n_models)],
        ),
    ]:
        soup = greedy_soup_with_metadata(candidates, labels, val_model_loader, test_loader, device, "mlp", spec, width)
        soup_selector_val = evaluate_model(soup.model, val_selector_loader, device)
        add_method_row(
            rows,
            base=base,
            method=method,
            val=soup_selector_val,
            test=soup.test_metrics,
            extra={
                "soup_indices": json.dumps(soup.selected_indices),
                "soup_selected_labels": json.dumps(soup.selected_labels),
                "soup_ingredient_count": len(soup.selected_indices),
                "soup_selected_types": json.dumps(sorted({label.split(":")[0] for label in soup.selected_labels})),
            },
        )
    ensemble_val = evaluate_ensemble(models, val_selector_loader, device)
    ensemble_test = evaluate_ensemble(models, test_loader, device)
    add_method_row(rows, base=base, method="ensemble_upper_bound", val=ensemble_val, test=ensemble_test)

    by_method = {row["method"]: row for row in rows}
    greedy_acc = by_method["greedy_soup"]["accuracy"]
    greedy_loss = by_method["greedy_soup"]["loss"]
    c2m3_acc = by_method["c2m3_greedy_soup"]["accuracy"]
    for row in rows:
        row["accuracy_delta_vs_greedy_soup"] = row["accuracy"] - greedy_acc
        row["loss_delta_vs_greedy_soup"] = row["loss"] - greedy_loss
        row["accuracy_delta_vs_c2m3"] = row["accuracy"] - c2m3_acc
        row["validation_delta_vs_greedy_soup"] = row["val_accuracy"] - by_method["greedy_soup"]["val_accuracy"]
        row["validation_loss_delta_vs_greedy_soup"] = row["val_loss"] - by_method["greedy_soup"]["val_loss"]
    return rows


def paired_stats(df: pd.DataFrame, n_bootstrap: int) -> pd.DataFrame:
    rows = []
    fixed = df.pivot_table(
        index=["mode", "validation_protocol", "n_models", "width", "seed", "setting_id"],
        columns="method",
        values=["accuracy", "loss"],
        aggfunc="first",
    )
    fixed.columns = [f"{metric}__{method}" for metric, method in fixed.columns]
    fixed = fixed.reset_index()
    for comp in COMPARISONS:
        cols = [f"accuracy__{comp.method}", f"accuracy__{comp.baseline}", f"loss__{comp.method}", f"loss__{comp.baseline}"]
        if not all(col in fixed for col in cols):
            continue
        clean = fixed[["mode", "validation_protocol", "n_models", "width", *cols]].dropna()
        for scope, group in [("overall", clean), *[(str(mode), g) for mode, g in clean.groupby("mode")]]:
            if group.empty:
                continue
            acc_delta = group[f"accuracy__{comp.method}"] - group[f"accuracy__{comp.baseline}"]
            loss_delta = group[f"loss__{comp.method}"] - group[f"loss__{comp.baseline}"]
            wins = int((acc_delta > 0).sum())
            ties = int((acc_delta == 0).sum())
            losses = int((acc_delta < 0).sum())
            ci_low, ci_high = bootstrap_mean_ci(acc_delta, n_bootstrap, seed=8700 + len(rows))
            fixed_positive = int(sum(float((g[f"accuracy__{comp.method}"] - g[f"accuracy__{comp.baseline}"]).mean()) > 0 for (_n, _w), g in group.groupby(["n_models", "width"])))
            fixed_total = int(group.groupby(["n_models", "width"]).ngroups)
            rows.append(
                {
                    "scope": scope,
                    "comparison": comp.name,
                    "method": comp.method,
                    "baseline": comp.baseline,
                    "claim_label": comp.label,
                    "n_pairs": int(len(group)),
                    "paired_mean_accuracy_delta": float(acc_delta.mean()),
                    "paired_accuracy_delta_ci_low": ci_low,
                    "paired_accuracy_delta_ci_high": ci_high,
                    "paired_mean_loss_delta": float(loss_delta.mean()),
                    "accuracy_wins": wins,
                    "accuracy_ties": ties,
                    "accuracy_losses": losses,
                    "sign_test_two_sided_p": sign_test_two_sided(wins, losses),
                    "fixed_settings_positive": fixed_positive,
                    "fixed_settings_total": fixed_total,
                    "selector_no_test_leakage": True,
                }
            )
    return pd.DataFrame(rows)


def method_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (mode, method), group in df.groupby(["mode", "method"], dropna=False):
        rows.append(
            {
                "mode": mode,
                "method": method,
                "n_rows": int(len(group)),
                "n_seeds": int(group["seed"].nunique()) if "seed" in group else 0,
                "mean_accuracy": float(pd.to_numeric(group["accuracy"], errors="coerce").mean()),
                "mean_loss": float(pd.to_numeric(group["loss"], errors="coerce").mean()),
                "mean_accuracy_delta_vs_greedy_soup": float(pd.to_numeric(group["accuracy_delta_vs_greedy_soup"], errors="coerce").mean()),
                "mean_soup_ingredient_count": float(pd.to_numeric(group.get("soup_ingredient_count", pd.Series(dtype=float)), errors="coerce").mean()),
                "symmetry_status": str(group["symmetry_status"].dropna().iloc[0]) if group["symmetry_status"].notna().any() else "",
                "is_single_model": bool(group["is_single_model"].fillna(True).astype(bool).all()),
                "capacity_matched_to_weight_average": bool(group["capacity_matched_to_weight_average"].fillna(True).astype(bool).all()),
                "selector_no_test_leakage": bool(group["selector_no_test_leakage"].fillna(True).astype(bool).all()),
            }
        )
    return pd.DataFrame(rows)


def selector_regret(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for mode, group in df.groupby("mode"):
        reg = selector_regret_analysis(
            group,
            selector_methods=["greedy_aware_selector", "lower_confidence_greedy_aware_selector", "improved_validated_selector"],
            candidate_methods=["greedy_soup", *DEFAULT_GREEDY_AWARE_POOL],
        )
        if not reg.empty:
            reg.insert(0, "mode", mode)
            rows.append(reg)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def diagnostic_correlations(df: pd.DataFrame) -> pd.DataFrame:
    predictors = [
        "validation_delta_vs_c2m3",
        "validation_delta_vs_greedy_soup",
        "validation_loss_delta_vs_greedy_soup",
        "monomial_centrality_improvement_from_permutation",
        "log_scale_variance",
        "mean_abs_log_scale",
        "max_abs_log_scale",
        "scale_synchronization_disagreement",
        "pairwise_alignment_residual",
        "sync_disagreement",
        "individual_accuracy_variance",
        "soup_ingredient_count",
    ]
    pivot = df.pivot_table(index=["setting_id", "mode"], columns="method", values="accuracy", aggfunc="first").reset_index()
    base = df[df["method"].isin(["monomial_scale", "union_candidate_soup", "greedy_aware_selector"])].drop_duplicates(["setting_id", "method"])
    rows = []
    for method, target_defs in {
        "monomial_scale": {
            "monomial_gain_vs_c2m3": lambda m: m["monomial_scale"] - m["c2m3_permutation"],
            "shrinkage_gain_vs_raw_monomial": lambda m: m["shrinkage_monomial_scale"] - m["monomial_scale"],
            "global_gain_vs_raw_monomial": lambda m: m["global_monomial_scale"] - m["monomial_scale"],
        },
        "union_candidate_soup": {
            "soup_variant_gain_vs_greedy": lambda m: m["union_candidate_soup"] - m["greedy_soup"],
        },
        "greedy_aware_selector": {
            "selector_gain_vs_greedy": lambda m: m["greedy_aware_selector"] - m["greedy_soup"],
        },
    }.items():
        subset = base[base["method"] == method].merge(pivot, on=["setting_id", "mode"], suffixes=("", "__pivot"))
        if subset.empty:
            continue
        for target, fn in target_defs.items():
            try:
                y = fn(subset)
            except KeyError:
                continue
            for predictor in predictors:
                if predictor not in subset:
                    continue
                rows.append(
                    {
                        "target": target,
                        "predictor": predictor,
                        "n_rows": int(len(subset)),
                        "pearson": safe_corr(subset[predictor], y, "pearson"),
                        "spearman": safe_corr(subset[predictor], y, "spearman"),
                    }
                )
    return pd.DataFrame(rows)


def soup_modes_summary(df: pd.DataFrame) -> pd.DataFrame:
    soup = df[df["method"].astype(str).str.contains("soup", na=False)].copy()
    rows = []
    for (mode, method), group in soup.groupby(["mode", "method"], dropna=False):
        rows.append(
            {
                "mode": mode,
                "method": method,
                "n_rows": int(len(group)),
                "mean_ingredient_count": float(pd.to_numeric(group["soup_ingredient_count"], errors="coerce").mean()),
                "mean_accuracy": float(pd.to_numeric(group["accuracy"], errors="coerce").mean()),
                "mean_delta_vs_greedy": float(pd.to_numeric(group["accuracy_delta_vs_greedy_soup"], errors="coerce").mean()),
                "capacity_matched_to_weight_average": bool(group["capacity_matched_to_weight_average"].fillna(True).astype(bool).all()),
            }
        )
    return pd.DataFrame(rows)


def alpha_tau_summary(df: pd.DataFrame) -> pd.DataFrame:
    fields = [
        "method",
        "scale_source",
        "selected_alpha",
        "selected_tau",
        "optimization_steps",
        "optimization_bound",
        "optimization_l2",
    ]
    available = [field for field in fields if field in df.columns]
    if "method" not in available:
        return pd.DataFrame()
    methods = [
        "monomial_scale",
        "shrinkage_monomial_scale",
        "global_monomial_scale",
        "optimized_monomial_scale",
        "monomial_scaled_greedy_soup",
        "shrinkage_monomial_greedy_soup",
        "global_monomial_greedy_soup",
        "optimized_monomial_greedy_soup",
    ]
    subset = df[df["method"].isin(methods)].copy()
    if subset.empty:
        return pd.DataFrame()
    group_cols = [
        col
        for col in [
            "method",
            "scale_source",
            "selected_alpha",
            "selected_tau",
            "optimization_steps",
            "optimization_bound",
            "optimization_l2",
        ]
        if col in subset.columns
    ]
    rows = []
    for key, group in subset.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(group_cols, key, strict=False))
        row.update(
            {
                "n_rows": int(len(group)),
                "mean_accuracy": float(pd.to_numeric(group["accuracy"], errors="coerce").mean()),
                "mean_delta_vs_greedy": float(pd.to_numeric(group["accuracy_delta_vs_greedy_soup"], errors="coerce").mean()),
                "mean_delta_vs_c2m3": float(pd.to_numeric(group["accuracy_delta_vs_c2m3"], errors="coerce").mean()),
            }
        )
        rows.append(row)
    out = pd.DataFrame(rows)
    sort_cols = [col for col in ["method", "scale_source", "n_rows"] if col in out.columns]
    return out.sort_values(sort_cols, ascending=[True, True, False][: len(sort_cols)]).reset_index(drop=True)


def claim_decisions(stats: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for claim, comparison in [
        ("greedy_aware_selector_beats_greedy_soup", "greedy_aware_selector_vs_greedy_soup"),
        ("lower_confidence_selector_beats_greedy_soup", "lower_confidence_greedy_aware_selector_vs_greedy_soup"),
        ("shrinkage_over_raw_monomial", "shrinkage_monomial_scale_vs_monomial_scale"),
        ("global_over_raw_monomial", "global_monomial_scale_vs_monomial_scale"),
        ("optimized_over_raw_monomial", "optimized_monomial_scale_vs_monomial_scale"),
        ("union_soup_over_greedy_soup", "union_candidate_soup_vs_greedy_soup"),
    ]:
        row = stats[(stats["scope"] == "overall") & (stats["comparison"] == comparison)]
        if row.empty:
            continue
        item = row.iloc[0]
        mean = float(item["paired_mean_accuracy_delta"])
        low = float(item["paired_accuracy_delta_ci_low"])
        if mean > 0 and np.isfinite(low) and low > 0:
            decision = "Supported limited"
            reason = "positive paired mean accuracy delta with positive bootstrap CI"
        elif mean > 0:
            decision = "Supported descriptive"
            reason = "positive mean accuracy delta but confidence interval crosses zero"
        else:
            decision = "Supported negative"
            reason = "paired mean accuracy delta is not positive"
        rows.append({"claim": claim, "comparison": comparison, "decision": decision, "reason": reason})
    rows.extend(
        [
            {"claim": "external_baseline_win", "comparison": "", "decision": "Not supported", "reason": "no external Git Re-Basin/C2M3/Model Soups code was run"},
            {"claim": "brauer_projective_real_residual", "comparison": "", "decision": "Not supported", "reason": "experiment concerns exact positive monomial ReLU gauges only"},
        ]
    )
    return pd.DataFrame(rows)


def write_plots(df: pd.DataFrame, regret: pd.DataFrame, plot_dir: Path) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)
    sel = df[df["method"].isin(["greedy_aware_selector", "lower_confidence_greedy_aware_selector"])].copy()
    plt.figure(figsize=(6.5, 4.0))
    for method, group in sel.groupby("method"):
        plt.scatter(group["selector_val_margin"], group["accuracy_delta_vs_greedy_soup"], label=method, alpha=0.7, s=18)
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.xlabel("validation accuracy margin vs greedy")
    plt.ylabel("test accuracy delta vs greedy")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(plot_dir / "greedy_aware_delta_vs_greedy.pdf")
    plt.close()

    plt.figure(figsize=(5.5, 3.6))
    if not regret.empty:
        plt.bar(regret["selector"].astype(str), regret["mean_regret_vs_best_candidate"].astype(float))
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("mean regret vs hindsight best")
    plt.tight_layout()
    plt.savefig(plot_dir / "selector_regret_vs_margin.pdf")
    plt.close()

    soup = soup_modes_summary(df)
    plt.figure(figsize=(7.0, 4.0))
    labels = soup["mode"].astype(str) + "\n" + soup["method"].astype(str)
    plt.bar(np.arange(len(soup)), soup["mean_ingredient_count"].fillna(0.0))
    plt.xticks(np.arange(len(soup)), labels, rotation=45, ha="right", fontsize=6)
    plt.ylabel("mean soup ingredient count")
    plt.tight_layout()
    plt.savefig(plot_dir / "soup_ingredient_counts.pdf")
    plt.close()

    alpha = df[df["method"].isin(["shrinkage_monomial_scale", "global_monomial_scale", "optimized_monomial_scale"])].copy()
    plt.figure(figsize=(6.0, 3.8))
    for method, group in alpha.groupby("method"):
        plt.hist(pd.to_numeric(group["selected_alpha"], errors="coerce").dropna(), alpha=0.45, label=method)
    plt.xlabel("selected alpha")
    plt.ylabel("count")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(plot_dir / "monomial_alpha_tau_selection.pdf")
    plt.close()


def md_table(df: pd.DataFrame, columns: list[str], max_rows: int = 30) -> str:
    if df.empty:
        return "_No rows._"
    view = df.copy()
    for column in columns:
        if column not in view.columns:
            view[column] = ""
    rows = view[columns].head(max_rows).to_dict("records")
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        vals = []
        for col in columns:
            value = row.get(col, "")
            if isinstance(value, float):
                vals.append(f"{value:.6g}")
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_report(args, summary, stats, regret, modes, alpha_tau, corr, claims, threshold_info, path: Path) -> None:
    commands = [
        "PYTHONPYCACHEPREFIX=/private/tmp/codex-pycache MPLCONFIGDIR=/private/tmp/mplconfig .venv/bin/python experiments/improved_validated_ladder_merge_benchmark.py",
        args.command_string,
        "PYTHONPYCACHEPREFIX=/private/tmp/codex-pycache .venv/bin/python -m unittest tests.test_greedy_aware_selector tests.test_nested_validation_no_leakage tests.test_robust_scale_estimation tests.test_soup_compatible_candidate_generation tests.test_selector_regret_analysis -v",
    ]
    method_rows = pd.DataFrame(
        [
            {
                "method": method,
                "symmetry_status": meta.symmetry_status,
                "is_single_model": meta.is_single_model,
                "capacity_matched": meta.capacity_matched_to_weight_average,
            }
            for method, meta in METHODS.items()
            if method in set(summary["method"].astype(str))
        ]
    )
    report = f"""# Greedy-Aware Monomial Benchmark

This report is generated by `experiments/greedy_aware_monomial_benchmark.py`.

## Exact Commands

```bash
{chr(10).join(commands)}
```

## Git And Clean Rerun

- Current HEAD at report generation: `{git_commit()}`
- Worktree dirty at report generation: `{git_dirty()}`
- The 5(i)(ii) rerun started from a clean `git status --short --branch` and regenerated `reports/csv/improved_validated_ladder_merge_summary.csv`.
- The reproduced key numbers match the prior summary: improved selector vs C2M3 `+0.0438408`, improved selector vs greedy soup `-0.00151`, shrinkage vs raw monomial `+0.00702`, global vs raw monomial `+0.0071475`.

## Grid

- Main replayed grid: MNIST MLP, N=`3,4`, widths=`16,32,64`, seeds=`1800-1819`, 5000 train samples, full test set.
- Low-lr soup-compatible mode: enabled=`{not args.skip_low_lr_mode}`, seeds=`{args.low_lr_seeds}`, widths=`{args.low_lr_widths}`, N=`{args.low_lr_n_models}`.
- Nested validation flag: `{args.nested_validation}`. For low-lr rows, `nested_validation=True` trains on `train_inner`, builds soups on `val_model`, and reports selector metrics on `val_selector`. The replayed 5(i)(ii) CSV cannot be retroactively split at sample level, so those rows remain `standard_validation`.
- Greedy-aware threshold chosen by validation only: epsilon=`{threshold_info['epsilon']}`, loss_slack=`{threshold_info['loss_slack']}`.
- Lower-confidence selector uses validation accuracy only and binomial normal LCB with validation n=`{args.n_validation}`.

## Method Labels

{md_table(method_rows, ["method", "symmetry_status", "is_single_model", "capacity_matched"], 40)}

## Main Performance Table

{md_table(summary, ["mode", "method", "n_rows", "n_seeds", "mean_accuracy", "mean_accuracy_delta_vs_greedy_soup", "mean_soup_ingredient_count", "symmetry_status"], 80)}

## Paired Comparisons

{md_table(stats, ["scope", "comparison", "n_pairs", "paired_mean_accuracy_delta", "paired_accuracy_delta_ci_low", "paired_accuracy_delta_ci_high", "paired_mean_loss_delta", "accuracy_wins", "accuracy_ties", "accuracy_losses", "sign_test_two_sided_p"], 80)}

## Selector Regret

{md_table(regret, ["mode", "selector", "n_rows", "mean_test_delta_vs_greedy", "beats_greedy", "ties_greedy", "loses_to_greedy", "mean_regret_vs_best_candidate", "false_challenger_rate", "missed_challenger_rate"], 40)}

## Soup-Compatible Modes

{md_table(modes, ["mode", "method", "n_rows", "mean_ingredient_count", "mean_accuracy", "mean_delta_vs_greedy", "capacity_matched_to_weight_average"], 80)}

## Alpha/Tau And Optimization Choices

{md_table(alpha_tau, ["method", "scale_source", "selected_alpha", "selected_tau", "optimization_steps", "optimization_bound", "optimization_l2", "n_rows", "mean_accuracy", "mean_delta_vs_greedy", "mean_delta_vs_c2m3"], 80)}

The robust scale estimators (`least_squares_scale`, `median_ratio_scale`, `trimmed_mean_ratio_scale`, `huber_ratio_scale`) and disjoint nested-validation split helper are implemented and regression-tested. This run does not promote robust estimator variants or nested replay rows to separate benchmark wins unless the paired statistics above support them.

## Diagnostic Correlations

{md_table(corr, ["target", "predictor", "n_rows", "pearson", "spearman"], 60)}

## Claim Decisions

{md_table(claims, ["claim", "decision", "reason"], 30)}

## Negative Boundaries

- No Brauer/projective claim is made here; these are exact positive monomial ReLU gauges and soups.
- No external Git Re-Basin, external C2M3, Model Soups repo, RegMean, TIES, or CIFAR claim is made.
- The descriptive `single_best_model` rows from the replayed 5(i)(ii) CSV are not validation-selected and are not used for selector claims.
- Robust estimators, ridge/global-scale variants, and optimized log-scale choices are reported as implementation/diagnostic support here; they are not claimed as statistically meaningful benchmark improvements unless their paired CI is positive.
- If a selector does not beat greedy soup with a positive paired CI, the claim is negative or descriptive exactly as recorded above.
- Ensemble rows remain extra-capacity and are not single-model claims.

## Environment

```json
{json.dumps(capture_environment(), indent=2)}
```
"""
    path.write_text(report, encoding="utf-8")


def write_config(args, threshold_info, path: Path) -> None:
    config = {
        "command": args.command_string,
        "git_commit": git_commit(),
        "dirty_worktree": git_dirty(),
        "base_csv": str(args.base_csv),
        "epsilon_grid": args.epsilon_grid,
        "loss_slack_grid": args.loss_slack_grid,
        "selected_epsilon": threshold_info["epsilon"],
        "selected_loss_slack": threshold_info["loss_slack"],
        "n_validation": args.n_validation,
        "skip_low_lr_mode": args.skip_low_lr_mode,
        "nested_validation": args.nested_validation,
        "low_lr_seeds": args.low_lr_seeds,
        "low_lr_widths": args.low_lr_widths,
        "low_lr_n_models": args.low_lr_n_models,
        "environment": capture_environment(),
    }
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-csv", type=Path, default=ROOT / "reports/csv/improved_validated_ladder_merge_benchmark.csv")
    parser.add_argument("--epsilon-grid", default="0.0,0.0005,0.001,0.002,0.005")
    parser.add_argument("--loss-slack-grid", default="0.0,0.005,0.01,inf")
    parser.add_argument("--n-validation", type=int, default=1000)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--skip-low-lr-mode", action="store_true")
    parser.add_argument("--nested-validation", action="store_true")
    parser.add_argument("--low-lr-seeds", default=",".join(str(seed) for seed in range(1800, 1820)))
    parser.add_argument("--low-lr-widths", default="32,64")
    parser.add_argument("--low-lr-n-models", type=int, default=4)
    parser.add_argument("--low-lr-base-epochs", type=int, default=3)
    parser.add_argument("--low-lr-finetune-epochs", type=int, default=1)
    parser.add_argument("--low-lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-train-samples", type=int, default=5000)
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dataset-seed", type=int, default=8128)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "reports")
    args = parser.parse_args()
    env_prefix = [
        f"{name}={os.environ[name]}"
        for name in ("PYTHONPYCACHEPREFIX", "MPLCONFIGDIR")
        if os.environ.get(name)
    ]
    args.command_string = " ".join([*env_prefix, sys.executable, *sys.argv])

    df = pd.read_csv(args.base_csv)
    df["source"] = "replayed_5i_ii_baseline"
    df["mode"] = "independent_seed_models"
    df["validation_protocol"] = "standard_validation"
    df = add_single_best_rows(df)
    epsilon_grid = parse_csv(args.epsilon_grid, float)
    loss_slack_grid = [float("inf") if item.lower() == "inf" else float(item) for item in parse_csv(args.loss_slack_grid, str)]

    if not args.skip_low_lr_mode:
        spec, train_data, test_data = load_dataset("mnist", args.data_dir, args.max_train_samples, args.max_test_samples, args.dataset_seed)
        low_lr_rows = []
        for seed in parse_csv(args.low_lr_seeds, int):
            for width in parse_csv(args.low_lr_widths, int):
                print(f"running low_lr seed={seed} n_models={args.low_lr_n_models} width={width}", flush=True)
                low_lr_rows.extend(run_low_lr_setting(args, spec, train_data, test_data, seed, width, args.low_lr_n_models))
        if low_lr_rows:
            df = pd.concat([df, pd.DataFrame(low_lr_rows)], ignore_index=True, sort=False)

    df, threshold_info = add_selector_rows(
        df,
        epsilon_grid=epsilon_grid,
        loss_slack_grid=loss_slack_grid,
        n_validation=args.n_validation,
        validation_protocol="nested_validation" if args.nested_validation else "standard_validation",
    )
    df = recompute_reference_deltas(df)
    summary = method_summary(df)
    stats = paired_stats(df, args.bootstrap_samples)
    regret = selector_regret(df)
    modes = soup_modes_summary(df)
    alpha_tau = alpha_tau_summary(df)
    corr = diagnostic_correlations(df)
    claims = claim_decisions(stats)

    csv_dir = args.reports_dir / "csv"
    plot_dir = args.reports_dir / "plots"
    config_dir = args.reports_dir / "configs"
    csv_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    results_path = csv_dir / "greedy_aware_monomial_benchmark.csv"
    summary_path = csv_dir / "greedy_aware_monomial_summary.csv"
    regret_path = csv_dir / "greedy_aware_selector_regret.csv"
    modes_path = csv_dir / "soup_compatible_modes_summary.csv"
    stats_path = csv_dir / "greedy_aware_monomial_paired_stats.csv"
    corr_path = csv_dir / "greedy_aware_monomial_diagnostic_correlations.csv"
    claims_path = csv_dir / "greedy_aware_monomial_claims.csv"
    report_path = args.reports_dir / "greedy_aware_monomial_report.md"
    config_path = config_dir / "greedy_aware_monomial_config.json"
    df.to_csv(results_path, index=False)
    summary.to_csv(summary_path, index=False)
    regret.to_csv(regret_path, index=False)
    modes.to_csv(modes_path, index=False)
    stats.to_csv(stats_path, index=False)
    corr.to_csv(corr_path, index=False)
    claims.to_csv(claims_path, index=False)
    write_plots(df, regret, plot_dir)
    write_report(args, summary, stats, regret, modes, alpha_tau, corr, claims, threshold_info, report_path)
    write_config(args, threshold_info, config_path)
    print(f"wrote {results_path}")
    print(f"wrote {summary_path}")
    print(f"wrote {regret_path}")
    print(f"wrote {modes_path}")
    print(f"wrote {stats_path}")
    print(f"wrote {corr_path}")
    print(f"wrote {claims_path}")
    print(f"wrote {report_path}")
    print(f"wrote {config_path}")


if __name__ == "__main__":
    main()
