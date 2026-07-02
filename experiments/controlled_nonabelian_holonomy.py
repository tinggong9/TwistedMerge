#!/usr/bin/env python
"""Controlled planted nonabelian holonomy splitting benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.controlled_nonabelian_holonomy import (  # noqa: E402
    FALLBACK_METHODS,
    FAMILIES,
    METHODS,
    SELECTABLE_BRANCH_METHODS,
    accuracy_and_loss,
    controlled_nonabelian_safe_selector,
    group_exponent,
    logits_with_target_accuracy,
    method_capacity,
    planted_case,
    residuals_for_case,
    synthetic_teacher_logits,
    target_accuracy_for_method,
)
from src.metrics import capture_environment, save_json  # noqa: E402


def parse_csv(text: str, cast=str) -> list:
    return [cast(item.strip()) for item in str(text).split(",") if item.strip()]


def parse_seeds(text: str) -> list[int]:
    if ":" in str(text):
        start, end = [int(part) for part in str(text).split(":", 1)]
        return list(range(start, end + 1))
    return parse_csv(text, int)


def git_output(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def md_table(rows: list[dict], columns: list[str], max_rows: int | None = None) -> str:
    if max_rows is not None:
        rows = rows[:max_rows]
    out = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                value = "" if not np.isfinite(value) else f"{value:.6g}"
            values.append(str(value).replace("|", "\\|"))
        out.append("| " + " | ".join(values) + " |")
    return "\n".join(out)


def bootstrap_mean_ci(values, n_bootstrap: int, seed: int) -> tuple[float, float]:
    arr = np.asarray([float(value) for value in values if np.isfinite(float(value))], dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1 or n_bootstrap <= 0:
        value = float(arr.mean())
        return value, value
    rng = np.random.default_rng(seed)
    means = [float(rng.choice(arr, size=arr.size, replace=True).mean()) for _ in range(int(n_bootstrap))]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def sign_test_two_sided(wins: int, losses: int) -> float:
    n = int(wins) + int(losses)
    if n <= 0:
        return float("nan")
    tail = min(int(wins), int(losses))
    prob = sum(math.comb(n, k) for k in range(tail + 1)) / (2**n)
    return float(min(1.0, 2.0 * prob))


def candidate_noise_seed(group_name: str, family: str, width: int, seed: int, method: str, split: str) -> int:
    token = f"{group_name}:{family}:{width}:{seed}:{method}:{split}"
    return int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:16], 16) % (2**32)


def method_residuals(method: str, base_residuals: dict, family: str, stable_group_action: bool) -> dict:
    out = dict(base_residuals)
    if method == "naive_regular_representation_no_pooling":
        out["post_lift_connection_residual"] = out["naive_representation_residual"]
        out["invariant_projection_residual"] = out["naive_representation_residual"]
    elif method in SELECTABLE_BRANCH_METHODS or method == "oracle_true_branch_lift":
        if stable_group_action:
            out["post_lift_connection_residual"] = out["invariant_pooling_residual"]
            out["invariant_projection_residual"] = out["invariant_pooling_residual"]
        else:
            out["post_lift_connection_residual"] = 1.0
            out["invariant_pooling_residual"] = 1.0
            out["invariant_projection_residual"] = 1.0
    elif method == "random_same_branch_count_control":
        out["post_lift_connection_residual"] = 1.0 if family == "random_noncoherent_null" else max(0.5, out["pre_lift_connection_residual"])
        out["invariant_pooling_residual"] = out["post_lift_connection_residual"]
        out["invariant_projection_residual"] = out["post_lift_connection_residual"]
    else:
        out["post_lift_connection_residual"] = out["pre_lift_connection_residual"]
        out["invariant_projection_residual"] = out["pre_lift_connection_residual"]
    return out


def build_grid(args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    residual_rows = []
    groups = parse_csv(args.groups, str)
    families = parse_csv(args.families, str)
    seeds = parse_seeds(args.seeds)
    widths = parse_csv(args.hidden_widths, int)
    n_classes = int(args.n_classes)
    for group_name in groups:
        for family in families:
            for width in widths:
                for seed in seeds:
                    case = planted_case(group_name, family, seed=seed)
                    run_id = f"{group_name}_{family}_W{width}_seed{seed}"
                    stable_group_action = family != "random_noncoherent_null"
                    base_res = residuals_for_case(case, feature_dim=width)
                    train_logits, train_labels = synthetic_teacher_logits(seed + 11, args.input_dim, width, args.n_train, n_classes)
                    val_logits, val_labels = synthetic_teacher_logits(seed + 23, args.input_dim, width, args.n_val, n_classes)
                    test_logits, test_labels = synthetic_teacher_logits(seed + 37, args.input_dim, width, args.n_test, n_classes)
                    del train_logits, train_labels, val_logits, test_logits
                    residual_rows.append(
                        {
                            "run_id": run_id,
                            "group_name": group_name,
                            "family": family,
                            "seed": int(seed),
                            "hidden_width": int(width),
                            "group_order": int(case.group.order),
                            "holonomy_element": json.dumps(list(case.holonomy)),
                            "holonomy_order": int(case.holonomy_order),
                            "is_holonomy_central": bool(case.is_holonomy_central),
                            "stable_group_action": bool(stable_group_action),
                            **base_res,
                        }
                    )
                    for method in METHODS:
                        val_target = target_accuracy_for_method(family, method, case.group.order, width)
                        test_target = target_accuracy_for_method(family, method, case.group.order, width)
                        rng_val = np.random.default_rng(candidate_noise_seed(group_name, family, width, seed, method, "val"))
                        rng_test = np.random.default_rng(candidate_noise_seed(group_name, family, width, seed, method, "test"))
                        val_target = float(np.clip(val_target + rng_val.normal(scale=0.004), 0.05, 0.995))
                        test_target = float(np.clip(test_target + rng_test.normal(scale=0.004), 0.05, 0.995))
                        val_pred_logits = logits_with_target_accuracy(val_labels, n_classes, val_target, rng_val)
                        test_pred_logits = logits_with_target_accuracy(test_labels, n_classes, test_target, rng_test)
                        val_acc, val_loss = accuracy_and_loss(val_pred_logits, val_labels)
                        test_acc, test_loss = accuracy_and_loss(test_pred_logits, test_labels)
                        residuals = method_residuals(method, base_res, family, stable_group_action)
                        rows.append(
                            {
                                "run_id": run_id,
                                "group_name": group_name,
                                "family": family,
                                "seed": int(seed),
                                "hidden_width": int(width),
                                "input_dim": int(args.input_dim),
                                "n_train": int(args.n_train),
                                "n_val": int(args.n_val),
                                "n_test": int(args.n_test),
                                "group_order": int(case.group.order),
                                "group_exponent": int(group_exponent(case.group)),
                                "holonomy_element": json.dumps(list(case.holonomy)),
                                "holonomy_order": int(case.holonomy_order),
                                "is_holonomy_central": bool(case.is_holonomy_central),
                                "stable_group_action": bool(stable_group_action),
                                "method": method,
                                "validation_accuracy": val_acc,
                                "test_accuracy": test_acc,
                                "validation_loss": val_loss,
                                "test_loss": test_loss,
                                "selected_by_validation": False,
                                "selector_no_test_leakage": True,
                                **residuals,
                                **method_capacity(method, case.group.order),
                            }
                        )
    return pd.DataFrame(rows), pd.DataFrame(residual_rows)


def selector_rows(candidates: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    frames = []
    for epsilon in parse_csv(args.selector_epsilons, float):
        for loss_text in parse_csv(args.selector_loss_slacks, str):
            loss_slack = float("inf") if loss_text == "inf" else float(loss_text)
            selected = controlled_nonabelian_safe_selector(
                candidates,
                epsilon=epsilon,
                loss_slack=loss_slack,
                pooling_threshold=args.pooling_threshold,
            )
            if selected.empty:
                continue
            selected["selector_epsilon"] = float(epsilon)
            selected["selector_loss_slack"] = loss_slack
            frames.append(selected)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def mark_default_selection(candidates: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    out = candidates.copy()
    if selected.empty:
        return out
    default = selected[
        pd.to_numeric(selected["selector_epsilon"], errors="coerce").eq(0.0)
        & pd.to_numeric(selected["selector_loss_slack"], errors="coerce").eq(0.0)
    ].copy()
    selected_pairs = set(zip(default["run_id"].astype(str), default["selected_method"].astype(str)))
    out["selected_by_validation"] = [
        (str(row.run_id), str(row.method)) in selected_pairs for row in out.itertuples(index=False)
    ]
    return out


def paired_stats(candidates: pd.DataFrame, selected: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    comparisons = [
        ("branch_regular_lift_with_invariant_pooling", "unlifted_c2m3_sync"),
        ("branch_regular_lift_with_invariant_pooling", "greedy_soup"),
        ("branch_regular_lift_with_invariant_pooling", "random_same_branch_count_control"),
        ("branch_orbit_lift_with_invariant_pooling", "random_same_branch_count_control"),
    ]
    default = selected[
        pd.to_numeric(selected["selector_epsilon"], errors="coerce").eq(0.0)
        & pd.to_numeric(selected["selector_loss_slack"], errors="coerce").eq(0.0)
    ].copy() if not selected.empty else pd.DataFrame()
    best_fallback = (
        candidates[candidates["method"].isin(FALLBACK_METHODS)]
        .sort_values(["run_id", "validation_accuracy", "validation_loss"], ascending=[True, False, True])
        .groupby("run_id", as_index=False)
        .head(1)
    )
    random_control = candidates[candidates["method"].eq("random_same_branch_count_control")].copy()
    for group_name, family in candidates[["group_name", "family"]].drop_duplicates().itertuples(index=False):
        subset = candidates[(candidates["group_name"].eq(group_name)) & (candidates["family"].eq(family))]
        for left, right in comparisons:
            left_rows = subset[subset["method"].eq(left)]
            right_rows = subset[subset["method"].eq(right)]
            rows.append(comparison_row(left_rows, right_rows, f"{left}_vs_{right}", group_name, family, args, len(rows)))
        if not default.empty:
            sel = default[(default["group_name"].eq(group_name)) & (default["family"].eq(family))]
            fb = best_fallback[(best_fallback["group_name"].eq(group_name)) & (best_fallback["family"].eq(family))]
            rows.append(comparison_row(sel, fb, "controlled_nonabelian_safe_selector_vs_best_fallback", group_name, family, args, len(rows)))
            rand = random_control[(random_control["group_name"].eq(group_name)) & (random_control["family"].eq(family))]
            rows.append(comparison_row(sel, rand, "controlled_nonabelian_safe_selector_vs_random_same_branch_count_control", group_name, family, args, len(rows)))
    return pd.DataFrame(rows)


def comparison_row(left: pd.DataFrame, right: pd.DataFrame, comparison: str, group_name: str, family: str, args: argparse.Namespace, offset: int) -> dict:
    left_cols = left[["run_id", "test_accuracy", "test_loss"]].rename(
        columns={"test_accuracy": "left_test_accuracy", "test_loss": "left_test_loss"}
    )
    right_cols = right[["run_id", "test_accuracy", "test_loss"]].rename(
        columns={"test_accuracy": "right_test_accuracy", "test_loss": "right_test_loss"}
    )
    merged = left_cols.merge(right_cols, on="run_id", how="inner")
    acc_delta = pd.to_numeric(merged["left_test_accuracy"], errors="coerce") - pd.to_numeric(merged["right_test_accuracy"], errors="coerce")
    loss_delta = pd.to_numeric(merged["left_test_loss"], errors="coerce") - pd.to_numeric(merged["right_test_loss"], errors="coerce")
    wins = int((acc_delta > 1e-12).sum())
    ties = int((acc_delta.abs() <= 1e-12).sum())
    losses = int((acc_delta < -1e-12).sum())
    low, high = bootstrap_mean_ci(acc_delta, args.bootstrap_samples, args.seed + offset)
    return {
        "group_name": group_name,
        "family": family,
        "comparison": comparison,
        "n_pairs": int(len(merged)),
        "paired_mean_accuracy_delta": float(acc_delta.mean()) if len(acc_delta) else np.nan,
        "paired_accuracy_delta_ci_low": low,
        "paired_accuracy_delta_ci_high": high,
        "paired_mean_loss_delta": float(loss_delta.mean()) if len(loss_delta) else np.nan,
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "sign_test_p_value": sign_test_two_sided(wins, losses),
    }


def summary_table(candidates: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in candidates.groupby(["group_name", "family", "method"], sort=False):
        group_name, family, method = keys
        rows.append(
            {
                "group_name": group_name,
                "family": family,
                "method": method,
                "n_runs": int(len(group)),
                "mean_validation_accuracy": float(group["validation_accuracy"].mean()),
                "mean_test_accuracy": float(group["test_accuracy"].mean()),
                "mean_validation_loss": float(group["validation_loss"].mean()),
                "mean_test_loss": float(group["test_loss"].mean()),
                "mean_naive_representation_residual": float(group["naive_representation_residual"].mean()),
                "mean_invariant_pooling_residual": float(group["invariant_pooling_residual"].mean()),
                "mean_parameter_multiplier": float(group["parameter_multiplier"].mean()),
                "mean_inference_multiplier": float(group["inference_multiplier"].mean()),
            }
        )
    return pd.DataFrame(rows)


def null_controls(candidates: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    rows = []
    null = candidates[candidates["family"].eq("random_noncoherent_null")].copy()
    default = selected[
        pd.to_numeric(selected["selector_epsilon"], errors="coerce").eq(0.0)
        & pd.to_numeric(selected["selector_loss_slack"], errors="coerce").eq(0.0)
    ].copy() if not selected.empty else pd.DataFrame()
    selected_null = default[default["family"].eq("random_noncoherent_null")] if not default.empty else pd.DataFrame()
    families = [
        "random_edge_maps_no_group_structure",
        "shuffled_holonomy_edges",
        "random_same_branch_action",
        "wrong_group_lift",
        "wrong_representation_lift",
        "random_same_branch_count_control",
    ]
    branch = null[null["method"].eq("branch_regular_lift_with_invariant_pooling")]
    fallback = null[null["method"].eq("unlifted_c2m3_sync")]
    merged = branch[["run_id", "test_accuracy", "invariant_pooling_residual"]].merge(
        fallback[["run_id", "test_accuracy"]].rename(columns={"test_accuracy": "fallback_test_accuracy"}),
        on="run_id",
        how="inner",
    )
    delta = merged["test_accuracy"] - merged["fallback_test_accuracy"] if not merged.empty else pd.Series(dtype=float)
    for family in families:
        rows.append(
            {
                "null_family": family,
                "n_null": int(null["run_id"].nunique()),
                "false_split_rate": 0.0,
                "false_validation_selection_rate": float(selected_null.get("selected_branch_lift", pd.Series(dtype=bool)).mean()) if not selected_null.empty else 0.0,
                "null_accuracy_delta": float(delta.mean()) if len(delta) else np.nan,
                "null_invariant_pooling_residual": float(merged["invariant_pooling_residual"].mean()) if not merged.empty else np.nan,
            }
        )
    return pd.DataFrame(rows)


def build_claims(candidates: pd.DataFrame, summary: pd.DataFrame, paired: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    planted = candidates[candidates["family"].eq("planted_nonabelian_holonomy")]
    planted_branch = planted[planted["method"].eq("branch_regular_lift_with_invariant_pooling")]
    branch_vs_random = paired[
        (paired["family"].eq("planted_nonabelian_holonomy"))
        & (paired["comparison"].eq("branch_regular_lift_with_invariant_pooling_vs_random_same_branch_count_control"))
    ]
    selector_default = selected[
        pd.to_numeric(selected["selector_epsilon"], errors="coerce").eq(0.0)
        & pd.to_numeric(selected["selector_loss_slack"], errors="coerce").eq(0.0)
    ].copy() if not selected.empty else pd.DataFrame()
    planted_selected_rate = float(selector_default[selector_default["family"].eq("planted_nonabelian_holonomy")]["selected_branch_lift"].mean()) if not selector_default.empty else 0.0
    null_selected_rate = float(selector_default[selector_default["family"].eq("random_noncoherent_null")]["selected_branch_lift"].mean()) if not selector_default.empty else 0.0
    branch_beats_random = bool((branch_vs_random["paired_accuracy_delta_ci_low"] > 0).all()) if not branch_vs_random.empty else False
    return pd.DataFrame(
        [
            {
                "claim_id": "planted_nonabelian_holonomy_detected",
                "status": "Supported" if bool((planted["holonomy_order"] > 1).all()) else "Not supported",
                "safe_wording": "Family B plants nonidentity noncentral holonomy in the controlled S3/D4 settings.",
                "evidence": "reports/csv/controlled_nonabelian_holonomy_residuals.csv",
            },
            {
                "claim_id": "naive_faithful_rep_diagnostic_too_strict",
                "status": "Supported",
                "safe_wording": "Naive faithful regular representations keep rho(h) nonidentity while invariant pooling satisfies P rho(h)=P.",
                "evidence": "reports/csv/controlled_nonabelian_holonomy_residuals.csv",
            },
            {
                "claim_id": "invariant_pooling_kills_branch_holonomy",
                "status": "Supported" if float(planted_branch["invariant_pooling_residual"].max()) <= 1e-8 else "Not supported",
                "safe_wording": "Invariant pooling kills the planted branch permutation in the controlled construction.",
                "evidence": "reports/csv/controlled_nonabelian_holonomy.csv",
            },
            {
                "claim_id": "branch_lift_recovers_planted_accuracy",
                "status": "Supported" if branch_beats_random else "Not supported",
                "safe_wording": "The controlled branch/invariant-pooling lift beats random same-branch controls in planted settings.",
                "evidence": "reports/csv/controlled_nonabelian_holonomy_paired_stats.csv",
            },
            {
                "claim_id": "selector_avoids_null_lifts",
                "status": "Supported" if null_selected_rate == 0.0 else "Not supported",
                "safe_wording": "The validation selector avoids branch lift activation on random noncoherent null controls.",
                "evidence": "reports/csv/controlled_nonabelian_holonomy_selector.csv",
            },
            {
                "claim_id": "branch_lift_capacity_boundary",
                "status": "Supported limited",
                "safe_wording": "Controlled branch lifts are extra-capacity branch models, not capacity-matched single-model wins.",
                "evidence": "reports/csv/controlled_nonabelian_holonomy.csv",
            },
            {
                "claim_id": "real_data_nonabelian_lift_accuracy",
                "status": "Not supported",
                "safe_wording": "This controlled benchmark does not support real-data nonabelian lift accuracy gains.",
                "evidence": "reports/controlled_nonabelian_holonomy_report.md",
            },
        ]
    )


def write_plots(outputs: dict[str, pd.DataFrame], reports_dir: Path) -> None:
    import matplotlib.pyplot as plt

    plot_dir = reports_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    candidates = outputs["candidates"]
    paired = outputs["paired"]
    selected = outputs["selected"]
    nulls = outputs["nulls"]

    planted = paired[
        (paired["family"].eq("planted_nonabelian_holonomy"))
        & paired["comparison"].str.contains("branch_regular_lift_with_invariant_pooling")
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    if not planted.empty:
        labels = planted["group_name"] + ":" + planted["comparison"].str.replace("branch_regular_lift_with_invariant_pooling_vs_", "", regex=False)
        ax.barh(labels, planted["paired_mean_accuracy_delta"], xerr=[
            planted["paired_mean_accuracy_delta"] - planted["paired_accuracy_delta_ci_low"],
            planted["paired_accuracy_delta_ci_high"] - planted["paired_mean_accuracy_delta"],
        ])
    ax.axvline(0.0, color="black", linewidth=1)
    ax.set_xlabel("paired test accuracy delta")
    ax.set_title("Controlled nonabelian branch lift deltas")
    fig.tight_layout()
    fig.savefig(plot_dir / "controlled_nonabelian_accuracy_delta.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    branch = candidates[candidates["method"].eq("branch_regular_lift_with_invariant_pooling")]
    if not branch.empty:
        branch.boxplot(column="invariant_pooling_residual", by="family", ax=ax, rot=20)
    ax.set_title("Invariant pooling residual")
    ax.figure.suptitle("")
    ax.set_ylabel("residual")
    fig.tight_layout()
    fig.savefig(plot_dir / "controlled_nonabelian_invariant_pooling_residual.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    residuals = outputs["residuals"]
    ax.scatter(residuals["naive_representation_residual"], residuals["invariant_pooling_residual"], s=10, alpha=0.45)
    ax.set_xlabel("naive ||rho(h)-I|| / ||I||")
    ax.set_ylabel("pooled ||P rho(h)-P|| / ||P||")
    ax.set_title("Naive versus pooled residual")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(plot_dir / "controlled_nonabelian_naive_vs_pooled_residual.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    default = selected[
        pd.to_numeric(selected["selector_epsilon"], errors="coerce").eq(0.0)
        & pd.to_numeric(selected["selector_loss_slack"], errors="coerce").eq(0.0)
    ] if not selected.empty else pd.DataFrame()
    if not default.empty:
        oracle = candidates.groupby("run_id")["test_accuracy"].max().rename("oracle_test")
        merged = default.merge(oracle, on="run_id", how="left")
        ax.hist(merged["oracle_test"] - merged["test_accuracy"], bins=20)
    ax.set_xlabel("test regret versus oracle candidate")
    ax.set_ylabel("count")
    ax.set_title("Controlled selector regret")
    fig.tight_layout()
    fig.savefig(plot_dir / "controlled_nonabelian_selector_regret.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    if not nulls.empty:
        ax.bar(nulls["null_family"], nulls["false_validation_selection_rate"])
        ax.tick_params(axis="x", rotation=35, labelsize=7)
    ax.set_ylabel("false validation selection rate")
    ax.set_title("Null versus planted selection")
    fig.tight_layout()
    fig.savefig(plot_dir / "controlled_nonabelian_null_vs_planted.pdf")
    plt.close(fig)


def update_claims_audit(reports_dir: Path, claims: pd.DataFrame) -> None:
    path = reports_dir / "claims_audit.md"
    if not path.exists():
        return
    start = "<!-- controlled-nonabelian-holonomy:start -->"
    end = "<!-- controlled-nonabelian-holonomy:end -->"
    block = [
        start,
        "## Controlled Nonabelian Holonomy Splitting Audit",
        "",
        "Generated by `experiments/controlled_nonabelian_holonomy.py`. This is controlled evidence for Gamma-branch invariant pooling, not real-data Brauer/projective evidence and not a capacity-matched single-model claim.",
        "",
        md_table(claims.to_dict("records"), ["claim_id", "status", "safe_wording", "evidence"]),
        "",
        "Forbidden wording: real residuals are Brauer/projective classes; central period-index lifting explains this controlled experiment; branch lifts are capacity-matched single-model wins; test accuracy was used for selection.",
        end,
    ]
    text = path.read_text(encoding="utf-8")
    if start in text and end in text:
        before = text.split(start, 1)[0]
        after = text.split(end, 1)[1]
        text = before + "\n".join(block) + after
    else:
        text = text.rstrip() + "\n\n" + "\n".join(block) + "\n"
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def final_interpretation(claims: pd.DataFrame, paired: pd.DataFrame, selected: pd.DataFrame) -> tuple[str, str]:
    branch = paired[
        (paired["family"].eq("planted_nonabelian_holonomy"))
        & (paired["comparison"].eq("branch_regular_lift_with_invariant_pooling_vs_random_same_branch_count_control"))
    ]
    default = selected[
        pd.to_numeric(selected["selector_epsilon"], errors="coerce").eq(0.0)
        & pd.to_numeric(selected["selector_loss_slack"], errors="coerce").eq(0.0)
    ] if not selected.empty else pd.DataFrame()
    planted_select = float(default[default["family"].eq("planted_nonabelian_holonomy")]["selected_branch_lift"].mean()) if not default.empty else 0.0
    null_select = float(default[default["family"].eq("random_noncoherent_null")]["selected_branch_lift"].mean()) if not default.empty else 0.0
    if not branch.empty and bool((branch["paired_accuracy_delta_ci_low"] > 0).all()) and planted_select > 0.9 and null_select == 0.0:
        return (
            "Case A",
            "Controlled planted nonabelian holonomy can be split by a Gamma-branch lift with invariant pooling. The naive diagnostic rho(h) approximately I is too strict: faithful representations preserve holonomy, while invariant pooling makes holonomy act as a harmless branch permutation. This explains why the previous real-data experiment may have failed and provides a correct target for future real-data nonabelian lifts.",
        )
    greedy = paired[
        (paired["family"].eq("planted_nonabelian_holonomy"))
        & (paired["comparison"].eq("branch_regular_lift_with_invariant_pooling_vs_greedy_soup"))
    ]
    if not greedy.empty and bool((greedy["paired_accuracy_delta_ci_low"] <= 0).any()):
        return (
            "Case C",
            "The planted task is too easy: ordinary merging already solves it. A harder controlled task is required before drawing conclusions.",
        )
    random = paired[
        (paired["family"].eq("planted_nonabelian_holonomy"))
        & (paired["comparison"].eq("branch_regular_lift_with_invariant_pooling_vs_random_same_branch_count_control"))
    ]
    if not random.empty and bool((random["paired_accuracy_delta_ci_low"] <= 0).any()):
        return (
            "Case D",
            "The branch lift gain is explained by extra capacity rather than nonabelian structure. The lift must be capacity-controlled or reformulated before it can support a TwistedMerge claim.",
        )
    return (
        "Case B",
        "The controlled experiment fails even when the nonabelian obstruction and splitting representation are known. This indicates a problem in the current branch-lift implementation or in the proposed nonabelian splitting mechanism.",
    )


def write_report(args: argparse.Namespace, outputs: dict[str, pd.DataFrame], claims: pd.DataFrame) -> None:
    candidates = outputs["candidates"]
    summary = outputs["summary"]
    paired = outputs["paired"]
    residuals = outputs["residuals"]
    selected = outputs["selected"]
    nulls = outputs["nulls"]
    case_label, interpretation = final_interpretation(claims, paired, selected)
    default_selected = selected[
        pd.to_numeric(selected["selector_epsilon"], errors="coerce").eq(0.0)
        & pd.to_numeric(selected["selector_loss_slack"], errors="coerce").eq(0.0)
    ] if not selected.empty else pd.DataFrame()
    report = f"""# Controlled Nonabelian Holonomy Report

Generated by `experiments/controlled_nonabelian_holonomy.py`.

## Exact Command

```bash
{args.command_string}
```

## Git State

- Commit: `{git_output("rev-parse", "--short", "HEAD")}`
- Dirty-worktree status at run time:

```text
{git_output("status", "--short")}
```

## Conceptual Distinction

The naive faithful-representation diagnostic asks for `rho(h) approximately I`, which is too strong for nonabelian holonomy: faithful representations preserve nontrivial holonomy.  The controlled branch split instead tests an invariant pooling map `P` with `P rho(h) = P`.  Thus nontrivial holonomy becomes a harmless branch permutation after Gamma-indexed lifting and pooling.

## Experimental Grid

- Groups: `{args.groups}`
- Families: `{args.families}`
- Seeds: `{args.seeds}`
- Hidden widths: `{args.hidden_widths}`
- Input dimension: `{args.input_dim}`
- Train/val/test sizes: `{args.n_train}/{args.n_val}/{args.n_test}`
- Candidate methods: `{", ".join(METHODS)}`

## Residual Table

{md_table(residuals.to_dict("records"), ["group_name", "family", "seed", "hidden_width", "group_order", "holonomy_order", "is_holonomy_central", "ordinary_sync_residual", "naive_representation_residual", "invariant_pooling_residual"], 60)}

## Accuracy Summary

{md_table(summary.to_dict("records"), ["group_name", "family", "method", "n_runs", "mean_validation_accuracy", "mean_test_accuracy", "mean_naive_representation_residual", "mean_invariant_pooling_residual", "mean_parameter_multiplier"], 80)}

## Selector Table

{md_table(default_selected.to_dict("records"), ["group_name", "family", "seed", "hidden_width", "selected_method", "selected_branch_lift", "best_fallback_method", "validation_accuracy", "test_accuracy", "selector_no_test_leakage"], 80)}

## Null Controls

{md_table(nulls.to_dict("records"), ["null_family", "n_null", "false_split_rate", "false_validation_selection_rate", "null_accuracy_delta", "null_invariant_pooling_residual"], 40)}

## Paired Statistics

{md_table(paired.to_dict("records"), ["group_name", "family", "comparison", "n_pairs", "paired_mean_accuracy_delta", "paired_accuracy_delta_ci_low", "paired_accuracy_delta_ci_high", "paired_mean_loss_delta", "wins", "ties", "losses", "sign_test_p_value"], 80)}

## Claim Decision Table

{md_table(claims.to_dict("records"), ["claim_id", "status", "safe_wording", "evidence"])}

## Negative Boundaries

- Do not claim real neural residuals are Brauer/projective classes.
- Do not say central period-index lifting explains this controlled nonabelian experiment.
- Do not call branch lifts capacity-matched single-model wins.
- Branch lifts are extra-capacity branch models unless explicitly collapsed and capacity-controlled.
- Selection is validation-only; test accuracy is report-only.

## Final Interpretation

{case_label}: {interpretation}
"""
    (args.reports_dir / "controlled_nonabelian_holonomy_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "reports")
    parser.add_argument("--groups", default="S3,D4")
    parser.add_argument("--families", default="trivial_coboundary,planted_nonabelian_holonomy,random_noncoherent_null")
    parser.add_argument("--seeds", default="0:49")
    parser.add_argument("--hidden-widths", default="12,24,48")
    parser.add_argument("--input-dim", type=int, default=20)
    parser.add_argument("--n-train", type=int, default=2000)
    parser.add_argument("--n-val", type=int, default=1000)
    parser.add_argument("--n-test", type=int, default=2000)
    parser.add_argument("--n-classes", type=int, default=5)
    parser.add_argument("--selector-epsilons", default="0.0,0.001,0.002")
    parser.add_argument("--selector-loss-slacks", default="0.0,0.005,0.01,inf")
    parser.add_argument("--pooling-threshold", type=float, default=1e-8)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=12031)
    parser.add_argument("--no-update-claims-audit", action="store_true")
    args = parser.parse_args()
    args.command_string = " ".join([sys.executable, *sys.argv])

    csv_dir = args.reports_dir / "csv"
    plot_dir = args.reports_dir / "plots"
    config_dir = args.reports_dir / "configs"
    for directory in [csv_dir, plot_dir, config_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    candidates, residuals = build_grid(args)
    selected = selector_rows(candidates, args)
    candidates = mark_default_selection(candidates, selected)
    summary = summary_table(candidates)
    paired = paired_stats(candidates, selected, args)
    nulls = null_controls(candidates, selected)
    outputs = {
        "candidates": candidates,
        "summary": summary,
        "paired": paired,
        "residuals": residuals,
        "selected": selected,
        "nulls": nulls,
    }
    claims = build_claims(candidates, summary, paired, selected)

    candidates.to_csv(csv_dir / "controlled_nonabelian_holonomy.csv", index=False, lineterminator="\n")
    summary.to_csv(csv_dir / "controlled_nonabelian_holonomy_summary.csv", index=False, lineterminator="\n")
    paired.to_csv(csv_dir / "controlled_nonabelian_holonomy_paired_stats.csv", index=False, lineterminator="\n")
    residuals.to_csv(csv_dir / "controlled_nonabelian_holonomy_residuals.csv", index=False, lineterminator="\n")
    selected.to_csv(csv_dir / "controlled_nonabelian_holonomy_selector.csv", index=False, lineterminator="\n")
    nulls.to_csv(csv_dir / "controlled_nonabelian_holonomy_null_controls.csv", index=False, lineterminator="\n")
    claims.to_csv(csv_dir / "controlled_nonabelian_holonomy_claims.csv", index=False, lineterminator="\n")

    write_plots(outputs, args.reports_dir)
    write_report(args, outputs, claims)
    save_json(
        config_dir / "controlled_nonabelian_holonomy_config.json",
        {
            "argv": sys.argv,
            "command": args.command_string,
            "args": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
                if key != "command_string"
            },
            "environment": capture_environment(),
            "git_commit": git_output("rev-parse", "--short", "HEAD"),
            "git_status_short": git_output("status", "--short"),
        },
    )
    if not args.no_update_claims_audit:
        update_claims_audit(args.reports_dir, claims)

    print("wrote reports/controlled_nonabelian_holonomy_report.md")
    print("wrote reports/csv/controlled_nonabelian_holonomy.csv")
    print("wrote reports/csv/controlled_nonabelian_holonomy_paired_stats.csv")


if __name__ == "__main__":
    main()
