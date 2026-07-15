#!/usr/bin/env python
"""Rerun obstruction predictor-target diagnostics from completed fixed-setting CSVs."""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import sys
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

TARGETS = (
    "weight_average_degradation_vs_best_single",
    "git_rebasin_degradation_vs_best_single",
    "c2m3_degradation_vs_best_single",
    "c2m3_delta_vs_git_rebasin",
    "c2m3_delta_vs_weight_average",
    "rank_lift_delta_vs_weight_average",
    "rank_lift_delta_vs_c2m3",
    "greedy_soup_delta_vs_weight_average",
    "linear_mode_connectivity_barrier",
    "c2m3_barrier_delta_vs_git_rebasin",
    "c2m3_barrier_delta_vs_weight_average",
    "monomial_barrier_delta_vs_c2m3",
    "weight_average_val_max_loss_barrier",
    "git_rebasin_val_max_loss_barrier",
    "c2m3_val_max_loss_barrier",
    "monomial_val_max_loss_barrier",
)

BARRIER_TARGETS = (
    "linear_mode_connectivity_barrier",
    "c2m3_barrier_delta_vs_git_rebasin",
    "c2m3_barrier_delta_vs_weight_average",
    "monomial_barrier_delta_vs_c2m3",
    "weight_average_val_max_loss_barrier",
    "git_rebasin_val_max_loss_barrier",
    "c2m3_val_max_loss_barrier",
    "monomial_val_max_loss_barrier",
)

BARRIER_METHOD_TARGETS = (
    ("weight_average", "weight_average_val_max_loss_barrier"),
    ("git_rebasin_pairwise_ref0", "git_rebasin_val_max_loss_barrier"),
    ("c2m3_synchronized", "c2m3_val_max_loss_barrier"),
    ("monomial_shrinkage", "monomial_val_max_loss_barrier"),
)

PREDICTORS = (
    "mean_cycle_score",
    "max_cycle_score",
    "nonidentity_triangle_fraction",
    "sync_disagreement",
    "pairwise_alignment_residual_mean",
    "activation_assignment_similarity_mean",
    "combined_obstruction_score",
    "monomial_defect_score",
    "mean_triangle_defect_rate",
)

GROUP_COLS = (
    "dataset",
    "architecture",
    "n_models",
    "width",
    "domain_shift",
    "matching",
    "alignment_source",
    "alignment_noise_fraction",
)

NUMERIC_CONTROLS = (
    "mean_individual_accuracy",
    "pairwise_alignment_residual_mean",
)

OUTPUT_STATS = "obstruction_predictor_target_stats.csv"
OUTPUT_REGRESSIONS = "real_obstruction_predictor_regressions.csv"
OUTPUT_BARRIER_STATS = "obstruction_barrier_predictor_stats.csv"
OUTPUT_REPORT = "obstruction_predictor_target_report.md"
OUTPUT_PLOT = "obstruction_predictor_target_grid.pdf"
OUTPUT_BARRIER_REPORT = "obstruction_barrier_predictor_report.md"
OUTPUT_BARRIER_PLOT = "obstruction_barrier_predictor_grid.pdf"
BARRIER_STABILITY_REPORT = "alignment_barrier_stability_report.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-csv",
        type=Path,
        default=ROOT / "reports" / "csv" / "fixed_setting_verification_runs.csv",
        help="Completed fixed-setting run CSV to analyze.",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=ROOT / "reports",
        help="Report root containing csv/ and plots/ directories.",
    )
    parser.add_argument(
        "--barriers-csv",
        type=Path,
        default=ROOT / "reports" / "csv" / "alignment_barrier_targets.csv",
        help="Optional alignment-barrier target CSV to merge into the fixed-setting run rows.",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument(
        "--max-plot-label-length",
        type=int,
        default=36,
        help="Soft wrap length for plot labels.",
    )
    return parser.parse_args()


def safe_mean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float(np.mean(values))


def fixed_setting_id(row: pd.Series | dict) -> str:
    return (
        f"{row['dataset']}_{row['architecture']}_N{int(row['n_models'])}_"
        f"W{int(row['width'])}_{row['domain_shift']}_{row['matching']}"
    )


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


def beta_from_design(design: np.ndarray, y: np.ndarray) -> float:
    try:
        beta = np.linalg.lstsq(design, y, rcond=None)[0][1]
    except np.linalg.LinAlgError:
        return float("nan")
    return float(beta)


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
    if design is None or y_f is None or status != "ok":
        return float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    sample_count = max(0, int(samples))
    if sample_count == 0:
        return float("nan"), float("nan")
    idx = rng.integers(0, n, size=(sample_count, n))
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


def target_family(target: str) -> str:
    if target == "weight_average_degradation_vs_best_single":
        return "raw_accuracy"
    if target in BARRIER_TARGETS:
        return "alignment_conditioned_barrier"
    if target == "greedy_soup_delta_vs_weight_average":
        return "validation_soup_accuracy"
    return "alignment_conditioned_accuracy"


def merge_barrier_targets(runs: pd.DataFrame, barriers_csv: Path) -> pd.DataFrame:
    if not barriers_csv.exists():
        return runs
    barriers = pd.read_csv(barriers_csv)

    # The checkpoint barrier run is computed only for observed alignments.  Add
    # the source columns before merging so injected-noise rows remain controls
    # rather than silently inheriting observed-path targets.
    if "alignment_source" not in barriers.columns:
        barriers["alignment_source"] = "observed"
    if "alignment_noise_fraction" not in barriers.columns:
        barriers["alignment_noise_fraction"] = 0.0

    fixed_keys = [
        "dataset",
        "architecture",
        "n_models",
        "width",
        "domain_shift",
        "matching",
        "seed",
        "alignment_source",
    ]
    key_cols = [col for col in fixed_keys if col in barriers.columns and col in runs.columns]
    if len(key_cols) < len(fixed_keys):
        return runs

    if "status" in barriers.columns:
        barriers = barriers[barriers["status"].astype(str) == "ok"].copy()

    pieces: list[pd.DataFrame] = []
    aggregate_cols = [col for col in BARRIER_TARGETS if col in barriers.columns]
    if aggregate_cols:
        pieces.append(barriers[key_cols + aggregate_cols].drop_duplicates(subset=key_cols, keep="first"))

    if "method" in barriers.columns and "val_max_loss_barrier" in barriers.columns:
        pivot = barriers.pivot_table(
            index=key_cols,
            columns="method",
            values="val_max_loss_barrier",
            aggfunc="first",
        )
        rename = {method: target for method, target in BARRIER_METHOD_TARGETS if method in pivot.columns}
        method_targets = pivot.rename(columns=rename).reset_index()
        keep = key_cols + [target for _method, target in BARRIER_METHOD_TARGETS if target in method_targets.columns]
        if len(keep) > len(key_cols):
            pieces.append(method_targets[keep])

    if not pieces:
        return runs
    barrier_rows = pieces[0]
    for piece in pieces[1:]:
        barrier_rows = barrier_rows.merge(piece, on=key_cols, how="outer")

    value_cols = [col for col in barrier_rows.columns if col not in key_cols]
    out = runs.copy()
    existing = [col for col in value_cols if col in out.columns]
    if existing:
        out = out.drop(columns=existing)
    return out.merge(barrier_rows, on=key_cols, how="left")


def load_config_command(reports_dir: Path) -> str:
    config_path = reports_dir / "configs" / "fixed_setting_verification_config.json"
    if not config_path.exists():
        return "not recorded"
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return "not readable"
    commands = payload.get("commands", [])
    if not commands:
        return "not recorded"
    return "\n".join(str(command) for command in commands)


def input_audit(reports_dir: Path) -> pd.DataFrame:
    csv_dir = reports_dir / "csv"
    names = [
        "fixed_setting_verification_runs.csv",
        "fixed_setting_verification_stats.csv",
        "fixed_setting_triangle_defects.csv",
        "fixed_setting_individual_models.csv",
        "real_obstruction_degradation.csv",
        "real_obstruction_summary.csv",
        "real_obstruction_paired_deltas.csv",
        "real_obstruction_predictor_regressions.csv",
        "alignment_barrier_targets.csv",
        "alignment_barrier_target_stats.csv",
        "obstruction_barrier_predictor_stats.csv",
    ]
    rows = []
    for name in names:
        path = csv_dir / name
        rows.append(
            {
                "input": f"reports/csv/{name}",
                "exists": path.exists(),
                "rows": count_csv_rows(path) if path.exists() else 0,
                "size_mb": round(path.stat().st_size / (1024 * 1024), 3) if path.exists() else 0.0,
            }
        )
    return pd.DataFrame(rows)


def barrier_loader_metadata(barriers_csv: Path) -> dict:
    if not barriers_csv.exists():
        return {
            "barrier_eval_scale": "barrier_csv_missing",
            "barrier_max_eval_batches": "",
            "barrier_citation_preferred": False,
        }
    try:
        barriers = pd.read_csv(barriers_csv, usecols=lambda col: col in {"max_eval_batches", "status"})
    except Exception:
        return {
            "barrier_eval_scale": "barrier_csv_unreadable",
            "barrier_max_eval_batches": "",
            "barrier_citation_preferred": False,
        }
    if "status" in barriers.columns:
        barriers = barriers[barriers["status"].astype(str) == "ok"].copy()
    if "max_eval_batches" not in barriers.columns or barriers.empty:
        return {
            "barrier_eval_scale": "unknown_eval_scale",
            "barrier_max_eval_batches": "",
            "barrier_citation_preferred": False,
        }
    caps = sorted({int(value) for value in pd.to_numeric(barriers["max_eval_batches"], errors="coerce").dropna()})
    if caps == [0]:
        scale = "full_loader"
        citation = True
    else:
        scale = "two_batch_or_capped_loader"
        citation = False
    return {
        "barrier_eval_scale": scale,
        "barrier_max_eval_batches": ",".join(str(cap) for cap in caps),
        "barrier_citation_preferred": citation,
    }


def read_markdown_table_after_heading(path: Path, heading: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    lines = path.read_text(encoding="utf-8").splitlines()
    start = None
    for idx, line in enumerate(lines):
        if line.strip() == heading:
            start = idx + 1
            break
    if start is None:
        return pd.DataFrame()
    table_lines = []
    for line in lines[start:]:
        stripped = line.strip()
        if stripped.startswith("## ") and table_lines:
            break
        if stripped.startswith("|"):
            table_lines.append(stripped)
        elif table_lines and stripped:
            break
    if len(table_lines) < 3:
        return pd.DataFrame()
    csv_lines = []
    for line in table_lines:
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if cells and all(set(cell) <= {"-", ":", " "} for cell in cells):
            continue
        csv_lines.append(",".join('"' + cell.replace('"', '""') + '"' for cell in cells))
    if len(csv_lines) < 2:
        return pd.DataFrame()
    return pd.read_csv(StringIO("\n".join(csv_lines)))


def load_barrier_claim_stability(reports_dir: Path) -> pd.DataFrame:
    stability = read_markdown_table_after_heading(
        reports_dir / BARRIER_STABILITY_REPORT,
        "## Supported Predictor-Target Claim Stability",
    )
    if stability.empty:
        return stability
    numeric_cols = [
        "n_models",
        "width",
        "previous_beta",
        "previous_ci_low",
        "previous_ci_high",
        "current_beta",
        "current_ci_low",
        "current_ci_high",
        "n_unique_seeds",
    ]
    for col in numeric_cols:
        if col in stability.columns:
            stability[col] = pd.to_numeric(stability[col], errors="coerce")
    if "stable_positive_gate" in stability.columns:
        stability["stable_positive_gate"] = stability["stable_positive_gate"].astype(str).str.lower().isin(
            {"true", "1", "yes"}
        )
    else:
        stability["stable_positive_gate"] = False
    return stability


def count_csv_rows(path: Path) -> int:
    with path.open("rb") as handle:
        lines = sum(1 for _ in handle)
    return max(lines - 1, 0)


def compute_rows(runs: pd.DataFrame, bootstrap_samples: int) -> pd.DataFrame:
    if "method" not in runs:
        raise ValueError("runs CSV must contain a method column")
    base = runs[runs["method"].astype(str) == "weight_average"].copy()
    if base.empty:
        raise ValueError("runs CSV contains no weight_average rows")

    rows: list[dict] = []
    for key, group in base.groupby(list(GROUP_COLS), dropna=False, sort=True):
        meta = dict(zip(GROUP_COLS, key))
        setting_id = fixed_setting_id(meta)
        n_rows = int(len(group))
        n_unique_seeds = int(group["seed"].nunique()) if "seed" in group else 0
        controls = [pd.to_numeric(group[col], errors="coerce").to_numpy() for col in NUMERIC_CONTROLS if col in group]
        is_observed = str(meta["alignment_source"]) == "observed" and float(meta["alignment_noise_fraction"]) == 0.0
        primary_evidence = bool(is_observed and n_unique_seeds >= 20)

        for target in TARGETS:
            target_missing = target not in group
            y = (
                np.full(n_rows, np.nan, dtype=float)
                if target_missing
                else pd.to_numeric(group[target], errors="coerce").to_numpy()
            )
            for predictor in PREDICTORS:
                predictor_missing = predictor not in group
                x = (
                    np.full(n_rows, np.nan, dtype=float)
                    if predictor_missing
                    else pd.to_numeric(group[predictor], errors="coerce").to_numpy()
                )
                beta, n_finite, fit_status = regression_beta(x, y, controls)
                ci_low, ci_high = (
                    (float("nan"), float("nan"))
                    if target_missing or predictor_missing or fit_status != "ok"
                    else bootstrap_beta_ci(
                        x,
                        y,
                        controls,
                        bootstrap_samples,
                        seed=9109 + len(rows) * 7919,
                    )
                )
                if target_missing:
                    claim_status = "target_not_run"
                    target_status = "not_run"
                elif predictor_missing or fit_status == "predictor_missing_or_constant":
                    claim_status = "unsupported_predictor_missing_or_constant"
                    target_status = "available"
                elif fit_status != "ok":
                    claim_status = f"unsupported_{fit_status}"
                    target_status = "available"
                elif not is_observed:
                    claim_status = "negative_control_not_primary_evidence"
                    target_status = "available"
                elif n_unique_seeds < 20:
                    claim_status = "unsupported_descriptive_n_below_20"
                    target_status = "available"
                elif math.isfinite(ci_low) and ci_low > 0.0:
                    claim_status = "positive_ci_gate_pending_stability"
                    target_status = "available"
                elif math.isfinite(ci_high) and ci_high < 0.0:
                    claim_status = "negative_association"
                    target_status = "available"
                else:
                    claim_status = "unsupported_ci_crosses_zero_or_unstable"
                    target_status = "available"

                rows.append(
                    {
                        **meta,
                        "fixed_setting_id": setting_id,
                        "target": target,
                        "outcome": target,
                        "outcome_family": target_family(target),
                        "predictor": predictor,
                        "regression_formula": (
                            f"{target} ~ {predictor} + mean_individual_accuracy "
                            "+ pairwise_alignment_residual_mean, stratified by fixed setting"
                        ),
                        "controls_used": (
                            "fixed_setting_strata(dataset,architecture,n_models,width,domain_shift,matching); "
                            "mean_individual_accuracy; pairwise_alignment_residual_mean"
                        ),
                        "n_rows": n_rows,
                        "n_unique_seeds": n_unique_seeds,
                        "n_finite": n_finite,
                        "predictor_beta": beta,
                        "predictor_beta_ci_low": ci_low,
                        "predictor_beta_ci_high": ci_high,
                        "mean_outcome": safe_mean(y),
                        "mean_predictor": safe_mean(x),
                        "mean_individual_accuracy": safe_mean(
                            pd.to_numeric(group.get("mean_individual_accuracy"), errors="coerce").to_numpy()
                        )
                        if "mean_individual_accuracy" in group
                        else float("nan"),
                        "mean_pairwise_alignment_residual": safe_mean(
                            pd.to_numeric(group.get("pairwise_alignment_residual_mean"), errors="coerce").to_numpy()
                        )
                        if "pairwise_alignment_residual_mean" in group
                        else float("nan"),
                        "target_status": target_status,
                        "fit_status": fit_status,
                        "claim_status": claim_status,
                        "support_scope": "not_supported",
                        "claim_supported": False,
                        "primary_evidence": primary_evidence,
                    }
                )
    result = pd.DataFrame(rows)
    return apply_support_stability(result)


def barrier_run_level_from_csv(barriers_csv: Path) -> pd.DataFrame:
    if not barriers_csv.exists():
        return pd.DataFrame()
    barriers = pd.read_csv(barriers_csv)
    if barriers.empty:
        return pd.DataFrame()
    if "status" in barriers.columns:
        barriers = barriers[barriers["status"].astype(str) == "ok"].copy()
    if barriers.empty:
        return pd.DataFrame()
    if "alignment_source" not in barriers.columns:
        barriers["alignment_source"] = "observed"
    if "alignment_noise_fraction" not in barriers.columns:
        barriers["alignment_noise_fraction"] = 0.0
    key_cols = [
        "setting_id",
        "run_id",
        "dataset",
        "architecture",
        "n_models",
        "width",
        "domain_shift",
        "matching",
        "seed",
        "alignment_source",
        "alignment_noise_fraction",
    ]
    if not all(col in barriers.columns for col in key_cols):
        return pd.DataFrame()
    predictors = [col for col in PREDICTORS if col in barriers.columns]
    controls = [col for col in NUMERIC_CONTROLS if col in barriers.columns]
    pieces: list[pd.DataFrame] = []
    aggregate_targets = [col for col in BARRIER_TARGETS if col in barriers.columns]
    keep_base = list(dict.fromkeys(key_cols + predictors + controls))
    if aggregate_targets:
        pieces.append(
            barriers[keep_base + aggregate_targets]
            .drop_duplicates(subset=key_cols, keep="first")
            .copy()
        )
    if "method" in barriers.columns and "val_max_loss_barrier" in barriers.columns:
        pivot = barriers.pivot_table(
            index=key_cols,
            columns="method",
            values="val_max_loss_barrier",
            aggfunc="first",
        )
        rename = {method: target for method, target in BARRIER_METHOD_TARGETS if method in pivot.columns}
        method_targets = pivot.rename(columns=rename).reset_index()
        method_target_cols = [target for _method, target in BARRIER_METHOD_TARGETS if target in method_targets.columns]
        if method_target_cols:
            base = barriers[keep_base].drop_duplicates(subset=key_cols, keep="first")
            pieces.append(base.merge(method_targets[key_cols + method_target_cols], on=key_cols, how="left"))
    if not pieces:
        return pd.DataFrame()
    out = pieces[0]
    for piece in pieces[1:]:
        out = out.merge(piece, on=keep_base, how="outer")
    return out


def compute_barrier_rows_from_barriers(barriers_csv: Path, bootstrap_samples: int) -> pd.DataFrame:
    base = barrier_run_level_from_csv(barriers_csv)
    if base.empty:
        return pd.DataFrame()
    rows: list[dict] = []
    for key, group in base.groupby(list(GROUP_COLS), dropna=False, sort=True):
        meta = dict(zip(GROUP_COLS, key))
        setting_id = fixed_setting_id(meta)
        n_rows = int(len(group))
        n_unique_seeds = int(group["seed"].nunique()) if "seed" in group else 0
        controls = [pd.to_numeric(group[col], errors="coerce").to_numpy() for col in NUMERIC_CONTROLS if col in group]
        is_observed = str(meta["alignment_source"]) == "observed" and float(meta["alignment_noise_fraction"]) == 0.0
        primary_evidence = bool(is_observed and n_unique_seeds >= 20)
        for target in BARRIER_TARGETS:
            target_missing = target not in group
            y = (
                np.full(n_rows, np.nan, dtype=float)
                if target_missing
                else pd.to_numeric(group[target], errors="coerce").to_numpy()
            )
            for predictor in PREDICTORS:
                predictor_missing = predictor not in group
                x = (
                    np.full(n_rows, np.nan, dtype=float)
                    if predictor_missing
                    else pd.to_numeric(group[predictor], errors="coerce").to_numpy()
                )
                beta, n_finite, fit_status = regression_beta(x, y, controls)
                ci_low, ci_high = (
                    (float("nan"), float("nan"))
                    if target_missing or predictor_missing or fit_status != "ok"
                    else bootstrap_beta_ci(
                        x,
                        y,
                        controls,
                        bootstrap_samples,
                        seed=50111 + len(rows) * 6151,
                    )
                )
                if target_missing:
                    claim_status = "target_not_run"
                    target_status = "not_run"
                elif predictor_missing or fit_status == "predictor_missing_or_constant":
                    claim_status = "unsupported_predictor_missing_or_constant"
                    target_status = "available"
                elif fit_status != "ok":
                    claim_status = f"unsupported_{fit_status}"
                    target_status = "available"
                elif not is_observed:
                    claim_status = "negative_control_not_primary_evidence"
                    target_status = "available"
                elif n_unique_seeds < 20:
                    claim_status = "unsupported_descriptive_n_below_20"
                    target_status = "available"
                elif math.isfinite(ci_low) and ci_low > 0.0:
                    claim_status = "positive_ci_gate_pending_stability"
                    target_status = "available"
                elif math.isfinite(ci_high) and ci_high < 0.0:
                    claim_status = "negative_association"
                    target_status = "available"
                else:
                    claim_status = "unsupported_ci_crosses_zero_or_unstable"
                    target_status = "available"
                rows.append(
                    {
                        **meta,
                        "fixed_setting_id": setting_id,
                        "target": target,
                        "outcome": target,
                        "outcome_family": target_family(target),
                        "predictor": predictor,
                        "regression_formula": (
                            f"{target} ~ {predictor} + available controls, "
                            "stratified by full-loader barrier fixed setting"
                        ),
                        "controls_used": (
                            "full_loader_barrier_strata(dataset,architecture,n_models,width,domain_shift,matching); "
                            + "; ".join([col for col in NUMERIC_CONTROLS if col in group])
                        ),
                        "n_rows": n_rows,
                        "n_unique_seeds": n_unique_seeds,
                        "n_finite": n_finite,
                        "predictor_beta": beta,
                        "predictor_beta_ci_low": ci_low,
                        "predictor_beta_ci_high": ci_high,
                        "mean_outcome": safe_mean(y),
                        "mean_predictor": safe_mean(x),
                        "mean_individual_accuracy": safe_mean(
                            pd.to_numeric(group.get("mean_individual_accuracy"), errors="coerce").to_numpy()
                        )
                        if "mean_individual_accuracy" in group
                        else float("nan"),
                        "mean_pairwise_alignment_residual": safe_mean(
                            pd.to_numeric(group.get("pairwise_alignment_residual_mean"), errors="coerce").to_numpy()
                        )
                        if "pairwise_alignment_residual_mean" in group
                        else float("nan"),
                        "target_status": target_status,
                        "fit_status": fit_status,
                        "claim_status": claim_status,
                        "support_scope": "not_supported",
                        "claim_supported": False,
                        "primary_evidence": primary_evidence,
                        "barrier_rows_source": "full_loader_barrier_csv_direct",
                    }
                )
    return apply_support_stability(pd.DataFrame(rows))


def apply_support_stability(stats: pd.DataFrame) -> pd.DataFrame:
    stats = stats.copy()
    observed = stats[
        (stats["alignment_source"].astype(str) == "observed")
        & (pd.to_numeric(stats["alignment_noise_fraction"], errors="coerce") == 0.0)
    ].copy()
    positive_sign_counts = (
        observed[np.isfinite(pd.to_numeric(observed["predictor_beta"], errors="coerce"))]
        .assign(_positive=pd.to_numeric(observed["predictor_beta"], errors="coerce") > 0.0)
        .groupby(["target", "predictor"], dropna=False)["_positive"]
        .sum()
        .to_dict()
    )
    for idx, row in stats.iterrows():
        if row["claim_status"] != "positive_ci_gate_pending_stability":
            continue
        key = (row["target"], row["predictor"])
        positive_count = int(positive_sign_counts.get(key, 0))
        if positive_count >= 2:
            stats.at[idx, "claim_status"] = "supported_positive_predictor_coefficient_replicated"
            stats.at[idx, "support_scope"] = "replicated_positive_sign"
        else:
            stats.at[idx, "claim_status"] = "supported_setting_specific_positive_predictor_coefficient"
            stats.at[idx, "support_scope"] = "setting_specific"
        stats.at[idx, "claim_supported"] = True
    return stats


def apply_full_loader_barrier_stability(
    stats: pd.DataFrame,
    reports_dir: Path,
    barriers_csv: Path,
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    stats = stats.copy()
    metadata = barrier_loader_metadata(barriers_csv)
    stability = load_barrier_claim_stability(reports_dir)
    barrier_mask = stats["target"].astype(str).isin(BARRIER_TARGETS)
    stats["barrier_eval_scale"] = ""
    stats["barrier_max_eval_batches"] = ""
    stats["barrier_citation_preferred"] = False
    stats["previous_two_batch_supported"] = False
    stats["previous_two_batch_beta"] = float("nan")
    stats["previous_two_batch_ci_low"] = float("nan")
    stats["previous_two_batch_ci_high"] = float("nan")
    stats["full_loader_beta_in_stability_report"] = float("nan")
    stats["full_loader_ci_low_in_stability_report"] = float("nan")
    stats["full_loader_ci_high_in_stability_report"] = float("nan")
    stats["barrier_stability_decision"] = ""
    stats.loc[barrier_mask, "barrier_eval_scale"] = metadata["barrier_eval_scale"]
    stats.loc[barrier_mask, "barrier_max_eval_batches"] = metadata["barrier_max_eval_batches"]
    stats.loc[barrier_mask, "barrier_citation_preferred"] = bool(metadata["barrier_citation_preferred"])

    if stability.empty:
        stats.loc[barrier_mask, "barrier_stability_decision"] = (
            "full_loader_no_two_batch_claim_comparison"
            if metadata["barrier_citation_preferred"]
            else "not_full_loader_citation_preferred"
        )
        return stats, metadata, stability

    key_cols = [
        col
        for col in ["dataset", "architecture", "n_models", "width", "domain_shift", "matching", "target", "predictor"]
        if col in stability.columns and col in stats.columns
    ]
    for claim in stability.to_dict("records"):
        mask = barrier_mask.copy()
        for col in key_cols:
            if col in {"n_models", "width"}:
                mask &= pd.to_numeric(stats[col], errors="coerce") == float(claim[col])
            else:
                mask &= stats[col].astype(str) == str(claim[col])
        if not mask.any():
            continue
        stable_gate = bool(claim.get("stable_positive_gate", False))
        current_supported = stats.loc[mask, "claim_supported"].astype(bool)
        stats.loc[mask, "previous_two_batch_supported"] = True
        stats.loc[mask, "previous_two_batch_beta"] = claim.get("previous_beta", float("nan"))
        stats.loc[mask, "previous_two_batch_ci_low"] = claim.get("previous_ci_low", float("nan"))
        stats.loc[mask, "previous_two_batch_ci_high"] = claim.get("previous_ci_high", float("nan"))
        stats.loc[mask, "full_loader_beta_in_stability_report"] = claim.get("current_beta", float("nan"))
        stats.loc[mask, "full_loader_ci_low_in_stability_report"] = claim.get("current_ci_low", float("nan"))
        stats.loc[mask, "full_loader_ci_high_in_stability_report"] = claim.get("current_ci_high", float("nan"))
        # For previously supported two-batch claims, prefer the benchmark series 20
        # full-loader stability recheck coefficients.  The direct barrier-CSV
        # fallback can lack controls that were available during the full-loader
        # barrier run, so it must not silently resurrect a rejected stability
        # claim.
        stats.loc[mask, "predictor_beta"] = claim.get("current_beta", float("nan"))
        stats.loc[mask, "predictor_beta_ci_low"] = claim.get("current_ci_low", float("nan"))
        stats.loc[mask, "predictor_beta_ci_high"] = claim.get("current_ci_high", float("nan"))
        if stable_gate and bool(metadata["barrier_citation_preferred"]):
            supported_mask = mask & current_supported
            stats.loc[supported_mask, "claim_status"] = "stable_full_loader_supported"
            stats.loc[supported_mask, "support_scope"] = "stable_full_loader_supported"
            stats.loc[supported_mask, "barrier_stability_decision"] = "stable_full_loader_supported"
            not_supported_mask = mask & ~current_supported
            stats.loc[not_supported_mask, "barrier_stability_decision"] = "full_loader_gate_stable_in_reference_but_current_row_not_supported"
        else:
            stats.loc[mask, "claim_supported"] = False
            stats.loc[mask, "claim_status"] = "not_stable_previous_two_batch_support_disappeared"
            stats.loc[mask, "support_scope"] = "not_stable_previous_two_batch"
            stats.loc[mask, "barrier_stability_decision"] = "not_stable_previous_two_batch_support_disappeared"

    no_prior_mask = barrier_mask & (stats["barrier_stability_decision"].astype(str) == "")
    stats.loc[no_prior_mask, "barrier_stability_decision"] = (
        "full_loader_citation_preferred_no_prior_supported_two_batch_claim"
        if metadata["barrier_citation_preferred"]
        else "not_full_loader_citation_preferred"
    )
    new_supported_mask = (
        no_prior_mask
        & stats["claim_supported"].astype(bool)
        & stats["barrier_citation_preferred"].astype(bool)
    )
    stats.loc[new_supported_mask, "claim_status"] = "full_loader_supported_no_prior_two_batch_claim"
    stats.loc[new_supported_mask, "support_scope"] = "full_loader_only"
    return stats, metadata, stability


def md_table(df: pd.DataFrame, cols: list[str], max_rows: int = 40) -> str:
    if df.empty:
        return "_None._"
    view = df.loc[:, [col for col in cols if col in df.columns]].head(max_rows).copy()
    integer_like = {
        "n_models",
        "width",
        "n_rows",
        "n_unique_seeds",
        "n_finite",
        "rows",
        "n_supported",
        "n_target_not_run",
    }
    for col in view.columns:
        if pd.api.types.is_bool_dtype(view[col]):
            view[col] = view[col].map(lambda value: "true" if bool(value) else "false")
        elif col in integer_like and pd.api.types.is_numeric_dtype(view[col]):
            view[col] = view[col].map(lambda value: "" if pd.isna(value) else str(int(round(float(value)))))
        elif pd.api.types.is_numeric_dtype(view[col]):
            view[col] = view[col].map(lambda value: "" if pd.isna(value) else f"{value:.4f}")
        else:
            view[col] = view[col].fillna("").astype(str)
    header = "| " + " | ".join(view.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(view.columns)) + " |"
    rows = ["| " + " | ".join(str(value) for value in row) + " |" for row in view.to_numpy()]
    suffix = ""
    if len(df) > max_rows:
        suffix = f"\n\n_Showing {max_rows} of {len(df)} rows._"
    return "\n".join([header, sep, *rows]) + suffix


def summarize_datasets(runs: pd.DataFrame) -> str:
    base = runs[runs["method"].astype(str) == "weight_average"].copy()
    summary = (
        base.groupby(["dataset", "architecture", "n_models", "width", "domain_shift", "alignment_source"], dropna=False)
        .agg(n_rows=("seed", "size"), n_unique_seeds=("seed", "nunique"))
        .reset_index()
        .sort_values(["dataset", "n_models", "domain_shift", "alignment_source"])
    )
    return md_table(
        summary,
        ["dataset", "architecture", "n_models", "width", "domain_shift", "alignment_source", "n_rows", "n_unique_seeds"],
        max_rows=40,
    )


def recommendation(stats: pd.DataFrame) -> str:
    observed_supported = stats[
        (stats["alignment_source"].astype(str) == "observed") & (stats["claim_supported"] == True)  # noqa: E712
    ]
    if observed_supported.empty:
        return (
            "No predictor-target claim passes the observed bootstrap gate. The paper should describe these diagnostics "
            "as negative/descriptive on the quality-gated real-network benchmark."
        )
    raw_supported = observed_supported[observed_supported["outcome_family"] == "raw_accuracy"]
    accuracy_supported = observed_supported[
        observed_supported["outcome_family"].isin(["alignment_conditioned_accuracy", "validation_soup_accuracy"])
    ]
    barrier_supported = observed_supported[observed_supported["outcome_family"] == "alignment_conditioned_barrier"]
    if raw_supported.empty and (not accuracy_supported.empty or not barrier_supported.empty):
        return (
            "Only alignment-conditioned targets pass the gate. Do not claim raw weight-average degradation prediction; "
            "state that selected obstruction statistics predict alignment-conditioned accuracy or barrier targets "
            "within this quality-gated fixed-setting run."
        )
    if not raw_supported.empty:
        return (
            "At least one raw target passes the gate, but wording should remain fixed-setting and predictor-specific; "
            "do not generalize beyond the quality-gated MNIST/Fashion-MNIST mlp2 benchmark."
        )
    return "Supported targets are secondary targets; keep the raw weight-average boundary explicit."


def target_support_sections(stats: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    observed = stats[stats["alignment_source"].astype(str) == "observed"].copy()
    supported = observed[observed["claim_supported"] == True].copy()  # noqa: E712
    raw = supported[supported["outcome_family"] == "raw_accuracy"].copy()
    conditioned = supported[
        supported["outcome_family"].isin(["alignment_conditioned_accuracy", "validation_soup_accuracy"])
    ].copy()
    barrier = supported[supported["outcome_family"] == "alignment_conditioned_barrier"].copy()
    unsupported_names = sorted(set(TARGETS) - set(supported["target"].astype(str).unique()))
    unsupported = pd.DataFrame({"target": unsupported_names})
    return raw, conditioned, barrier, unsupported


def write_report(args: argparse.Namespace, runs: pd.DataFrame, stats: pd.DataFrame, audit: pd.DataFrame) -> None:
    report_path = args.reports_dir / OUTPUT_REPORT
    observed = stats[stats["alignment_source"].astype(str) == "observed"].copy()
    injected = stats[stats["alignment_source"].astype(str) != "observed"].copy()
    supported = observed[observed["claim_supported"] == True].copy()  # noqa: E712
    setting_specific = supported[supported["support_scope"] == "setting_specific"].copy()
    raw_supported, conditioned_supported, barrier_supported, unsupported_targets = target_support_sections(stats)
    barrier_rows = stats[stats["target"].astype(str).isin(BARRIER_TARGETS)].copy()
    barrier_scales = sorted(set(barrier_rows.get("barrier_eval_scale", pd.Series(dtype=str)).dropna().astype(str))) if not barrier_rows.empty else []
    barrier_caps = sorted(set(barrier_rows.get("barrier_max_eval_batches", pd.Series(dtype=str)).dropna().astype(str))) if not barrier_rows.empty else []
    citation_preferred = bool(
        not barrier_rows.empty
        and "barrier_citation_preferred" in barrier_rows
        and barrier_rows["barrier_citation_preferred"].astype(bool).any()
    )
    stability_rows = barrier_rows[
        barrier_rows.get("previous_two_batch_supported", pd.Series(False, index=barrier_rows.index)).astype(bool)
    ].copy()
    datasets = sorted(str(item) for item in runs["dataset"].dropna().unique())
    architectures = sorted(str(item) for item in runs["architecture"].dropna().unique())
    fake_rows = runs[runs["dataset"].astype(str).str.contains("fake", case=False, na=False)]
    command = " ".join([sys.executable, *sys.argv])
    source_command = load_config_command(args.reports_dir)
    report = f"""# Obstruction Predictor Target Diagnostics

Generated by `experiments/obstruction_predictor_target_diagnostics.py` from completed quality-gated fixed-setting CSVs.

## Rerun Command

```bash
{command}
```

## Source Quality-Gated Commands

```bash
{source_command}
```

## Data Audit

- Source run rows: {len(runs)}
- Weight-average diagnostic rows: {int((runs["method"].astype(str) == "weight_average").sum())}
- Datasets used as evidence: {", ".join(datasets)}
- Architectures used as evidence: {", ".join(architectures)}
- Fake/smoke dataset rows: {len(fake_rows)}
- Bootstrap samples: {args.bootstrap_samples}
- Barrier target CSV: `{args.barriers_csv}`
- Barrier evaluation scale: {", ".join(barrier_scales) if barrier_scales else "not available"}
- Barrier max eval batches: {", ".join(barrier_caps) if barrier_caps else "not available"} (`0` means full loader)
- Citation-preferred barrier outputs: {"yes" if citation_preferred else "no"}
- Platform: {platform.platform()}

{md_table(audit, ["input", "exists", "rows", "size_mb"], max_rows=20)}

## Fixed Settings

{summarize_datasets(runs)}

## Scope And Gate

- Observed alignment rows are the only primary evidence.
- Injected-noise rows are retained only as negative/control diagnostics.
- Each fixed setting is kept separate by dataset, architecture, model count, width, domain shift, matching, and alignment source.
- Regressions control for mean individual accuracy and pairwise alignment residual within each fixed setting.
- Dataset, model count, and domain shift are handled by fixed-setting stratification rather than pooled smoke-data mixing.
- A predictor-target row is supported only when it has at least 20 unique seeds, an observed alignment source, and a positive bootstrap lower bound for the predictor coefficient.
- If the positive sign is not repeated in a secondary setting, the row is explicitly labeled setting-specific.
- Raw accuracy targets and alignment-conditioned barrier targets are reported separately.
- Barrier targets are validation-loss interpolation barriers merged from `alignment_barrier_targets.csv`; test barriers remain evaluation-only in the barrier report.
- Barrier rows are merged by dataset, architecture, model count, width, domain shift, matching, seed, and alignment source; method-specific barriers are pivoted from the barrier method rows.
- Full-loader barrier targets (`max_eval_batches=0`) are citation-preferred. Previous two-batch/debug barrier claims are used only for stability comparison.

## Supported Raw Targets

{md_table(raw_supported, ["dataset", "n_models", "width", "domain_shift", "target", "predictor", "n_unique_seeds", "predictor_beta", "predictor_beta_ci_low", "predictor_beta_ci_high", "support_scope", "claim_status"], max_rows=60)}

## Supported Alignment-Conditioned Targets

{md_table(conditioned_supported, ["dataset", "n_models", "width", "domain_shift", "target", "predictor", "n_unique_seeds", "predictor_beta", "predictor_beta_ci_low", "predictor_beta_ci_high", "support_scope", "claim_status"], max_rows=80)}

## Supported Barrier Targets

{md_table(barrier_supported, ["dataset", "n_models", "width", "domain_shift", "target", "predictor", "n_unique_seeds", "predictor_beta", "predictor_beta_ci_low", "predictor_beta_ci_high", "barrier_eval_scale", "barrier_stability_decision", "support_scope", "claim_status"], max_rows=80)}

## Full-Loader Barrier Stability

This section compares previous two-batch/debug supported barrier claims against the current full-loader barrier rows. Rows marked `not_stable_previous_two_batch_support_disappeared` should not be cited as supported. Rows marked `stable_full_loader_supported` are the citation-preferred full-loader survivors.

{md_table(stability_rows, ["dataset", "n_models", "width", "domain_shift", "target", "predictor", "previous_two_batch_beta", "previous_two_batch_ci_low", "previous_two_batch_ci_high", "predictor_beta", "predictor_beta_ci_low", "predictor_beta_ci_high", "barrier_stability_decision", "claim_status"], max_rows=80)}

## Unsupported Targets

{md_table(unsupported_targets, ["target"], max_rows=20)}

## Setting-Specific Supported Rows

{md_table(setting_specific, ["dataset", "n_models", "width", "domain_shift", "target", "predictor", "predictor_beta", "predictor_beta_ci_low", "predictor_beta_ci_high", "claim_status"], max_rows=80)}

## Recommendation For Paper Wording

{recommendation(stats)}

## Observed Predictor-Target Rows

{md_table(observed, ["dataset", "n_models", "width", "domain_shift", "target", "outcome_family", "predictor", "n_unique_seeds", "predictor_beta", "predictor_beta_ci_low", "predictor_beta_ci_high", "claim_status"], max_rows=120)}

## Injected-Noise Controls

{md_table(injected, ["dataset", "n_models", "width", "domain_shift", "alignment_noise_fraction", "target", "predictor", "n_unique_seeds", "claim_status"], max_rows=60)}
"""
    report_path.write_text(report, encoding="utf-8")


def write_barrier_report(args: argparse.Namespace, stats: pd.DataFrame, audit: pd.DataFrame) -> None:
    report_path = args.reports_dir / OUTPUT_BARRIER_REPORT
    observed = stats[
        (stats["alignment_source"].astype(str) == "observed")
        & (pd.to_numeric(stats["alignment_noise_fraction"], errors="coerce") == 0.0)
    ].copy()
    barrier = observed[observed["target"].astype(str).isin(BARRIER_TARGETS)].copy()
    raw = observed[observed["outcome_family"].astype(str) == "raw_accuracy"].copy()
    supported_barrier = barrier[barrier["claim_supported"] == True].copy()  # noqa: E712
    supported_raw = raw[raw["claim_supported"] == True].copy()  # noqa: E712
    citation_preferred = bool(
        "barrier_citation_preferred" in barrier
        and not barrier.empty
        and barrier["barrier_citation_preferred"].astype(bool).any()
    )
    barrier_scales = sorted(set(barrier.get("barrier_eval_scale", pd.Series(dtype=str)).dropna().astype(str))) if not barrier.empty else []
    barrier_caps = sorted(set(barrier.get("barrier_max_eval_batches", pd.Series(dtype=str)).dropna().astype(str))) if not barrier.empty else []
    stability_rows = barrier[
        barrier.get("previous_two_batch_supported", pd.Series(False, index=barrier.index)).astype(bool)
    ].copy()
    unavailable = stats[
        stats["target"].astype(str).isin(BARRIER_TARGETS)
        & (stats["target_status"].astype(str) == "not_run")
    ].copy()
    if supported_barrier.empty and supported_raw.empty:
        conclusion = "Neither raw accuracy nor barrier targets pass the observed bootstrap gate."
    elif not supported_barrier.empty and supported_raw.empty:
        conclusion = (
            "Barrier-style targets pass in selected fixed settings while raw weight-average degradation does not. "
            "The supported wording should be barrier-specific, not a raw merge-degradation prediction claim."
        )
    elif supported_barrier.empty and not supported_raw.empty:
        conclusion = "Raw accuracy targets pass but barrier targets do not; do not promote a barrier-prediction claim."
    else:
        conclusion = "Both raw and barrier targets have supported rows; keep every claim fixed-setting and predictor-specific."

    target_summary = (
        observed.groupby(["target", "outcome_family"], dropna=False)
        .agg(
            n_rows=("target", "size"),
            n_supported=("claim_supported", "sum"),
            n_target_not_run=("target_status", lambda values: int((values.astype(str) == "not_run").sum())),
        )
        .reset_index()
        .sort_values(["outcome_family", "target"])
    )
    supported_display = supported_barrier.sort_values(
        ["target", "dataset", "n_models", "domain_shift", "predictor"]
    )
    report = f"""# Obstruction Barrier Predictor Diagnostics

Generated by `experiments/obstruction_predictor_target_diagnostics.py`.

## Scope

- This is the barrier-focused companion to `reports/obstruction_predictor_target_report.md`.
- Barrier targets are validation-loss interpolation barriers from `reports/csv/alignment_barrier_targets.csv`.
- Test barriers remain evaluation-only in `reports/alignment_barrier_targets_report.md`; this predictor report uses validation barrier targets.
- Observed alignment rows with at least 20 unique seeds are the only primary evidence.
- Injected-noise rows are not primary evidence.
- Regressions control for mean individual accuracy and pairwise alignment residual within fixed-setting strata.
- Fixed-setting strata are dataset, architecture, model count, width, domain shift, matching, alignment source, and alignment-noise fraction.
- Current barrier rows use evaluation scale `{", ".join(barrier_scales) if barrier_scales else "not available"}` with max eval batches `{", ".join(barrier_caps) if barrier_caps else "not available"}`; `0` means full loader.
- Full-loader barrier rows are citation-preferred: {"yes" if citation_preferred else "no"}.
- Previous two-batch/debug barrier claims are retained only as a stability comparison.

## Data Audit

{md_table(audit, ["input", "exists", "rows", "size_mb"], max_rows=20)}

## Barrier Target Availability

{md_table(target_summary[target_summary["target"].astype(str).isin(BARRIER_TARGETS)], ["target", "outcome_family", "n_rows", "n_supported", "n_target_not_run"], max_rows=20)}

## Supported Barrier Rows

{md_table(supported_display, ["dataset", "n_models", "width", "domain_shift", "target", "predictor", "n_unique_seeds", "predictor_beta", "predictor_beta_ci_low", "predictor_beta_ci_high", "barrier_eval_scale", "barrier_stability_decision", "support_scope", "claim_status"], max_rows=120)}

## Two-Batch Vs Full-Loader Claim Stability

Rows below were supported by the previous two-batch/debug barrier run and rechecked against the current full-loader barriers. Disappearing support is marked `not_stable_previous_two_batch_support_disappeared`; surviving support is marked `stable_full_loader_supported`.

{md_table(stability_rows, ["dataset", "n_models", "width", "domain_shift", "target", "predictor", "previous_two_batch_beta", "previous_two_batch_ci_low", "previous_two_batch_ci_high", "predictor_beta", "predictor_beta_ci_low", "predictor_beta_ci_high", "barrier_stability_decision", "claim_status"], max_rows=120)}

## Raw Accuracy Gate

{md_table(supported_raw, ["dataset", "n_models", "width", "domain_shift", "target", "predictor", "n_unique_seeds", "predictor_beta", "predictor_beta_ci_low", "predictor_beta_ci_high", "support_scope", "claim_status"], max_rows=60)}

## Not-Run Barrier Rows

{md_table(unavailable, ["dataset", "n_models", "width", "domain_shift", "alignment_source", "target", "predictor", "claim_status"], max_rows=40)}

## Conclusion

{conclusion}

## Claim Boundary

This report tests whether obstruction/alignment diagnostics predict barrier-style targets better than raw weight-average degradation. It does not claim a method-performance win and does not use injected-noise rows as primary evidence.
"""
    report_path.write_text(report, encoding="utf-8")


def wrap_label(label: str, width: int) -> str:
    if len(label) <= width:
        return label
    parts = label.split("_")
    lines: list[str] = []
    current = ""
    for part in parts:
        candidate = part if not current else f"{current}_{part}"
        if len(candidate) > width and current:
            lines.append(current)
            current = part
        else:
            current = candidate
    if current:
        lines.append(current)
    return "\n".join(lines)


def write_plot(
    args: argparse.Namespace,
    stats: pd.DataFrame,
    output_name: str = OUTPUT_PLOT,
    targets: tuple[str, ...] = TARGETS,
    title: str = "Observed predictor coefficients on full quality-gated fixed-setting data",
) -> None:
    import matplotlib.pyplot as plt

    path = args.reports_dir / "plots" / output_name
    observed = stats[
        (stats["alignment_source"].astype(str) == "observed")
        & (stats["target_status"].astype(str) != "not_run")
        & (stats["target"].astype(str).isin(targets))
    ].copy()
    height = max(4.8, 1.0 + 0.55 * len(targets))
    fig, ax = plt.subplots(figsize=(13.0, height))
    if observed.empty:
        ax.text(0.5, 0.5, "No observed predictor-target rows", ha="center", va="center")
        ax.set_axis_off()
    else:
        observed["predictor_beta"] = pd.to_numeric(observed["predictor_beta"], errors="coerce")
        pivot = (
            observed.groupby(["target", "predictor"], dropna=False)["predictor_beta"]
            .mean()
            .unstack("predictor")
            .reindex(index=list(targets), columns=list(PREDICTORS))
        )
        values = pivot.to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        vmax = max(float(np.max(np.abs(finite))), 1e-12) if finite.size else 1.0
        image = ax.imshow(values, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
        fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02, label="mean controlled predictor coefficient")
        supported = {
            (str(row.target), str(row.predictor))
            for row in observed.itertuples()
            if bool(getattr(row, "claim_supported", False))
        }
        setting_specific = {
            (str(row.target), str(row.predictor))
            for row in observed.itertuples()
            if str(getattr(row, "support_scope", "")) == "setting_specific"
        }
        for y_idx, target in enumerate(pivot.index):
            for x_idx, predictor in enumerate(pivot.columns):
                key = (str(target), str(predictor))
                if key in supported:
                    marker = "s" if key in setting_specific else "*"
                    ax.text(
                        x_idx,
                        y_idx,
                        marker,
                        ha="center",
                        va="center",
                        color="black",
                        fontsize=13,
                        fontweight="bold",
                    )
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels(
            [wrap_label(label, args.max_plot_label_length) for label in pivot.columns],
            rotation=35,
            ha="right",
            fontsize=8,
        )
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels([wrap_label(label, 34) for label in pivot.index], fontsize=8)
        ax.set_title(title)
        ax.set_xlabel("Predictor")
        ax.set_ylabel("Target")
        ax.text(
            0.0,
            -0.18,
            "* replicated positive-sign support; s setting-specific support; injected controls excluded from heatmap",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    (args.reports_dir / "csv").mkdir(parents=True, exist_ok=True)
    (args.reports_dir / "plots").mkdir(parents=True, exist_ok=True)
    runs = pd.read_csv(args.runs_csv)
    runs = merge_barrier_targets(runs, args.barriers_csv)
    stats = compute_rows(runs, args.bootstrap_samples)
    direct_barrier_stats = compute_barrier_rows_from_barriers(args.barriers_csv, args.bootstrap_samples)
    if not direct_barrier_stats.empty:
        stats = pd.concat(
            [
                stats[~stats["target"].astype(str).isin(BARRIER_TARGETS)].copy(),
                direct_barrier_stats,
            ],
            ignore_index=True,
            sort=False,
        )
    stats, _barrier_metadata, _claim_stability = apply_full_loader_barrier_stability(
        stats,
        args.reports_dir,
        args.barriers_csv,
    )
    barrier_stats = stats[stats["target"].astype(str).isin(BARRIER_TARGETS)].copy()

    stats_path = args.reports_dir / "csv" / OUTPUT_STATS
    regressions_path = args.reports_dir / "csv" / OUTPUT_REGRESSIONS
    barrier_stats_path = args.reports_dir / "csv" / OUTPUT_BARRIER_STATS
    stats.to_csv(stats_path, index=False, lineterminator="\n")
    stats.to_csv(regressions_path, index=False, lineterminator="\n")
    barrier_stats.to_csv(barrier_stats_path, index=False, lineterminator="\n")
    audit = input_audit(args.reports_dir)
    write_plot(args, stats)
    write_plot(
        args,
        barrier_stats,
        output_name=OUTPUT_BARRIER_PLOT,
        targets=BARRIER_TARGETS,
        title="Observed predictor coefficients for validation barrier targets",
    )
    write_report(args, runs, stats, audit)
    write_barrier_report(args, stats, audit)

    print(f"wrote {stats_path}")
    print(f"wrote {regressions_path}")
    print(f"wrote {barrier_stats_path}")
    print(f"wrote {args.reports_dir / OUTPUT_REPORT}")
    print(f"wrote {args.reports_dir / OUTPUT_BARRIER_REPORT}")
    print(f"wrote {args.reports_dir / 'plots' / OUTPUT_PLOT}")
    print(f"wrote {args.reports_dir / 'plots' / OUTPUT_BARRIER_PLOT}")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from exc
