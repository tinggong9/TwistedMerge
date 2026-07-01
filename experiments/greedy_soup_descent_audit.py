#!/usr/bin/env python
"""Audit greedy soup as validation-selected empirical descent.

The fixed-setting verifier stores final greedy-soup rows and per-local-model
validation/test metrics, but not the full candidate soup validation trajectory.
This audit reconstructs the validation candidate order and final accepted
ingredients from saved metadata, then marks which descent checks are directly
observable and which are algorithm-implied but not logged.
"""

from __future__ import annotations

import argparse
import ast
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model_merging_benchmark import format_markdown_table  # noqa: E402


SETTING_COLS = ["dataset", "architecture", "n_models", "width", "domain_shift", "matching"]
RUN_KEY_COLS = ["setting_id", "run_id", *SETTING_COLS, "seed"]
TOL = 1e-12


def safe_float(value) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def parse_indices(value) -> list[int]:
    if isinstance(value, list):
        return [int(item) for item in value]
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
    except Exception:
        return []
    if isinstance(parsed, (list, tuple)):
        return [int(item) for item in parsed]
    return []


def md_table(df: pd.DataFrame, columns: list[str], max_rows: int = 30) -> str:
    if df.empty:
        return "_No rows._"
    rows = df.head(max_rows).copy()
    for col in columns:
        if col not in rows:
            rows[col] = ""
    return format_markdown_table(rows[columns].to_dict("records"), columns)


def add_selector_metadata(greedy: pd.DataFrame, selector_path: Path) -> pd.DataFrame:
    if not selector_path.exists():
        greedy = greedy.copy()
        greedy["diagnostic_selector_present"] = False
        greedy["diagnostic_validation_safe"] = np.nan
        return greedy
    selector = pd.read_csv(selector_path)
    if selector.empty or "selector" not in selector:
        greedy = greedy.copy()
        greedy["diagnostic_selector_present"] = False
        greedy["diagnostic_validation_safe"] = np.nan
        return greedy
    selector = selector[selector["selector"].astype(str) == "greedy_baseline_selector"].copy()
    keep = [
        col
        for col in [
            "run_id",
            "selected_method",
            "selection_reason",
            "greedy_val_accuracy",
            "greedy_test_accuracy",
            "single_best_val_accuracy",
            "single_best_test_accuracy",
            "validation_safe",
            "test_used_for_selection",
        ]
        if col in selector
    ]
    if "run_id" not in keep:
        greedy = greedy.copy()
        greedy["diagnostic_selector_present"] = False
        greedy["diagnostic_validation_safe"] = np.nan
        return greedy
    selector = selector[keep].rename(
        columns={
            "selected_method": "diagnostic_selected_method",
            "selection_reason": "diagnostic_selection_reason",
            "greedy_val_accuracy": "diagnostic_greedy_val_accuracy",
            "greedy_test_accuracy": "diagnostic_greedy_test_accuracy",
            "single_best_val_accuracy": "diagnostic_single_best_val_accuracy",
            "single_best_test_accuracy": "diagnostic_single_best_test_accuracy",
            "validation_safe": "diagnostic_validation_safe",
            "test_used_for_selection": "diagnostic_test_used_for_selection",
        }
    )
    out = greedy.merge(selector, on="run_id", how="left")
    out["diagnostic_selector_present"] = out["diagnostic_selected_method"].notna()
    return out


def load_inputs(args):
    runs = pd.read_csv(args.runs_csv)
    individuals = pd.read_csv(args.individuals_csv)
    greedy = runs[
        (runs["method"].astype(str) == "greedy_soup")
        & (runs["alignment_source"].astype(str) == "observed")
        & (pd.to_numeric(runs["alignment_noise_fraction"], errors="coerce").fillna(0.0) == 0.0)
    ].copy()
    greedy = add_selector_metadata(greedy, args.selector_csv)
    return runs, individuals, greedy


def reconstruct_candidate_audit(greedy: pd.DataFrame, individuals: pd.DataFrame) -> pd.DataFrame:
    rows = []
    individual_groups = {run_id: group.copy() for run_id, group in individuals.groupby("run_id", dropna=False)}
    for greedy_row in greedy.itertuples(index=False):
        run_id = str(greedy_row.run_id)
        group = individual_groups.get(run_id)
        if group is None or group.empty:
            continue
        group = group.copy()
        group["val_accuracy_num"] = pd.to_numeric(group["val_accuracy"], errors="coerce")
        group["model_index_num"] = pd.to_numeric(group["model_index"], errors="coerce").astype(int)
        # The implementation stores tuples as (validation_accuracy, model_index)
        # and calls sorted(..., reverse=True), so ties prefer higher model_index.
        group = group.sort_values(["val_accuracy_num", "model_index_num"], ascending=[False, False], kind="stable")
        order = [int(value) for value in group["model_index_num"].tolist()]
        selected = parse_indices(getattr(greedy_row, "selection_indices", ""))
        selected_set = set(selected)
        best_val_model = group.iloc[0]
        best_test_model = individuals[individuals["run_id"].astype(str) == run_id].sort_values(
            ["test_accuracy", "model_index"],
            ascending=[False, False],
            kind="stable",
        ).iloc[0]

        greedy_val_accuracy = safe_float(getattr(greedy_row, "val_accuracy", float("nan")))
        greedy_val_loss = safe_float(getattr(greedy_row, "val_loss", float("nan")))
        greedy_test_accuracy = safe_float(getattr(greedy_row, "test_accuracy", float("nan")))
        greedy_test_loss = safe_float(getattr(greedy_row, "test_loss", float("nan")))
        greedy_val_risk = 1.0 - greedy_val_accuracy
        greedy_test_risk = 1.0 - greedy_test_accuracy
        best_val_accuracy = safe_float(best_val_model["val_accuracy"])
        best_val_risk = 1.0 - best_val_accuracy
        best_test_accuracy = safe_float(best_test_model["test_accuracy"])
        best_test_risk = 1.0 - best_test_accuracy
        best_test_val_risk = 1.0 - safe_float(best_test_model["val_accuracy"])
        generalization_proxy = abs(greedy_test_risk - greedy_val_risk) + abs(best_test_risk - best_test_val_risk)
        validation_descent_margin = best_val_risk - greedy_val_risk
        test_excess_vs_best_test = greedy_test_risk - best_test_risk

        current_val_accuracy = float("-inf")
        current_indices: list[int] = []
        for candidate_rank, item in enumerate(group.itertuples(index=False), start=1):
            candidate_index = int(item.model_index_num)
            candidate_individual_val_accuracy = safe_float(item.val_accuracy)
            candidate_individual_val_risk = 1.0 - candidate_individual_val_accuracy
            candidate_individual_test_accuracy = safe_float(item.test_accuracy)
            candidate_individual_test_risk = 1.0 - candidate_individual_test_accuracy
            if candidate_rank == 1:
                decision = "accepted_initial_best_validation_model"
                accepted = True
                decision_source = "direct_individual_validation_metric"
                before_accuracy = float("nan")
                after_accuracy = candidate_individual_val_accuracy
                before_risk = float("nan")
                after_risk = candidate_individual_val_risk
                margin = float("nan")
                current_val_accuracy = candidate_individual_val_accuracy
                current_indices = [candidate_index]
                after_indices = current_indices.copy()
                test_risk_after_accepted = candidate_individual_test_risk
            elif candidate_index in selected_set:
                decision = "accepted_final_or_intermediate_soup"
                accepted = True
                decision_source = "final_selection_indices_without_step_metric"
                before_accuracy = current_val_accuracy
                before_risk = 1.0 - before_accuracy
                if set(current_indices + [candidate_index]) == selected_set:
                    after_accuracy = greedy_val_accuracy
                    after_risk = greedy_val_risk
                    margin = after_accuracy - before_accuracy
                    test_risk_after_accepted = greedy_test_risk
                    decision_source = "final_greedy_row_metric"
                else:
                    after_accuracy = float("nan")
                    after_risk = float("nan")
                    margin = float("nan")
                    test_risk_after_accepted = float("nan")
                current_indices.append(candidate_index)
                current_val_accuracy = after_accuracy if math.isfinite(after_accuracy) else current_val_accuracy
                after_indices = current_indices.copy()
            else:
                decision = "rejected_inferred_from_final_selection"
                accepted = False
                decision_source = "decision_inferred_candidate_soup_metric_not_logged"
                before_accuracy = current_val_accuracy
                before_risk = 1.0 - before_accuracy
                after_accuracy = float("nan")
                after_risk = float("nan")
                margin = float("nan")
                after_indices = current_indices.copy()
                test_risk_after_accepted = float("nan")

            row = {col: getattr(greedy_row, col) for col in RUN_KEY_COLS if hasattr(greedy_row, col)}
            row.update(
                {
                    "candidate_rank": candidate_rank,
                    "candidate_model_index": candidate_index,
                    "candidate_order": str(order),
                    "candidate_order_source": "individual_validation_accuracy_descending",
                    "final_selection_indices": str(selected),
                    "accepted": accepted,
                    "decision": decision,
                    "decision_source": decision_source,
                    "soup_indices_before": str(current_indices[:-1] if accepted and candidate_rank > 1 else current_indices if not accepted else []),
                    "soup_indices_after": str(after_indices),
                    "validation_accuracy_before_candidate": before_accuracy,
                    "validation_accuracy_after_candidate": after_accuracy,
                    "validation_risk_before_candidate": before_risk,
                    "validation_risk_after_candidate": after_risk,
                    "validation_accuracy_margin_after_minus_before": margin,
                    "validation_risk_descent_margin_before_minus_after": -margin if math.isfinite(margin) else float("nan"),
                    "candidate_individual_val_accuracy": candidate_individual_val_accuracy,
                    "candidate_individual_val_loss": safe_float(item.val_loss),
                    "candidate_individual_test_accuracy": candidate_individual_test_accuracy,
                    "candidate_individual_test_loss": safe_float(item.test_loss),
                    "candidate_individual_val_risk": candidate_individual_val_risk,
                    "candidate_individual_test_risk": candidate_individual_test_risk,
                    "greedy_val_accuracy": greedy_val_accuracy,
                    "greedy_val_loss": greedy_val_loss,
                    "greedy_test_accuracy": greedy_test_accuracy,
                    "greedy_test_loss": greedy_test_loss,
                    "greedy_val_risk": greedy_val_risk,
                    "greedy_test_risk": greedy_test_risk,
                    "best_validation_model_index": int(best_val_model["model_index"]),
                    "best_validation_model_val_accuracy": best_val_accuracy,
                    "best_validation_model_test_accuracy": safe_float(best_val_model["test_accuracy"]),
                    "best_validation_model_val_risk": best_val_risk,
                    "best_test_model_index": int(best_test_model["model_index"]),
                    "best_test_model_val_accuracy": safe_float(best_test_model["val_accuracy"]),
                    "best_test_model_test_accuracy": best_test_accuracy,
                    "best_test_model_test_risk": best_test_risk,
                    "greedy_validation_descent_margin_vs_best_val_risk": validation_descent_margin,
                    "greedy_test_excess_risk_vs_best_test": test_excess_vs_best_test,
                    "generalization_proxy_abs_gaps": generalization_proxy,
                    "test_proxy_bound_margin": generalization_proxy - test_excess_vs_best_test,
                    "candidate_after_metric_logged": bool(math.isfinite(after_accuracy)),
                    "rejected_margin_empirically_checkable": False if not accepted else bool(math.isfinite(margin)),
                    "test_risk_after_accepted_candidate": test_risk_after_accepted,
                    "diagnostic_selector_present": bool(getattr(greedy_row, "diagnostic_selector_present", False)),
                    "diagnostic_validation_safe": getattr(greedy_row, "diagnostic_validation_safe", np.nan),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def build_summary(audit: pd.DataFrame) -> pd.DataFrame:
    if audit.empty:
        return pd.DataFrame()
    run_level = audit.groupby("run_id", dropna=False).agg(
        setting_id=("setting_id", "first"),
        dataset=("dataset", "first"),
        architecture=("architecture", "first"),
        n_models=("n_models", "first"),
        width=("width", "first"),
        domain_shift=("domain_shift", "first"),
        matching=("matching", "first"),
        seed=("seed", "first"),
        accepted_candidates=("accepted", "sum"),
        rejected_candidates=("accepted", lambda values: int((~values.astype(bool)).sum())),
        logged_candidate_steps=("candidate_after_metric_logged", "sum"),
        unlogged_candidate_steps=("candidate_after_metric_logged", lambda values: int((~values.astype(bool)).sum())),
        rejected_unlogged_candidates=("decision", lambda values: int((values.astype(str) == "rejected_inferred_from_final_selection").sum())),
        validation_monotonicity_violations=(
            "validation_accuracy_margin_after_minus_before",
            lambda values: int((pd.to_numeric(values, errors="coerce").dropna() < -TOL).sum()),
        ),
        greedy_val_accuracy=("greedy_val_accuracy", "first"),
        greedy_test_accuracy=("greedy_test_accuracy", "first"),
        greedy_val_risk=("greedy_val_risk", "first"),
        greedy_test_risk=("greedy_test_risk", "first"),
        best_validation_model_val_accuracy=("best_validation_model_val_accuracy", "first"),
        best_validation_model_test_accuracy=("best_validation_model_test_accuracy", "first"),
        best_validation_model_val_risk=("best_validation_model_val_risk", "first"),
        best_test_model_test_accuracy=("best_test_model_test_accuracy", "first"),
        best_test_model_test_risk=("best_test_model_test_risk", "first"),
        greedy_validation_descent_margin_vs_best_val_risk=("greedy_validation_descent_margin_vs_best_val_risk", "first"),
        greedy_test_excess_risk_vs_best_test=("greedy_test_excess_risk_vs_best_test", "first"),
        generalization_proxy_abs_gaps=("generalization_proxy_abs_gaps", "first"),
        test_proxy_bound_margin=("test_proxy_bound_margin", "first"),
        diagnostic_selector_present=("diagnostic_selector_present", "first"),
    ).reset_index()
    run_level["greedy_val_risk_le_best_single_val_risk"] = (
        run_level["greedy_val_risk"] <= run_level["best_validation_model_val_risk"] + TOL
    )
    run_level["test_within_validation_gap_plus_generalization_proxy"] = run_level["test_proxy_bound_margin"] >= -TOL
    run_level["rejected_margin_check_status"] = np.where(
        run_level["rejected_unlogged_candidates"] > 0,
        "not_empirically_logged_algorithm_implied",
        "no_rejected_candidates",
    )

    group_cols = SETTING_COLS
    summary = run_level.groupby(group_cols, dropna=False).agg(
        summary_type=("run_id", lambda _: "setting_summary"),
        n_runs=("run_id", "count"),
        n_unique_seeds=("seed", "nunique"),
        mean_accepted_candidates=("accepted_candidates", "mean"),
        mean_rejected_candidates=("rejected_candidates", "mean"),
        total_accepted_candidates=("accepted_candidates", "sum"),
        total_rejected_candidates=("rejected_candidates", "sum"),
        validation_monotonicity_violations=("validation_monotonicity_violations", "sum"),
        greedy_val_risk_le_best_single_val_risk_violations=(
            "greedy_val_risk_le_best_single_val_risk",
            lambda values: int((~values.astype(bool)).sum()),
        ),
        test_proxy_bound_violations=("test_within_validation_gap_plus_generalization_proxy", lambda values: int((~values.astype(bool)).sum())),
        rejected_candidates_unlogged=("rejected_unlogged_candidates", "sum"),
        mean_validation_descent_margin_vs_best_val_risk=("greedy_validation_descent_margin_vs_best_val_risk", "mean"),
        min_validation_descent_margin_vs_best_val_risk=("greedy_validation_descent_margin_vs_best_val_risk", "min"),
        mean_test_excess_risk_vs_best_test=("greedy_test_excess_risk_vs_best_test", "mean"),
        mean_generalization_proxy_abs_gaps=("generalization_proxy_abs_gaps", "mean"),
        mean_test_proxy_bound_margin=("test_proxy_bound_margin", "mean"),
        mean_greedy_test_accuracy=("greedy_test_accuracy", "mean"),
        mean_best_validation_model_test_accuracy=("best_validation_model_test_accuracy", "mean"),
        mean_best_test_model_test_accuracy=("best_test_model_test_accuracy", "mean"),
        diagnostic_selector_coverage=("diagnostic_selector_present", "mean"),
    ).reset_index()
    summary["validation_descent_supported"] = summary["greedy_val_risk_le_best_single_val_risk_violations"].eq(0) & summary[
        "validation_monotonicity_violations"
    ].eq(0)
    summary["rejected_margin_check_status"] = np.where(
        summary["rejected_candidates_unlogged"] > 0,
        "not_empirically_logged_algorithm_implied",
        "checked_or_no_rejections",
    )
    summary["claim_decision"] = np.where(
        summary["validation_descent_supported"],
        "supports_empirical_validation_descent_final_only",
        "falsifies_validation_descent",
    )
    return summary


def build_theorem_table(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    rows = []
    for row in summary.itertuples(index=False):
        base = {col: getattr(row, col) for col in SETTING_COLS}
        rows.append(
            {
                **base,
                "theorem_check": "greedy_validation_risk_le_best_single_validation_risk",
                "n_runs": int(row.n_runs),
                "violations": int(row.greedy_val_risk_le_best_single_val_risk_violations),
                "check_status": "supported" if int(row.greedy_val_risk_le_best_single_val_risk_violations) == 0 else "falsified",
                "boundary": "final accepted greedy row only; candidate soup after-metrics were not fully logged",
            }
        )
        rows.append(
            {
                **base,
                "theorem_check": "test_risk_within_validation_gap_plus_generalization_proxy",
                "n_runs": int(row.n_runs),
                "violations": int(row.test_proxy_bound_violations),
                "check_status": "supported_proxy" if int(row.test_proxy_bound_violations) == 0 else "proxy_violated",
                "boundary": "evaluation-only test accounting; not a proof of test optimality",
            }
        )
        rows.append(
            {
                **base,
                "theorem_check": "rejected_candidates_have_non_positive_validation_margin",
                "n_runs": int(row.n_runs),
                "violations": "",
                "check_status": str(row.rejected_margin_check_status),
                "boundary": "rejected candidate soup validation metrics are not saved in current artifacts",
            }
        )
    return pd.DataFrame(rows)


def plot_margins(summary: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    if summary.empty:
        for ax in axes:
            ax.text(0.5, 0.5, "No summary rows", ha="center", va="center")
            ax.set_axis_off()
    else:
        labels = [
            f"{row.dataset}\nN={int(row.n_models)} {row.domain_shift}\n{row.matching}"
            for row in summary.itertuples(index=False)
        ]
        x = np.arange(len(summary))
        axes[0].bar(x, summary["mean_validation_descent_margin_vs_best_val_risk"], color="tab:green", alpha=0.78)
        axes[0].axhline(0.0, color="black", linewidth=0.8)
        axes[0].set_title("Validation risk descent margin")
        axes[0].set_ylabel("best single val risk - greedy val risk")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        axes[0].grid(True, axis="y", alpha=0.25)

        axes[1].bar(x, summary["mean_test_excess_risk_vs_best_test"], color="tab:blue", alpha=0.78)
        axes[1].axhline(0.0, color="black", linewidth=0.8)
        axes[1].set_title("Evaluation-only test excess")
        axes[1].set_ylabel("greedy test risk - best local test risk")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        axes[1].grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_report(args, audit: pd.DataFrame, summary: pd.DataFrame, theorem: pd.DataFrame, path: Path) -> None:
    n_runs = int(summary["n_runs"].sum()) if not summary.empty else 0
    total_rejected = int(summary["total_rejected_candidates"].sum()) if not summary.empty else 0
    diagnostic_coverage = float(summary["diagnostic_selector_coverage"].mean()) if "diagnostic_selector_coverage" in summary else float("nan")
    rejected_status = (
        "not empirically logged for rejected candidate soups"
        if not summary.empty and (summary["rejected_candidates_unlogged"] > 0).any()
        else "checked or no rejected candidates"
    )
    descent_supported = bool(not summary.empty and summary["validation_descent_supported"].all())
    report = f"""# Greedy Soup Descent Audit

Generated by `experiments/greedy_soup_descent_audit.py`.

## Exact Command

```bash
{args.command_string}
```

## Claim Under Audit

Greedy soup can be interpreted as empirical descent on the validation objective used by the implementation: a candidate is accepted only when validation accuracy does not decrease. This report uses validation risk `1 - validation_accuracy` because the saved greedy-soup implementation accepts by validation accuracy, not by validation loss.

Decision: `{"supports_validation_descent_final_only" if descent_supported else "falsifies_validation_descent"}`.

## Data Boundary

The fixed-setting artifacts save final greedy-soup metrics, final selected ingredient indices, and individual local-model validation/test metrics. They do not save the full validation metric for every rejected averaged candidate soup. Therefore, the candidate order and final accepted ingredient set are reconstructed, final validation descent is directly checked, and rejected-candidate margins are labeled `{rejected_status}` rather than treated as observed measurements.

This is evidence for validation-selected descent, not a proof of test optimality.

## Outputs

- `reports/csv/greedy_soup_descent_audit.csv`
- `reports/csv/greedy_soup_descent_summary.csv`
- `reports/csv/greedy_soup_descent_theorem_checks.csv`
- `reports/plots/greedy_soup_validation_margins.pdf`
- `reports/greedy_soup_descent_audit.md`

## Overall Counts

- Observed fixed-setting greedy runs audited: `{n_runs}`
- Candidate rows reconstructed: `{len(audit)}`
- Rejected candidates: `{total_rejected}`
- Diagnostic selector CSV run-id coverage for these exact fixed-setting rows: `{diagnostic_coverage:.4f}`

## Setting Summary

{md_table(summary, ["dataset", "architecture", "n_models", "width", "domain_shift", "matching", "n_runs", "mean_accepted_candidates", "mean_rejected_candidates", "validation_monotonicity_violations", "greedy_val_risk_le_best_single_val_risk_violations", "test_proxy_bound_violations", "mean_validation_descent_margin_vs_best_val_risk", "mean_test_excess_risk_vs_best_test", "rejected_margin_check_status", "claim_decision"], 40)}

## Theorem-Check Table

{md_table(theorem, ["dataset", "architecture", "n_models", "width", "domain_shift", "matching", "theorem_check", "n_runs", "violations", "check_status", "boundary"], 80)}

## Candidate-Sequence Sample

{md_table(audit, ["setting_id", "seed", "candidate_rank", "candidate_model_index", "candidate_order", "decision", "decision_source", "candidate_individual_val_accuracy", "validation_accuracy_before_candidate", "validation_accuracy_after_candidate", "validation_accuracy_margin_after_minus_before"], 30)}

## Interpretation

The saved artifacts support the narrow claim that final greedy-soup outputs are validation-safe relative to the best validation local model in the fixed-setting runs. They do not support a stronger claim that every rejected averaged candidate has an empirically measured non-positive validation margin, because those candidate-after metrics were not logged. Test-set columns are evaluation-only diagnostics and should not be used as selection evidence.
"""
    path.write_text(report, encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-csv", type=Path, default=ROOT / "reports" / "csv" / "fixed_setting_verification_runs.csv")
    parser.add_argument("--individuals-csv", type=Path, default=ROOT / "reports" / "csv" / "fixed_setting_individual_models.csv")
    parser.add_argument("--selector-csv", type=Path, default=ROOT / "reports" / "csv" / "diagnostic_method_selector.csv")
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "reports")
    args = parser.parse_args()
    args.command_string = " ".join([sys.executable, *sys.argv])
    return args


def main() -> None:
    args = parse_args()
    _runs, individuals, greedy = load_inputs(args)
    audit = reconstruct_candidate_audit(greedy, individuals)
    summary = build_summary(audit)
    theorem = build_theorem_table(summary)

    csv_dir = args.reports_dir / "csv"
    plot_dir = args.reports_dir / "plots"
    csv_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    audit.to_csv(csv_dir / "greedy_soup_descent_audit.csv", index=False, lineterminator="\n")
    summary.to_csv(csv_dir / "greedy_soup_descent_summary.csv", index=False, lineterminator="\n")
    theorem.to_csv(csv_dir / "greedy_soup_descent_theorem_checks.csv", index=False, lineterminator="\n")
    plot_margins(summary, plot_dir / "greedy_soup_validation_margins.pdf")
    write_report(args, audit, summary, theorem, args.reports_dir / "greedy_soup_descent_audit.md")

    print(f"wrote {csv_dir / 'greedy_soup_descent_audit.csv'}")
    print(f"wrote {csv_dir / 'greedy_soup_descent_summary.csv'}")
    print(f"wrote {args.reports_dir / 'greedy_soup_descent_audit.md'}")


if __name__ == "__main__":
    main()
