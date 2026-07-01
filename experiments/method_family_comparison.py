#!/usr/bin/env python
"""Lightweight appendix comparison across model-merging method families.

This script aggregates existing MNIST/Fashion-MNIST MLP artifacts rather than
launching a new large training run.  It keeps independent-seed/rebasin rows
separate from shared-base task-vector rows and records structural coverage
without folding heterogeneous metrics into a single leaderboard.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.method_family_baselines import structural_coverage_matrix  # noqa: E402


INDEPENDENT_METHODS = {
    "weight_average": "Weight averaging",
    "greedy_soup": "Model Soups / greedy soup",
    "c2m3_permutation": "C2M3-style permutation synchronization",
    "improved_validated_selector": "TwistedMerge / TwistedMerge++ selector",
}

SHARED_METHODS = {
    "weight_average": "Weight averaging",
    "greedy_soup": "Model Soups / greedy soup",
    "slerp_sequential": "SLERP-style sequential soup",
    "task_arithmetic": "Task Arithmetic",
    "ties_merging": "TIES-style merging",
    "dare": "DARE-style merging",
}


def safe_float(value) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def fmt(value, digits: int = 4) -> str:
    number = safe_float(value)
    if not math.isfinite(number):
        return ""
    return f"{number:.{digits}f}"


def md_table(rows: list[dict], columns: list[str], max_rows: int | None = None) -> str:
    if max_rows is not None:
        rows = rows[:max_rows]
    out = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                value = fmt(value)
            values.append(str(value).replace("|", "\\|"))
        out.append("| " + " | ".join(values) + " |")
    return "\n".join(out)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def build_independent_summary(reports_dir: Path) -> pd.DataFrame:
    external = read_csv(reports_dir / "csv" / "external_baseline_comparison_summary.csv")
    overall = external[external["scope"].astype(str).eq("overall")].copy()
    rows: list[dict] = []
    for method, display in INDEPENDENT_METHODS.items():
        match = overall[overall["method"].astype(str).eq(method)]
        if match.empty:
            rows.append(
                {
                    "regime": "independent_seed_rebasin",
                    "method": method,
                    "display_name": display,
                    "status": "missing_existing_artifact",
                    "implementation_kind": "internal",
                    "n_rows": 0,
                }
            )
            continue
        item = match.iloc[0].to_dict()
        rows.append(
            {
                "regime": "independent_seed_rebasin",
                "method": method,
                "display_name": display,
                "status": "evaluated_existing_external_baseline_artifact",
                "implementation_kind": item.get("implementation_kind", "internal"),
                "n_rows": int(safe_float(item.get("n_rows", 0))),
                "n_seeds": int(safe_float(item.get("n_seeds", 0))),
                "mean_test_accuracy": safe_float(item.get("mean_test_accuracy")),
                "paired_delta_vs_greedy_soup": safe_float(item.get("paired_mean_accuracy_delta_vs_greedy_soup")),
                "paired_delta_vs_greedy_soup_ci_low": safe_float(item.get("paired_accuracy_delta_vs_greedy_soup_ci_low")),
                "paired_delta_vs_greedy_soup_ci_high": safe_float(item.get("paired_accuracy_delta_vs_greedy_soup_ci_high")),
                "paired_delta_vs_c2m3": safe_float(item.get("paired_mean_accuracy_delta_vs_internal_c2m3")),
                "paired_delta_vs_c2m3_ci_low": safe_float(item.get("paired_accuracy_delta_vs_internal_c2m3_ci_low")),
                "paired_delta_vs_c2m3_ci_high": safe_float(item.get("paired_accuracy_delta_vs_internal_c2m3_ci_high")),
                "validation_selected_hyperparameters": "validation accuracy/loss" if bool(item.get("uses_validation_data", False)) else "none",
                "source_csv": "reports/csv/external_baseline_comparison_summary.csv",
                "note": "MNIST one-hidden-layer MLP; faithful in-repo baselines, not official external code.",
            }
        )

    slerp_path = reports_dir / "csv" / "slerp_barrier_geometry.csv"
    if slerp_path.exists():
        slerp = read_csv(slerp_path)
        filt = slerp[
            slerp["regime"].astype(str).eq("fixed_independent_seed")
            & slerp["path_method"].astype(str).eq("slerp_interpolation")
        ].copy()
        rows.append(
            {
                "regime": "independent_seed_rebasin",
                "method": "slerp_pairwise_midpoint",
                "display_name": "SLERP-style pairwise midpoint",
                "status": "path_geometry_existing_artifact",
                "implementation_kind": "internal SLERP-style path baseline",
                "n_rows": int(len(filt)),
                "n_seeds": int(filt["seed"].nunique()) if not filt.empty else 0,
                "mean_test_accuracy": safe_float(pd.to_numeric(filt.get("test_accuracy_t05"), errors="coerce").mean()) if not filt.empty else float("nan"),
                "paired_delta_vs_greedy_soup": float("nan"),
                "paired_delta_vs_greedy_soup_ci_low": float("nan"),
                "paired_delta_vs_greedy_soup_ci_high": float("nan"),
                "paired_delta_vs_c2m3": float("nan"),
                "paired_delta_vs_c2m3_ci_low": float("nan"),
                "paired_delta_vs_c2m3_ci_high": float("nan"),
                "validation_selected_hyperparameters": "t=0.5 pairwise midpoint; no validation selection",
                "source_csv": "reports/csv/slerp_barrier_geometry.csv",
                "note": "Pairwise path midpoint accuracy, not a multi-model merged soup; included as an internal SLERP-style diagnostic.",
            }
        )
    return pd.DataFrame(rows)


def build_shared_base_summary(reports_dir: Path) -> pd.DataFrame:
    shared = read_csv(reports_dir / "csv" / "same_base_task_vector_extended_summary.csv")
    rows: list[dict] = []
    for method, display in SHARED_METHODS.items():
        subset = shared[(shared["method"].astype(str).eq(method)) & (shared["status"].astype(str).eq("ok"))]
        if subset.empty:
            rows.append(
                {
                    "regime": "shared_base_task_vector",
                    "method": method,
                    "display_name": display,
                    "status": "missing_existing_artifact",
                    "n_settings": 0,
                    "source_csv": "reports/csv/same_base_task_vector_extended_summary.csv",
                }
            )
            continue
        rows.append(
            {
                "regime": "shared_base_task_vector",
                "method": method,
                "display_name": display,
                "status": "evaluated_existing_same_base_artifact",
                "n_settings": int(len(subset)),
                "n_unique_seeds_min": int(pd.to_numeric(subset["n_unique_seeds"], errors="coerce").min()),
                "mean_test_accuracy_across_settings": safe_float(pd.to_numeric(subset["mean_average_test_accuracy"], errors="coerce").mean()),
                "best_setting_test_accuracy": safe_float(pd.to_numeric(subset["mean_average_test_accuracy"], errors="coerce").max()),
                "mean_delta_vs_greedy_soup": safe_float(pd.to_numeric(subset["mean_delta_vs_greedy_soup"], errors="coerce").mean()),
                "min_delta_vs_greedy_ci_low": safe_float(pd.to_numeric(subset["delta_vs_greedy_ci_low"], errors="coerce").min()),
                "max_delta_vs_greedy_ci_high": safe_float(pd.to_numeric(subset["delta_vs_greedy_ci_high"], errors="coerce").max()),
                "validation_selected_hyperparameters": (
                    "alpha grid" if method == "task_arithmetic" else "density/alpha grid" if method == "ties_merging" else "drop-rate/mask/alpha grid" if method == "dare" else "validation greedy order" if method == "greedy_soup" else "sequential equal-weight SLERP" if method == "slerp_sequential" else "none"
                ),
                "source_csv": "reports/csv/same_base_task_vector_extended_summary.csv",
                "note": "Common-base MLP task-vector setting; Task Arithmetic/TIES/DARE are internal style implementations.",
            }
        )
    rows.extend(
        [
            {
                "regime": "shared_base_task_vector",
                "method": "c2m3_permutation",
                "display_name": "C2M3-style synchronization",
                "status": "not_run_secondary_regime_mismatch",
                "n_settings": 0,
                "validation_selected_hyperparameters": "none",
                "source_csv": "reports/csv/same_base_task_vector_extended_summary.csv",
                "note": "C2M3 is an independent-seed/rebasin diagnostic; common-base rows do not intentionally create permutation mismatch.",
            },
            {
                "regime": "shared_base_task_vector",
                "method": "twistedmerge_selector",
                "display_name": "TwistedMerge / TwistedMerge++ selector",
                "status": "not_run_not_primary_for_common_base_task_vectors",
                "n_settings": 0,
                "validation_selected_hyperparameters": "validation selector not replayed in this regime",
                "source_csv": "reports/csv/same_base_task_vector_extended_summary.csv",
                "note": "TwistedMerge structural diagnostics remain relevant, but the existing shared-base artifact is a task-vector baseline benchmark.",
            },
        ]
    )
    return pd.DataFrame(rows)


def build_paired_rows(independent: pd.DataFrame, shared: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    tm = independent[independent["method"].eq("improved_validated_selector")]
    if not tm.empty:
        item = tm.iloc[0]
        rows.append(
            {
                "regime": "independent_seed_rebasin",
                "comparison": "TwistedMerge selector vs greedy soup",
                "method": "improved_validated_selector",
                "baseline": "greedy_soup",
                "paired_mean_delta": item.get("paired_delta_vs_greedy_soup"),
                "ci_low": item.get("paired_delta_vs_greedy_soup_ci_low"),
                "ci_high": item.get("paired_delta_vs_greedy_soup_ci_high"),
                "claim_reading": "not_supported_win" if safe_float(item.get("paired_delta_vs_greedy_soup_ci_low")) <= 0 else "supported_exact_setting_win",
            }
        )
        rows.append(
            {
                "regime": "independent_seed_rebasin",
                "comparison": "TwistedMerge selector vs C2M3",
                "method": "improved_validated_selector",
                "baseline": "c2m3_permutation",
                "paired_mean_delta": item.get("paired_delta_vs_c2m3"),
                "ci_low": item.get("paired_delta_vs_c2m3_ci_low"),
                "ci_high": item.get("paired_delta_vs_c2m3_ci_high"),
                "claim_reading": "supported_internal_c2m3_delta" if safe_float(item.get("paired_delta_vs_c2m3_ci_low")) > 0 else "descriptive_or_not_supported",
            }
        )
    for method in ["task_arithmetic", "ties_merging", "dare"]:
        item = shared[shared["method"].eq(method)]
        if item.empty:
            continue
        row = item.iloc[0]
        rows.append(
            {
                "regime": "shared_base_task_vector",
                "comparison": f"{row['display_name']} vs greedy soup",
                "method": method,
                "baseline": "greedy_soup",
                "paired_mean_delta": row.get("mean_delta_vs_greedy_soup"),
                "ci_low": row.get("min_delta_vs_greedy_ci_low"),
                "ci_high": row.get("max_delta_vs_greedy_ci_high"),
                "claim_reading": "mixed_exact_setting_task_vector" if safe_float(row.get("min_delta_vs_greedy_ci_low")) <= 0 else "supported_shared_base_delta",
            }
        )
    return pd.DataFrame(rows)


def build_claim_rows(independent: pd.DataFrame, shared: pd.DataFrame, coverage: pd.DataFrame) -> pd.DataFrame:
    tm = independent[independent["method"].eq("improved_validated_selector")]
    greedy = independent[independent["method"].eq("greedy_soup")]
    highest = "Not supported"
    reason = "The independent-seed MNIST MLP summary has greedy soup above the improved selector, and the shared-base task-vector benchmark is a separate regime."
    if not tm.empty and not greedy.empty and safe_float(tm.iloc[0].get("mean_test_accuracy")) > safe_float(greedy.iloc[0].get("mean_test_accuracy")):
        highest = "Partially supported"
        reason = "TwistedMerge is above greedy soup in the limited independent-seed aggregate, but this remains narrow."

    tm_cov = coverage[coverage["method"].eq("twistedmerge_selector")]
    best_cov = int(pd.to_numeric(coverage["qualitative_coverage_score"], errors="coerce").max())
    tm_score = int(pd.to_numeric(tm_cov["qualitative_coverage_score"], errors="coerce").iloc[0]) if not tm_cov.empty else -1
    well_rounded_status = "Supported" if tm_score == best_cov else "Partially supported"

    rows = [
        {
            "claim_id": "twistedmerge_highest_accuracy",
            "claim": "TwistedMerge is the highest-accuracy method among all compared methods.",
            "status": highest,
            "evidence": "reports/csv/method_family_comparison_independent.csv; reports/csv/method_family_comparison_shared_base.csv",
            "safe_wording": "Do not claim highest accuracy unless the exact paired comparison supports it.",
            "reason": reason,
        },
        {
            "claim_id": "twistedmerge_most_well_rounded",
            "claim": "TwistedMerge is the most well-rounded framework among the compared methods.",
            "status": well_rounded_status,
            "evidence": "reports/csv/method_family_structural_coverage.csv",
            "safe_wording": "TwistedMerge is the most well-rounded framework among the methods studied, while greedy soup remains a very strong pure-accuracy baseline.",
            "reason": "The coverage matrix marks TwistedMerge as the only compared family covering validation selection, gauge correction, cycle/holonomy diagnostics, central/projective obstruction detection, and conservative no-lift behavior. This is a qualitative structural statement, not an averaged metric leaderboard.",
        },
        {
            "claim_id": "greedy_soup_strong_accuracy_baseline",
            "claim": "Greedy soup remains a strong pure-accuracy baseline.",
            "status": "Supported",
            "evidence": "reports/csv/external_baseline_comparison_summary.csv; reports/csv/same_base_task_vector_extended_summary.csv",
            "safe_wording": "Greedy soup remains a strong pure-accuracy boundary baseline.",
            "reason": "Greedy soup is the top or near-top independent-seed pure-accuracy baseline, and task-vector methods are interpreted in a separate shared-base regime.",
        },
        {
            "claim_id": "task_vector_methods_fixed_trivialization",
            "claim": "Task Arithmetic, TIES, and DARE are best interpreted as fixed-trivialization/task-vector methods.",
            "status": "Supported",
            "evidence": "reports/same_base_task_vector_extended.md; reports/csv/method_family_comparison_shared_base.csv",
            "safe_wording": "Task Arithmetic/TIES/DARE are evaluated only in shared-base task-vector settings here.",
            "reason": "The report keeps these methods out of independent-seed rebasin claims and uses the common-base task-vector artifact for their accuracy rows.",
        },
        {
            "claim_id": "slerp_path_geometry_not_obstruction",
            "claim": "SLERP improves interpolation geometry inside a fixed parameter chart but does not address gauge/descent obstruction.",
            "status": "Partially supported",
            "evidence": "reports/slerp_barrier_geometry_report.md; reports/csv/method_family_structural_coverage.csv",
            "safe_wording": "SLERP is an internal path-geometry baseline in a fixed chart; it does not provide gauge synchronization or obstruction diagnostics.",
            "reason": "The existing SLERP audit is path-geometry evidence and did not show an average barrier reduction; the no-gauge/no-obstruction part is structural.",
        },
        {
            "claim_id": "c2m3_permutation_not_full_taxonomy",
            "claim": "C2M3 addresses permutation cycle consistency but not the full obstruction taxonomy.",
            "status": "Supported",
            "evidence": "reports/csv/method_family_structural_coverage.csv; reports/full_capacity_claim_audit.md",
            "safe_wording": "C2M3-style synchronization is the permutation cycle-consistency baseline, not a full residual taxonomy or period-index detector.",
            "reason": "The coverage matrix gives C2M3 permutation and cycle/holonomy coverage but not monomial scaling, central/projective detection, or full conservative lift rejection.",
        },
    ]
    return pd.DataFrame(rows)


def write_latex_table(df: pd.DataFrame, path: Path, columns: list[str], caption: str, label: str) -> None:
    def tex(value) -> str:
        text = str(value)
        for old, new in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")]:
            text = text.replace(old, new)
        return text

    lines = [f"\\begin{{table}}[t]", "\\centering", f"\\caption{{{tex(caption)}}}", f"\\label{{{tex(label)}}}"]
    lines.append("\\begin{tabular}{" + "l" * len(columns) + "}")
    lines.append("\\toprule")
    lines.append(" & ".join(tex(col) for col in columns) + r" \\")
    lines.append("\\midrule")
    for _, row in df[columns].iterrows():
        values = []
        for col in columns:
            val = row[col]
            if isinstance(val, float):
                val = fmt(val)
            values.append(tex(val))
        lines.append(" & ".join(values) + r" \\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def update_claims_audit(claims: pd.DataFrame, reports_dir: Path) -> None:
    path = reports_dir / "claims_audit.md"
    if not path.exists():
        return
    start = "<!-- method-family-comparison:start -->"
    end = "<!-- method-family-comparison:end -->"
    table_rows = claims[["claim_id", "status", "safe_wording", "reason"]].to_dict("records")
    block = [
        start,
        "## Method Family Comparison Appendix",
        "",
        "This section is generated by `experiments/method_family_comparison.py`. It is an appendix-level comparison across internal/fair-style method families, not an official external-baseline claim.",
        "",
        md_table(table_rows, ["claim_id", "status", "safe_wording", "reason"]),
        "",
        "Claim boundary: the supported wording is `TwistedMerge is the most well-rounded framework among the methods studied, while greedy soup remains a very strong pure-accuracy baseline.` Do not claim broad highest-accuracy or official implementation wins from this artifact.",
        end,
        "",
    ]
    text = path.read_text(encoding="utf-8")
    if start in text and end in text:
        before = text.split(start, 1)[0]
        after = text.split(end, 1)[1]
        text = before + "\n".join(block) + after
    else:
        text = text.rstrip() + "\n\n" + "\n".join(block)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_report(independent: pd.DataFrame, shared: pd.DataFrame, paired: pd.DataFrame, coverage: pd.DataFrame, claims: pd.DataFrame, reports_dir: Path) -> None:
    tm = independent[independent["method"].eq("improved_validated_selector")]
    tm_delta_greedy = fmt(tm.iloc[0]["paired_delta_vs_greedy_soup"]) if not tm.empty else ""
    tm_delta_c2m3 = fmt(tm.iloc[0]["paired_delta_vs_c2m3"]) if not tm.empty else ""
    report = f"""# Method Family Comparison Appendix

Generated by `experiments/method_family_comparison.py`.

## Executive Summary

This lightweight appendix aggregates existing in-repository MNIST/Fashion-MNIST MLP artifacts rather than running a new large benchmark. It compares method families mentioned in the introduction: weight averaging, Model Soups/greedy soup, C2M3-style synchronization, TwistedMerge/TwistedMerge++ selectors, internal SLERP-style baselines, Task Arithmetic, TIES-style merging, and DARE-style merging.

Main conclusion: TwistedMerge is the most well-rounded framework among the methods studied, while greedy soup remains a very strong pure-accuracy baseline. The highest-accuracy claim is not supported by this appendix.

## Exact Command

```bash
PYTHONPYCACHEPREFIX=/private/tmp/codex-pycache .venv/bin/python experiments/method_family_comparison.py --reports-dir reports
```

## Experimental Setup

- Independent-seed/rebasin evidence is read from `reports/csv/external_baseline_comparison_summary.csv` and the fixed-independent SLERP path rows in `reports/csv/slerp_barrier_geometry.csv`.
- Shared-base task-vector evidence is read from `reports/csv/same_base_task_vector_extended_summary.csv`.
- SLERP, Task Arithmetic, TIES-style, and DARE-style vector mechanics are implemented in `src/method_family_baselines.py` and covered by unit tests.
- Task Arithmetic/TIES/DARE are not applied as primary methods to independently initialized models.
- All external-method rows are internal or faithful-style implementations unless the source report explicitly says official code was run.
- Runtime is not remeasured here because this appendix aggregates existing artifacts; original source reports/configs should be used for training/evaluation runtime details.

## Independent-Seed Accuracy Table

{md_table(independent.to_dict('records'), ['method', 'display_name', 'status', 'n_rows', 'n_seeds', 'mean_test_accuracy', 'paired_delta_vs_greedy_soup', 'paired_delta_vs_c2m3', 'validation_selected_hyperparameters', 'source_csv'])}

## Shared-Base Task-Vector Accuracy Table

{md_table(shared.to_dict('records'), ['method', 'display_name', 'status', 'n_settings', 'n_unique_seeds_min', 'mean_test_accuracy_across_settings', 'mean_delta_vs_greedy_soup', 'min_delta_vs_greedy_ci_low', 'validation_selected_hyperparameters', 'source_csv'])}

## Paired Comparisons

{md_table(paired.to_dict('records'), ['regime', 'comparison', 'paired_mean_delta', 'ci_low', 'ci_high', 'claim_reading'])}

For the independent-seed MNIST MLP summary, the TwistedMerge improved selector has paired mean delta `{tm_delta_greedy}` versus greedy soup and `{tm_delta_c2m3}` versus internal C2M3. This supports the internal-C2M3 comparison but not a greedy-soup win.

## Structural Coverage Matrix

The qualitative coverage score below is a coverage count, not an accuracy score and not a paper leaderboard.

{md_table(coverage.to_dict('records'), ['method', 'validation_selection', 'pairwise_gauge_synchronization', 'permutation_gauge_handling', 'monomial_relu_scaling_gauge_handling', 'coordinatewise_sign_or_sparsity', 'cycle_holonomy_diagnostic', 'central_projective_obstruction_detection', 'conservative_rejection_no_lift', 'common_base_required', 'qualitative_coverage_score'])}

## Claim Audit

{md_table(claims.to_dict('records'), ['claim_id', 'status', 'safe_wording', 'reason'])}

## Claim Boundaries

- Do not claim TwistedMerge is the highest-accuracy method among all compared methods.
- Do not claim broad foundation-model or broad vision results from this MNIST/Fashion-MNIST MLP appendix.
- Do not claim official implementations for SLERP, C2M3, Model Soups, Task Arithmetic, TIES, or DARE unless a separate official-code run exists.
- Do not claim TwistedMerge beats greedy soup unless an exact paired comparison has a positive confidence interval.
- Task Arithmetic, TIES, and DARE rows belong to the shared-base/fixed-trivialization regime.
"""
    (reports_dir / "method_family_comparison_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--no-update-claims-audit", action="store_true")
    args = parser.parse_args()

    reports_dir = args.reports_dir
    csv_dir = reports_dir / "csv"
    table_dir = reports_dir / "tables"
    csv_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    independent = build_independent_summary(reports_dir)
    shared = build_shared_base_summary(reports_dir)
    coverage = pd.DataFrame(structural_coverage_matrix())
    paired = build_paired_rows(independent, shared)
    claims = build_claim_rows(independent, shared, coverage)

    independent.to_csv(csv_dir / "method_family_comparison_independent.csv", index=False, lineterminator="\n")
    shared.to_csv(csv_dir / "method_family_comparison_shared_base.csv", index=False, lineterminator="\n")
    paired.to_csv(csv_dir / "method_family_comparison_paired.csv", index=False, lineterminator="\n")
    coverage.to_csv(csv_dir / "method_family_structural_coverage.csv", index=False, lineterminator="\n")
    claims.to_csv(csv_dir / "method_family_claim_audit.csv", index=False, lineterminator="\n")

    write_latex_table(
        coverage,
        table_dir / "method_family_structural_coverage.tex",
        ["method", "validation_selection", "pairwise_gauge_synchronization", "permutation_gauge_handling", "monomial_relu_scaling_gauge_handling", "coordinatewise_sign_or_sparsity", "cycle_holonomy_diagnostic", "central_projective_obstruction_detection", "conservative_rejection_no_lift"],
        "Qualitative structural coverage by method family.",
        "tab:method-family-coverage",
    )
    write_report(independent, shared, paired, coverage, claims, reports_dir)
    if not args.no_update_claims_audit:
        update_claims_audit(claims, reports_dir)

    print("wrote reports/method_family_comparison_report.md")
    print("wrote reports/csv/method_family_comparison_independent.csv")
    print("wrote reports/csv/method_family_comparison_shared_base.csv")
    print("wrote reports/csv/method_family_structural_coverage.csv")


if __name__ == "__main__":
    main()
