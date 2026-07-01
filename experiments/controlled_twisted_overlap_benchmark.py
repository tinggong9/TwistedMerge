#!/usr/bin/env python
"""Controlled neural overlap benchmark with known central twists."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.controlled_twisted_overlaps import (  # noqa: E402
    EXTRA_CONTROL_ALIASES,
    METHODS,
    bootstrap_mean_ci,
    build_controlled_case,
    defect_rows_for_case,
    evaluate_methods,
    pairwise_rows_for_case,
    save_local_checkpoints,
)
from src.metrics import capture_environment, save_json  # noqa: E402


def parse_csv(text: str, cast=str) -> list:
    return [cast(item.strip()) for item in str(text).split(",") if item.strip()]


def parse_seeds(text: str) -> list[int]:
    text = str(text).strip()
    if "," in text:
        return parse_csv(text, int)
    if "-" in text:
        start, end = text.split("-", 1)
        start_i, end_i = int(start), int(end)
        step = 1 if end_i >= start_i else -1
        return list(range(start_i, end_i + step, step))
    if ":" in text:
        start, end = text.split(":", 1)
        start_i, end_i = int(start), int(end)
        step = 1 if end_i >= start_i else -1
        return list(range(start_i, end_i + step, step))
    return [int(text)]


def json_compact(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def parse_extra_controls(text: str) -> tuple[str, ...]:
    controls = []
    for item in parse_csv(text, str):
        if item not in EXTRA_CONTROL_ALIASES:
            raise ValueError(f"unknown extra control: {item}")
        controls.append(EXTRA_CONTROL_ALIASES[item])
    return tuple(dict.fromkeys(controls))


def family_numeric_summary(case, triangle_rows: list[dict]) -> dict[str, float | int | bool | str]:
    centrality = np.asarray([float(row["centrality_residual"]) for row in triangle_rows], dtype=float)
    defect = np.asarray([float(row["defect_to_true_twist_residual"]) for row in triangle_rows], dtype=float)
    alpha_signs = [int(row["true_alpha_sign"]) for row in triangle_rows if int(row["true_alpha_sign"]) != 0]
    h2_product = int(np.prod(alpha_signs)) if alpha_signs else 0
    central_negative_faces = sum(1 for sign in alpha_signs if sign < 0)
    if case.is_coboundary is None:
        coboundary_residual = float("nan")
    else:
        coboundary_residual = 0.0 if case.is_coboundary else 1.0
    return {
        "centrality_residual_mean": float(centrality.mean()) if len(centrality) else float("nan"),
        "defect_to_true_twist_residual_mean": float(defect.mean()) if len(defect) else float("nan"),
        "def_c_residual": float(defect.mean()) if len(defect) else float("nan"),
        "coboundary_residual": coboundary_residual,
        "h2_product_sign": h2_product,
        "central_negative_faces": central_negative_faces,
        "central_twist_claim_allowed": bool(case.central_twist_claim_allowed),
        "is_coboundary": case.is_coboundary if case.is_coboundary is not None else "",
    }


def run_case(args, family: str, width: int, seed: int):
    case = build_controlled_case(
        family=family,
        width=width,
        n_models=args.n_models,
        seed=seed,
        samples_per_chart=args.samples_per_chart,
        samples_per_overlap=args.samples_per_overlap,
        branch_count=args.branch_count,
    )
    checkpoint_rows = save_local_checkpoints(case, args.reports_dir / "checkpoints" / "controlled_twisted_overlap")
    triangles = defect_rows_for_case(case)
    family_summary = family_numeric_summary(case, triangles)
    method_rows = evaluate_methods(case, args.extra_controls_parsed)
    pairwise = pairwise_rows_for_case(case)
    base = {
        "family": family,
        "seed": seed,
        "width": width,
        "n_models": args.n_models,
        "epochs": args.epochs,
        "samples_per_chart": args.samples_per_chart,
        "samples_per_overlap": args.samples_per_overlap,
        "branch_count": args.branch_count,
        "overlap_ids_json": json_compact({"-".join(map(str, face)): overlap_id for face, overlap_id in case.overlap_ids.items()}),
        "true_alpha_json": json_compact({"-".join(map(str, face)): int(sign) for face, sign in case.alpha_signs.items()}),
        "target_sign_json": json_compact({"-".join(map(str, face)): int(sign) for face, sign in case.target_signs.items()}),
        "checkpoint_metadata_json": json_compact(checkpoint_rows),
        "controlled_evidence_type": "exact_constructed_neural_overlap",
        "real_model_evidence": False,
        "notes": case.notes,
        **family_summary,
    }
    rows = []
    for row in method_rows:
        payload = {**base, **row}
        payload["branch_assignment_json"] = json_compact(payload.pop("branch_assignment"))
        if isinstance(payload.get("router_branch_scores_json"), dict):
            payload["router_branch_scores_json"] = json_compact(payload["router_branch_scores_json"])
        rows.append(payload)
    pairwise_rows = [{**base, **row} for row in pairwise]
    triangle_rows = [{**base, **row} for row in triangles]
    return rows, pairwise_rows, triangle_rows


def summarize(df: pd.DataFrame, bootstrap_samples: int) -> pd.DataFrame:
    rng = np.random.default_rng(24681357)
    rows = []
    group_cols = ["family", "width", "method"]
    for key, group in df.groupby(group_cols, dropna=False):
        meta = dict(zip(group_cols, key))
        acc = pd.to_numeric(group["test_accuracy"], errors="coerce").to_numpy()
        loss = pd.to_numeric(group["test_loss"], errors="coerce").to_numpy()
        acc_low, acc_high = bootstrap_mean_ci(acc, bootstrap_samples, rng)
        method = str(meta["method"])
        if bool(group.get("is_extra_control", pd.Series([False])).fillna(False).astype(bool).any()):
            status = "control_method_descriptive"
        elif method == "learned_context_router":
            status = "learned_router_diagnostic"
        else:
            status = "method_descriptive"
        rows.append(
            {
                "summary_type": "method",
                **meta,
                "comparison": "",
                "n_rows": int(len(group)),
                "n_unique_seeds": int(group["seed"].nunique()),
                "mean_test_accuracy": float(np.nanmean(acc)),
                "test_accuracy_ci_low": acc_low,
                "test_accuracy_ci_high": acc_high,
                "mean_test_loss": float(np.nanmean(loss)),
                "mean_delta": float("nan"),
                "delta_ci_low": float("nan"),
                "delta_ci_high": float("nan"),
                "claim_status": status,
            }
        )
    comparisons = [
        ("twisted_q2_branch", "ordinary_weight_average"),
        ("twisted_q2_branch", "c2m3_synchronized"),
        ("twisted_q2_branch", "random_branch_ensemble"),
        ("twisted_q2_branch", "validation_selected_branch_ensemble"),
        ("twisted_q2_branch", "c2m3_cluster_branch_ensemble"),
        ("twisted_q2_branch", "wrong_twist_control"),
        ("twisted_q2_branch", "wrong_context_control"),
        ("twisted_q2_branch", "learned_context_router"),
        ("twisted_q2_branch", "distilled_twisted_single_model"),
        ("twisted_q2_branch", "parameter_matched_wide_control"),
        ("twisted_q2_branch", "no_twist_branch_control"),
        ("c2m3_synchronized", "ordinary_weight_average"),
    ]
    for (family, width), group in df.groupby(["family", "width"], dropna=False):
        pivot = group.pivot_table(index="seed", columns="method", values="test_accuracy", aggfunc="mean")
        for left, right in comparisons:
            if left not in pivot.columns or right not in pivot.columns:
                continue
            delta = (pivot[left] - pivot[right]).dropna().to_numpy(dtype=float)
            low, high = bootstrap_mean_ci(delta, bootstrap_samples, rng)
            mean_delta = float(np.nanmean(delta)) if len(delta) else float("nan")
            comparison = f"{left}_vs_{right}"
            if family == "random_noncentral":
                status = "noncentral_control_not_promoted"
            elif family == "mu2_coboundary" and comparison == "c2m3_synchronized_vs_ordinary_weight_average":
                status = "supported_coboundary_sync" if mean_delta > 0 and low > 0 else "descriptive"
            elif family == "mu2_nontrivial_h2" and left == "twisted_q2_branch" and right in {
                "random_branch_ensemble",
                "validation_selected_branch_ensemble",
                "c2m3_cluster_branch_ensemble",
                "c2m3_synchronized",
            }:
                status = "supported_controlled_rank_lift" if mean_delta > 0 and low > 0 else "descriptive"
            elif family == "mu2_nontrivial_h2" and right in {"wrong_twist_control", "wrong_context_control"}:
                status = "supported_q2_beats_wrong_control" if mean_delta > 0 and low > 0 else "wrong_control_not_beaten"
            elif family == "mu2_nontrivial_h2" and right == "no_twist_branch_control":
                status = "supported_q2_beats_no_twist_branch" if mean_delta > 0 and low > 0 else "no_twist_branch_matches"
            elif family == "mu2_nontrivial_h2" and right == "learned_context_router":
                status = "learned_router_matches_supplied_context" if abs(mean_delta) <= 1e-12 else "learned_router_gap"
            elif family == "mu2_nontrivial_h2" and right == "distilled_twisted_single_model":
                status = "distillation_failed_branch_remains_extra_capacity" if mean_delta > 0 and low > 0 else "distillation_matches_single_model"
            elif family == "mu2_nontrivial_h2" and right == "parameter_matched_wide_control":
                status = "supported_not_explained_by_parameter_matched_wide" if mean_delta > 0 and low > 0 else "wide_control_matches_weaken_to_charted_representation"
            else:
                status = "descriptive"
            rows.append(
                {
                    "summary_type": "paired_delta",
                    "family": family,
                    "width": width,
                    "method": left,
                    "comparison": comparison,
                    "n_rows": int(len(delta)),
                    "n_unique_seeds": int(len(delta)),
                    "mean_test_accuracy": float("nan"),
                    "test_accuracy_ci_low": float("nan"),
                    "test_accuracy_ci_high": float("nan"),
                    "mean_test_loss": float("nan"),
                    "mean_delta": mean_delta,
                    "delta_ci_low": low,
                    "delta_ci_high": high,
                    "claim_status": status,
                }
            )
    return pd.DataFrame(rows)


def md_table(df: pd.DataFrame, columns: list[str], max_rows: int = 40) -> str:
    if df.empty:
        return "_No rows._"
    part = df.head(max_rows).copy()
    for col in columns:
        if col not in part:
            part[col] = ""
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in part.iterrows():
        values = []
        for col in columns:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.4f}" if np.isfinite(value) else "nan")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_latex_table(summary: pd.DataFrame, path: Path) -> None:
    method = summary[(summary["summary_type"] == "method") & (summary["method"].isin(["ordinary_weight_average", "c2m3_synchronized", "twisted_q2_branch", "validation_selected_branch_ensemble"]))].copy()
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "\\begin{tabular}{llrrr}",
        "\\toprule",
        "family & method & width & mean acc. & claim status\\\\",
        "\\midrule",
    ]
    for _, row in method.sort_values(["family", "width", "method"]).iterrows():
        lines.append(
            f"{row['family']} & {row['method']} & {int(row['width'])} & "
            f"{float(row['mean_test_accuracy']):.3f} & {row['claim_status']}\\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_defect_vs_merge_loss(df: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    data = df[df["method"].isin(["ordinary_weight_average", "c2m3_synchronized", "twisted_q2_branch"])].copy()
    data["merge_loss"] = 1.0 - pd.to_numeric(data["test_accuracy"], errors="coerce")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for (family, method), group in data.groupby(["family", "method"]):
        ax.scatter(
            group["coboundary_residual"].fillna(group["centrality_residual_mean"]),
            group["merge_loss"],
            s=30,
            alpha=0.72,
            label=f"{family} {method}",
        )
    ax.set_xlabel("coboundary residual (central rows) or centrality residual (noncentral control)")
    ax.set_ylabel("test merge loss")
    ax.set_title("Controlled defect residual versus merge loss")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_rank_lift_delta(summary: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    data = summary[
        (summary["summary_type"] == "paired_delta")
        & summary["comparison"].isin(
            [
                "twisted_q2_branch_vs_random_branch_ensemble",
                "twisted_q2_branch_vs_validation_selected_branch_ensemble",
                "twisted_q2_branch_vs_c2m3_cluster_branch_ensemble",
                "twisted_q2_branch_vs_c2m3_synchronized",
            ]
        )
    ].copy()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    if data.empty:
        ax.text(0.5, 0.5, "No paired deltas", ha="center", va="center")
    else:
        data = data.sort_values(["family", "width", "comparison"])
        labels = [f"{row.family}\nW{int(row.width)}\n{row.comparison.replace('twisted_q2_branch_vs_', '')}" for row in data.itertuples()]
        x = np.arange(len(data))
        y = data["mean_delta"].astype(float)
        low = y - data["delta_ci_low"].astype(float)
        high = data["delta_ci_high"].astype(float) - y
        ax.bar(x, y, color="tab:purple", alpha=0.75)
        ax.errorbar(x, y, yerr=[low, high], fmt="none", ecolor="black", capsize=3)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("q=2 branch accuracy delta")
        ax.set_title("Rank-lift branch delta versus capacity-matched baselines")
        ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def hardening_gate_text(summary: pd.DataFrame) -> str:
    nontrivial = summary[(summary["summary_type"] == "paired_delta") & (summary["family"] == "mu2_nontrivial_h2")].copy()
    if nontrivial.empty:
        return "No nontrivial `mu2_nontrivial_h2` hardening rows were produced."
    statuses = set(nontrivial["claim_status"].astype(str))
    wrong_controls = nontrivial[
        nontrivial["comparison"].isin(
            [
                "twisted_q2_branch_vs_wrong_twist_control",
                "twisted_q2_branch_vs_wrong_context_control",
            ]
        )
    ]
    wrong_ok = not wrong_controls.empty and bool((wrong_controls["claim_status"] == "supported_q2_beats_wrong_control").all())
    router = nontrivial[nontrivial["comparison"] == "twisted_q2_branch_vs_learned_context_router"]
    router_text = (
        "The learned validation-only context router matches the supplied-context q=2 branch on held-out overlap samples."
        if not router.empty and bool((router["claim_status"] == "learned_router_matches_supplied_context").all())
        else "The learned validation-only context router does not fully match the supplied-context q=2 branch."
        if not router.empty
        else "The learned validation-only context router was not run."
    )
    distill = nontrivial[nontrivial["comparison"] == "twisted_q2_branch_vs_distilled_twisted_single_model"]
    distill_text = (
        "Distillation into a single context-free model fails to match the q=2 branch, so the branch result remains extra-capacity/charted."
        if not distill.empty and bool((distill["claim_status"] == "distillation_failed_branch_remains_extra_capacity").all())
        else "Distillation matches the q=2 branch in at least one setting, so no extra-capacity branch advantage is claimed there."
        if not distill.empty
        else "The distilled single-model control was not run."
    )
    wide = nontrivial[nontrivial["comparison"] == "twisted_q2_branch_vs_parameter_matched_wide_control"]
    wide_text = (
        "The q=2 branch beats the parameter-matched wide ordinary control, so this run is not explained by ordinary width alone."
        if not wide.empty and bool((wide["claim_status"] == "supported_not_explained_by_parameter_matched_wide").all())
        else "The parameter-matched wide control matches the q=2 branch in at least one setting; weaken to a charted-representation claim there."
        if not wide.empty
        else "The parameter-matched wide control was not run."
    )
    wrong_text = (
        "The q=2 branch beats both wrong-twist and wrong-context controls in every nontrivial h2 setting."
        if wrong_ok
        else "The q=2 branch does not beat every wrong-twist/wrong-context control; treat supplied-context results as descriptive."
    )
    no_twist = nontrivial[nontrivial["comparison"] == "twisted_q2_branch_vs_no_twist_branch_control"]
    no_twist_text = (
        "The q=2 branch also beats the same-branch-count no-twist control."
        if not no_twist.empty and bool((no_twist["claim_status"] == "supported_q2_beats_no_twist_branch").all())
        else "The no-twist branch control was not beaten in every setting."
        if not no_twist.empty
        else "The no-twist branch control was not run."
    )
    return "\n".join(
        [
            f"- {wrong_text}",
            f"- {router_text}",
            f"- {distill_text}",
            f"- {wide_text}",
            f"- {no_twist_text}",
            f"- Status labels present: `{', '.join(sorted(statuses))}`.",
        ]
    )


def write_report(
    args,
    runs: pd.DataFrame,
    summary: pd.DataFrame,
    pairwise: pd.DataFrame,
    triangles: pd.DataFrame,
    controls: pd.DataFrame,
    report_path: Path,
) -> None:
    method_cols = [
        "family",
        "width",
        "method",
        "n_rows",
        "mean_test_accuracy",
        "test_accuracy_ci_low",
        "test_accuracy_ci_high",
        "mean_test_loss",
    ]
    delta_cols = ["family", "width", "comparison", "n_rows", "mean_delta", "delta_ci_low", "delta_ci_high", "claim_status"]
    method_summary = summary[summary["summary_type"] == "method"].copy()
    deltas = summary[summary["summary_type"] == "paired_delta"].copy()
    control_summary = summary[
        (summary["summary_type"] == "method")
        & (summary["method"].astype(str).isin(controls["method"].astype(str).unique() if not controls.empty else []))
    ].copy()
    local_exact = runs.groupby(["family", "width"])["local_model_accuracy"].mean().reset_index()
    pairwise_exact = pairwise.groupby(["family", "width"])["pairwise_alignment_residual"].mean().reset_index()
    triangle_exact = (
        triangles.groupby(["family", "width"])[["defect_to_true_twist_residual_mean", "centrality_residual_mean"]]
        .mean()
        .reset_index()
    )
    exact_rows = local_exact.merge(pairwise_exact, on=["family", "width"], how="left").merge(
        triangle_exact, on=["family", "width"], how="left"
    )
    report = f"""# Controlled Twisted-Overlap Benchmark

This report is generated by `experiments/controlled_twisted_overlap_benchmark.py`.

## Exact Command

```bash
{args.command_string}
```

## Construction

The benchmark uses exact one-hidden-layer ReLU MLPs with paired hidden units `ReLU(z), ReLU(-z)`. The central `mu_2` element is an exact hidden-unit swap, so pairwise overlap alignments are exact neural symmetries. Triangle contexts are indexed by the four faces of the tetrahedral sphere.

This is controlled obstruction evidence, not real-model evidence. It is deliberately separate from MNIST/Fashion/CIFAR model-merging experiments.

## Outputs

- `reports/csv/controlled_twisted_overlap.csv`
- `reports/csv/controlled_twisted_overlap_pairwise.csv`
- `reports/csv/controlled_twisted_overlap_triangles.csv`
- `reports/csv/controlled_twisted_overlap_summary.csv`
- `reports/csv/controlled_twisted_overlap_controls.csv`
- `reports/tables/controlled_twisted_overlap_table.tex`
- `reports/plots/controlled_twisted_overlap_defect_vs_merge_loss.pdf`
- `reports/plots/controlled_twisted_overlap_rank_lift_delta.pdf`
- `reports/configs/controlled_twisted_overlap_config.json`

## Exactness Checks

{md_table(exact_rows, ["family", "width", "local_model_accuracy", "pairwise_alignment_residual", "defect_to_true_twist_residual_mean", "centrality_residual_mean"], 20)}

## Method Summary

{md_table(method_summary, method_cols, 80)}

## Paired Deltas

{md_table(deltas, delta_cols, 80)}

## Hardening Controls

{hardening_gate_text(summary)}

{md_table(control_summary, method_cols, 80)}

## Claim Boundaries

- Coboundary rows support the controlled claim that cycle-consistent synchronization can absorb an edge-coboundary central sign without a branch lift.
- Nontrivial `mu2_nontrivial_h2` rows support only a controlled branch-prediction claim. Supplied-context q=2 branch results are reported separately from validation-learned router results.
- If the distilled single model or parameter-matched wide control matches q=2 in a setting, the corresponding claim is weakened to a charted-representation claim for that setting.
- Random noncentral rows are negative controls and are not promoted to central-twist or Brauer/projective claims.
- This benchmark does not claim that natural MNIST/Fashion/CIFAR model merging has the same obstruction.

## Row Counts

- Main rows: `{len(runs)}`
- Control rows: `{len(controls)}`
- Pairwise rows: `{len(pairwise)}`
- Triangle rows: `{len(triangles)}`

## Environment

```json
{json.dumps(capture_environment(), indent=2)}
```
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--twist-family", default="mu2_coboundary,mu2_nontrivial_h2,random_noncentral")
    parser.add_argument("--n-models", type=int, default=4)
    parser.add_argument("--widths", default="32,64")
    parser.add_argument("--seeds", default="5000-5029")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--samples-per-chart", type=int, default=2000)
    parser.add_argument("--samples-per-overlap", type=int, default=1000)
    parser.add_argument("--branch-count", type=int, default=2)
    parser.add_argument("--extra-controls", default="")
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "reports")
    args = parser.parse_args()
    args.command_string = " ".join([sys.executable, *sys.argv])
    args.extra_controls_parsed = parse_extra_controls(args.extra_controls)

    families = parse_csv(args.twist_family, str)
    widths = parse_csv(args.widths, int)
    seeds = parse_seeds(args.seeds)

    all_rows = []
    all_pairwise = []
    all_triangles = []
    for family in families:
        for width in widths:
            for seed in seeds:
                print(f"running family={family} width={width} seed={seed}", flush=True)
                rows, pairwise_rows, triangle_rows = run_case(args, family, width, seed)
                all_rows.extend(rows)
                all_pairwise.extend(pairwise_rows)
                all_triangles.extend(triangle_rows)

    runs = pd.DataFrame(all_rows)
    pairwise = pd.DataFrame(all_pairwise)
    triangles = pd.DataFrame(all_triangles)
    controls = runs[runs["is_extra_control"].fillna(False).astype(bool)].copy() if "is_extra_control" in runs else pd.DataFrame()
    summary = summarize(runs, args.bootstrap_samples)

    csv_dir = args.reports_dir / "csv"
    plot_dir = args.reports_dir / "plots"
    table_dir = args.reports_dir / "tables"
    config_dir = args.reports_dir / "configs"
    csv_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    runs_path = csv_dir / "controlled_twisted_overlap.csv"
    pairwise_path = csv_dir / "controlled_twisted_overlap_pairwise.csv"
    triangles_path = csv_dir / "controlled_twisted_overlap_triangles.csv"
    summary_path = csv_dir / "controlled_twisted_overlap_summary.csv"
    controls_path = csv_dir / "controlled_twisted_overlap_controls.csv"
    runs.to_csv(runs_path, index=False, lineterminator="\n")
    pairwise.to_csv(pairwise_path, index=False, lineterminator="\n")
    triangles.to_csv(triangles_path, index=False, lineterminator="\n")
    summary.to_csv(summary_path, index=False, lineterminator="\n")
    controls.to_csv(controls_path, index=False, lineterminator="\n")
    write_latex_table(summary, table_dir / "controlled_twisted_overlap_table.tex")
    plot_defect_vs_merge_loss(runs, plot_dir / "controlled_twisted_overlap_defect_vs_merge_loss.pdf")
    plot_rank_lift_delta(summary, plot_dir / "controlled_twisted_overlap_rank_lift_delta.pdf")
    write_report(args, runs, summary, pairwise, triangles, controls, args.reports_dir / "controlled_twisted_overlap_report.md")
    save_json(
        config_dir / "controlled_twisted_overlap_config.json",
        {
            "argv": sys.argv,
            "families": families,
            "widths": widths,
            "seeds": seeds,
            "extra_controls": list(args.extra_controls_parsed),
            "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "environment": capture_environment(),
        },
    )
    print(f"wrote {runs_path}")
    print(f"wrote {pairwise_path}")
    print(f"wrote {triangles_path}")
    print(f"wrote {summary_path}")
    print(f"wrote {controls_path}")
    print(f"wrote {args.reports_dir / 'controlled_twisted_overlap_report.md'}")


if __name__ == "__main__":
    main()
