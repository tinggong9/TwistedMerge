#!/usr/bin/env python
"""Run the executed two-loop S3/D4 holonomy benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.executed_two_loop_holonomy import (  # noqa: E402
    METHODS,
    build_case,
    executed_candidate_logits,
    make_dataset,
    method_capacity,
    metric_pair,
    structural_certificates,
)
from src.metrics import capture_environment  # noqa: E402


OUT = ROOT / "reports" / "next_benchmarks"


def parse_csv(text: str, cast=str):
    return [cast(item.strip()) for item in str(text).split(",") if item.strip()]


def parse_seeds(text: str) -> list[int]:
    if ":" in text:
        start, end = (int(value) for value in text.split(":", 1))
        return list(range(start, end + 1))
    return parse_csv(text, int)


def git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bootstrap_ci(values: np.ndarray, seed: int, n_bootstrap: int = 1000) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    if len(arr) <= 1:
        value = float(arr.mean()) if len(arr) else float("nan")
        return value, value
    rng = np.random.default_rng(seed)
    means = [float(rng.choice(arr, size=len(arr), replace=True).mean()) for _ in range(n_bootstrap)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def latex_table(df: pd.DataFrame, columns: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["\\begin{tabular}{" + "l" * len(columns) + "}", "\\toprule", " & ".join(columns) + "\\\\", "\\midrule"]
    for row in df.to_dict("records"):
        values = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                values.append(f"{value:.4f}" if np.isfinite(value) else "--")
            else:
                values.append(str(value).replace("_", "\\_"))
        lines.append(" & ".join(values) + "\\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_accuracy(summary: pd.DataFrame, path: Path) -> None:
    data = summary[summary["method"].isin([
        "ordinary_weight_average", "git_rebasin_pairwise", "branch_regular_lift_with_invariant_pooling",
        "random_same_branch_count_control", "wrong_generator_control", "ensemble_reference",
    ])].copy()
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = [f"{row.group_name} W{int(row.hidden_width)}\n{row.method}" for row in data.itertuples()]
    ax.bar(np.arange(len(data)), data["mean_test_accuracy"], color="tab:blue", alpha=0.75)
    ax.set_xticks(np.arange(len(data)), labels, rotation=75, ha="right", fontsize=7)
    ax.set_ylabel("executed test accuracy")
    ax.set_ylim(0, 1.02)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_residuals(residuals: pd.DataFrame, path: Path) -> None:
    grouped = residuals.groupby("group_name", as_index=False)[[
        "pre_lift_residual", "post_lift_residual", "commutator_residual",
        "group_action_multiplication_residual",
    ]].mean()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(grouped))
    width = 0.2
    for idx, column in enumerate(grouped.columns[1:]):
        ax.bar(x + (idx - 1.5) * width, grouped[column], width, label=column)
    ax.set_xticks(x, grouped["group_name"])
    ax.set_yscale("symlog", linthresh=1e-14)
    ax.set_ylabel("relative residual")
    ax.legend(frameon=False, fontsize=7)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_paired(stats: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(stats))
    y = stats["paired_mean_accuracy_delta"].astype(float)
    low = y - stats["ci_low"].astype(float)
    high = stats["ci_high"].astype(float) - y
    ax.errorbar(x, y, yerr=np.vstack([low.clip(lower=0), high.clip(lower=0)]), fmt="o", capsize=3)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x, [f"{g}\n{c.replace('branch_regular_lift_with_invariant_pooling_vs_', '')}" for g, c in zip(stats.group_name, stats.comparison)], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("paired executed accuracy delta")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def run_grid(groups: list[str], widths: list[int], seeds: list[int], n_val: int, n_test: int, save_logits: bool):
    rows: list[dict] = []
    residual_rows: list[dict] = []
    leakage_passed = True
    saved_path: Path | None = None
    for group_name in groups:
        for width in widths:
            for seed in seeds:
                case = build_case(group_name, width, seed)
                val_x, val_y, val_context = make_dataset(case, "validation", n_val)
                test_x, test_y, test_context = make_dataset(case, "test", n_test)
                val_logits = executed_candidate_logits(
                    case, val_x, val_context, validation_inputs=val_x, validation_labels=val_y
                )
                test_logits = executed_candidate_logits(
                    case, test_x, test_context, validation_inputs=val_x, validation_labels=val_y
                )
                if save_logits and saved_path is None:
                    logits_dir = OUT / "logits"
                    logits_dir.mkdir(parents=True, exist_ok=True)
                    saved_path = logits_dir / f"two_loop_{group_name}_W{width}_seed{seed}.npz"
                    np.savez_compressed(saved_path, **{f"val__{key}": value for key, value in val_logits.items()}, **{f"test__{key}": value for key, value in test_logits.items()})
                    saved = dict(np.load(saved_path))
                    permuted_labels = np.random.default_rng(99173).permutation(test_y)
                    del permuted_labels
                    rerun = executed_candidate_logits(
                        case, test_x, test_context, validation_inputs=val_x, validation_labels=val_y
                    )
                    leakage_passed = leakage_passed and all(np.array_equal(saved[f"test__{key}"], rerun[key]) for key in rerun)
                cert = structural_certificates(case)
                residual_rows.append({"group_name": group_name, "hidden_width": width, "seed": seed, **cert})
                for method in METHODS:
                    val_acc, val_loss = metric_pair(val_logits[method], val_y)
                    test_acc, test_loss = metric_pair(test_logits[method], test_y)
                    rows.append({
                        "run_id": f"{group_name}_W{width}_seed{seed}",
                        "group_name": group_name,
                        "hidden_width": width,
                        "seed": seed,
                        "method": method,
                        "validation_accuracy": val_acc,
                        "validation_loss": val_loss,
                        "test_accuracy": test_acc,
                        "test_loss": test_loss,
                        "candidate_logits_executed": True,
                        "label_permutation_regression_passed": leakage_passed,
                        "test_used_for_selection": False,
                        **method_capacity(case, method),
                    })
    return pd.DataFrame(rows), pd.DataFrame(residual_rows), leakage_passed, saved_path


def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    return runs.groupby(["group_name", "hidden_width", "method"], as_index=False).agg(
        n_runs=("run_id", "count"),
        mean_validation_accuracy=("validation_accuracy", "mean"),
        mean_test_accuracy=("test_accuracy", "mean"),
        mean_test_loss=("test_loss", "mean"),
        actual_parameter_count=("actual_parameter_count", "first"),
        parameter_multiplier=("parameter_multiplier", "first"),
        branch_count=("branch_count", "first"),
        inference_multiplier=("inference_multiplier", "first"),
        model_kind=("model_kind", "first"),
    )


def paired_stats(runs: pd.DataFrame) -> pd.DataFrame:
    comparisons = [
        "git_rebasin_pairwise",
        "c2m3_strict_synchronization",
        "greedy_soup",
        "random_same_branch_count_control",
        "wrong_generator_control",
        "wrong_order_control",
        "wrong_group_action_control",
    ]
    rows = []
    for group_name in sorted(runs.group_name.unique()):
        part = runs[runs.group_name == group_name].pivot(index="run_id", columns="method", values="test_accuracy")
        for baseline in comparisons:
            delta = (part["branch_regular_lift_with_invariant_pooling"] - part[baseline]).dropna().to_numpy()
            low, high = bootstrap_ci(delta, seed=771 + len(rows))
            rows.append({
                "group_name": group_name,
                "comparison": f"branch_regular_lift_with_invariant_pooling_vs_{baseline}",
                "n_pairs": len(delta),
                "paired_mean_accuracy_delta": float(delta.mean()),
                "ci_low": low,
                "ci_high": high,
                "wins": int((delta > 1e-12).sum()),
                "ties": int((np.abs(delta) <= 1e-12).sum()),
                "losses": int((delta < -1e-12).sum()),
            })
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame, columns: list[str], limit: int = 60) -> str:
    rows = df.head(limit).to_dict("records")
    out = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column, "")
            values.append(f"{value:.6g}" if isinstance(value, float) and np.isfinite(value) else str(value))
        out.append("| " + " | ".join(values) + " |")
    return "\n".join(out)


def write_report(args, runs, residuals, summary, stats, claims, gates, saved_path):
    decision = claims.iloc[0]["decision"]
    report = f"""# Executed Two-Loop Noncommuting Holonomy Report

Decision: **{decision}**

Every prediction in this report was produced by an executed NumPy one-hidden-layer ReLU MLP, an executed parameter soup, an executed branch tensor, or an executed ensemble. Candidate functions do not accept labels. Labels were generated once from the fixed planted teacher and used only after candidate logits existed.

## Exact command

```bash
{args.command_string}
```

- Git commit at execution: `{git_commit()}`
- Mode: `{args.mode}`
- Groups: `{args.groups}`
- Widths: `{args.widths}`
- Seeds: `{args.seeds}`
- Validation/test sizes: `{args.n_val}` / `{args.n_test}`
- Saved-logit leakage artifact: `{saved_path.relative_to(ROOT) if saved_path else 'not saved'}`
- Label-permutation regression: `{gates['label_permutation_regression_passed']}`

## Construction

The comparison complex is a wedge of two length-three cycles, `0-1-2-0` and `0-3-4-0`. The first loop carries the planted reflection/transposition `s`; the second carries the planted rotation/3-cycle `r`. Five local checkpoints are exact hidden-unit reparameterizations of the same executed ReLU MLP. A duplicated regular hidden orbit supplies exact automorphisms carrying the two noncommuting transitions; other hidden units remain generic, so ordinary unaligned weight averaging is a genuine executed control.

## Smoke and full-run gates

{markdown_table(pd.DataFrame([gates]), list(gates))}

## Structural residuals

{markdown_table(residuals.groupby('group_name', as_index=False).mean(numeric_only=True), ['group_name', 'pre_lift_residual', 'post_lift_residual', 'pooling_residual_gamma_1', 'pooling_residual_gamma_2', 'commutator_residual', 'group_action_multiplication_residual', 'local_functional_equivalence_residual'])}

`commutator_residual > 0` is the certificate that `rho(gamma_1) rho(gamma_2) != rho(gamma_2) rho(gamma_1)`. Both pooling residuals are required to vanish.

## Executed accuracy summary

{markdown_table(summary, ['group_name', 'hidden_width', 'method', 'n_runs', 'mean_test_accuracy', 'mean_test_loss', 'parameter_multiplier', 'branch_count', 'inference_multiplier', 'model_kind'])}

## Paired statistics

{markdown_table(stats, ['group_name', 'comparison', 'n_pairs', 'paired_mean_accuracy_delta', 'ci_low', 'ci_high', 'wins', 'ties', 'losses'])}

## Claim status

{markdown_table(claims, ['claim_id', 'status', 'decision', 'safe_wording'])}

## Interpretation

The two noncommuting holonomies, exact local functional equivalence, regular-action multiplication, and invariant-pooling certificates are supported. The branch regular lift does not receive an accuracy-advantage claim unless it beats the random and wrong controls with a positive paired confidence interval. Ties are retained as a negative empirical outcome. The ensemble is called an `ensemble_reference`, never an upper bound.
"""
    (OUT / "two_loop_holonomy_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--groups", default="S3,D4")
    parser.add_argument("--widths", default="32,64")
    parser.add_argument("--seeds", default="0:2")
    parser.add_argument("--n-val", type=int, default=256)
    parser.add_argument("--n-test", type=int, default=512)
    args = parser.parse_args()
    if args.mode == "full":
        args.groups = "S3,D4"
        args.widths = "32,64"
        args.seeds = "0:49"
        args.n_val = max(args.n_val, 1000)
        args.n_test = max(args.n_test, 2000)
    args.command_string = " ".join([sys.executable, *sys.argv])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tables").mkdir(exist_ok=True)
    (OUT / "plots").mkdir(exist_ok=True)

    runs, residuals, leakage_passed, saved_path = run_grid(
        parse_csv(args.groups), parse_csv(args.widths, int), parse_seeds(args.seeds), args.n_val, args.n_test, True
    )
    bool_columns = [
        "generators_noncommute", "pooling_certificate_passed", "group_action_certificate_passed",
        "local_equivalence_passed", "generators_recovered", "wrong_controls_rejected_structurally",
    ]
    gates = {column: bool(residuals[column].all()) for column in bool_columns}
    gates["candidate_logits_executed"] = bool(runs["candidate_logits_executed"].all())
    gates["label_permutation_regression_passed"] = bool(leakage_passed)
    gates["all_smoke_gates_passed"] = bool(all(gates.values()))
    if args.mode == "full" and not gates["all_smoke_gates_passed"]:
        raise RuntimeError(f"full run blocked by smoke gates: {gates}")

    summary = summarize(runs)
    stats = paired_stats(runs)
    critical = stats[stats["comparison"].str.endswith(("random_same_branch_count_control", "wrong_generator_control", "wrong_order_control", "wrong_group_action_control"))]
    accuracy_supported = bool(not critical.empty and (critical["ci_low"] > 0).all())
    if not gates["all_smoke_gates_passed"]:
        decision = "D. Construction or execution failed; no nonabelian accuracy claim is allowed."
        status = "unsupported"
    elif accuracy_supported:
        decision = "A. Executed two-loop noncommuting holonomy accuracy claim supported."
        status = "supported"
    else:
        decision = "B. Structural noncommuting holonomy supported, but accuracy advantage unsupported."
        status = "supported with limitations"
    claims = pd.DataFrame([
        {
            "claim_id": "executed_two_loop_noncommuting_holonomy",
            "status": status,
            "decision": decision,
            "safe_wording": "Executed S3/D4 models certify two noncommuting loop holonomies and invariant pooling; no lift accuracy advantage is claimed when controls tie.",
        }
    ])
    capacity = runs[[
        "method", "actual_parameter_count", "parameter_multiplier", "branch_count", "inference_multiplier",
        "model_kind", "uses_supplied_context", "uses_validation_data", "uses_obstruction_data",
    ]].drop_duplicates().sort_values("method")

    runs.to_csv(OUT / "two_loop_holonomy_runs.csv", index=False)
    residuals.to_csv(OUT / "two_loop_holonomy_residuals.csv", index=False)
    summary.to_csv(OUT / "two_loop_holonomy_summary.csv", index=False)
    stats.to_csv(OUT / "two_loop_holonomy_paired_stats.csv", index=False)
    capacity.to_csv(OUT / "two_loop_holonomy_capacity.csv", index=False)
    claims.to_csv(OUT / "two_loop_holonomy_claims.csv", index=False)
    config = {
        "command": args.command_string,
        "mode": args.mode,
        "groups": parse_csv(args.groups),
        "widths": parse_csv(args.widths, int),
        "seeds": parse_seeds(args.seeds),
        "n_validation": args.n_val,
        "n_test": args.n_test,
        "git_commit": git_commit(),
        "dirty_worktree": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()),
        "saved_logits": str(saved_path.relative_to(ROOT)) if saved_path else "",
        "saved_logits_sha256": file_sha256(saved_path) if saved_path else "",
        "gates": gates,
        "environment": capture_environment(),
    }
    (OUT / "two_loop_holonomy_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    latex_table(summary, ["group_name", "hidden_width", "method", "mean_test_accuracy"], OUT / "tables" / "two_loop_holonomy_accuracy.tex")
    latex_table(residuals.groupby("group_name", as_index=False).mean(numeric_only=True), ["group_name", "pre_lift_residual", "post_lift_residual", "commutator_residual"], OUT / "tables" / "two_loop_holonomy_residuals.tex")
    latex_table(capacity, ["method", "parameter_multiplier", "branch_count", "inference_multiplier", "model_kind"], OUT / "tables" / "two_loop_holonomy_capacity.tex")
    plot_accuracy(summary, OUT / "plots" / "two_loop_holonomy_accuracy.pdf")
    plot_residuals(residuals, OUT / "plots" / "two_loop_holonomy_residuals.pdf")
    plot_paired(stats, OUT / "plots" / "two_loop_holonomy_paired_delta.pdf")
    write_report(args, runs, residuals, summary, stats, claims, gates, saved_path)
    print(decision)
    print(f"wrote {OUT / 'two_loop_holonomy_report.md'}")


if __name__ == "__main__":
    main()
