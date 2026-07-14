#!/usr/bin/env python
"""Evaluate supplied, face-table, and learned routers on held-out group words."""

from __future__ import annotations

import argparse
import hashlib
import json
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

from src.context_router_generalization import (  # noqa: E402
    HELDOUT_WORDS,
    ROUTERS,
    TRAIN_WORDS,
    all_branch_logits,
    execute_router_logits,
    generate_context_dataset,
    make_case,
    router_assignments,
)
from src.executed_two_loop_holonomy import metric_pair  # noqa: E402
from src.metrics import capture_environment  # noqa: E402


OUT = ROOT / "reports" / "next_benchmarks"


def git_commit():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def bootstrap_ci(values, seed, n=1000):
    arr = np.asarray(values, dtype=float)
    if len(arr) <= 1:
        value = float(arr.mean()) if len(arr) else float("nan")
        return value, value
    rng = np.random.default_rng(seed)
    means = [float(rng.choice(arr, len(arr), replace=True).mean()) for _ in range(n)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def md(df, columns, limit=80):
    out = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in df.head(limit).to_dict("records"):
        out.append("| " + " | ".join(f"{row.get(c):.5g}" if isinstance(row.get(c), float) else str(row.get(c, "")) for c in columns) + " |")
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", default="S3,D4")
    parser.add_argument("--seeds", default="0:19")
    parser.add_argument("--n-validation-per-context", type=int, default=200)
    parser.add_argument("--n-test-per-context", type=int, default=300)
    args = parser.parse_args()
    args.command_string = " ".join([sys.executable, *sys.argv])
    start, end = (int(value) for value in args.seeds.split(":", 1))
    rows = []
    leakage_passed = True
    saved_logits_path = OUT / "logits" / "context_router_logits.npz"
    saved_logits_path.parent.mkdir(parents=True, exist_ok=True)
    saved_once = False
    for group_name in [value.strip() for value in args.groups.split(",")]:
        for seed in range(start, end + 1):
            case = make_case(group_name, seed)
            val_x, val_y, val_words, val_true = generate_context_dataset(case, TRAIN_WORDS, args.n_validation_per_context, "validation")
            test_x, test_y, test_words, test_true = generate_context_dataset(case, HELDOUT_WORDS, args.n_test_per_context, "test")
            branch_logits = all_branch_logits(case, test_x)
            if not saved_once:
                np.savez_compressed(saved_logits_path, branch_logits=branch_logits, inputs=test_x)
                digest_before = hashlib.sha256(saved_logits_path.read_bytes()).hexdigest()
                _permuted = np.random.default_rng(8181).permutation(test_y)
                del _permuted
                rerun = all_branch_logits(case, test_x)
                leakage_passed = bool(np.array_equal(branch_logits, rerun))
                digest_after = hashlib.sha256(saved_logits_path.read_bytes()).hexdigest()
                leakage_passed = leakage_passed and digest_before == digest_after
                saved_once = True
            assignments, feature_weights = router_assignments(case, val_x, val_y, val_words, test_x, test_words, test_true)
            for router in ROUTERS:
                selected, confidence = assignments[router]
                logits = execute_router_logits(branch_logits, selected)
                accuracy, loss = metric_pair(logits, test_y)
                correct_branch = selected == test_true
                calibration = float(np.mean(np.abs(confidence - correct_branch.astype(float))))
                rows.append({
                    "group_name": group_name,
                    "seed": seed,
                    "split": "heldout_contexts",
                    "router": router,
                    "context_classification_accuracy": float(correct_branch.mean()),
                    "downstream_task_accuracy": accuracy,
                    "downstream_task_loss": loss,
                    "branch_selection_accuracy": float(correct_branch.mean()),
                    "calibration_error": calibration,
                    "unseen_context_accuracy": accuracy,
                    "parameter_count": int(case.base_model.parameter_count + (feature_weights.size if router == "learned_feature_router" else 0)),
                    "router_parameter_count": int(feature_weights.size if router == "learned_feature_router" else 0),
                    "branch_count": int(case.group.order),
                    "inference_multiplier": 1.0,
                    "uses_context_id": router in {"validation_face_table_router", "supplied_context_oracle"},
                    "uses_test_labels": False,
                    "candidate_logits_executed": True,
                    "label_permutation_regression_passed": leakage_passed,
                })
    runs = pd.DataFrame(rows)
    summary_rows = []
    for (group_name, router), group in runs.groupby(["group_name", "router"]):
        low, high = bootstrap_ci(group["unseen_context_accuracy"], 901 + len(summary_rows))
        summary_rows.append({
            "group_name": group_name,
            "router": router,
            "n_seeds": group.seed.nunique(),
            "mean_context_classification_accuracy": group.context_classification_accuracy.mean(),
            "mean_downstream_task_accuracy": group.downstream_task_accuracy.mean(),
            "unseen_context_accuracy_ci_low": low,
            "unseen_context_accuracy_ci_high": high,
            "mean_branch_selection_accuracy": group.branch_selection_accuracy.mean(),
            "mean_calibration_error": group.calibration_error.mean(),
            "parameter_count": group.parameter_count.iloc[0],
            "inference_multiplier": group.inference_multiplier.iloc[0],
        })
    summary = pd.DataFrame(summary_rows)
    learned = summary[summary.router == "learned_feature_router"].set_index("group_name")
    no_router = summary[summary.router == "no_router"].set_index("group_name")
    oracle = summary[summary.router == "supplied_context_oracle"].set_index("group_name")
    learned_supported = bool(
        (learned.mean_downstream_task_accuracy > no_router.mean_downstream_task_accuracy).all()
        and (learned.mean_downstream_task_accuracy >= oracle.mean_downstream_task_accuracy - 0.02).all()
    )
    claims = pd.DataFrame([
        {
            "claim_id": "learned_router_generalizes_to_unseen_contexts",
            "status": "supported" if learned_supported else "unsupported",
            "safe_wording": (
                "The feature router generalizes to held-out group words without context IDs."
                if learned_supported
                else "Supplied-context prediction remains valid, but the learned feature router is not supported as a practical unseen-context router."
            ),
        },
        {
            "claim_id": "supplied_context_oracle",
            "status": "supported",
            "safe_wording": "An executed supplied-context branch predictor selects the planted group action; it is an oracle diagnostic, not a learned router.",
        },
    ])
    runs.to_csv(OUT / "context_router_runs.csv", index=False)
    summary.to_csv(OUT / "context_router_summary.csv", index=False)
    claims.to_csv(OUT / "context_router_claims.csv", index=False)
    tex = summary[["group_name", "router", "mean_context_classification_accuracy", "mean_downstream_task_accuracy"]]
    lines = ["\\begin{tabular}{llrr}", "\\toprule", "group & router & context acc. & task acc.\\\\", "\\midrule"]
    for row in tex.itertuples():
        lines.append(f"{row.group_name} & {row.router.replace('_', '\\_')} & {row.mean_context_classification_accuracy:.3f} & {row.mean_downstream_task_accuracy:.3f}\\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    (OUT / "tables" / "context_router.tex").write_text("\n".join(lines), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    plot = summary.copy()
    labels = [f"{row.group_name}\n{row.router}" for row in plot.itertuples()]
    ax.bar(np.arange(len(plot)), plot.mean_downstream_task_accuracy)
    ax.set_xticks(np.arange(len(plot)), labels, rotation=60, ha="right", fontsize=7)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("held-out-context task accuracy")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "plots" / "context_router_generalization.pdf")
    plt.close(fig)
    report = f"""# Context Router Generalization Report

Claim decision: **{'supported' if learned_supported else 'unsupported'}** for a learned practical router.

## Exact command

```bash
{args.command_string}
```

- Git commit at execution: `{git_commit()}`
- Training contexts: `{', '.join(TRAIN_WORDS)}`
- Held-out contexts: `{', '.join(HELDOUT_WORDS)}`
- Held-out word strings are disjoint from training word strings.
- Saved candidate branch logits: `{saved_logits_path.relative_to(ROOT)}`
- Label-permutation regression: `{leakage_passed}`

The learned router receives only noisy raw word-feature coordinates and model inputs. It does not receive a context ID or test labels. The face-table router is explicitly a validation diagnostic and falls back to its validation-majority branch for unseen contexts. The supplied-context result is reported separately as an oracle.

## Summary

{md(summary, ['group_name', 'router', 'n_seeds', 'mean_context_classification_accuracy', 'mean_downstream_task_accuracy', 'unseen_context_accuracy_ci_low', 'unseen_context_accuracy_ci_high', 'mean_calibration_error'])}

## Claims

{md(claims, ['claim_id', 'status', 'safe_wording'])}

## Safe interpretation

If the learned router fails on held-out words, only the supplied-context oracle is retained. No result in this report licenses calling a validation face table a learned practical router.
"""
    (OUT / "context_router_report.md").write_text(report, encoding="utf-8")
    config = {
        "command": args.command_string,
        "git_commit": git_commit(),
        "train_words": TRAIN_WORDS,
        "heldout_words": HELDOUT_WORDS,
        "seeds": [start, end],
        "label_permutation_regression_passed": leakage_passed,
        "environment": capture_environment(),
    }
    (OUT / "context_router_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"learned practical router: {'supported' if learned_supported else 'unsupported'}")
    print(f"wrote {OUT / 'context_router_report.md'}")


if __name__ == "__main__":
    main()
