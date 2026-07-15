#!/usr/bin/env python
"""Generate reports/summary.md from experiment outputs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def fmt(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for _, row in df[columns].iterrows():
        rows.append("| " + " | ".join(fmt(row[col]) for col in columns) + " |")
    return "\n".join([header, sep, *rows])


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    mu2 = pd.read_csv(REPORTS / "csv" / "synthetic_mu2_summary.csv")
    u1 = pd.read_csv(REPORTS / "csv" / "synthetic_u1_summary.csv")
    rank = pd.read_csv(REPORTS / "csv" / "rank_lift_ablation_summary.csv")
    mu2_cfg = load_json(REPORTS / "configs" / "synthetic_mu2_config.json")
    u1_cfg = load_json(REPORTS / "configs" / "synthetic_u1_config.json")
    rank_cfg = load_json(REPORTS / "configs" / "rank_lift_ablation_config.json")
    mnist_status = load_json(REPORTS / "configs" / "mnist_model_merging_status.json")
    cifar_status = load_json(REPORTS / "configs" / "cifar_model_merging_status.json")
    env = mu2_cfg["environment"]

    mu2_brief = mu2[
        [
            "flip_prob",
            "obstruction_score_mean",
            "naive_accuracy_mean",
            "rank_lift_accuracy_mean",
            "oracle_accuracy_mean",
            "rank_lift_gain_mean",
        ]
    ]
    u1_brief = u1[
        [
            "noise_std",
            "obstruction_score_mean",
            "naive_accuracy_mean",
            "rank_lift_accuracy_mean",
            "oracle_accuracy_mean",
            "rank_lift_gain_mean",
        ]
    ]
    rank_brief = rank[["experiment", "rank", "effective_rank", "mean", "std"]]

    summary = f"""# TwistedMerge Experimental Summary

Generated from local artifacts in `reports/` on 2026-06-29.

## Exact commands run

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-synthetic.txt
.venv/bin/python experiments/synthetic_mu2_obstruction.py
.venv/bin/python experiments/synthetic_u1_obstruction.py
.venv/bin/python experiments/rank_lift_ablation.py
.venv/bin/python experiments/mnist_model_merging.py --check-only || true
.venv/bin/python experiments/cifar_model_merging.py --check-only || true
.venv/bin/python experiments/generate_summary.py
```

An earlier run failed before producing all outputs because Pandas 3.x required `jinja2` for LaTeX table generation. `jinja2>=3.1` was added to the requirements and the synthetic experiments were rerun successfully.

## Hardware and software environment

- Platform: {env["platform"]}
- Machine: {env["machine"]}
- Processor: {env["processor"]}
- Python: {env["python"]}
- NumPy: {env["packages"]["numpy"]}
- Pandas: {env["packages"]["pandas"]}
- Matplotlib: {env["packages"]["matplotlib"]}
- Torch: {env["packages"]["torch"]}
- Torchvision: {env["packages"]["torchvision"]}

## Metrics

- `obstruction_score`: normalized triangle cocycle inconsistency.
- `oracle_accuracy`: test accuracy of the unmerged local synthetic models.
- `naive_accuracy`: test accuracy after descended single-gauge global merge.
- `rank_lift_accuracy`: test accuracy after rank-lifted branch merge with validation branch selection.
- `naive_failure`: `oracle_accuracy - naive_accuracy`.
- `rank_lift_gain`: `rank_lift_accuracy - naive_accuracy`.

## mu_2 synthetic obstruction

CSV: `reports/csv/synthetic_mu2_results.csv`  
Plot: `reports/plots/synthetic_mu2_obstruction.png`  
LaTeX table: `reports/tables/synthetic_mu2_summary.tex`

Correlation between obstruction score and descended-merge failure: `{mu2_cfg["correlations"]["obstruction_vs_naive_failure"]:.3f}`.

{markdown_table(mu2_brief, list(mu2_brief.columns))}

## U(1) synthetic obstruction

CSV: `reports/csv/synthetic_u1_results.csv`  
Plot: `reports/plots/synthetic_u1_obstruction.png`  
LaTeX table: `reports/tables/synthetic_u1_summary.tex`

Correlation between obstruction score and descended-merge failure: `{u1_cfg["correlations"]["obstruction_vs_naive_failure"]:.3f}`.

{markdown_table(u1_brief, list(u1_brief.columns))}

## Rank-lift ablation

CSV: `reports/csv/rank_lift_ablation.csv`  
Plot: `reports/plots/rank_lift_ablation.png`  
LaTeX table: `reports/tables/rank_lift_ablation.tex`

{markdown_table(rank_brief, list(rank_brief.columns))}

## Claim status

| Claim | Status | Evidence |
| --- | --- | --- |
| Cocycle obstruction predicts descended/global merge failure on synthetic mu_2. | Supported, moderate. | Obstruction/failure correlation is `{mu2_cfg["correlations"]["obstruction_vs_naive_failure"]:.3f}`. At flip probability `0.40`, mean descended accuracy is `{mu2_brief.iloc[-1]["naive_accuracy_mean"]:.3f}` versus `{mu2_brief.iloc[0]["naive_accuracy_mean"]:.3f}` at no flips. |
| Cocycle obstruction predicts descended/global merge failure on synthetic U(1). | Supported. | Obstruction/failure correlation is `{u1_cfg["correlations"]["obstruction_vs_naive_failure"]:.3f}`. At noise std `1.20`, mean descended accuracy is `{u1_brief.iloc[-1]["naive_accuracy_mean"]:.3f}` versus `{u1_brief.iloc[0]["naive_accuracy_mean"]:.3f}` at zero noise. |
| Rank-lifted merging fixes mu_2 synthetic failure. | Partially supported. | At flip probability `0.40`, rank-lift accuracy is `{mu2_brief.iloc[-1]["rank_lift_accuracy_mean"]:.3f}` versus descended `{mu2_brief.iloc[-1]["naive_accuracy_mean"]:.3f}`. It does not reach oracle `{mu2_brief.iloc[-1]["oracle_accuracy_mean"]:.3f}`. |
| Rank-lifted merging fixes U(1) synthetic failure. | Weakly supported at high obstruction only. | At noise std `1.20`, rank-lift accuracy is `{u1_brief.iloc[-1]["rank_lift_accuracy_mean"]:.3f}` versus descended `{u1_brief.iloc[-1]["naive_accuracy_mean"]:.3f}`. Lower-noise settings show no measurable gain in this implementation. |
| MNIST model-merging claims. | Unsupported. | Status: `{mnist_status["status"]}`. Reason: {mnist_status["reason"]} |
| CIFAR model-merging claims. | Unsupported. | Status: `{cifar_status["status"]}`. Reason: {cifar_status["reason"]} |
| Comparisons against Git Re-Basin, C2M3, Model Soups, RegMean, TIES, mergekit/MergeBench. | Unsupported. | Baselines are listed in the README as future wrappers; no external baseline code was run. |

## Reproducibility notes

- Fixed seeds are encoded in each experiment config under `reports/configs/`.
- Synthetic runs do not require external repositories.
- PyTorch and torchvision were not installed for this run, so image experiments remain scaffolds only.
- The rank-lifted merge uses validation labels for branch selection; future comparisons should account for this when comparing against baselines that do not use validation branch selection.
"""
    (REPORTS / "summary.md").write_text(summary, encoding="utf-8")
    print(f"wrote {REPORTS / 'summary.md'}")


if __name__ == "__main__":
    main()
