#!/usr/bin/env python3
"""E5: integrity-check the established 120-setting practical selector."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.future_benchmark_common import OUT, ROOT, sha256_file, stage_result, write_csv

DEST = OUT / "emergency"


def main() -> None:
    source = ROOT / "reports" / "overnight_program" / "practical_selector_runs.csv"
    if not source.exists():
        stage_result("E5", "failed", "established practical-selector run ledger is missing")
        raise SystemExit(1)
    frame = pd.read_csv(source)
    setting_column = "setting_id" if "setting_id" in frame else "setting"
    settings = int(frame[setting_column].nunique())
    hashes = sorted(set(frame.get("saved_logits_sha256", frame.get("logits_sha256", pd.Series(dtype=str))).dropna().astype(str)))
    leakage_column = "label_permutation_regression_passed" if "label_permutation_regression_passed" in frame else "leakage_hash_passed"
    leakage = bool(frame[leakage_column].all()) if leakage_column in frame else False
    methods = frame.method.unique().tolist()
    selector_name = next((name for name in methods if "selector" in name and "twisted" in name.lower()), "twistedmerge_selector")
    greedy_name = next((name for name in methods if "ordinary_greedy" in name or name == "greedy_soup"), None)
    if greedy_name is None:
        greedy_name = next(name for name in methods if "greedy" in name)
    pivot = frame[frame.method.isin([selector_name, greedy_name])].pivot_table(index=setting_column, columns="method", values="accuracy")
    paired = []
    if selector_name in pivot and greedy_name in pivot:
        for key, row in pivot.iterrows():
            paired.append({"setting_id": key, "selector_accuracy": row[selector_name], "greedy_accuracy": row[greedy_name], "selector_minus_greedy": row[selector_name] - row[greedy_name]})
    activation_columns = [column for column in frame.columns if "activation" in column or column in {"central_selected", "nonabelian_selected", "lift_activated"}]
    activations = {column: float(frame[column].astype(float).sum()) for column in activation_columns}
    verified = frame.copy()
    verified["source_sha256"] = sha256_file(source)
    verified.to_csv(DEST / "practical_selector_verified.csv", index=False)
    write_csv(DEST / "practical_selector_paired.csv", paired)
    pd.DataFrame(paired).to_latex(DEST / "tables" / "practical_selector.tex", index=False, float_format="%.6f")
    integrity = settings == 120 and leakage and all(value == 0 for value in activations.values())
    delta = float(pd.DataFrame(paired).selector_minus_greedy.mean()) if paired else float("nan")
    (DEST / "practical_selector_report.md").write_text(f"# Practical-selector integrity freeze\n\nThe established fresh ledger contains `{settings}` matched settings. Saved-logit permutation checks pass: `{leakage}`. Recorded central/nonabelian activation totals are `{activations}`. The paired selector-minus-greedy mean is `{delta:+.6f}` and is retained without reinterpretation.\n", encoding="utf-8")
    stage_result("E5", "clean-freeze" if integrity else "failed", f"settings={settings}; leakage={leakage}; selector-minus-greedy={delta:+.6f}", settings=settings, activations=activations)
    if not integrity:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
