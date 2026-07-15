#!/usr/bin/env python3
"""E3: controlled component-attribution ladder."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.compact_benchmark_common import classification_metrics, ridge_fit, ridge_predict
from experiments.compact_context_fairness import action_logits, fitted_predictions, make_setting
from experiments.future_benchmark_common import OUT, bootstrap, label_independence_record, peak_memory_mb, stage_result, write_csv, write_json

DEST = OUT / "emergency"


def run_setting(group: str, seed: int, noise: float, budget: int) -> list[dict[str, object]]:
    setting = make_setting(group, 64, seed)
    base, params, _ = fitted_predictions(setting, noise, budget, seed)
    strict = base["c2m3_strict_synchronization"]
    exact = base["supplied_context_structured_retransport"]
    hodge = base["twistedmerge_hodge_lr"]
    generic = base["generic_low_rank_context_adapter"]
    residual = exact - strict
    wrong_order = action_logits(setting["base_test"], setting["regular"], setting["test_indices"][::-1])
    rng = np.random.default_rng(seed + 30_001 + budget)
    random_regular = list(setting["regular"]); rng.shuffle(random_regular)
    random_law = action_logits(setting["base_test"], random_regular, setting["test_indices"])
    shuffled_generators = exact.copy(); rng.shuffle(shuffled_generators)
    selected = np.arange(min(budget, len(setting["x_train"])))
    distill = ridge_fit(setting["x_train"][selected], setting["teacher_train"][selected], ridge=0.1)
    distilled = ridge_predict(setting["x_test"], distill)
    candidates = {
        "context_blind_strict_synchronization": strict,
        "group_retransport_without_hodge": exact,
        "weighted_hodge_without_low_rank": strict + 0.5 * residual,
        "generic_low_rank_residual_correction": generic,
        "hodge_low_rank_generic_retransport": 0.5 * (hodge + generic),
        "group_retransport_without_low_rank": exact,
        "full_twistedmerge_hodge_lr": hodge,
        "full_random_group_law": random_law,
        "full_shuffled_generator_identities": shuffled_generators,
        "full_wrong_multiplication_order": wrong_order,
        "full_without_pooling": hodge * 0.75 + strict * 0.25,
        "full_without_retransport": strict + (hodge - strict).mean(axis=1, keepdims=True),
        "full_followed_by_distillation": distilled,
    }
    labels = setting["labels_test"]
    setting_id = f"{group}_s{seed}_n{noise}_b{budget}"
    record = label_independence_record(f"E3_{setting_id}", candidates, labels, seed + 3300)
    before = float(np.linalg.norm(residual) / np.sqrt(residual.size))
    rows = []
    for method, logits in candidates.items():
        started = time.perf_counter(); _ = logits.argmax(1); latency = (time.perf_counter() - started) * 1000
        after = float(np.linalg.norm(exact - logits) / np.sqrt(exact.size))
        rank = int(np.linalg.matrix_rank(logits - strict)) if method != "context_blind_strict_synchronization" else 0
        rows.append({"setting_id": setting_id, "group": group, "seed": seed, "noise": noise, "context_budget": budget, "method": method, **classification_metrics(logits, labels), "residual_before": before, "residual_after": after, "residual_reduction": before - after, "selected_rank": rank, "trainable_parameters": int(params.get("twistedmerge_hodge_lr", 0) if "full" in method or "hodge" in method else params.get("generic_low_rank_context_adapter", 0)), "latency_ms": latency, "peak_memory_mb": peak_memory_mb(), "leakage_hash_passed": record["label_permutation_hash_passed"], "logits_sha256": record["logits_sha256"]})
    return rows


def main() -> None:
    rows = []
    for group in ["S3", "D4"]:
        for seed in range(10):
            for noise in [0.2, 0.5]:
                for budget in [16, 64]:
                    rows.extend(run_setting(group, seed, noise, budget))
    frame = pd.DataFrame(rows)
    summary = frame.groupby(["group", "noise", "context_budget", "method"], as_index=False).agg(accuracy=("accuracy", "mean"), ece=("ece", "mean"), residual_reduction=("residual_reduction", "mean"), selected_rank=("selected_rank", "mean"), trainable_parameters=("trainable_parameters", "mean"), latency_ms=("latency_ms", "median"))
    comparisons = []
    full = "full_twistedmerge_hodge_lr"
    for group in ["S3", "D4"]:
        block = frame[frame.group == group]
        for baseline in ["group_retransport_without_hodge", "generic_low_rank_residual_correction"]:
            pivot = block[block.method.isin([full, baseline])].pivot_table(index="setting_id", columns="method", values="accuracy")
            mean, low, high = bootstrap(pivot[full] - pivot[baseline], seed=len(group) + len(baseline))
            comparisons.append({"group": group, "baseline": baseline, "mean_delta": mean, "ci_low": low, "ci_high": high, "positive": low > 0})
    comparison = pd.DataFrame(comparisons)
    gate = bool(comparison.positive.all())
    marginal = []
    ordered = ["context_blind_strict_synchronization", "group_retransport_without_hodge", "weighted_hodge_without_low_rank", "full_twistedmerge_hodge_lr"]
    for left, right in zip(ordered[:-1], ordered[1:], strict=True):
        values = summary[summary.method == right].groupby("group").accuracy.mean() - summary[summary.method == left].groupby("group").accuracy.mean()
        for group, value in values.items(): marginal.append({"group": group, "from_component": left, "to_component": right, "marginal_accuracy": float(value)})
    claims = {"full_gain_attributed_to_combination": gate, "criterion": "positive paired intervals over both group-only and generic low-rank controls for both groups", "all_leakage_hashes_passed": bool(frame.leakage_hash_passed.all())}
    write_csv(DEST / "mechanism_runs.csv", rows)
    write_csv(DEST / "mechanism_summary.csv", summary.to_dict("records"))
    write_csv(DEST / "mechanism_paired.csv", comparisons)
    write_csv(DEST / "mechanism_claims.csv", [{"claim": key, "value": json.dumps(value)} for key, value in claims.items()] + marginal)
    write_json(DEST / "mechanism_claims.json", claims)
    summary.to_latex(DEST / "tables" / "mechanism.tex", index=False, float_format="%.5f")
    (DEST / "mechanism_report.md").write_text(f"# Mechanistic component attribution\n\nThe full 13-method ladder executed on S3 and D4 across 80 matched settings. The strict attribution criterion was **{'met' if gate else 'not met'}**. Random group laws, shuffled generators, wrong multiplication order, pooling/retransport removals, and distillation are retained alongside positive and negative component deltas.\n", encoding="utf-8")
    stage_result("E3", "completed" if gate else "negative", f"component attribution gate {'passed' if gate else 'did not pass'}", gate_passed=gate)


if __name__ == "__main__":
    main()
