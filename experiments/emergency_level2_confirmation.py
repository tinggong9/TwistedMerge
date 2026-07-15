#!/usr/bin/env python3
"""E1: independent S3/D4 controlled confirmation on fresh seeds."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.compact_benchmark_common import classification_metrics
from experiments.compact_context_fairness import action_logits, fitted_predictions, make_setting
from experiments.future_benchmark_common import LOCAL, OUT, bootstrap, label_independence_record, peak_memory_mb, stage_result, write_csv, write_json

DEST = OUT / "emergency"
GENERIC = ["generic_mixture_of_experts", "learned_unconstrained_matrix_context_action", "generic_low_rank_context_adapter"]


def variants(group: str, seed: int, noise: float, budget: int) -> tuple[dict[str, np.ndarray], dict[str, int], dict[str, object]]:
    setting = make_setting(group, 64, seed)
    base, params, _ = fitted_predictions(setting, noise, budget, seed)
    rng = np.random.default_rng(710_000 + seed + int(noise * 1000) + budget)
    classes = len(setting["regular"])
    random_table = list(setting["regular"])
    rng.shuffle(random_table)
    random_law = action_logits(setting["base_test"], random_table, setting["test_indices"])
    exact = base["supplied_context_structured_retransport"]
    hodge = base["twistedmerge_hodge_lr"]
    generic_lr = base["generic_low_rank_context_adapter"]
    candidates = {
        "twistedmerge_hodge_lr": hodge,
        "generic_mixture_of_experts": base["generic_mixture_of_experts"],
        "learned_unconstrained_matrix_context_action": base["learned_matrix_context_action"],
        "generic_low_rank_context_adapter": generic_lr,
        "group_structured_without_hodge": exact,
        "hodge_lr_generic_retransport": 0.5 * (hodge + generic_lr),
        "random_multiplication_table_control": random_law,
        "shuffled_context_control": base["shuffled_context_control"],
    }
    counts = {
        "twistedmerge_hodge_lr": params["twistedmerge_hodge_lr"],
        "generic_mixture_of_experts": params["generic_mixture_of_experts"],
        "learned_unconstrained_matrix_context_action": params["learned_matrix_context_action"],
        "generic_low_rank_context_adapter": params["generic_low_rank_context_adapter"],
        "group_structured_without_hodge": 0,
        "hodge_lr_generic_retransport": params["twistedmerge_hodge_lr"] + classes * classes,
        "random_multiplication_table_control": 0,
        "shuffled_context_control": 0,
    }
    return candidates, counts, setting


def evaluate(group: str, seed: int, noise: float, budget: int) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    candidates, counts, setting = variants(group, seed, noise, budget)
    labels = setting["labels_test"]
    setting_id = f"{group}_w64_s{seed}_n{noise:.1f}_b{budget}"
    record = label_independence_record(f"E1_{setting_id}", candidates, labels, seed + 991)
    if not record["label_permutation_hash_passed"]:
        raise RuntimeError("saved-logit permutation regression failed")
    masks = {
        "standard": np.ones(len(labels), dtype=bool),
        "words_length_4_5": np.array([len(word) in {4, 5} for word in setting["test_words"]]),
        "heldout_rs_after_sr": np.array([tuple(word[:2]) == ("r", "s") for word in setting["test_words"]]),
        "heldout_element": setting["test_indices"] == int(np.bincount(setting["test_indices"]).argmin()),
        "context_distribution_shift": np.arange(len(labels)) % 3 == 0,
    }
    rows = []
    generalization = []
    for method, logits in candidates.items():
        started = time.perf_counter()
        _ = logits.argmax(axis=1)
        latency = (time.perf_counter() - started) * 1000
        for split, mask in masks.items():
            if not mask.any():
                continue
            metrics = classification_metrics(logits[mask], labels[mask])
            row = {
                "setting_id": setting_id,
                "group": group,
                "width": 64,
                "seed": seed,
                "noise": noise,
                "context_budget": budget,
                "split": split,
                "method": method,
                **metrics,
                "trainable_parameters": counts[method],
                "stored_parameters": counts[method],
                "branch_count": len(setting["regular"]) if "retransport" in method or "mixture" in method else 1,
                "latency_ms": latency,
                "peak_memory_mb": peak_memory_mb(),
                "leakage_hash_passed": True,
                "logits_sha256": record["logits_sha256"],
            }
            (rows if split == "standard" else generalization).append(row)
    # Two genuinely inferred-context diagnostics, trained without an explicit group element.
    for source, train_x, test_x in [("input_only", setting["x_train"], setting["x_test"]), ("alignment_residual_only", np.abs(setting["base_train"]), np.abs(setting["base_test"]))]:
        from experiments.compact_benchmark_common import ridge_fit, ridge_predict

        chosen = np.arange(min(budget, len(train_x)))
        router = ridge_fit(train_x[chosen], np.eye(len(setting["regular"]))[setting["train_indices"][chosen]], ridge=0.1)
        predicted = ridge_predict(test_x, router).argmax(axis=1)
        logits = action_logits(setting["base_test"], setting["regular"], predicted)
        generalization.append({
            "setting_id": setting_id,
            "group": group,
            "width": 64,
            "seed": seed,
            "noise": noise,
            "context_budget": budget,
            "split": source,
            "method": "twistedmerge_hodge_lr",
            **classification_metrics(logits, labels),
            "trainable_parameters": int(router.size),
            "stored_parameters": int(router.size),
            "branch_count": len(setting["regular"]),
            "latency_ms": float("nan"),
            "peak_memory_mb": peak_memory_mb(),
            "leakage_hash_passed": True,
            "logits_sha256": record["logits_sha256"],
        })
    return rows, generalization


def main() -> None:
    rows, generalization = [], []
    for group in ["S3", "D4"]:
        for seed in range(20, 30):
            for noise in [0.2, 0.5] + ([1.0] if seed < 25 else []):
                for budget in [16, 64]:
                    current, extra = evaluate(group, seed, noise, budget)
                    rows.extend(current)
                    generalization.extend(extra)
    frame = pd.DataFrame(rows)
    summaries = frame.groupby(["group", "noise", "context_budget", "method"], as_index=False).agg(accuracy=("accuracy", "mean"), loss=("loss", "mean"), ece=("ece", "mean"), trainable_parameters=("trainable_parameters", "mean"), latency_ms=("latency_ms", "median"))
    paired = []
    conditions = {}
    for group in ["S3", "D4"]:
        passed = 0
        for noise in [0.2, 0.5]:
            for budget in [16, 64]:
                block = frame[(frame.group == group) & (frame.noise == noise) & (frame.context_budget == budget)]
                means = block[block.method.isin(GENERIC)].groupby("method").accuracy.mean()
                best = str(means.idxmax())
                pivot = block[block.method.isin(["twistedmerge_hodge_lr", best])].pivot(index="seed", columns="method", values="accuracy")
                delta = pivot["twistedmerge_hodge_lr"] - pivot[best]
                mean, low, high = bootstrap(delta, seed=int(noise * 1000) + budget)
                positive = low > 0
                passed += int(positive)
                paired.append({"group": group, "noise": noise, "context_budget": budget, "best_generic": best, "mean_delta": mean, "ci_low": low, "ci_high": high, "positive_interval": positive})
        gen = pd.DataFrame(generalization)
        held = gen[(gen.group == group) & gen.split.isin(["words_length_4_5", "heldout_rs_after_sr", "heldout_element", "context_distribution_shift"])]
        regrets = 1 - held.groupby("method").accuracy.mean()
        structured_regret = float(regrets.get("twistedmerge_hodge_lr", 1.0))
        generic_regret = float(regrets.reindex(GENERIC).min())
        conditions[group] = {"positive_conditions": passed, "condition_A": passed >= 3, "condition_B": False, "condition_C": structured_regret < generic_regret, "structured_worst_case_regret": structured_regret, "best_generic_worst_case_regret": generic_regret}
    gate = all(item["condition_A"] or item["condition_B"] or item["condition_C"] for item in conditions.values())
    claims = {"independent_gate_passed": gate, "conditions": conditions, "fresh_seed_block": "20:29", "all_leakage_hashes_passed": bool(frame.leakage_hash_passed.all())}
    write_csv(DEST / "level2_runs.csv", rows)
    write_csv(DEST / "level2_summary.csv", summaries.to_dict("records"))
    write_csv(DEST / "level2_paired.csv", paired)
    write_csv(DEST / "level2_efficiency.csv", summaries[["group", "noise", "context_budget", "method", "accuracy", "trainable_parameters", "latency_ms"]].to_dict("records"))
    write_csv(DEST / "level2_generalization.csv", generalization)
    write_csv(DEST / "level2_claims.csv", [{"claim": key, "value": json.dumps(value, sort_keys=True)} for key, value in claims.items()])
    write_json(DEST / "level2_claims.json", claims)
    summaries.to_latex(DEST / "tables" / "level2_main.tex", index=False, float_format="%.5f")
    pd.DataFrame(paired).to_latex(DEST / "tables" / "level2_efficiency.tex", index=False, float_format="%.5f")
    fig, ax = plt.subplots(figsize=(7, 4))
    for method in ["twistedmerge_hodge_lr", *GENERIC]:
        block = summaries[summaries.method == method].groupby("noise").accuracy.mean()
        ax.plot(block.index, block.values, marker="o", label=method)
    ax.set(xlabel="Context corruption", ylabel="Accuracy", ylim=(0, 1.02))
    ax.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(DEST / "plots" / "level2_noise.pdf"); plt.close(fig)
    gen_frame = pd.DataFrame(generalization)
    fig, ax = plt.subplots(figsize=(8, 4))
    gen_frame[gen_frame.method.isin(["twistedmerge_hodge_lr", *GENERIC])].groupby(["split", "method"]).accuracy.mean().unstack().plot.bar(ax=ax)
    ax.set(ylabel="Accuracy", ylim=(0, 1.02)); fig.tight_layout(); fig.savefig(DEST / "plots" / "level2_generalization.pdf"); plt.close(fig)
    (DEST / "level2_report.md").write_text(f"# Independent controlled confirmation\n\nFresh seeds 20--29 were executed for S3 and D4 at width 64 with matched context budgets and corruption. The independent gate was **{'passed' if gate else 'not passed'}**. Noise 1.0 is retained as a boundary condition, and all inferred-context and held-out-composition results remain in the generalization ledger.\n", encoding="utf-8")
    stage_result("E1", "confirmation" if gate else "negative", f"independent controlled gate {'passed' if gate else 'did not pass'} on fresh seeds", gate_passed=gate)


if __name__ == "__main__":
    main()
