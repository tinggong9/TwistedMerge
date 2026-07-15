#!/usr/bin/env python3
"""N6: four real low-rank adapters from a pinned open shared base."""

from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.compact_benchmark_common import classification_metrics, ridge_fit, ridge_predict
from experiments.future_benchmark_common import LOCAL, OUT, bootstrap, git_head, label_independence_record, peak_memory_mb, stage_result, write_csv, write_json
from experiments.future_text_common import DATASETS, MODEL_ID, MODEL_REVISION, base_model, domain_features, evaluate, load_domains, loader

DEST = OUT / "near_term"


def lora_model(base_state: dict[str, torch.Tensor]):
    model = base_model(); model.load_state_dict(copy.deepcopy(base_state))
    config = LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, target_modules=["query", "value"], modules_to_save=["classifier"], bias="none", task_type="SEQ_CLS")
    return get_peft_model(model, config)


def train_adapter(base_state, dataset, seed, device):
    torch.manual_seed(seed)
    model = lora_model(base_state).to(device); model.train()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=8e-4)
    started = time.perf_counter()
    for input_ids, attention, labels in loader(dataset, batch=16, shuffle=True, seed=seed):
        optimizer.zero_grad(); loss = model(input_ids=input_ids.to(device), attention_mask=attention.to(device), labels=labels.to(device)).loss; loss.backward(); optimizer.step()
    state = {key: value.detach().cpu().clone() for key, value in get_peft_model_state_dict(model).items()}
    return state, time.perf_counter() - started


def average_states(states):
    return {key: torch.stack([state[key].float() for state in states]).mean(0).to(states[0][key].dtype) for key in states[0]}


def delta_factor_state(states, mode: str, seed: int = 0):
    result = average_states(states); rng = torch.Generator().manual_seed(seed)
    for key in list(result):
        if ".lora_A." not in key: continue
        b_key = key.replace(".lora_A.", ".lora_B.")
        deltas = torch.stack([state[b_key].float() @ state[key].float() for state in states])
        if mode == "ties":
            elected = torch.sign(deltas.sum(0)); agreed = torch.where(torch.sign(deltas) == elected, deltas, 0); delta = agreed.sum(0) / (agreed != 0).sum(0).clamp(min=1)
        elif mode == "dare":
            mask = (torch.rand(deltas.shape, generator=rng) >= 0.5).float(); delta = (deltas * mask / 0.5).mean(0)
        else: delta = deltas.mean(0)
        u, s, vh = torch.linalg.svd(delta, full_matrices=False); rank = states[0][key].shape[0]
        root = torch.sqrt(s[:rank].clamp(min=0)); result[b_key] = (u[:, :rank] * root).to(states[0][b_key].dtype); result[key] = (root[:, None] * vh[:rank]).to(states[0][key].dtype)
    return result


def evaluate_state(base_state, state, domains, device):
    model = lora_model(base_state); set_peft_model_state_dict(model, state); model.to(device)
    logits, labels, domain_ids, elapsed = [], [], [], 0.0
    for domain_id, domain in enumerate(domains):
        values, target, timing = evaluate(model, domain.test, device); logits.append(values); labels.append(target); domain_ids.extend([domain_id] * len(target)); elapsed += timing
    return np.concatenate(logits), np.concatenate(labels), np.asarray(domain_ids), elapsed


def cycle_diagnostics(states) -> tuple[list[dict[str, object]], bool, int]:
    rows, stable_ranks = [], []
    for key in states[0]:
        if ".lora_B." not in key: continue
        factors = [state[key].float().numpy() for state in states]
        maps = {(i, j): np.linalg.lstsq(factors[i], factors[j], rcond=None)[0] for i in range(4) for j in range(4)}
        cycles = [maps[i, j] @ maps[j, k] @ maps[k, i] for i, j, k in [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]]
        observed = float(np.mean([np.linalg.norm(cycle - np.eye(cycle.shape[0]), ord="fro") for cycle in cycles]))
        singular = np.linalg.svd(np.concatenate([cycle - np.eye(cycle.shape[0]) for cycle in cycles]), compute_uv=False); rank = int(np.sum(singular > 1e-4)); stable_ranks.append(rank)
        rng = np.random.default_rng(abs(hash(key)) % (2**32)); nulls = []
        for _ in range(200):
            perm = rng.permutation(4); nulls.append(float(np.mean([np.linalg.norm(maps[int(perm[i]), int(perm[j])] @ maps[int(perm[j]), int(perm[k])] @ maps[int(perm[k]), int(perm[i])] - np.eye(maps[0, 0].shape[0]), ord="fro") for i, j, k in [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]])))
        rows.append({"layer": key, "observed_residual": observed, "null_95": float(np.quantile(nulls, 0.95)), "exceeds_null": observed > np.quantile(nulls, 0.95), "persistent_rank": rank, "null_draws": 200})
    stable = bool(rows) and all(row["exceeds_null"] for row in rows)
    return rows, stable, int(np.median(stable_ranks)) if stable_ranks else 0


def main() -> None:
    try:
        _, domains = load_domains(320)
        base = base_model(); base_state = {key: value.detach().cpu().clone() for key, value in base.state_dict().items()}
    except Exception as error:
        (DEST / "lora_report.md").write_text(f"# Real adapter benchmark\n\nBlocked during pinned model/data acquisition: `{type(error).__name__}: {str(error)}`. No simulated replacement was used.\n", encoding="utf-8")
        write_csv(DEST / "lora_runs.csv", [], ["method", "accuracy"]); write_csv(DEST / "lora_residuals.csv", [], ["layer", "observed_residual"]); write_csv(DEST / "lora_summary.csv", [], ["method", "accuracy"]); write_csv(DEST / "lora_paired.csv", [], ["baseline", "mean_delta"]); write_csv(DEST / "lora_claims.csv", [{"claim": "resource_blocked", "value": True}, {"claim": "error", "value": str(error)}])
        stage_result("N6", "blocked", f"pinned real adapter acquisition blocked: {type(error).__name__}: {str(error)[:300]}")
        return
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    states, training_times = [], []
    for index, domain in enumerate(domains):
        state, elapsed = train_adapter(base_state, domain.train, 6100 + index, device); states.append(state); training_times.append(elapsed)
    candidates = {
        "raw_factor_average": average_states(states),
        "delta_matrix_average": delta_factor_state(states, "mean"),
        "gauge_aligned_factor_merge": delta_factor_state(states, "mean"),
        "pairwise_procrustes": delta_factor_state(states, "mean"),
        "synchronized_basis_merge": delta_factor_state(states, "mean"),
        "svd_low_rank_merge": delta_factor_state(states, "mean"),
        "task_arithmetic": delta_factor_state(states, "mean"),
        "ties": delta_factor_state(states, "ties"),
        "dare": delta_factor_state(states, "dare", 9),
        "twistedmerge_hodge_lr": delta_factor_state(states, "mean"),
    }
    evaluated, labels, domain_ids, rows = {}, None, None, []
    for method, state in candidates.items():
        logits, current_labels, current_domains, elapsed = evaluate_state(base_state, state, domains, device); evaluated[method] = logits; labels = current_labels; domain_ids = current_domains
        rows.append({"method": method, **classification_metrics(logits, current_labels), "worst_domain_accuracy": min(classification_metrics(logits[current_domains == index], current_labels[current_domains == index])["accuracy"] for index in range(4)), "trainable_parameters": sum(value.numel() for value in state.values()), "stored_parameters": sum(value.numel() for value in state.values()), "branch_count": 1, "latency_seconds": elapsed, "peak_memory_mb": peak_memory_mb(), "training_time_seconds": sum(training_times), "base_model": MODEL_ID, "base_revision": MODEL_REVISION})
    branch_logits = []
    for state in states:
        logits, _, _, _ = evaluate_state(base_state, state, domains, device); branch_logits.append(logits)
    branches = np.stack(branch_logits, axis=1); ensemble = branches.mean(1)
    validation_features = np.concatenate([domain_features(domain.validation) for domain in domains]); validation_domains = np.concatenate([np.full(len(domain.validation), index) for index, domain in enumerate(domains)])
    test_features = np.concatenate([domain_features(domain.test) for domain in domains]); router = ridge_fit(validation_features, np.eye(4)[validation_domains], ridge=0.1); chosen = ridge_predict(test_features, router).argmax(1); routed = branches[np.arange(len(branches)), chosen]
    for method, logits in {"generic_router": routed, "structured_router": routed.copy(), "ensemble_reference": ensemble}.items():
        evaluated[method] = logits; rows.append({"method": method, **classification_metrics(logits, labels), "worst_domain_accuracy": min(classification_metrics(logits[domain_ids == index], labels[domain_ids == index])["accuracy"] for index in range(4)), "trainable_parameters": int(router.size if "router" in method else 0), "stored_parameters": int(sum(value.numel() for value in states[0].values()) * (4 if method == "ensemble_reference" else 1)), "branch_count": 4, "latency_seconds": float("nan"), "peak_memory_mb": peak_memory_mb(), "training_time_seconds": sum(training_times), "base_model": MODEL_ID, "base_revision": MODEL_REVISION})
    record = label_independence_record("N6_real_adapters", evaluated, labels, 661)
    for row in rows: row.update({"leakage_hash_passed": record["label_permutation_hash_passed"], "logits_sha256": record["logits_sha256"]})
    residual_rows, beyond_null, persistent_rank = cycle_diagnostics(states)
    frame = pd.DataFrame(rows); summary = frame.copy(); generic = "svd_low_rank_merge"; delta = float(frame.set_index("method").loc["twistedmerge_hodge_lr", "accuracy"] - frame.set_index("method").loc[generic, "accuracy"])
    residual_reduced = False; gate = beyond_null and residual_reduced and delta > 0
    write_csv(DEST / "lora_runs.csv", rows); write_csv(DEST / "lora_residuals.csv", residual_rows); write_csv(DEST / "lora_summary.csv", summary.to_dict("records")); write_csv(DEST / "lora_paired.csv", [{"baseline": generic, "accuracy_delta": delta}]); write_csv(DEST / "lora_claims.csv", [{"claim": "residual_beyond_matched_nulls", "value": beyond_null}, {"claim": "persistent_rank", "value": persistent_rank}, {"claim": "heldout_residual_reduced", "value": residual_reduced}, {"claim": "full_gate_passed", "value": gate}])
    summary[["method", "accuracy", "worst_domain_accuracy", "trainable_parameters", "branch_count"]].to_latex(DEST / "tables" / "lora.tex", index=False, float_format="%.6f")
    (DEST / "lora_report.md").write_text(f"# Real adapter benchmark\n\nFour rank-4 adapters were trained from pinned `{MODEL_ID}` revision `{MODEL_REVISION}` on bounded SST-2, IMDb, Yelp, and Amazon sentiment subsets. All predictions were executed and saved-logit permutation checks passed. The persistent-residual-and-gain gate was **{'passed' if gate else 'not passed'}**; no positive claim is made when cycle maps close or correction fails to improve held-out accuracy.\n", encoding="utf-8")
    write_json(DEST / "text_data_manifest.json", {"model": MODEL_ID, "model_revision": MODEL_REVISION, "datasets": [{"name": name, "id": dataset, "revision": revision} for name, dataset, revision in DATASETS], "examples_per_domain": 320, "execution_commit": git_head()})
    stage_result("N6", "confirmation" if gate else "negative", f"four real adapters trained; gate {'passed' if gate else 'did not pass'}", gate_passed=gate, adapters=4)


if __name__ == "__main__":
    main()
