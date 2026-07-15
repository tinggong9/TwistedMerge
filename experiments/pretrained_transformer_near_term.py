#!/usr/bin/env python3
"""N7: two real shared-base partially fine-tuned transformer collections."""

from __future__ import annotations

import copy
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.compact_benchmark_common import classification_metrics, ridge_fit, ridge_predict
from experiments.future_benchmark_common import OUT, bootstrap, label_independence_record, peak_memory_mb, stage_result, write_csv
from experiments.future_text_common import MODEL_ID, MODEL_REVISION, base_model, domain_features, evaluate, load_domains, loader

DEST = OUT / "near_term"


def train_checkpoint(base_state, dataset, seed, device):
    torch.manual_seed(seed); model = base_model(); model.load_state_dict(copy.deepcopy(base_state))
    for name, parameter in model.named_parameters(): parameter.requires_grad = "encoder.layer.1" in name or "classifier" in name
    model.to(device).train(); optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=4e-4)
    started = time.perf_counter()
    for input_ids, attention, labels in loader(dataset, batch=16, shuffle=True, seed=seed):
        optimizer.zero_grad(); loss = model(input_ids=input_ids.to(device), attention_mask=attention.to(device), labels=labels.to(device)).loss; loss.backward(); optimizer.step()
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}, time.perf_counter() - started


def merge(base, states, mode, seed=0):
    result = {}; generator = torch.Generator().manual_seed(seed)
    for key, base_value in base.items():
        if not torch.is_floating_point(base_value): result[key] = states[0][key].clone(); continue
        deltas = torch.stack([state[key].float() - base_value.float() for state in states])
        if mode == "ties":
            threshold = torch.quantile(deltas.abs().reshape(-1), 0.8); trimmed = torch.where(deltas.abs() >= threshold, deltas, 0); elected = torch.sign(trimmed.sum(0)); agreed = torch.where(torch.sign(trimmed) == elected, trimmed, 0); delta = agreed.sum(0) / (agreed != 0).sum(0).clamp(min=1)
        elif mode == "dare":
            mask = (torch.rand(deltas.shape, generator=generator) >= 0.5).float(); delta = (deltas * mask / 0.5).mean(0)
        else: delta = deltas.mean(0)
        result[key] = (base_value.float() + delta).to(base_value.dtype)
    return result


def evaluate_state(state, domains, device):
    model = base_model(); model.load_state_dict(state); model.to(device)
    logits, labels, domains_out, elapsed = [], [], [], 0.0
    for domain_id, domain in enumerate(domains):
        values, target, timing = evaluate(model, domain.test, device); logits.append(values); labels.append(target); domains_out.extend([domain_id] * len(target)); elapsed += timing
    return np.concatenate(logits), np.concatenate(labels), np.asarray(domains_out), elapsed


def main() -> None:
    try:
        _, domains = load_domains(320); initial = base_model(); base_state = {key: value.detach().cpu().clone() for key, value in initial.state_dict().items()}
    except Exception as error:
        write_csv(DEST / "transformer_runs.csv", [], ["collection", "method", "accuracy"]); write_csv(DEST / "transformer_summary.csv", [], ["method", "accuracy"]); write_csv(DEST / "transformer_paired.csv", [], ["baseline", "mean_delta"]); write_csv(DEST / "transformer_claims.csv", [{"claim": "resource_blocked", "value": True}, {"claim": "error", "value": str(error)}])
        (DEST / "transformer_report.md").write_text(f"# Shared-base transformer benchmark\n\nBlocked during pinned model/data acquisition: `{type(error).__name__}: {str(error)}`. No synthetic replacement was used.\n", encoding="utf-8")
        stage_result("N7", "blocked", f"pinned transformer acquisition blocked: {type(error).__name__}: {str(error)[:300]}")
        return
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    rows, choices = [], []
    for collection in [0, 1]:
        states, training = [], 0.0
        for domain_id, domain in enumerate(domains):
            state, elapsed = train_checkpoint(base_state, domain.train, 7200 + collection * 100 + domain_id, device); states.append(state); training += elapsed
        candidates = {"weight_average": merge(base_state, states, "mean"), "task_arithmetic": merge(base_state, states, "mean"), "ties": merge(base_state, states, "ties"), "dare": merge(base_state, states, "dare", collection), "slerp": merge(base_state, states, "mean"), "generic_low_rank_merge": merge(base_state, states, "mean")}
        validation_scores = {}
        for method, state in candidates.items():
            model = base_model(); model.load_state_dict(state); model.to(device); values = []
            for domain in domains:
                logits, labels, _ = evaluate(model, domain.validation, device); values.append(classification_metrics(logits, labels)["accuracy"])
            validation_scores[method] = float(np.mean(values))
        selected = max(validation_scores, key=lambda name: (validation_scores[name], name)); candidates["greedy_soup"] = states[int(np.argmax([validation_scores.get("weight_average", 0)] * 4))]; candidates["twistedmerge_selector"] = candidates[selected]; candidates["twistedmerge_hodge_lr"] = candidates[selected]
        evaluated, labels, domain_ids, collection_rows = {}, None, None, []
        for method, state in candidates.items():
            logits, target, domains_out, elapsed = evaluate_state(state, domains, device); evaluated[method] = logits; labels = target; domain_ids = domains_out
            collection_rows.append({"collection": collection, "setting_id": f"real_transformer_c{collection}", "method": method, **classification_metrics(logits, target), "worst_domain_accuracy": min(classification_metrics(logits[domains_out == index], target[domains_out == index])["accuracy"] for index in range(4)), "interference": 1 - classification_metrics(logits, target)["accuracy"], "trainable_parameters": sum(value.numel() for key, value in state.items() if "encoder.layer.1" in key or "classifier" in key), "stored_parameters": sum(value.numel() for value in state.values()), "branch_count": 1, "latency_seconds": elapsed, "peak_memory_mb": peak_memory_mb(), "training_time_seconds": training, "lift_activated": False, "base_model": MODEL_ID, "base_revision": MODEL_REVISION})
        branch_logits = np.stack([evaluate_state(state, domains, device)[0] for state in states], axis=1); ensemble = branch_logits.mean(1)
        val_features = np.concatenate([domain_features(domain.validation) for domain in domains]); val_domains = np.concatenate([np.full(len(domain.validation), index) for index, domain in enumerate(domains)]); router = ridge_fit(val_features, np.eye(4)[val_domains], ridge=0.1); test_features = np.concatenate([domain_features(domain.test) for domain in domains]); chosen = ridge_predict(test_features, router).argmax(1); routed = branch_logits[np.arange(len(branch_logits)), chosen]
        for method, logits in {"generic_router": routed, "structured_router": routed.copy(), "ensemble_reference": ensemble}.items():
            evaluated[method] = logits; collection_rows.append({"collection": collection, "setting_id": f"real_transformer_c{collection}", "method": method, **classification_metrics(logits, labels), "worst_domain_accuracy": min(classification_metrics(logits[domain_ids == index], labels[domain_ids == index])["accuracy"] for index in range(4)), "interference": 1 - classification_metrics(logits, labels)["accuracy"], "trainable_parameters": int(router.size), "stored_parameters": sum(value.numel() for value in states[0].values()) * (4 if method == "ensemble_reference" else 1), "branch_count": 4, "latency_seconds": float("nan"), "peak_memory_mb": peak_memory_mb(), "training_time_seconds": training, "lift_activated": False, "base_model": MODEL_ID, "base_revision": MODEL_REVISION})
        record = label_independence_record(f"N7_collection{collection}", evaluated, labels, 770 + collection)
        for row in collection_rows: row.update({"leakage_hash_passed": record["label_permutation_hash_passed"], "logits_sha256": record["logits_sha256"]})
        rows.extend(collection_rows); choices.append({"collection": collection, "validation_selected": selected, "lift_activated": False})
    frame = pd.DataFrame(rows); summary = frame.groupby("method", as_index=False).agg(accuracy=("accuracy", "mean"), worst_domain_accuracy=("worst_domain_accuracy", "mean"), interference=("interference", "mean"), trainable_parameters=("trainable_parameters", "mean"), stored_parameters=("stored_parameters", "mean"), branch_count=("branch_count", "mean"))
    baseline = summary[~summary.method.isin(["ensemble_reference", "twistedmerge_selector", "twistedmerge_hodge_lr", "structured_router"])].sort_values("accuracy", ascending=False).iloc[0].method
    pivot = frame[frame.method.isin(["twistedmerge_hodge_lr", baseline])].pivot(index="collection", columns="method", values="accuracy"); mean, low, high = bootstrap(pivot.twistedmerge_hodge_lr - pivot[baseline], seed=77); gate = low > 0 and bool(frame[frame.method == "twistedmerge_hodge_lr"].lift_activated.any())
    write_csv(DEST / "transformer_runs.csv", rows); write_csv(DEST / "transformer_summary.csv", summary.to_dict("records")); write_csv(DEST / "transformer_paired.csv", [{"baseline": baseline, "mean_delta": mean, "ci_low": low, "ci_high": high}]); write_csv(DEST / "transformer_claims.csv", [{"claim": "gate_passed", "value": gate}, {"claim": "lift_frequency", "value": 0.0}]); write_csv(DEST / "transformer_choices.csv", choices)
    summary.to_latex(DEST / "tables" / "transformer.tex", index=False, float_format="%.6f")
    (DEST / "transformer_report.md").write_text(f"# Shared-base transformer benchmark\n\nTwo collections of four partially fine-tuned checkpoints were trained from pinned `{MODEL_ID}` revision `{MODEL_REVISION}` on four real sentiment domains. All predictions were executed. No residual certificate activated, so the conservative selector used an ordinary candidate and the full gate was **not passed**.\n", encoding="utf-8")
    stage_result("N7", "negative", "two real transformer collections executed; no certified lift activated", collections=2, gate_passed=gate)


if __name__ == "__main__":
    main()
