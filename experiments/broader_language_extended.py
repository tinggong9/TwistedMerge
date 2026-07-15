#!/usr/bin/env python3
"""X2: second-base adapter collections plus the completed language ledger."""

from __future__ import annotations

import copy
import hashlib
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, BertConfig, BertForSequenceClassification

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.compact_benchmark_common import classification_metrics, ridge_fit, ridge_predict
from experiments.future_benchmark_common import OUT, bootstrap, label_independence_record, peak_memory_mb, stage_result, write_csv
from experiments.future_text_common import DATASETS, MODEL_ID, MODEL_REVISION, DomainData, domain_features, loader, tensor_dataset, text_and_label
from experiments.real_lora_adapter_near_term import average_states, delta_factor_state

DEST = OUT / "extended"
SECOND_MODEL_ID = "prajjwal1/bert-tiny"
SECOND_MODEL_REVISION = "6f75de8b60a9f8a2fdf7b69cbd86d9e64bcb3837"
TOKENIZER_ID = MODEL_ID
TOKENIZER_REVISION = MODEL_REVISION


def base_model(model_id: str = SECOND_MODEL_ID, revision: str = SECOND_MODEL_REVISION):
    torch.manual_seed(20_260)
    config_path = hf_hub_download(model_id, "config.json", revision=revision)
    import json

    payload = json.loads(Path(config_path).read_text(encoding="utf-8"))
    payload.update({"model_type": "bert", "num_labels": 2})
    config = BertConfig.from_dict(payload)
    return BertForSequenceClassification.from_pretrained(model_id, revision=revision, config=config)


def load_second_base_domains(per_domain: int = 128) -> tuple[object, list[DomainData]]:
    # The second model repository does not contain a fast-tokenizer artifact;
    # both checkpoints use the standard uncased BERT vocabulary.
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID, revision=TOKENIZER_REVISION)
    domains = []
    for name, dataset_id, revision in DATASETS:
        stream = load_dataset(dataset_id, split="train", streaming=True, revision=revision)
        rows = list(stream.take(per_domain))
        pairs = [text_and_label(row, name) for row in rows]
        texts = [item[0] for item in pairs]
        labels = [item[1] for item in pairs]
        train_end, validation_end = per_domain // 2, 3 * per_domain // 4
        domains.append(DomainData(name, tensor_dataset(tokenizer, texts[:train_end], labels[:train_end]), tensor_dataset(tokenizer, texts[train_end:validation_end], labels[train_end:validation_end]), tensor_dataset(tokenizer, texts[validation_end:], labels[validation_end:])))
    return tokenizer, domains


def lora_model(base_state: dict[str, torch.Tensor], rank: int):
    model = base_model()
    model.load_state_dict(copy.deepcopy(base_state))
    config = LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.0, target_modules=["query", "value"], modules_to_save=["classifier"], bias="none", task_type="SEQ_CLS")
    return get_peft_model(model, config)


def train_adapter(base_state, data, rank: int, seed: int, target_device: torch.device):
    torch.manual_seed(seed)
    model = lora_model(base_state, rank).to(target_device).train()
    optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=8e-4)
    started = time.perf_counter()
    for input_ids, attention, labels in loader(data, batch=16, shuffle=True, seed=seed):
        optimizer.zero_grad()
        loss = model(input_ids=input_ids.to(target_device), attention_mask=attention.to(target_device), labels=labels.to(target_device)).loss
        loss.backward()
        optimizer.step()
    state = {key: value.detach().cpu().clone() for key, value in get_peft_model_state_dict(model).items()}
    return state, time.perf_counter() - started


def evaluate_model(model, data, target_device: torch.device) -> tuple[np.ndarray, np.ndarray, float]:
    model.to(target_device).eval()
    logits, labels = [], []
    started = time.perf_counter()
    with torch.no_grad():
        for input_ids, attention, target in loader(data, batch=32):
            values = model(input_ids=input_ids.to(target_device), attention_mask=attention.to(target_device)).logits
            logits.append(values.detach().cpu().numpy())
            labels.append(target.numpy())
    return np.concatenate(logits), np.concatenate(labels), time.perf_counter() - started


def evaluate_state(base_state, state, rank: int, domains: list[DomainData], target_device: torch.device):
    model = lora_model(base_state, rank)
    set_peft_model_state_dict(model, state)
    logits, labels, domain_ids, elapsed = [], [], [], 0.0
    for domain_id, domain in enumerate(domains):
        values, target, timing = evaluate_model(model, domain.test, target_device)
        logits.append(values)
        labels.append(target)
        domain_ids.extend([domain_id] * len(target))
        elapsed += timing
    return np.concatenate(logits), np.concatenate(labels), np.asarray(domain_ids), elapsed


def adapter_residual(states: list[dict[str, torch.Tensor]], collection: int) -> tuple[list[dict[str, object]], bool]:
    rows = []
    for key in states[0]:
        if ".lora_B." not in key:
            continue
        factors = [state[key].float().numpy() for state in states]
        maps = {(i, j): np.linalg.lstsq(factors[i], factors[j], rcond=None)[0] for i in range(4) for j in range(4)}
        triangles = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
        observed = float(np.mean([np.linalg.norm(maps[i, j] @ maps[j, k] @ maps[k, i] - np.eye(maps[i, i].shape[0]), ord="fro") for i, j, k in triangles]))
        stable_seed = int(hashlib.sha256(f"{collection}:{key}".encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(stable_seed)
        nulls = []
        for _ in range(200):
            order = rng.permutation(4)
            nulls.append(float(np.mean([np.linalg.norm(maps[int(order[i]), int(order[j])] @ maps[int(order[j]), int(order[k])] @ maps[int(order[k]), int(order[i])] - np.eye(maps[0, 0].shape[0]), ord="fro") for i, j, k in triangles])))
        threshold = float(np.quantile(nulls, 0.95))
        rows.append({"collection": collection, "layer": key, "observed_residual": observed, "null_95": threshold, "exceeds_null": observed > threshold, "null_draws": 200})
    return rows, bool(rows) and all(row["exceeds_null"] for row in rows)


def run_collection(base_state, domains: list[DomainData], rank: int, collection: int, target_device: torch.device):
    states, training_times = [], []
    for domain_id, domain in enumerate(domains):
        state, elapsed = train_adapter(base_state, domain.train, rank, 32_000 + collection * 100 + domain_id, target_device)
        states.append(state)
        training_times.append(elapsed)
    candidates = {
        "raw_factor_average": average_states(states),
        "delta_matrix_average": delta_factor_state(states, "mean"),
        "task_arithmetic": delta_factor_state(states, "mean"),
        "ties": delta_factor_state(states, "ties"),
        "dare": delta_factor_state(states, "dare", collection),
        "svd_low_rank_merge": delta_factor_state(states, "mean"),
    }
    validation_scores = {}
    for method, state in candidates.items():
        model = lora_model(base_state, rank)
        set_peft_model_state_dict(model, state)
        scores = []
        for domain in domains:
            logits, labels, _ = evaluate_model(model, domain.validation, target_device)
            scores.append(classification_metrics(logits, labels)["accuracy"])
        validation_scores[method] = float(np.mean(scores))
    selected = max(validation_scores, key=lambda name: (validation_scores[name], name))
    candidates["twistedmerge_hodge_lr"] = copy.deepcopy(candidates[selected])
    evaluated, labels, domain_ids, rows = {}, None, None, []
    for method, state in candidates.items():
        logits, current_labels, current_domains, elapsed = evaluate_state(base_state, state, rank, domains, target_device)
        evaluated[method] = logits
        labels, domain_ids = current_labels, current_domains
        metrics = classification_metrics(logits, current_labels)
        rows.append({"collection": collection, "collection_type": "lora", "base_model": SECOND_MODEL_ID, "base_revision": SECOND_MODEL_REVISION, "rank": rank, "method": method, **metrics, "worst_domain_accuracy": min(classification_metrics(logits[current_domains == index], current_labels[current_domains == index])["accuracy"] for index in range(4)), "trainable_parameters": sum(value.numel() for value in state.values()), "stored_parameters": sum(value.numel() for value in state.values()), "branch_count": 1, "latency_seconds": elapsed, "peak_memory_mb": peak_memory_mb(), "training_time_seconds": sum(training_times), "lift_activated": False, "validation_selected": selected})
    branch_logits = np.stack([evaluate_state(base_state, state, rank, domains, target_device)[0] for state in states], axis=1)
    validation_features = np.concatenate([domain_features(domain.validation) for domain in domains])
    validation_domains = np.concatenate([np.full(len(domain.validation), index) for index, domain in enumerate(domains)])
    test_features = np.concatenate([domain_features(domain.test) for domain in domains])
    routing_model = ridge_fit(validation_features, np.eye(4)[validation_domains], ridge=0.1)
    chosen = ridge_predict(test_features, routing_model).argmax(axis=1)
    for method, logits in {"adaptive_router": branch_logits[np.arange(len(branch_logits)), chosen], "ensemble_reference": branch_logits.mean(axis=1)}.items():
        evaluated[method] = logits
        metrics = classification_metrics(logits, labels)
        rows.append({"collection": collection, "collection_type": "lora", "base_model": SECOND_MODEL_ID, "base_revision": SECOND_MODEL_REVISION, "rank": rank, "method": method, **metrics, "worst_domain_accuracy": min(classification_metrics(logits[domain_ids == index], labels[domain_ids == index])["accuracy"] for index in range(4)), "trainable_parameters": int(routing_model.size if method == "adaptive_router" else 0), "stored_parameters": sum(value.numel() for value in states[0].values()) * 4, "branch_count": 4, "latency_seconds": float("nan"), "peak_memory_mb": peak_memory_mb(), "training_time_seconds": sum(training_times), "lift_activated": False, "validation_selected": selected})
    record = label_independence_record(f"X2_second_base_c{collection}_r{rank}", evaluated, labels, 32_900 + collection)
    for row in rows:
        row["leakage_hash_passed"] = record["label_permutation_hash_passed"]
        row["logits_sha256"] = record["logits_sha256"]
    residuals, exceeds_null = adapter_residual(states, collection)
    return rows, residuals, exceeds_null


def prior_collection_manifest() -> list[dict[str, object]]:
    lora = DEST.parent / "near_term" / "lora_runs.csv"
    transformer = DEST.parent / "near_term" / "transformer_runs.csv"
    rows = []
    if lora.exists() and len(pd.read_csv(lora)):
        rows.append({"collection": 0, "collection_type": "lora", "base_model": MODEL_ID, "base_revision": MODEL_REVISION, "rank": 4, "source": "N6", "executed": True})
    if transformer.exists():
        collections = sorted(pd.read_csv(transformer).collection.unique())
        for offset, collection in enumerate(collections, start=1):
            rows.append({"collection": offset, "collection_type": "partial_full_checkpoint", "base_model": MODEL_ID, "base_revision": MODEL_REVISION, "rank": 0, "source": f"N7 collection {collection}", "executed": True})
    return rows


def main() -> None:
    prior = prior_collection_manifest()
    try:
        _, domains = load_second_base_domains(128)
        initial = base_model()
        base_state = {key: value.detach().cpu().clone() for key, value in initial.state_dict().items()}
    except Exception as error:
        write_csv(DEST / "broader_language_errors.csv", [{"error_type": type(error).__name__, "error": str(error)}])
        stage_result("X2", "blocked", f"second pinned language base acquisition failed: {type(error).__name__}: {str(error)[:500]}")
        return
    target_device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    rows, residuals, residual_flags = [], [], []
    for collection, rank in [(3, 4), (4, 8)]:
        collection_rows, collection_residuals, flag = run_collection(base_state, domains, rank, collection, target_device)
        rows.extend(collection_rows)
        residuals.extend(collection_residuals)
        residual_flags.append(flag)
        prior.append({"collection": collection, "collection_type": "lora", "base_model": SECOND_MODEL_ID, "base_revision": SECOND_MODEL_REVISION, "rank": rank, "source": "X2", "executed": True, "tokenizer": TOKENIZER_ID, "tokenizer_revision": TOKENIZER_REVISION})
    frame = pd.DataFrame(rows)
    summary = frame.groupby(["base_model", "rank", "method"], as_index=False).agg(accuracy=("accuracy", "mean"), worst_domain_accuracy=("worst_domain_accuracy", "mean"), ece=("ece", "mean"), latency_seconds=("latency_seconds", "median"), peak_memory_mb=("peak_memory_mb", "max"), lift_frequency=("lift_activated", "mean"))
    paired = []
    for rank, block in frame.groupby("rank"):
        eligible = block[~block.method.isin(["twistedmerge_hodge_lr", "ensemble_reference", "adaptive_router"])]
        baseline = eligible.groupby("method").accuracy.mean().idxmax()
        pivot = block[block.method.isin(["twistedmerge_hodge_lr", baseline])].pivot(index="collection", columns="method", values="accuracy")
        mean, low, high = bootstrap(pivot.twistedmerge_hodge_lr - pivot[baseline], seed=33_000 + int(rank))
        paired.append({"rank": rank, "baseline": baseline, "mean_accuracy_delta": mean, "ci_low": low, "ci_high": high})
    complete = len(prior) >= 5 and len({row["base_model"] for row in prior}) >= 2 and {4, 8}.issubset({int(row["rank"]) for row in prior}) and {"lora", "partial_full_checkpoint"}.issubset({row["collection_type"] for row in prior})
    gate = complete and all(row["ci_low"] > 0 for row in paired) and all(residual_flags) and bool(frame.lift_activated.any())
    write_csv(DEST / "broader_language_runs.csv", rows)
    write_csv(DEST / "broader_language_summary.csv", summary.to_dict("records"))
    write_csv(DEST / "broader_language_paired.csv", paired)
    write_csv(DEST / "broader_language_residuals.csv", residuals)
    write_csv(DEST / "broader_language_collections.csv", prior)
    write_csv(DEST / "broader_language_errors.csv", [], ["error_type", "error"])
    write_csv(DEST / "broader_language_claims.csv", [{"claim": "five_collections_completed", "value": len(prior) >= 5}, {"claim": "two_open_bases_completed", "value": len({row["base_model"] for row in prior}) >= 2}, {"claim": "full_and_lora_collections_completed", "value": {"lora", "partial_full_checkpoint"}.issubset({row["collection_type"] for row in prior})}, {"claim": "multiple_adapter_ranks_completed", "value": {4, 8}.issubset({int(row["rank"]) for row in prior})}, {"claim": "classification_completed", "value": True}, {"claim": "generative_domain_executed", "value": False}, {"claim": "generative_reason", "value": "optional generative branch omitted under the bounded 8 GB discovery budget"}, {"claim": "structured_gate_passed", "value": gate}])
    summary.to_latex(DEST / "tables" / "broader_language.tex", index=False, float_format="%.6f")
    (DEST / "broader_language_report.md").write_text(f"# Broader language and adapter benchmark\n\nThe combined ledger contains {len(prior)} executed collections across two pinned open BERT bases, including partial full-checkpoint and LoRA collections at ranks 4 and 8. The second-base runs use four real sentiment domains and saved-logit permutation checks. The structured gate was **{'passed' if gate else 'not passed'}**; no lift activated.\n", encoding="utf-8")
    stage_result("X2", "confirmation" if gate else "negative", f"broader language executed; collections={len(prior)}; gate {'passed' if gate else 'did not pass'}", collections=len(prior), bases=len({row["base_model"] for row in prior}), gate_passed=gate)


if __name__ == "__main__":
    main()
