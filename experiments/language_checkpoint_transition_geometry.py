#!/usr/bin/env python3
"""C3: hidden-subspace geometry of four real partially fine-tuned BERT checkpoints."""

from __future__ import annotations

import copy
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.future_text_common import DATASETS, MODEL_ID, MODEL_REVISION, base_model, evaluate, load_domains, loader
from experiments.next_program_common import OUT, TMP, classification_metrics, git_head, paired_bootstrap, provenance, save_logits_before_labels, torch_device, write_csv, write_json

SCRIPT = Path(__file__).resolve()
DEST = OUT / "extended"
DEVICE = torch_device()
SUBSPACES = ("attention_output", "mlp_intermediate", "final_hidden_state")


def train_checkpoint(base_state, dataset, seed: int):
    torch.manual_seed(seed); model = base_model(); model.load_state_dict(copy.deepcopy(base_state))
    for name, parameter in model.named_parameters(): parameter.requires_grad = "encoder.layer.1" in name or "classifier" in name
    model.to(DEVICE).train(); optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=4e-4)
    started = time.perf_counter()
    for input_ids, attention, labels in loader(dataset, batch=16, shuffle=True, seed=seed):
        optimizer.zero_grad(set_to_none=True); loss = model(input_ids=input_ids.to(DEVICE), attention_mask=attention.to(DEVICE), labels=labels.to(DEVICE)).loss; loss.backward(); optimizer.step()
    return model.eval(), time.perf_counter() - started


def hidden_subspaces(model, dataset):
    storage = {name: [] for name in SUBSPACES}; handles = []
    handles.append(model.bert.encoder.layer[-1].attention.output.dense.register_forward_hook(lambda module, inputs, output: storage["attention_output"].append(output[:, 0].detach().cpu())))
    handles.append(model.bert.encoder.layer[-1].intermediate.dense.register_forward_hook(lambda module, inputs, output: storage["mlp_intermediate"].append(output[:, 0].detach().cpu())))
    with torch.no_grad():
        for input_ids, attention, _ in loader(dataset, batch=32):
            output = model(input_ids=input_ids.to(DEVICE), attention_mask=attention.to(DEVICE), output_hidden_states=True, return_dict=True)
            storage["final_hidden_state"].append(output.hidden_states[-1][:, 0].detach().cpu())
    for handle in handles: handle.remove()
    return {name: torch.cat(values).numpy() for name, values in storage.items()}


def reduced(values: list[np.ndarray], width: int = 24):
    joined = np.concatenate(values); mean = joined.mean(0, keepdims=True); _, _, vt = np.linalg.svd(joined - mean, full_matrices=False); basis = vt[: min(width, vt.shape[0])].T
    return [(value - mean) @ basis for value in values], basis


def maps(values: list[np.ndarray], indices=None):
    chosen = np.arange(len(values[0])) if indices is None else indices; output = {}
    for left in range(4):
        for right in range(4):
            if left == right: continue
            u, _, vt = np.linalg.svd(values[left][chosen].T @ values[right][chosen], full_matrices=False); output[left, right] = u @ vt
    return output


def diagnostics(fitted):
    identity = np.eye(next(iter(fitted.values())).shape[0]); cycles = []
    for a, b, c in ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)): cycles.append(fitted[a, b] @ fitted[b, c] @ fitted[c, a] - identity)
    residual = max(np.linalg.norm(value, "fro") / np.sqrt(len(identity)) for value in cycles); singular = np.linalg.svd(np.concatenate(cycles), compute_uv=False); rank = int(np.sum(singular > max(1e-6, singular[0] * 0.05)))
    return float(residual), rank


def merge_states(base_state, states):
    output = {}
    for name, base in base_state.items(): output[name] = torch.stack([state[name].float() for state in states]).mean(0).to(base.dtype) if torch.is_floating_point(base) else states[0][name]
    return output


def main() -> None:
    _, domains = load_domains(160); initial = base_model(); base_state = {name: value.detach().cpu().clone() for name, value in initial.state_dict().items()}
    models = []; states = []; training_times = []
    for index, domain in enumerate(domains):
        model, elapsed = train_checkpoint(base_state, domain.train, 151_000_000 + index); models.append(model); training_times.append(elapsed); states.append({name: value.detach().cpu().clone() for name, value in model.state_dict().items()})
    calibration = domains[0].validation
    hidden = [hidden_subspaces(model, calibration) for model in models]
    checkpoint_dir = TMP / "checkpoints" / "language"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "checkpoints": states,
            "datasets": [{"name": name, "id": dataset_id, "revision": revision} for name, dataset_id, revision in DATASETS],
        },
        checkpoint_dir / "partial_finetunes.pt",
    )
    np.savez_compressed(
        checkpoint_dir / "calibration_features.npz",
        **{
            f"checkpoint_{index}_{subspace}": values[subspace]
            for index, values in enumerate(hidden)
            for subspace in SUBSPACES
        },
    )
    transitions = []; stability = []; nulls = []; final_basis = None; final_maps = None
    for subspace in SUBSPACES:
        values, basis = reduced([item[subspace] for item in hidden]); fitted = maps(values); residual, rank = diagnostics(fitted)
        fit = np.mean([np.linalg.norm(values[i] @ fitted[i, j] - values[j]) / max(np.linalg.norm(values[j]), 1e-12) for i in range(4) for j in range(i + 1, 4)])
        transitions.append({"subspace": subspace, "pairwise_fit": float(fit), "cycle_residual": residual, "residual_rank": rank, "calibration_examples": len(values[0]), "adapter_subspace": False})
        rng = np.random.default_rng(151_100_000 + len(subspace))
        for resample in range(5):
            indices = rng.choice(len(values[0]), size=len(values[0]), replace=True); sampled_residual, sampled_rank = diagnostics(maps(values, indices)); stability.append({"subspace": subspace, "resample": resample, "cycle_residual": sampled_residual, "residual_rank": sampled_rank})
        edge_values = list(fitted.values()); identity = np.eye(edge_values[0].shape[0])
        for draw in range(200):
            chosen = rng.choice(len(edge_values), 3, replace=False); statistic = np.linalg.norm(edge_values[chosen[0]] @ edge_values[chosen[1]] @ edge_values[chosen[2]] - identity, "fro") / np.sqrt(len(identity)); nulls.append({"subspace": subspace, "null_family": "edge_shuffle", "draw": draw, "statistic": float(statistic)})
        if subspace == "final_hidden_state": final_basis, final_maps = basis, fitted
    merged = base_model(); merged.load_state_dict(merge_states(base_state, states)); merged.to(DEVICE).eval()
    logits = []; labels = []; domain_ids = []
    for domain_index, domain in enumerate(domains):
        values, target, _ = evaluate(merged, domain.test, DEVICE); logits.append(values); labels.append(target); domain_ids.extend([domain_index] * len(target))
    merged_logits = np.concatenate(logits); all_labels = np.concatenate(labels)
    # Feature-aligned correction is executed on each domain's own test inputs.
    corrected_parts = []
    for domain in domains:
        per_model = [hidden_subspaces(model, domain.test)["final_hidden_state"] for model in models]
        joined_mean = np.concatenate([item["final_hidden_state"] for item in hidden]).mean(0, keepdims=True)
        projected = [(value - joined_mean) @ final_basis for value in per_model]
        aligned = [projected[0]] + [projected[index] @ final_maps[index, 0] for index in range(1, 4)]
        reconstructed = np.mean(aligned, axis=0) @ final_basis.T + joined_mean
        with torch.no_grad(): corrected_parts.append(models[0].classifier(models[0].dropout(torch.tensor(reconstructed, dtype=torch.float32, device=DEVICE))).cpu().numpy())
    corrected_logits = np.concatenate(corrected_parts)
    candidates = {"weight_average": merged_logits, "feature_aligned_correction": corrected_logits}
    ledger = save_logits_before_labels("language_checkpoint_geometry", candidates, all_labels, 151_900_000)
    runs = []
    for name, values in candidates.items(): runs.append({"method": name, **classification_metrics(values, all_labels), "worst_domain_accuracy": min(classification_metrics(values[np.asarray(domain_ids) == index], all_labels[np.asarray(domain_ids) == index])["accuracy"] for index in range(4)), "training_time_seconds": sum(training_times), "logits_sha256": ledger["logits_sha256"], "label_permutation_hash_passed": bool(ledger["candidate_hashes_unchanged"] and ledger["file_hash_unchanged"]), **provenance(SCRIPT, "python experiments/language_checkpoint_transition_geometry.py", 0)})
    residual_survives = all(float(row["cycle_residual"]) > max(float(null["statistic"]) for null in nulls if null["subspace"] == row["subspace"]) for row in transitions)
    rank_stable = all(len({row["residual_rank"] for row in stability if row["subspace"] == subspace}) == 1 for subspace in SUBSPACES)
    improvement = next(row["accuracy"] for row in runs if row["method"] == "feature_aligned_correction") - next(row["accuracy"] for row in runs if row["method"] == "weight_average")
    claims = [{"claim": "residual_exceeds_nulls", "value": residual_survives}, {"claim": "rank_stable", "value": rank_stable}, {"claim": "correction_reduces_interference", "value": improvement > 0}, {"claim": "adapter_subspace_available", "value": False}, {"claim": "complete_language_gate_passed", "value": residual_survives and rank_stable and improvement > 0}]
    write_csv(DEST / "language_runs.csv", runs); write_csv(DEST / "language_transitions.csv", transitions); write_csv(DEST / "language_stability.csv", stability); write_csv(DEST / "language_nulls.csv", nulls); write_csv(DEST / "language_claims.csv", claims)
    write_json(DEST / "language_manifest.json", {"model": MODEL_ID, "revision": MODEL_REVISION, "datasets": [{"name": name, "id": dataset_id, "revision": revision} for name, dataset_id, revision in DATASETS], "examples_per_domain": 160, "execution_commit": git_head()})
    (DEST / "language_report.md").write_text(
        "# Language checkpoint transition geometry\n\n"
        f"Execution commit: `{git_head()}`. Four checkpoints from pinned `{MODEL_ID}` revision `{MODEL_REVISION}` were "
        "partially fine-tuned on real SST-2, IMDb, Yelp, and Amazon sentiment subsets. Attention-output, MLP-intermediate, "
        "and final-hidden subspaces were measured with five resamples and 200 edge-shuffle nulls. No adapter was present, "
        f"so adapter-subspace results are marked unavailable. The aligned correction changed accuracy by `{improvement:+.6f}`; "
        f"the complete gate {'passed' if residual_survives and rank_stable and improvement > 0 else 'did not pass'}.\n",
        encoding="utf-8",
    )


if __name__ == "__main__": main()
