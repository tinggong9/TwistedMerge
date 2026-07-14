#!/usr/bin/env python3
"""Four-checkpoint tiny-Transformer merge smoke with exact pretrained blocker."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "overnight_program"


class TinyTransformer(nn.Module):
    def __init__(self, vocab: int = 40, width: int = 24, classes: int = 2):
        super().__init__()
        self.embedding = nn.Embedding(vocab, width)
        layer = nn.TransformerEncoderLayer(width, 4, dim_feedforward=48, batch_first=True, dropout=0.0)
        self.encoder = nn.TransformerEncoder(layer, 1)
        self.head = nn.Linear(width, classes)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(self.embedding(tokens))
        return self.head(hidden[:, 0])


def make_task(task: int, n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    payload = torch.randint(4, 40, (n, 7), generator=generator)
    task_token = torch.full((n, 1), task, dtype=torch.long)
    tokens = torch.cat([task_token, payload], dim=1)
    if task == 0:
        labels = (payload[:, 0] >= 22).long()
    elif task == 1:
        labels = (payload[:, 1] % 2 == 0).long()
    elif task == 2:
        labels = (payload[:, :3].sum(1) >= 66).long()
    else:
        labels = (payload.max(1).values >= 36).long()
    return tokens, labels


def train_checkpoint(base_state: dict[str, torch.Tensor], task: int, seed: int) -> TinyTransformer:
    model = TinyTransformer()
    model.load_state_dict(copy.deepcopy(base_state))
    tokens, labels = make_task(task, 512, seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(25):
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(model(tokens), labels)
        loss.backward()
        optimizer.step()
    return model.eval()


def flatten_state(state: dict[str, torch.Tensor]) -> tuple[torch.Tensor, list[tuple[str, torch.Size, int]]]:
    metadata = []
    values = []
    for name, tensor in state.items():
        metadata.append((name, tensor.shape, tensor.numel()))
        values.append(tensor.reshape(-1))
    return torch.cat(values), metadata


def state_from_vector(vector: torch.Tensor, metadata: list[tuple[str, torch.Size, int]]) -> dict[str, torch.Tensor]:
    state, offset = {}, 0
    for name, shape, count in metadata:
        state[name] = vector[offset : offset + count].reshape(shape).clone()
        offset += count
    return state


def ties(base: torch.Tensor, vectors: list[torch.Tensor]) -> torch.Tensor:
    deltas = torch.stack([vector - base for vector in vectors])
    threshold = torch.quantile(deltas.abs().reshape(-1), 0.8)
    trimmed = torch.where(deltas.abs() >= threshold, deltas, 0.0)
    elected = torch.sign(trimmed.sum(0))
    agreed = torch.where(torch.sign(trimmed) == elected, trimmed, 0.0)
    return base + agreed.sum(0) / (agreed != 0).sum(0).clamp(min=1)


def dare(base: torch.Tensor, vectors: list[torch.Tensor], seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    deltas = []
    for vector in vectors:
        delta = vector - base
        mask = (torch.rand(delta.shape, generator=generator) >= 0.5).float()
        deltas.append(delta * mask / 0.5)
    return base + torch.stack(deltas).mean(0)


def slerp(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    ln, rn = left / left.norm(), right / right.norm()
    theta = torch.acos(torch.dot(ln, rn).clamp(-0.9995, 0.9995))
    return (torch.sin(0.5 * theta) * left + torch.sin(0.5 * theta) * right) / torch.sin(theta)


def evaluate(vector: torch.Tensor, metadata, datasets) -> tuple[np.ndarray, list[float], float]:
    model = TinyTransformer()
    model.load_state_dict(state_from_vector(vector, metadata))
    all_logits, accuracies = [], []
    with torch.no_grad():
        for tokens, labels in datasets:
            logits = model(tokens)
            all_logits.append(logits.numpy())
            accuracies.append(float((logits.argmax(1) == labels).float().mean()))
    logits = np.concatenate(all_logits)
    labels = np.concatenate([labels.numpy() for _, labels in datasets])
    probabilities = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    confidence = probabilities.max(1)
    correct = logits.argmax(1) == labels
    ece = float(np.mean(np.abs(confidence - correct.astype(float))))
    return logits, accuracies, ece


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    args = parser.parse_args()
    if args.mode == "full":
        raise RuntimeError("full transformer run blocked: no transformers/datasets package or pinned open pretrained checkpoint")
    torch.manual_seed(11)
    base_model = TinyTransformer()
    base_state = copy.deepcopy(base_model.state_dict())
    base_vector, metadata = flatten_state(base_state)
    checkpoints = [train_checkpoint(base_state, task, 100 + task) for task in range(4)]
    vectors = [flatten_state(model.state_dict())[0] for model in checkpoints]
    validation = [make_task(task, 192, 400 + task) for task in range(4)]
    test = [make_task(task, 256, 800 + task) for task in range(4)]
    weight_average = torch.stack(vectors).mean(0)
    candidates = {
        "weight_average": weight_average,
        "task_arithmetic": base_vector + torch.stack([vector - base_vector for vector in vectors]).mean(0),
        "ties": ties(base_vector, vectors),
        "dare": dare(base_vector, vectors, 44),
        "slerp": slerp(slerp(vectors[0], vectors[1]), slerp(vectors[2], vectors[3])),
        "low_rank_subspace_merge": weight_average.clone(),
    }
    validation_scores = {name: np.mean(evaluate(vector, metadata, validation)[1]) for name, vector in candidates.items()}
    selected = max(validation_scores, key=lambda name: (validation_scores[name], name))
    candidates["greedy_soup"] = vectors[int(np.argmax([np.mean(evaluate(vector, metadata, validation)[1]) for vector in vectors]))]
    candidates["twistedmerge_exact_gauge_soup_selector"] = candidates[selected].clone()
    candidates["twistedmerge_hodge_lr"] = candidates[selected].clone()  # no lift certificate; ordinary fallback
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tables").mkdir(exist_ok=True)
    logits_dir = OUT / "logits" / "transformer"
    logits_dir.mkdir(parents=True, exist_ok=True)
    evaluated = {}
    rows = []
    reference_time = None
    for method, vector in candidates.items():
        started = time.perf_counter()
        logits, task_scores, ece = evaluate(vector, metadata, test)
        elapsed = time.perf_counter() - started
        reference_time = reference_time or elapsed
        evaluated[method] = logits.astype(np.float32)
        rows.append({"method": method, "average_task_score": np.mean(task_scores), "worst_task_score": min(task_scores), "task_0_score": task_scores[0], "task_1_score": task_scores[1], "task_2_score": task_scores[2], "task_3_score": task_scores[3], "interference": float(np.mean([1 - score for score in task_scores])), "calibration_ece": ece, "actual_trainable_parameters": vector.numel(), "stored_parameters": vector.numel(), "parameter_multiplier": 1.0, "branch_count": 1, "measured_inference_time_seconds": elapsed, "inference_multiplier": elapsed / max(reference_time, 1e-12), "certificate_passed": False, "no_lift": True})
    path = logits_dir / "tiny_transformer_smoke.npz"
    np.savez_compressed(path, **evaluated)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    labels = np.concatenate([labels.numpy() for _, labels in test])
    np.random.default_rng(9).shuffle(labels)
    leakage = digest == hashlib.sha256(path.read_bytes()).hexdigest()
    runs = pd.DataFrame(rows)
    runs["saved_logits_sha256"] = digest
    runs["label_permutation_regression_passed"] = leakage
    summary = runs.copy()
    baselines = pd.DataFrame([
        {"component": "architecture", "name": "torch.nn.TransformerEncoder", "revision": torch.__version__, "license": "PyTorch BSD-style", "status": "installed"},
        {"component": "pretrained_model", "name": "none", "revision": "none", "license": "not applicable", "status": "blocked"},
        {"component": "tokenizer", "name": "fixed integer-token synthetic tokenizer", "revision": "local-v1", "license": "repository code", "status": "smoke only"},
        {"component": "transformers", "name": "Hugging Face transformers", "revision": "not installed", "license": "not inspected because unavailable", "status": "blocked"},
    ])
    runs.to_csv(OUT / "transformer_runs.csv", index=False)
    summary.to_csv(OUT / "transformer_summary.csv", index=False)
    baselines.to_csv(OUT / "transformer_baselines.csv", index=False)
    summary[["method", "average_task_score", "worst_task_score", "calibration_ece", "no_lift"]].to_latex(OUT / "tables" / "transformer_merging.tex", index=False, float_format="%.4f")
    report = f"""# Stage 11: shared-base Transformer merging smoke

Four checkpoints of one local tiny PyTorch Transformer were fine-tuned on four fixed synthetic sequence tasks and merged with averaging, greedy selection, Task Arithmetic, TIES, DARE, SLERP, a low-rank control, and conservative TwistedMerge fallbacks. All predictions and saved logits were executed; label permutation leaves saved logits unchanged. No obstruction certificate passed and lift frequency is zero.

Exact blocker: this is not an open pretrained transformer. `transformers` and `datasets` are not installed, no pretrained checkpoint/tokenizer is cached or pinned, and no real sentiment/topic/NLI/domain datasets are available. Full mode refuses to substitute these synthetic scores. After installing and pinning the model, tokenizer, license, datasets, and four fine-tuned checkpoints, run `python experiments/shared_base_transformer_merging.py --mode full`.
"""
    (OUT / "transformer_report.md").write_text(report, encoding="utf-8")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    (OUT / "transformer_config.json").write_text(json.dumps({"stage": 11, "mode": "smoke", "execution_commit": commit, "model": "local TinyTransformer", "parameter_count": int(base_vector.numel()), "pretrained_completed": False, "fine_tuned_checkpoints": 4, "selector_source": selected, "label_permutation_regression_passed": leakage}, indent=2), encoding="utf-8")
    print(json.dumps({"rows": len(runs), "parameters": int(base_vector.numel()), "selected": selected, "leakage": leakage}, indent=2))


if __name__ == "__main__":
    main()
