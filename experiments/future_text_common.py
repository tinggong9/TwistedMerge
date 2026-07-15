#!/usr/bin/env python3
"""Pinned, bounded real-text data and model helpers for future benchmarks."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from experiments.future_benchmark_common import DATA

MODEL_ID = "google/bert_uncased_L-2_H-128_A-2"
MODEL_REVISION = "30b0a37ccaaa32f332884b96992754e246e48c5f"
DATASETS = [
    ("sst2", "stanfordnlp/sst2", "8d51e7e4887a4caaa95b3fbebbf53c0490b58bbb"),
    ("imdb", "imdb", "e6281661ce1c48d982bc483cf8a173c1bbeb5d31"),
    ("yelp", "fancyzhx/yelp_polarity", "bbf1c97a1f0cf005e5aded43839fd814654a1557"),
    ("amazon", "fancyzhx/amazon_polarity", "9d9c45c18f8c3cf1b23a3c27917b60cbf28f3289"),
]


@dataclass
class DomainData:
    name: str
    train: TensorDataset
    validation: TensorDataset
    test: TensorDataset


def configure_hf() -> None:
    root = DATA / "huggingface"
    root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(root))
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "20")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def text_and_label(row: dict, name: str) -> tuple[str, int]:
    if name == "sst2": return str(row["sentence"]), int(row["label"])
    if name == "imdb": return str(row["text"]), int(row["label"])
    if name == "yelp": return str(row["text"]), int(row["label"])
    return f"{row.get('title', '')} {row.get('content', '')}", int(row["label"])


def tensor_dataset(tokenizer, texts: list[str], labels: list[int]) -> TensorDataset:
    encoded = tokenizer(texts, padding="max_length", truncation=True, max_length=64, return_tensors="pt")
    return TensorDataset(encoded["input_ids"], encoded["attention_mask"], torch.tensor(labels, dtype=torch.long))


def load_domains(per_domain: int = 320) -> tuple[object, list[DomainData]]:
    configure_hf()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    domains = []
    for name, dataset_id, revision in DATASETS:
        stream = load_dataset(dataset_id, split="train", streaming=True, revision=revision)
        rows = list(stream.take(per_domain))
        pairs = [text_and_label(row, name) for row in rows]
        texts, labels = [item[0] for item in pairs], [item[1] for item in pairs]
        train_end, validation_end = per_domain // 2, 3 * per_domain // 4
        domains.append(DomainData(name, tensor_dataset(tokenizer, texts[:train_end], labels[:train_end]), tensor_dataset(tokenizer, texts[train_end:validation_end], labels[train_end:validation_end]), tensor_dataset(tokenizer, texts[validation_end:], labels[validation_end:])))
    return tokenizer, domains


def base_model():
    configure_hf()
    return AutoModelForSequenceClassification.from_pretrained(MODEL_ID, revision=MODEL_REVISION, num_labels=2)


def loader(dataset: TensorDataset, batch: int = 16, shuffle: bool = False, seed: int = 0) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch, shuffle=shuffle, generator=generator)


def evaluate(model, dataset: TensorDataset, device: torch.device) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval(); outputs, labels = [], []
    started = torch.mps.Event(enable_timing=False) if False else None
    import time
    begin = time.perf_counter()
    with torch.no_grad():
        for input_ids, attention, target in loader(dataset, batch=32):
            logits = model(input_ids=input_ids.to(device), attention_mask=attention.to(device)).logits
            outputs.append(logits.detach().cpu().numpy()); labels.append(target.numpy())
    return np.concatenate(outputs), np.concatenate(labels), time.perf_counter() - begin


def domain_features(dataset: TensorDataset) -> np.ndarray:
    ids, attention, _ = dataset.tensors
    length = attention.sum(1).numpy().astype(float)
    mean = (ids * attention).sum(1).numpy() / np.maximum(length, 1)
    std = np.sqrt((((ids.numpy() - mean[:, None]) * attention.numpy()) ** 2).sum(1) / np.maximum(length, 1))
    return np.column_stack([length, mean / 30_000, std / 30_000, np.ones(len(ids))])
