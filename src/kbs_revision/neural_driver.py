"""Bounded CLI operations for real local training and software-only integration."""
from __future__ import annotations

import gc
import json
from pathlib import Path
import time

import torch

from .neural_runner import StageProfiler, atomic_json, execute_config, source_fingerprint
from .protocol import ROLES, digest, file_hash, load_config
from .task_data import load_split, prepare_splits, software_only_fixture_records


def software_smoke(out, root):
    from .language_backend import backend_smoke, create_tiny_fixture, train_local_adapter
    from .external_methods import verify_external_parity
    out, root = Path(out), Path(root)
    begin = time.perf_counter()
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        records, sources = software_only_fixture_records()
        split_dir = out / "software_only_splits"
        splits = prepare_splits(records, sources, split_dir, seed=17, counts=dict.fromkeys(ROLES, 1))
        atomic_json(out / "split_manifest.json", {"status": "software_only", "real_benchmark_status": "blocked",
            "software_manifest": str(split_dir.relative_to(out) / "split_manifest.json"),
            "sha256": file_hash(split_dir / "split_manifest.json"), "near_duplicate_audit": splits["near_duplicate_audit"],
            "final_test_evaluated": False})
        train, development, selection = [load_split(split_dir / "split_manifest.json", role) for role in ("train", "pilot", "D_select")]
        smoke = backend_smoke(out / "backend_fixture", {"train": [r for r in train if r["task_id"] == "BoolQ"], "pilot": development})
        if not smoke["passed"]:
            raise ValueError("backend software smoke failed")
        atomic_json(out / "backend_smoke.json", smoke)
        external = verify_external_parity()
        atomic_json(out / "external_method_parity.json", external)
        if not external["passed"]:
            raise ValueError("external fixture parity failed")
        cohort = out / "software_cohort"
        cohort.mkdir(exist_ok=False)
        base = cohort / "base"
        base_manifest = create_tiny_fixture(base, train + development + selection)
        atomic_json(cohort / "model_manifest.json", base_manifest)
        adapter_specs, training_costs = [], []
        timer = StageProfiler("cpu")
        for index, task in enumerate(sorted(splits["tasks"])):
            path = cohort / task
            with timer.stage("specialist_training", task_id=task):
                model, tokenizer, adapter = train_local_adapter(base, base_manifest,
                    [r for r in train if r["task_id"] == task], seed=901 + index,
                    rank=2, alpha=4, dropout=0, steps=4, out=path,
                    group_id="software-only-cohort-1", software_only=True)
            adapter_specs.append({**adapter, "path": str(path.resolve())})
            training_costs.append({"task_id": task, **adapter["training"]})
            del model, tokenizer
            gc.collect()
        atomic_json(out / "model_and_adapter_manifest.json", {"status": "software_only",
            "base": base_manifest, "adapters": adapter_specs, "independent_research_training_groups": 0,
            "planned_real_base": "Qwen/Qwen2.5-1.5B", "real_cohort_status": "blocked"})
        config = {"phase": "software_only", "backend": "modern_language_v1", "device": "cpu", "dtype": "float32",
            "model": {"local_path": str(base.resolve())}, "model_manifest_path": str((cohort / "model_manifest.json").resolve()),
            "split_manifest_path": str((split_dir / "split_manifest.json").resolve()),
            "candidate_methods": ["compact_qr_svd", "dense_reference_svd", "exact_concatenation", "task_arithmetic", "ties",
                                  "dare_ties", "naive_factor_average", "pairwise_alignment", "global_synchronization", "knots", "core_space"],
            "method_options": {"knots": {"density": 1.0}, "core_space": {"density": .5},
                               "ties": {"density": .5}, "dare_ties": {"density": .5, "seed": 0}},
            "output_rank": 2, "merge_weights": [.25] * 4, "fallback": "compact_qr_svd", "test_access": False,
            "policies": ["P0", "P1", "P2", "P3", "P4"], "pairwise_threshold": .05,
            "resource_limits": {"cpu_seconds": 300, "disk_bytes": 200000000, "pilot_accelerator_hours": 0},
            "episodes": [{"episode_id": "software-four-task-cohort", "tasks": sorted(splits["tasks"]), "adapters": adapter_specs}]}
        atomic_json(out / "software_episode_config.json", config)
        results = execute_config(config, out / "software_episodes", root=root)
        resumed = execute_config(config, out / "software_episodes", root=root, resume=True)
        if not all(row["reused"] for row in resumed):
            raise ValueError("completed episodes were duplicated during resume")
        result = {"status": "software_only", "passed": True, "backend_smoke": "backend_smoke.json",
            "external_parity": "external_method_parity.json", "episodes": results,
            "resume_reused_without_reexecution": True, "training_costs": training_costs,
            "training_stage_costs": timer.rows, "source_hashes": source_fingerprint(root),
            "wall_seconds": time.perf_counter() - begin, "accelerator_hours": 0,
            "final_test_evaluated": False, "independent_research_training_groups": 0,
            "research_pilot_status": "blocked"}
        atomic_json(out / "software_integration.json", result)
        return result
    finally:
        torch.set_num_threads(previous_threads)


def prepare_local_splits(config_path, out, root):
    config = load_config(config_path)
    raw = Path(root) / config["dataset_records_path"]
    if file_hash(raw) != config["dataset_records_sha256"]:
        raise ValueError("local dataset input hash mismatch")
    records = [json.loads(line) for line in raw.read_text().splitlines() if line.strip()]
    return prepare_splits(records, config["source_metadata"], out, seed=config["split_seed"],
                          counts=config.get("split_counts"), ratios=config.get("split_ratios"))


def train_cohort(config_path, out, root):
    """Train a predeclared, local compatible task cohort; no final-test loading."""
    from .language_backend import train_local_adapter, validate_base_manifest
    from .resource_guard import preflight, BudgetGuard
    config = load_config(config_path)
    root, out = Path(root), Path(out)
    errors = preflight(config, root)
    if errors:
        raise ValueError("training resource admission failed: " + "; ".join(errors))
    budget = BudgetGuard(config, out)
    base_path = root / config["model"]["local_path"]
    base_manifest = json.loads((root / config["model_manifest_path"]).read_text())
    validate_base_manifest(base_path, base_manifest)
    records = load_split(root / config["split_manifest_path"], "train")
    recipe = config["training"]
    specs, timer = [], StageProfiler(config.get("device", "cpu"))
    for group in config["training_groups"]:
        for task in config["tasks"]:
            budget.check()
            seed = group["task_seeds"][task]
            path = out / group["group_id"] / task
            with timer.stage("specialist_training", task_id=task, group_id=group["group_id"]):
                model, tokenizer, spec = train_local_adapter(base_path, base_manifest,
                    [r for r in records if r["task_id"] == task], seed=seed, out=path,
                    group_id=group["group_id"], device=config.get("device", "cpu"),
                    dtype={"float32": torch.float32, "bfloat16": torch.bfloat16}[config.get("dtype", "float32")], **recipe)
            specs.append({**spec, "path": str(path.resolve())})
            del model, tokenizer
            gc.collect()
    value = {"status": "trained", "base_manifest": base_manifest, "adapters": specs,
             "costs": timer.rows, "final_test_accessed": False, "configuration_sha256": digest(config),
             "resource_receipt": budget.check()}
    atomic_json(out / "model_and_adapter_manifest.json", value)
    return value
