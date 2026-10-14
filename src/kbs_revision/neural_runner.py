"""Atomic episode receipts and measured matched-candidate language execution.

This module never downloads data/weights. Pilot evaluation uses development
roles; final D_test evaluation requires an explicit confirmatory configuration.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import resource
import tempfile
import threading
import time
import traceback
import uuid

import psutil
import torch

from .protocol import canonical_bytes, digest, file_hash


def atomic_json(path, value):
    """Publish once through an fsynced temporary file and exclusive hard link."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)  # EEXIST prevents replacement, including races.
    finally:
        Path(temporary).unlink(missing_ok=True)


def source_fingerprint(root):
    root = Path(root)
    files = [root / "experiments/kbs_revision.py"] + sorted((root / "src/kbs_revision").glob("*.py"))
    return {str(path.relative_to(root)): file_hash(path) for path in files}


class EpisodeStore:
    def __init__(self, out, configuration, *, inputs):
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.fingerprint = digest({"configuration": configuration, "inputs": inputs})
        receipt = self.out / "run_identity.json"
        expected = {"fingerprint": self.fingerprint, "configuration": configuration, "inputs": inputs}
        if receipt.exists():
            if json.loads(receipt.read_text()) != expected:
                raise ValueError("run inputs/configuration/source differ; use a fresh versioned run")
        else:
            atomic_json(receipt, expected)

    def _path(self, episode_id):
        if not episode_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in episode_id):
            raise ValueError("episode id must be a safe identifier")
        return self.out / "episodes" / episode_id

    def completed(self, episode_id):
        path = self._path(episode_id)
        if not path.exists():
            return None
        receipt = json.loads((path / "completed.json").read_text())
        if receipt["fingerprint"] != self.fingerprint or receipt["episode_id"] != episode_id:
            raise ValueError("completed episode fingerprint mismatch")
        if set(receipt["artifacts"]) != {p.name for p in path.iterdir() if p.name != "completed.json"}:
            raise ValueError("completed episode artifact set changed")
        for name, expected in receipt["artifacts"].items():
            if file_hash(path / name) != expected:
                raise ValueError("completed episode checksum mismatch")
        return json.loads((path / "result.json").read_text())

    def execute(self, episode_id, function, *, resume=False):
        existing = self.completed(episode_id)
        if existing is not None:
            if not resume:
                raise FileExistsError("episode already completed; request --resume")
            return existing, True
        path = self._path(episode_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        lock = path.parent / f".{episode_id}.lock"
        if lock.exists():
            owner = json.loads(lock.read_text())
            if owner.get("fingerprint") != self.fingerprint:
                raise ValueError("lock belongs to another run fingerprint")
            if psutil.pid_exists(owner["pid"]):
                raise FileExistsError("episode is locked by a live process")
            lock.unlink()  # Only a dead owner with this run's fingerprint.
        lock_descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.write(lock_descriptor, canonical_bytes({"pid": os.getpid(), "fingerprint": self.fingerprint}))
        temporary = Path(tempfile.mkdtemp(prefix=f".{episode_id}-", dir=path.parent))
        try:
            value = function()
            if not isinstance(value, dict):
                raise TypeError("episode callback must return a JSON mapping")
            atomic_json(temporary / "result.json", value)
            receipt = {"fingerprint": self.fingerprint, "episode_id": episode_id,
                       "artifacts": {"result.json": file_hash(temporary / "result.json")}}
            atomic_json(temporary / "completed.json", receipt)
            os.rename(temporary, path)
            return value, False
        except Exception as error:
            failure = {"fingerprint": self.fingerprint, "episode_id": episode_id,
                       "status": "failed", "error_type": type(error).__name__, "error": str(error),
                       "traceback": traceback.format_exc(), "time_unix": time.time()}
            failure_path = self.out / "failures" / f"{episode_id}-{uuid.uuid4().hex}.json"
            atomic_json(failure_path, failure)
            # Failed/partial bytes are retained for diagnosis, not recognized as complete.
            partial_path = failure_path.with_suffix(".partial")
            os.rename(temporary, partial_path)
            raise
        finally:
            os.close(lock_descriptor)
            lock.unlink(missing_ok=True)


def synchronize(device):
    kind = torch.device(device).type
    if kind == "cuda":
        torch.cuda.synchronize(device)
    elif kind == "mps":
        torch.mps.synchronize()


class StageProfiler:
    """Synchronized wall times, sampled host RSS and stage accelerator peaks."""
    def __init__(self, device="cpu"):
        self.device = torch.device(device)
        self.rows = []

    @contextmanager
    def stage(self, stage, **labels):
        synchronize(self.device)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        process = psutil.Process()
        peak_rss = [process.memory_info().rss]
        stop = threading.Event()
        def sample():
            while not stop.wait(.005):
                peak_rss[0] = max(peak_rss[0], process.memory_info().rss)
        monitor = threading.Thread(target=sample, daemon=True)
        monitor.start()
        begin = time.perf_counter()
        status = "completed"
        try:
            yield
        except Exception:
            status = "failed"
            raise
        finally:
            synchronize(self.device)
            seconds = time.perf_counter() - begin
            stop.set()
            monitor.join()
            peak_rss[0] = max(peak_rss[0], process.memory_info().rss)
            self.rows.append({"stage": stage, "seconds": seconds, "device": str(self.device),
                "cpu_peak_rss_bytes": peak_rss[0], "rss_sampling_seconds": .005,
                "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(self.device) if self.device.type == "cuda" else None,
                "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(self.device) if self.device.type == "cuda" else None,
                "mps_current_allocated_bytes": torch.mps.current_allocated_memory() if self.device.type == "mps" else None,
                "mps_peak_limitation": "MPS API does not expose CUDA-style stage peak allocated/reserved" if self.device.type == "mps" else None,
                "status": status, **labels})

    def seconds(self, stages):
        return sum(row["seconds"] for row in self.rows if row["stage"] in stages)


def evaluate_summary(value):
    """Normalize the backend's measured example records without changing metrics."""
    rows = value["rows"] if isinstance(value, dict) else value
    if not rows:
        raise ValueError("no scored examples; cannot create a successful metric")
    tasks = {}
    for row in rows:
        tasks.setdefault(row["task_id"], []).append(row)
    losses = {task: sum(float(r["bounded_loss"]) for r in values) / len(values) for task, values in tasks.items()}
    return {"rows": rows, "per_task_loss": losses, "macro_loss": sum(losses.values()) / len(losses)}


def execute_episode(configuration, base_manifest, adapter_specs, select_records, evaluation_records, *, phase="pilot", budget_check=lambda: None):
    """One matched candidate episode, using immutable trained adapters.

    P3 is the fully gated Hodge rule (old ungated P3 is retained only in legacy
    policy code). P3/P4 receive identical information. Without a justified
    coefficient/uncertainty model the structural branch abstains. This does not
    invent a latent finite class or an application-grounded overlap geometry.
    """
    from . import language_backend as backend
    from .policies import Candidate, Evidence, decide
    device = configuration.get("device", "cpu")
    dtype = {"float32": torch.float32, "float64": torch.float64, "bfloat16": torch.bfloat16}[configuration.get("dtype", "float32")]
    profiler = StageProfiler(device)
    cohorts, specialist_metrics = [], []
    base_path = configuration["model"]["local_path"]
    with profiler.stage("reference_model_loading", method_id="base"):
        ordinary, tokenizer = backend.load_local_base(base_path, base_manifest, device=device, dtype=dtype,
                                                       software_only=phase == "software_only")
    with profiler.stage("base_inference", method_id="base"):
        base_metrics = evaluate_summary(backend.evaluate_model(ordinary, tokenizer, evaluation_records))
    del ordinary
    gc.collect()
    for spec in adapter_specs:
        budget_check()
        with profiler.stage("model_loading", method_id="specialist", task_id=spec["task_id"]):
            base, tokenizer = backend.load_local_base(base_path, base_manifest, device=device, dtype=dtype,
                                                      software_only=phase == "software_only")
            adapted = backend.load_local_adapter(base, spec["path"], spec, base_manifest, software_only=phase == "software_only")
        with profiler.stage("specialist_inference", method_id="specialist", task_id=spec["task_id"]):
            records = [r for r in evaluation_records if r["task_id"] == spec["task_id"]]
            specialist_metrics.append(evaluate_summary(backend.evaluate_model(adapted, tokenizer, records)))
        with profiler.stage("factor_extraction", method_id="specialist"):
            cohorts.append(backend.extract_lora_updates(adapted))
        del adapted, base
        gc.collect()
    generation, validation, evaluated, candidate_metadata = {}, {}, {}, {}
    methods = configuration["candidate_methods"]
    rank = configuration["output_rank"]
    for method in methods:
        budget_check()
        with profiler.stage("candidate_generation", method_id=method):
            factors, metadata = backend.generate_candidate(cohorts, method, rank, weights=configuration.get("merge_weights"),
                **configuration.get("method_options", {}).get(method, {}))
            candidate_metadata[method] = metadata
            generation[method] = factors
        with profiler.stage("model_loading", method_id=method, role="validation"):
            base, tokenizer = backend.load_local_base(base_path, base_manifest, device=device, dtype=dtype,
                                                      software_only=phase == "software_only")
            model = backend.attach_candidate(base, factors)
        with profiler.stage("validation_selection", method_id=method):
            validation[method] = evaluate_summary(backend.evaluate_model(model, tokenizer, select_records))
        candidate_metadata[method]["full_model_parameter_count"] = sum(p.numel() for p in model.parameters())
        del model, base, factors
        gc.collect()
    with profiler.stage("alignment_fitting"):
        # Factor-space pairwise quality is descriptive. It is not a calibrated
        # confidence radius for a latent neural transition/coefficient model.
        pairwise = []
        for layer in sorted(cohorts[0]):
            values = [group[layer] for group in cohorts]
            for i in range(len(values)):
                for j in range(i):
                    a, b = values[i].B.double(), values[j].B.double()
                    q, _, vh = torch.linalg.svd(a.mH @ b, full_matrices=False)
                    residual = torch.linalg.vector_norm(a @ (q @ vh) - b) / torch.linalg.vector_norm(b).clamp_min(1e-12)
                    pairwise.append(float(residual))
        pairwise_loss = sum(pairwise) / len(pairwise) if pairwise else 0.0
    with profiler.stage("structural_applicability"):
        evidence = Evidence(coefficient_model="inapplicable", pairwise_loss=pairwise_loss)
        structural_record = {"status": "inapplicable", "coefficient_model": "not_justified",
            "uncertainty_model": "not_justified", "full_common_context_complex": "requires audited D_K probe construction",
            "exact_trivial": None, "repairable_to_tolerance": None, "separated_from_coboundaries": None,
            "natural_central_class_asserted": False, "pairwise_quality": pairwise_loss,
            "reason": "No independently justified structural/coefficient/confidence model; structural branch abstains."}
    structural_methods = set(configuration.get("structural_candidate_methods", ["pairwise_alignment", "global_synchronization", "naive_factor_average"]))
    candidates = [Candidate(method, validation[method]["macro_loss"], method in structural_methods) for method in methods]
    decisions = {}
    for policy in configuration.get("policies", ["P0", "P1", "P2", "P3", "P4"]):
        with profiler.stage("policy_rule", policy_id=policy):
            decisions[policy] = decide("P3G" if policy == "P3" else policy, candidates, configuration.get("fallback", "compact_qr_svd"),
                evidence, pairwise_threshold=configuration.get("pairwise_threshold", .05))
    frozen_actions = {policy: value["selected_action"] for policy, value in decisions.items()}
    frozen_actions_sha256 = digest(frozen_actions)
    # All actions are fixed before any candidate final/development evaluation.
    for method in methods:
        budget_check()
        with profiler.stage("model_loading", method_id=method, role="evaluation"):
            base, tokenizer = backend.load_local_base(base_path, base_manifest, device=device, dtype=dtype,
                                                       software_only=phase == "software_only")
            model = backend.attach_candidate(base, generation.pop(method))
        with profiler.stage("evaluation_inference", method_id=method):
            evaluated[method] = evaluate_summary(backend.evaluate_model(model, tokenizer, evaluation_records))
        del model, base
        gc.collect()
    policies = {}
    for policy in configuration.get("policies", ["P0", "P1", "P2", "P3", "P4"]):
        record = decisions[policy]
        selected = record["selected_action"]
        # Matched candidate experiment: every selecting policy pays the entire
        # generation and validation scan actually used by the shared panel.
        selected_inference = sum(r["seconds"] for r in profiler.rows if r["stage"] == "evaluation_inference" and r.get("method_id") == selected)
        if policy == "P0":
            merge_cost = sum(r["seconds"] for r in profiler.rows if r["stage"] == "candidate_generation" and r.get("method_id") == selected)
            selection_cost = 0.0
            diagnosis_cost = 0.0
        else:
            merge_cost = profiler.seconds({"candidate_generation"})
            selection_cost = profiler.seconds({"validation_selection"})
            diagnosis_cost = profiler.seconds({"alignment_fitting", "structural_applicability"}) if policy in ("P3", "P4") else profiler.seconds({"alignment_fitting"}) if policy == "P2" else 0.0
        extraction_cost = profiler.seconds({"factor_extraction"})
        rule_cost = sum(r["seconds"] for r in profiler.rows if r["stage"] == "policy_rule" and r.get("policy_id") == policy)
        loading_cost = sum(r["seconds"] for r in profiler.rows if r["stage"] == "model_loading" and (
            r.get("method_id") == "specialist" or
            (r.get("method_id") == selected and r.get("role") == "evaluation") or
            (policy != "P0" and r.get("role") == "validation")))
        policies[policy] = {**record, "policy_id": policy, "evaluation": evaluated[selected],
            "merge_seconds": merge_cost, "selection_seconds": selection_cost, "diagnosis_seconds": diagnosis_cost,
            "factor_extraction_seconds": extraction_cost, "policy_rule_seconds": rule_cost,
            "inference_seconds": selected_inference, "merging_selection_seconds": extraction_cost + merge_cost + selection_cost + diagnosis_cost + rule_cost,
            "model_loading_seconds_separate": loading_cost,
            "end_to_end_with_loading_seconds": loading_cost + extraction_cost + merge_cost + selection_cost + diagnosis_cost + rule_cost,
            "cost_scope": "attributed from shared matched-candidate panel; loading+merge+select, inference separate; not measured standalone-policy latency",
            "candidate_generation_savings_claim": False, "cost_experiment": "matched_candidate",
            "parameter_count": candidate_metadata[selected]["full_model_parameter_count"]}
    return {"phase": phase, "status": "completed", "base_revision": base_manifest["revision"],
            "base_metrics": base_metrics, "candidate_metrics": evaluated, "candidate_validation": validation, "candidate_metadata": candidate_metadata,
            "specialist_metrics": specialist_metrics, "policies": policies, "structural_record": structural_record,
            "stage_costs": profiler.rows, "final_test_evaluated": phase == "confirmatory",
            "software_only": phase == "software_only", "actions_frozen_before_candidate_evaluation": True,
            "frozen_actions_sha256": frozen_actions_sha256, "certificate_computation_status": "not_run; coefficient model inapplicable"}


def execute_config(configuration, out, *, root, resume=False):
    """Run resolved local episodes with source/input matching and no duplicates."""
    from .task_data import validate_split_manifest, load_split
    from .language_backend import validate_base_manifest, validate_adapter_manifest
    from .resource_guard import preflight, BudgetGuard
    root = Path(root)
    if configuration["phase"] == "confirmatory":
        require_confirmatory_access(configuration, out, root)
    configuration = copy.deepcopy(configuration)
    for field in ("model_manifest_path", "split_manifest_path"):
        configuration[field] = str((root / configuration[field]).resolve())
    configuration["model"]["local_path"] = str((root / configuration["model"]["local_path"]).resolve())
    for episode in configuration["episodes"]:
        if "model" in episode:
            raise ValueError("episode cannot override the frozen common base")
        for spec in episode["adapters"]:
            spec["path"] = str((root / spec["path"]).resolve())
    resource_errors = preflight(configuration, root)
    if resource_errors:
        raise ValueError("resource admission failed: " + "; ".join(resource_errors))
    model_manifest = json.loads(Path(configuration["model_manifest_path"]).read_text())
    split_path = Path(configuration["split_manifest_path"])
    split_manifest = json.loads(split_path.read_text())
    errors = validate_split_manifest(split_path)
    if errors:
        raise ValueError("split/leakage audit failed: " + "; ".join(errors))
    phase = configuration["phase"]
    if phase not in {"pilot", "confirmatory", "software_only"}:
        raise ValueError("unsupported phase")
    software_only = phase == "software_only"
    if software_only and configuration.get("device", "cpu") != "cpu":
        raise ValueError("software-only fixture is CPU-only")
    if not software_only and split_manifest.get("not_benchmark_evidence"):
        raise ValueError("software-only splits cannot enter a research cohort")
    validate_base_manifest(configuration["model"]["local_path"], model_manifest, software_only=software_only)
    for episode in configuration["episodes"]:
        if len({spec["group_id"] for spec in episode["adapters"]}) != 1:
            raise ValueError("all specialists in an episode must share an independent group identity")
        if set(episode["tasks"]) != {spec["task_id"] for spec in episode["adapters"]}:
            raise ValueError("episode requires exactly one specialist per declared task")
        if len(episode["tasks"]) != len(episode["adapters"]):
            raise ValueError("repeated specialist tasks are not independent task members")
        for spec in episode["adapters"]:
            validate_adapter_manifest(spec["path"], spec, model_manifest, software_only=software_only)
    evaluation_role = "D_test" if phase == "confirmatory" else "pilot"
    def records(role):
        return load_split(split_path, role, allow_final_test=phase == "confirmatory")
    inputs = {"source": source_fingerprint(root), "model_manifest": file_hash(configuration["model_manifest_path"]),
              "split_manifest": file_hash(split_path), "split_files": {role: value["sha256"] for role, value in split_manifest["splits"].items()},
              "adapters": {spec["path"]: spec["files"] for ep in configuration["episodes"] for spec in ep["adapters"]}}
    store = EpisodeStore(out, configuration, inputs=inputs)
    budget = BudgetGuard(configuration, out)
    results = []
    ids = [ep["episode_id"] for ep in configuration["episodes"]]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate episode IDs in plan")
    for episode in configuration["episodes"]:
        budget.check()
        task_set = set(episode["tasks"])
        selected = [r for r in records("D_select") if r["task_id"] in task_set]
        evaluated = [r for r in records(evaluation_role) if r["task_id"] in task_set]
        value, reused = store.execute(episode["episode_id"], lambda: execute_episode(
            {**configuration, **episode}, model_manifest, episode["adapters"], selected, evaluated, phase=phase,
            budget_check=budget.check), resume=resume)
        results.append({"episode_id": episode["episode_id"], "reused": reused, "result": value})
    atomic_json(Path(out) / f"budget_receipt-{uuid.uuid4().hex}.json", budget.check())
    return results


def require_confirmatory_access(configuration, out, root):
    """Reject phase-only or software-GO authorization for final labels."""
    out = Path(out)
    if configuration.get("test_access") is not True:
        raise PermissionError("confirmatory final-test access is not explicitly frozen")
    path = out / "freeze_manifest.json"
    if not path.is_file():
        raise PermissionError("final-test access requires GO and a frozen protocol")
    receipt = json.loads(path.read_text())
    if receipt.get("pilot_decision") != "GO" or receipt.get("software_only") is not False:
        raise PermissionError("a real pilot GO is required before final-test access")
    if receipt.get("config_sha256") != digest(configuration) or receipt.get("source_hashes") != source_fingerprint(root):
        raise PermissionError("frozen configuration/source hash mismatch")
    decision_path = Path(root) / configuration.get("pilot_decision_path", "__missing__")
    if not decision_path.is_file() or file_hash(decision_path) != receipt.get("pilot_decision_sha256"):
        raise PermissionError("real pilot GO evidence hash mismatch")
    decision = json.loads(decision_path.read_text())
    if decision.get("decision") != "GO" or decision.get("software_only") is not False:
        raise PermissionError("pilot decision is not a real GO")
