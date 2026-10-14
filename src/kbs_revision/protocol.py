"""Immutable configuration and split guards for the KBS experiment program."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import yaml

ROLES = ("train", "pilot", "D_K", "D_align", "D_cert", "D_select", "D_test")
METHODS = ("base", "specialists", "task_arithmetic", "compact_qr_svd", "dense_reference_svd",
           "exact_concatenation", "naive_factor_average", "pairwise_alignment", "global_synchronization",
           "ties", "dare_ties", "knots", "core_space")
IMPLEMENTED_NEURAL_BACKENDS = frozenset({"modern_language_v1"})


def canonical_bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def digest(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def load_config(path):
    value = yaml.safe_load(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError("configuration must be a mapping")
    return value


def split_errors(splits, root):
    errors, seen_ids, seen_text = [], {}, {}
    for role in ROLES:
        spec = splits.get(role, {})
        filename = spec.get("path")
        if not filename:
            errors.append(f"missing split manifest: {role}")
            continue
        path = Path(root) / filename
        if not path.is_file() or file_hash(path) != spec.get("sha256"):
            errors.append(f"missing or hash-mismatched split: {role}")
            continue
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if not records:
            errors.append(f"empty split: {role}")
        for record in records:
            # Content is hashed as well as IDs: changing IDs does not hide exact duplicates.
            key = (record.get("task_id"), record.get("example_id"))
            if None in key:
                errors.append(f"unidentified example in {role}")
            if key in seen_ids:
                errors.append(f"duplicate example across/within splits: {role}, {seen_ids[key]}")
            seen_ids[key] = role
            content = record.get("normalized_text_sha256")
            if not isinstance(content, str) or not re.fullmatch(r"[a-f0-9]{64}", content):
                errors.append(f"missing normalized content hash in {role}")
            elif content in seen_text:
                errors.append(f"duplicate normalized text: {role}, {seen_text[content]}")
            seen_text[content] = role
    if not splits.get("near_duplicate_audit", {}).get("passed"):
        errors.append("near-duplicate leakage audit not passed")
    return errors


def readiness_errors(config, root):
    errors = []
    model = config.get("model", {})
    if not re.fullmatch(r"[a-f0-9]{40}", str(model.get("revision", ""))):
        errors.append("unresolved base model revision")
    if not model.get("local_path") or not (Path(root) / model["local_path"]).is_dir():
        errors.append("base model unavailable locally")
    if not model.get("license_reviewed"):
        errors.append("model license not recorded/reviewed")
    endpoints = config.get("endpoints", {})
    for field in ("efficacy", "efficiency", "task_losses", "task_weights", "primary_comparisons",
                  "multiplicity", "statistical_unit"):
        if not endpoints.get(field):
            errors.append(f"missing primary analysis definition: {field}")
    if endpoints.get("claim_noninferiority") and not endpoints.get("noninferiority_justification"):
        errors.append("noninferiority claim lacks a task-specific justification")
    tasks = config.get("tasks", [])
    weights = endpoints.get("task_weights", {})
    if len(tasks) < 4 or set(weights) != set(tasks) or any(w != 1 / len(tasks) for w in weights.values()):
        errors.append("four-task equal-weight population is not fixed")
    limits = config.get("resource_limits", {})
    for field in ("cpu_seconds", "pilot_accelerator_hours", "total_accelerator_hours", "disk_bytes"):
        if not isinstance(limits.get(field), (int, float)) or limits[field] <= 0:
            errors.append(f"unrecorded/nonpositive resource limit: {field}")
    baselines = config.get("baselines", {})
    for method in METHODS:
        spec = baselines.get(method, {})
        status = spec.get("status")
        if status == "not_applicable" and spec.get("reason"):
            continue
        if status != "parity_passed" or not spec.get("definition") or not spec.get("evidence_sha256"):
            errors.append(f"unresolved baseline: {method}")
    if config.get("backend") not in IMPLEMENTED_NEURAL_BACKENDS:
        errors.append("no implemented modern language execution backend")
    if config.get("backend") == "modern_language_v1":
        model_manifest_path = config.get("model_manifest_path")
        if not model_manifest_path or not (Path(root) / model_manifest_path).is_file():
            errors.append("missing validated model/tokenizer manifest")
        else:
            try:
                from .language_backend import validate_base_manifest
                value = json.loads((Path(root) / model_manifest_path).read_text())
                validate_base_manifest(Path(root) / model["local_path"], value)
            except (ValueError, KeyError, TypeError, OSError) as error:
                errors.append(f"invalid model manifest: {error}")
        split_manifest_path = config.get("split_manifest_path")
        if not split_manifest_path or not (Path(root) / split_manifest_path).is_file():
            errors.append("missing actual seven-role split manifest")
        else:
            from .task_data import validate_split_manifest
            errors += validate_split_manifest(Path(root) / split_manifest_path)
            manifest_data = json.loads((Path(root) / split_manifest_path).read_text())
            if manifest_data.get("not_benchmark_evidence"):
                errors.append("software-only splits cannot enter the real language protocol")
    adapters = config.get("adapters", []) or [spec for episode in config.get("episodes", []) for spec in episode.get("adapters", [])]
    if not adapters:
        errors.append("no compatible multilayer trained adapters")
    groups = set()
    for adapter in adapters:
        groups.add(adapter.get("group_id"))
        if config.get("backend") == "modern_language_v1":
            try:
                from .language_backend import validate_adapter_manifest
                base_manifest = json.loads((Path(root) / config["model_manifest_path"]).read_text())
                validate_adapter_manifest(Path(root) / adapter["path"], adapter, base_manifest)
            except (ValueError, KeyError, TypeError, OSError) as error:
                errors.append(f"invalid compatible adapter: {error}")
            continue
        for field in ("group_id", "training_seed", "task_id", "base_model_revision", "rank", "scale",
                      "target_modules", "orientation", "tokenizer_revision", "head_convention"):
            if adapter.get(field) is None:
                errors.append(f"adapter lacks {field}")
        if adapter.get("base_model_revision") != model.get("revision") or adapter.get("parameterization") != "lora":
            errors.append("incompatible base or unsupported adapter parameterization")
        path = Path(root) / adapter.get("path", "__missing__")
        if not path.is_file() or file_hash(path) != adapter.get("sha256"):
            errors.append("missing/hash-mismatched trained adapter")
    if config.get("phase") == "confirmatory" and (None in groups or len(groups) < 5):
        errors.append("confirmatory inference requires at least five independent training groups")
    if config.get("backend") != "modern_language_v1":
        errors += split_errors(config.get("splits", {}), root)
    else:
        from .resource_guard import preflight
        errors += preflight(config, root)
    if config.get("phase") == "confirmatory":
        decision = config.get("pilot_decision_path")
        if not decision or not (Path(root) / decision).is_file():
            errors.append("confirmatory protocol requires a real pilot GO receipt")
        else:
            result = json.loads((Path(root) / decision).read_text())
            if result.get("decision") != "GO" or result.get("software_only") is not False:
                errors.append("software-only or non-GO pilot cannot authorize confirmatory protocol")
    return sorted(set(errors))


def freeze(config, root, out):
    errors = readiness_errors(config, root)
    if errors:
        raise ValueError("protocol not ready: " + "; ".join(errors))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    encoded = canonical_bytes(config)
    # Exclusive creation makes preregistration immutable.
    with (out / "preregistration.yaml").open("xb") as handle:
        handle.write(encoded)  # JSON is also valid YAML.
    with (out / "preregistration.sha256").open("x") as handle:
        handle.write(hashlib.sha256(encoded).hexdigest() + "\n")
    if config.get("phase") == "confirmatory":
        from .neural_runner import atomic_json, source_fingerprint
        atomic_json(out / "freeze_manifest.json", {"config_sha256": digest(config), "source_hashes": source_fingerprint(root),
            "pilot_decision": "GO", "software_only": False,
            "pilot_decision_sha256": file_hash(Path(root) / config["pilot_decision_path"])})


def check_frozen(config_path, out):
    out = Path(out)
    frozen_path = out / "preregistration.yaml"
    if not frozen_path.is_file():
        raise ValueError("missing frozen preregistration")
    expected = (out / "preregistration.sha256").read_text().strip()
    if file_hash(frozen_path) != expected or canonical_bytes(load_config(config_path)) != frozen_path.read_bytes():
        raise ValueError("frozen configuration hash mismatch")
    return load_config(config_path), expected
