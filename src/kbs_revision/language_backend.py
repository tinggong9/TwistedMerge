"""Local, safetensors-only Qwen/PEFT training, merging, and scored inference.

The smoke fixture is a randomly initialized two-block Qwen2, explicitly
software-only. It does not establish Qwen2.5-1.5B research readiness. Real runs
must supply a reviewed manifest and the pinned base below; no download or
remote Python is permitted by any loader in this module. Ordinary causal LoRA
with a shared frozen language head is the supported scope.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import resource
import sys
import time
from typing import Mapping, Sequence

import torch
from safetensors import safe_open

from .merge_baselines import (
    LoRAUpdate, aligned_factor_average, compact_qr_svd, concatenate_exact,
    dare_ties_effective, dense_reference_svd, effective_delta,
    naive_factor_average, ties_effective, weighted_sum,
)
from .protocol import digest, file_hash

BASE_ID = "Qwen/Qwen2.5-1.5B"
BASE_REVISION = "8faed761d45a263340a0528343f099c05c9a4323"
TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                   "vocab.json", "merges.txt", "added_tokens.json")
METHODS = ("compact_qr_svd", "dense_reference_svd", "exact_concatenation",
           "task_arithmetic", "naive_factor_average", "pairwise_alignment",
           "global_synchronization", "ties", "dare_ties", "knots", "core_space")


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _reviewed_license(manifest):
    if manifest.get("license_reviewed") is not True or not isinstance(manifest.get("license"), str) or not manifest["license"].strip():
        raise ValueError("license and explicit license_reviewed are required")


def _checked_files(path, expected, *, label):
    """Verify every recorded artifact before any model/tensor deserialization."""
    path = Path(path).resolve()
    if not isinstance(expected, dict) or not expected:
        raise ValueError(f"nonempty {label} SHA256 manifest required")
    for name, sha in expected.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or not re.fullmatch(r"[a-f0-9]{64}", str(sha)):
            raise ValueError(f"invalid {label} artifact identity")
        artifact = path / relative
        if not artifact.is_file() or not artifact.resolve().is_relative_to(path) or file_hash(artifact) != sha:
            raise ValueError(f"missing or hash-mismatched {label} artifact: {name}")


def _safe_directory(path):
    path = Path(path)
    if not path.is_dir():
        raise ValueError("local artifact directory is unavailable")
    forbidden = [p.name for p in path.rglob("*") if p.is_file() and p.suffix.lower() in {".bin", ".pt", ".pth", ".pkl", ".pickle"}]
    if forbidden:
        raise ValueError("pickle artifacts are unsupported: " + ", ".join(sorted(forbidden)))
    return path


def _synchronize(device):
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def tokenizer_fingerprint(path):
    path = Path(path)
    return {name: file_hash(path / name) for name in TOKENIZER_FILES if (path / name).is_file()}


def validate_base_manifest(path, manifest, *, software_only=False):
    path = _safe_directory(path)
    _reviewed_license(manifest)
    if manifest.get("software_only", False) != software_only:
        raise ValueError("software-only fixture must be explicitly opted into")
    if not software_only and (manifest.get("model_id"), manifest.get("revision")) != (BASE_ID, BASE_REVISION):
        raise ValueError("base model ID/revision differs from the planned pinned Qwen base")
    if not re.fullmatch(r"[a-f0-9]{40}", str(manifest.get("revision", ""))):
        raise ValueError("base revision must be immutable 40-character identity")
    _checked_files(path, manifest.get("files"), label="base")
    actual = {p.name for p in path.glob("*.safetensors")}
    recorded = {name for name in manifest["files"] if name.endswith(".safetensors")}
    if not actual or recorded != actual:
        raise ValueError("all base safetensors weights must be recorded exactly")
    if "config.json" not in manifest["files"]:
        raise ValueError("base config.json must be hash pinned")
    config = _json(path / "config.json")
    if config.get("auto_map") or config.get("quantization_config"):
        raise ValueError("remote code and quantized bases are unsupported")
    if config.get("model_type") != "qwen2" or config.get("architectures") != ["Qwen2ForCausalLM"] or manifest.get("architecture") != "Qwen2ForCausalLM":
        raise ValueError("only the reviewed Qwen2ForCausalLM architecture is supported")
    if config.get("num_hidden_layers", 0) < 2:
        raise ValueError("multilayer backend requires at least two transformer blocks")
    if not software_only and (config.get("hidden_size"), config.get("num_hidden_layers"), config.get("num_attention_heads"), config.get("num_key_value_heads")) != (1536, 28, 12, 2):
        raise ValueError("architecture dimensions differ from the pinned 1.5B Qwen base")
    if manifest.get("head_convention") != "shared_frozen_causal_lm_head":
        raise ValueError("shared frozen causal language head must be declared")
    tokenizer = manifest.get("tokenizer_files")
    _checked_files(path, tokenizer, label="tokenizer")
    if tokenizer != tokenizer_fingerprint(path) or "tokenizer_config.json" not in tokenizer:
        raise ValueError("complete tokenizer files must be hash pinned")
    if _json(path / "tokenizer_config.json").get("auto_map"):
        raise ValueError("remote tokenizer code is unsupported")
    index = path / "model.safetensors.index.json"
    if index.exists():
        if "model.safetensors.index.json" not in manifest["files"]:
            raise ValueError("sharded safetensors index must be pinned")
        shards = set(_json(index).get("weight_map", {}).values())
        if shards != recorded:
            raise ValueError("sharded safetensors index does not match pinned weights")
    return {"status": "validated", "model_id": manifest["model_id"], "revision": manifest["revision"],
            "architecture": manifest["architecture"], "num_hidden_layers": config["num_hidden_layers"],
            "tokenizer_sha256": digest(tokenizer), "head_convention": manifest["head_convention"],
            "software_only": software_only}


def load_local_base(path, manifest, *, device="cpu", dtype=torch.float32, software_only=False):
    validate_base_manifest(path, manifest, software_only=software_only)
    if dtype not in {torch.float32, torch.float64, torch.bfloat16, torch.float16}:
        raise ValueError("unsupported model dtype")
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("requested CUDA device is unavailable")
    if str(device).startswith("mps") and not torch.backends.mps.is_available():
        raise ValueError("requested MPS device is unavailable")
    from transformers import AutoTokenizer, PreTrainedTokenizerFast, Qwen2ForCausalLM
    model, loading = Qwen2ForCausalLM.from_pretrained(str(path), local_files_only=True,
                                                     use_safetensors=True, dtype=dtype,
                                                     attn_implementation="eager", output_loading_info=True)
    if any(loading.get(name) for name in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError("base safetensors do not contain a complete compatible model state")
    # AutoTokenizer follows Qwen's architecture class, which would reinterpret
    # the software fixture's WordLevel JSON as Qwen BPE in Transformers 5.
    loader = PreTrainedTokenizerFast if software_only else AutoTokenizer
    tokenizer = loader.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        raise ValueError("pinned tokenizer must declare a pad token")
    if len(tokenizer) > model.config.vocab_size:
        raise ValueError("tokenizer vocabulary exceeds base embedding/head capacity")
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model._kbs_base_manifest_digest = digest(manifest)
    _synchronize(device)
    return model, tokenizer


def validate_adapter_manifest(path, manifest, base_manifest, *, software_only=False):
    path = _safe_directory(path)
    _reviewed_license(manifest)
    if manifest.get("software_only", False) != software_only or base_manifest.get("software_only", False) != software_only:
        raise ValueError("software-only adapter cannot enter a research cohort")
    for name, expected in (("base_model_id", base_manifest.get("model_id")), ("base_revision", base_manifest.get("revision")),
                           ("architecture", base_manifest.get("architecture")), ("head_convention", base_manifest.get("head_convention")),
                           ("tokenizer_sha256", digest(base_manifest.get("tokenizer_files")))):
        if manifest.get(name) != expected:
            raise ValueError(f"adapter {name} is incompatible with base")
    if not isinstance(manifest.get("group_id"), str) or not manifest["group_id"] or not isinstance(manifest.get("training_seed"), int):
        raise ValueError("independent training group and seed must be recorded")
    if manifest.get("task_id") not in {"BoolQ", "PIQA", "ARC-Challenge", "GSM8K"}:
        raise ValueError("adapter training task must be recorded")
    _checked_files(path, manifest.get("files"), label="adapter")
    if set(manifest["files"]) != {"adapter_config.json", "adapter_model.safetensors"}:
        raise ValueError("ordinary adapter_config and safetensors weights required")
    if {p.name for p in path.glob("*.safetensors")} != {"adapter_model.safetensors"}:
        raise ValueError("unrecorded adapter safetensors artifacts are unsupported")
    config = _json(path / "adapter_config.json")
    if config.get("peft_type") != "LORA" or config.get("task_type") != "CAUSAL_LM":
        raise ValueError("ordinary causal LoRA is required")
    if config.get("base_model_name_or_path") != base_manifest["model_id"] or config.get("revision") != base_manifest["revision"]:
        raise ValueError("adapter config does not pin the common base identity")
    unsupported = ("use_dora", "use_qalora", "use_bdlora", "alora_invocation_tokens", "layer_replication",
                   "modules_to_save", "trainable_token_indices", "target_parameters", "rank_pattern", "alpha_pattern", "lora_bias",
                   "exclude_modules", "layers_to_transform", "layers_pattern", "loftq_config", "megatron_config", "arrow_config")
    if any(config.get(key) for key in unsupported) or config.get("bias", "none") != "none" or config.get("fan_in_fan_out"):
        raise ValueError("DoRA, quantized, patterned, bias, or trainable-head adapters are unsupported")
    if sorted(config.get("target_modules", [])) != sorted(TARGET_MODULES) or sorted(manifest.get("target_modules", [])) != sorted(TARGET_MODULES):
        raise ValueError("adapter must target all q/k/v/o attention projections")
    for field, config_field in (("rank", "r"), ("alpha", "lora_alpha"), ("dropout", "lora_dropout"), ("use_rslora", "use_rslora")):
        if manifest.get(field) != config.get(config_field, False if field == "use_rslora" else None):
            raise ValueError(f"adapter {field} differs from config")
    if isinstance(manifest["rank"], bool) or not isinstance(manifest["rank"], int) or manifest["rank"] <= 0:
        raise ValueError("adapter rank must be positive")
    if not math.isfinite(float(manifest["alpha"])) or manifest["alpha"] <= 0 or not 0 <= manifest["dropout"] < 1:
        raise ValueError("invalid adapter alpha/dropout")
    with safe_open(path / "adapter_model.safetensors", framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        if not keys or any(not re.fullmatch(r"base_model\.model\.model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)\.lora_[AB]\.weight", key) for key in keys):
            raise ValueError("adapter tensors include unsupported modules/head or unexpected names")
        pairs = {}
        for key in keys:
            stem, letter = key.rsplit(".lora_", 1)
            pairs.setdefault(stem, {})[letter[0]] = handle.get_slice(key).get_shape()
        if len(pairs) < 8 or len(keys) != 2 * len(pairs):
            raise ValueError("adapter must contain A/B factors on multiple target layers")
        layers = {}
        for stem, pair in pairs.items():
            if set(pair) != {"A", "B"} or len(pair["A"]) != 2 or len(pair["B"]) != 2 or pair["A"][0] != manifest["rank"] or pair["B"][1] != manifest["rank"]:
                raise ValueError("adapter tensor rank/shapes disagree with config")
            layer = re.search(r"layers\.(\d+)", stem).group(1)
            layers.setdefault(layer, set()).add(stem.rsplit(".", 1)[-1])
        if any(names != set(TARGET_MODULES) for names in layers.values()):
            raise ValueError("each adapted layer must contain all q/k/v/o modules")
    return {"status": "validated", "group_id": manifest["group_id"], "training_seed": manifest["training_seed"],
            "adapted_layers": len(layers), "target_module_count": len(pairs), "software_only": software_only}


def load_local_adapter(base_model, path, manifest, base_manifest, *, software_only=False):
    validate_adapter_manifest(path, manifest, base_manifest, software_only=software_only)
    if hasattr(base_model, "peft_config"):
        raise ValueError("adapter must attach to a clean common base")
    if getattr(base_model, "_kbs_base_manifest_digest", None) != digest(base_manifest):
        raise ValueError("base model was not loaded from this validated immutable manifest")
    expected = {}
    for name, module in base_model.named_modules():
        if name.rsplit(".", 1)[-1] in TARGET_MODULES and isinstance(module, torch.nn.Linear):
            expected["base_model.model." + name + ".lora_A.weight"] = [manifest["rank"], module.in_features]
            expected["base_model.model." + name + ".lora_B.weight"] = [module.out_features, manifest["rank"]]
    with safe_open(Path(path) / "adapter_model.safetensors", framework="pt", device="cpu") as handle:
        if set(handle.keys()) != set(expected) or any(handle.get_slice(name).get_shape() != shape for name, shape in expected.items()):
            raise ValueError("adapter tensor module identities/shapes differ from complete common base targets")
    from peft import PeftModel
    model = PeftModel.from_pretrained(base_model, str(path), is_trainable=False,
                                     local_files_only=True, use_safetensors=True)
    model.eval()
    factors = extract_lora_updates(model)
    if len(factors) != model.config.num_hidden_layers * len(TARGET_MODULES):
        raise ValueError("adapter must cover every planned target module in the common base")
    return model


def extract_lora_updates(model, adapter_name="default"):
    """Return CPU float64 effective-update factors with PEFT scaling preserved."""
    result = {}
    for name, module in model.named_modules():
        if hasattr(module, "lora_A") and adapter_name in module.lora_A:
            if getattr(module, "use_dora", {}).get(adapter_name, False):
                raise ValueError("DoRA effective updates are unsupported")
            result[name] = LoRAUpdate(module.lora_B[adapter_name].weight.detach().to(device="cpu", dtype=torch.float64).clone(),
                                      module.lora_A[adapter_name].weight.detach().to(device="cpu", dtype=torch.float64).clone(),
                                      float(module.scaling[adapter_name]))
    if not result:
        raise ValueError("no ordinary LoRA target modules were found")
    return result


def _matrix_factor(delta, rank):
    # Exact when rank is full; explicit rank compression otherwise.
    u, singular, vh = torch.linalg.svd(delta, full_matrices=False)
    k = min(rank, len(singular))
    root = singular[:k].sqrt()
    return LoRAUpdate(u[:, :k] * root, root[:, None] * vh[:k])


def generate_candidate(cohorts: Sequence[Mapping[str, LoRAUpdate]], method, rank=None, weights=None, *, density=0.5, seed=0,
                       weight_semantics="official_equal"):
    if method not in METHODS or not cohorts:
        raise ValueError("supported method and a nonempty adapter cohort are required")
    if any(set(cohort) != set(cohorts[0]) for cohort in cohorts):
        raise ValueError("cohort target modules differ")
    if rank is not None and (isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0):
        raise ValueError("output rank must be positive")
    if method in {"compact_qr_svd", "dense_reference_svd"} and rank is None:
        raise ValueError("SVD candidate requires fixed output rank")
    started = time.perf_counter()
    result = {}
    if method == "knots":
        from .external_methods import knots_merge_layers
        result = knots_merge_layers({name: [cohort[name] for cohort in cohorts] for name in sorted(cohorts[0])},
                                    rank=rank, weights=weights, density=density, weight_semantics=weight_semantics)
    for name in (() if method == "knots" else sorted(cohorts[0])):
        updates = [cohort[name] for cohort in cohorts]
        if method == "compact_qr_svd":
            merged = compact_qr_svd(updates, rank, weights)
        elif method == "dense_reference_svd":
            merged = dense_reference_svd(updates, rank, weights)
        elif method == "exact_concatenation":
            merged = concatenate_exact(updates, weights)
        elif method == "naive_factor_average":
            merged = naive_factor_average(updates, weights)
        elif method in {"pairwise_alignment", "global_synchronization"}:
            merged = aligned_factor_average(updates, weights, method="pairwise" if method == "pairwise_alignment" else "global")
        elif method == "core_space":
            from .external_methods import core_space_merge
            merged = core_space_merge(updates, rank=rank, weights=weights, density=density, weight_semantics=weight_semantics)
        else:
            if method == "task_arithmetic":
                delta = weighted_sum(updates, weights)
            elif method == "ties":
                delta = ties_effective(updates, weights, density=density)
            else:
                delta = dare_ties_effective(updates, weights, density=density, seed=seed)
            merged = _matrix_factor(delta, min(delta.shape) if rank is None else rank)
        if method in {"naive_factor_average", "pairwise_alignment", "global_synchronization"} and rank is not None and rank != merged.rank:
            raise ValueError("factor-averaging capacity equals input rank; requested output rank must match")
        result[name] = merged
    elapsed = time.perf_counter() - started
    alignment = method in {"pairwise_alignment", "global_synchronization"}
    default_one = method in {"ties", "dare_ties", "knots", "core_space"}
    effective_weights = list(weights) if weights is not None else [1.0 if default_one else 1.0 / len(cohorts)] * len(cohorts)
    return result, {"method": method, "weights": effective_weights,
                    "weight_semantics": "verbatim explicit weights; equal average default except PEFT TIES/DARE and external KnOTS/Core Space default ones",
                    "input_module_count": len(result), "requested_output_rank": rank,
                    "output_ranks": {name: update.rank for name, update in result.items()},
                    "output_parameters": sum(update.A.numel() + update.B.numel() for update in result.values()),
                    "deployment_model_count": 1, "inference_latency_multiplier": None,
                    "alignment_fitting_seconds": elapsed if alignment else 0.0,
                    "candidate_generation_seconds": 0.0 if alignment else elapsed,
                    "timer_scope": "all layers, including factor validation; alignment stage includes whitening, synchronization, factor averaging",
                    "effective_density": density if method in {"ties", "dare_ties", "knots", "core_space"} else None,
                    "external_weight_semantics": weight_semantics if method in {"knots", "core_space"} else None,
                    "seed": seed if method == "dare_ties" else None,
                    "compression": "post-merge rank compression adaptation" if method in {"knots", "core_space"} and rank is not None else
                                   "source supported-basis capacity" if method in {"knots", "core_space"} else
                                   "rank-sum exact capacity" if method == "exact_concatenation" else
                                   "effective delta SVD after tensor kernel" if method in {"ties", "dare_ties", "task_arithmetic"} else method}


def attach_candidate(base_model, factors):
    """Attach unit-scaled output factors; the base's language head is untouched."""
    if hasattr(base_model, "peft_config") or not factors:
        raise ValueError("candidate requires a clean base and nonempty factors")
    expected = {"base_model.model." + name for name, module in base_model.named_modules()
                if name.rsplit(".", 1)[-1] in TARGET_MODULES and isinstance(module, torch.nn.Linear)}
    if set(factors) != expected:
        raise ValueError("candidate factor modules must exactly cover the base q/k/v/o modules")
    if any(update.rank <= 0 for update in factors.values()):
        raise ValueError("zero-rank adapters are unsupported by PEFT attachment")
    from peft import LoraConfig, get_peft_model
    rank_pattern = {name.removeprefix("base_model.model."): update.rank for name, update in factors.items()}
    alpha_pattern = dict(rank_pattern)  # alpha/r = 1, scaling already absorbed.
    default_rank = next(iter(factors.values())).rank
    config = LoraConfig(r=default_rank, lora_alpha=default_rank, lora_dropout=0.0,
                        rank_pattern=rank_pattern, alpha_pattern=alpha_pattern,
                        target_modules=list(TARGET_MODULES), task_type="CAUSAL_LM", bias="none")
    model = get_peft_model(base_model, config)
    modules = dict(model.named_modules())
    with torch.no_grad():
        for name, update in factors.items():
            module = modules[name]
            if module.lora_A["default"].weight.shape != update.A.shape or module.lora_B["default"].weight.shape != update.B.shape:
                raise ValueError("candidate ambient shape differs from base projection")
            if module.scaling["default"] != 1.0:
                raise RuntimeError("candidate scaling was not unit normalized")
            module.lora_A["default"].weight.copy_(update.scale * update.A)
            module.lora_B["default"].weight.copy_(update.B)
    return model.eval()


def _tokens(tokenizer, text):
    result = tokenizer.encode(text, add_special_tokens=False)
    if not result:
        raise ValueError("empty tokenized prompt/completion")
    return result


def _context_guard(model, ids):
    if len(ids) > model.config.max_position_embeddings:
        raise ValueError("record exceeds pinned model context; truncation is not permitted")


def choice_loglikelihood(model, tokenizer, prompt, choice):
    prefix = _tokens(tokenizer, prompt)
    completion = _tokens(tokenizer, " " + choice)
    ids = prefix + completion
    _context_guard(model, ids)
    device = next(model.parameters()).device
    tensor = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(input_ids=tensor, attention_mask=torch.ones_like(tensor), use_cache=False).logits[0]
        scores = logits[len(prefix) - 1:-1].float().log_softmax(-1)
        targets = tensor[0, len(prefix):]
        return float(scores.gather(1, targets[:, None]).sum())


def evaluate_model(model, tokenizer, records, *, max_new_tokens=96):
    from .task_data import prepare_record, choose_prediction, score_prediction
    if not records:
        raise ValueError("evaluation records must be nonempty")
    device = next(model.parameters()).device
    _synchronize(device)
    started = time.perf_counter()
    model.eval()
    rows = []
    for record in records:
        prepared = prepare_record(record)
        row_start = time.perf_counter()
        if prepared["choices"] is not None:
            scores = [choice_loglikelihood(model, tokenizer, prepared["prompt"], choice) for choice in prepared["choices"]]
            prediction = choose_prediction(prepared["task"], scores, record)
        else:
            ids = _tokens(tokenizer, prepared["prompt"])
            if len(ids) + max_new_tokens > model.config.max_position_embeddings:
                raise ValueError("generation budget exceeds pinned context")
            inputs = torch.tensor([ids], dtype=torch.long, device=device)
            with torch.no_grad():
                generated = model.generate(inputs, attention_mask=torch.ones_like(inputs), do_sample=False,
                                           max_new_tokens=max_new_tokens, pad_token_id=tokenizer.pad_token_id,
                                           eos_token_id=tokenizer.eos_token_id, use_cache=True)
            prediction = tokenizer.decode(generated[0, len(ids):], skip_special_tokens=True)
            scores = None
        score = score_prediction(prepared["task"], prediction, record)
        _synchronize(device)
        rows.append({"task_id": prepared["task"], "example_id": prepared["example_id"], "prediction": prediction,
                     "choice_loglikelihoods": scores, "score": score, "correct": bool(score),
                     "bounded_task_error": 1 - score, "bounded_loss": 1 - score,
                     "inference_seconds": time.perf_counter() - row_start})
    return {"rows": rows, "accuracy": sum(row["score"] for row in rows) / len(rows),
            "bounded_loss": sum(row["bounded_loss"] for row in rows) / len(rows),
            "inference_seconds": time.perf_counter() - started,
            "scoring": "sum full independently tokenized leading-space completion causal loglikelihood; no length normalization; deterministic greedy GSM8K",
            "deployment_model_count": 1, "inference_latency_multiplier": None}


def _parameter_hash(model, *, base_only=False):
    sha = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if base_only and "lora_" in name:
            continue
        # PEFT prepends module names; compare values and original parameter order.
        sha.update(parameter.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return sha.hexdigest()


def _artifact_manifest(path, base_manifest, *, group_id, seed, task_id, rank, alpha, dropout, use_rslora=False):
    return {"base_model_id": base_manifest["model_id"], "base_revision": base_manifest["revision"],
            "architecture": base_manifest["architecture"], "target_modules": list(TARGET_MODULES),
            "rank": rank, "alpha": alpha, "dropout": dropout, "use_rslora": use_rslora,
            "head_convention": base_manifest["head_convention"], "tokenizer_sha256": digest(base_manifest["tokenizer_files"]),
            "license": base_manifest["license"], "license_reviewed": True,
            "group_id": group_id, "training_seed": seed, "task_id": task_id,
            "software_only": base_manifest.get("software_only", False),
            "files": {name: file_hash(Path(path) / name) for name in ("adapter_config.json", "adapter_model.safetensors")}}


def train_local_adapter(base_path, base_manifest, records, *, seed, rank, alpha, dropout, steps, out, group_id,
                        learning_rate=0.02, device="cpu", dtype=torch.float32, software_only=False):
    """Train only LoRA from a freshly loaded, identical frozen base per call.

    This intentionally simple fixed-step, batch-one runner consumes actual
    training records. It is suitable for smoke and bounded local pilots; model,
    split, learning rate, context and step choices need protocol predeclaration.
    """
    from .task_data import prepare_record
    from peft import LoraConfig, get_peft_model
    if not records or isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
        raise ValueError("nonempty training records and positive fixed steps required")
    if not isinstance(seed, int) or not isinstance(group_id, str) or not group_id:
        raise ValueError("training seed and independent group identity required")
    if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0 or not math.isfinite(float(alpha)) or not alpha > 0 or not 0 <= dropout < 1 or not math.isfinite(float(learning_rate)) or not learning_rate > 0:
        raise ValueError("invalid declared LoRA/training hyperparameters")
    prepared = [prepare_record(record) for record in records]
    tasks = {record["task"] for record in prepared}
    if len(tasks) != 1:
        raise ValueError("each specialist adapter trains on one declared task")
    out = Path(out)
    if out.exists():
        raise FileExistsError("training output already exists")
    started = time.perf_counter()
    base, tokenizer = load_local_base(base_path, base_manifest, device=device, dtype=dtype, software_only=software_only)
    loading_seconds = time.perf_counter() - started
    base_hash = _parameter_hash(base)
    devices = [torch.device(device).index or 0] if str(device).startswith("cuda") else []
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            torch.use_deterministic_algorithms(True)
            config = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=dropout,
                                target_modules=list(TARGET_MODULES), task_type="CAUSAL_LM", bias="none",
                                base_model_name_or_path=base_manifest["model_id"], revision=base_manifest["revision"])
            model = get_peft_model(base, config)
            # PEFT may replace this with the local config name. Save the declared immutable identity.
            model.peft_config["default"].base_model_name_or_path = base_manifest["model_id"]
            model.peft_config["default"].revision = base_manifest["revision"]
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=learning_rate, weight_decay=0)
            order = list(range(len(prepared)))
            random.Random(seed).shuffle(order)
            losses = []
            _synchronize(device)
            fit_start = time.perf_counter()
            model.train()
            for step in range(steps):
                record = prepared[order[step % len(order)]]
                prefix = _tokens(tokenizer, record["prompt"])
                answer = "#### " + record["answer"] if record["task"] == "GSM8K" else record["answer"]
                completion = _tokens(tokenizer, " " + answer)
                if tokenizer.eos_token_id is not None:
                    completion.append(tokenizer.eos_token_id)
                ids = prefix + completion
                _context_guard(model, ids)
                inputs = torch.tensor([ids], dtype=torch.long, device=device)
                labels = inputs.clone()
                labels[:, :len(prefix)] = -100
                optimizer.zero_grad(set_to_none=True)
                loss = model(input_ids=inputs, attention_mask=torch.ones_like(inputs), labels=labels, use_cache=False).loss
                if not torch.isfinite(loss):
                    raise RuntimeError("nonfinite training loss")
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            _synchronize(device)
            fitting_seconds = time.perf_counter() - fit_start
            model.eval()
            if _parameter_hash(model, base_only=True) != base_hash:
                raise RuntimeError("common frozen base/head changed during training")
            out.parent.mkdir(parents=True, exist_ok=True)
            temporary = out.with_name(out.name + ".incomplete")
            temporary.mkdir(exist_ok=False)
            try:
                model.save_pretrained(temporary, safe_serialization=True, save_embedding_layers=False)
                manifest = _artifact_manifest(temporary, base_manifest, group_id=group_id, seed=seed, task_id=next(iter(tasks)),
                                              rank=rank, alpha=alpha, dropout=dropout)
                manifest["training"] = {"steps": steps, "learning_rate": learning_rate, "batch_size": 1,
                                        "record_ids": [record["example_id"] for record in prepared],
                                        "records_sha256": digest(records), "base_parameter_sha256": base_hash,
                                        "backend_source_sha256": file_hash(__file__),
                                        "losses": losses, "model_loading_seconds": loading_seconds,
                                        "training_seconds": fitting_seconds, "common_base_unchanged": True}
                validate_adapter_manifest(temporary, manifest, base_manifest, software_only=software_only)
                (temporary / "training_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
                os.replace(temporary, out)
            except BaseException:
                # Preserve failed output for the caller's error journal.
                raise
            return model, tokenizer, manifest
    finally:
        torch.use_deterministic_algorithms(previous_deterministic, warn_only=previous_warn_only)


def _rss():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def create_tiny_fixture(path, records):
    """Write a safe local random two-block Qwen plus a WordLevel tokenizer."""
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from .task_data import prepare_record
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    words = {"no", "yes", "1", "2", "A", "B", "C", "D", "####"}
    for record in records:
        prepared = prepare_record(record)
        words.update(re.findall(r"\w+|[^\w\s]", prepared["prompt"] + " " + prepared["answer"]))
    vocabulary = {word: i for i, word in enumerate(["[PAD]", "[BOS]", "[EOS]", "[UNK]", *sorted(words)])}
    core = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    core.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=core, pad_token="[PAD]", bos_token="[BOS]", eos_token="[EOS]", unk_token="[UNK]")
    tokenizer.save_pretrained(path)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(418)
        config = Qwen2Config(vocab_size=len(tokenizer), hidden_size=16, intermediate_size=32,
                             num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
                             max_position_embeddings=256, bos_token_id=1, eos_token_id=2,
                             pad_token_id=0, attention_dropout=0.0, tie_word_embeddings=True)
        model = Qwen2ForCausalLM(config)
        model.save_pretrained(path, safe_serialization=True)
    files = {p.name: file_hash(p) for p in path.iterdir() if p.is_file() and p.name not in TOKENIZER_FILES}
    manifest = {"model_id": "software-only/random-Qwen2-two-block", "revision": digest(files)[:40],
                "architecture": "Qwen2ForCausalLM", "license": "CC0-1.0 (locally generated random fixture weights)",
                "license_reviewed": True, "head_convention": "shared_frozen_causal_lm_head",
                "software_only": True, "files": files, "tokenizer_files": tokenizer_fingerprint(path)}
    validate_base_manifest(path, manifest, software_only=True)
    return manifest


def backend_smoke(out, records=None):
    """Measured CPU end-to-end software smoke; never accesses D_test records."""
    if records is None:
        records = {
            "train": [{"task": "BoolQ", "example_id": "smoke-train-1", "passage": "Copper conducts electricity.", "question": "Does copper conduct electricity?", "label": True},
                      {"task": "BoolQ", "example_id": "smoke-train-2", "passage": "Glass does not conduct electricity.", "question": "Does glass conduct electricity?", "label": False}],
            "pilot": [{"task": "BoolQ", "example_id": "smoke-pilot-1", "passage": "Maples are trees.", "question": "Is a maple a tree?", "label": True},
                      {"task": "BoolQ", "example_id": "smoke-pilot-2", "passage": "A stone is not a bird.", "question": "Is a stone a bird?", "label": False}]}
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    previous_threads = torch.get_num_threads()
    started = time.perf_counter()
    torch.set_num_threads(2)
    try:
        # Vocabulary derives from train + pilot only, never final held-out records.
        base_path = out / "tiny_base"
        manifest = create_tiny_fixture(base_path, records["train"] + records["pilot"])
        model_a, tokenizer, adapter_a = train_local_adapter(base_path, manifest, records["train"], seed=301, rank=2, alpha=4,
                                                          dropout=0, steps=4, out=out / "adapter_a", group_id="software-only-group-1", software_only=True)
        factors_a = extract_lora_updates(model_a)
        logits_inputs = torch.tensor([tokenizer.encode("Answer yes", add_special_tokens=False)])
        with torch.no_grad():
            expected_logits = model_a(logits_inputs).logits.detach()
        common_base, _ = load_local_base(base_path, manifest, software_only=True)
        reloaded = load_local_adapter(common_base, out / "adapter_a", adapter_a, manifest, software_only=True)
        with torch.no_grad():
            observed_logits = reloaded(logits_inputs).logits.detach()
        reload_error = float((expected_logits - observed_logits).abs().max())
        repeat, _, _ = train_local_adapter(base_path, manifest, records["train"], seed=301, rank=2, alpha=4,
                                          dropout=0, steps=4, out=out / "adapter_repeat", group_id="software-only-group-1-repeat", software_only=True)
        repeat_factors = extract_lora_updates(repeat)
        deterministic_error = max(float((factors_a[name].B - repeat_factors[name].B).abs().max()) for name in factors_a)
        deterministic_error = max(deterministic_error, max(float((factors_a[name].A - repeat_factors[name].A).abs().max()) for name in factors_a))
        model_b, _, adapter_b = train_local_adapter(base_path, manifest, records["train"], seed=302, rank=2, alpha=4,
                                                   dropout=0, steps=4, out=out / "adapter_b", group_id="software-only-group-2", software_only=True)
        factors_b = extract_lora_updates(model_b)
        compact, compact_meta = generate_candidate([factors_a, factors_b], "compact_qr_svd", 2, [0.4, 0.6])
        dense, dense_meta = generate_candidate([factors_a, factors_b], "dense_reference_svd", 2, [0.4, 0.6])
        exact, exact_meta = generate_candidate([factors_a, factors_b], "exact_concatenation", 2, [0.4, 0.6])
        parity_error = max(float((effective_delta(compact[name]) - effective_delta(dense[name])).abs().max()) for name in compact)
        exact_error = max(float((effective_delta(exact[name]) - weighted_sum([factors_a[name], factors_b[name]], [0.4, 0.6])).abs().max()) for name in exact)
        base, _ = load_local_base(base_path, manifest, software_only=True)
        head_before = base.lm_head.weight.detach().clone()
        base_predictions = evaluate_model(base, tokenizer, records["pilot"])
        specialist_predictions = [evaluate_model(model_a, tokenizer, records["pilot"]),
                                  evaluate_model(model_b, tokenizer, records["pilot"])]
        candidate = attach_candidate(base, compact)
        head_preserved = torch.equal(candidate.base_model.model.lm_head.weight, head_before)
        predictions = evaluate_model(candidate, tokenizer, records["pilot"])
        predictions_repeat = evaluate_model(candidate, tokenizer, records["pilot"])
        prediction_determinism = [{k: v for k, v in row.items() if k != "inference_seconds"} for row in predictions["rows"]] == [{k: v for k, v in row.items() if k != "inference_seconds"} for row in predictions_repeat["rows"]]
        passed = reload_error == 0 and deterministic_error == 0 and parity_error < 1e-10 and exact_error < 1e-10 and head_preserved and prediction_determinism
        result = {"status": "software_only", "passed": passed, "device": "cpu", "threads": 2,
                  "model": "random two-block Qwen2ForCausalLM with locally generated WordLevel tokenizer",
                  "model_manifest": manifest, "adapter_manifests": [adapter_a, adapter_b],
                  "multilayer_target_count": len(factors_a), "training_steps_per_adapter": 4,
                  "same_seed_factor_max_absolute_error": deterministic_error, "safe_reload_logits_max_absolute_error": reload_error,
                  "compact_dense_effective_delta_max_absolute_error": parity_error, "exact_weighted_delta_max_absolute_error": exact_error,
                  "shared_head_preserved": head_preserved, "predictions_deterministic": prediction_determinism,
                  "ordinary_base_scored_predictions": base_predictions,
                  "ordinary_specialist_scored_predictions": specialist_predictions,
                  "merged_scored_predictions": predictions, "baseline_capacity": [compact_meta, dense_meta, exact_meta],
                  "selection_policy_outcome_status": "not_run_software_loading_training_inference_smoke_only",
                  "wall_seconds": time.perf_counter() - started, "host_peak_rss_bytes": _rss(),
                  "gpu_peak_allocated_bytes": None, "gpu_peak_reserved_bytes": None, "accelerator_hours": 0,
                  "final_test_accessed": False, "independent_research_training_groups": 0,
                  "scope": "software exercise only; fixture task accuracy is not neural decision-utility evidence"}
        (out / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
        return result
    finally:
        torch.set_num_threads(previous_threads)
