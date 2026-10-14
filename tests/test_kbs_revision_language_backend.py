"""Real tiny-model integration and loader-boundary checks, all software-only."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

from src.kbs_revision.language_backend import (
    TARGET_MODULES, attach_candidate, backend_smoke, choice_loglikelihood,
    create_tiny_fixture, evaluate_model, extract_lora_updates, generate_candidate,
    load_local_adapter, load_local_base, train_local_adapter,
    validate_adapter_manifest, validate_base_manifest,
)
from src.kbs_revision.merge_baselines import LoRAUpdate, effective_delta, weighted_sum
from src.kbs_revision.protocol import file_hash


@pytest.fixture(scope="module")
def records():
    return [
        {"task": "BoolQ", "example_id": "software-1", "passage": "Water is a liquid.", "question": "Is water liquid?", "label": True},
        {"task": "BoolQ", "example_id": "software-2", "passage": "Rock is not a bird.", "question": "Is a rock a bird?", "label": False},
    ]


@pytest.fixture(scope="module", autouse=True)
def bounded_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def fixture(tmp_path_factory, records):
    root = tmp_path_factory.mktemp("kbs-real-transformer")
    base = root / "base"
    manifest = create_tiny_fixture(base, records)
    model, tokenizer, adapter = train_local_adapter(base, manifest, records, seed=73, rank=2, alpha=6,
                                                    dropout=.1, steps=3, out=root / "adapter", group_id="software-group-1",
                                                    software_only=True)
    return root, base, manifest, model, tokenizer, adapter


def test_local_base_has_two_real_qwen_blocks_and_roundtripped_wordlevel(fixture):
    _, base, manifest, _, _, _ = fixture
    model, tokenizer = load_local_base(base, manifest, software_only=True)
    assert model.config.num_hidden_layers == 2
    assert len(model.model.layers) == 2
    assert tokenizer.encode(" yes", add_special_tokens=False)
    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert next(model.parameters()).device.type == "cpu"
    assert all(path.suffix != ".bin" for path in base.iterdir())


def test_tiny_fixture_cannot_enter_pinned_research_cohort(fixture):
    _, base, manifest, _, _, _ = fixture
    with pytest.raises(ValueError, match="software-only"):
        validate_base_manifest(base, manifest)
    altered = {**manifest, "software_only": False}
    with pytest.raises(ValueError, match="pinned Qwen"):
        validate_base_manifest(base, altered)


def test_base_manifest_checks_weights_and_complete_tokenizer(fixture):
    _, base, manifest, _, _, _ = fixture
    altered = copy.deepcopy(manifest)
    altered["files"]["model.safetensors"] = "0" * 64
    with pytest.raises(ValueError, match="hash-mismatched"):
        validate_base_manifest(base, altered, software_only=True)
    altered = copy.deepcopy(manifest)
    altered["tokenizer_files"].pop("tokenizer_config.json")
    with pytest.raises(ValueError, match="complete tokenizer"):
        validate_base_manifest(base, altered, software_only=True)


@pytest.mark.parametrize("field,value", [("auto_map", {"AutoModel": "untrusted.py"}), ("quantization_config", {"load_in_4bit": True}),
                                        ("architectures", ["Qwen2ForSequenceClassification"])])
def test_base_rejects_remote_quantized_or_classifier_even_with_new_hash(fixture, tmp_path, field, value):
    import shutil
    _, base, manifest, _, _, _ = fixture
    path = tmp_path / "base"
    shutil.copytree(base, path)
    config = json.loads((path / "config.json").read_text())
    config[field] = value
    (path / "config.json").write_text(json.dumps(config))
    altered = copy.deepcopy(manifest)
    altered["files"]["config.json"] = file_hash(path / "config.json")
    with pytest.raises(ValueError, match="unsupported|architecture"):
        validate_base_manifest(path, altered, software_only=True)


def test_pickle_only_artifacts_rejected_before_loading(fixture, tmp_path):
    import shutil
    _, base, manifest, _, _, _ = fixture
    path = tmp_path / "base"
    shutil.copytree(base, path)
    (path / "pytorch_model.bin").write_bytes(b"not loaded")
    with pytest.raises(ValueError, match="pickle"):
        load_local_base(path, manifest, software_only=True)


def test_missing_base_weight_rejected_instead_of_random_initialization(fixture, tmp_path):
    import shutil
    from safetensors.torch import load_file, save_file
    _, base, manifest, _, _, _ = fixture
    path = tmp_path / "base"
    shutil.copytree(base, path)
    state = load_file(path / "model.safetensors")
    state.pop("model.layers.1.self_attn.q_proj.weight")
    save_file(state, path / "model.safetensors")
    altered = copy.deepcopy(manifest)
    altered["files"]["model.safetensors"] = file_hash(path / "model.safetensors")
    with pytest.raises(ValueError, match="complete compatible"):
        load_local_base(path, altered, software_only=True)


def test_adapter_contains_all_target_layers_with_nonunit_scaling(fixture):
    _, _, _, model, _, adapter = fixture
    updates = extract_lora_updates(model)
    assert len(updates) == 8
    assert {name.rsplit(".", 1)[-1] for name in updates} == set(TARGET_MODULES)
    assert all(update.scale == 3 for update in updates.values())
    assert all(update.B.dtype == torch.float64 and update.B.device.type == "cpu" for update in updates.values())
    assert adapter["training"]["common_base_unchanged"]
    assert len(adapter["training"]["losses"]) == 3
    for name, update in updates.items():
        torch.testing.assert_close(effective_delta(update).float(), dict(model.named_modules())[name].get_delta_weight("default"))


@pytest.mark.parametrize("field,value", [("base_revision", "0" * 40), ("tokenizer_sha256", "0" * 64), ("rank", 8),
                                        ("head_convention", "trainable_classifier"), ("license_reviewed", False)])
def test_incompatible_adapter_manifest_rejected(fixture, field, value):
    root, _, manifest, _, _, adapter = fixture
    altered = {**adapter, field: value}
    with pytest.raises(ValueError):
        validate_adapter_manifest(root / "adapter", altered, manifest, software_only=True)


@pytest.mark.parametrize("field,value", [("use_dora", True), ("use_qalora", True), ("modules_to_save", ["lm_head"]),
                                        ("bias", "all"), ("target_modules", ["q_proj"]), ("rank_pattern", {"q_proj": 7})])
def test_unsupported_peft_variants_rejected_even_with_pinned_config(fixture, tmp_path, field, value):
    import shutil
    root, _, manifest, _, _, adapter = fixture
    path = tmp_path / "adapter"
    shutil.copytree(root / "adapter", path)
    config = json.loads((path / "adapter_config.json").read_text())
    config[field] = value
    (path / "adapter_config.json").write_text(json.dumps(config))
    altered = copy.deepcopy(adapter)
    altered["files"]["adapter_config.json"] = file_hash(path / "adapter_config.json")
    with pytest.raises(ValueError, match="unsupported|target"):
        validate_adapter_manifest(path, altered, manifest, software_only=True)


def test_safe_reload_preserves_exact_logits_and_scores(fixture):
    root, base, manifest, model, tokenizer, adapter = fixture
    clean, _ = load_local_base(base, manifest, software_only=True)
    reloaded = load_local_adapter(clean, root / "adapter", adapter, manifest, software_only=True)
    inputs = torch.tensor([tokenizer.encode("Answer yes", add_special_tokens=False)])
    with torch.no_grad():
        torch.testing.assert_close(model(inputs).logits, reloaded(inputs).logits, rtol=0, atol=0)
    assert extract_lora_updates(reloaded).keys() == extract_lora_updates(model).keys()


def test_adapter_rejects_base_not_loaded_from_validated_manifest(fixture):
    from transformers import Qwen2ForCausalLM
    root, _, manifest, model, _, adapter = fixture
    unvalidated = Qwen2ForCausalLM(model.config)
    with pytest.raises(ValueError, match="validated immutable"):
        load_local_adapter(unvalidated, root / "adapter", adapter, manifest, software_only=True)


def test_wrong_adapter_layer_ids_rejected_before_peft_partial_load(fixture, tmp_path):
    import shutil
    from safetensors.torch import load_file, save_file
    root, base, manifest, _, _, adapter = fixture
    path = tmp_path / "adapter"
    shutil.copytree(root / "adapter", path)
    state = load_file(path / "adapter_model.safetensors")
    save_file({name.replace(".layers.1.", ".layers.9."): value for name, value in state.items()}, path / "adapter_model.safetensors")
    altered = copy.deepcopy(adapter)
    altered["files"]["adapter_model.safetensors"] = file_hash(path / "adapter_model.safetensors")
    clean, _ = load_local_base(base, manifest, software_only=True)
    with pytest.raises(ValueError, match="module identities/shapes"):
        load_local_adapter(clean, path, altered, manifest, software_only=True)
    assert not hasattr(clean, "peft_config")


def test_same_seed_training_is_exact_and_preserves_caller_rng(fixture, records):
    root, base, manifest, model, _, _ = fixture
    before = torch.random.get_rng_state()
    trained, _, _ = train_local_adapter(base, manifest, records, seed=73, rank=2, alpha=6, dropout=.1, steps=3,
                                       out=root / "repeat", group_id="software-group-repeat", software_only=True)
    assert torch.equal(before, torch.random.get_rng_state())
    expected, actual = extract_lora_updates(model), extract_lora_updates(trained)
    for name in expected:
        torch.testing.assert_close(expected[name].A, actual[name].A, atol=0, rtol=0)
        torch.testing.assert_close(expected[name].B, actual[name].B, atol=0, rtol=0)
    with pytest.raises(FileExistsError):
        train_local_adapter(base, manifest, records, seed=73, rank=2, alpha=6, dropout=.1, steps=3,
                            out=root / "repeat", group_id="software-group-repeat", software_only=True)


def test_multilayer_compact_dense_and_exact_weighted_semantics(fixture):
    _, _, _, model, _, _ = fixture
    first = extract_lora_updates(model)
    second = {name: LoRAUpdate(update.B * .7, update.A * 1.3, update.scale * 2) for name, update in first.items()}
    compact, compact_meta = generate_candidate([first, second], "compact_qr_svd", 2, [.8, -.3])
    dense, _ = generate_candidate([first, second], "dense_reference_svd", 2, [.8, -.3])
    exact, exact_meta = generate_candidate([first, second], "exact_concatenation", 2, [.8, -.3])
    assert exact_meta["output_parameters"] == compact_meta["output_parameters"] * 2
    assert set(exact_meta["output_ranks"].values()) == {4}
    for name in compact:
        torch.testing.assert_close(effective_delta(compact[name]), effective_delta(dense[name]), atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(effective_delta(exact[name]), weighted_sum([first[name], second[name]], [.8, -.3]), atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("method", ["task_arithmetic", "naive_factor_average", "pairwise_alignment", "global_synchronization", "ties", "dare_ties", "knots", "core_space"])
def test_every_candidate_runs_on_actual_trained_multilayer_factors(fixture, method):
    _, _, _, model, _, _ = fixture
    factors = extract_lora_updates(model)
    result, metadata = generate_candidate([factors, factors], method, 2, [.5, .5], density=.8, seed=15)
    assert len(result) == 8
    assert metadata["output_parameters"] > 0
    assert metadata["deployment_model_count"] == 1
    assert metadata["inference_latency_multiplier"] is None
    assert all(torch.isfinite(effective_delta(update)).all() for update in result.values())


def test_attached_candidate_matches_explicit_effective_delta_model_and_preserves_head(fixture, records):
    _, base, manifest, original, tokenizer, _ = fixture
    factors = extract_lora_updates(original)
    compact, _ = generate_candidate([factors, factors], "compact_qr_svd", 2)
    clean, _ = load_local_base(base, manifest, software_only=True)
    explicit, _ = load_local_base(base, manifest, software_only=True)
    head = clean.lm_head.weight.detach().clone()
    with torch.no_grad():
        for name, update in compact.items():
            explicit.get_submodule(name.removeprefix("base_model.model.")).weight.add_(effective_delta(update).float())
    candidate = attach_candidate(clean, compact)
    assert torch.equal(candidate.base_model.model.lm_head.weight, head)
    assert all(update.scale == 1 for update in extract_lora_updates(candidate).values())
    inputs = torch.tensor([tokenizer.encode("Answer yes", add_special_tokens=False)])
    with torch.no_grad():
        torch.testing.assert_close(candidate(inputs).logits, explicit(inputs).logits, atol=1e-7, rtol=1e-6)
    scored = evaluate_model(candidate, tokenizer, records)
    assert len(scored["rows"]) == 2
    assert 0 <= scored["accuracy"] <= 1
    assert all(isinstance(row["correct"], bool) and row["bounded_loss"] in {0, 1} for row in scored["rows"])


def test_choice_score_is_sum_of_actual_causal_logprobabilities(fixture):
    _, _, _, model, tokenizer, _ = fixture
    prompt, completion = "Answer", "yes no"
    prefix, suffix = tokenizer.encode(prompt, add_special_tokens=False), tokenizer.encode(" " + completion, add_special_tokens=False)
    ids = torch.tensor([prefix + suffix])
    with torch.no_grad():
        logits = model(ids).logits[0, len(prefix)-1:-1].log_softmax(-1)
        expected = float(logits.gather(1, torch.tensor(suffix)[:, None]).sum())
    assert choice_loglikelihood(model, tokenizer, prompt, completion) == pytest.approx(expected, abs=1e-7)


def test_context_limit_and_candidate_capacity_errors_are_explicit(fixture):
    _, _, _, model, tokenizer, _ = fixture
    with pytest.raises(ValueError, match="context"):
        choice_loglikelihood(model, tokenizer, "yes " * 300, "no")
    factors = extract_lora_updates(model)
    with pytest.raises(ValueError, match="capacity"):
        generate_candidate([factors], "naive_factor_average", rank=3)
    with pytest.raises(ValueError, match="target modules differ"):
        generate_candidate([factors, {}], "compact_qr_svd", rank=2)


def test_real_end_to_end_software_smoke_is_honestly_labeled(tmp_path):
    result = backend_smoke(tmp_path / "smoke")
    assert result["passed"]
    assert result["status"] == "software_only"
    assert result["multilayer_target_count"] == 8
    assert result["safe_reload_logits_max_absolute_error"] == 0
    assert result["same_seed_factor_max_absolute_error"] == 0
    assert result["accelerator_hours"] == 0
    assert result["independent_research_training_groups"] == 0
    assert result["final_test_accessed"] is False
    assert result["wall_seconds"] > 0
    assert json.loads((tmp_path / "smoke" / "result.json").read_text())["passed"]
