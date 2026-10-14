import copy
import json

import pytest

from src.kbs_revision.protocol import ROLES, canonical_bytes, digest, file_hash
from src.kbs_revision.task_data import (
    choose_prediction, extract_gsm8k_answer, load_split, normalize_text,
    prepare_record, prepare_splits, score_prediction, software_only_fixture_records,
    validate_split_manifest,
)


def _fixture(tmp_path, name="splits", records=None, sources=None):
    if records is None:
        records, sources = software_only_fixture_records()
    root = tmp_path / name
    manifest = prepare_splits(records, sources, root, seed=17, counts=dict.fromkeys(ROLES, 1))
    return root, manifest


def _write_manifest(root, manifest):
    manifest["manifest_sha256"] = digest({key: value for key, value in manifest.items() if key != "manifest_sha256"})
    (root / "split_manifest.json").write_bytes(canonical_bytes(manifest))


def _replace_role(root, manifest, role, records):
    path = root / manifest["splits"][role]["path"]
    path.write_bytes(b"".join(canonical_bytes(record) for record in records))
    manifest["splits"][role]["sha256"] = file_hash(path)
    _write_manifest(root, manifest)


def test_all_task_prompts_and_choice_scores_are_explicit():
    records, _ = software_only_fixture_records()
    prepared = {record["task"]: prepare_record(record) for record in records[:4]}
    assert prepared["BoolQ"]["choices"] == ["no", "yes"]
    assert prepared["PIQA"]["choices"] == ["1", "2"]
    assert prepared["ARC-Challenge"]["choices"] == ["A", "B"]
    assert prepared["GSM8K"]["choices"] is None
    assert prepared["GSM8K"]["answer"] == "5"
    for record in records[:3]:
        item = prepare_record(record)
        assert item["prompt"].endswith("Answer:")
        assert choose_prediction(record["task"], [-3., -3.], record) == item["choices"][0]
        assert choose_prediction(record["task"], [-3., -2.], record) == item["choices"][1]
        assert score_prediction(record["task"], item["answer"], record) == 1.
        assert score_prediction(record["task"], "a full explanation", record) == 0.


def test_arc_preserves_numeric_labels_and_source_order():
    record = {"task": "ARC-Challenge", "example_id": "numeric", "question": "What floats?",
              "choices": [{"label": "2", "text": "stone"}, {"label": "1", "text": "cork"}], "answerKey": "1"}
    assert prepare_record(record)["choices"] == ["2", "1"]
    assert choose_prediction("ARC-Challenge", {"1": -2., "2": -2.}, record) == "2"
    assert score_prediction("ARC-Challenge", "1", record) == 1.


@pytest.mark.parametrize("prediction, expected", [
    ("Reasoning gives 25. #### 1,000.00", "1000"),
    ("Reasoning. Final answer: -02.50", "-2.5"),
    ("The answer is 42\n", "42"), ("#### ２.０", "2"),
    ("#### -0.000", "0"), ("#### 1\n#### 2", "2"),
    ("42", None), ("Reasoning has 42 apples.", None),
    ("#### 2 or 3", None), ("#### 1/2", None), ("#### 2 apples", None),
    ("#### 2.0.0", None), ("#### 2%", None), ("#### NaN", None),
])
def test_gsm8k_extraction_is_predeclared(prediction, expected):
    assert extract_gsm8k_answer(prediction) == expected


def test_gsm8k_exact_match_does_not_score_intermediate_numbers():
    record = {"task": "GSM8K", "example_id": "sum", "question": "What is two plus three?", "answer": "2+3=5. #### 5"}
    assert score_prediction("GSM8K", "I considered 5 but conclude #### 6", record) == 0.
    assert score_prediction("GSM8K", "Final answer: 5.000", record) == 1.
    with pytest.raises(ValueError, match="unambiguous"):
        prepare_record(dict(record, answer="The answer is unspecified"))


def test_scoring_rejects_bad_schema_and_nonfinite_scores():
    records, _ = software_only_fixture_records()
    with pytest.raises(ValueError, match="finite"):
        choose_prediction("BoolQ", [-1., float("nan")], records[0])
    with pytest.raises(ValueError, match="BoolQ label"):
        prepare_record(dict(records[0], label="yes"))
    with pytest.raises(ValueError, match="prediction task"):
        score_prediction("PIQA", "yes", records[0])
    with pytest.raises(ValueError, match="answerKey"):
        prepare_record(dict(records[2], answerKey="C"))


def test_normalization_keeps_arithmetic_distinctions():
    assert normalize_text("  Ａ   has 1,000.00 ITEMS ") == "a has 1000 items"
    assert normalize_text("x has -2 stones") != normalize_text("x has 2 stones")
    assert normalize_text("x has 2.5 stones") != normalize_text("x has 25 stones")
    assert normalize_text("x has 2/3 stones") != normalize_text("x has 23 stones")
    assert normalize_text("x has 2 stones") != normalize_text("x has 3 stones")


def test_seven_actual_jsonl_roles_and_order_independent_plan(tmp_path):
    records, sources = software_only_fixture_records()
    root, manifest = _fixture(tmp_path, records=records, sources=sources)
    other, reordered = _fixture(tmp_path, "reordered", list(reversed(records)), sources)
    assert manifest == reordered
    assert validate_split_manifest(root / "split_manifest.json") == []
    assert manifest["status"] == "software_only"
    assert manifest["not_benchmark_evidence"]
    assert manifest["near_duplicate_audit"]["passed"]
    assert manifest["near_duplicate_audit"]["records"] == 28
    for role in ROLES:
        assert (root / f"{role}.jsonl").read_bytes() == (other / f"{role}.jsonl").read_bytes()
        assert manifest["splits"][role]["count"] == 4


def test_immutable_resume_validates_full_evidence_and_plan(tmp_path):
    records, sources = software_only_fixture_records()
    root, manifest = _fixture(tmp_path, records=records, sources=sources)
    before = {path.name: path.read_bytes() for path in root.iterdir()}
    with pytest.raises(FileExistsError, match="immutable"):
        prepare_splits(records, sources, root, seed=17, counts=dict.fromkeys(ROLES, 1))
    assert prepare_splits(records, sources, root, seed=17, counts=dict.fromkeys(ROLES, 1), resume=True) == manifest
    assert before == {path.name: path.read_bytes() for path in root.iterdir()}
    with pytest.raises(ValueError, match="resume"):
        prepare_splits(records, sources, root, seed=18, counts=dict.fromkeys(ROLES, 1), resume=True)
    (root / "pilot.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="resume"):
        prepare_splits(records, sources, root, seed=17, counts=dict.fromkeys(ROLES, 1), resume=True)


def test_final_test_loading_requires_deliberate_access(tmp_path):
    root, _ = _fixture(tmp_path)
    assert len(load_split(root / "split_manifest.json", "pilot")) == 4
    with pytest.raises(PermissionError, match="final test"):
        load_split(root / "split_manifest.json", "D_test")
    assert len(load_split(root / "split_manifest.json", "D_test", allow_final_test=True)) == 4


def test_public_test_records_never_become_training_or_calibration(tmp_path):
    records, sources = software_only_fixture_records()
    for record in records:
        if record["example_id"] == "fixture-6":
            record["source_split"] = "test"
    root, _ = _fixture(tmp_path, records=records, sources=sources)
    for role in ROLES:
        actual = [json.loads(line) for line in (root / f"{role}.jsonl").read_text().splitlines()]
        assert all(record["source_split"] == ("test" if role == "D_test" else "train") for record in actual)
    for record in records:
        if record["example_id"] == "fixture-5":
            record["source_split"] = "validation"
    with pytest.raises(ValueError, match="insufficient eligible"):
        _fixture(tmp_path, "not-enough-train", records, sources)
    assert not (tmp_path / "not-enough-train").exists()


def test_source_versions_and_licenses_are_pinned_before_any_write(tmp_path):
    records, sources = software_only_fixture_records()
    sources["BoolQ"]["revision"] = "main"
    with pytest.raises(ValueError, match="immutable"):
        _fixture(tmp_path, records=records, sources=sources)
    assert not (tmp_path / "splits").exists()
    _, sources = software_only_fixture_records()
    sources["BoolQ"]["license_reviewed"] = False
    with pytest.raises(ValueError, match="license"):
        _fixture(tmp_path, records=records, sources=sources)
    _, sources = software_only_fixture_records()
    records[0]["source_revision"] = "a" * 40
    with pytest.raises(ValueError, match="revision mismatch"):
        _fixture(tmp_path, records=records, sources=sources)


def test_exact_duplicates_hidden_by_new_ids_fail_before_write(tmp_path):
    records, sources = software_only_fixture_records()
    boolq = [record for record in records if record["task"] == "BoolQ"]
    boolq[1].update(passage=boolq[0]["passage"], question=boolq[0]["question"])
    with pytest.raises(ValueError, match="exact_duplicates"):
        _fixture(tmp_path, records=records, sources=sources)
    assert not (tmp_path / "splits").exists()


def test_same_source_identity_with_different_content_is_rejected(tmp_path):
    records, sources = software_only_fixture_records()
    records[4]["example_id"] = records[0]["example_id"]
    with pytest.raises(ValueError, match="identity_duplicates"):
        _fixture(tmp_path, records=records, sources=sources)


def test_case_unicode_numeric_spelling_duplicate_is_detected(tmp_path):
    records, sources = software_only_fixture_records()
    boolq = [record for record in records if record["task"] == "BoolQ"]
    boolq[1].update(passage=boolq[0]["passage"].upper().replace("2", "２.０"),
                    question=boolq[0]["question"].upper().replace("2", "２.０"))
    with pytest.raises(ValueError, match="exact_duplicates"):
        _fixture(tmp_path, records=records, sources=sources)


def test_near_duplicate_minor_edit_blocks_cross_role_leakage(tmp_path):
    records, sources = software_only_fixture_records()
    boolq = [record for record in records if record["task"] == "BoolQ"]
    boolq[0]["passage"] = "A basket contains many polished blue stones beside a green wooden bridge overlooking the quiet river."
    boolq[0]["question"] = "Is the basket beside a bridge?"
    boolq[1]["passage"] = boolq[0]["passage"].replace("polished", "polishes")
    boolq[1]["question"] = boolq[0]["question"]
    with pytest.raises(ValueError, match="near_duplicates"):
        _fixture(tmp_path, records=records, sources=sources)


def test_rehashed_tampering_cannot_hide_false_record_hash_or_false_audit(tmp_path):
    root, manifest = _fixture(tmp_path)
    role = "pilot"
    records = [json.loads(line) for line in (root / f"{role}.jsonl").read_text().splitlines()]
    records[0]["normalized_text_sha256"] = "a" * 64
    records[0]["content_sha256"] = "b" * 64
    _replace_role(root, manifest, role, records)
    errors = validate_split_manifest(root / "split_manifest.json")
    assert any("normalized_text_sha256 mismatch" in error for error in errors)
    assert any("content hash" in error for error in errors)
    root2, manifest2 = _fixture(tmp_path, "false-audit")
    manifest2["near_duplicate_audit"]["compared_pairs"] += 1
    _write_manifest(root2, manifest2)
    assert any("claimed leakage audit" in error for error in validate_split_manifest(root2 / "split_manifest.json"))


def test_rehashed_public_test_misuse_and_content_changes_are_recomputed(tmp_path):
    root, manifest = _fixture(tmp_path)
    records = [json.loads(line) for line in (root / "D_cert.jsonl").read_text().splitlines()]
    records[0]["source_split"] = "test"
    records[0]["content_sha256"] = digest({key: value for key, value in records[0].items()
                                         if key not in {"content_sha256", "text_sha256", "normalized_text_sha256"}})
    _replace_role(root, manifest, "D_cert", records)
    errors = validate_split_manifest(root / "split_manifest.json")
    assert "recomputed split leakage audit failed" in errors
    assert "input population fingerprint mismatch" in errors


def test_manifest_paths_do_not_escape_root(tmp_path):
    root, manifest = _fixture(tmp_path)
    manifest["splits"]["pilot"]["path"] = "../outside.jsonl"
    _write_manifest(root, manifest)
    assert any("escapes manifest root" in error for error in validate_split_manifest(root / "split_manifest.json"))


def test_ratio_partition_consumes_inputs_and_preserves_nonempty_roles(tmp_path):
    records, sources = software_only_fixture_records()
    manifest = prepare_splits(records, sources, tmp_path / "ratios", ratios=dict.fromkeys(ROLES, 1))
    assert manifest["unused_records"] == []
    assert validate_split_manifest(tmp_path / "ratios" / "split_manifest.json") == []
    with pytest.raises(ValueError, match="positive"):
        prepare_splits(records, sources, tmp_path / "bad-ratios", ratios=dict.fromkeys(ROLES, 0))


def test_explicit_count_plan_records_unused_local_examples(tmp_path):
    records, sources = software_only_fixture_records()
    extra = copy.deepcopy(records[0])
    extra.update(example_id="extra", passage="A red lamp lights 99 books.", question="Does a red lamp light 99 books?")
    root, manifest = _fixture(tmp_path, records=records + [extra], sources=sources)
    assert len(manifest["unused_records"]) == 1
    assert validate_split_manifest(root / "split_manifest.json") == []
