"""Offline language-task preparation, fixed scoring, and immutable split evidence.

This module never downloads a dataset. ``prepare_splits`` accepts already local
records plus pinned source metadata. Its JSONL files contain labels, so callers
must use ``load_split(..., allow_final_test=False)`` during pilot work. A source
``validation``/``test`` split is eligible only for ``D_test``; calibration and
training always come from source ``train``. A disjoint train holdout is permitted
when public evaluation data have not been supplied.

Multiple-choice candidates are scored by the *sum* of causal log probabilities
of the complete candidate continuation (including the fixed leading space).
No length normalization or answer-text guessing is used. GSM8K uses greedy
generation and exact equality of canonical finite decimals following an explicit
final-answer marker. These rules are software definitions, not a validated
benchmark result or a frozen confirmatory population.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from difflib import SequenceMatcher
import hashlib
import json
import math
from pathlib import Path
import re
import unicodedata

from .protocol import ROLES, canonical_bytes, digest, file_hash

TASKS = ("BoolQ", "PIQA", "ARC-Challenge", "GSM8K")
SCHEMA_VERSION = 1
NORMALIZATION_VERSION = "nfkc-casefold-space-numeric-preserving-v1"
PROMPT_VERSION = "plain-causal-task-prompts-v1"
NEAR_THRESHOLD = .95
DEFAULT_RATIOS = dict(zip(ROLES, (.60, .05, .05, .05, .05, .05, .15)))
_PIN = re.compile(r"(?:[a-f0-9]{40}|[a-f0-9]{64})\Z")
_NUMBER = re.compile(r"(?<![\w.])[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?![\w.])")
SCORING_SPEC = {
    "prompt_version": PROMPT_VERSION,
    "multiple_choice": {
        "score": "sum causal log-probabilities of every continuation token",
        "continuation_prefix": " ", "length_normalization": False,
        "tie_break": "first candidate in source order", "metric": "accuracy",
    },
    "GSM8K": {
        "decoding": "greedy", "metric": "exact-match",
        "extraction": "last explicit ####, Final answer:, or answer is marker; one finite decimal only",
        "numeric_equivalence": "Decimal canonicalization; commas and currency dollars accepted; no fractions/units/rounding",
        "missing_or_ambiguous_answer": "incorrect",
    },
}


def _task(record):
    task = record.get("task", record.get("task_id"))
    if task not in TASKS:
        raise ValueError(f"unsupported task: {task!r}")
    if record.get("task") and record.get("task_id") and record["task"] != record["task_id"]:
        raise ValueError("task and task_id disagree")
    return task


def _text(record, field):
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"missing/non-string {field}")
    return value.strip()


def _canonical_decimal(value):
    value = unicodedata.normalize("NFKC", str(value)).strip()
    if not re.fullmatch(r"[+-]?\$?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?", value):
        return None
    try:
        number = Decimal(value.replace("$", "").replace(",", ""))
    except InvalidOperation:
        return None
    if not number.is_finite():
        return None
    if number == 0:
        return "0"
    return format(number, "f").rstrip("0").rstrip(".") if "." in format(number, "f") else format(number, "f")


def extract_gsm8k_answer(text):
    """Extract the last explicit final-answer marker; never score a stray number.

    A final answer must be a single decimal without additional prose or units.
    Subsequent whitespace-only lines are allowed. Multiple numbers, fractions,
    percentages, exponents and a bare unmarked answer are invalid by definition.
    """
    if not isinstance(text, str):
        return None
    text = unicodedata.normalize("NFKC", text)
    markers = list(re.finditer(r"####|final\s+answer\s*:|answer\s+is\s*:?", text, re.I))
    if not markers:
        return None
    return _canonical_decimal(text[markers[-1].end():].strip())


def prepare_record(record):
    """Validate native task fields and return fixed prompt/candidates/target.

    Native schemas: BoolQ ``passage,question,label`` (bool or 0/1); PIQA
    ``goal,sol1,sol2,label`` (0/1); ARC ``question,choices,answerKey`` where
    choices are ``{label: [...], text: [...]}`` or a list of label/text mappings;
    GSM8K ``question,answer`` with the source's explicit ``####`` answer marker.
    The returned candidates exclude their common fixed leading space.
    """
    task = _task(record)
    example_id = record.get("example_id", record.get("id"))
    if example_id is None or isinstance(example_id, (dict, list, bool)) or not str(example_id).strip():
        raise ValueError("missing/invalid example_id")
    choices = None
    if task == "BoolQ":
        passage, question = _text(record, "passage"), _text(record, "question")
        label = record.get("label", record.get("answer"))
        if type(label) not in (bool, int) or label not in (False, True, 0, 1):
            raise ValueError("BoolQ label must be bool or 0/1")
        prompt = f"Passage: {passage}\nQuestion: {question}\nAnswer yes or no.\nAnswer:"
        choices, answer = ["no", "yes"], "yes" if label else "no"
    elif task == "PIQA":
        goal, sol1, sol2 = (_text(record, key) for key in ("goal", "sol1", "sol2"))
        label = record.get("label")
        if type(label) is not int or label not in (0, 1):
            raise ValueError("PIQA label must be 0/1")
        prompt = f"Goal: {goal}\n1. {sol1}\n2. {sol2}\nChoose the best solution, 1 or 2.\nAnswer:"
        choices, answer = ["1", "2"], str(label + 1)
    elif task == "ARC-Challenge":
        question, raw = _text(record, "question"), record.get("choices")
        if isinstance(raw, dict):
            labels, texts = raw.get("label", []), raw.get("text", [])
            if len(labels) != len(texts):
                raise ValueError("ARC choice labels/texts have different lengths")
            pairs = list(zip(labels, texts))
        elif isinstance(raw, list) and all(isinstance(item, dict) for item in raw):
            pairs = [(item.get("label"), item.get("text")) for item in raw]
        else:
            raise ValueError("invalid ARC choices")
        if len(pairs) < 2 or any(not isinstance(label, str) or not label.strip() or
                                not isinstance(text, str) or not text.strip() for label, text in pairs):
            raise ValueError("invalid ARC choice label/text")
        choices = [label.strip() for label, _ in pairs]
        if len(set(choices)) != len(choices):
            raise ValueError("duplicate ARC choice label")
        answer = str(record.get("answerKey", "")).strip()
        if answer not in choices:
            raise ValueError("ARC answerKey not in choices")
        rendered = "\n".join(f"{label}. {text.strip()}" for label, text in pairs)
        prompt = f"Question: {question}\n{rendered}\nChoose the correct option label.\nAnswer:"
    else:
        question, target = _text(record, "question"), _text(record, "answer")
        answer = extract_gsm8k_answer(target)
        if answer is None:
            raise ValueError("GSM8K source answer lacks an unambiguous explicit final decimal")
        prompt = f"Question: {question}\nSolve step by step, then write the final numerical answer after ####.\nAnswer:"
    return {"task": task, "task_id": task, "example_id": str(example_id), "prompt": prompt,
            "answer": answer, "choices": choices,
            "scoring": "causal_choice_loglikelihood" if choices else "greedy_gsm8k_exact_match",
            "prompt_version": PROMPT_VERSION}


def choose_prediction(task, choice_loglikelihoods, record):
    """Select the largest finite total log likelihood, breaking ties by source order."""
    prepared = prepare_record(record)
    if task != prepared["task"] or prepared["choices"] is None:
        raise ValueError("choice scoring requires the matching multiple-choice task")
    choices = prepared["choices"]
    if isinstance(choice_loglikelihoods, dict) and set(choice_loglikelihoods) != set(choices):
        raise ValueError("choice likelihoods must define exactly the source candidates")
    try:
        scores = ([float(choice_loglikelihoods[label]) for label in choices]
                  if isinstance(choice_loglikelihoods, dict) else [float(value) for value in choice_loglikelihoods])
    except (TypeError, ValueError) as exc:
        raise ValueError("choice likelihoods must be numeric") from exc
    if len(scores) != len(choices) or any(not math.isfinite(value) for value in scores):
        raise ValueError("choice likelihoods must be a finite score for every candidate")
    return choices[max(range(len(choices)), key=lambda index: float(scores[index]))]


def score_prediction(task, prediction, record):
    """Return bounded accuracy/exact-match for one prediction under frozen rules."""
    prepared = prepare_record(record)
    if task != prepared["task"]:
        raise ValueError("prediction task does not match record")
    actual = extract_gsm8k_answer(prediction) if task == "GSM8K" else str(prediction).strip()
    return float(actual is not None and actual == prepared["answer"])


def normalize_text(text):
    """Case/Unicode/space normalization preserving numeric meaning and punctuation.

    Decimal spellings ``1,000.0`` and ``1000`` match. Distinct numbers, minus
    signs, decimal points and fractions are retained; replacing all numerals by
    one placeholder would falsely collapse different arithmetic problems.
    """
    text = unicodedata.normalize("NFKC", text).casefold()
    text = text.translate(str.maketrans({"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u2212": "-"}))
    text = _NUMBER.sub(lambda match: _canonical_decimal(match.group()) or match.group(), text)
    return " ".join(text.split())


def _sha_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _record_content(record):
    return {key: value for key, value in record.items() if key not in
            {"content_sha256", "normalized_text_sha256", "text_sha256"}}


def _source_metadata(sources, task):
    source = sources.get(task)
    if not isinstance(source, dict):
        raise ValueError(f"missing source metadata for {task}")
    for field in ("dataset_id", "revision", "license"):
        if not isinstance(source.get(field), str) or not source[field].strip():
            raise ValueError(f"source {task} lacks {field}")
    if not _PIN.fullmatch(source["revision"]):
        raise ValueError(f"source {task} revision must be immutable git40 or content SHA256")
    if source.get("license_reviewed") is not True:
        raise ValueError(f"source {task} license not reviewed")
    if type(source.get("software_only", False)) is not bool:
        raise ValueError("source software_only flag must be boolean")
    if source.get("software_only") and not source["dataset_id"].startswith("software-only/"):
        raise ValueError("software-only source must use a software-only/ dataset identity")
    return source


def _manifest_record(raw, sources):
    prepared = prepare_record(raw)
    task = prepared["task"]
    source = _source_metadata(sources, task)
    source_split = raw.get("source_split")
    if source_split not in ("train", "validation", "test"):
        raise ValueError("source_split must be train, validation, or test")
    if raw.get("source_revision", source["revision"]) != source["revision"]:
        raise ValueError("record/source revision mismatch")
    payload = {key: value for key, value in raw.items() if key not in {
        "prompt", "text", "text_sha256", "normalized_text_sha256", "content_sha256", "role", "answer_target"}}
    payload.update({"task": task, "task_id": task, "example_id": prepared["example_id"],
                    "source_dataset": source["dataset_id"], "source_revision": source["revision"],
                    "source_split": source_split, "software_only": source.get("software_only", False),
                    "prompt_version": PROMPT_VERSION, "text": prepared["prompt"]})
    payload["text_sha256"] = _sha_text(payload["text"])
    payload["normalized_text_sha256"] = _sha_text(normalize_text(payload["text"]))
    payload["content_sha256"] = digest(_record_content(payload))
    return payload


def leakage_audit(role_records):
    """Recompute exact and conservative near-duplicate findings from actual text.

    Near candidates share a normalized three-token shingle (inverted index).
    Cross-role pairs with identical numeric inventories, >= 40 normalized
    characters and SequenceMatcher >= .95 are flagged for review. No assertion
    of universal semantic deduplication is made: this is a reproducible lexical
    audit, and any flagged pair blocks split preparation.
    """
    rows = [(role, record, normalize_text(_leakage_input_text(record)))
            for role in ROLES for record in role_records.get(role, [])]
    identities, exact = defaultdict(list), defaultdict(list)
    public_errors = []
    for role, record, normalized in rows:
        ref = {"role": role, "task_id": record["task_id"], "example_id": record["example_id"]}
        identities[(record["source_dataset"], record["source_split"], record["task_id"], record["example_id"])].append(ref)
        exact[_sha_text(normalized)].append(ref)
        if role != "D_test" and record["source_split"] != "train":
            public_errors.append(ref)
    identity_duplicates = [items for items in identities.values() if len(items) > 1]
    exact_duplicates = [items for items in exact.values() if len(items) > 1]
    inverted, candidate_pairs = defaultdict(list), set()
    for index, (role, _, normalized) in enumerate(rows):
        # Prompt scaffolding is not a useful index key and makes the candidate
        # set quadratic. Retain only raw task text shingles below instead.
        source_text = _leakage_input_text(rows[index][1])
        source_tokens = re.findall(r"\w+|[^\w\s]", normalize_text(source_text))
        shingles = {tuple(source_tokens[position:position + 3]) for position in range(len(source_tokens) - 2)}
        for shingle in sorted(shingles):
            inverted[shingle].append(index)
    excluded_common_shingles = 0
    for indices in inverted.values():
        # Explicitly bound ubiquitous language shingles. Exact duplicates are
        # still checked for the entire population, without this lexical limit.
        if len(indices) > 128:
            excluded_common_shingles += 1
            continue
        for position, prior in enumerate(indices):
            for index in indices[position + 1:]:
                if rows[prior][0] != rows[index][0]:
                    candidate_pairs.add((prior, index))
    near, compared = [], 0
    for left, right in sorted(candidate_pairs):
        lr, lrec, ltext = rows[left]
        rr, rrec, rtext = rows[right]
        if ltext == rtext or min(len(ltext), len(rtext)) < 40:
            continue
        if Counter(_NUMBER.findall(ltext)) != Counter(_NUMBER.findall(rtext)):
            continue
        if min(len(ltext), len(rtext)) / max(len(ltext), len(rtext)) < NEAR_THRESHOLD:
            continue
        compared += 1
        similarity = SequenceMatcher(None, ltext, rtext, autojunk=False).ratio()
        if similarity >= NEAR_THRESHOLD:
            near.append({"left": {"role": lr, "task_id": lrec["task_id"], "example_id": lrec["example_id"]},
                         "right": {"role": rr, "task_id": rrec["task_id"], "example_id": rrec["example_id"]},
                         "similarity": similarity})
    passed = not (identity_duplicates or exact_duplicates or near or public_errors)
    return {"algorithm": "cross-role shared-source-token-trigram plus SequenceMatcher",
            "normalization_version": NORMALIZATION_VERSION, "threshold": NEAR_THRESHOLD,
            "numeric_inventory_must_match": True, "minimum_characters": 40,
            "max_shingle_document_frequency": 128, "excluded_common_shingles": excluded_common_shingles,
            "scope": "lexical review; semantic paraphrases are not guaranteed detected",
            "records": len(rows), "candidate_pairs": len(candidate_pairs), "compared_pairs": compared,
            "identity_duplicates": identity_duplicates, "exact_duplicates": exact_duplicates,
            "near_duplicates": near, "public_evaluation_reused": public_errors, "passed": passed}


def _leakage_input_text(record):
    task = record["task_id"]
    if task == "BoolQ":
        return record["passage"] + "\n" + record["question"]
    if task == "PIQA":
        return "\n".join(record[key] for key in ("goal", "sol1", "sol2"))
    if task == "ARC-Challenge":
        choices = record["choices"]
        choice_texts = choices["text"] if isinstance(choices, dict) else [item["text"] for item in choices]
        return "\n".join([record["question"], *choice_texts])
    return record["question"]


def _allocate(size, ratios):
    if set(ratios) != set(ROLES) or any(not math.isfinite(float(value)) or value <= 0 for value in ratios.values()):
        raise ValueError("ratios must define all seven roles with positive finite weights")
    if size < len(ROLES):
        raise ValueError("at least seven unique source examples per task are required")
    total = sum(ratios.values())
    remaining = size - len(ROLES)
    quotas = {role: remaining * ratios[role] / total for role in ROLES}
    counts = {role: 1 + int(quotas[role]) for role in ROLES}
    for role in sorted(ROLES, key=lambda item: (-(quotas[item] % 1), ROLES.index(item)))[:size - sum(counts.values())]:
        counts[role] += 1
    return counts


def prepare_splits(records, source_metadata, out, *, seed=0, counts=None, ratios=None, resume=False):
    """Write deterministic, immutable seven-role JSONL and split_manifest.json.

    ``counts`` gives one positive count per role *per task*, or maps each task
    to such a role mapping. Without counts, positive ``ratios`` consume all
    supplied examples and enforce at least one example/task/role. Extra records
    under an explicit count plan are documented as unused. Ordering is a stable
    SHA256 of seed/task/source identity/content, independent of input ordering.
    Public validation/test records are used only for D_test. All other roles
    are partitioned from source train; insufficient data fail before writing.
    ``resume`` succeeds only for an identical, independently revalidated plan.
    """
    if counts is not None and ratios is not None:
        raise ValueError("choose counts or ratios")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    if not isinstance(source_metadata, dict):
        raise ValueError("source metadata must be a task mapping")
    sources = json.loads(canonical_bytes(source_metadata))
    prepared = [_manifest_record(raw, sources) for raw in records]
    if not prepared:
        raise ValueError("no local dataset records supplied")
    tasks = sorted({record["task_id"] for record in prepared})
    sources = {task: _source_metadata(sources, task) for task in tasks}
    grouped = defaultdict(list)
    for record in prepared:
        grouped[record["task_id"]].append(record)
    role_records, allocations, unused = {role: [] for role in ROLES}, {}, []
    for task in tasks:
        task_records = sorted(grouped[task], key=lambda record: digest({
            "seed": seed, "task": task, "source_split": record["source_split"],
            "example_id": record["example_id"], "content_sha256": record["content_sha256"]}))
        plan = (counts[task] if task in (counts or {}) and isinstance(counts[task], dict) else counts)
        plan = _allocate(len(task_records), ratios or DEFAULT_RATIOS) if plan is None else dict(plan)
        if set(plan) != set(ROLES) or any(type(value) is not int or value <= 0 for value in plan.values()):
            raise ValueError("counts must provide positive integers for all seven roles")
        train = [record for record in task_records if record["source_split"] == "train"]
        heldout = [record for record in task_records if record["source_split"] != "train"]
        test_pool = heldout if heldout else train
        required_train = sum(plan[role] for role in ROLES if role != "D_test") + (0 if heldout else plan["D_test"])
        if len(train) < required_train or len(test_pool) < plan["D_test"]:
            raise ValueError(f"insufficient eligible train/heldout data for {task}")
        selected_ids = set()
        for role in ROLES:
            pool = test_pool if role == "D_test" and heldout else train
            chosen = pool[:plan[role]]
            del pool[:plan[role]]
            for record in chosen:
                selected_ids.add(id(record))
                assigned = dict(record, role=role)
                assigned["content_sha256"] = digest(_record_content(assigned))
                role_records[role].append(assigned)
        unused.extend({"task_id": task, "example_id": record["example_id"], "source_split": record["source_split"],
                       "content_sha256": record["content_sha256"]}
                      for record in task_records if id(record) not in selected_ids)
        allocations[task] = plan
    audit = leakage_audit(role_records)
    if not audit["passed"]:
        raise ValueError("split leakage audit failed: " + json.dumps(audit, sort_keys=True))
    encoded = {role: b"".join(canonical_bytes(record) for record in role_records[role]) for role in ROLES}
    manifest = {"schema_version": SCHEMA_VERSION, "status": "software_only" if all(
        source.get("software_only") for source in sources.values()) else "prepared_not_frozen",
        "not_benchmark_evidence": all(source.get("software_only") for source in sources.values()),
        "seed": seed, "source_metadata": sources, "tasks": tasks, "allocation": allocations,
        "input_fingerprint": digest(sorted(record["content_sha256"] for record in prepared)),
        "unused_records": unused, "scoring_spec": SCORING_SPEC, "near_duplicate_audit": audit,
        "final_test_access": "prohibited during pilot; explicit allow_final_test required",
        "splits": {role: {"path": f"{role}.jsonl", "sha256": hashlib.sha256(encoded[role]).hexdigest(),
                         "count": len(role_records[role]), "task_counts": dict(Counter(
                             record["task_id"] for record in role_records[role]))} for role in ROLES}}
    manifest["manifest_sha256"] = digest(manifest)
    out = Path(out)
    if out.exists():
        if not resume:
            raise FileExistsError(f"immutable split output already exists: {out}")
        errors = validate_split_manifest(out / "split_manifest.json")
        if errors or json.loads((out / "split_manifest.json").read_text()) != manifest:
            raise ValueError("resume split plan differs or stored evidence failed validation: " + "; ".join(errors))
        return manifest
    out.mkdir(parents=True, exist_ok=False)
    for role in ROLES:
        with (out / f"{role}.jsonl").open("xb") as handle:
            handle.write(encoded[role])
    with (out / "split_manifest.json").open("xb") as handle:
        handle.write(canonical_bytes(manifest))
    return manifest


def validate_split_manifest(manifest_or_path, root=None):
    """Return errors after re-reading records and recomputing every claimed audit.

    No claimed ``passed`` flag, text hash or content hash is trusted. Relative
    paths may not escape root. This validates the delivered population, not
    unseen source-corpus exclusion or a universal semantic duplicate guarantee.
    """
    errors = []
    try:
        if isinstance(manifest_or_path, (str, Path)):
            path = Path(manifest_or_path)
            manifest = json.loads(path.read_text())
            root = path.parent if root is None else Path(root)
        else:
            manifest = manifest_or_path
            if root is None:
                raise ValueError("root required for an in-memory split manifest")
            root = Path(root)
        if not isinstance(manifest, dict) or manifest.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported split manifest schema")
        if digest({key: value for key, value in manifest.items() if key != "manifest_sha256"}) != manifest.get("manifest_sha256"):
            errors.append("split manifest checksum mismatch")
        if manifest.get("scoring_spec") != SCORING_SPEC:
            errors.append("task scoring specification mismatch")
        if set(manifest.get("splits", {})) != set(ROLES):
            errors.append("split manifest must define exactly seven roles")
        role_records = {role: [] for role in ROLES}
        for role in ROLES:
            try:
                spec = manifest["splits"][role]
                relative = Path(spec["path"])
                target = (root / relative).resolve()
                if relative.is_absolute() or not target.is_relative_to(root.resolve()):
                    raise ValueError("split path escapes manifest root")
                if file_hash(target) != spec.get("sha256"):
                    errors.append(f"split file checksum mismatch: {role}")
                lines = target.read_text().splitlines()
                if any(not line.strip() for line in lines):
                    errors.append(f"blank/noncanonical JSONL line: {role}")
                records = [json.loads(line) for line in lines if line.strip()]
                role_records[role] = records
                if not records or len(records) != spec.get("count"):
                    errors.append(f"empty/count-mismatched split: {role}")
                if dict(Counter(record.get("task_id") for record in records)) != spec.get("task_counts"):
                    errors.append(f"task counts mismatch: {role}")
                for record in records:
                    task = _task(record)
                    source = _source_metadata(manifest["source_metadata"], task)
                    original = _manifest_record(record, manifest["source_metadata"])
                    if record.get("role") != role:
                        errors.append(f"record role mismatch: {role}")
                    for field in ("source_dataset", "source_revision", "software_only", "text", "text_sha256", "normalized_text_sha256", "prompt_version"):
                        if record.get(field) != original[field]:
                            errors.append(f"record {field} mismatch: {role}/{record.get('example_id')}")
                    if record.get("content_sha256") != digest(_record_content(record)):
                        errors.append(f"record content hash mismatch: {role}/{record.get('example_id')}")
                    if record.get("source_revision") != source["revision"]:
                        errors.append(f"record source revision mismatch: {role}")
            except (OSError, ValueError, KeyError, TypeError) as exc:
                errors.append(f"invalid split {role}: {exc}")
        try:
            actual_audit = leakage_audit(role_records)
            if actual_audit != manifest.get("near_duplicate_audit"):
                errors.append("claimed leakage audit differs from recomputed records")
            if not actual_audit["passed"]:
                errors.append("recomputed split leakage audit failed")
            actual_tasks = sorted({record["task_id"] for records in role_records.values() for record in records})
            if manifest.get("tasks") != actual_tasks:
                errors.append("manifest task population differs from records")
            for task in actual_tasks:
                expected = {role: sum(record["task_id"] == task for record in role_records[role]) for role in ROLES}
                if expected != manifest.get("allocation", {}).get(task) or any(count == 0 for count in expected.values()):
                    errors.append(f"allocation/nonempty task-role mismatch: {task}")
            input_hashes = [digest(_record_content({key: value for key, value in record.items() if key != "role"}))
                            for records in role_records.values() for record in records]
            for record in manifest.get("unused_records", []):
                value = record.get("content_sha256")
                if not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{64}", value):
                    raise ValueError("unused record lacks source content checksum")
                input_hashes.append(value)
            if digest(sorted(input_hashes)) != manifest.get("input_fingerprint"):
                errors.append("input population fingerprint mismatch")
            software_only = all(_source_metadata(manifest["source_metadata"], task).get("software_only") for task in actual_tasks)
            if manifest.get("status") != ("software_only" if software_only else "prepared_not_frozen") or manifest.get("not_benchmark_evidence") != software_only:
                errors.append("software-only benchmark status mismatch")
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"invalid leakage audit records: {exc}")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        errors.append(f"invalid split manifest: {exc}")
    return sorted(set(errors))


def load_split(manifest_path, role, *, allow_final_test=False):
    """Load validated records; deliberate access is required to read D_test labels."""
    if role not in ROLES:
        raise ValueError("unknown data role")
    if role == "D_test" and not allow_final_test:
        raise PermissionError("final test access prohibited during pilot")
    errors = validate_split_manifest(manifest_path)
    if errors:
        raise ValueError("invalid split evidence: " + "; ".join(errors))
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    return [json.loads(line) for line in (manifest_path.parent / manifest["splits"][role]["path"]).read_text().splitlines()]


def software_only_fixture_records():
    """Return 28 invented records plus source metadata for seven-role software QA.

    Four task schemas are exercised. These are not benchmark data, model training
    groups or evidence about neural task accuracy. Each task supplies seven
    distinct source-train records, sufficient for counts={role: 1 for role in ROLES}.
    """
    records = []
    names = ("amber", "birch", "coral", "dahlia", "elm", "fern", "granite")
    for index, name in enumerate(names):
        shared = {"example_id": f"fixture-{index}", "source_split": "train"}
        records.append(dict(shared, task="BoolQ", passage=f"The {name} container holds {index + 2} blue stones.",
                            question=f"Does the {name} container hold {index + 2} blue stones?", label=True))
        records.append(dict(shared, task="PIQA", goal=f"Carry {index + 2} {name} stones safely.",
                            sol1=f"Place the {index + 2} stones in a sturdy basket.",
                            sol2=f"Balance the {index + 2} stones on a loose thread.", label=0))
        records.append(dict(shared, task="ARC-Challenge", question=f"Which object can hold {index + 2} {name} stones?",
                            choices={"label": ["A", "B"], "text": ["a basket", "a ray of sunlight"]}, answerKey="A"))
        records.append(dict(shared, task="GSM8K", question=f"A {name} box contains {index + 2} stones. Add {index + 3} stones. How many are in it?",
                            answer=f"Add the counts. #### {2 * index + 5}"))
    sources = {task: {"dataset_id": f"software-only/{task}", "revision": digest({"task": task, "fixture_version": 1}),
                      "license": "CC0-1.0", "license_reviewed": True, "software_only": True,
                      "origin": "invented software fixture; no benchmark population"} for task in TASKS}
    return records, sources
