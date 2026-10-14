"""Read-only evidence inventory and historical scalar recomputation.

No checkpoint deserialization, remote code execution, weight download, training,
or write is performed here. CSV data rows are numbered from one, excluding the
header; ``csv_line`` is the corresponding physical line for these flat CSVs.
"""
from __future__ import annotations

import concurrent.futures
import csv
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
from datetime import datetime, timezone
import urllib.request

import numpy as np


LORA = "reports/practical_twistedmerge/lora_practical_followup/real_adapter_gauge"
SELECTOR = "reports/post_iclr_v2/selector_attribution"
DIAGNOSTIC = "reports/next_benchmarks"
NATURAL = "reports/holonomy_applications/application_B_brauer_certificate"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(root: Path, *args: str) -> str | None:
    result = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def _artifact(root: Path, name: str, rows: list[int] | None = None) -> dict:
    path = root / name
    result = {"path": name, "exists": path.is_file()}
    if path.is_file():
        result.update(sha256=sha256(path), bytes=path.stat().st_size,
                      artifact_commit=_git(root, "log", "-1", "--format=%H", "--", name))
    if rows is not None:
        result.update(data_rows=rows, csv_lines=[i + 1 for i in rows])
    return result


def _csv(root: Path, name: str) -> list[dict]:
    path = root / name
    if not path.is_file():
        raise FileNotFoundError(name)
    with path.open(newline="") as stream:
        return [dict(row, source_data_row=i) for i, row in enumerate(csv.DictReader(stream), 1)]


def _json(root: Path, name: str) -> dict:
    return json.loads((root / name).read_text())


def _config(root: Path, name: str) -> dict:
    data = _json(root, name)
    execution = next((data[k] for k in ["execution_commit", "git_commit_at_execution", "git_commit"]
                      if k in data), None)
    command = next((data[k] for k in ["exact_command", "command", "confirmatory_command"]
                    if k in data), None)
    dirty = next((data[k] for k in ["worktree_dirty_during_execution", "git_worktree_dirty_at_execution"]
                  if k in data), None)
    return {"artifact": _artifact(root, name), "execution_commit": execution,
            "recorded_command": command, "worktree_dirty_at_execution": dirty,
            "execution_provenance_complete": dirty is False,
            "limitation": ("Dirty execution: a commit alone does not identify executed source."
                           if dirty else "Execution source hashes or dirty-tree status not recorded."
                           if dirty is None else None)}


def _legacy_diagnostics(root: Path) -> dict:
    name = f"{DIAGNOSTIC}/diagnostic_prediction_runs.csv"
    rows = _csv(root, name)
    y = np.asarray([float(r["weight_average_degradation"]) for r in rows])
    folds = np.asarray([r["setting_group"] for r in rows])
    summaries = _csv(root, f"{DIAGNOSTIC}/diagnostic_prediction_summary.csv")
    values = {}
    for predictor in ["cycle_residual", "pairwise_alignment_loss"]:
        x = np.asarray([float(r[predictor]) for r in rows])
        predicted = np.empty(len(y))
        for fold in sorted(set(folds)):
            train = folds != fold
            test = ~train
            design = np.column_stack([np.ones(int(train.sum())), x[train]])
            coef = np.linalg.lstsq(design, y[train], rcond=None)[0]
            predicted[test] = coef[0] + coef[1] * x[test]
        r2 = float(1 - np.sum((y - predicted) ** 2) / np.sum((y - y.mean()) ** 2))
        record = next(r for r in summaries if r["predictor"] == predictor
                      and r["target"] == "weight_average_degradation")
        expected = float(record["heldout_r2"])
        values[predictor] = {"heldout_r2": r2, "recorded_heldout_r2": expected,
                             "absolute_recompute_error": abs(r2 - expected),
                             "summary_source_data_row": record["source_data_row"]}
    return {"claim_id": "legacy_natural_diagnostic_prediction", "status": "legacy_reproduced",
            "n_checkpoint_collections": len(rows), "n_distinct_seed_labels": len({r["seed"] for r in rows}),
            "fold_definition": "leave one entire n_models,width setting out; ordinary scalar least squares",
            "recomputation": values, "provenance": _config(root, f"{DIAGNOSTIC}/diagnostic_prediction_config.json"),
            "sources": [_artifact(root, name, list(range(1, len(rows) + 1))),
                        _artifact(root, f"{DIAGNOSTIC}/diagnostic_prediction_summary.csv",
                                  [values[k]["summary_source_data_row"] for k in values]),
                        _artifact(root, "reports/csv/improved_validated_ladder_merge_benchmark.csv"),
                        _artifact(root, "experiments/heldout_diagnostic_prediction.py")],
            "limitations": ["Historical image-MLP collections are not a new multilayer language-model cohort.",
                            "120 collections are not 120 independent training groups.",
                            "Closure and exact distance to coboundaries were unavailable; harmful-merge AUC has one class."]}


def _legacy_selector(root: Path) -> dict:
    rows = _csv(root, f"{SELECTOR}/selectors.csv")
    by_setting = {}
    for row in rows:
        by_setting.setdefault(row["setting_id"], {})[row["selector"]] = row
    paired = []
    group_values = {}
    for setting in sorted(by_setting):
        pair = by_setting[setting]
        a, b = pair["A5"], pair["B0"]
        delta = float(a["test_accuracy"]) - float(b["test_accuracy"])
        seed = int(a["seed"])
        group_values.setdefault(seed, []).append(delta)
        paired.append({"setting_id": setting, "group_id": seed, "paired_accuracy_delta": delta,
                       "source_rows": [a["source_data_row"], b["source_data_row"]]})
    grouped = np.asarray([np.mean(group_values[s]) for s in sorted(group_values)])
    # Original implementation: baseline B0 is comparison index 5, 4000 draws.
    rng = np.random.default_rng(9915)
    draws = rng.choice(grouped, size=(4000, len(grouped)), replace=True).mean(axis=1)
    ci = np.percentile(draws, [2.5, 97.5])
    recorded = next(r for r in _csv(root, f"{SELECTOR}/paired.csv") if r["comparison"] == "A5_minus_B0")
    means = {method: float(np.mean([float(r["test_accuracy"]) for r in rows if r["selector"] == method]))
             for method in ["A5", "B0"]}
    recorded_ci = [float(recorded["paired_accuracy_delta_ci_low"]),
                   float(recorded["paired_accuracy_delta_ci_high"])]
    return {"claim_id": "legacy_matched_selector_attribution", "status": "legacy_reproduced",
            "n_exact_settings": len(paired), "n_training_groups": len(grouped),
            "recomputation": {"mean_test_accuracy": means, "paired_mean_accuracy_delta": float(grouped.mean()),
                              "paired_ci_95": ci.tolist(), "recorded_ci_95": recorded_ci,
                              "maximum_ci_recompute_error": float(np.max(np.abs(ci - recorded_ci))),
                              "bootstrap_draws": 4000, "bootstrap_seed": 9915,
                              "bootstrap_unit": "training-group seed; average widths/counts within group"},
            "per_group": [{"group_id": s, "paired_accuracy_delta": float(np.mean(group_values[s])),
                           "n_dependent_settings": len(group_values[s])} for s in sorted(group_values)],
            "per_setting": paired,
            "provenance": _config(root, f"{SELECTOR}/config.json"),
            "sources": [_artifact(root, f"{SELECTOR}/selectors.csv",
                                  sorted(i for p in paired for i in p["source_rows"])),
                        _artifact(root, f"{SELECTOR}/paired.csv", [recorded["source_data_row"]]),
                        _artifact(root, "experiments/post_iclr_selector_attribution.py")],
            "limitations": ["Pool counts and validation allowance matched; candidate-generation kernels/cost did not.",
                            "Official cores were adapter-assisted; C2M3 source was dirty.",
                            "Historical MNIST selector evidence is separate from the new language study."]}


def _legacy_lora(root: Path) -> dict:
    rows = _csv(root, f"{LORA}/runs.csv")
    primary = [r for r in rows if r["claim_scope"] == "primary_well_conditioned"]
    groups = sorted({int(r["group_seed"]) for r in primary})
    method_values = {}
    for method in ["naive_factor_average", "global_synchronization", "full_delta_svd", "cycle_aware_alignment"]:
        selected = [r for r in primary if r["method"] == method]
        original = {}
        for row in selected:
            group = int(row["group_seed"])
            value = float(row["unscrambled_test_accuracy"])
            if group in original and abs(original[group] - value) > 1e-12:
                raise ValueError(f"Inconsistent historical reference accuracy in {method}/{group}")
            original[group] = value
        method_values[method] = {
            "mean_unscrambled_test_accuracy": float(np.mean(list(original.values()))),
            "per_group_unscrambled_test_accuracy": original,
            "maximum_relative_merged_update_change": max(float(r["relative_delta_change_from_unscrambled"]) for r in selected),
            "maximum_test_logit_change": max(float(r["maximum_test_logit_change"]) for r in selected),
            "maximum_test_prediction_disagreement": max(float(r["test_prediction_disagreement"]) for r in selected),
            "source_data_rows": [r["source_data_row"] for r in selected]}
    cycle = [r for r in primary if r["method"] == "cycle_aware_alignment"]
    decisions = {d: sum(r["decision"] == d for r in cycle) for d in sorted({r["decision"] for r in cycle})}
    cfg = _json(root, f"{LORA}/config.json")
    source_hashes = [{"path": p, "recorded_sha256": h, "current_sha256": sha256(root / p)
                      if (root / p).is_file() else None} for p, h in cfg.get("source_hashes", {}).items()]
    return {"claim_id": "legacy_single_feature_layer_lora", "status": "legacy_reproduced",
            "n_training_groups": len(groups), "group_ids": groups,
            "adapters_per_group": cfg["adapters_per_group"], "adapter_rank": cfg["adapter_rank"],
            "lora_layers": 1, "factor_shapes": {"B": [64, 4], "A": [4, 64]},
            "classification_head_shape": [10, 64], "base_model": "frozen ResNet18 projected CIFAR10 features",
            "n_primary_rows": len(primary), "n_all_rows": len(rows),
            "stage_counts": {s: sum(r["stage"] == s for r in rows) for s in sorted({r["stage"] for r in rows})},
            "recomputation": {"methods": method_values, "cycle_aware_decision_counts": decisions,
                              "full_delta_svd_mean_accuracy_advantage_over_global":
                              method_values["full_delta_svd"]["mean_unscrambled_test_accuracy"]
                              - method_values["global_synchronization"]["mean_unscrambled_test_accuracy"]},
            "provenance": _config(root, f"{LORA}/config.json"),
            "training_commit": cfg["training_commit"], "source_corpus_commit": cfg["source_corpus_commit"],
            "execution_source_hashes": source_hashes,
            "sources": [_artifact(root, f"{LORA}/runs.csv", [r["source_data_row"] for r in primary]),
                        _artifact(root, "reports/practical_twistedmerge/lora_practical_followup/reuse_manifest.csv"),
                        _artifact(root, "experiments/real_lora_gauge_stability.py")],
            "limitations": ["One feature-layer update plus an averaged head; not multilayer PEFT transformers.",
                            "Gauge scrambles are dependent representations, not independent training groups.",
                            "The final CSV combines three pilot groups with two extension groups.",
                            "Ill-conditioned boundary rows excluded from the primary panel but retained in the source CSV."]}


def _legacy_natural(root: Path) -> dict:
    name = f"{NATURAL}/candidate_classifications.csv"
    rows = _csv(root, name)
    positive = [r for r in rows if r["brauer_like_candidate"].lower() == "true"]
    return {"claim_id": "legacy_no_natural_certificate", "status": "legacy_reproduced",
            "n_threshold_rows": len(rows), "n_seed_estimator_combinations":
            len({(r["corpus_seed"], r["transition_method"]) for r in rows}),
            "n_certified_natural_candidates": len(positive),
            "classification_counts": {s: sum(r["classification"] == s for r in rows)
                                      for s in sorted({r["classification"] for r in rows})},
            "execution_commits": sorted({r["execution_commit"] for r in rows}),
            "recorded_command": "<repository-root>/.venv/bin/python experiments/holonomy_application_B.py --mode confirmatory",
            "sources": [_artifact(root, name, list(range(1, len(rows) + 1))),
                        _artifact(root, f"{NATURAL}/config.json"),
                        _artifact(root, f"{NATURAL}/implementation_manifest.md")],
            "limitations": ["Three threshold versions per seed/estimator are dependent checks.",
                            "This reports the tested corpus; absence of certification is not a universal absence theorem."]}


def _legacy_tetra(root: Path) -> dict:
    # Independent exhaustive finite-cochain check; does not treat oracle face
    # values as edge-derived GL transitions.
    faces = list(itertools.combinations(range(4), 3))
    edges = list(itertools.combinations(range(4), 2))
    face_values = [-1, 1, 1, 1]
    witnesses = []
    for values in itertools.product([-1, 1], repeat=6):
        edge = dict(zip(edges, values))
        produced = [edge[(i, j)] * edge[(j, k)] * edge[(i, k)] for i, j, k in faces]
        if produced == face_values:
            witnesses.append(values)
    manifest = _json(root, f"{DIAGNOSTIC}/central_reproduction_manifest.json")
    return {"claim_id": "legacy_tetrahedral_oracle_correction", "status": "legacy_reproduced",
            "ground_truth_construction_type": "oracle_face_assignment", "faces": faces,
            "face_values": face_values, "all_64_mu2_edge_assignments_checked": True,
            "mu2_edge_witness_count": len(witnesses),
            "constant_central_GL_edge_realizable": False,
            "interpretation": "The one-negative-face class on the simply connected tetrahedral boundary is an oracle cochain. Under constant inverse-consistent edge maps with central residuals it is not realizable; finite mu2 enumeration alone only checks mu2-valued edges. The general simply-connected no-go is tested separately.",
            "recorded_command": manifest.get("command"), "execution_commit": manifest.get("git_commit"),
            "sources": [_artifact(root, "src/simplicial_mu2.py"),
                        _artifact(root, "experiments/synthetic_h2_mu2_obstruction.py"),
                        _artifact(root, f"{DIAGNOSTIC}/central_reproduction_manifest.json"),
                        _artifact(root, f"{DIAGNOSTIC}/central_mu2_structural.csv")],
            "limitations": ["Historical oracle prediction results cannot be relabeled as natural or edge-realizable central obstruction evidence."]}


def _checkpoint_inventory(root: Path) -> dict:
    manifests = ["reports/practical_twistedmerge/lora_practical_followup/reuse_manifest.csv",
                 f"{SELECTOR}/checkpoint_manifest.csv"]
    records = []
    hash_cache = {}
    missing_manifests = []
    for manifest in manifests:
        if not (root / manifest).is_file():
            missing_manifests.append(manifest)
            continue
        for row in _csv(root, manifest):
            keys = [k for k in row if k.endswith("_path") or k == "checkpoint"]
            for key in keys:
                raw = row[key]
                path = Path(raw)
                if not path.is_absolute():
                    path = root / path
                hkey = key.removesuffix("_path") + "_sha256"
                expected = row.get(hkey)
                exists = path.is_file()
                if exists and path not in hash_cache:
                    hash_cache[path] = sha256(path)
                actual = hash_cache.get(path)
                # Absolute paths stay in the author inventory; reproducibility
                # export is responsible for scrubbing private/local paths.
                records.append({"manifest_path": manifest, "source_data_row": row["source_data_row"],
                                "kind": key, "local_path": str(path), "exists": exists,
                                "recorded_sha256": expected, "actual_sha256": actual,
                                "hash_matches": actual == expected if exists and expected else None,
                                "bytes": path.stat().st_size if exists else None})
    cache_roots = {Path.home() / ".cache/huggingface/hub"}
    for key in ["HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"]:
        if os.environ.get(key):
            cache_roots.add(Path(os.environ[key]))
    if os.environ.get("HF_HOME"):
        cache_roots.add(Path(os.environ["HF_HOME"]) / "hub")
    snapshots = []
    for cache in sorted(cache_roots):
        if cache.exists():
            for model in sorted(cache.glob("models--*")):
                revisions = sorted(p.name for p in (model / "snapshots").glob("*") if p.is_dir())
                files = [p for p in model.rglob("*") if p.is_file() and p.suffix in [".bin", ".safetensors"]]
                snapshots.append({"model_cache_name": model.name, "local_path": str(model),
                                  "snapshot_revisions": revisions, "weight_file_count": len(files),
                                  "weight_bytes": sum(p.stat().st_size for p in files)})
    return {"checkpoint_deserialization_performed": False, "manifest_files": records,
            "missing_manifests": missing_manifests,
            "n_unique_available_manifest_files": len(hash_cache),
            "huggingface_cache_roots": [str(p) for p in sorted(cache_roots)],
            "huggingface_model_caches": snapshots,
            "language_model_cache_available": any(s["weight_file_count"] for s in snapshots),
            "search_boundary": "Manifest paths plus default/environment-declared Hugging Face hub caches; arbitrary filesystem model stores are not exhaustively searched.",
            "additional_legacy_language_artifact": _artifact(root, "reports/future_program/extended/broader_language_collections.csv"),
            "legacy_language_limitation": "Five prior tiny-BERT collections on two bases have a negative structured gate; they are not a compatible four-task 1-3B primary cohort."}


def _fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "TwistedMerge-read-only-evidence-audit"})
    with urllib.request.urlopen(request, timeout=15) as response:
        data = response.read(4 * 1024 * 1024 + 1)
    if len(data) > 4 * 1024 * 1024:
        raise ValueError("Metadata/source exceeds 4 MiB audit limit")
    return data


def _source_file(repo: str, commit: str, path: str) -> dict:
    url = f"https://raw.githubusercontent.com/{repo}/{commit}/{path}"
    try:
        data = _fetch(url)
        return {"path": path, "url": url, "accessible": True,
                "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    except Exception as error:
        return {"path": path, "url": url, "accessible": False,
                "error": f"{type(error).__name__}: {error}"}


def _external_repo(repo: str) -> dict:
    url = f"https://api.github.com/repos/{repo}"
    try:
        details = json.loads(_fetch(url))
        commit_api = url + "/commits?per_page=1"
        commit = json.loads(_fetch(commit_api))[0]["sha"]
        tree = json.loads(_fetch(url + f"/git/trees/{commit}?recursive=1"))
        paths = [r["path"] for r in tree["tree"]]
        files = {"gstoica27/KnOTS": ["LICENSE", "configs/llama8B_r16_knots_ties.py", "training_scripts/nli_training.py", "task_merger.py"],
                 "apanariello4/core-space-merging": ["LICENSE", "configs/llama8B_r16_tv.py", "task_merger.py"],
                 "ContinuumCoder/Higher-Order-Model-Merge": ["README.md", "src/topology/hodge.py"],
                 "huggingface/peft": ["LICENSE", "src/peft/tuners/lora/model.py"]}[repo]
        return {"repository": repo, "url": details["html_url"], "api_url": url,
                "api_accessible": True, "commit": commit, "commit_api_url": commit_api,
                "license": details.get("license", {}).get("spdx_id") if details.get("license") else None,
                "configuration_paths": [p for p in paths if p.startswith("configs/")],
                "reviewed_source_files": [_source_file(repo, commit, path) for path in files],
                "integration_status": "not_integrated", "remote_code_executed": False,
                "limitation": "Source/configuration review only; no official execution or parity-tested adaptation is claimed."}
    except Exception as error:
        return {"repository": repo, "api_url": url, "api_accessible": False,
                "integration_status": "blocked", "error": f"{type(error).__name__}: {error}"}


def _external_model(model: str) -> dict:
    url = "https://huggingface.co/api/models/" + model
    try:
        data = json.loads(_fetch(url))
        result = {"model_id": model, "url": "https://huggingface.co/" + model,
                  "api_url": url, "api_accessible": True, "revision": data["sha"],
                  "gated": data.get("gated"), "private": data.get("private"),
                  "license": data.get("cardData", {}).get("license"),
                  "available_files": [s["rfilename"] for s in data.get("siblings", [])],
                  "weights_downloaded": False, "remote_code_executed": False}
        cfg_name = "adapter_config.json" if "hoffman-lab/" in model else "config.json" if "Qwen/" in model else None
        if cfg_name:
            cfg_url = f"https://huggingface.co/{model}/resolve/{data['sha']}/{cfg_name}"
            raw = _fetch(cfg_url)
            result.update(config_url=cfg_url, config_sha256=hashlib.sha256(raw).hexdigest(),
                          config=json.loads(raw))
        if "hoffman-lab/" in model:
            result["limitation"] = "One released cohort, no independent training-seed provenance; adapter card license unspecified; only pickle adapter weights listed; classifier/head/tokenizer conventions require explicit parity review."
        return result
    except Exception as error:
        return {"model_id": model, "api_url": url, "api_accessible": False,
                "error": f"{type(error).__name__}: {error}"}


def external_inventory() -> dict:
    repos = ["gstoica27/KnOTS", "apanariello4/core-space-merging",
             "ContinuumCoder/Higher-Order-Model-Merge", "huggingface/peft"]
    models = ["meta-llama/Meta-Llama-3-8B", "Qwen/Qwen2.5-1.5B"] + [
        "hoffman-lab/KnOTS-Llama3_8B_lora_R16_" + task
        for task in ["snli", "mnli", "sick", "qnli", "rte", "scitail"]]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        repository_records = list(executor.map(_external_repo, repos))
        model_records = list(executor.map(_external_model, models))
    return {"checked_utc": datetime.now(timezone.utc).isoformat(),
            "access_mode": "Unauthenticated read-only metadata and source text; no model downloads or execution",
            "repositories": repository_records, "models": model_records,
            "official_language_setup": {"base_model": "meta-llama/Meta-Llama-3-8B", "rank": 16,
                "lora_alpha": 16, "lora_dropout": 0.1, "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
                "tasks": ["snli", "mnli", "sick", "qnli", "rte", "scitail"],
                "core_space_language_config_default": "full-space task vectors, not automatically core-space TIES",
                "status": "blocked", "reason": "Base license/access and adapter licenses/head provenance unresolved; no compatible local cached cohort or official parity run."},
            "literature_sources": [
                {"title": "KnOTS ICLR 2025", "url": "https://proceedings.iclr.cc/paper_files/paper/2025/hash/0d4f8a5109c5083b5307fcd0bddae7a7-Abstract-Conference.html", "browser_access": "accessible"},
                {"title": "Core Space NeurIPS 2025, arXiv v4", "url": "https://arxiv.org/abs/2509.17786v4", "browser_access": "accessible"},
                {"title": "PEFT merging documentation", "url": "https://huggingface.co/blog/peft_merging", "browser_access": "accessible", "operator_warning": "Factorwise linear/TIES names do not imply effective-update Task Arithmetic/TIES parity."},
                {"title": "Beyond Pairwise Hodge", "url": "https://openreview.net/forum?id=5IsDyRzKks", "browser_access": "challenge_blocked; repository README/source accessible"},
                {"title": "Learn then Test", "url": "https://arxiv.org/abs/2110.01052", "browser_access": "accessible"},
                {"title": "KBS author guide", "url": "https://www.sciencedirect.com/journal/knowledge-based-systems/publish/guide-for-authors", "browser_access": "blocked_internal_error"}]}


def inventory(repo_root: str | Path, *, include_external: bool = False) -> dict:
    """Return reusable evidence, exact scalar reanalyses, and explicit blockers."""
    root = Path(repo_root).resolve()
    historical = []
    for function in [_legacy_lora, _legacy_diagnostics, _legacy_selector, _legacy_natural, _legacy_tetra]:
        try:
            historical.append(function(root))
        except Exception as error:
            historical.append({"claim_id": function.__name__, "status": "blocked",
                               "failure_reason": f"{type(error).__name__}: {error}"})
    older = "7a0620bb19dffba97012350b6ffd20684bcbe220"
    later = "b0e1ac47e4b0d7c559f37ba8bc3634ba18f3ca1c"
    ancestry = subprocess.run(["git", "merge-base", "--is-ancestor", older, later], cwd=root,
                              capture_output=True).returncode
    required = ["README.md", "reports/final_claim_ledger.md", "reports/claims_audit.md", "reports/full_capacity_claim_audit.md"]
    extra = ["reports/post_iclr_experiment_gap_audit.md", "reports/post_iclr_v2/selector_attribution/report.md",
             "reports/practical_twistedmerge/lora_practical_followup/final_assessment.md",
             "reports/future_program/extended/broader_language_report.md", "reports/holonomy_applications/final_assessment.md"]
    result = {"schema_version": 1, "status": "legacy_reproduced" if all(
        h["status"] == "legacy_reproduced" for h in historical) else "blocked",
              "checked_utc": datetime.now(timezone.utc).isoformat(),
              "repo_root": str(root), "head_commit": _git(root, "rev-parse", "HEAD"),
              "historical": historical, "required_audit_sources": [_artifact(root, p) for p in required + extra],
              "provenance_resolution": {
                  "older_commit": older, "older_subject": _git(root, "show", "-s", "--format=%s", older),
                  "older_role": "Spatial-output biomedical benchmark snapshot; documented starting baseline of post-ICLR gap audit, before later selector and LoRA studies.",
                  "later_commit": later, "later_subject": _git(root, "show", "-s", "--format=%s", later),
                  "later_role": "Later merge snapshot including model-lineage and preceding selector/LoRA evidence; repository HEAD at task start.",
                  "older_is_ancestor_of_later": ancestry == 0 if ancestry in (0, 1) else None,
                  "conclusion": "Distinct snapshot roles; neither is the execution commit for every historical experiment. Link numbers to their experiment-specific config and artifacts.",
                  "bibliography_limitation": "No manuscript or bibliography is present/edited in this audit; its citation mismatch is resolved by repository chronology, not by inspecting manuscript bytes."},
              "checkpoints": _checkpoint_inventory(root)}
    result["external"] = external_inventory() if include_external else {
        "status": "not_checked", "reason": "Read-only external API audit disabled for this invocation."}
    result["items"] = [{"item": h["claim_id"], "existing_artifact": h.get("sources", []),
                        "provenance_complete": h.get("provenance", {}).get("execution_provenance_complete", False),
                        "can_reuse": h["status"] == "legacy_reproduced", "action_needed":
                        "Use only as historical evidence; preserve source hashes and listed provenance limitations."}
                       for h in historical]
    result["baseline_integration"] = [
        {"method_id": method, "status": "not_integrated", "source_url": url, "fidelity": "no official neural result or parity-tested adaptation",
         "action_needed": action}
        for method, url, action in [
            ("KnOTS", "https://github.com/gstoica27/KnOTS", "Run a reviewed pinned implementation on compatible licensed checkpoints and verify adapter scaling/head parity."),
            ("Core_Space", "https://github.com/apanariello4/core-space-merging", "Define and parity-test the actual core-space operator; compact weighted-sum QR/SVD does not reproduce nonlinear core-space TIES."),
            ("official_Hodge", "https://github.com/ContinuumCoder/Higher-Order-Model-Merge", "Resolve absent repository license and task/setup mismatch; keep degree-1 scalar barrier analysis distinct from degree-2 central residuals.")]]
    return result
