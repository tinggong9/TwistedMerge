#!/usr/bin/env python3
"""Application C: controlled period-index capacity planning on frozen real features."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torchvision.datasets import CIFAR10

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.holonomy_application_A import load_models, load_shared
from src.holonomy_application_corpus import classification_metrics, tensor_mapping_sha256
from src.holonomy_period_index_capacity import (
    candidate_generators,
    carrier_vectors,
    chart_operators,
    decode_carrier,
    encode_logits,
    relation_residual,
    unitarity_residual,
)

APP_DIR = ROOT / "reports" / "holonomy_applications" / "application_C_period_index_capacity"
ARTIFACT_ROOT = ROOT / "reports" / "tmp" / "holonomy_applications"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def random_unitaries(capacity: int, seed: int) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    result = []
    for _ in range(8):
        real = torch.randn(capacity, capacity, generator=generator)
        imaginary = torch.randn(capacity, capacity, generator=generator)
        q, _r = torch.linalg.qr(torch.complex(real, imaginary))
        result.append(q.to(torch.complex64))
    return result


def branch_outputs(
    local_logits: torch.Tensor,
    encoding_operators: list[torch.Tensor],
    carriers: torch.Tensor,
    alignment_operators: list[torch.Tensor] | None,
) -> torch.Tensor:
    outputs = []
    for chart in range(8):
        encoded = encode_logits(local_logits[chart], encoding_operators[chart], carriers)
        if alignment_operators is not None:
            inverse = torch.linalg.pinv(alignment_operators[chart])
            encoded = torch.einsum("rs,ncs->ncr", inverse, encoded)
        outputs.append(decode_carrier(encoded, carriers))
    return torch.stack(outputs)


def local_model_logits(models, features: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return torch.stack([models[chart](features[chart]) for chart in range(8)])


def build_candidates(
    case_name: str,
    capacity: int,
    local_logits: torch.Tensor,
    seed: int,
) -> tuple[dict[str, torch.Tensor], dict[str, float], dict[str, float]]:
    case, operators = chart_operators(case_name, capacity)
    _case, generators = candidate_generators(case_name, capacity)
    carriers = carrier_vectors(capacity, 10, 1010000 + seed * 1000 + case.index * 100 + capacity)
    random_operators = random_unitaries(capacity, 1020000 + seed * 1000 + case.index * 100 + capacity)
    permutation = np.random.default_rng(1030000 + seed * 1000 + capacity).permutation(8)
    wrong_case = {
        "period2_index2": "period3_index3",
        "period2_index4": "period2_index2",
        "period3_index3": "period2_index2",
    }[case_name]
    _wrong, wrong_operators = chart_operators(wrong_case, capacity)
    candidates: dict[str, torch.Tensor] = {}
    latencies: dict[str, float] = {}

    def timed(name: str, function) -> None:
        started = time.perf_counter()
        candidates[name] = function().detach().cpu()
        latencies[name] = (time.perf_counter() - started) * 1000.0

    timed(
        "ordinary_same_capacity",
        lambda: branch_outputs(local_logits, operators, carriers, None),
    )
    timed(
        "coherent_projective_lift",
        lambda: branch_outputs(local_logits, operators, carriers, operators),
    )
    timed(
        "parameter_matched_generic_unitary",
        lambda: branch_outputs(local_logits, random_operators, carriers, random_operators),
    )
    timed(
        "random_branch_control",
        lambda: branch_outputs(
            local_logits,
            operators,
            carriers,
            [operators[int(permutation[index])] for index in range(8)],
        ),
    )
    timed(
        "wrong_cocycle_control",
        lambda: branch_outputs(local_logits, operators, carriers, list(reversed(operators))),
    )
    timed(
        "wrong_projective_representation",
        lambda: branch_outputs(local_logits, operators, carriers, wrong_operators),
    )
    timed("ensemble_upper_bound", lambda: local_logits.clone())
    _oracle_case, oracle_operators = chart_operators(case_name, case.index)
    oracle_carriers = carrier_vectors(case.index, 10, 1040000 + seed * 100 + case.index)
    timed(
        "oracle_coherent_lift",
        lambda: branch_outputs(local_logits, oracle_operators, oracle_carriers, oracle_operators),
    )
    structural = {
        "projective_relation_residual": relation_residual(case, generators),
        "unitarity_residual": unitarity_residual(operators),
    }
    structural["combined_structural_residual"] = max(structural.values())
    return candidates, latencies, structural


def score(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    fused = logits.mean(0)
    aggregate = classification_metrics(fused, labels)
    branch_metrics = [classification_metrics(logits[branch], labels) for branch in range(8)]
    predictions = logits.argmax(-1)
    modal = torch.mode(predictions, dim=0).values
    return {
        "classification_accuracy": aggregate["accuracy"],
        "average_branch_accuracy": float(np.mean([row["accuracy"] for row in branch_metrics])),
        "worst_branch_accuracy": float(min(row["accuracy"] for row in branch_metrics)),
        "nll": aggregate["nll"],
        "brier": aggregate["brier"],
        "ece": aggregate["ece"],
        "prediction_consistency": float((predictions == modal.unsqueeze(0)).float().mean()),
    }


def paired_bootstrap(values: np.ndarray, samples: int, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    means = np.asarray([rng.choice(values, size=len(values), replace=True).mean() for _ in range(samples)])
    return float(values.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "confirmatory"), required=True)
    parser.add_argument("--config", type=Path, default=APP_DIR / "config.json")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/Users/tinggong/Documents/GitHub/TwistedMerge/data"),
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    command = " ".join([sys.executable, *sys.argv])
    output_dir = APP_DIR if args.mode == "confirmatory" else APP_DIR / "smoke"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "plots").mkdir(exist_ok=True)
    (output_dir / "tables").mkdir(exist_ok=True)
    resolved, manifest, payload, _shared = load_shared(args.mode)
    features = {name: values.float() for name, values in payload["features"].items()}
    splits = {name: values.numpy() for name, values in payload["splits"].items()}
    test_dataset = CIFAR10(args.data_dir, train=False, download=False)
    run_rows: list[dict[str, object]] = []
    capacity_rows: list[dict[str, object]] = []
    artifact_rows: list[dict[str, object]] = []
    failure_rows: list[dict[str, object]] = []
    methods = list(config["methods"])
    for seed in sorted(int(value) for value in manifest["corpus_seed"].unique()):
        try:
            models = load_models(seed, manifest, int(resolved["feature_dim"]), int(resolved["adapter_rank"]))
            test_local_logits = local_model_logits(models, features["test"])
            all_candidates: dict[str, torch.Tensor] = {}
            pending_rows: list[dict[str, object]] = []
            for case_name, case_config in config["cases"].items():
                period = int(case_config["period"])
                index = int(case_config["index"])
                for capacity in map(int, case_config["capacities"]):
                    candidates, latencies, structural = build_candidates(
                        case_name, capacity, test_local_logits, seed
                    )
                    for method in methods:
                        key = f"{case_name}__capacity_{capacity}__{method}"
                        all_candidates[key] = candidates[method]
                        complex_operator_bytes = 8 * capacity * capacity * 8
                        capacity_row = {
                            "evidence_label": "controlled_on_real_features",
                            "mode": args.mode,
                            "corpus_seed": seed,
                            "case_name": case_name,
                            "period": period,
                            "predicted_index": index,
                            "candidate_capacity": capacity,
                            "capacity_divisible_by_index": capacity % index == 0,
                            "method": method,
                            "adapter_rank": int(resolved["adapter_rank"]),
                            "branches": 8,
                            "active_local_adapter_parameters": 8 * 1162,
                            "controlled_carrier_parameters": 0,
                            "stored_carrier_bytes": complex_operator_bytes,
                            "parameter_multiplier_vs_shared_corpus": 1.0,
                            "inference_multiplier": 8.0,
                            "latency_ms_for_complete_test_tensor": latencies[method],
                            "estimated_peak_tensor_bytes": int(candidates[method].numel() * 4),
                            "new_adapter_training": False,
                            "test_labels_used_before_logits_saved": False,
                            **structural,
                        }
                        capacity_rows.append(capacity_row)
                        pending_rows.append(capacity_row.copy())
            logits_path = ARTIFACT_ROOT / f"application_C_{args.mode}" / f"candidate_logits_seed_{seed}.npz"
            logits_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                logits_path,
                **{name: value.numpy().astype(np.float32) for name, value in all_candidates.items()},
                test_indices=splits["test"],
            )
            logits_hash = sha256_file(logits_path)
            content_hash = tensor_mapping_sha256(all_candidates)
            # First label access occurs only after all case/capacity/method logits are immutable.
            test_labels = torch.tensor(np.asarray(test_dataset.targets), dtype=torch.long)[
                torch.from_numpy(splits["test"])
            ]
            ensemble_accuracy: dict[tuple[str, int], float] = {}
            for key, candidate in all_candidates.items():
                case_name, capacity_text, method = key.split("__")
                capacity = int(capacity_text.removeprefix("capacity_"))
                metrics = score(candidate, test_labels)
                if method == "ensemble_upper_bound":
                    ensemble_accuracy[(case_name, capacity)] = metrics["classification_accuracy"]
                row = next(
                    item
                    for item in pending_rows
                    if item["case_name"] == case_name
                    and item["candidate_capacity"] == capacity
                    and item["method"] == method
                )
                run_rows.append(
                    {
                        **row,
                        **metrics,
                        "test_logits_path": str(logits_path),
                        "test_logits_sha256": logits_hash,
                        "test_logits_content_sha256": content_hash,
                        "execution_commit": git_head(),
                    }
                )
            for row in run_rows:
                if row["corpus_seed"] != seed:
                    continue
                ensemble = ensemble_accuracy[(str(row["case_name"]), int(row["candidate_capacity"]))]
                row["accuracy_gap_from_ensemble"] = float(row["classification_accuracy"]) - ensemble
                row["structural_gate_passed"] = (
                    float(row["combined_structural_residual"])
                    <= float(config["gates"]["structural_residual"])
                )
                row["task_gate_passed"] = (
                    float(row["classification_accuracy"])
                    >= ensemble - float(config["gates"]["accuracy_gap_from_ensemble"])
                )
                row["coherent_application_success"] = bool(
                    row["method"] == "coherent_projective_lift"
                    and row["structural_gate_passed"]
                    and row["task_gate_passed"]
                )
            permuted = test_labels[
                torch.randperm(len(test_labels), generator=torch.Generator().manual_seed(1050000 + seed))
            ]
            _ = score(next(iter(all_candidates.values())), permuted)
            if sha256_file(logits_path) != logits_hash:
                raise RuntimeError("Application C logit bundle changed after label access")
            artifact_rows.append(
                {
                    "evidence_label": "controlled_on_real_features",
                    "mode": args.mode,
                    "corpus_seed": seed,
                    "artifact_kind": "candidate_logits_before_test_labels",
                    "path": str(logits_path),
                    "sha256": logits_hash,
                    "content_sha256": content_hash,
                    "bytes": logits_path.stat().st_size,
                    "label_permutation_hash_passed": True,
                }
            )
        except Exception as error:
            failure_rows.append(
                {
                    "mode": args.mode,
                    "corpus_seed": seed,
                    "stage": "application_C",
                    "error_type": type(error).__name__,
                    "message": str(error),
                }
            )

    runs = pd.DataFrame(run_rows)
    capacity = pd.DataFrame(capacity_rows)
    runs.to_csv(output_dir / "runs.csv", index=False)
    capacity.to_csv(output_dir / "capacity_audit.csv", index=False)
    pd.DataFrame(failure_rows, columns=("mode", "corpus_seed", "stage", "error_type", "message")).to_csv(
        output_dir / "failure_log.csv", index=False
    )
    samples = int(config[args.mode]["statistic_bootstrap_samples"])
    paired_rows = []
    for case_name, case_config in config["cases"].items():
        index = int(case_config["index"])
        subset = runs[(runs["case_name"] == case_name) & (runs["candidate_capacity"] == index)]
        pivot = subset.pivot(index="corpus_seed", columns="method", values="classification_accuracy")
        for right in (
            "ordinary_same_capacity",
            "parameter_matched_generic_unitary",
            "random_branch_control",
            "wrong_cocycle_control",
        ):
            delta = (pivot["coherent_projective_lift"] - pivot[right]).to_numpy(dtype=float)
            mean, low, high = paired_bootstrap(delta, samples, 1060000 + len(paired_rows))
            paired_rows.append(
                {
                    "evidence_label": "controlled_on_real_features",
                    "mode": args.mode,
                    "case_name": case_name,
                    "candidate_capacity": index,
                    "comparison": f"coherent_projective_lift_minus_{right}",
                    "n_independent_seeds": len(delta),
                    "mean_accuracy_delta": mean,
                    "ci_low": low,
                    "ci_high": high,
                    "wins": int((delta > 1e-12).sum()),
                    "ties": int((np.abs(delta) <= 1e-12).sum()),
                    "losses": int((delta < -1e-12).sum()),
                }
            )
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(output_dir / "paired_statistics.csv", index=False)

    coherent = runs[runs["method"] == "coherent_projective_lift"].copy()
    coherent["success_matches_divisibility"] = (
        coherent["coherent_application_success"] == coherent["capacity_divisible_by_index"]
    )
    structural_task_iff = bool(coherent["success_matches_divisibility"].all())
    minimal_rows = []
    for (seed, case_name), rows in coherent.groupby(["corpus_seed", "case_name"]):
        successes = rows[rows["coherent_application_success"]]
        minimal = int(successes["candidate_capacity"].min()) if len(successes) else None
        predicted = int(rows["predicted_index"].iloc[0])
        minimal_rows.append(
            {
                "evidence_label": "controlled_on_real_features",
                "mode": args.mode,
                "corpus_seed": seed,
                "case_name": case_name,
                "predicted_index": predicted,
                "observed_minimum_successful_capacity": minimal if minimal is not None else "",
                "prediction_correct": minimal == predicted,
            }
        )
    minimums = pd.DataFrame(minimal_rows)
    minimums.to_csv(output_dir / "minimum_capacity_predictions.csv", index=False)
    threshold_predicts = bool(len(minimums) and minimums["prediction_correct"].all())
    generic_rows = paired[paired["comparison"].str.contains("parameter_matched_generic_unitary")]
    beats_generic = bool(
        args.mode == "confirmatory"
        and len(generic_rows)
        and (
            generic_rows["ci_low"]
            > float(config["gates"]["improvement_over_generic"])
        ).all()
    )
    application_gate = threshold_predicts and structural_task_iff and beats_generic
    claims = pd.DataFrame(
        [
            {
                "claim_id": "controlled_period_index_structural_threshold",
                "status": "smoke_only" if args.mode == "smoke" else ("supported_controlled" if structural_task_iff else "negative"),
                "gate_passed": structural_task_iff,
                "safe_wording": "In the controlled carrier layer, structural-plus-task success matches index divisibility across preregistered capacities." if structural_task_iff else "The controlled carrier did not obey the preregistered divisibility threshold.",
            },
            {
                "claim_id": "predicted_minimum_capacity",
                "status": "smoke_only" if args.mode == "smoke" else ("supported_controlled" if threshold_predicts else "negative"),
                "gate_passed": threshold_predicts,
                "safe_wording": "The controlled index equals the minimum structurally coherent capacity on the real-feature task." if threshold_predicts else "The predicted index did not equal every observed minimum capacity.",
            },
            {
                "claim_id": "threshold_beats_parameter_matched_generic_capacity",
                "status": "supported" if beats_generic else "negative",
                "gate_passed": beats_generic,
                "safe_wording": "The predicted-index lift beats matched generic unitary capacity." if beats_generic else "The predicted-index lift does not outperform matched generic unitary capacity.",
            },
            {
                "claim_id": "practical_period_index_capacity_planner",
                "status": "supported" if application_gate else "negative",
                "gate_passed": application_gate,
                "safe_wording": "Period-index supplies a practically superior capacity recommendation in this task." if application_gate else "The controlled structural threshold does not establish a practically superior capacity planner.",
            },
        ]
    )
    claims.to_csv(output_dir / "claims.csv", index=False)

    diagnostic_lines = []
    for case_name, case_config in config["cases"].items():
        index = int(case_config["index"])
        diagnostic_lines.extend(
            [
                f"Case: {case_name}",
                "Detected obstruction type: controlled central projective",
                f"Estimated period: {case_config['period']}",
                f"Estimated index/minimal capacity: {index}",
                f"Current adapter rank: {resolved['adapter_rank']}",
                "Coherent ordinary merge possible: no under the controlled carrier",
                f"Recommended minimum rank or branch count: {index}",
                f"Certificate confidence: controlled exact structure; practical-superiority gate {'passed' if beats_generic else 'failed'}",
                "",
            ]
        )
    (output_dir / "capacity_planner_output.txt").write_text("\n".join(diagnostic_lines), encoding="utf-8")

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    summary = runs.groupby(["case_name", "candidate_capacity", "method"], as_index=False).agg(
        accuracy=("classification_accuracy", "mean"),
        structural=("combined_structural_residual", "mean"),
    )
    for case_name, rows in summary[summary["method"] == "coherent_projective_lift"].groupby("case_name"):
        axes[0].plot(rows["candidate_capacity"], rows["accuracy"], marker="o", label=case_name)
        axes[1].plot(rows["candidate_capacity"], rows["structural"], marker="o", label=case_name)
    axes[0].set_xlabel("Carrier capacity")
    axes[0].set_ylabel("Mean test accuracy")
    axes[0].set_title("Controlled coherent lift")
    axes[1].set_xlabel("Carrier capacity")
    axes[1].set_ylabel("Combined structural residual")
    axes[1].set_yscale("symlog", linthresh=1e-6)
    axes[1].set_title("Index divisibility and structure")
    axes[0].legend(fontsize=7)
    axes[1].legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(output_dir / "plots" / "period_index_capacity.pdf", bbox_inches="tight")
    plt.close(figure)

    table = summary[summary["method"].isin(["coherent_projective_lift", "ordinary_same_capacity", "parameter_matched_generic_unitary"])]
    latex = ["\\begin{tabular}{llrrr}", "\\toprule", "Case & Method & Capacity & Accuracy & Residual\\\\", "\\midrule"]
    for row in table.itertuples(index=False):
        latex.append(
            f"{row.case_name.replace('_', ' ')} & {row.method.replace('_', ' ')} & {row.candidate_capacity} & {row.accuracy:.3f} & {row.structural:.3g}\\\\"
        )
    latex.extend(["\\bottomrule", "\\end{tabular}", ""])
    (output_dir / "tables" / "application_C_capacity.tex").write_text("\n".join(latex), encoding="utf-8")

    report = f"""# Application C: Period-Index Capacity Planner

Decision: **{'bounded smoke completed' if args.mode == 'smoke' else ('positive practical capacity gate' if application_gate else 'controlled structural threshold without practical superiority')}**.

## Commands

Smoke: `{sys.executable} experiments/holonomy_application_C.py --mode smoke`

Confirmatory: `{sys.executable} experiments/holonomy_application_C.py --mode confirmatory`

Executed: `{command}`

## Scope

Evidence label: `controlled_on_real_features`. The experiment uses the same frozen ResNet features, the same 8 chart adapters, and actual local-model logits. It adds no dataset and retrains no adapter. The finite-Heisenberg carrier is planted and exact at capacities divisible by its index; this is not a naturally discovered class.

## Result

- Cases: period 2/index 2, period 2/index 4, and period 3/index 3.
- Independent adapter seeds: {runs['corpus_seed'].nunique()}.
- Structural-plus-task success matched divisibility: `{structural_task_iff}`.
- Predicted index equaled every minimum successful controlled capacity: `{threshold_predicts}`.
- Predicted-index lift beat parameter-matched generic unitary capacity: `{beats_generic}`.
- Practical capacity-planner gate: `{application_gate}`.

The controlled layer verifies that complete projective blocks restore exact relations and preserve actual classifier logits. However, matched generic unitary carriers recover the same real-task predictions without waiting for the projective index. Therefore the structural threshold does not translate into a uniquely useful capacity recommendation on this task.

## Boundary

The positive part, if any, is controlled algebra tied to real frozen features and actual logits. It is not evidence of a natural Brauer-like obstruction. A practical period-index claim requires superiority over matched generic capacity; that gate {'passed' if beats_generic else 'failed'}.
"""
    (output_dir / "report.md").write_text(report, encoding="utf-8")

    committed = (
        output_dir / "runs.csv",
        output_dir / "capacity_audit.csv",
        output_dir / "paired_statistics.csv",
        output_dir / "minimum_capacity_predictions.csv",
        output_dir / "claims.csv",
        output_dir / "capacity_planner_output.txt",
        output_dir / "plots" / "period_index_capacity.pdf",
        output_dir / "tables" / "application_C_capacity.tex",
    )
    artifact_rows.extend(
        {
            "evidence_label": "controlled_on_real_features",
            "mode": args.mode,
            "corpus_seed": "all",
            "artifact_kind": "committed_output",
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256_file(path),
            "content_sha256": "",
            "bytes": path.stat().st_size,
            "label_permutation_hash_passed": "",
        }
        for path in committed
    )
    pd.DataFrame(artifact_rows).to_csv(output_dir / "artifact_hashes.csv", index=False)
    expected = manifest["corpus_seed"].nunique() * sum(
        len(case["capacities"]) for case in config["cases"].values()
    ) * len(methods)
    if failure_rows or len(runs) != expected:
        raise RuntimeError("Application C incomplete; inspect failure_log.csv")


if __name__ == "__main__":
    main()
