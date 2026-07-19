#!/usr/bin/env python3
"""Application D: grouped holonomy-aware mergeability linter."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.holonomy_application_A import load_models, load_shared
from src.holonomy_mergeability_linter import (
    coverage_risk_rows,
    double_holdout_predictions,
    linter_metrics,
    reliability_rows,
)

APP_DIR = ROOT / "reports" / "holonomy_applications" / "application_D_mergeability_linter"
HOLONOMY_ROOT = ROOT / "reports" / "holonomy_applications"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def phase_dir(name: str, mode: str) -> Path:
    base = HOLONOMY_ROOT / name
    return base if mode == "confirmatory" else base / "smoke"


def parameter_distances(mode: str) -> dict[int, float]:
    resolved, manifest, _payload, _shared = load_shared(mode)
    result = {}
    for seed in sorted(int(value) for value in manifest["corpus_seed"].unique()):
        models = load_models(seed, manifest, int(resolved["feature_dim"]), int(resolved["adapter_rank"]))
        vectors = []
        for model in models:
            vector = np.concatenate(
                [
                    model.effective_adapter().detach().numpy().reshape(-1),
                    model.head.weight.detach().numpy().reshape(-1),
                    model.head.bias.detach().numpy().reshape(-1),
                ]
            )
            vectors.append(vector)
        values = []
        for left, right in combinations(vectors, 2):
            denominator = max((np.linalg.norm(left) + np.linalg.norm(right)) / 2.0, 1e-12)
            values.append(float(np.linalg.norm(left - right) / denominator))
        result[seed] = float(np.mean(values))
    return result


def seed_diagnostics(mode: str) -> dict[int, dict[str, float]]:
    a_dir = phase_dir("application_A_holonomy", mode)
    b_dir = phase_dir("application_B_brauer_certificate", mode)
    a_runs = pd.read_csv(a_dir / "runs.csv")
    transitions = pd.read_csv(a_dir / "transitions.csv")
    loops = pd.read_csv(a_dir / "holonomy_loops.csv")
    certificates = pd.read_csv(b_dir / "candidate_classifications.csv")
    distances = parameter_distances(mode)
    result = {}
    for seed in sorted(int(value) for value in a_runs["corpus_seed"].unique()):
        selected_method = str(
            a_runs[a_runs["corpus_seed"] == seed]["selected_transition_method"].iloc[0]
        )
        transition = transitions[
            (transitions["corpus_seed"] == seed)
            & (transitions["transition_method"] == selected_method)
        ]
        loop = loops[
            (loops["corpus_seed"] == seed)
            & (loops["transition_method"] == selected_method)
            & (loops["selected_transition_method"] == True)
        ]
        ordinary_loops = loop[loop["loop_name"] != "rotation_reflection_commutator"]
        commutator = loop[loop["loop_name"] == "rotation_reflection_commutator"]
        certificate = certificates[
            (certificates["corpus_seed"] == seed)
            & (certificates["transition_method"] == selected_method)
            & (certificates["threshold_level"] == "medium")
        ].iloc[0]
        instability = max(
            float(certificate["max_bootstrap_centrality_ci_high"]),
            float(certificate["max_bootstrap_phase_instability"]),
        )
        root_residual = float(certificate["max_root_residual"])
        root_margin = float(certificate["minimum_root_margin"])
        result[seed] = {
            "mean_heldout_pairwise_residual": float(transition["heldout_overlap_residual"].mean()),
            "mean_inverse_consistency_residual": float(transition["inverse_consistency_residual"].mean()),
            "global_synchronization_residual": float(
                ordinary_loops["connection_synchronization_residual"].mean()
            ),
            "mean_parameter_distance": distances[seed],
            "max_loop_holonomy_norm": float(ordinary_loops["identity_distance"].max()),
            "loop_commutator_norm": float(commutator["identity_distance"].mean()),
            "mean_loop_spectral_radius": float(ordinary_loops["spectral_radius"].mean()),
            "mean_loop_absolute_phase": float(ordinary_loops["mean_absolute_eigenphase"].mean()),
            "centrality_residual": float(certificate["max_centrality_residual"]),
            "torsion_confidence": float(root_margin / max(root_margin + root_residual, 1e-12)),
            "cocycle_residual": float(certificate["max_cocycle_residual"]),
            "coboundary_residual": float(certificate["coboundary_residual"]),
            "bootstrap_stability": 1.0 - min(instability, 1.0),
            "natural_connection_trivial_indicator": float(
                certificate["classification"] == "trivial_coboundary"
            ),
        }
    return result


def assemble_dataset(mode: str, config: dict[str, object]) -> pd.DataFrame:
    a_dir = phase_dir("application_A_holonomy", mode)
    c_dir = phase_dir("application_C_period_index_capacity", mode)
    a_runs = pd.read_csv(a_dir / "runs.csv")
    c_runs = pd.read_csv(c_dir / "runs.csv")
    diagnostics = seed_diagnostics(mode)
    rows: list[dict[str, object]] = []
    harm_margin = float(config["harm_margin"])
    benefit_margin = float(config["benefit_margin"])
    observation = 0
    for seed, seed_runs in a_runs.groupby("corpus_seed"):
        seed = int(seed)
        reference = float(
            seed_runs[seed_runs["method"] == "prediction_ensemble_upper_bound"][
                "ordinary_test_accuracy"
            ].iloc[0]
        )
        global_accuracy = float(
            seed_runs[seed_runs["method"] == "global_c2m3_synchronization"][
                "ordinary_test_accuracy"
            ].iloc[0]
        )
        orbit_accuracy = float(
            seed_runs[seed_runs["method"] == "orbit_branch_invariant_pooling"][
                "ordinary_test_accuracy"
            ].iloc[0]
        )
        raw_accuracy = float(
            seed_runs[seed_runs["method"] == "raw_parameter_average"][
                "ordinary_test_accuracy"
            ].iloc[0]
        )
        for run in seed_runs.itertuples(index=False):
            harmful = float(run.ordinary_test_accuracy) < reference - harm_margin
            rows.append(
                {
                    "observation_id": f"D{observation:04d}",
                    "evidence_label": "natural_measured",
                    "corpus_seed": seed,
                    "setting_family": "natural_application_A",
                    "fusion_method": run.method,
                    "reference_accuracy": reference,
                    "observed_accuracy": float(run.ordinary_test_accuracy),
                    "ordinary_fusion_harmful": int(harmful),
                    "ordinary_fusion_safe": int(not harmful),
                    "gauge_sync_sufficient": int(global_accuracy >= reference - harm_margin),
                    "branch_lift_beneficial": int(orbit_accuracy >= raw_accuracy + benefit_margin),
                    "projective_rank_expansion_required": 0,
                    "abstention_recommended": int(harmful),
                    "ordinary_prediction_disagreement": 1.0 - float(run.prediction_consistency),
                    "estimated_period": 0.0,
                    "estimated_index": 0.0,
                    "candidate_capacity": float(run.branch_count),
                    "controlled_structural_residual": 0.0,
                    "capacity_divisible_by_index": 0.0,
                    "controlled_projective_indicator": 0.0,
                    "fusion_parameter_multiplier": float(run.parameter_multiplier_vs_single),
                    **diagnostics[seed],
                }
            )
            observation += 1

    for (seed, case_name, capacity), setting in c_runs.groupby(
        ["corpus_seed", "case_name", "candidate_capacity"]
    ):
        seed = int(seed)
        capacity = int(capacity)
        reference = float(
            setting[setting["method"] == "ensemble_upper_bound"]["classification_accuracy"].iloc[0]
        )
        ordinary = float(
            setting[setting["method"] == "ordinary_same_capacity"]["classification_accuracy"].iloc[0]
        )
        coherent = float(
            setting[setting["method"] == "coherent_projective_lift"]["classification_accuracy"].iloc[0]
        )
        for run in setting.itertuples(index=False):
            harmful = float(run.classification_accuracy) < reference - harm_margin
            rank_required = not bool(run.capacity_divisible_by_index)
            controlled_diagnostics = diagnostics[seed].copy()
            controlled_diagnostics.update(
                {
                    "centrality_residual": min(float(run.combined_structural_residual), 1.0),
                    "torsion_confidence": float(not rank_required),
                    "cocycle_residual": min(float(run.projective_relation_residual), 1.0),
                    "coboundary_residual": float(not rank_required),
                    "bootstrap_stability": 1.0,
                    "natural_connection_trivial_indicator": 0.0,
                }
            )
            rows.append(
                {
                    "observation_id": f"D{observation:04d}",
                    "evidence_label": "controlled_on_real_features",
                    "corpus_seed": seed,
                    "setting_family": str(case_name),
                    "fusion_method": run.method,
                    "reference_accuracy": reference,
                    "observed_accuracy": float(run.classification_accuracy),
                    "ordinary_fusion_harmful": int(harmful),
                    "ordinary_fusion_safe": int(not harmful),
                    "gauge_sync_sufficient": 0,
                    "branch_lift_beneficial": int(coherent >= ordinary + benefit_margin),
                    "projective_rank_expansion_required": int(rank_required),
                    "abstention_recommended": int(harmful or (rank_required and run.method == "ordinary_same_capacity")),
                    "ordinary_prediction_disagreement": 1.0 - float(run.prediction_consistency),
                    "estimated_period": float(run.period),
                    "estimated_index": float(run.predicted_index),
                    "candidate_capacity": float(capacity),
                    "controlled_structural_residual": min(float(run.combined_structural_residual), 1.0),
                    "capacity_divisible_by_index": float(bool(run.capacity_divisible_by_index)),
                    "controlled_projective_indicator": 1.0,
                    "fusion_parameter_multiplier": float(run.parameter_multiplier_vs_shared_corpus),
                    **controlled_diagnostics,
                }
            )
            observation += 1
    return pd.DataFrame(rows)


def grouped_metric_bootstrap(
    targets: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    cells: np.ndarray,
    samples: int,
    seed: int,
) -> list[dict[str, float | str]]:
    rng = np.random.default_rng(seed)
    unique_cells = np.unique(cells)
    values = {"auroc": [], "auprc": [], "brier_delta": []}
    for _ in range(samples):
        chosen = rng.choice(unique_cells, size=len(unique_cells), replace=True)
        indices = np.concatenate([np.flatnonzero(cells == cell) for cell in chosen])
        selected_targets = targets[indices]
        if len(np.unique(selected_targets)) < 2:
            continue
        baseline_metrics = linter_metrics(selected_targets, baseline[indices])
        candidate_metrics = linter_metrics(selected_targets, candidate[indices])
        values["auroc"].append(candidate_metrics.auroc - baseline_metrics.auroc)
        values["auprc"].append(candidate_metrics.auprc - baseline_metrics.auprc)
        values["brier_delta"].append(candidate_metrics.brier - baseline_metrics.brier)
    rows = []
    for metric, observations in values.items():
        array = np.asarray(observations, dtype=float)
        rows.append(
            {
                "metric": metric,
                "mean_delta_full_minus_baseline": float(array.mean()),
                "ci_low": float(np.quantile(array, 0.025)),
                "ci_high": float(np.quantile(array, 0.975)),
                "bootstrap_samples_retained": len(array),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "confirmatory"), required=True)
    parser.add_argument("--config", type=Path, default=APP_DIR / "config.json")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    command = " ".join([sys.executable, *sys.argv])
    output_dir = APP_DIR if args.mode == "confirmatory" else APP_DIR / "smoke"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "plots").mkdir(exist_ok=True)
    (output_dir / "tables").mkdir(exist_ok=True)
    dataset = assemble_dataset(args.mode, config)
    dataset.to_csv(output_dir / "dataset.csv", index=False)
    seeds = dataset["corpus_seed"].to_numpy()
    families = dataset["setting_family"].to_numpy()
    cells = np.asarray([f"{seed}:{family}" for seed, family in zip(seeds, families, strict=True)])
    outcomes = (
        "ordinary_fusion_harmful",
        "gauge_sync_sufficient",
        "branch_lift_beneficial",
        "projective_rank_expansion_required",
        "abstention_recommended",
    )
    prediction_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    coefficient_rows: list[dict[str, object]] = []
    coverage_rows: list[dict[str, object]] = []
    calibration_rows: list[dict[str, object]] = []
    capacity_rows: list[dict[str, object]] = []
    failure_rows: list[dict[str, object]] = []
    probabilities_by_key: dict[tuple[str, str], np.ndarray] = {}
    for outcome in outcomes:
        targets = dataset[outcome].to_numpy(dtype=int)
        adequate = (
            dataset["corpus_seed"].nunique() >= int(config["minimum_independent_seeds"])
            and int(targets.sum()) >= int(config["minimum_positive_examples"])
            and int((1 - targets).sum()) >= int(config["minimum_negative_examples"])
        )
        feature_sets = config["feature_sets"] if outcome == "ordinary_fusion_harmful" else {
            "full_holonomy_projective": config["feature_sets"]["full_holonomy_projective"]
        }
        for feature_set, columns in feature_sets.items():
            if not adequate:
                metric_rows.append(
                    {
                        "evidence_label": "diagnostic_only",
                        "mode": args.mode,
                        "outcome": outcome,
                        "feature_set": feature_set,
                        "status": "inadequate_sample",
                        "rows": len(targets),
                        "positive_examples": int(targets.sum()),
                        "negative_examples": int((1 - targets).sum()),
                        "independent_seeds": dataset["corpus_seed"].nunique(),
                    }
                )
                continue
            features = dataset[list(columns)].to_numpy(dtype=float)
            try:
                probabilities, folds, coefficients = double_holdout_predictions(
                    features,
                    targets,
                    seeds,
                    families,
                    random_state=1100000 + len(metric_rows),
                )
                probabilities_by_key[(outcome, feature_set)] = probabilities
                metrics = linter_metrics(targets, probabilities)
                metric_rows.append(
                    {
                        "evidence_label": "diagnostic_only",
                        "mode": args.mode,
                        "outcome": outcome,
                        "feature_set": feature_set,
                        "status": "evaluated_double_holdout",
                        "rows": len(targets),
                        "positive_examples": int(targets.sum()),
                        "negative_examples": int((1 - targets).sum()),
                        "independent_seeds": dataset["corpus_seed"].nunique(),
                        "independent_seed_family_cells": len(np.unique(cells)),
                        **metrics.__dict__,
                        "regret_vs_oracle_action": 1.0 - metrics.accuracy,
                    }
                )
                for index, probability in enumerate(probabilities):
                    prediction_rows.append(
                        {
                            "observation_id": dataset.iloc[index]["observation_id"],
                            "corpus_seed": int(seeds[index]),
                            "setting_family": families[index],
                            "outcome": outcome,
                            "feature_set": feature_set,
                            "target": int(targets[index]),
                            "probability": float(probability),
                            "predicted": int(probability >= 0.5),
                        }
                    )
                for fold in folds:
                    fold_rows.append({"outcome": outcome, "feature_set": feature_set, **fold})
                for fold_index, coefficients_for_fold in enumerate(coefficients):
                    for name, value in zip(columns, coefficients_for_fold, strict=True):
                        coefficient_rows.append(
                            {
                                "outcome": outcome,
                                "feature_set": feature_set,
                                "fold_index": fold_index,
                                "feature": name,
                                "logistic_coefficient_standardized": float(value),
                            }
                        )
                for row in coverage_risk_rows(targets, probabilities):
                    coverage_rows.append({"outcome": outcome, "feature_set": feature_set, **row})
                for row in reliability_rows(targets, probabilities):
                    calibration_rows.append({"outcome": outcome, "feature_set": feature_set, **row})
                capacity_rows.append(
                    {
                        "outcome": outcome,
                        "feature_set": feature_set,
                        "features": len(columns),
                        "logistic_parameters": len(columns) + 1,
                        "folds": len(folds),
                        "training_rows_min": min(int(row["train_rows"]) for row in folds),
                        "training_rows_max": max(int(row["train_rows"]) for row in folds),
                        "new_model_corpus": False,
                    }
                )
            except Exception as error:
                failure_rows.append(
                    {
                        "mode": args.mode,
                        "outcome": outcome,
                        "feature_set": feature_set,
                        "error_type": type(error).__name__,
                        "message": str(error),
                    }
                )

    metrics_frame = pd.DataFrame(metric_rows)
    predictions_frame = pd.DataFrame(prediction_rows)
    folds_frame = pd.DataFrame(fold_rows)
    coefficients_frame = pd.DataFrame(coefficient_rows)
    coverage_frame = pd.DataFrame(coverage_rows)
    calibration_frame = pd.DataFrame(calibration_rows)
    metrics_frame.to_csv(output_dir / "metrics.csv", index=False)
    predictions_frame.to_csv(output_dir / "predictions.csv", index=False)
    folds_frame.to_csv(output_dir / "fold_assignments.csv", index=False)
    coefficients_frame.to_csv(output_dir / "coefficients.csv", index=False)
    coverage_frame.to_csv(output_dir / "coverage_risk.csv", index=False)
    calibration_frame.to_csv(output_dir / "calibration.csv", index=False)
    pd.DataFrame(capacity_rows).to_csv(output_dir / "capacity_audit.csv", index=False)
    pd.DataFrame(failure_rows, columns=("mode", "outcome", "feature_set", "error_type", "message")).to_csv(
        output_dir / "failure_log.csv", index=False
    )
    outcome_summary = dataset[list(outcomes)].agg(["sum", "mean"]).T.reset_index().rename(
        columns={"index": "outcome", "sum": "positive_examples", "mean": "positive_fraction"}
    )
    outcome_summary.to_csv(output_dir / "outcome_summary.csv", index=False)

    baseline_name = str(config["primary_baseline"])
    full_name = str(config["primary_model"])
    paired_rows = []
    gate_passed = False
    key_baseline = ("ordinary_fusion_harmful", baseline_name)
    key_full = ("ordinary_fusion_harmful", full_name)
    if key_baseline in probabilities_by_key and key_full in probabilities_by_key:
        paired_rows = grouped_metric_bootstrap(
            dataset["ordinary_fusion_harmful"].to_numpy(dtype=int),
            probabilities_by_key[key_baseline],
            probabilities_by_key[key_full],
            cells,
            int(config[args.mode]["bootstrap_samples"]),
            1110000,
        )
        paired_lookup = {row["metric"]: row for row in paired_rows}
        discrimination = (
            float(paired_lookup["auroc"]["ci_low"]) > float(config["gate"]["minimum_auroc_delta"])
            or float(paired_lookup["auprc"]["ci_low"]) > float(config["gate"]["minimum_auprc_delta"])
        )
        calibration = float(paired_lookup["brier_delta"]["ci_high"]) <= float(
            config["gate"]["maximum_brier_increase"]
        )
        gate_passed = args.mode == "confirmatory" and discrimination and calibration
    pd.DataFrame(paired_rows).to_csv(output_dir / "paired_statistics.csv", index=False)

    recommended_capacity_accuracy = float(
        pd.read_csv(phase_dir("application_C_period_index_capacity", args.mode) / "minimum_capacity_predictions.csv")[
            "prediction_correct"
        ].mean()
    )
    claims = pd.DataFrame(
        [
            {
                "claim_id": "holonomy_projective_features_add_value",
                "status": "supported_diagnostic" if gate_passed else ("smoke_only" if args.mode == "smoke" else "negative"),
                "gate_passed": gate_passed,
                "safe_wording": "Full holonomy/projective features improve strict double-held-out harmful-fusion prediction over pairwise-plus-sync features." if gate_passed else "Full holonomy/projective features do not pass the incremental-value gate over pairwise-plus-sync features.",
            },
            {
                "claim_id": "holonomy_aware_auditing_improves_merge_abstain",
                "status": "supported_diagnostic" if gate_passed else ("smoke_only" if args.mode == "smoke" else "negative"),
                "gate_passed": gate_passed,
                "safe_wording": "The bounded linter improves merge/abstain decisions on unseen seed-family cells." if gate_passed else "The bounded linter does not establish improved merge/abstain decisions.",
            },
            {
                "claim_id": "recommended_capacity_accuracy",
                "status": "controlled_only",
                "gate_passed": recommended_capacity_accuracy == 1.0,
                "safe_wording": f"Controlled recommended-capacity accuracy is {recommended_capacity_accuracy:.3f}; it is inherited from Application C and is not a natural prediction claim.",
            },
        ]
    )
    claims.to_csv(output_dir / "claims.csv", index=False)

    if key_baseline in probabilities_by_key and key_full in probabilities_by_key:
        targets = dataset["ordinary_fusion_harmful"].to_numpy(dtype=int)
        figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
        for name, color in ((baseline_name, "#777777"), (full_name, "#2f6f9f")):
            probabilities = probabilities_by_key[("ordinary_fusion_harmful", name)]
            false_positive, true_positive, _ = roc_curve(targets, probabilities)
            precision, recall, _ = precision_recall_curve(targets, probabilities)
            axes[0].plot(false_positive, true_positive, label=name, color=color)
            axes[1].plot(recall, precision, label=name, color=color)
        axes[0].plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=0.8)
        axes[0].set_xlabel("False-positive rate")
        axes[0].set_ylabel("True-positive rate")
        axes[0].set_title("Double-held-out ROC")
        axes[1].set_xlabel("Recall")
        axes[1].set_ylabel("Precision")
        axes[1].set_title("Double-held-out precision-recall")
        axes[0].legend(fontsize=7)
        axes[1].legend(fontsize=7)
        figure.tight_layout()
        figure.savefig(output_dir / "plots" / "linter_discrimination.pdf", bbox_inches="tight")
        plt.close(figure)

        figure, axis = plt.subplots(figsize=(6.5, 4.5))
        for name in (baseline_name, full_name):
            rows = coverage_frame[
                (coverage_frame["outcome"] == "ordinary_fusion_harmful")
                & (coverage_frame["feature_set"] == name)
            ]
            axis.plot(rows["coverage"], rows["risk"], marker="o", label=name)
        axis.set_xlabel("Coverage")
        axis.set_ylabel("Decision risk")
        axis.set_title("Merge/abstain coverage-risk")
        axis.legend(fontsize=7)
        figure.tight_layout()
        figure.savefig(output_dir / "plots" / "coverage_risk.pdf", bbox_inches="tight")
        plt.close(figure)

    primary_metrics = metrics_frame[metrics_frame["outcome"] == "ordinary_fusion_harmful"]
    latex = ["\\begin{tabular}{lrrrr}", "\\toprule", "Features & AUROC & AUPRC & Brier & Avoidance\\\\", "\\midrule"]
    for row in primary_metrics.itertuples(index=False):
        if row.status != "evaluated_double_holdout":
            continue
        latex.append(
            f"{row.feature_set.replace('_', ' ')} & {row.auroc:.3f} & {row.auprc:.3f} & {row.brier:.3f} & {row.harmful_avoidance:.3f}\\\\"
        )
    latex.extend(["\\bottomrule", "\\end{tabular}", ""])
    (output_dir / "tables" / "application_D_linter.tex").write_text("\n".join(latex), encoding="utf-8")

    report = f"""# Application D: Holonomy-Aware Mergeability Linter

Decision: **{'bounded smoke; sample inadequate' if args.mode == 'smoke' else ('incremental diagnostic gate passed' if gate_passed else 'no incremental holonomy/projective value')}**.

## Commands

Smoke: `{sys.executable} experiments/holonomy_application_D.py --mode smoke`

Confirmatory: `{sys.executable} experiments/holonomy_application_D.py --mode confirmatory`

Executed: `{command}`

## Leakage and model boundary

The linter uses only accumulated A-C outputs plus shared adapter checkpoint metadata. It creates no new image/model corpus. Every prediction is double held out: the test observation's corpus seed and its entire setting family are both absent from training. The model class is logistic regression for every feature set; no tree or boosted fallback was tried after seeing results.

## Data and outcomes

- Observation rows: {len(dataset)}.
- Independent corpus seeds: {dataset['corpus_seed'].nunique()}.
- Seed-family cells: {len(np.unique(cells))}.
- Setting families: {sorted(dataset['setting_family'].unique())}.
- Outcome counts: `{outcome_summary.set_index('outcome')['positive_examples'].to_dict()}`.
- Recommended-capacity accuracy inherited from controlled Application C: `{recommended_capacity_accuracy:.3f}`.

## Primary result

- Baseline: `{baseline_name}`.
- Full diagnostic: `{full_name}`.
- Incremental discrimination/calibration gate: `{gate_passed}`.

{primary_metrics.to_csv(index=False)}

The only allowed positive claim would require holonomy/projective features to improve double-held-out discrimination without worsening Brier score. That gate {'passed' if gate_passed else 'did not pass'}. Controlled capacity labels remain separate from natural mergeability evidence.
"""
    (output_dir / "report.md").write_text(report, encoding="utf-8")

    committed = [
        output_dir / "dataset.csv",
        output_dir / "metrics.csv",
        output_dir / "predictions.csv",
        output_dir / "fold_assignments.csv",
        output_dir / "coefficients.csv",
        output_dir / "coverage_risk.csv",
        output_dir / "calibration.csv",
        output_dir / "capacity_audit.csv",
        output_dir / "outcome_summary.csv",
        output_dir / "paired_statistics.csv",
        output_dir / "claims.csv",
        output_dir / "tables" / "application_D_linter.tex",
    ]
    for plot_name in ("linter_discrimination.pdf", "coverage_risk.pdf"):
        path = output_dir / "plots" / plot_name
        if path.exists():
            committed.append(path)
    artifact_rows = [
        {
            "evidence_label": "diagnostic_only",
            "mode": args.mode,
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in committed
    ]
    pd.DataFrame(artifact_rows).to_csv(output_dir / "artifact_hashes.csv", index=False)
    if failure_rows:
        raise RuntimeError("Application D incomplete; inspect failure_log.csv")


if __name__ == "__main__":
    main()
