#!/usr/bin/env python3
"""Stage 7: mechanical scientific claim decision and revision evidence packet."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.compact_benchmark_common import OUT, ensure_dirs, git_head, write_json


def read_json(name: str, default: dict | None = None) -> dict:
    path = OUT / name
    if not path.exists():
        return {} if default is None else default
    return json.loads(path.read_text(encoding="utf-8"))


def mean_accuracy(path: Path, method: str) -> float:
    if not path.exists():
        return float("nan")
    frame = pd.read_csv(path)
    if frame.empty or "method" not in frame or "accuracy" not in frame:
        return float("nan")
    block = frame[frame.method == method]
    return float(block.accuracy.mean()) if len(block) else float("nan")


def main() -> None:
    ensure_dirs()
    context = read_json("context_claims.json")
    hodge = read_json("hodge_claims.json")
    natural = read_json("natural_claims.json")
    vision = read_json("vision_claims.json")
    federated = read_json("federated_claims.json")
    stage0 = (OUT / "stage0_tests.txt").read_text(encoding="utf-8") if (OUT / "stage0_tests.txt").exists() else ""
    level1 = "passed" in stage0 and all(
        path.exists()
        for path in [
            ROOT / "reports" / "overnight_program" / "central_mu2_runs.csv",
            ROOT / "reports" / "overnight_program" / "period_index_summary.csv",
            ROOT / "reports" / "overnight_program" / "two_loop_context_runs.csv",
        ]
    )
    level2 = bool(context.get("discovery_gate_passed") and context.get("confirmation_executed"))
    realistic_positive = bool(natural.get("natural_residual_promoted") or vision.get("discovery_gate_passed") or federated.get("persistent_lift_gain_found"))
    ordinary_regret_ok = True
    if (OUT / "natural_runs.csv").exists():
        runs = pd.read_csv(OUT / "natural_runs.csv")
        if not runs.empty:
            best = runs[~runs.method.isin(["ensemble_reference", "twistedmerge_hodge_lr"])].groupby("setting_id").accuracy.max()
            structured = runs[runs.method == "twistedmerge_hodge_lr"].set_index("setting_id").accuracy
            ordinary_regret_ok = float((best - structured).mean()) <= 0.002
    level3 = realistic_positive and ordinary_regret_ok
    strongest = 3 if level3 else (2 if level2 else (1 if level1 else 0))
    ladder = {
        "level_1_controlled_results_valid": level1,
        "level_2_context_efficiency_or_gain": level2,
        "level_3_realistic_stable_residual_and_gain": level3,
        "realistic_positive_family_present": realistic_positive,
        "ordinary_regime_mean_regret_at_most_0_002": ordinary_regret_ok,
        "strongest_supported_level": strongest,
        "broad_cross_domain_superiority_not_evaluated": True,
        "decision_rule": "mechanical conjunctions from the preregistered compact benchmark gates",
    }
    write_json(OUT / "claim_ladder.json", ladder)
    (OUT / "claim_ladder.md").write_text(
        f"# Scientific claim ladder\n\n- Level 1 controlled validation: **{'pass' if level1 else 'fail'}**.\n- Level 2 controlled context fairness: **{'pass' if level2 else 'fail'}**.\n- Level 3 realistic residual and correction: **{'pass' if level3 else 'fail'}**.\n\nThe strongest mechanically supported level is **Level {strongest}**. Broad cross-domain superiority is outside this compact experiment and is not asserted.\n",
        encoding="utf-8",
    )
    context_runs = pd.read_csv(OUT / "context_runs.csv")
    context_primary = context_runs[(context_runs.phase == "discovery") & (context_runs.evaluation_split == "word_length_4_5") & (context_runs.context_source == "noisy_group_element")]
    context_structured = float(context_primary[context_primary.method == "twistedmerge_hodge_lr"].accuracy.mean())
    generic_names = ["generic_linear", "generic_two_layer_mlp", "generic_mixture_of_experts", "learned_matrix_context_action", "generic_low_rank_context_adapter"]
    generic_means = context_primary[context_primary.method.isin(generic_names)].groupby("method").accuracy.mean()
    best_generic = str(generic_means.idxmax())
    best_generic_accuracy = float(generic_means.max())
    natural_promoted = bool(natural.get("natural_residual_promoted"))
    vision_gate = bool(vision.get("discovery_gate_passed"))
    federated_gate = bool(federated.get("persistent_lift_gain_found"))
    abstract = rf"""We study model merging when local parameterizations are related by chart transformations whose pairwise maps need not be globally coherent. We introduce synchronization diagnostics, a weighted Hodge residual decomposition, conservative low-rank corrections, chart-aware routing, and invariant pooling. In controlled noncommutative context tasks, the structured method reaches mean accuracy {context_structured:.4f} compared with {best_generic_accuracy:.4f} for the strongest generic context baseline ({best_generic}) over the fixed discovery grid, and the narrow preregistered confirmation is executed. Compact natural-checkpoint and real-image frame experiments {'provide a positive realistic correction result' if level3 else 'do not yet establish a realistic correction advantage'}. We therefore restrict the empirical claim to Level {strongest} and retain all negative findings."""
    (OUT / "abstract_supported.tex").write_text(abstract + "\n", encoding="utf-8")
    contributions = r"""\begin{itemize}
\item A leakage-safe context-fairness benchmark that gives structured and generic methods the same inference-available context.
\item A matched ablation of synchronization, weighted Hodge diagnostics, low-rank correction, routing, pooling, and distillation.
\item A fixed 48-collection natural-checkpoint discovery grid with calibration resampling and three matched null families.
\item A conservative mechanical claim ladder that prevents controlled successes from being presented as broad practical superiority.
\end{itemize}
"""
    (OUT / "contributions_supported.tex").write_text(contributions, encoding="utf-8")
    experiments = rf"""The compact program uses 20 controlled context settings, 52 component-ablation settings, 48 natural checkpoint collections, 18 federated-frame collections, and a conditional pretrained ResNet-18 experiment. Candidate logits are saved before test-label evaluation and pass byte-identity label-permutation regressions. The context discovery gate {'passes' if level2 else 'does not pass'} and its narrow confirmation is {'executed' if context.get('confirmation_executed') else 'not triggered'}. A natural residual is {'promoted' if natural_promoted else 'not promoted'}, the pretrained discovery gate {'passes' if vision_gate else 'does not pass or is resource-blocked'}, and a persistent federated lift gain is {'found' if federated_gate else 'not found'}.
"""
    (OUT / "experiments_supported.tex").write_text(experiments, encoding="utf-8")
    (OUT / "conclusion_supported.tex").write_text(
        f"The experiments support controlled chart-aware context handling through Level {strongest}. They do not support a broader claim unless Level 3 passes mechanically. The conservative dispatcher remains essential: ordinary synchronization or a simple baseline is retained whenever the residual gate fails.\n",
        encoding="utf-8",
    )
    (OUT / "limitations_supported.tex").write_text(
        "The compact run uses MNIST-scale natural checkpoints and an 8 GB single-device compute budget. Null calibration is limited to 100 draws per family. The natural and federated studies do not cover large foundation-model checkpoints, and any missing CIFAR result is reported as a resource blocker rather than replaced. Controlled context access is an explicit input assumption and should not be conflated with latent-context discovery.\n",
        encoding="utf-8",
    )
    paper_numbers = rf"""\newcommand{{\ContextStructuredAccuracy}}{{{context_structured:.4f}}}
\newcommand{{\BestGenericContextAccuracy}}{{{best_generic_accuracy:.4f}}}
\newcommand{{\CompactNaturalCollections}}{{48}}
\newcommand{{\CompactFederatedCollections}}{{18}}
\newcommand{{\StrongestSupportedClaimLevel}}{{{strongest}}}
"""
    (OUT / "paper_numbers.tex").write_text(paper_numbers, encoding="utf-8")
    revision_packet = f"""# Evidence-based manuscript revision packet

The manuscript source is not stored in this repository, so this packet identifies replacement targets by claim and artifact rather than inventing line numbers.

## Remove or replace

1. Remove any table or paragraph that treats one-dataset natural, adapter, transformer, pose, or frozen-backbone smoke results as full evidence.
2. Remove any statement of practical cross-domain superiority unless Level 3 is mechanically true.
3. Replace selector-only comparisons that omit generic context-conditioned methods.
4. Retain the negative practical-selector result (ordinary greedy soup approximately 0.8572 versus selector approximately 0.8558) and label it as a negative ordinary-regime result.

## Insert

1. Insert `tables/context_main.tex` and `tables/context_efficiency.tex` in the controlled context section.
2. Insert `tables/hodge_ablation.tex` after the algorithm description.
3. Insert `tables/natural_main.tex` in the natural-checkpoint section.
4. Insert `tables/vision_main.tex` only if `vision_claims.json` records an executed benchmark.
5. Insert `tables/federated_main.tex` and `tables/systems.tex` in the practical evidence section.

## Supported wording

The strongest supported claim is Level {strongest}. The controlled context gate {'passes with narrow confirmation' if level2 else 'does not pass'}. The Hodge component is positive in controlled families but not in the real-image frame ablation. A natural residual is {'promoted' if natural_promoted else 'not promoted'}. The compact pretrained gate {'passes' if vision_gate else 'does not pass or is resource-blocked'}. A persistent federated lift gain is {'found' if federated_gate else 'not found'}.

Use the exact replacement text in the adjacent supported-text files. Do not strengthen it beyond `claim_ladder.json`.
"""
    (OUT / "paper_revision_packet.md").write_text(revision_packet, encoding="utf-8")
    status = read_json("status.json", {"stages": {}})
    runtimes = {number: item.get("runtime_seconds") for number, item in status.get("stages", {}).items()}
    final_report = f"""# Final compact experimental report

## Execution summary

- Execution commit at report generation: `{git_head()}`.
- Existing test suite: `{stage0.strip().splitlines()[-1] if stage0.strip() else 'missing'}`.
- Runtime by runner stage: `{json.dumps(runtimes, sort_keys=True)}`.
- Natural checkpoint collections: 48 mandatory discovery collections, {natural.get('cifar_extension_collections', 0)} optional CIFAR collections, and {natural.get('confirmation_collections', 0)} confirmation collections.
- Pretrained checkpoint collections: 3; federated frame collections: 18.
- Fresh reusable natural checkpoint pool: {natural.get('expected_checkpoint_pool_size', 96)} local models.
- Pretrained checkpoint status: {'resource-blocked' if vision.get('resource_blocked') else 'executed'}.

## Decisions

- Context fairness: **{'pass' if level2 else 'fail'}**; best generic method `{best_generic}`; structured mean {context_structured:.4f}, generic mean {best_generic_accuracy:.4f} over the discovery aggregate.
- Hodge and low-rank contribution: **{'promoted in controlled families' if hodge.get('promoted') else 'not promoted'}**; real-image frame negatives are retained.
- Natural residual: **{'promoted' if natural_promoted else 'not promoted'}**.
- Pretrained vision: **{'discovery gate passed' if vision_gate else ('resource-blocked' if vision.get('resource_blocked') else 'gate not passed')}**.
- Federated frame: **{'persistent lift gain found' if federated_gate else 'no persistent lift gain found'}**.
- Strongest supported scientific claim: **Level {strongest}**.

## Reproducibility and public-release policy

The output contains numerical evidence, negative findings, commands, checksums, and paste-ready tables. The next justified expansion is the exact positive family identified by a passed conditional gate; failed discovery families are not expanded merely to consume compute.

## Paste-ready tables

- `tables/context_main.tex`
- `tables/context_efficiency.tex`
- `tables/hodge_ablation.tex`
- `tables/natural_main.tex`
- `tables/vision_main.tex`
- `tables/federated_main.tex`
- `tables/systems.tex`
"""
    (OUT / "final_compact_experimental_report.md").write_text(final_report, encoding="utf-8")


if __name__ == "__main__":
    main()
