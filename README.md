# TwistedMerge

Code and reproducible artifacts for the paper project **"Descent Obstructions and Twisted Sheaves in Learning Theory."**

TwistedMerge studies model merging as a descent problem. Pairwise alignments between models define transition maps; triangle products define cycle defects; residual structure controls which merges are ordinary, cycle-consistent, diagnostic, or lift-based.

At a high level, the evidence pipeline is:

```text
pairwise alignments -> triangle defects -> residual type -> merge operator or diagnostic gate
```

## Project Status

This repository is research code with generated evidence reports. The strongest results are controlled obstruction witnesses and conservative diagnostics. Practical neural-network results are intentionally scoped to the exact datasets, architectures, seeds, and gates recorded in `reports/`.

Current evidence supports these public-facing themes:

- Controlled `mu_2`, finite-index, period-index, time-frequency, and nonabelian holonomy experiments realize explicit obstruction and rank-threshold behavior.
- Controlled twisted-overlap and planted-obstruction runs connect cycle defects with merge degradation in settings where the obstruction is known.
- ReLU-compatible exact gauges for MLPs and small CNNs give limited validation-gated improvements over internal C2M3-style synchronization on selected MNIST and Fashion-MNIST settings.
- Greedy soup remains the main boundary baseline across practical neural-network experiments.
- Real MNIST, Fashion-MNIST, and CIFAR residual diagnostics mostly fall into noncentral, diagnostic, or negative categories under the tested structure groups.
- CIFAR experiments pass a bounded no-BatchNorm base-accuracy gate; exact-gauge effects there are descriptive appendix evidence.
- Official external-code integration attempts are documented separately from faithful in-repo baseline comparisons.

For claim status, start with:

- `reports/claims_audit.md`
- `reports/full_capacity_claim_audit.md`
- `reports/final_evidence_freeze_manifest.md`
- `reports/final_claim_ledger.md`

## Repository Map

```text
src/
  Core libraries for cocycles, pairwise alignment, model merging,
  monomial/channel gauges, holonomy diagnostics, period-index detectors,
  block gauges, residual peeling, metrics, and plotting.

experiments/
  Reproducible experiment entry points. The scripts write reports,
  CSVs, plots, and config snapshots under reports/.

reports/
  Generated evidence, tables, plots, LaTeX snippets, configs, and
  claim-audit files. This is the main public record of completed runs.

external_baselines/
  Documentation for official-code integration attempts and external
  baseline provenance.

tests/
  Unit and smoke tests for controlled algebraic cases, detector gates,
  exact-gauge preservation, selectors, and report-generation helpers.
```

## Conceptual Layers

The repository is easiest to read in three layers:

- Theory: cocycles, descent defects, period-index thresholds, finite-index lifts, and controlled obstruction witnesses.
- Merge operators: weight averaging, Git-ReBasin-style alignment, C2M3-style synchronization, exact monomial/channel gauges, soups, and branch/rank-lift comparisons.
- Diagnostics: residual taxonomy, cycle scores, block gauges, holonomy splitting, time-frequency chart recovery, capacity audits, and claim ledgers.

This split keeps model-producing methods separate from diagnostics that measure residual structure. The main practical reading frame is:

| Residual type | Repository treatment |
| --- | --- |
| Permutation residual | Git-ReBasin-style alignment and C2M3-style synchronization |
| Positive scale or channel gauge residual | Exact ReLU-compatible monomial and CNN channel gauge merges |
| Controlled central or projective residual | Finite-index, period-index, and rank-lift controlled experiments |
| Noncentral or unstable residual | Diagnostic report, fallback selector, or abstention |
| Extra-capacity lift | Branch comparisons with capacity metadata and matched baselines |

## Setup

Synthetic and algebraic experiments use a small dependency set:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-synthetic.txt
```

Image/model-merging experiments use the PyTorch dependency set:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The scripts run on CPU. Repeated-seed image experiments can take substantial time.

## Main Experiment Families

### Controlled obstruction witnesses

These scripts cover explicit synthetic cocycle, central twist, period-index, time-frequency, nonabelian holonomy, and block-gauge cases.

```bash
source .venv/bin/activate
python experiments/synthetic_mu2_obstruction.py
python experiments/synthetic_u1_obstruction.py
python experiments/synthetic_h2_mu2_obstruction.py
python experiments/twisted_merge_algorithm_demo.py
python experiments/finite_index_twist_absorption.py
python experiments/period_index_central_benchmark.py
python experiments/time_frequency_period_index_benchmark.py
python experiments/controlled_nonabelian_holonomy.py
python experiments/block_gauge_phase_diagram.py
```

Representative outputs:

- `reports/synthetic_obstruction_report.md`
- `reports/twisted_merge_algorithm_report.md`
- `reports/finite_index_twist_report.md`
- `reports/period_index_central_report.md`
- `reports/time_frequency_period_index_report.md`
- `reports/controlled_nonabelian_holonomy_report.md`
- `reports/block_gauge_phase_diagram_report.md`

### Real fixed-setting verification

The current paper-grade real verification entry point is:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/codex-pycache MPLCONFIGDIR=/private/tmp/mplconfig \
.venv/bin/python experiments/model_merging_fixed_setting_verification.py \
  --datasets mnist,fashion_mnist \
  --architecture mlp2 \
  --model-counts 3,4 \
  --widths 128 \
  --domain-shifts none,input_noise \
  --seeds 4100:4129 \
  --epochs 10 \
  --max-train-samples 10000 \
  --max-test-samples 2000 \
  --batch-size 128 \
  --lr 0.001 \
  --optimizer adamw \
  --weight-decay 0.0001 \
  --scheduler cosine \
  --matching activation,weight \
  --bootstrap-samples 2000 \
  --alignment-noise-levels 0.15 \
  --rank-lift-branches 2 \
  --feature-batches 8 \
  --device auto
```

Primary outputs:

- `reports/fixed_setting_verification_report.md`
- `reports/fixed_setting_full_run_interpretation.md`
- `reports/real_obstruction_degradation_report.md`
- `reports/csv/fixed_setting_verification_runs.csv`
- `reports/csv/fixed_setting_verification_stats.csv`
- `reports/csv/fixed_setting_triangle_defects.csv`
- `reports/csv/real_obstruction_predictor_regressions.csv`

The recorded gates track individual-model quality, observed-seed count, bootstrap confidence intervals, injected-noise controls, selector validation usage, and capacity metadata.

### Exact ReLU-compatible gauges

The ReLU-compatible gauge experiments cover positive monomial gauges for MLPs and positive channel gauges for small CNNs.

Useful entry points and artifacts:

- `src/monomial_gauge_alignment.py`
- `src/cnn_channel_gauge.py`
- `experiments/greedy_aware_monomial_benchmark.py`
- `experiments/fashion_mnist_cnn_channel_gauge_confirmatory.py`
- `reports/monomial_gauge_alignment_report.md`
- `reports/fashion_mnist_improved_ladder_report.md`
- `reports/fashion_mnist_cnn_channel_gauge_confirmatory_report.md`

These experiments report same-capacity gauge-preserving merges separately from branch ensembles, soups, and upper-bound ensembles.

### Holonomy, quotient, and residual-peeling diagnostics

The repository includes several conservative residual-analysis pipelines:

- `experiments/small_quotient_holonomy_splitting.py`
- `experiments/group_cohomology_torsion_hunting.py`
- `experiments/nonabelian_holonomy_splitting.py`
- `experiments/primary_holonomy_splitting.py`
- `experiments/loss_aware_primary_peeling_smoke.py`
- `experiments/primary_residual_peeling_smoke_v2.py`

Recent large selector output from `primary_holonomy_splitting.py` is stored as GitHub-safe shards:

- `reports/csv/primary_holonomy_selector_results_manifest.csv`
- `reports/csv/primary_holonomy_selector_results_part_000.csv`
- `reports/csv/primary_holonomy_selector_results_part_001.csv`
- `reports/csv/primary_holonomy_selector_results_part_002.csv`
- `reports/csv/primary_holonomy_selector_results_part_003.csv`
- `reports/csv/primary_holonomy_selector_results_part_004.csv`
- `reports/csv/primary_holonomy_selector_results_part_005.csv`

The manifest gives row ranges and byte sizes for each shard.

### Bridge and boundary datasets

Fashion-MNIST, rotated-MNIST, colored-MNIST, and CIFAR runs are organized as boundary checks for practical exact-gauge behavior:

- `reports/fashion_mnist_improved_ladder_report.md`
- `reports/fashion_mnist_cnn_channel_gauge_confirmatory_report.md`
- `reports/bridge_dataset_channel_gauge_expansion.md`
- `reports/cifar_rescue_or_no_go_report.md`
- `reports/cifar_final_channel_gauge_confirmatory_report.md`
- `reports/cifar_bridge_boundary_summary.md`

These reports separate internal C2M3-style comparisons, greedy-soup comparisons, exact-gauge diagnostics, and base-accuracy gates.

### External baselines

Official-code integration status lives in:

- `external_baselines/README.md`
- `external_baselines/OFFICIAL_INTEGRATION.md`
- `external_baselines/NSD_INTEGRATION.md`
- `reports/external_baseline_comparison.md`
- `reports/official_external_baseline_attempt.md`
- `reports/nsd_official_integration_report.md`

The official-code documents record repository URLs, licenses, commit hashes, environment checks, integration blockers, and exact settings where official code was attempted.

## Recommended Reading Order

1. `reports/final_evidence_freeze_manifest.md`
2. `reports/final_claim_ledger.md`
3. `reports/claims_audit.md`
4. `reports/full_capacity_claim_audit.md`
5. `reports/paper_level_decision_after_35791f7.md`
6. `reports/results_narrative_after_35791f7.md`
7. Dataset-specific reports listed in the relevant section above

## Reproducibility Conventions

Each substantial experiment records:

- command-line arguments,
- random seeds,
- environment metadata where available,
- generated CSV row counts,
- validation/test split usage,
- bootstrap intervals,
- paired deltas,
- capacity and inference multipliers when relevant.

Smoke tests are plumbing checks. Paper-grade runs use the report-specific gates and repeated-seed settings described in the corresponding markdown report.

## Artifact Hygiene

Large local data, virtual environments, caches, and long-run checkpoints are ignored by Git. Public CSV artifacts are kept below GitHub blob limits through compact scalar tables, gzip shards, or plain CSV shards with manifests.

Relevant compaction artifacts:

- `reports/csv/fixed_setting_large_artifacts_manifest.csv`
- `reports/csv/fixed_setting_large_artifacts/*.csv.gz`
- `reports/csv/primary_holonomy_selector_results_manifest.csv`

Large local checkpoint directories under `reports/checkpoints/`, including repeated-seed task-vector checkpoints, can be regenerated from the recorded commands. They are useful for audit and reruns on the same machine.

## Development Checks

Run focused tests while editing a method family:

```bash
source .venv/bin/activate
PYTHONPYCACHEPREFIX=/private/tmp/codex-pycache python -m pytest tests/test_primary_holonomy.py
PYTHONPYCACHEPREFIX=/private/tmp/codex-pycache python -m pytest tests/test_monomial_gauge_alignment.py
PYTHONPYCACHEPREFIX=/private/tmp/codex-pycache python -m pytest tests/test_cnn_channel_gauge.py
```

Before publishing changes:

```bash
git diff --check
git rev-list --objects origin/main..HEAD \
  | git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' \
  | awk '$1=="blob" && $3 >= 95000000 {printf "%.1f MB %s\n", $3/1048576, $4}'
```

The second command prints oversized blobs in commits waiting to be pushed.
