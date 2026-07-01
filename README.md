# TwistedMerge

Reproducible experiments for the paper project **"Twisted Sheaves and Descent Obstructions in Learning Theory."**

The central empirical question is:

> Do cocycle or cohomological obstruction scores predict failure of global model merging, and can cycle-consistent, twisted, or rank-lifted procedures reduce that failure?

This repository is intentionally claim-audited. Synthetic obstruction experiments are independent of external model-merging repositories. Real MNIST/Fashion-MNIST model-merging runs are included, but the reports only promote claims that pass the recorded gates.

## Current Evidence Snapshot

- Controlled synthetic `mu_2` and `H^2(mu_2)` experiments are implemented and reported.
- The TwistedMerge prototype detects controlled finite central twist failures and a q=2 branch lift recovers prediction in that controlled setting.
- The quality-gated real fixed-setting run has good individual models, but only one primary observed Fashion-MNIST setting passes the strict obstruction-prediction gate. The other observed settings remain unsupported.
- The current real run does not support broad MNIST/Fashion-MNIST obstruction prediction, external-baseline superiority, or a claim that rank lifts are capacity-matched single-model improvements.
- External baseline integrations are documented separately; do not read them as official validation unless the report says the official code ran for that exact setting.

The most important claim-boundary files are:

- `reports/claims_audit.md`
- `reports/full_capacity_claim_audit.md`
- `reports/fixed_setting_verification_report.md`
- `reports/fixed_setting_full_run_interpretation.md`
- `reports/real_obstruction_degradation_report.md`

## Repository Layout

```text
src/
  cocycles.py
  alignment.py
  twisted_merge.py
  twisted_merge_algorithm.py
  monomial_gauge_alignment.py
  model_merging_benchmark.py
  synthetic_tasks.py
  models.py
  metrics.py
  plotting.py
experiments/
  synthetic_mu2_obstruction.py
  synthetic_u1_obstruction.py
  synthetic_h2_mu2_obstruction.py
  twisted_merge_algorithm_demo.py
  model_merging_benchmark.py
  model_merging_fixed_setting_verification.py
  compact_fixed_setting_outputs.py
  train_quality_sweep.py
reports/
  csv/
  plots/
  tables/
  configs/
external_baselines/
tests/
```

## Install

For synthetic experiments:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-synthetic.txt
```

For PyTorch image/model-merging experiments:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The code is CPU-runnable by design, but full repeated-seed runs can still take a while.

## Synthetic Experiments

```bash
source .venv/bin/activate
python experiments/synthetic_mu2_obstruction.py
python experiments/synthetic_u1_obstruction.py
python experiments/synthetic_h2_mu2_obstruction.py
python experiments/twisted_merge_algorithm_demo.py
python experiments/rank_lift_ablation.py
```

Key outputs:

- `reports/synthetic_obstruction_report.md`
- `reports/twisted_merge_algorithm_report.md`
- `reports/twisted_merge_algorithm_verification.md`
- `reports/csv/`
- `reports/plots/`

## Training Quality Sweep

Use this before making real model-merging claims. It measures individual model quality only.

```bash
source .venv/bin/activate
PYTHONPYCACHEPREFIX=/private/tmp/codex-pycache MPLCONFIGDIR=/private/tmp/mplconfig \
.venv/bin/python experiments/train_quality_sweep.py
```

Key outputs:

- `reports/csv/training_quality_sweep.csv`
- `reports/training_quality_sweep_report.md`

The current quality gate used `mlp2`, width `128`, AdamW, cosine scheduling, 10 epochs, and 10000 train samples.

## Real Fixed-Setting Verification

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

- `reports/csv/fixed_setting_verification_runs.csv`
- `reports/csv/fixed_setting_verification_stats.csv`
- `reports/csv/fixed_setting_triangle_defects.csv`
- `reports/csv/fixed_setting_individual_models.csv`
- `reports/csv/real_obstruction_predictor_regressions.csv`
- `reports/fixed_setting_verification_report.md`
- `reports/fixed_setting_full_run_interpretation.md`

Claim gate:

- Each fixed setting needs at least 20 observed seeds.
- Individual mean accuracy should clear the dataset quality gate.
- Predictor support requires positive Pearson, positive Spearman, and a positive bootstrap Pearson lower bound.
- Injected-noise rows are controls, not primary evidence.

## Large Artifact Compaction

Full fixed-setting verifier outputs can contain large pairwise and layerwise permutation maps. To keep GitHub blobs below size limits, the current committed CSVs keep scalar metrics inline and move bulky map fields into deterministic gzip shards:

```bash
.venv/bin/python experiments/compact_fixed_setting_outputs.py --rows-per-shard 2000
```

Manifest and shards:

- `reports/csv/fixed_setting_large_artifacts_manifest.csv`
- `reports/csv/fixed_setting_large_artifacts/*.csv.gz`

The compact CSVs include `large_field_shard` and `large_field_row` pointers back to the moved raw fields. Local checkpoints from long verifier runs are intentionally ignored by Git.

## Monomial Gauge Experiments

Positive ReLU-compatible monomial gauges are implemented for supported MLP paths. These rows are separate from the Prompt 11 `activation,weight` fixed-setting run unless explicitly requested with monomial matching modes.

Useful artifacts:

- `src/monomial_gauge_alignment.py`
- `reports/monomial_gauge_alignment_report.md`
- `reports/csv/monomial_fixed_setting_runs.csv`
- `reports/csv/monomial_triangle_defects.csv`

Implementation support is not the same as a performance claim. The claim audit remains authoritative.

## External Baselines

In-repo baselines include:

- ordinary weight averaging,
- greedy model soup,
- Git-ReBasin-style pairwise permutation alignment,
- C2M3-style cycle-consistent synchronization,
- ensemble upper bound,
- cycle-aware rank-lifted branch ensemble.

External references:

- Git Re-Basin: <https://github.com/samuela/git-re-basin>
- C2M3 cycle-consistent model merging: <https://github.com/crisostomi/cycle-consistent-model-merging>
- Model Soups: <https://github.com/mlfoundations/model-soups>

Current external integration reports separate official-code attempts from faithful in-repo surrogates:

- `reports/external_baseline_comparison.md`
- `reports/nsd_official_integration_report.md`
- `external_baselines/NSD_INTEGRATION.md`

## Reporting Rule

Do not claim success from theory or intent. A supported claim must cite generated data and exact commands. If a setting is negative, mixed, underpowered, extra-capacity, validation-selected, or only a smoke test, the report should say so plainly.
