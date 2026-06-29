# TwistedMerge

Reproducible experiments for the paper project **"Twisted Sheaves and Descent Obstructions in Learning Theory."**

The central question is empirical:

> Do cohomological/cocycle obstruction scores predict failure of global model merging, and can twisted or rank-lifted merging reduce that failure?

This repository starts with independent synthetic obstruction experiments that do not require external model-merging repositories. MNIST/CIFAR and baseline wrappers are included as extension points, but the generated report only claims what has been run locally.

## Repository Layout

```text
src/
  cocycles.py          # MU(2)/U(1) cocycle generation, obstruction scores, synchronization
  alignment.py         # sign and phase alignment utilities
  twisted_merge.py     # descended/global merge and rank-lifted merge routines
  twisted_merge_algorithm.py # prototype TwistedMerge algorithm
  synthetic_tasks.py   # reproducible synthetic binary classification tasks
  models.py            # PyTorch model definitions for image experiments
  metrics.py           # accuracy, losses, summaries, environment capture
  plotting.py          # CSV-to-plot and LaTeX table helpers
experiments/
  synthetic_mu2_obstruction.py
  synthetic_u1_obstruction.py
  synthetic_h2_mu2_obstruction.py
  twisted_merge_algorithm_demo.py
  rank_lift_ablation.py
  model_merging_benchmark.py
  mnist_model_merging.py
  cifar_model_merging.py
reports/
  csv/
  plots/
  tables/
  configs/
  summary.md
```

## Install

For the synthetic experiments:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-synthetic.txt
```

For image experiments and PyTorch model merging:

```bash
python -m pip install -r requirements.txt
```

## Run the Synthetic Experiments

```bash
source .venv/bin/activate
python experiments/synthetic_mu2_obstruction.py
python experiments/synthetic_u1_obstruction.py
python experiments/synthetic_h2_mu2_obstruction.py
python experiments/twisted_merge_algorithm_demo.py
python experiments/rank_lift_ablation.py
```

Outputs are written under `reports/csv`, `reports/plots`, `reports/tables`, and `reports/configs`.

## Run the Small Model-Merging Benchmark

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
python experiments/model_merging_benchmark.py \
  --datasets mnist,cifar10 \
  --model-counts 2,3 \
  --widths 8,16 \
  --domain-shifts none,input_noise \
  --epochs 1 \
  --max-train-samples 384 \
  --max-test-samples 256 \
  --batch-size 128 \
  --device cpu
```

Outputs are written to `reports/model_merging_report.md`, `reports/checkpoints/`, `reports/csv/`, and `reports/plots/`.

## What the Synthetic Experiments Test

### H^2(mu_2) obstruction on a tetrahedral sphere

`experiments/synthetic_h2_mu2_obstruction.py` uses the boundary of a tetrahedron, a triangulated 2-sphere with nontrivial `H^2(-, mu_2)`. It constructs a trivial face cocycle and a nontrivial face cocycle with one negative triangle. Pairwise edge alignments are locally exact, while triple-overlap defects carry the prescribed central twist. The report is written to `reports/synthetic_obstruction_report.md`.

### TwistedMerge prototype

`experiments/twisted_merge_algorithm_demo.py` runs the prototype `TwistedMerge` algorithm. It computes triangle defects, tries gauge trivialization, checks whether defects match a finite central `mu_2` twist, and builds a q=2 doubled branch representation with the central sign acting by a 2x2 branch-swap matrix. The report is written to `reports/twisted_merge_algorithm_report.md`.

### mu_2 / sign cocycle

Local models are binary classifiers whose weights are related by hidden sign gauges. Pairwise alignment observations are corrupted by sign flips. The obstruction score is the fraction of frustrated triangles:

```text
s_ij * s_jk * s_ki = -1
```

The descended merge synchronizes a single global sign gauge and averages aligned weights. The rank-lifted merge keeps two sign branches and uses a validation split to select the branch for each local task.

### U(1) / phase cocycle

Local models are binary classifiers with weights rotated in two-dimensional feature blocks. Pairwise alignments are phase observations corrupted by angular noise. The obstruction score is the mean normalized triangle holonomy:

```text
abs(wrap(phi_ij + phi_jk + phi_ki)) / pi
```

The descended merge phase-synchronizes a single global gauge. The rank-lifted merge keeps a finite bank of phase branches and validates the branch per task.

## Baselines To Add Or Wrap

`experiments/model_merging_benchmark.py` implements small in-repo MLP/CNN baselines:

- ordinary weight averaging,
- greedy model soup,
- Git-Re-Basin-style pairwise permutation alignment,
- C2M3-style cycle-consistent permutation synchronization,
- ensemble upper bound,
- cycle-aware rank-lifted branch ensemble.

The first implementation intentionally does not vendor external code. These are the intended larger comparison points:

- Git Re-Basin: <https://github.com/samuela/git-re-basin>
- C2M3 cycle-consistent model merging: <https://github.com/crisostomi/cycle-consistent-model-merging>
- Model Soups: <https://github.com/mlfoundations/model-soups>
- Optional: RegMean, TIES-Merging, mergekit/MergeBench.

When those are added, reports should separate synthetic-only claims from image/model-merging claims.

## Reporting Rule

Do not claim success from theory or intent. A supported claim must cite generated data in `reports/summary.md`, including:

1. exact commands run,
2. hardware/software environment,
3. metrics,
4. tables,
5. whether each claim is supported or unsupported.
