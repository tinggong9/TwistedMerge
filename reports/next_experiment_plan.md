# Next Experiment Plan

This plan is for strengthening the paper **"Descent Obstructions and Twisted Sheaves in Learning Theory."** It is a planning artifact only: no planned result below is a claim of success. The repo already contains many useful components, but the real-model obstruction-to-degradation evidence is still underpowered and does not yet save all reproducibility artifacts required by the paper.

## Current Repository Inventory

### Already Implemented

Synthetic obstruction experiments:

- `experiments/synthetic_mu2_obstruction.py`
  - Uses `src.cocycles.sample_mu2_cocycle`, `mu2_triangle_obstruction`, `estimate_mu2_gauges_spectral`, `descended_mu2_merge`, and `rank_lift_mu2_merge`.
  - Outputs `reports/csv/synthetic_mu2_results.csv`, `reports/csv/synthetic_mu2_summary.csv`, `reports/tables/synthetic_mu2_summary.tex`, `reports/plots/synthetic_mu2_obstruction.png`, and `reports/configs/synthetic_mu2_config.json`.
  - Records obstruction score, naive failure, rank-lift gain, edge agreement, gauge accuracy, seeds, and hyperparameters.

- `experiments/synthetic_u1_obstruction.py`
  - Uses `sample_u1_cocycle`, `u1_triangle_obstruction`, `estimate_u1_phases_spectral`, `descended_u1_merge`, and `rank_lift_u1_merge`.
  - Outputs analogous U(1) CSV, table, plot, and config files.
  - Records phase residuals, obstruction score, naive failure, rank-lift gain, branch count, seeds, and hyperparameters.

- `experiments/synthetic_h2_mu2_obstruction.py`
  - Uses `src.simplicial_mu2` to build the tetrahedral-sphere `H^2(mu_2)` witness.
  - Produces `reports/synthetic_obstruction_report.md`, `reports/csv/synthetic_h2_mu2_obstruction.csv`, `reports/plots/synthetic_h2_mu2_obstruction_vs_failure.png`, `reports/plots/synthetic_h2_mu2_rank_success.png`, and config JSON.
  - Current report supports the controlled statement that the nontrivial rank-1 case has local/pairwise zero loss but ordinary global merge loss `0.250`, while rank `2` absorbs the sign twist in that construction.

Core TwistedMerge prototype:

- `src/twisted_merge.py`
  - Implements descended and rank-lifted merge routines for NumPy vector classifiers: `descended_mu2_merge`, `rank_lift_mu2_merge`, `descended_u1_merge`, and `rank_lift_u1_merge`.

- `src/twisted_merge_algorithm.py`
  - Implements the auditable vector `TwistedMerge` prototype.
  - Computes pairwise maps `g_ij`, triangle defects `c_ijk = g_ij g_jk g_ki`, gauge synchronization residuals, central twist residuals, ordinary merge, cycle-consistent merge, and a q=2 `mu_2` branch model.
  - Important boundary: the `TwistedVectorModel` is branch-prediction machinery; non-coboundary `H^2` transition-level descent is not solved by the current prototype unless an edge cochain exists.

Model-merging benchmark:

- `src/model_merging_benchmark.py`
  - Defines `PermutableMLP` and `PermutableCNN`, data loaders for MNIST, Fashion-MNIST, CIFAR-10, and fake datasets, activation/weight permutation matching, `cycle_score`, `synchronize_permutations`, model averaging, greedy soup, checkpoint saving, and `rank_lifted_branch_models`.
  - Current `rank_lifted_branch_models` returns a branch ensemble. It is explicitly extra capacity and is not capacity matched to a single merged model.

- `experiments/model_merging_benchmark.py`
  - Benchmark mode writes `reports/csv/model_merging_benchmark.csv`, `reports/csv/model_merging_individual_models.csv`, `reports/csv/model_merging_cycle_defects.csv`, plots, checkpoints, and `reports/model_merging_report.md`.
  - Verification mode writes `reports/csv/model_merging_verification.csv`, `reports/csv/model_merging_stats.csv`, `reports/plots/model_merging_cycle_score_fixed_N.pdf`, config JSON, and `reports/model_merging_verification_report.md`.
  - Verification mode already supports repeated seeds, fixed `N`, bootstrap correlation intervals, observed activation matching, injected pairwise permutation noise, ordinary weight average, greedy soup, Git-ReBasin-style pairwise alignment, C2M3-style synchronization, rank-lift branch ensemble, and ensemble upper bound.

Existing evidence reports:

- `reports/model_merging_verification_report.md`
  - Runs MNIST MLP, `N=3,4`, widths `16,32`, 5 seeds, 3 epochs, 2k train samples, 1k test samples.
  - Current fixed-setting rows have only 5 seeds and mark cycle-score prediction as unsupported or descriptive.
  - Rank lift is correctly labeled "branch ensemble / extra capacity."

- `reports/planted_obstruction_model_merging_report.md`
  - Uses functionally equivalent MNIST MLP copies and planted alignment defects.
  - Supports that planted central `mu_2`-like cycle score tracks Git-ReBasin-style pairwise degradation in that controlled design.
  - Does not show rank lift improves beyond C2M3.

- `reports/fashion_mnist_improved_ladder_report.md` and `reports/fashion_mnist_cnn_channel_gauge_confirmatory_report.md`
  - Show exact ReLU-compatible monomial/channel gauge improvements over faithful internal C2M3-style baselines in limited Fashion-MNIST settings.
  - Greedy soup remains a strong boundary baseline.
  - These are practical gauge-selection results, not direct obstruction-to-degradation verification artifacts.

- `reports/results_narrative_after_35791f7.md` and `reports/paper_level_decision_after_35791f7.md`
  - Current paper framing is conservative: controlled mathematics plus limited exact-gauge practical wins, no broad model-merging, greedy-soup, official-baseline, CIFAR, or real Brauer claim.

### Missing Or Underpowered

The paper still needs a real-model verification layer that saves all raw obstruction artifacts:

- raw pairwise alignments `g_ij`, not only scalar cycle scores;
- activation overlap/statistics used to estimate each `g_ij`;
- triangle defect rows with explicit compositions `c_ijk = g_ij g_jk g_ki`;
- obstruction residuals `Def(c)` at triangle, setting, and method levels;
- confidence intervals for obstruction-to-degradation correlations at fixed settings;
- fixed-setting repeated-seed MNIST and Fashion-MNIST runs with enough seeds;
- capacity-matched rank-lift baselines, because the current rank-lift row is an extra-capacity branch ensemble.

## Proposed Experiment A: Real Obstruction-Degradation Verification

Purpose: test whether real model-merging obstruction residuals predict ordinary merge degradation at fixed dataset/architecture/width/model-count settings.

### New Files

- `experiments/real_obstruction_degradation.py`
  - New high-level experiment script.
  - Reuses training/evaluation utilities from `src.model_merging_benchmark`.
  - Calls new artifact helpers listed below.

- `src/obstruction_artifacts.py`
  - New helper module for serialization and residual accounting.
  - Proposed functions:
    - `permutation_to_json(perm: np.ndarray) -> str`
    - `permutation_from_json(text: str) -> np.ndarray`
    - `pairwise_alignment_rows(pairwise, run_meta) -> list[dict]`
    - `activation_stat_rows(features_by_model, pairwise, run_meta) -> list[dict]`
    - `triangle_defect_rows(pairwise, n_models, width, run_meta) -> list[dict]`
    - `setting_obstruction_summary(triangle_rows) -> dict`
    - `bootstrap_ci(values, seed, n_bootstrap) -> tuple[float, float]`
    - `paired_delta_rows(wide_results, baseline_cols) -> list[dict]`

- `tests/test_obstruction_artifacts.py`
  - Unit tests for permutation serialization, composition, triangle defect identity/nonidentity, and summary statistics.

### Required Functionality

For each setting and seed, save:

- checkpoints:
  - `reports/checkpoints/real_obstruction/{setting_id}/model_{i}.pt`
  - `weight_average.pt`
  - `git_rebasin_pairwise.pt`
  - `c2m3_cycle_consistent.pt`
  - rank-lift and capacity-matched branch baseline checkpoints when enabled.

- pairwise alignment rows:
  - `run_id`, `setting_id`, `dataset`, `architecture`, `n_models`, `width`, `seed`;
  - `i`, `j`;
  - `matching` (`activation` and optionally `weight`);
  - serialized permutation `g_ij_perm`;
  - pairwise activation correlation score;
  - feature overlap stats: mean norm, std norm, mean max similarity, assignment similarity, matched-feature correlation.

- triangle defect rows:
  - `i`, `j`, `k`;
  - serialized `g_ij`, `g_jk`, `g_ki`;
  - serialized `c_ijk_perm`;
  - `cycle_defect_frobenius`;
  - `cycle_defect_hamming`;
  - `fixed_point_fraction`;
  - `is_identity_defect`;
  - `Def_c_triangle = norm(P(c_ijk) - I) / sqrt(2 * width)`.

- setting-level obstruction summaries:
  - mean, max, median, and quantiles of `Def_c_triangle`;
  - nonidentity triangle fraction;
  - synchronization residual from `synchronize_permutations`;
  - number of triangles;
  - bootstrap intervals where applicable.

- method metrics:
  - validation/test accuracy and loss;
  - merge degradation versus best single model and mean individual model;
  - deltas versus weight average, Git-ReBasin-style pairwise alignment, C2M3-style synchronization, greedy soup, rank-lift branch ensemble, and capacity-matched branch baselines;
  - `uses_validation_data`, `is_single_model`, `capacity_matched_to_weight_average`, `inference_multiplier`, and `parameter_multiplier`.

### Main Fixed Settings

Run two main fixed-setting repeated-seed experiments:

1. MNIST MLP:
   - dataset: `mnist`
   - architecture: `mlp`
   - model counts: `N=3,4`
   - widths: `32,64`
   - seeds: `2000..2029` for the main `N=4,width=64` setting; `2010..2029` or 20 seeds for remaining settings if time allows.
   - epochs: `5`
   - train samples: `12000`
   - test samples: full test if feasible, otherwise `5000`.

2. Fashion-MNIST MLP:
   - dataset: `fashion_mnist`
   - architecture: `mlp`
   - model counts: `N=3,4`
   - widths: `32,64,128`
   - seeds: at least 20 for `N=4,width=64`; at least 10 for secondary settings.
   - epochs: `5`
   - train samples: `12000`
   - test samples: full test if feasible, otherwise `5000`.

Use `matching=activation` for the main claim. Add `matching=weight` only as a sensitivity check.

### Commands After Implementation

Main MNIST run:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/codex-pycache MPLCONFIGDIR=/private/tmp/codex-mpl \
.venv/bin/python experiments/real_obstruction_degradation.py \
  --datasets mnist \
  --architecture mlp \
  --model-counts 3,4 \
  --widths 32,64 \
  --main-setting mnist:mlp:4:64 \
  --main-seeds 2000-2029 \
  --secondary-seeds 2030-2049 \
  --epochs 5 \
  --max-train-samples 12000 \
  --max-test-samples 0 \
  --batch-size 128 \
  --matching activation \
  --save-checkpoints \
  --bootstrap-samples 5000 \
  --device auto
```

Main Fashion-MNIST run:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/codex-pycache MPLCONFIGDIR=/private/tmp/codex-mpl \
.venv/bin/python experiments/real_obstruction_degradation.py \
  --datasets fashion_mnist \
  --architecture mlp \
  --model-counts 3,4 \
  --widths 32,64,128 \
  --main-setting fashion_mnist:mlp:4:64 \
  --main-seeds 3000-3029 \
  --secondary-seeds 3030-3049 \
  --epochs 5 \
  --max-train-samples 12000 \
  --max-test-samples 0 \
  --batch-size 128 \
  --matching activation \
  --save-checkpoints \
  --bootstrap-samples 5000 \
  --device auto
```

Sensitivity run:

```bash
.venv/bin/python experiments/real_obstruction_degradation.py \
  --datasets mnist,fashion_mnist \
  --architecture mlp \
  --model-counts 4 \
  --widths 64 \
  --main-seeds 4000-4009 \
  --epochs 5 \
  --max-train-samples 12000 \
  --max-test-samples 5000 \
  --batch-size 128 \
  --matching weight \
  --bootstrap-samples 2000 \
  --device auto
```

### Expected Outputs

- `reports/real_obstruction_degradation_report.md`
- `reports/csv/real_obstruction_degradation.csv`
- `reports/csv/real_obstruction_pairwise_alignments.csv`
- `reports/csv/real_obstruction_triangle_defects.csv`
- `reports/csv/real_obstruction_activation_stats.csv`
- `reports/csv/real_obstruction_summary.csv`
- `reports/csv/real_obstruction_paired_deltas.csv`
- `reports/tables/real_obstruction_degradation_table.tex`
- `reports/tables/real_obstruction_correlation_table.tex`
- `reports/plots/real_obstruction_defect_vs_weight_degradation.pdf`
- `reports/plots/real_obstruction_defect_vs_git_degradation.pdf`
- `reports/plots/real_obstruction_defect_vs_c2m3_delta.pdf`
- `reports/plots/real_obstruction_activation_overlap_vs_defect.pdf`
- `reports/configs/real_obstruction_degradation_config.json`

### Acceptance Criteria

Implemented:

- The script runs without modifying existing benchmark outputs unless explicitly requested.
- All pairwise `g_ij` and triangle `c_ijk` rows are saved.
- Every run row has seed, hyperparameters, checkpoint path, validation/test split metadata, and method capacity metadata.
- Tests verify permutation composition and defect calculations on controlled examples.

Run:

- Main MNIST fixed setting `N=4,width=64` has at least 30 seeds.
- Main Fashion-MNIST fixed setting `N=4,width=64` has at least 30 seeds.
- Secondary settings have at least 10 seeds.
- Individual model accuracy is high enough to avoid plumbing-only interpretation: target `>0.85` MNIST and `>0.75` Fashion-MNIST mean best-individual accuracy.

Supported by data only if:

- At a fixed setting, `Def(c)` versus ordinary merge degradation has Spearman and Pearson intervals with positive lower bound, or the report marks the claim unsupported.
- The association survives at least one secondary fixed setting or is explicitly called setting-specific.
- Injected-alignment rows are kept as sensitivity/negative-control rows and are not used as independent evidence that real residuals predict weight averaging.

## Proposed Experiment B: Capacity-Matched Rank-Lift Baselines

Purpose: separate "rank-lift helps because it uses obstruction structure" from "rank-lift helps because it has more branches or inference capacity."

### Current Gap

The existing `twisted_rank_lift_2` in `experiments/model_merging_benchmark.py` is a branch ensemble and is marked:

- `is_single_model = False`
- `capacity_matched_to_weight_average = False`
- `method_note = rank-lifted branch ensemble, extra capacity`

This is correct, but it means the paper cannot claim a capacity-matched single-model win from current rank-lift rows.

### New Baselines

Add branch-capacity matched baselines to `src/model_merging_benchmark.py` or a new `src/rank_lift_baselines.py`:

- `random_branch_ensemble(aligned_models, n_branches, seed)`
  - Same number of branch models and same inference multiplier as the rank-lift branch ensemble.
  - Randomly partitions aligned models into `B` branches and averages each partition.

- `validation_branch_ensemble(models, val_loader, n_branches)`
  - Same branch count as rank lift.
  - Selects the top `B` individual models by validation accuracy; no obstruction or pairwise defect information.

- `c2m3_cluster_branch_ensemble(aligned_synced, n_branches)`
  - Clusters aligned models using only post-C2M3 distances or validation scores, not triangle defects.

- Optional harder baseline: `distilled_rank_lift_single_model(branches, train_loader, val_loader, architecture, width)`
  - Distills rank-lift branch ensemble into the original architecture.
  - This is capacity matched to weight averaging only if parameter count and inference multiplier are checked and recorded.
  - It should be treated as a separate distillation experiment, not as the primary rank-lift claim.

### Required Outputs

Add rows to `reports/csv/real_obstruction_degradation.csv`:

- `twisted_rank_lift_B`
- `random_branch_ensemble_B`
- `validation_branch_ensemble_B`
- `c2m3_cluster_branch_ensemble_B`
- optional `distilled_twisted_rank_lift_single`
- optional `distilled_random_branch_single`

Each row must include:

- `branch_count`
- `parameter_multiplier`
- `inference_multiplier`
- `capacity_matched_to_rank_lift`
- `capacity_matched_to_weight_average`
- `uses_obstruction_residual`
- `uses_validation_data`

### Acceptance Criteria

Implemented:

- At least one branch-capacity matched non-obstruction baseline exists and is tested.
- The report distinguishes "matched to rank-lift branch capacity" from "matched to one original model."

Run:

- Capacity-matched rank-lift comparisons run on MNIST and Fashion-MNIST main fixed settings.
- Each comparison has at least 20 paired seeds.

Supported by data only if:

- `twisted_rank_lift_B - random_branch_ensemble_B` has positive paired CI lower bound.
- `twisted_rank_lift_B - validation_branch_ensemble_B` has positive paired CI lower bound.
- If only the extra-capacity branch rank lift improves over single-model baselines, the paper must state "extra-capacity branch lift" and not "capacity-matched merge."

## Proposed Experiment C: Controlled Twisted Overlap Benchmark

Purpose: create a controlled neural setting, richer than the current vector toy, where a known twist controls overlap labels and a rank-lifted model has a real reason to help.

### New Files

- `experiments/controlled_twisted_overlap_benchmark.py`
- `src/controlled_twisted_overlaps.py`
- `tests/test_controlled_twisted_overlaps.py`

### Design

Create local charts/tasks indexed by vertices and overlap contexts indexed by triangles. Train or instantiate local MLPs whose hidden features are related by exact sign/permutation gauges. Supply a known central `mu_2` face twist `alpha_ijk`.

Evaluate:

- ordinary weight average;
- cycle-consistent merge when `alpha` is coboundary;
- twisted q=2 branch model using the supplied or learned triangle context;
- random branch ensemble with the same branch count;
- validation-selected branch ensemble with the same branch count;
- ensemble upper bound.

Save:

- local checkpoints;
- overlap datasets;
- true edge gauges;
- observed edge alignments `g_ij`;
- true and observed triangle defects `c_ijk`;
- `Def(c)` residuals;
- branch assignments;
- validation/test metrics;
- bootstrap CIs.

### Commands After Implementation

```bash
.venv/bin/python experiments/controlled_twisted_overlap_benchmark.py \
  --twist-family mu2_coboundary,mu2_nontrivial_h2,random_noncentral \
  --n-models 4 \
  --widths 32,64 \
  --seeds 5000-5029 \
  --epochs 5 \
  --samples-per-chart 2000 \
  --samples-per-overlap 1000 \
  --branch-count 2 \
  --bootstrap-samples 5000 \
  --device auto
```

### Expected Outputs

- `reports/controlled_twisted_overlap_report.md`
- `reports/csv/controlled_twisted_overlap.csv`
- `reports/csv/controlled_twisted_overlap_pairwise.csv`
- `reports/csv/controlled_twisted_overlap_triangles.csv`
- `reports/csv/controlled_twisted_overlap_summary.csv`
- `reports/tables/controlled_twisted_overlap_table.tex`
- `reports/plots/controlled_twisted_overlap_defect_vs_merge_loss.pdf`
- `reports/plots/controlled_twisted_overlap_rank_lift_delta.pdf`
- `reports/configs/controlled_twisted_overlap_config.json`

### Acceptance Criteria

Implemented:

- The script can generate both coboundary and non-coboundary central `mu_2` cases and a noncentral negative control.
- It stores true `alpha_ijk`, observed `c_ijk`, and overlap data identifiers.

Run:

- At least 30 seeds for the main `width=64` setting.
- At least 10 seeds for `width=32`.

Supported by data only if:

- Nontrivial central cases show ordinary/cycle-consistent failure and rank-lift improvement over branch-capacity matched non-obstruction baselines.
- Coboundary cases are solved by cycle-consistent synchronization without needing branch lift.
- Noncentral controls do not get promoted to central-twist claims.

## Proposed Experiment D: Upgrade Existing Verification Mode Or Deprecate It

Purpose: avoid two divergent real-model benchmark scripts.

Two acceptable paths:

1. Extend `experiments/model_merging_benchmark.py --mode verification` to call `src.obstruction_artifacts`.
2. Keep the existing verification script unchanged and mark `experiments/real_obstruction_degradation.py` as the paper-grade successor.

Recommended path: keep existing verification mode as a historical baseline and implement the new paper-grade script separately. This avoids breaking existing reports while allowing richer outputs.

### Required Additions If Extending Existing Verification Mode

- Add flags:
  - `--save-pairwise-alignments`
  - `--save-activation-stats`
  - `--save-triangle-defects`
  - `--rank-lift-capacity-baselines random,validation,c2m3_cluster`
  - `--report-prefix real_obstruction`
- Save the same outputs as Experiment A.

## Proposed Experiment E: Report And Claim-Audit Integration

After Experiments A-C run, update:

- `reports/claims_audit.md`
- `reports/results_narrative_after_35791f7.md` or a successor synthesis file
- `reports/tables/venue_claim_matrix_after_35791f7.tex` only if claims materially change
- `reports/full_capacity_claim_audit.md` only if new methods are implemented

Do not update the paper thesis until data exist.

## Plotting And Tables

New plotting functions can live in `src/plotting.py` or `src/obstruction_artifacts.py`:

- `plot_defect_vs_degradation(summary_csv, output_pdf)`
- `plot_rank_lift_capacity_matched_delta(paired_csv, output_pdf)`
- `plot_activation_overlap_vs_defect(activation_csv, triangle_csv, output_pdf)`
- `write_obstruction_latex_tables(summary_csv, paired_csv, tables_dir)`

Required plots:

- `Def(c)` versus weight-average merge degradation;
- `Def(c)` versus Git-ReBasin-style pairwise merge degradation;
- `Def(c)` versus C2M3 gain/loss;
- rank-lift branch delta versus capacity-matched branch baselines;
- activation overlap/alignment confidence versus triangle defect.

## Claim Rules For The Next Reports

Always separate:

- implemented: code path exists and tests pass;
- run: command executed and outputs exist;
- supported by data: confidence interval or explicitly stated acceptance criterion passes.

Forbidden unless directly supported:

- "obstruction residuals predict ordinary merge degradation" without fixed-setting repeated-seed CIs;
- "rank lift helps" without comparison to capacity-matched branch baselines;
- "rank lift is capacity matched to a single model" unless a same-parameter/inference distilled or compressed model is implemented and verified;
- "real residuals are Brauer/projective/period-index classes";
- "TwistedMerge beats greedy soup";
- "TwistedMerge beats official external baselines";
- "CIFAR confirms the method."

Allowed if supported:

- "In fixed MNIST/Fashion-MNIST MLP settings, permutation obstruction residuals predict ordinary merge degradation."
- "In controlled twisted-overlap settings, rank-lifted branch models improve over ordinary/cycle-consistent merges and branch-capacity matched controls."
- "The detector remains conservative: real MNIST/Fashion-MNIST residuals need not be central/projective to be useful diagnostics."
- "The current rank-lift evidence is extra-capacity unless the capacity-matched branch controls are beaten."

## Minimal Next PR Scope

For the next implementation PR, do only:

1. add `src/obstruction_artifacts.py`;
2. add `tests/test_obstruction_artifacts.py`;
3. add `experiments/real_obstruction_degradation.py`;
4. implement branch-capacity matched baselines in `src/rank_lift_baselines.py`;
5. generate an initial MNIST dry-run with 2 seeds and fake/MNIST small settings to validate output schemas;
6. do not update claims except to say the machinery is implemented and dry-run outputs exist.

Full runs should be a separate PR or commit because they will create many checkpoints and large CSVs.
