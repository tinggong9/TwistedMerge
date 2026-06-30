# Claims Audit

This file tracks which claims are supported by current repository artifacts and which claims remain unsupported.

## Supported

| Claim | Status | Evidence |
| --- | --- | --- |
| TwistedMerge can detect failed gauge synchronization in a controlled `mu_2` central-twist example. | Supported | `tests/test_twisted_merge_algorithm.py` checks that the finite central twist has `status == "failed"` for `q=1` and `gauge.success == False`. |
| `q=2` branch lift can recover perfect prediction in the controlled finite central `mu_2` example. | Supported | `tests/test_twisted_merge_algorithm.py` checks `status == "twisted_rank_lifted"`, `twist_residual == 0`, and twisted zero-one loss `0`. |
| The code separates finite central coboundary twists from the nonzero `H^2(mu_2)` tetrahedral obstruction. | Supported | `tests/test_twisted_merge_algorithm.py` checks the H2 tetrahedral twist is non-coboundary and is not absorbed by the current `TwistedMerge` algorithm. |
| Lifted transition maps are no longer trivial placeholders for the finite central coboundary example. | Supported | `test_lifted_transition_maps_encode_nontrivial_edge_sign` checks that a nontrivial lifted edge uses `rho(-1)` rather than `rho(+1)`. |
| The model-merging benchmark now includes fixed-`N` repeated-seed MNIST checks and controlled injected-alignment negative controls. | Supported | `reports/model_merging_verification_report.md` and `reports/csv/model_merging_verification.csv` cover MNIST MLP, `N=3,4`, widths `16,32`, five seeds, and injected pairwise alignment noise. |
| CIFAR should remain plumbing-only in the current benchmark artifacts. | Supported | `reports/model_merging_verification_report.md` records that prior smoke-run CIFAR individual accuracy maxed at `0.1328`, below the `0.20` threshold for non-plumbing claims. |
| The planted-obstruction benchmark uses functionally equivalent MNIST MLP copies before merging. | Supported | `reports/planted_obstruction_model_merging_report.md` reports mean base accuracy `0.8632`, max copy accuracy std `0`, and max logit disagreement about `2.9e-6`. |
| In the planted central `mu_2` alignment-observation benchmark, planted cycle score predicts pairwise Git-ReBasin merge degradation. | Supported descriptive | `reports/csv/planted_obstruction_stats.csv` reports central `git_rebasin_pairwise` Spearman `0.8741` and monotone mean degradation `0.0000 -> 0.0031 -> 0.0149 -> 0.0387`. |
| Cycle-consistent synchronization fixes the one-edge planted inconsistency in the planted benchmark. | Supported descriptive | `reports/csv/planted_obstruction_stats.csv` reports C2M3 high-defect degradation `0.0000` for both central and random planted families. |
| TwistedMerge++ contains C2M3-style synchronization as the trivial/resolved-residual case. | Supported | `tests/test_twisted_merge_plus.py` checks zero defects select `untwisted_c2m3`; `reports/twisted_merge_plus_report.md` shows C2M3 selected for a C2M3-fixable one-edge permutation outlier. |
| TwistedMerge++ distinguishes C2M3-fixable permutation noise from central/twist residuals. | Supported | `tests/test_twisted_merge_plus.py` checks one-edge permutation noise is classified as `edge_outlier_or_noise`, while the finite sign residual is classified as `central_coboundary`. |
| TwistedMerge++ activates lifted maps only for finite central coboundary residuals in the current demo/tests. | Supported | `tests/test_twisted_merge_plus.py` checks the central coboundary case builds nontrivial `rho(beta_ij) tensor G_ij` maps and random/noncentral plus nonzero-H2 cases do not build lifted maps. |
| Finite torsion/projective defects of order `d` have an explicit rank-`d` absorption in the clock-shift toy model. | Supported | `tests/test_finite_index_twists.py` checks the clock/shift relation, and `reports/csv/finite_index_twist_summary.csv` reports minimal success rank equals `d` for primitive and nonprimitive cases. |
| Ranks not divisible by the torsion order `d` are excluded by the determinant obstruction in the finite-index toy model. | Supported | `tests/test_finite_index_twists.py` checks determinant rejection for nondivisible ranks; `reports/csv/finite_index_twist_absorption.csv` preserves failed ranks with nonzero residuals. |
| The finite-index experiment realizes a period/index threshold in a controlled algebraic setting. | Supported | `reports/finite_index_twist_report.md` proves the determinant obstruction and reports success exactly when `d` divides candidate rank `r`. |
| The central period-index benchmark realizes controlled examples with period `d` and index `d^k`. | Supported | `tests/test_period_index_central.py` checks the k-pair Heisenberg relations and index growth; `reports/csv/period_index_central_summary.csv` reports minimal success rank equals `d^k`. |
| Period divisibility alone is not enough in the k-pair central/projective benchmark. | Supported | `tests/test_period_index_central.py` checks `d=3,k=2` ranks `3` and `6` fail while rank `9` succeeds; `reports/period_index_central_report.md` reports the same explicit example. |
| TwistedMerge++ detects controlled period-index central projective systems with index larger than period. | Supported | `tests/test_twisted_merge_plus_period_index.py` checks `d=2,k=2`, `d=2,k=3`, and `d=3,k=2`; `reports/twisted_merge_plus_period_index_report.md` reports period `d` and index `d^k` rows. |
| TwistedMerge++ rejects candidate ranks divisible by period but not by index in controlled period-index cases. | Supported | `tests/test_twisted_merge_plus_period_index.py` checks rank `2` fails for `d=2,k=2`, ranks `2,4` fail for `d=2,k=3`, and ranks `3,6` fail for `d=3,k=2`. |
| TwistedMerge++ activates a period-index projective/Morita lift exactly when the computed index divides the candidate rank in controlled cases. | Supported | `tests/test_twisted_merge_plus_period_index.py` checks rank `4` succeeds for `d=2,k=2`, rank `8` succeeds for `d=2,k=3`, and rank `9` succeeds for `d=3,k=2`; the demo selects `period_index_projective_morita_lift` only on index-divisible rows. |
| The TwistedMerge++ period-index detector reduces to the previous finite-index threshold for one Heisenberg pair. | Supported | `test_k1_reduces_to_finite_index_detector` compares k=1 period-index decisions against `evaluate_rank_absorption`; scenario `d2_k1_rank2` reports period and index both equal `2`. |
| TwistedMerge++ detects finite-index projective residuals in controlled clock-shift examples. | Supported | `tests/test_twisted_merge_plus_finite_index.py` and `reports/twisted_merge_plus_finite_index_report.md` detect order-3 and order-2 scalar projective phases. |
| TwistedMerge++ rejects candidate lift ranks not divisible by the detected order. | Supported | `test_order3_insufficient_rank_is_obstructed` checks order `3`, rank `2` is classified as `finite_index_projective_obstructed`; scenario E1 reports the same. |
| TwistedMerge++ activates a finite-index projective lift when the rank threshold is met. | Supported | `test_order3_sufficient_rank_activates_projective_lift` and scenarios E2-E4 select `finite_index_projective_lift` with near-zero lift residual. |
| TwistedMerge++ still gives C2M3/edge-outlier handling priority over finite-index language when synchronization diagnostics explain the issue. | Supported | `test_c2m3_resolved_case_keeps_c2m3_priority` and scenario E6 select `c2m3_cycle_consistent` with no detected finite-index order. |
| The finite-index residual miner correctly identifies clock-shift positive controls. | Supported | `tests/test_finite_index_residual_mining.py` checks exact clock-shift controls, and `reports/finite_index_residual_mining_report.md` reports detected orders `2`, `3`, and `4`. |
| Real MNIST activation-permutation residuals were mined for finite-index scalar/projective structure. | Supported | `reports/csv/finite_index_residual_mining.csv` contains 50 real MNIST triangle residual rows from `N=3,4`, widths `16,32`, five seeds, plus positive controls. |
| The default MNIST residual-mining run found no strict/medium finite-index scalar candidates. | Supported negative result | `reports/finite_index_residual_mining_report.md` reports real MNIST candidate fraction `0.0000` at strict, medium, and loose thresholds; mean centrality is about `0.9620`. |
| The detector distinguishes central finite-index projective residuals from noncentral permutation holonomy. | Supported | `tests/test_noncentral_holonomy.py` and `reports/noncentral_holonomy_ladder_report.md` classify clock-shift controls as `central_finite_index_projective` and the `S_3` commutator as `noncentral_permutation_holonomy`. |
| Controlled `S_3` permutation commutators are noncentral and not Brauer/projective scalar classes. | Supported | `test_s3_commutator_is_noncentral` verifies the `(12),(23)` commutator is a noncentral 3-cycle; `reports/noncentral_vs_brauer_note.tex` records the paper-ready example. |
| Sampled real MNIST permutation residuals are better described as noncentral permutation holonomy than finite-index scalar twists in the current mined data. | Supported negative result | `reports/noncentral_holonomy_ladder_report.md` samples the ten most finite-index-like real MNIST residual rows and labels all as `noncentral_permutation_holonomy` with `not_brauer_noncentral`. |
| StructureGroupLadder distinguishes permutation, signed, monomial/projective, block, and GL residual diagnostics. | Supported | `tests/test_structure_group_ladder.py` exercises the ladder levels; `reports/csv/structure_group_ladder_mining.csv` contains one row per level for synthetic controls and real MNIST triangles. |
| Controlled signed or monomial examples can reveal central `mu_2` or finite-index projective residuals. | Supported | `reports/structure_group_ladder_report.md` reports `signed_mu2_central` as `central_mu2_candidate`, order-3 clock-shift rank 2 as obstructed, and rank 3 as `finite_index_projective_lift`. |
| Controlled noncentral examples remain noncentral and are not mislabeled as Brauer/projective by the ladder. | Supported | `tests/test_structure_group_ladder.py` checks the `S_3` and GL controls are noncentral; the ladder report labels them `noncentral_permutation_holonomy` and `gl_noncentral_holonomy`. |
| Real MNIST residuals were tested across permutation, signed, monomial, low-rank GL, and block-orthogonal structure levels. | Supported negative result | `reports/csv/structure_group_ladder_summary.csv` covers signed/monomial/GL rows, while `reports/csv/block_orthogonal_ladder_summary.csv` covers block sizes `2,4,8`; both report zero real central/projective candidates. |
| The actionable ladder merge benchmark was run on MNIST MLPs. | Supported | `reports/structure_group_ladder_merge_report.md` and `reports/csv/structure_group_ladder_merge_benchmark.csv` cover MNIST ReLU MLP, `N=3,4`, widths `16,32`, five seeds, and methods including weight average, greedy soup, C2M3, signed, monomial scale, GL diagnostic, and ensemble. |
| Monomial positive scaling was evaluated as an exact ReLU reparameterization before averaging. | Supported | `src/ladder_merge_methods.py` applies positive hidden-unit scaling with inverse outgoing adjustment; `reports/structure_group_ladder_merge_report.md` labels `monomial_scale` as `exact_relu_positive_scale_symmetry`. |
| In the current MNIST ladder merge benchmark, monomial scaling gives a descriptive mean gain over C2M3 but not over greedy soup. | Supported descriptive | `reports/csv/structure_group_ladder_merge_summary.csv` reports monomial mean accuracy deltas versus C2M3 of `0.0090` to `0.0160`, while deltas versus greedy soup are negative in all fixed settings. |
| On the validated MNIST MLP benchmark, the validation-selected ladder selector improves over the repo's internal C2M3-style baseline as a single-model capacity-matched method. | Supported limited | `reports/csv/validated_ladder_merge_summary.csv` reports `validated_ladder_selector_vs_c2m3_permutation` paired mean accuracy delta `0.0088`, bootstrap CI `[0.0059, 0.0121]`, wins/ties/losses `71/28/21`, and positive mean deltas in all six fixed settings. |
| On the validated MNIST MLP benchmark, monomial positive scaling improves over the repo's internal C2M3-style baseline as an exact ReLU reparameterization. | Supported limited | `reports/csv/validated_ladder_merge_summary.csv` reports `monomial_scale_vs_c2m3_permutation` paired mean accuracy delta `0.0059`, bootstrap CI `[0.0020, 0.0098]`, wins/ties/losses `78/1/41`, and positive mean deltas in five of six fixed settings. |
| The validated ladder selector avoids test-set selection leakage in the current benchmark implementation. | Supported | `reports/validated_ladder_merge_report.md` records that selection uses validation accuracy/loss only; `reports/csv/validated_ladder_merge_summary.csv` reports `selector_no_test_leakage = True` for all selector behavior rows. |
| Monomial-scaled greedy soup does not improve over ordinary greedy soup in the validated MNIST MLP run. | Supported negative result | `reports/csv/validated_ladder_merge_summary.csv` reports `monomial_scaled_greedy_soup_vs_greedy_soup` paired mean accuracy delta `-0.0002`, CI `[-0.0007, 0.0002]`, and wins/ties/losses `7/106/7`. |
| The current monomial centrality-improvement diagnostic does not strongly or consistently predict monomial accuracy gain. | Supported negative result | `reports/csv/validated_ladder_merge_summary.csv` reports overall Pearson/Spearman correlations `0.1893/0.2868`, with mixed fixed-setting correlations including negative values and one fixed setting with negative mean monomial accuracy delta. |
| Block-orthogonal synthetic controls are implemented and distinguish recovered rotations, noncentral block holonomy, and scalar block phases. | Supported | `tests/test_block_gauge_alignment.py` checks rotation recovery, noncentral block commutators, and scalar `-I` block phase detection. |
| Real block-orthogonal diagnostics were evaluated on MNIST MLPs. | Supported | `reports/block_orthogonal_ladder_report.md` and `reports/csv/block_orthogonal_ladder.csv` cover MNIST ReLU MLP, `N=3,4`, widths `16,32`, block sizes `2,4,8`, and five seeds. |
| In the current real MNIST block-orthogonal run, block gauges do not reduce residual centrality on average and do not produce scalar/projective candidates. | Supported negative result | `reports/csv/block_orthogonal_ladder_summary.csv` reports real block mean centrality improvements from permutation to block of `-0.0050`, `-0.0090`, and `-0.0100`, and central/projective candidate fraction `0.0000` for block sizes `2,4,8`. |
| Block-orthogonal merge performance was not evaluated for the ReLU MLP run. | Supported negative result | `reports/block_orthogonal_ladder_report.md` labels block-orthogonal rotations as feature-space diagnostics, and `reports/csv/block_orthogonal_ladder.csv` preserves `merge_evaluated = False`. |
| Global block synchronization recovers planted globally consistent block gauges. | Supported | `tests/test_global_block_synchronization.py` checks exact recovery for planted gauges, and `reports/global_block_synchronization_report.md` reports zero synthetic connection residual for `planted_recoverable_block_rotations`. |
| Global block synchronization projects block maps to cycle-consistent gauges while preserving an explicit connection-residual honesty check. | Supported descriptive | `reports/csv/global_block_synchronization_summary.csv` reports zero post-projection cycle/centrality for global rows and nonzero real connection residuals, so this is diagnostic projection rather than a proof of exact real-data descent. |
| Learned block partitions are implemented deterministically for activation-correlation and output-weight similarity choices. | Supported | `tests/test_learned_block_partition.py` checks deterministic clustering and required inputs; `reports/configs/global_block_synchronization_config.json` records enabled partition methods. |
| The current global/learned block MNIST run finds no scalar finite-order projective candidates. | Supported negative result | `reports/global_block_synchronization_report.md` reports real MNIST central/projective candidate fraction `0.0000` across permutation, monomial, low-rank GL, pairwise block, and global block diagnostic rows. |
| Learned block partitions do not improve observed pairwise-block centrality over contiguous blocks in the current MNIST run. | Supported negative result | `reports/global_block_synchronization_report.md` states learned partitions do not reduce mean observed pairwise-block centrality versus contiguous; `reports/csv/global_block_synchronization_summary.csv` records contiguous block size 2 mean centrality `0.5155`, lower than activation-correlation `0.6295` and output-weight similarity `0.5478`. |

## Not Yet Supported

| Claim | Status | Reason |
| --- | --- | --- |
| TwistedMerge beats external model-merging baselines. | Not yet supported | No external Git Re-Basin, C2M3, Model Soups, RegMean, TIES, or mergekit/MergeBench implementation has been run. |
| TwistedMerge solves natural MNIST/CIFAR merging. | Not yet supported | `reports/model_merging_verification_report.md` is stronger for MNIST than the smoke run, but it still reports descriptive prototype baselines and excludes CIFAR as near chance. |
| TwistedMerge fully trivializes a nonzero `H^2(mu_2)` class as an ordinary untwisted vector bundle. | Not yet supported | The nonzero tetrahedral `H^2` class is explicitly non-coboundary. Current `TwistedMerge` does not construct an edge-level untwisted descent for it. |
| The branch-prediction lift is a complete transition-map-level twisted sheaf descent implementation. | Not yet supported | The q=2 branch result is a controlled prediction-level sanity check. It is not a proof of full sheaf-level descent in the non-coboundary case. |
| Cycle obstruction score predicts weight-average merge degradation beyond the trivial number-of-models confound. | Not yet supported | In `reports/csv/model_merging_stats.csv`, fixed-`N` observed correlations are marked unsupported: `N=3` Pearson `-0.0347`, `N=4` Pearson `-0.3622`, and bootstrap intervals cross zero. |
| C2M3-style alignment or TwistedMerge/rank-lifted branching gives a statistically meaningful improvement over weight averaging on MNIST. | Not yet supported | Mean deltas in `reports/csv/model_merging_stats.csv` are descriptive; fixed setting rows have only five seeds and no significance claim. |
| TwistedMerge/rank-lifted branching beats greedy soup as a single-model baseline. | Not yet supported | `reports/model_merging_verification_report.md` labels `twisted_rank_lift_2` as a branch ensemble with extra capacity; fixed-`N` deltas versus greedy soup are mixed or negative. |
| The planted obstruction score generally predicts every kind of model-merging degradation. | Not yet supported | In `reports/planted_obstruction_model_merging_report.md`, weight averaging is constant across planted levels because it does not use alignments, and the random noncentral Git-ReBasin trend is weaker than the central trend. |
| TwistedMerge/rank-lifted branching adds benefit beyond C2M3 in the planted model-merging benchmark. | Not yet supported | `reports/csv/planted_obstruction_stats.csv` reports rank-lift accuracy delta versus C2M3 is `0.0000` for central and random planted families. |
| Rank-lift helps only when the planted defect is central/twist-like in the model-merging benchmark. | Not yet supported | The planted benchmark shows no rank-lift gain beyond C2M3 for either central or random defects, so central selectivity is not established. |
| TwistedMerge++ beats C2M3 on broad real MNIST/CIFAR model merging. | Not yet supported | `reports/validated_ladder_merge_report.md` supports a limited MNIST MLP win over the repo's internal C2M3-style baseline only; it is not CIFAR, not external C2M3, and not a broad architecture/dataset claim. |
| TwistedMerge++ beats an external C2M3 implementation. | Not yet supported | The validated benchmark uses the repository's internal C2M3-style permutation synchronization baseline, not the external C2M3 codebase. |
| TwistedMerge++ beats greedy soup as a single-model baseline. | Not yet supported | `reports/csv/validated_ladder_merge_summary.csv` reports `validated_ladder_selector_vs_greedy_soup` paired mean accuracy delta `-0.0328`, CI `[-0.0383, -0.0274]`, and wins/ties/losses `7/1/112`. |
| Monomial-scaled greedy soup improves over ordinary greedy soup. | Not yet supported | `reports/csv/validated_ladder_merge_summary.csv` reports no positive paired mean accuracy delta and a confidence interval crossing zero. |
| Monomial centrality improvement is a reliable predictor of monomial accuracy gain. | Not yet supported | `reports/csv/validated_ladder_merge_summary.csv` shows only weak overall correlation and inconsistent fixed-setting correlations. |
| Block-orthogonal alignment improves C2M3 merge accuracy on real MNIST/CIFAR. | Not yet supported | `reports/block_orthogonal_ladder_report.md` does not evaluate block-orthogonal merge performance for ReLU MLPs because general block rotations are not exact ReLU parameter symmetries. |
| TwistedMerge++ solves natural model merging. | Not yet supported | The current TwistedMerge++ artifacts test residual classification and selection logic only. Natural MNIST/CIFAR claims remain governed by `reports/model_merging_verification_report.md`. |
| TwistedMerge++ trivializes a nonzero `H^2(mu_2)` class as an ordinary vector bundle. | Not yet supported | `tests/test_twisted_merge_plus.py` and `reports/twisted_merge_plus_report.md` label the nonzero tetrahedral class as branch-only extra-capacity behavior, not ordinary untwisted descent. |
| TwistedMerge++ rank-lift gives a capacity-matched single-model improvement. | Not yet supported | The branch path is explicitly labeled `branch_lift_extra_capacity`; no capacity-matched single-model comparison has been run. |
| Real neural model-merging defects have the exact finite-index clock-shift form. | Not yet supported | `reports/finite_index_twist_report.md` is a controlled algebraic toy experiment, not a learned MNIST/CIFAR defect identification result. |
| Real neural model-merging defects have the higher period-index form `period=d, index=d^k`. | Not yet supported | `reports/period_index_central_report.md` is a controlled finite Heisenberg benchmark and explicitly does not claim MNIST/CIFAR residuals are Brauer/projective classes. |
| TwistedMerge++ beats C2M3 because of period-index detection. | Not yet supported | `reports/twisted_merge_plus_period_index_report.md` is a controlled selector demo; it does not run natural MNIST/CIFAR model merging or compare against C2M3 performance. |
| Every torsion cohomology class in the paper's broad setting is trivialized on the original cover. | Not yet supported | The finite-index report says the defect is absorbed by a finite-rank projective/Morita lift and explicitly not that the original class vanishes. |
| Branch/projective finite-index lift is a capacity-matched single merged model. | Not yet supported | `reports/finite_index_twist_report.md` labels the branch/projective proxy as extra capacity. |
| Pure permutation C2M3 residuals are the same as scalar finite-index twists. | Not yet supported | `reports/finite_index_residual_mining_report.md` shows MNIST permutation residuals are generally noncentral and do not pass scalar finite-index thresholds. |
| Real neural defects are Brauer classes. | Not yet supported | `reports/noncentral_holonomy_ladder_report.md` explicitly separates noncentral permutation holonomy from central Brauer/projective candidates. |
| Noncentral regular branch lifts are capacity-matched single merged models. | Not yet supported | `src/noncentral_holonomy.py` labels the regular representation construction `noncentral_regular_branch_lift_extra_capacity`; no compression/capacity-matched result is provided. |
| Enlarging the structure group automatically reveals finite-index torsion in real models. | Not yet supported | `reports/structure_group_ladder_report.md` and `reports/block_orthogonal_ladder_report.md` test signed, monomial, low-rank GL, and block-orthogonal diagnostics on real MNIST and still report zero central/projective candidates. |
| Signed or full-GL transforms are exact single-model merges for ReLU MLPs. | Not yet supported | `reports/structure_group_ladder_merge_report.md` labels signed permutation as `heuristic_relu_not_exact` and low-rank GL as `diagnostic_not_single_model_for_relu`. |
| Block-orthogonal ReLU transforms are exact single-model symmetries. | Not yet supported | `reports/block_orthogonal_ladder_report.md` explicitly treats block rotations as feature-space diagnostics for ReLU MLPs, not exact parameter symmetries. |
| Block-orthogonal gauges reveal Brauer/projective classes in real neural residuals. | Not yet supported | `reports/csv/block_orthogonal_ladder_summary.csv` reports zero real central/projective block candidates for all tested block sizes. |
| Global or learned block synchronization turns real MNIST block gauges into an exact same-architecture ReLU merge. | Not yet supported | `reports/global_block_synchronization_report.md` explicitly marks block-orthogonal rows as diagnostics only and preserves ReLU-compatible accuracy reporting for C2M3 permutation, positive monomial scaling, greedy soup, and ensemble baselines. |
| Learned block discovery improves real MNIST residual diagnostics over contiguous blocks. | Not yet supported | The current run reports the opposite for observed pairwise-block centrality, so any learned-block improvement claim needs new evidence. |
| Global block synchronization reveals Brauer/projective classes in real neural residuals. | Not yet supported | The global block run reports zero scalar/projective candidates and uses the connection residual rather than post-projection cycle score as the real-data honesty check. |

## Current Artifact Map

| Artifact | Role |
| --- | --- |
| `tests/test_twisted_merge_algorithm.py` | Regression tests for TwistedMerge success/failure modes. |
| `reports/twisted_merge_algorithm_verification.md` | Verification report for the hardened prototype. |
| `reports/twisted_merge_algorithm_report.md` | Demo report for the prototype algorithm. |
| `reports/synthetic_obstruction_report.md` | Separate nonzero `H^2(mu_2)` obstruction witness. |
| `reports/model_merging_report.md` | Small MNIST/CIFAR model-merging benchmark smoke report. |
| `reports/model_merging_verification_report.md` | Fixed-`N`, repeated-seed MNIST model-merging verification report. |
| `reports/csv/model_merging_verification.csv` | Per-baseline rows for observed and injected-alignment verification settings. |
| `reports/csv/model_merging_stats.csv` | Correlations, bootstrap intervals, deltas, and negative-result labels for verification settings. |
| `reports/planted_obstruction_model_merging_report.md` | Causally planted cycle-obstruction model-merging report using exact hidden-permutation MNIST MLP copies. |
| `reports/csv/planted_obstruction_model_merging.csv` | Per-seed planted central/random obstruction benchmark rows. |
| `reports/csv/planted_obstruction_stats.csv` | Trend and method-delta statistics for the planted obstruction benchmark. |
| `src/twisted_merge_plus.py` | TwistedMerge++ residual classifier and merge selector prototype. |
| `tests/test_twisted_merge_plus.py` | Regression tests for TwistedMerge++ classifications and lifted-map activation. |
| `experiments/twisted_merge_plus_demo.py` | Fast synthetic demo/report generator for TwistedMerge++. |
| `reports/twisted_merge_plus_report.md` | TwistedMerge++ verification report with scenario table and negative-result boundaries. |
| `reports/csv/twisted_merge_plus_demo.csv` | Per-scenario TwistedMerge++ selector diagnostics. |
| `src/finite_index_twists.py` | Clock-shift, determinant obstruction, direct-sum lift, and finite torsion metadata utilities. |
| `tests/test_finite_index_twists.py` | Regression tests for finite-index rank threshold behavior. |
| `experiments/finite_index_twist_absorption.py` | Finite-index torsion/projective twist absorption experiment generator. |
| `reports/finite_index_twist_report.md` | Report proving and testing the period/index rank threshold. |
| `reports/finite_index_twist_theorem.tex` | LaTeX appendix snippet for the determinant obstruction theorem. |
| `reports/csv/finite_index_twist_absorption.csv` | Per-rank finite-index threshold sweep. |
| `reports/csv/finite_index_twist_summary.csv` | Per-case finite-index threshold summary. |
| `src/period_index_central.py` | k-pair finite Heisenberg central period-index generators, relation checks, and rank-obstruction utilities. |
| `tests/test_period_index_central.py` | Regression tests for period `d`, index `d^k`, period-only failures, direct-sum lifts, and no same-cover trivialization claim. |
| `experiments/period_index_central_benchmark.py` | Central period-index benchmark generator over `d=2,3,4` and `k=1,2,3` cases. |
| `reports/period_index_central_report.md` | Report for controlled central/projective period-index evidence and negative boundaries. |
| `reports/period_index_central_theorem.tex` | LaTeX theorem snippet for the k-pair finite Heisenberg period-index benchmark. |
| `reports/csv/period_index_central_benchmark.csv` | Per-rank central period-index threshold sweep. |
| `reports/csv/period_index_central_summary.csv` | Per-case central period-index summary with period-only obstructed ranks. |
| `src/period_index_detector.py` | Conservative detector for scalar central commutators, independent Heisenberg pairs, and period-index rank decisions. |
| `tests/test_twisted_merge_plus_period_index.py` | Regression tests for TwistedMerge++ period-index success, period-only obstruction, unknown-index central cases, noncentral rejection, and k=1 reduction. |
| `experiments/twisted_merge_plus_period_index_demo.py` | Demo generator for TwistedMerge++ period-index selector behavior. |
| `reports/twisted_merge_plus_period_index_report.md` | Scenario report for TwistedMerge++ period-index integration and negative boundaries. |
| `reports/csv/twisted_merge_plus_period_index_demo.csv` | Per-scenario TwistedMerge++ period-index diagnostics. |
| `tests/test_twisted_merge_plus_finite_index.py` | Regression tests for TwistedMerge++ finite-index projective classifications. |
| `experiments/twisted_merge_plus_finite_index_demo.py` | Demo generator for TwistedMerge++ finite-index selector behavior. |
| `reports/twisted_merge_plus_finite_index_report.md` | Scenario report for finite-index TwistedMerge++ integration. |
| `reports/csv/twisted_merge_plus_finite_index_demo.csv` | Per-scenario finite-index TwistedMerge++ diagnostics. |
| `experiments/mine_finite_index_residuals.py` | Mines real MNIST activation-permutation residuals for scalar finite-index structure. |
| `tests/test_finite_index_residual_mining.py` | Unit tests for residual-mining detector helpers and negative controls. |
| `reports/finite_index_residual_mining_report.md` | Report for positive controls and real MNIST finite-index residual mining. |
| `reports/csv/finite_index_residual_mining.csv` | Per-triangle residual-mining rows. |
| `reports/csv/finite_index_residual_mining_summary.csv` | Threshold sensitivity and grouped mining summary. |
| `reports/plots/finite_index_residual_phase_histogram.pdf` | Phase-angle histogram for mined residuals and controls. |
| `src/noncentral_holonomy.py` | Helpers for permutation commutators, scalar centrality, noncentral holonomy classification, and regular branch-lift labeling. |
| `tests/test_noncentral_holonomy.py` | Regression tests separating central finite-index projective defects from noncentral permutation/matrix holonomy. |
| `experiments/noncentral_holonomy_ladder.py` | Controlled ladder experiment comparing clock-shift, `S_3`, GL, and mined MNIST residual samples. |
| `reports/noncentral_holonomy_ladder_report.md` | Report distinguishing central Brauer/projective candidates from noncentral holonomy. |
| `reports/noncentral_vs_brauer_note.tex` | Paper-ready note on why noncentral permutation holonomy is not a scalar Brauer/projective class. |
| `reports/csv/noncentral_holonomy_ladder.csv` | Per-example ladder classifications and MNIST residual sample interpretations. |
| `src/structure_group_ladder.py` | Structure-group ladder diagnostics over permutation, signed, monomial, block, and low-rank GL levels. |
| `tests/test_structure_group_ladder.py` | Regression tests for C2M3 priority, signed `mu_2`, finite-index projective, noncentral, and real-row no-overclaim behavior. |
| `experiments/structure_group_ladder_mining.py` | Synthetic controls and MNIST MLP residual re-mining across the implemented ladder levels. |
| `reports/structure_group_ladder_report.md` | Report for structure-group ladder controls and real MNIST ladder mining. |
| `reports/csv/structure_group_ladder_mining.csv` | Per-triangle, per-level ladder diagnostics for controls and MNIST. |
| `reports/csv/structure_group_ladder_summary.csv` | Summary statistics by source and structure-group level. |
| `src/ladder_merge_methods.py` | Exact ReLU positive scaling transform, heuristic signed transform, and method metadata for ladder merge benchmarking. |
| `experiments/structure_group_ladder_merge_benchmark.py` | Actionable MNIST MLP benchmark comparing weight average, greedy soup, C2M3, signed, monomial scale, GL diagnostic, and ensemble. |
| `reports/structure_group_ladder_merge_report.md` | Performance report for actionable structure-group ladder merge methods. |
| `reports/csv/structure_group_ladder_merge_benchmark.csv` | Per-setting, per-method ladder merge performance rows. |
| `reports/csv/structure_group_ladder_merge_summary.csv` | Fixed-setting summary statistics and deltas versus C2M3, weight averaging, and greedy soup. |
| `src/block_gauge_alignment.py` | Contiguous block partitioning, orthogonal Procrustes, and block transition-map estimation helpers. |
| `tests/test_block_gauge_alignment.py` | Regression tests for block rotation recovery, noncentral block holonomy, scalar block phase detection, and real-row overclaim prevention. |
| `experiments/block_orthogonal_ladder_experiment.py` | MNIST block-orthogonal ladder diagnostic experiment generator. |
| `reports/block_orthogonal_ladder_report.md` | Report for synthetic block controls and real MNIST block-orthogonal diagnostics. |
| `reports/csv/block_orthogonal_ladder.csv` | Per-triangle, per-level block-orthogonal ladder rows. |
| `reports/csv/block_orthogonal_ladder_summary.csv` | Summary by source, block size, and structure-group level. |
| `src/global_block_synchronization.py` | Connection-Laplacian global block-gauge synchronization and residual diagnostics. |
| `src/learned_block_partition.py` | Deterministic contiguous, activation-correlation, and output-weight-similarity block partition helpers. |
| `tests/test_global_block_synchronization.py` | Regression tests for planted global gauges, noisy projection residuals, noncentral holonomy rejection, and scalar block phase detection. |
| `tests/test_learned_block_partition.py` | Regression tests for deterministic learned block clustering and input validation. |
| `experiments/global_block_synchronization_experiment.py` | Synthetic controls and MNIST diagnostic experiment for global/learned block synchronization. |
| `reports/global_block_synchronization_report.md` | Report for synthetic controls, real MNIST global block diagnostics, learned-block comparison, and negative boundaries. |
| `reports/csv/global_block_synchronization.csv` | Per-setting global block synchronization diagnostics. |
| `reports/csv/global_block_synchronization_summary.csv` | Grouped diagnostic summary for permutation, monomial, GL, pairwise block, and global block rows. |
| `reports/configs/global_block_synchronization_config.json` | Saved configuration and environment metadata for the global block synchronization run. |
| `experiments/validated_ladder_merge_benchmark.py` | Validated MNIST MLP benchmark for C2M3, monomial scaling, validation-selected ladder merge, greedy soup variants, and ensemble. |
| `reports/validated_ladder_merge_report.md` | Report with paired statistics, selector behavior, residual correlations, and claim decisions for the validated ladder benchmark. |
| `reports/csv/validated_ladder_merge_benchmark.csv` | Per-setting validated ladder merge benchmark rows. |
| `reports/csv/validated_ladder_merge_summary.csv` | Method summaries, paired comparisons, selector behavior, residual correlations, and claim decisions. |
| `reports/plots/validated_ladder_merge_delta_vs_c2m3.pdf` | Scatter plot of monomial centrality improvement versus monomial accuracy delta over C2M3. |
