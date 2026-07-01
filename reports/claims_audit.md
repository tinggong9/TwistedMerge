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
| The fixed-setting verification script implements the stronger repeated-seed obstruction-correlation gate for real small neural networks. | Supported implementation | `experiments/model_merging_fixed_setting_verification.py` writes fixed-setting run, statistics, triangle-defect, and individual-model CSVs plus plots/report; claims remain gated by `n_rows >= 20` observed rows and bootstrap CIs. |
| ReLU-compatible monomial gauge alignment is implemented for one-hidden-layer MLPs. | Supported implementation | `src/monomial_gauge_alignment.py` estimates activation/weight permutation plus positive hidden-unit scales, applies inverse outgoing classifier scaling, computes monomial triangle defects, and `tests/test_monomial_gauge_alignment.py` checks exact function preservation and scale-defect detection. |
| The earlier small-CNN CIFAR probes were plumbing-only, but the bounded rescue now clears the CIFAR meaningful-accuracy gate. | Supported limited | `reports/model_merging_verification_report.md` records that prior smoke-run CIFAR individual accuracy maxed at `0.1328`, and `reports/cifar_or_colored_mnist_feasibility.md` records a gated CIFAR probe test accuracy of `0.2480`; `reports/cifar_rescue_or_no_go_report.md` records max individual accuracy `0.6583`, above the `0.60` meaningful threshold. |
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
| TwistedMerge++ estimates period-index thresholds from a central commutator matrix, not only labeled Heisenberg pairs. | Supported | `tests/test_period_index_commutator_matrix.py` checks shuffled generator names and TwistedMerge++ integration; `reports/period_index_commutator_matrix_report.md` reports `detector_mode = commutator_matrix` rows. |
| Rank-deficient central commutator forms produce a smaller certified index than the number of supplied generators would naively suggest. | Supported | `test_rank_deficient_form_has_smaller_index` checks four generators with one active pair have alternating rank `2` and index `3`; the report includes `rank_deficient_d3_four_generators`. |
| The commutator-matrix detector rejects ranks divisible by period but not by certified index. | Supported | `reports/period_index_commutator_matrix_report.md` reports `heisenberg_d2_k2_rank2` and `heisenberg_d3_k2_rank3` as `period_divisible_index_obstructed` with selected method `none`. |
| Noncentral commutators are rejected by the commutator-matrix detector. | Supported | `test_noncentral_commutator_rejected` and the report's `noncentral_control` row return `not_central_projective` rather than a Brauer/projective lift. |
| Unsupported central commutator cases with unknown index are not overclaimed. | Supported | `test_unknown_index_not_overclaimed` and the report's `mixed_period_common_d12_unknown` row return `central_projective_index_unknown` with selected method `none`. |
| The robust period-index detector recovers certified period/index data under small controlled noise. | Supported | `tests/test_robust_period_index_detector.py` checks small unitary and entrywise projected-unitary noise; `reports/robust_period_index_detector_report.md` reports certified small-noise Heisenberg rows. |
| The robust period-index detector marks loose noisy central/projective candidates uncertain or rejects large-noise cases instead of lifting them. | Supported | `test_medium_noise_uncertain_not_lifted` and `test_large_noise_rejected` check no lift is selected; `reports/csv/robust_period_index_detector.csv` reports uncertain rows with `selected_method = none` and rejected rows as noncentral. |
| Period-divisible but index-obstructed ranks remain rejected under certified noisy period-index detection. | Supported | `test_period_divisible_index_obstructed_under_noise` checks noisy `d=3,k=2` ranks `3` and `6`; the robust report's rank-divisibility table keeps both rows at `selected_method = none`. |
| Automatic period-index generator mining works on a synthetic loop-holonomy system. | Supported | `test_generator_mining_on_synthetic_loops` and the robust report's `synthetic_loop_mining_d2_k2` row recover period `2`, index `4`, and a certified mined candidate. |
| Robust period-index detection rejects noncentral controls. | Supported | `test_noncentral_control_rejected_even_with_low_noise` checks permutation and random-GL controls; the robust report's noncentral table labels all controls `not_central_projective`. |
| Robust period-index detection has calibrated certification/uncertainty/rejection behavior over multiple controlled noise seeds. | Supported | `experiments/robust_period_index_calibration.py` generated `reports/csv/robust_period_index_calibration.csv` with 20 seeds per setting, 11,200 rows, and the report summarizes certification, uncertainty, and rejection transitions by noise level. |
| The calibrated robust detector has zero false period-index lift rate on the tested noncentral and trivial controls. | Supported | `reports/csv/robust_period_index_calibration_summary.csv` reports zero false-positive lift rates for `s3_permutation_noncentral`, random GL, random unitary, nearly-scalar noncentral, and abelian trivial controls. |
| Period-divisible but index-obstructed ranks remain non-lifting across the calibrated noisy period-index runs. | Supported | `tests/test_robust_period_index_calibration.py` checks obstructed rows never select a lift; the calibration summary reports zero `false_lift_rate` for rank-divisibility rows. |
| The robust detector threshold recommendation is empirically calibrated on controlled central and noncentral rows. | Supported | `reports/csv/robust_period_index_calibration_threshold_policies.csv` recommends centrality/phase tolerance `3e-4` with confidence margin `0.25`, with zero false-positive central and lift rates on tested controls. |
| Finite time-frequency shift and modulation operators give a natural signal-domain central projective relation. | Supported | `tests/test_time_frequency_period_index_benchmark.py` checks `M T = zeta T M` and realification of the scalar commutator; `reports/time_frequency_period_index_report.md` explains the finite Heisenberg signal symmetry. |
| The commutator-matrix detector recovers period `d` and index `d^k` from known finite time-frequency chart operators. | Supported | `tests/test_time_frequency_period_index_benchmark.py` checks `d=2,k=2` and `d=3,k=2`; `reports/csv/time_frequency_period_index_summary.csv` reports certified known-operator chart rows. |
| Time-frequency period-divisible but index-obstructed ranks are rejected before lift selection. | Supported | `test_period_divisible_but_index_obstructed` checks `d=3,k=2` ranks `3` and `6`; `reports/time_frequency_period_index_report.md` reports rejected period-divisible and rank-obstructed rows. |
| Input-learned paired chart maps recover finite time-frequency period/index structure in the clean paired-data setting. | Supported | `tests/test_time_frequency_learned_charts.py` checks input least-squares recovery for `d=2,k=2` and `d=3,k=2`; `reports/csv/time_frequency_learned_chart_summary.csv` reports certification rate `1.0` and correct period/index rate `1.0` for zero-noise input least-squares rows. |
| Learned time-frequency chart maps reject period-divisible but index-obstructed ranks when the learned period/index is certified. | Supported | `test_input_least_squares_period_divisible_rank_rejected` checks ranks `3` and `6` reject for `d=3,k=2`; `reports/time_frequency_learned_chart_report.md` reports period-divisible rows with `selected_method = none`. |
| Full-dimensional linear autoencoder chart maps can recover the finite time-frequency period/index structure in the clean controlled setting. | Supported limited | `reports/csv/time_frequency_learned_chart_summary.csv` reports certified zero-noise full-dimensional `linear_autoencoder_chart` rows, while noisy rows are rejected rather than lifted. |
| Supervised encoder chart features did not certify the finite time-frequency period/index structure in the current run. | Supported negative result | `tests/test_time_frequency_learned_charts.py` checks no-overclaim behavior; `reports/csv/time_frequency_learned_chart_summary.csv` reports supervised encoder rows rejected with zero false lift rate. |
| Learned chart recovery has zero false period-index lift rate on the tested noisy, supervised, and random noncentral controls. | Supported | `reports/csv/time_frequency_learned_chart_summary.csv` reports `false_lift_rate = 0` for all grouped rows, and `test_noncentral_negative_chart_control` rejects random chart maps. |
| Structure-preserving denoising can improve learned finite time-frequency chart period-index recovery under small chart noise. | Supported limited | `reports/time_frequency_denoised_chart_report.md` reports `d=2,k=2`, noise `1e-4`, rank `4` certification/lift rate improves from raw least squares `0.5` to `0.9` for nearest-unitary, complex-unitary, global synchronization, and unitary-global synchronization rows. |
| Denoised learned time-frequency chart maps recover certified period/index at nonzero noise in the controlled benchmark. | Supported limited | `reports/csv/time_frequency_denoised_chart_summary.csv` reports certified index-rank rows at nonzero noise, including `d=2,k=2` noise `1e-4` and `d=3,k=2` noise `3e-4`, with correct period and index when certified. |
| Denoised learned chart recovery keeps false period-index lift rate zero on the tested ranks and seeds. | Supported | `reports/time_frequency_denoised_chart_report.md` reports `all_tested_methods` false-lift rate `0` over 2,400 rows; `tests/test_time_frequency_chart_denoising.py` checks projected noncentral maps do not lift. |
| Denoised learned chart recovery still rejects period-divisible but index-obstructed ranks when period/index are certified. | Supported | `tests/test_time_frequency_chart_denoising.py` checks `d=3,k=2` ranks `3` and `6` stay `period_divisible_index_obstructed` with `selected_method = none`; the denoised rank-threshold report table records the same policy. |
| Unitary projection and global chart synchronization reduce learned operator error for noisy finite time-frequency chart maps. | Supported diagnostic | `reports/csv/time_frequency_denoised_chart_summary.csv` reports mean denoised operator error below raw least-squares error for nearest-unitary, complex-unitary, global synchronization, and unitary-global synchronization methods; this is diagnostic and not used alone as a lift certificate. |
| Nearest finite-Heisenberg projection improves certified noisy learned-chart recovery when the projection residual gate accepts. | Supported limited | `reports/time_frequency_heisenberg_projection_report.md` reports projection methods recover certified index-rank rows at noise levels where baseline denoising rejects, e.g. `d=2,k=2` noise `3e-4` and `1e-3`, with positive `certification_gain_over_best_previous`. |
| Projection residual thresholds prevent canonical replacement from becoming an automatic lift. | Supported | `reports/csv/time_frequency_heisenberg_projection_summary.csv` groups by residual threshold and reports declining acceptance at larger noise; rows with large residual keep `selected_method = none` even though canonical projected generators are detector-certifiable. |
| Nearest finite-Heisenberg projection keeps false lift rate zero on tested noncentral, trivial, and wrong-period controls. | Supported | `reports/time_frequency_heisenberg_projection_report.md` reports 12,800 benchmark rows, 800 negative-control rows, zero false lifts, and zero accepted controls; `tests/test_nearest_heisenberg_projection.py` checks random noncentral and wrong-period rejection. |
| Period-divisible but index-obstructed ranks remain non-lifting after finite-Heisenberg projection. | Supported | `tests/test_nearest_heisenberg_projection.py` checks `d=3,k=2` ranks `3` and `6` stay `heisenberg_projection_index_obstructed` with `selected_method = none`; the report rank-threshold table records obstructed rows for `d=2,k=2` and `d=3,k=2`. |
| Finite-Heisenberg projection extends certified recovery beyond previous denoising in the controlled time-frequency benchmark. | Supported limited | `reports/csv/time_frequency_heisenberg_projection_summary.csv` reports positive `certification_gain_over_best_previous` for projection methods under accepted residual thresholds, while preserving zero false-lift and negative-control acceptance rates. |
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
| The improved validation-selected selector beats the repo's internal C2M3-style baseline on the MNIST MLP benchmark. | Supported limited | `reports/csv/improved_validated_ladder_merge_summary.csv` reports `improved_validated_selector_vs_c2m3_permutation` paired mean accuracy delta `0.0438`, bootstrap CI `[0.0372, 0.0508]`, wins/ties/losses `117/0/3`, and positive mean deltas in all six fixed settings. |
| Shrinkage monomial scaling improves over raw monomial scaling on the MNIST MLP benchmark. | Supported limited | `reports/csv/improved_validated_ladder_merge_summary.csv` reports `shrinkage_monomial_scale_vs_monomial_scale` paired mean accuracy delta `0.0070`, bootstrap CI `[0.0044, 0.0102]`, wins/ties/losses `85/6/29`, and positive mean deltas in all six fixed settings. |
| Global least-squares monomial synchronization improves over raw monomial scaling on the MNIST MLP benchmark. | Supported limited | `reports/csv/improved_validated_ladder_merge_summary.csv` reports `global_monomial_scale_vs_monomial_scale` paired mean accuracy delta `0.0071`, bootstrap CI `[0.0045, 0.0101]`, wins/ties/losses `85/4/31`, and positive mean deltas in all six fixed settings. |
| The improved selector, shrinkage/global scaling grids, and union-candidate soup preserve validation-only selection in the current benchmark. | Supported | `tests/test_validation_selector_no_leakage.py` checks selector helpers do not use test metrics; `reports/improved_validated_ladder_merge_report.md` records all method choices use validation accuracy/loss only and the selector behavior table has `selector_no_test_leakage = True`. |
| Union candidate soup remains one capacity-matched MLP rather than an ensemble in the current implementation. | Supported | `tests/test_union_candidate_soup.py` checks the soup output has the reference MLP parameter count; `reports/improved_validated_ladder_merge_report.md` labels `union_candidate_soup` as single-model capacity-matched and reports its candidate pool. |
| Improved validated selector and union candidate soup do not beat ordinary greedy soup in the current MNIST MLP run. | Supported negative result | `reports/csv/improved_validated_ladder_merge_summary.csv` reports `improved_validated_selector_vs_greedy_soup` paired mean accuracy delta `-0.0015`, CI `[-0.0023, -0.0008]`, and `union_candidate_soup_vs_greedy_soup` delta `-0.0007`, CI `[-0.0011, -0.0003]`. |
| Monomial-scaled soup variants do not improve ordinary greedy soup in the current MNIST MLP run. | Supported negative result | `reports/csv/improved_validated_ladder_merge_summary.csv` reports raw/shrinkage/global/optimized monomial soup deltas versus greedy soup of `-0.0003`, `-0.0003`, `-0.0004`, and `-0.0002`, with no fixed setting positive. |
| Greedy-aware monomial selectors were evaluated on the full MNIST replay plus low-lr soup-compatible regime. | Supported | `reports/greedy_aware_monomial_report.md` covers 120 independent-seed rows plus 40 low-lr fine-tune rows, with selector rows generated after all regimes are assembled and selection marked validation-only. |
| The conservative greedy-aware selector does not beat ordinary greedy soup overall in the current benchmark. | Supported negative result | `reports/csv/greedy_aware_monomial_paired_stats.csv` reports `greedy_aware_selector_vs_greedy_soup` overall paired mean accuracy delta `-0.001089`, CI `[-0.001692, -0.000559]`. |
| The lower-confidence greedy-aware selector avoids harmful departures but does not improve ordinary greedy soup. | Supported negative result | `reports/csv/greedy_aware_monomial_paired_stats.csv` reports exact ties versus greedy soup overall, and `reports/csv/greedy_aware_selector_regret.csv` reports false challenger rate `0.000` for both tested modes. |
| Low-lr fine-tuning creates a more soup-compatible descriptive regime, but it is not enough for an overall greedy-soup win. | Supported descriptive | In `reports/csv/greedy_aware_monomial_paired_stats.csv`, low-lr `greedy_aware_selector_vs_greedy_soup` has mean delta `0.0001725`, CI `[0.000025, 0.000335]`, while the overall selector claim remains negative. |
| Union candidate soup increases ingredient count but does not beat greedy soup overall. | Supported negative result | `reports/csv/soup_compatible_modes_summary.csv` reports low-lr union candidate mean ingredient count `3.7` versus greedy soup `1.5`, but `reports/csv/greedy_aware_monomial_paired_stats.csv` reports overall union candidate mean delta `-0.000479`, CI `[-0.000823, -0.000154]`. |
| Shrinkage and global positive monomial scaling continue to improve raw monomial scaling in the greedy-aware rerun. | Supported limited | `reports/csv/greedy_aware_monomial_paired_stats.csv` reports shrinkage-over-raw delta `0.007020`, CI `[0.004448, 0.010043]`, and global-over-raw delta `0.007148`, CI `[0.004716, 0.010096]`. |
| Robust scale estimators and nested-validation splits are implemented but not promoted to benchmark-win claims. | Supported | `tests/test_robust_scale_estimation.py` and `tests/test_nested_validation_no_leakage.py` cover the helpers; `reports/greedy_aware_monomial_report.md` records that robust variants and nested replay rows are not claimed without paired statistical support. |
| Validation loss optimization over log-scales gives only a descriptive, statistically unsupported improvement over raw monomial scaling in the current MNIST MLP run. | Supported descriptive | `reports/csv/improved_validated_ladder_merge_summary.csv` reports `optimized_monomial_scale_vs_monomial_scale` paired mean accuracy delta `0.0009`, CI `[-0.0030, 0.0053]`, and fixed-setting positives in only two of six settings. |
| Fashion-MNIST MLP validation tests whether the improved MNIST selector result persists beyond MNIST. | Supported limited | `reports/fashion_mnist_improved_ladder_report.md` and `reports/csv/fashion_mnist_improved_ladder_summary.csv` record `positive paired mean, positive bootstrap CI, and majority fixed-setting support`. |
| Fashion-MNIST provides an additional greedy-soup boundary check for the improved selector. | Supported negative result | `reports/csv/fashion_mnist_improved_ladder_summary.csv` records `paired mean test accuracy delta is not positive`. |
| Fashion-MNIST residuals do not support a mostly noncentral final-decision claim in this run. | Supported negative result | `reports/fashion_mnist_improved_ladder_report.md` records `literal noncentral final-decision fraction=0.2857; many remaining settings are GL-diagnostic-only rather than central/projective`. |
| Fashion-MNIST residuals remain non-Brauer under the tested structure groups. | Supported limited | `reports/fashion_mnist_improved_ladder_report.md` records `non-Brauer/noncentral fraction=1.0000, central/projective candidate fraction=0.0000, finite-index candidate fraction=0.0000`. |
| Fashion-MNIST residual taxonomy remains useful beyond MNIST without claiming Brauer classes. | Supported limited | `reports/fashion_mnist_improved_ladder_report.md` records `taxonomy separates noncentral final decisions, GL-diagnostic-only reductions, and zero central/projective finite-index candidates`. |
| The Fashion-MNIST residual detector remains conservative about finite-index/Brauer structure. | Supported limited | `reports/csv/fashion_mnist_improved_ladder_summary.csv` records `detector keeps finite-index and central/projective candidate fractions low on real Fashion-MNIST residuals`. |
| The repo now documents license-clean external baseline integration status for Git Re-Basin, C2M3, and Model Soups. | Supported | `external_baselines/README.md` records paper names, official URLs, MIT licenses for the three required external repos, integration mode, deviations, validation usage, capacity matching, inference cost, and fairness boundaries. |
| The external-baseline comparison uses the same MNIST MLP checkpoints and splits for every method in each setting. | Supported | `reports/external_baseline_comparison.md` records `N=3,4`, widths `32,64`, seeds `1800..1804`, shared checkpoints under `reports/checkpoints/external_baselines/`, and validation-only selection; `reports/csv/external_baseline_comparison.csv` has 160 rows over 20 settings. |
| On the external-baseline MNIST MLP run, the improved selector outperforms the internal C2M3-style baseline. | Supported limited | `reports/csv/external_baseline_comparison_summary.csv` reports `improved_validated_selector` paired mean accuracy delta versus `c2m3_permutation` `0.0209`, bootstrap CI `[0.0135, 0.0296]`, over 20 paired settings. |
| On the external-baseline MNIST MLP run, the improved selector does not beat the faithful greedy Model Soups baseline. | Supported negative result | `reports/csv/external_baseline_comparison_summary.csv` reports `improved_validated_selector` paired mean accuracy delta versus `greedy_soup` `-0.0024`, bootstrap CI `[-0.0042, -0.0009]`. |
| On the external-baseline MNIST MLP run, monomial-scaled greedy soup does not beat ordinary greedy soup. | Supported negative result | `reports/csv/external_baseline_comparison_summary.csv` reports `monomial_scaled_greedy_soup` paired mean accuracy delta versus `greedy_soup` `-0.0014`, bootstrap CI `[-0.0028, -0.0001]`. |
| Official external-code integration was attempted and documented, but produced no official baseline metrics. | Supported negative result | `external_baselines/OFFICIAL_INTEGRATION.md` and `reports/official_external_baseline_attempt.md` record cloned official Git Re-Basin, C2M3, and Model Soups commits/licenses, environment probes, checkpoint-interface mismatches, and the decision not to generate official baseline CSV/table artifacts. |
| Block-orthogonal synthetic controls are implemented and distinguish recovered rotations, noncentral block holonomy, and scalar block phases. | Supported | `tests/test_block_gauge_alignment.py` checks rotation recovery, noncentral block commutators, and scalar `-I` block phase detection. |
| Real block-orthogonal diagnostics were evaluated on MNIST MLPs. | Supported | `reports/block_orthogonal_ladder_report.md` and `reports/csv/block_orthogonal_ladder.csv` cover MNIST ReLU MLP, `N=3,4`, widths `16,32`, block sizes `2,4,8`, and five seeds. |
| In the current real MNIST block-orthogonal run, block gauges do not reduce residual centrality on average and do not produce scalar/projective candidates. | Supported negative result | `reports/csv/block_orthogonal_ladder_summary.csv` reports real block mean centrality improvements from permutation to block of `-0.0050`, `-0.0090`, and `-0.0100`, and central/projective candidate fraction `0.0000` for block sizes `2,4,8`. |
| Block-orthogonal merge performance was not evaluated for the ReLU MLP run. | Supported negative result | `reports/block_orthogonal_ladder_report.md` labels block-orthogonal rotations as feature-space diagnostics, and `reports/csv/block_orthogonal_ladder.csv` preserves `merge_evaluated = False`. |
| Global block synchronization recovers planted globally consistent block gauges. | Supported | `tests/test_global_block_synchronization.py` checks exact recovery for planted gauges, and `reports/global_block_synchronization_report.md` reports zero synthetic connection residual for `planted_recoverable_block_rotations`. |
| Global block synchronization projects block maps to cycle-consistent gauges while preserving an explicit connection-residual honesty check. | Supported descriptive | `reports/csv/global_block_synchronization_summary.csv` reports zero post-projection cycle/centrality for global rows and nonzero real connection residuals, so this is diagnostic projection rather than a proof of exact real-data descent. |
| Optimized global block synchronization recovers planted block gauges and rejects fake projected-cycle traps by calibrated connection residual. | Supported | `tests/test_optimized_global_block_synchronization.py` checks exact planted recovery and projection-trap rejection; `reports/optimized_global_block_synchronization_report.md` records fake projection rows with zero projected cycle score but rejected connection residual. |
| Residual-optimized global block synchronization can reduce connection residual beyond the spectral initializer in a controlled noisy-observation example. | Supported descriptive | `reports/csv/optimized_global_block_synchronization_paired_stats.csv` reports `optimized_sync_vs_spectral_connection_residual` mean delta `-0.00264068` for the controlled synthetic case; this is not claimed as a real-model benchmark. |
| Learned block partitions are implemented deterministically for activation-correlation and output-weight similarity choices. | Supported | `tests/test_learned_block_partition.py` checks deterministic clustering and required inputs; `reports/configs/global_block_synchronization_config.json` records enabled partition methods. |
| Validation-selected global activation blocks recover a planted non-contiguous positive-control partition. | Supported | `tests/test_global_learned_block_partition.py` checks recovery of blocks `{0,2}` and `{1,3}`; `reports/csv/optimized_global_block_synchronization_paired_stats.csv` reports learned versus contiguous validation residual delta `-0.956981`. |
| Block-compatible exact block-gauge averaging works in the controlled linear-hidden architecture. | Supported descriptive | `tests/test_block_compatible_merge.py` checks logit preservation, same parameter count, and aligned averaging; `reports/csv/optimized_global_block_synchronization_summary.csv` reports block-compatible aligned-average pseudo-label accuracy `1.0` versus unaligned weight average `0.9664`. |
| The current global/learned block MNIST run finds no scalar finite-order projective candidates. | Supported negative result | `reports/global_block_synchronization_report.md` reports real MNIST central/projective candidate fraction `0.0000` across permutation, monomial, low-rank GL, pairwise block, and global block diagnostic rows. |
| Learned block partitions do not improve observed pairwise-block centrality over contiguous blocks in the current MNIST run. | Supported negative result | `reports/global_block_synchronization_report.md` states learned partitions do not reduce mean observed pairwise-block centrality versus contiguous; `reports/csv/global_block_synchronization_summary.csv` records contiguous block size 2 mean centrality `0.5155`, lower than activation-correlation `0.6295` and output-weight similarity `0.5478`. |
| Strict block-gauge calibration with a numerical floor rejects fake projection traps and noncentral block holonomy. | Supported | `reports/csv/block_gauge_phase_diagram_summary.csv` reports strict false-accept rate `0` for both `fake_projection_trap` and `noncentral_block_holonomy`; `reports/block_gauge_branch_closure_report.md` records raw threshold `4.47628e-16`, effective threshold `1e-12`, and clean-worktree generation. |
| Scalar `mu_2` block phases are detected before projection in the phase-diagram benchmark. | Supported | `reports/csv/block_gauge_phase_diagram_summary.csv` reports scalar candidate fraction `1.0` for `scalar_block_phase_mu2`, and `reports/block_gauge_phase_diagram_report.md` keeps this separate from projected-cycle evidence. |
| Residual-optimized global block synchronization reduces connection residual over spectral on the multi-seed synthetic grid. | Supported descriptive | `reports/csv/block_gauge_phase_diagram_paired_stats.csv` reports 16,380 optimized-vs-spectral pairs, mean residual reduction `0.01245`, bootstrap CI `[0.0119235, 0.0129811]`, wins/ties/losses `4117/12263/0`. |
| Learned non-contiguous block discovery beats contiguous blocks on planted positive controls. | Supported | `reports/csv/block_gauge_phase_diagram_paired_stats.csv` reports 20/20 wins and mean validation-residual delta `-0.285714` for activation, output-weight, residual-greedy, and validation-selected learned partitions versus contiguous. |
| Exact global block gauges are accepted up to numerical tolerance after the calibration-floor fix. | Supported | `reports/csv/block_gauge_phase_diagram_summary.csv` reports strict acceptance rate `1.0` and false-reject rate `0.0` for `exact_global_block_gauge`; `reports/csv/block_gauge_acceptance_by_noise.csv` reports acceptance rate `1.0` for exact gauges at every listed noise bucket. |
| The block-compatible linear-hidden MNIST task supports capacity-matched block-gauge aligned averaging over unaligned averaging. | Supported descriptive | `reports/block_compatible_learning_report.md` reports optimized aligned average mean accuracy `0.61995` versus unaligned weight average `0.42835`, paired mean delta `0.1916`, and capacity-matched `True` for the aligned average. |
| The block-compatible aligned average does not beat greedy soup in the current exact linear-hidden benchmark. | Supported negative result | `reports/csv/block_gauge_phase_diagram_paired_stats.csv` reports optimized aligned average versus greedy soup mean accuracy delta `-0.00055`, CI `[-0.00165, 0]`, wins/ties/losses `0/19/1`. |
| Real ReLU MLP block-gauge rows remain diagnostic-only under the improved phase-diagram run. | Supported negative result | `reports/relu_block_diagnostic_report.md` reports `block_merge_reported=False` and scalar candidate fraction `0` for ReLU diagnostic rows; block-orthogonal ReLU merge accuracy is not reported. |
| The 5(j) block-gauge branch is experimentally closed as a diagnostic and controlled exact-merge component. | Supported limited | `reports/block_gauge_branch_closure_report.md` records success criteria A-H as supported while explicitly excluding exact ReLU block-orthogonal merging, real Brauer/projective residuals in MNIST, CIFAR, and broad natural model-merging claims. |

## Fashion-MNIST Greedy-Safe Selector And CNN Extension

| Claim | Status | Evidence |
| --- | --- | --- |
| A Fashion-MNIST greedy-safe selector can avoid harmful departures from greedy soup in the replayed 5(m) MLP candidate table. | Supported limited | `reports/csv/fashion_mnist_greedy_safe_selector_summary.csv` reports bootstrap/regret selector rows with false challenger rate `0.0000`, mean delta versus greedy soup `0.0000`, and mean delta versus internal C2M3 `0.067223`. |
| The Fashion-MNIST greedy-safe selector matches greedy soup while preserving gains over internal C2M3. | Supported limited | `reports/fashion_mnist_greedy_safe_selector_report.md` records safe rows that choose greedy soup in all 35 settings, match greedy exactly, and retain the C2M3 gain inherited from the 5(m) method pool. |
| The Fashion-MNIST greedy-safe selector beats greedy soup overall. | Not yet supported | The best greedy-safe selector rows have mean delta versus greedy soup `0.000000` with bootstrap CI `[0.000000, 0.000000]`, so this is a match, not a win. |
| Exact positive ReLU channel gauges extend the MLP monomial-scaling idea to the tested small CNN channel/hidden units. | Supported | `tests/test_cnn_channel_gauge.py` verifies exact logit preservation for channel permutations, positive channel scalings, and combined gauges, with unchanged parameter count and inference-cost proxy. |
| In the initial Fashion-MNIST CNN benchmark, shrinkage/global channel scaling gives a descriptive improvement over channel-permutation C2M3. | Supported descriptive | `reports/csv/fashion_mnist_cnn_ladder_summary.csv` reports shrinkage/global channel-scale mean deltas over C2M3 of `0.012733` and `0.014667`, with bootstrap CIs touching zero over three N=3 seeds. |
| Raw positive channel scaling improves over channel-permutation C2M3 in the initial CNN benchmark. | Supported negative result | `reports/csv/fashion_mnist_cnn_ladder_summary.csv` reports raw positive channel scale mean delta versus C2M3 `-0.000567`, CI `[-0.020900, 0.028300]`. |
| The CNN channel-gauge branch beats greedy soup. | Not yet supported | `reports/fashion_mnist_cnn_ladder_report.md` reports greedy soup mean test accuracy `0.824267`; shrinkage/global channel-scale rows remain below greedy soup, and the greedy-safe selector matches greedy with zero delta. |
| CNN residuals are Brauer or period-index classes. | Not yet supported | `reports/fashion_mnist_cnn_ladder_report.md` records zero central/projective and finite-index candidate fractions and explicitly does not run a CNN Brauer/period-index detector. |

## Sheaf/GNN Optional Diagnostic

| Claim | Status | Evidence |
| --- | --- | --- |
| A small, license-clean sheaf/GNN feasibility pass documents how Neural Sheaf Diffusion could be used without blocking model-merging experiments. | Supported | `reports/sheaf_gnn_feasibility.md` records the official Neural Sheaf Diffusion datasets, Apache-2.0 license, PyG dependency blocker, learned-map storage path, triangle/cycle measurement plan, and minimal local run. |
| Official Neural Sheaf Diffusion can run as an optional external smoke test in a separate PyG environment. | Supported limited | `external_baselines/NSD_INTEGRATION.md` and `reports/nsd_official_integration_report.md` document the separate Python 3.9/PyG environment and a tiny WebKB Texas BundleSheaf run with test accuracy `0.6486`. |
| Learned official NSD BundleSheaf transport caches can be post-processed for triangle cycle diagnostics. | Supported diagnostic | `experiments/nsd_official_cycle_diagnostics.py` generated `reports/csv/nsd_cycle_diagnostics.csv` with 67 Texas triangles and non-null cycle scores for unweighted connection and weighted cache variants. |
| Cycle inconsistency can be measured as a diagnostic of learned sheaf transports on small synthetic heterophilic graphs. | Supported limited | `reports/sheaf_gnn_optional_report.md` and `reports/csv/sheaf_gnn_cycle_diagnostics.csv` report 18 sheaf rows over target heterophily `0.25,0.55,0.85`, with 42-99 triangles per graph and non-null learned triangle cycle-inconsistency scores. |
| The optional cycle-regularized sheaf row reduces measured cycle inconsistency in the synthetic smoke run. | Supported descriptive | `reports/sheaf_gnn_optional_report.md` reports mean cycle inconsistency `1.2220` for `rotation_sheaf` versus `0.4597` for `rotation_sheaf_cycle_reg`; this is not a general accuracy claim. |
| Cycle inconsistency strongly explains heterophily or test accuracy on this run. | Not yet supported | `reports/sheaf_gnn_optional_report.md` reports weak synthetic-run correlations: cycle-vs-test-accuracy Pearson `-0.1139` and heterophily-vs-cycle Pearson `-0.0501`. |
| Official NSD cycle regularization improves accuracy or learned-map consistency. | Not yet supported | `reports/nsd_official_integration_report.md` records that the official map cache is created with `clone().detach()`, so the cycle-regularizer attempt was not applied. |

## Fashion-MNIST CNN Channel-Gauge Confirmatory Benchmark

| Claim | Status | Evidence |
| --- | --- | --- |
| `cnn_exact_channel_gauges_preserve_logits` / exact CNN channel gauges preserve logits | Supported | `reports/fashion_mnist_cnn_channel_gauge_confirmatory_report.md` records tests/test_cnn_channel_gauge.py covers permutation, scaling, combined gauges, conv-to-conv, conv-to-linear, hidden scaling, parameter count, and inference-cost proxy. |
| `cnn_shrinkage_channel_scale_over_c2m3_confirmed` / shrinkage channel scale over C2M3 | Supported limited | `reports/fashion_mnist_cnn_channel_gauge_confirmatory_report.md` records positive paired mean=0.002967, CI lower=0.000313, fixed positives=2/2. |
| `cnn_global_channel_scale_over_c2m3_confirmed` / global channel scale over C2M3 | Supported limited | `reports/fashion_mnist_cnn_channel_gauge_confirmatory_report.md` records positive paired mean=0.003180, CI lower=0.000420, fixed positives=2/2. |
| `cnn_optimized_channel_scale_over_c2m3_confirmed` / optimized channel scale over C2M3 | Supported limited | `reports/fashion_mnist_cnn_channel_gauge_confirmatory_report.md` records positive paired mean=0.009333, CI lower=0.004787, fixed positives=2/2. |
| `cnn_channel_candidate_soup_over_greedy_soup` / channel candidate soup over greedy soup | Supported descriptive | `reports/fashion_mnist_cnn_channel_gauge_confirmatory_report.md` records positive paired mean=0.000647, CI=[0.000000,0.001553], fixed positives=2/2. |
| `cnn_greedy_safe_selector_matches_or_beats_greedy_soup` / greedy-safe selector versus greedy soup | Supported descriptive | `reports/fashion_mnist_cnn_channel_gauge_confirmatory_report.md` records positive paired mean=0.000653, CI=[0.000000,0.001493], fixed positives=2/2. |
| `cnn_channel_gauge_generalizes_mlp_exact_gauge_story` / CNN channel gauge generalizes exact-gauge story | Supported limited | `reports/fashion_mnist_cnn_channel_gauge_confirmatory_report.md` records exactness tests pass; confirmatory performance status follows shrinkage/global/optimized C2M3 comparisons. |
| `cnn_residuals_are_brauer_or_period_index` / CNN residuals are Brauer or period-index | Not yet supported | `reports/fashion_mnist_cnn_channel_gauge_confirmatory_report.md` records central/projective and finite-index candidate fractions are zero under tested diagnostics. |

## CIFAR Or Rotated-MNIST Feasibility

| Claim | Status | Evidence |
| --- | --- | --- |
| Rotated-MNIST is feasible as a bridge dataset for the small CNN channel-gauge benchmark. | Supported limited | `reports/cifar_or_colored_mnist_feasibility.md` reports rotated-MNIST individual max accuracy `0.9283` over two `N=3` settings, above the `0.80` bridge threshold; `reports/csv/cifar_or_colored_mnist_feasibility_summary.csv` marks `rotated_mnist_feasibility_status` as `Supported limited`. |
| On the rotated-MNIST bridge run, greedy soup and the greedy-safe selector outperform C2M3-style channel permutation descriptively. | Supported limited | `reports/csv/cifar_or_colored_mnist_feasibility_summary.csv` reports mean test accuracy `0.924583` for greedy soup and greedy-safe selector versus `0.836667` for `c2m3_channel_permutation`, with paired mean delta `0.087917` and CI `[0.062500, 0.113333]`; the selector chose greedy soup in both settings. |
| Positive channel scaling beats C2M3 on the rotated-MNIST bridge run. | Not yet supported | `reports/csv/cifar_or_colored_mnist_feasibility_summary.csv` reports `positive_channel_scale` mean delta versus C2M3 `0.001250` with CI `[-0.004167, 0.006667]`, so the effect is descriptive at best. |
| CIFAR-10 remains below the plumbing threshold in the gated small-CNN probe. | Supported negative result | The CIFAR probe in `reports/cifar_or_colored_mnist_feasibility.md` reports test accuracy `0.2480`, below the `0.45` plumbing threshold and far below the `0.60` meaningful-claim threshold, so CIFAR merge methods were not run. |

## Bridge Dataset Channel-Gauge Expansion

| Claim | Status | Evidence |
| --- | --- | --- |
| Bridge datasets support the same C2M3-versus-greedy boundary pattern. | Supported limited | `reports/bridge_dataset_channel_gauge_expansion.md` reports 17 bridge settings across rotated-MNIST angles `15,25,45`, colored-MNIST, and `N=3,4`; on the main rotated-25 `N=3` 10-seed setting, greedy soup beats C2M3-style channel synchronization by `0.089670`, CI `[0.067857,0.112412]`, and over all bridge settings the delta is `0.084724`, CI `[0.063629,0.103986]`. |
| The expanded bridge run clears the bridge base-accuracy gate. | Supported limited | `reports/csv/bridge_dataset_channel_gauge_expansion_summary.csv` records `bridge_accuracy_gate` as `Supported limited`, with minimum setting-level individual max accuracy `0.9103` above the `0.80` bridge threshold. |
| Greedy-safe selection preserves the greedy-soup bridge result in this run. | Supported descriptive | `reports/bridge_dataset_channel_gauge_expansion.md` reports greedy-safe selector mean delta versus greedy soup `0.000000`; the selector chose `greedy_soup` in all 17 settings using validation metrics only. |
| Optimized channel scaling improves over C2M3-style synchronization on bridge datasets but remains below greedy soup. | Supported limited | `reports/csv/bridge_dataset_channel_gauge_expansion_summary.csv` reports overall `optimized_channel_scale` delta versus C2M3 `0.014729`, CI `[0.010211,0.019659]`, while its delta versus greedy soup is `-0.069994`. |
| Bridge results imply CIFAR or general vision performance. | Not yet supported | `reports/bridge_dataset_channel_gauge_expansion.md` explicitly labels rotated/colored-MNIST as bridge datasets only and states that these results do not imply CIFAR or general vision performance. |

## CIFAR Rescue Or No-Go Gate

| Claim | Status | Evidence |
| --- | --- | --- |
| The bounded CIFAR rescue clears the meaningful base-accuracy gate for a no-BatchNorm CNN. | Supported limited | `reports/cifar_rescue_or_no_go_report.md` reports normalized/augmented no-BatchNorm CNN probes at `0.6503` and `0.6057` test accuracy, and the merge setting reaches max individual accuracy `0.6583`, above the `0.60` meaningful threshold. |
| CIFAR does not need to be formally excluded solely for low base accuracy in the current artifacts. | Supported limited | `reports/csv/cifar_rescue_or_no_go_summary.csv` marks `cifar_rescue_or_no_go_status` as `Meaningful CIFAR gate passed`; this reverses the earlier no-go only for the bounded rescued CNN setting. |
| On the one-setting CIFAR rescue run, greedy soup and the greedy-safe selector are descriptively above C2M3-style channel synchronization. | Supported descriptive | `reports/csv/cifar_rescue_or_no_go_summary.csv` reports greedy soup and greedy-safe selector test accuracy `0.658333` versus C2M3-style channel synchronization `0.487000`, with delta `0.171333`; because there is one merge setting, this is descriptive rather than a robust paired-statistical claim. |
| Exact positive channel scaling does not improve over C2M3-style channel synchronization on the CIFAR rescue run. | Supported negative result | `reports/csv/cifar_rescue_or_no_go_summary.csv` reports `positive_channel_scale` test accuracy `0.438333`, below C2M3-style channel synchronization `0.487000`, with delta `-0.048667`. |
| The CIFAR ensemble row is an upper bound with extra inference and parameter cost. | Supported | `reports/cifar_rescue_or_no_go_report.md` reports `ensemble_upper_bound` test accuracy `0.675000`, while the CSV records `single_model=False`, `capacity_matched=False`, and parameter/inference multipliers `3.0`. |


## Final CIFAR Channel-Gauge Confirmatory Run

| Claim | Status | Evidence |
| --- | --- | --- |
| `cifar_base_accuracy_gate_passed_final` / base accuracy gate passed | Supported limited | `reports/cifar_final_channel_gauge_confirmatory_report.md` records all final settings pass; minimum setting max individual accuracy=0.6464. |
| `cifar_shrinkage_channel_scale_over_c2m3` / shrinkage channel scale over C2M3 | Supported descriptive | `reports/cifar_final_channel_gauge_confirmatory_report.md` records positive paired mean=0.000160, CI=[-0.000640,0.001280], fixed positives=1/1. |
| `cifar_global_channel_scale_over_c2m3` / global channel scale over C2M3 | Supported descriptive | `reports/cifar_final_channel_gauge_confirmatory_report.md` records positive paired mean=0.000040, CI=[-0.001900,0.002280], fixed positives=1/1. |
| `cifar_optimized_channel_scale_over_c2m3` / optimized channel scale over C2M3 | Supported descriptive | `reports/cifar_final_channel_gauge_confirmatory_report.md` records positive paired mean=0.000540, CI=[-0.000880,0.002740], fixed positives=1/1. |
| `cifar_union_candidate_soup_over_greedy_soup` / union candidate soup over greedy soup | Supported descriptive | `reports/cifar_final_channel_gauge_confirmatory_report.md` records positive paired mean=0.000440, CI=[0.000000,0.001320], fixed positives=1/1. |
| `cifar_greedy_safe_selector_over_or_matches_greedy_soup` / greedy-safe selector over or matches greedy soup | Supported limited | `reports/cifar_final_channel_gauge_confirmatory_report.md` records matches baseline with paired mean=0.000000, CI=[0.000000,0.000000]. |
| `cifar_exact_channel_gauge_methods_capacity_matched` / exact channel-gauge methods capacity matched | Supported limited | `reports/cifar_final_channel_gauge_confirmatory_report.md` records exact=True, capacity_matched=True for C2M3 and scale rows; no BatchNorm gauge claim is made. |
| `cifar_general_model_merging_win` / general CIFAR model-merging win | Not yet supported | `reports/cifar_final_channel_gauge_confirmatory_report.md` records bounded no-BatchNorm CIFAR setting only; no external official baseline, SOTA, BatchNorm, or broad CIFAR claim. |
| `cifar_branch_closed_for_current_paper` / CIFAR branch closed for current paper | Supported limited | `reports/cifar_final_channel_gauge_confirmatory_report.md` records final bounded CIFAR run completed; base gate passes, exact channel gauges are descriptive only, union soup CI touches zero, and the report closes CIFAR as an appendix boundary. |
| `cifar_residuals_are_brauer_or_period_index` / CIFAR residuals are Brauer or period-index | Not yet supported | `reports/cifar_final_channel_gauge_confirmatory_report.md` records this run records channel residual diagnostics but does not find or certify Brauer/period-index CIFAR residual classes. |

## CIFAR And Bridge Appendix

| Claim | Status | Evidence |
| --- | --- | --- |
| CIFAR is included as a bounded appendix boundary, not as a broad model-merging win. | Supported limited | `reports/cifar_bridge_boundary_summary.md` and `reports/latex/cifar_bridge_appendix.tex` consolidate the failed CIFAR probe (`0.2480`), bounded rescue gate pass (`0.658333` max individual accuracy), final CIFAR confirmatory run (`0.650620` mean individual max accuracy), descriptive exact channel-gauge deltas, greedy-soup boundary behavior, and bridge-only rotated/colored-MNIST results. |
| CIFAR confirms the main method. | Not yet supported | `reports/cifar_bridge_boundary_summary.md` explicitly frames CIFAR as an appendix boundary: exact CIFAR channel-gauge gains are tiny/descriptive with intervals crossing zero or touching zero, greedy soup remains the boundary, and bridge datasets are not promoted to CIFAR or general vision claims. |

## External Integration Appendix

| Claim | Status | Evidence |
| --- | --- | --- |
| Official integrations are documented; main comparisons remain faithful in-repo unless official runs succeeded. | Supported | `reports/external_integration_summary.md` and `reports/latex/external_integration_appendix.tex` summarize official Git Re-Basin, C2M3, Model Soups, and Neural Sheaf Diffusion integration status, including repository URLs, licenses, attempted environments, official-run status, blockers, in-repo faithful surrogates or wrappers, and claim boundaries. Git Re-Basin, C2M3, and Model Soups did not produce official baseline metrics; NSD ran only a tiny Texas smoke test and post-hoc diagnostic wrapper. |
| Official external code confirms the method. | Not yet supported | `reports/external_integration_summary.md` states that no official model-merging baseline ran on the exact TwistedMerge checkpoint set and that official NSD is only an optional smoke test unrelated to TwistedMerge model-merging performance. |

## Unified Quantitative Obstruction Chain

| Claim | Status | Evidence |
| --- | --- | --- |
| Different residual scores predict different downstream decisions; a single scalar obstruction does not explain all cases. | Supported descriptive | `reports/unified_quantitative_obstruction_chain.md` aligns 51,895 rows from 19 source families. Link tests show planted cycle score predicts Git-ReBasin-style pairwise degradation (Spearman `0.795903`), validation deltas predict selector gain as a separate selection signal (Spearman `0.552456`), learned-operator error predicts certification negatively (Spearman `-0.835969`), projection residual gates keep accepted false-lift rate at `0`, period-index divisibility matches selected lifts at rate `1.0`, and the block-gauge phase diagram has 12,240 projected-cycle trap rows where connection residual is needed. |
| Every merging failure is Brauer/projective. | Not yet supported | `reports/unified_quantitative_obstruction_chain.md` explicitly separates permutation synchronization, validation-selected soups, exact positive scale gauges, noncentral diagnostics, projection-gated period-index rows, sheaf diagnostics, and dataset gates; it repeats that the unified chain does not claim every merging failure is Brauer/projective. |

## Paper-Ready Post-35791f7 Synthesis

| Claim | Status | Evidence |
| --- | --- | --- |
| The current paper framing is a mathematical and diagnostic model-merging contribution with limited exact-gauge wins over faithful internal C2M3-style baselines. | Supported synthesis | `reports/results_narrative_after_35791f7.md`, `reports/latex/results_section_after_35791f7.tex`, and `reports/latex/main_claims_table_after_35791f7.tex` synthesize the final CIFAR boundary, bridge expansion, MNIST/Fashion exact-gauge results, residual taxonomy, period-index/time-frequency detectors, block-gauge gate, official-baseline status, and limitations. |
| The post-35791f7 synthesis supports a broad SOTA, official-baseline, greedy-soup, broad CIFAR, or real-neural Brauer/period-index win. | Not yet supported | The synthesis files explicitly preserve the negative boundaries from the underlying reports: no official external-code metrics were produced, greedy soup remains a boundary baseline, CIFAR exact-gauge effects are descriptive only, and real neural residuals remain non-Brauer/noncentral under tested diagnostics. |

## Full Capacity And Claim-Boundary Audit

| Claim | Status | Evidence |
| --- | --- | --- |
| `full_capacity_claim_audit_created` / authoritative capacity and symmetry registry exists | Supported | `reports/full_capacity_claim_audit.md` and `reports/csv/full_capacity_claim_audit.csv` enumerate current methods and diagnostics with output type, multipliers, exact-symmetry flags, validation-selection status, official-code status, extra-capacity status, residual gates, scopes, and allowed/forbidden paper claims. |
| `capacity_claim_boundaries_are_explicit` / broad overclaim boundaries are machine-readable | Supported | `reports/csv/full_capacity_claim_audit.csv` records per-row `paper_claim_allowed` and `paper_claim_forbidden` fields, including explicit boundaries for external official code, CIFAR, BatchNorm, ReLU block rotations, period-index lifts, and sheaf/GNN diagnostics. |

## Not Yet Supported

| Claim | Status | Reason |
| --- | --- | --- |
| TwistedMerge beats external model-merging baselines. | Not yet supported | `reports/external_baseline_comparison.md` compares documented faithful in-repo baselines, not official external code execution; `reports/official_external_baseline_attempt.md` records no successful official-code runs; and the improved selector remains below faithful greedy soup with paired mean accuracy delta `-0.0024`. |
| TwistedMerge solves natural MNIST/CIFAR merging. | Not yet supported | `reports/model_merging_verification_report.md` is stronger for MNIST than the smoke run, but it still reports descriptive prototype baselines and excludes CIFAR as near chance. |
| The rotated-MNIST bridge result proves CIFAR or broad vision generality. | Not yet supported | The bridge run is rotated-MNIST only; the later CIFAR rescue is a bounded no-BatchNorm CNN run with one merge setting, not broad vision generality. |
| The expanded rotated/colored-MNIST bridge run proves CIFAR or general vision performance. | Not yet supported | `reports/bridge_dataset_channel_gauge_expansion.md` covers MNIST-derived bridge datasets only; it does not evaluate CIFAR and explicitly forbids promoting the bridge result to CIFAR/general-vision claims. |
| The CIFAR rescue proves a general CIFAR model-merging win. | Not yet supported | `reports/cifar_rescue_or_no_go_report.md` clears the meaningful base-accuracy gate, but method comparisons are one-setting descriptive rows; no broad or multi-seed CIFAR method-win claim is established. |
| Positive channel scaling improves CIFAR merging. | Not yet supported | The bounded CIFAR rescue reports `positive_channel_scale` below C2M3-style channel synchronization by `-0.048667` test accuracy. |
| CIFAR confirms the main method. | Not yet supported | `reports/cifar_bridge_boundary_summary.md` says CIFAR is a bounded appendix boundary, not a broad model-merging win; the final CIFAR exact channel-gauge deltas versus C2M3 are descriptive and the strongest union-soup delta versus greedy soup has a confidence interval touching zero. |
| Official external code confirms the method. | Not yet supported | `reports/external_integration_summary.md` records that official Git Re-Basin, C2M3, and Model Soups did not run on the exact checkpoint set, and that the official Neural Sheaf Diffusion run was a tiny related smoke test with post-hoc diagnostics rather than a model-merging comparison. |
| Every merging failure is Brauer/projective. | Not yet supported | `reports/unified_quantitative_obstruction_chain.md` gives explicit counter-accounting: different residual scores govern synchronization degradation, selector gain, detector certification, lift decisions, and dataset gates, and noncentral/permutation/soup cases are not promoted to Brauer/projective failures. |
| TwistedMerge fully trivializes a nonzero `H^2(mu_2)` class as an ordinary untwisted vector bundle. | Not yet supported | The nonzero tetrahedral `H^2` class is explicitly non-coboundary. Current `TwistedMerge` does not construct an edge-level untwisted descent for it. |
| The branch-prediction lift is a complete transition-map-level twisted sheaf descent implementation. | Not yet supported | The q=2 branch result is a controlled prediction-level sanity check. It is not a proof of full sheaf-level descent in the non-coboundary case. |
| Cycle obstruction score predicts weight-average merge degradation beyond the trivial number-of-models confound. | Supported narrow | `reports/fixed_setting_verification_report.md` and `reports/fixed_setting_full_run_interpretation.md` record the Prompt 11 quality-gated `mlp2` run on MNIST and Fashion-MNIST with 30 observed seeds per fixed setting. Individual-model accuracy passes the gate. One primary observed setting passes the predefined gate: Fashion-MNIST, `N=3`, no shift, activation matching, Pearson `0.2420`, bootstrap Pearson lower bound `0.0065`, and Spearman `0.2629`. The other 15 observed settings remain unsupported, so this is not a broad predictor claim. |
| C2M3-style alignment or TwistedMerge/rank-lifted branching gives a statistically meaningful improvement over weight averaging on MNIST. | Not yet supported | `reports/fixed_setting_verification_report.md` includes repeated-seed internal method deltas, but this audit does not promote a general method-win claim: the comparisons are faithful in-repo baselines, rank-lift rows are branch ensembles with extra capacity, and official external baselines still did not run on the exact checkpoint set. |
| TwistedMerge/rank-lifted branching beats greedy soup as a single-model baseline. | Not yet supported | `reports/model_merging_verification_report.md` labels `twisted_rank_lift_2` as a branch ensemble with extra capacity; fixed-`N` deltas versus greedy soup are mixed or negative. |
| The planted obstruction score generally predicts every kind of model-merging degradation. | Not yet supported | In `reports/planted_obstruction_model_merging_report.md`, weight averaging is constant across planted levels because it does not use alignments, and the random noncentral Git-ReBasin trend is weaker than the central trend. |
| TwistedMerge/rank-lifted branching adds benefit beyond C2M3 in the planted model-merging benchmark. | Not yet supported | `reports/csv/planted_obstruction_stats.csv` reports rank-lift accuracy delta versus C2M3 is `0.0000` for central and random planted families. |
| Rank-lift helps only when the planted defect is central/twist-like in the model-merging benchmark. | Not yet supported | The planted benchmark shows no rank-lift gain beyond C2M3 for either central or random defects, so central selectivity is not established. |
| TwistedMerge++ beats C2M3 on broad real MNIST/CIFAR model merging. | Not yet supported | `reports/validated_ladder_merge_report.md` supports a limited MNIST MLP win over the repo's internal C2M3-style baseline only; it is not CIFAR, not external C2M3, and not a broad architecture/dataset claim. |
| TwistedMerge++ beats an external C2M3 implementation. | Not yet supported | `reports/official_external_baseline_attempt.md` records that official C2M3 did not run on the exact MNIST MLP checkpoint set; the external-baseline layer still uses the repository's faithful internal C2M3-style permutation synchronization baseline, not official C2M3 output. |
| TwistedMerge++ beats Git Re-Basin. | Not yet supported | `reports/official_external_baseline_attempt.md` records that official Git Re-Basin did not run on the exact MNIST MLP checkpoint set; `reports/external_baseline_comparison.md` runs only a faithful Git-ReBasin-style pairwise alignment baseline. |
| TwistedMerge++ beats Model Soups. | Not yet supported | `reports/official_external_baseline_attempt.md` records that official Model Soups did not run on the exact MNIST MLP checkpoint set; `reports/external_baseline_comparison.md` runs a faithful greedy soup baseline and reports the improved selector below greedy soup, with paired mean accuracy delta `-0.0024`, CI `[-0.0042, -0.0009]`. |
| TwistedMerge++ beats greedy soup as a single-model baseline. | Not yet supported | The stronger improved benchmark reports `improved_validated_selector_vs_greedy_soup` paired mean accuracy delta `-0.0015`, CI `[-0.0023, -0.0008]`; the external-baseline run reports `-0.0024`, CI `[-0.0042, -0.0009]`. |
| Monomial-scaled greedy soup improves over ordinary greedy soup. | Not yet supported | `reports/csv/improved_validated_ladder_merge_summary.csv` reports raw/shrinkage/global/optimized monomial soup deltas versus greedy soup are nonpositive; `reports/csv/external_baseline_comparison_summary.csv` reports `monomial_scaled_greedy_soup` delta `-0.0014`, CI `[-0.0028, -0.0001]`. |
| Greedy-aware TwistedMerge++ beats greedy soup overall. | Not yet supported | `reports/csv/greedy_aware_monomial_paired_stats.csv` reports overall `greedy_aware_selector_vs_greedy_soup` paired mean accuracy delta `-0.001089`, CI `[-0.001692, -0.000559]`; the low-lr mode-specific positive result is not an overall win. |
| Lower-confidence greedy-aware selection beats greedy soup. | Not yet supported | `reports/csv/greedy_aware_monomial_paired_stats.csv` reports all ties versus greedy soup and mean delta `0.000000`, not a positive improvement. |
| Soup-compatible candidate generation beats greedy soup overall. | Not yet supported | `reports/csv/greedy_aware_monomial_paired_stats.csv` reports `union_candidate_soup_vs_greedy_soup` overall mean delta `-0.000479`, CI `[-0.000823, -0.000154]`. |
| Robust scale estimator variants or nested validation prove a new accuracy gain. | Not yet supported | The helpers are implemented and tested, but `reports/greedy_aware_monomial_report.md` does not claim a separate paired benchmark win for robust estimators or nested-validation replay rows. |
| Monomial centrality improvement is a reliable predictor of monomial accuracy gain. | Not yet supported | `reports/csv/validated_ladder_merge_summary.csv` shows only weak overall correlation and inconsistent fixed-setting correlations. |
| Monomial triangle residuals are more predictive than permutation cycle residuals in fixed-setting real model merging. | Not yet supported | `reports/monomial_gauge_alignment_report.md` is descriptive unless a monomial fixed setting reaches the same repeated-seed gate as the main verifier; no new predictor claim is promoted by the Prompt 7 smoke run. |
| Block-orthogonal alignment improves C2M3 merge accuracy on real MNIST/CIFAR. | Not yet supported | `reports/block_orthogonal_ladder_report.md` does not evaluate block-orthogonal merge performance for ReLU MLPs because general block rotations are not exact ReLU parameter symmetries. |
| Optimized global block synchronization improves natural MNIST/CIFAR merge accuracy. | Not yet supported | `reports/optimized_global_block_synchronization_report.md` and `reports/block_compatible_learning_report.md` cover controlled synthetic and exact linear-hidden MNIST settings; real ReLU rows remain diagnostic-only and CIFAR is not a supported block-gauge result. |
| TwistedMerge++ solves natural model merging. | Not yet supported | The current TwistedMerge++ artifacts test residual classification and selection logic only. Natural MNIST/CIFAR claims remain governed by `reports/model_merging_verification_report.md`. |
| TwistedMerge++ trivializes a nonzero `H^2(mu_2)` class as an ordinary vector bundle. | Not yet supported | `tests/test_twisted_merge_plus.py` and `reports/twisted_merge_plus_report.md` label the nonzero tetrahedral class as branch-only extra-capacity behavior, not ordinary untwisted descent. |
| TwistedMerge++ rank-lift gives a capacity-matched single-model improvement. | Not yet supported | The branch path is explicitly labeled `branch_lift_extra_capacity`; no capacity-matched single-model comparison has been run. |
| Real neural model-merging defects have the exact finite-index clock-shift form. | Not yet supported | `reports/finite_index_twist_report.md` is a controlled algebraic toy experiment, not a learned MNIST/CIFAR defect identification result. |
| Real neural model-merging defects have the higher period-index form `period=d, index=d^k`. | Not yet supported | `reports/period_index_central_report.md` is a controlled finite Heisenberg benchmark and explicitly does not claim MNIST/CIFAR residuals are Brauer/projective classes. |
| TwistedMerge++ beats C2M3 because of period-index detection. | Not yet supported | `reports/twisted_merge_plus_period_index_report.md` is a controlled selector demo; it does not run natural MNIST/CIFAR model merging or compare against C2M3 performance. |
| Natural MNIST/CIFAR residuals contain central period-index classes found by the robust detector. | Not yet supported | `reports/robust_period_index_detector_report.md` uses controlled synthetic Heisenberg and loop-mining examples only; it does not mine MNIST/CIFAR residuals. |
| Arbitrary independently trained neural hidden layers naturally expose central period-index classes. | Not yet supported | `reports/time_frequency_learned_chart_report.md` and `reports/time_frequency_denoised_chart_report.md` support controlled finite time-frequency paired chart recovery only; supervised/hidden-feature and MNIST/CIFAR period-index claims remain unsupported. |
| Projection-large or uncertain finite-Heisenberg projection candidates are valid lifts. | Not yet supported | `reports/time_frequency_heisenberg_projection_report.md` requires residual acceptance plus robust period/index certification and records rejected large-residual rows with `selected_method = none`. |
| Canonical finite-Heisenberg replacement without small learned-to-projected residual is learned recovery. | Not yet supported | `src/nearest_heisenberg_projection.py` rejects canonical replacement unless the commutator form matches and the projection residual threshold accepts; the report lists this as a negative boundary. |
| Uncertain robust period-index candidates are valid lifts. | Not yet supported | The robust detector explicitly labels loose-threshold cases `central_projective_candidate_uncertain`, and TwistedMerge++ keeps `selected_method = none` for those rows. |
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
| Post-projection block cycle score alone proves descent. | Not yet supported | `reports/block_gauge_phase_diagram_report.md` includes fake projection traps whose projected cycle score is near zero while connection residual remains large and strict calibration rejects the row. |
| Block-compatible identity/linear results imply natural ReLU or CIFAR block-gauge performance. | Not yet supported | The new block-compatible improvement is restricted to an exact linear-hidden architecture with identity activation; ReLU diagnostics are explicitly marked diagnostic-only. |
| Twisted sheaf cycle regularization improves GNNs in general. | Not yet supported | `reports/sheaf_gnn_optional_report.md` is a 27-row synthetic smoke test, and `reports/nsd_official_integration_report.md` is a tiny official smoke run with post-hoc diagnostics only. Neither is a broad real-dataset benchmark or an applied official cycle-regularization test. |

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
| `reports/fixed_setting_verification_report.md` | Stronger fixed-setting repeated-seed verification report for cycle residual versus ordinary merge degradation. |
| `reports/csv/fixed_setting_verification_runs.csv` | Per-method fixed-setting rows including observed/injected alignment labels and method-capacity metadata. |
| `reports/csv/fixed_setting_verification_stats.csv` | Fixed-setting Pearson/Spearman/bootstrap and controlled regression statistics. |
| `reports/csv/fixed_setting_triangle_defects.csv` | Per-triangle permutation/cocycle defect rows. |
| `reports/csv/fixed_setting_individual_models.csv` | Per-local-model validation/test accuracy and checkpoint metadata. |
| `reports/monomial_gauge_alignment_report.md` | Fixed-setting monomial gauge report comparing permutation and positive-monomial alignment for one-hidden-layer ReLU MLPs. |
| `reports/csv/monomial_fixed_setting_runs.csv` | Per-method rows for monomial fixed-setting runs, including scale statistics and functional preservation errors. |
| `reports/csv/monomial_triangle_defects.csv` | Per-triangle rows for monomial matching settings, including monomial defect scores. |
| `experiments/full_capacity_claim_audit.py` | Generates the repository-wide capacity, exact-symmetry, diagnostic, validation-selection, official-code, extra-capacity, residual-gate, and claim-boundary registry. |
| `reports/full_capacity_claim_audit.md` | Markdown report rendering the full capacity and claim-boundary audit. |
| `reports/csv/full_capacity_claim_audit.csv` | Authoritative full-width capacity and claim-boundary table. |
| `reports/tables/full_capacity_claim_audit.tex` | LaTeX longtable rendering of the audit. |
| `reports/cifar_bridge_boundary_summary.md` | Concise CIFAR/bridge appendix summary covering the failed CIFAR gate, bounded rescue, final confirmatory run, exact-gauge descriptiveness, greedy-soup boundary, and bridge-only limitations. |
| `reports/latex/cifar_bridge_appendix.tex` | LaTeX appendix section and compact table for CIFAR and bridge boundary evidence. |
| `reports/external_integration_summary.md` | Reproducibility appendix summary for official Git Re-Basin, C2M3, Model Soups, and Neural Sheaf Diffusion integration attempts, blockers, surrogates, and claim boundaries. |
| `reports/latex/external_integration_appendix.tex` | LaTeX appendix section/table documenting official external-code integration status and faithful in-repo comparison boundaries. |
| `experiments/unified_quantitative_obstruction_chain.py` | Aggregates existing obstruction, detector, gate, selector, model-merging, block-gauge, time-frequency, and optional sheaf/GNN CSV artifacts into one quantitative chain. |
| `reports/unified_quantitative_obstruction_chain.md` | Unified obstruction-chain report with link tests for cycle scores, validation deltas, operator/projection residuals, period-index decisions, dataset gates, and residual taxonomy. |
| `reports/csv/unified_quantitative_obstruction_chain.csv` | Row-level normalized obstruction-chain table with aligned residual, detector, lift, gate, validation, and merge-degradation fields. |
| `reports/plots/unified_obstruction_chain.pdf` | Four-panel diagnostic plot for planted degradation, selector deltas, operator-error certification, and projection-residual gates. |
| `reports/latex/quantitative_obstruction_theorem_candidate.tex` | Cautious LaTeX theorem-candidate snippet stating the artifact-scoped multi-residual decision chain. |
| `reports/results_narrative_after_35791f7.md` | Paper-ready Results and Contributions narrative after the final CIFAR boundary and bridge expansion. |
| `reports/latex/results_section_after_35791f7.tex` | Paste-ready LaTeX Results and Contributions section preserving the current claim boundaries. |
| `reports/latex/main_claims_table_after_35791f7.tex` | Paste-ready LaTeX table of supported claims and forbidden overclaims after the final CIFAR run. |
| `experiments/cifar_or_colored_mnist_feasibility.py` | Rotated-MNIST bridge and gated CIFAR probe feasibility benchmark for small CNN channel-gauge merging. |
| `reports/cifar_or_colored_mnist_feasibility.md` | Report for bridge/CIFAR accuracy gates, merge method summaries, and plumbing-only boundaries. |
| `reports/csv/cifar_or_colored_mnist_feasibility.csv` | Per-setting bridge/CIFAR feasibility rows with method metrics, threshold status, and validation-only selector metadata. |
| `reports/csv/cifar_or_colored_mnist_feasibility_summary.csv` | Method summaries and claim-gate decisions for rotated-MNIST and CIFAR. |
| `reports/configs/cifar_or_colored_mnist_feasibility_config.json` | Saved command, thresholds, git state, and environment metadata for the feasibility run. |
| `experiments/bridge_dataset_channel_gauge_expansion.py` | Expanded rotated/colored-MNIST bridge benchmark over angles, `N=3,4`, 10 main seeds, exact channel gauges, C2M3-style synchronization, and greedy baselines. |
| `reports/bridge_dataset_channel_gauge_expansion.md` | Bridge expansion report with bridge-only claim decisions, main-setting method table, overall boundary comparisons, and setting coverage. |
| `reports/csv/bridge_dataset_channel_gauge_expansion.csv` | Per-setting bridge expansion rows for all methods and validation/test metrics. |
| `reports/csv/bridge_dataset_channel_gauge_expansion_summary.csv` | Bridge expansion method summaries, paired comparisons, and claim-decision rows. |
| `reports/tables/bridge_dataset_channel_gauge_expansion.tex` | LaTeX table for the main rotated-25 `N=3` bridge setting. |
| `experiments/cifar_rescue_or_no_go.py` | Bounded CIFAR rescue/no-go gate with larger no-BatchNorm CNNs, normalization/augmentation, thresholded merge evaluation, and ensemble upper bound. |
| `reports/cifar_rescue_or_no_go_report.md` | CIFAR rescue report with gate decision, probe rows, merge/diagnostic rows, and claim boundaries. |
| `reports/csv/cifar_rescue_or_no_go.csv` | Per-setting CIFAR rescue rows for probes, merge methods, selectors, and ensemble upper bound. |
| `reports/csv/cifar_rescue_or_no_go_summary.csv` | CIFAR rescue method summaries and formal gate decision. |
| `experiments/cifar_final_channel_gauge_confirmatory.py` | Final bounded CIFAR-10 no-BatchNorm CNN channel-gauge confirmatory benchmark. |
| `reports/cifar_final_channel_gauge_confirmatory_report.md` | Final CIFAR report with exact command, method table, paired summaries, diagnostics, negative results, and branch decision. |
| `reports/csv/cifar_final_channel_gauge_confirmatory.csv` | Per-setting CIFAR final channel-gauge benchmark rows. |
| `reports/csv/cifar_final_channel_gauge_confirmatory_summary.csv` | CIFAR final method summaries, paired comparisons, selector behavior, and claim decisions. |
| `reports/tables/cifar_final_channel_gauge_confirmatory_table.tex` | LaTeX table for the final CIFAR channel-gauge benchmark. |
| `reports/plots/cifar_final_delta_vs_c2m3.pdf` | CIFAR final method deltas versus C2M3-style channel synchronization. |
| `reports/plots/cifar_final_delta_vs_greedy_soup.pdf` | CIFAR final method deltas versus greedy soup. |
| `reports/plots/cifar_final_selector_choices.pdf` | CIFAR final greedy-safe selector choice counts. |
| `reports/plots/cifar_final_channel_residuals.pdf` | CIFAR final channel residual diagnostics. |
| `reports/configs/cifar_final_channel_gauge_confirmatory_config.json` | Saved command and runtime metadata for the final CIFAR benchmark. |
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
| `tests/test_period_index_commutator_matrix.py` | Regression tests for commutator-matrix rank/index detection, rank-deficient forms, shuffled generators, composite brute force, unknown index, and noncentral rejection. |
| `experiments/period_index_commutator_matrix_demo.py` | Demo generator for central commutator-matrix period-index detection. |
| `reports/period_index_commutator_matrix_report.md` | Report for general commutator-matrix detector behavior, mathematical explanation, and negative boundaries. |
| `reports/csv/period_index_commutator_matrix_demo.csv` | Per-scenario central commutator-matrix period-index diagnostics. |
| `src/period_index_mining.py` | Noise models, unitary projection, noncentral controls, and synthetic loop-holonomy generator mining for robust period-index diagnostics. |
| `tests/test_robust_period_index_detector.py` | Regression tests for robust noisy detection, uncertain no-lift behavior, noncentral rejection, unknown-index no-overclaiming, synthetic mining, and TwistedMerge++ integration. |
| `experiments/robust_period_index_detector.py` | Controlled robust period-index detector experiment and report generator. |
| `reports/robust_period_index_detector_report.md` | Report for robust/noisy period-index detection, certified-only threshold policy, synthetic generator mining, and negative boundaries. |
| `reports/csv/robust_period_index_detector.csv` | Per-scenario robust period-index detector diagnostics and safety pass/fail rows. |
| `reports/csv/robust_period_index_detector_summary.csv` | Grouped robust detector summary by source, noise type, noise level, and detector status. |
| `tests/test_robust_period_index_calibration.py` | Regression tests for calibration CSV generation, exact zero-noise certification, obstructed-rank no-lift behavior, negative controls, trivial controls, and certified-only threshold recommendation. |
| `experiments/robust_period_index_calibration.py` | Multi-seed robust period-index calibration experiment over controlled central cases, rank-divisibility rows, noncentral controls, and threshold policies. |
| `reports/robust_period_index_calibration_report.md` | Report for multi-seed robust period-index calibration, noise transitions, false-positive rates, threshold recommendation, and negative boundaries. |
| `reports/csv/robust_period_index_calibration.csv` | Per-seed calibration rows for central positives, rank-divisibility rows, noncentral negatives, and trivial abelian controls. |
| `reports/csv/robust_period_index_calibration_summary.csv` | Grouped calibration rates for certification, uncertainty, rejection, false lifts, false positives, and mean detector metrics. |
| `reports/csv/robust_period_index_calibration_threshold_policies.csv` | Threshold-policy grid and recommended calibrated tolerance/margin row. |
| `reports/plots/robust_period_index_certification_rate.pdf` | Certification-rate curves by noise level for calibrated central positive cases. |
| `reports/plots/robust_period_index_false_positive_rate.pdf` | False-positive central/lift rate curves for calibrated negative controls. |
| `reports/plots/robust_period_index_noise_phase_diagram.pdf` | Aggregate certified/uncertain/rejected noise phase diagram for positive central cases. |
| `src/time_frequency_benchmark.py` | Finite time-frequency shift/modulation operators, realification utilities, known chart generators, and chirp/Gabor dataset helpers. |
| `tests/test_time_frequency_period_index_benchmark.py` | Regression tests for the time-frequency relation, realification, known chart period-index detection, rank obstruction, dataset shapes, and no-MNIST/CIFAR boundary. |
| `experiments/time_frequency_period_index_benchmark.py` | Natural finite time-frequency known-operator chart benchmark generator. |
| `reports/time_frequency_period_index_report.md` | Report for finite time-frequency period-index detection, rank thresholds, learned-chart non-evaluation, and negative boundaries. |
| `reports/csv/time_frequency_period_index_benchmark.csv` | Per-seed, per-rank time-frequency known-operator chart diagnostics. |
| `reports/csv/time_frequency_period_index_summary.csv` | Grouped certification, lift, pass-rate, and orbit-invariant classifier summary for the time-frequency benchmark. |
| `reports/plots/time_frequency_period_index_rank_threshold.pdf` | Lift-selection rate by candidate rank for finite time-frequency cases. |
| `reports/plots/time_frequency_period_index_detection_rates.pdf` | Detector-status rates for finite time-frequency known-operator chart cases. |
| `reports/configs/time_frequency_period_index_config.json` | Saved configuration, environment metadata, scope note, and level-2 non-evaluation marker for the time-frequency benchmark. |
| `src/time_frequency_learned_charts.py` | Paired input least-squares recovery, real-block-to-complex conversion, full-dimensional linear autoencoder chart recovery, supervised feature transition diagnostics, and noncentral controls for time-frequency charts. |
| `tests/test_time_frequency_learned_charts.py` | Regression tests for paired sample ids, learned chart recovery, rank obstruction, noisy no-lift behavior, encoder safety, noncentral controls, and report scope boundaries. |
| `experiments/time_frequency_learned_chart_benchmark.py` | Learned chart recovery benchmark over known-operator, input least-squares, linear autoencoder, and supervised encoder levels. |
| `reports/time_frequency_learned_chart_report.md` | Report for learned time-frequency chart recovery, method separation, rank thresholds, exploratory encoder boundaries, and negative claims. |
| `reports/csv/time_frequency_learned_chart_benchmark.csv` | Per-seed, per-rank learned chart recovery diagnostics and certified-only lift decisions. |
| `reports/csv/time_frequency_learned_chart_summary.csv` | Grouped learned chart certification, false-lift, rank-obstruction, operator-error, residual, and accuracy rates. |
| `reports/plots/time_frequency_learned_chart_operator_error.pdf` | Learned chart operator-error curves by method and noise level. |
| `reports/plots/time_frequency_learned_chart_certification_rate.pdf` | Certification-rate curves for known, input-learned, autoencoder, and supervised chart levels. |
| `reports/plots/time_frequency_learned_chart_rank_threshold.pdf` | Zero-noise lift-selection rate by candidate rank for learned chart levels. |
| `reports/configs/time_frequency_learned_chart_config.json` | Saved configuration, calibrated threshold policy, environment metadata, and scope boundaries for learned chart recovery. |
| `src/time_frequency_chart_denoising.py` | Denoised learned-chart recovery methods: validation-selected ridge, nearest orthogonal/unitary projections, complex-unitary projection, and pairwise global chart synchronization. |
| `tests/test_time_frequency_chart_denoising.py` | Regression tests for unitary projection, projected noncentral no-lift behavior, synchronization consistency reduction, low-noise denoised recovery, rank obstruction, and report smoke generation. |
| `experiments/time_frequency_denoised_chart_benchmark.py` | Multi-seed denoised learned-chart benchmark over calibrated robust period-index detection, rank thresholds, false lifts, and operator-error diagnostics. |
| `reports/time_frequency_denoised_chart_report.md` | Report for denoised learned-chart certification gains, operator-error changes, false-lift table, rank thresholds, and negative boundaries. |
| `reports/csv/time_frequency_denoised_chart_benchmark.csv` | Per-seed, per-noise, per-rank denoised learned-chart diagnostics with raw and denoised errors/residuals. |
| `reports/csv/time_frequency_denoised_chart_summary.csv` | Grouped denoised learned-chart certification, lift, false-lift, rank-obstruction, operator-error, and synchronization summaries. |
| `reports/plots/time_frequency_denoised_certification_rate.pdf` | Certification-rate curves for denoised learned-chart methods on index-divisible ranks. |
| `reports/plots/time_frequency_denoised_operator_error.pdf` | Operator-error curves comparing raw and denoised learned chart maps. |
| `reports/plots/time_frequency_denoised_rank_threshold.pdf` | Zero-noise lift-selection rate by candidate rank for denoised learned-chart methods. |
| `reports/configs/time_frequency_denoised_chart_config.json` | Saved denoised benchmark configuration, environment metadata, calibrated thresholds, and scope boundaries. |
| `src/nearest_heisenberg_projection.py` | Conservative commutator-form finite-Heisenberg projection with unitary preprocessing, exponent-matrix fitting, residual gating, and certified-only lift decisions. |
| `tests/test_nearest_heisenberg_projection.py` | Regression tests for exact/noisy projection recovery, large-residual rejection, noncentral and wrong-period controls, rank obstruction, index-rank lifting, and report smoke generation. |
| `experiments/time_frequency_heisenberg_projection_benchmark.py` | Multi-seed nearest finite-Heisenberg projection benchmark over learned chart maps, residual thresholds, rank decisions, and negative controls. |
| `reports/time_frequency_heisenberg_projection_report.md` | Report for Heisenberg projection certification gains, residual thresholds, rank thresholds, false lifts, negative controls, and negative boundaries. |
| `reports/csv/time_frequency_heisenberg_projection_benchmark.csv` | Per-seed, per-noise, per-threshold Heisenberg projection diagnostics with before/after detector status and residual gating. |
| `reports/csv/time_frequency_heisenberg_projection_summary.csv` | Grouped Heisenberg projection acceptance, certification gains, false lifts, false-positive central rates, and residual/error means. |
| `reports/plots/time_frequency_heisenberg_projection_certification_rate.pdf` | Effective certification-rate curves after residual gating. |
| `reports/plots/time_frequency_heisenberg_projection_residual.pdf` | Projection-residual curves by method and noise level. |
| `reports/plots/time_frequency_heisenberg_projection_false_lift.pdf` | False-lift-rate curves for projection and baseline methods. |
| `reports/configs/time_frequency_heisenberg_projection_config.json` | Saved Heisenberg projection benchmark configuration, calibrated thresholds, method list, and scope boundaries. |
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
| `src/block_sync_calibration.py` | Calibrated connection-residual acceptance and projection-trap classification helpers. |
| `src/block_compatible_merge.py` | Exact block-gauge reparameterization helpers for the controlled linear-hidden architecture. |
| `tests/test_global_block_synchronization.py` | Regression tests for planted global gauges, noisy projection residuals, noncentral holonomy rejection, and scalar block phase detection. |
| `tests/test_learned_block_partition.py` | Regression tests for deterministic learned block clustering and input validation. |
| `tests/test_optimized_global_block_synchronization.py` | Regression tests for optimized residual sync, planted gauge recovery, and projection-trap rejection. |
| `tests/test_global_learned_block_partition.py` | Regression tests for global learned non-contiguous block recovery and validation-only selection. |
| `tests/test_block_sync_calibration.py` | Regression tests for accepted-sync calibration and evidence classification. |
| `tests/test_block_compatible_merge.py` | Regression tests for exact linear-hidden block-gauge transforms and aligned averaging. |
| `experiments/global_block_synchronization_experiment.py` | Synthetic controls and MNIST diagnostic experiment for global/learned block synchronization. |
| `experiments/optimized_global_block_synchronization.py` | Controlled optimized-sync, calibration, learned-block, and linear-hidden block-compatible experiment generator. |
| `reports/global_block_synchronization_report.md` | Report for synthetic controls, real MNIST global block diagnostics, learned-block comparison, and negative boundaries. |
| `reports/optimized_global_block_synchronization_report.md` | Verification report for Prompt 5(j)(ii), including exact commands, calibration, synthetic controls, and negative boundaries. |
| `reports/csv/global_block_synchronization.csv` | Per-setting global block synchronization diagnostics. |
| `reports/csv/global_block_synchronization_summary.csv` | Grouped diagnostic summary for permutation, monomial, GL, pairwise block, and global block rows. |
| `reports/csv/optimized_global_block_synchronization.csv` | Per-row optimized sync, learned-block, and block-compatible control results. |
| `reports/csv/optimized_global_block_synchronization_summary.csv` | Grouped optimized sync and block-compatible summaries. |
| `reports/csv/optimized_global_block_synchronization_paired_stats.csv` | Paired deltas and decision labels for optimized sync, learned blocks, and block-compatible aligned averaging. |
| `reports/plots/optimized_global_block_connection_residuals.pdf` | Connection-residual plot for controlled optimized-sync rows. |
| `reports/plots/learned_blocks_vs_contiguous.pdf` | Validation residual comparison for learned versus contiguous block positive control. |
| `reports/plots/block_compatible_merge_accuracy.pdf` | Pseudo-label accuracy plot for the controlled linear-hidden block-compatible merge path. |
| `reports/configs/global_block_synchronization_config.json` | Saved configuration and environment metadata for the global block synchronization run. |
| `reports/configs/optimized_global_block_synchronization_config.json` | Saved configuration, calibration, and environment metadata for the optimized block synchronization run. |
| `experiments/block_gauge_phase_diagram.py` | Multi-seed block-gauge phase diagram, learned-block benchmark, exact block-compatible MNIST task, and ReLU diagnostic-only report generator. |
| `tests/test_block_gauge_phase_diagram.py` | Regression tests for exact gauge acceptance, noncentral rejection, fake projection traps, and scalar phase detection. |
| `tests/test_block_sync_calibration_phase_diagram.py` | Regression tests for strict, balanced, and loose diagnostic block-sync policies. |
| `tests/test_block_sync_calibration_floor.py` | Regression tests for raw/effective threshold separation, numerical-floor acceptance, fake/noncentral rejection, and scalar `mu_2` detection. |
| `tests/test_optimized_global_block_multi_seed.py` | Regression test that residual optimization does not worsen planted exact gauges over multiple seeds. |
| `tests/test_learned_block_partition_statistics.py` | Regression test that learned non-contiguous block controls beat contiguous partitions statistically. |
| `tests/test_block_compatible_learning_task.py` | Regression tests for exact linear-hidden block-gauge logit preservation, capacity matching, and ReLU diagnostic-only rows. |
| `tests/test_block_gauge_branch_closure.py` | Regression tests for closure-output generation, threshold-column reporting, ReLU diagnostic-only rows, and capacity-matched block-compatible aligned averaging. |
| `reports/block_gauge_phase_diagram_report.md` | Multi-seed phase-diagram report with calibration policy table, paired residual statistics, learned-block table, scalar/projective table, and claim decisions. |
| `reports/block_compatible_learning_report.md` | Exact linear-hidden MNIST block-compatible learning report with accuracy/loss table and paired deltas versus unaligned averaging and greedy soup. |
| `reports/relu_block_diagnostic_report.md` | Diagnostic-only ReLU block report with scalar/projective candidate fraction, residuals, and ReLU-compatible baseline references. |
| `reports/block_gauge_branch_closure_report.md` | Final paper-oriented 5(j) closure report with clean-run metadata, calibration floor table, phase claims, learned-block controls, block-compatible learning, ReLU negatives, and boundaries. |
| `reports/csv/block_gauge_phase_diagram.csv` | Per-row synthetic phase-diagram diagnostics over families, seeds, widths, block sizes, noise levels, and methods. |
| `reports/csv/block_gauge_phase_diagram_summary.csv` | Grouped phase-diagram acceptance, false-accept/reject, scalar-candidate, and residual summaries. |
| `reports/csv/block_gauge_phase_diagram_paired_stats.csv` | Bootstrap paired statistics for optimized-vs-spectral sync, learned-block controls, and block-compatible learning deltas. |
| `reports/csv/block_gauge_acceptance_by_noise.csv` | Acceptance, false-accept, false-reject, scalar-candidate, and residual summaries grouped by true family, noise level, and policy. |
| `reports/csv/learned_block_partition_benchmark.csv` | Per-seed learned non-contiguous block partition recovery and validation-residual rows. |
| `reports/csv/block_compatible_learning_benchmark.csv` | Per-seed exact linear-hidden MNIST block-compatible method rows with accuracy, loss, capacity, and residual metadata. |
| `reports/csv/relu_block_diagnostic_benchmark.csv` | ReLU diagnostic-only block rows sourced from the existing real MNIST diagnostic run. |
| `reports/plots/block_gauge_phase_diagram_acceptance.pdf` | Strict acceptance-rate phase-diagram plot by family and noise level. |
| `reports/plots/block_gauge_connection_residual_vs_noise.pdf` | Connection-residual plot by family and noise level. |
| `reports/plots/optimized_vs_spectral_residual_delta.pdf` | Optimized-minus-spectral residual improvement plot over the synthetic grid. |
| `reports/plots/learned_block_recovery.pdf` | Learned-block recovery and validation-residual plot. |
| `reports/plots/block_compatible_learning_accuracy.pdf` | Exact linear-hidden MNIST block-compatible method accuracy plot. |
| `reports/plots/relu_block_diagnostic_residuals.pdf` | ReLU diagnostic residual plot, with no block-orthogonal merge performance claim. |
| `reports/configs/block_gauge_phase_diagram_config.json` | Saved phase-diagram configuration, command, git state, policy table, and environment metadata. |
| `reports/configs/block_compatible_learning_config.json` | Saved exact block-compatible learning configuration and environment metadata. |
| `reports/configs/relu_block_diagnostic_config.json` | Saved ReLU diagnostic-only configuration and environment metadata. |
| `reports/configs/block_gauge_branch_closure_config.json` | Saved closure-run configuration with raw/effective threshold metadata and `dirty_worktree=false`. |
| `experiments/validated_ladder_merge_benchmark.py` | Validated MNIST MLP benchmark for C2M3, monomial scaling, validation-selected ladder merge, greedy soup variants, and ensemble. |
| `reports/validated_ladder_merge_report.md` | Report with paired statistics, selector behavior, residual correlations, and claim decisions for the validated ladder benchmark. |
| `reports/csv/validated_ladder_merge_benchmark.csv` | Per-setting validated ladder merge benchmark rows. |
| `reports/csv/validated_ladder_merge_summary.csv` | Method summaries, paired comparisons, selector behavior, residual correlations, and claim decisions. |
| `reports/plots/validated_ladder_merge_delta_vs_c2m3.pdf` | Scatter plot of monomial centrality improvement versus monomial accuracy delta over C2M3. |
| `src/improved_monomial_merge.py` | Shrinkage/clipped log-scale gauges, global positive scale synchronization, validation-only selectors, log-scale optimization, and union candidate soup helpers. |
| `tests/test_monomial_shrinkage.py` | Regression tests for alpha-zero C2M3 recovery, alpha-one raw monomial recovery, clipping/shrinkage, and exact ReLU output preservation. |
| `tests/test_global_monomial_synchronization.py` | Regression tests for planted global per-model scale recovery and reference log-scale estimates. |
| `tests/test_validation_selector_no_leakage.py` | Regression tests that validation selectors use validation metrics and margin rules only. |
| `tests/test_union_candidate_soup.py` | Regression test that union candidate soup outputs one capacity-matched MLP rather than an ensemble. |
| `experiments/improved_validated_ladder_merge_benchmark.py` | MNIST MLP benchmark for shrinkage/global/optimized monomial scaling, improved validation selector, and soup-competitive variants. |
| `reports/improved_validated_ladder_merge_report.md` | Report for validation-optimized monomial scaling, selector behavior, soup variants, paired comparisons, diagnostics, and negative boundaries. |
| `reports/csv/improved_validated_ladder_merge_benchmark.csv` | Per-setting improved validated ladder benchmark rows. |
| `reports/csv/improved_validated_ladder_merge_summary.csv` | Method summaries, paired comparisons, selector behavior, alpha/tau selection, soup behavior, residual correlations, and claim decisions. |
| `reports/plots/improved_validated_ladder_delta_vs_greedy.pdf` | Scatter plot of validation accuracy against test accuracy delta versus greedy soup. |
| `reports/plots/scale_shrinkage_alpha_selection.pdf` | Histogram of validation-selected shrinkage alpha values. |
| `reports/configs/improved_validated_ladder_merge_config.json` | Saved configuration and environment metadata for the improved validated ladder run. |
| `src/greedy_aware_monomial.py` | Conservative greedy-aware selectors, lower-confidence selector, robust positive-scale estimators, nested validation split helper, and selector-regret utilities. |
| `experiments/greedy_aware_monomial_benchmark.py` | Greedy-aware monomial benchmark generator over replayed MNIST rows plus low-lr soup-compatible fine-tuning rows. |
| `tests/test_greedy_aware_selector.py` | Regression tests for greedy fallback, challenger acceptance, loss slack, and no test-metric usage by the lower-confidence selector. |
| `tests/test_nested_validation_no_leakage.py` | Regression test for disjoint `train_inner`, `val_model`, and `val_selector` splits. |
| `tests/test_robust_scale_estimation.py` | Regression tests for positive finite robust scale estimates and outlier sensitivity. |
| `tests/test_soup_compatible_candidate_generation.py` | Regression tests for alpha-zero/alpha-one monomial behavior and capacity-matched soup output. |
| `tests/test_selector_regret_analysis.py` | Regression test for selector-regret accounting on a toy table. |
| `reports/greedy_aware_monomial_report.md` | Report for greedy-aware selectors, soup-compatible low-lr mode, paired comparisons, selector regret, alpha/tau behavior, diagnostics, and negative boundaries. |
| `reports/csv/greedy_aware_monomial_benchmark.csv` | Per-setting, per-method greedy-aware monomial benchmark rows. |
| `reports/csv/greedy_aware_monomial_summary.csv` | Method summaries for replayed and low-lr greedy-aware rows. |
| `reports/csv/greedy_aware_monomial_paired_stats.csv` | Paired bootstrap statistics for selector, soup, and monomial-scale comparisons. |
| `reports/csv/greedy_aware_selector_regret.csv` | Selector-regret, false-challenger, and missed-challenger diagnostics. |
| `reports/csv/soup_compatible_modes_summary.csv` | Soup ingredient count and accuracy summaries by mode and soup method. |
| `reports/csv/greedy_aware_monomial_diagnostic_correlations.csv` | Pearson/Spearman diagnostic correlations for monomial and selector gains. |
| `reports/csv/greedy_aware_monomial_claims.csv` | Machine-readable claim decisions for the greedy-aware run. |
| `reports/plots/greedy_aware_delta_vs_greedy.pdf` | Scatter plot of selector validation margin versus test accuracy delta over greedy soup. |
| `reports/plots/selector_regret_vs_margin.pdf` | Selector-regret plot for greedy-aware and lower-confidence selectors. |
| `reports/plots/soup_ingredient_counts.pdf` | Soup ingredient count plot by mode and soup method. |
| `reports/plots/monomial_alpha_tau_selection.pdf` | Histogram of selected monomial alpha values. |
| `reports/configs/greedy_aware_monomial_config.json` | Saved configuration and environment metadata for the greedy-aware monomial run. |
| `experiments/fashion_mnist_improved_ladder.py` | Fashion-MNIST MLP benchmark for improved validated ladder selector, monomial scaling variants, soup baselines, and residual taxonomy. |
| `reports/fashion_mnist_improved_ladder_report.md` | Fashion-MNIST validation report with paired method comparisons, selector behavior, residual taxonomy, plots, and claim decisions. |
| `reports/csv/fashion_mnist_improved_ladder.csv` | Per-setting Fashion-MNIST improved ladder rows. |
| `reports/csv/fashion_mnist_improved_ladder_summary.csv` | Fashion-MNIST method summaries, all paired comparisons, residual taxonomy, correlations, and claim decisions. |
| `reports/plots/fashion_ladder_delta_vs_c2m3.pdf` | Fashion-MNIST test accuracy deltas versus internal C2M3-style synchronization. |
| `reports/plots/fashion_ladder_delta_vs_greedy_soup.pdf` | Fashion-MNIST test accuracy deltas versus ordinary greedy soup. |
| `reports/plots/fashion_residual_taxonomy.pdf` | Fashion-MNIST residual taxonomy fractions across fixed settings. |
| `reports/tables/fashion_ladder_table.tex` | LaTeX summary table for the Fashion-MNIST improved ladder benchmark. |
| `reports/configs/fashion_mnist_improved_ladder_config.json` | Saved configuration and environment metadata for the Fashion-MNIST improved ladder run. |
| `src/greedy_safe_selector.py` | Greedy-safe validation-only selector modes: fixed margin, loss-aware margin, bootstrap/LCB, nested validation, and regret bound. |
| `tests/test_greedy_safe_selector.py` | Regression tests for greedy fallback, challenger acceptance, bootstrap/LCB behavior, nested accept split, and conservative regret-bound selection. |
| `experiments/fashion_mnist_greedy_safe_selector.py` | Fashion-MNIST MLP greedy-safe selector replay over the 5(m) candidate table. |
| `reports/fashion_mnist_greedy_safe_selector_report.md` | Report for Fashion-MNIST greedy-safe selector modes, false challenger rates, regret, paired deltas, and negative boundaries. |
| `reports/csv/fashion_mnist_greedy_safe_selector.csv` | Per-setting Fashion-MNIST MLP candidate and greedy-safe selector rows. |
| `reports/csv/fashion_mnist_greedy_safe_selector_summary.csv` | Greedy-safe selector method summaries, paired comparisons, choice counts, and claim decisions. |
| `reports/plots/fashion_greedy_safe_delta_vs_greedy_soup.pdf` | Fashion-MNIST greedy-safe validation margin versus test delta plot. |
| `reports/plots/fashion_greedy_safe_regret.pdf` | Greedy-safe selector regret plot for the lowest-false-challenger rows. |
| `reports/tables/fashion_greedy_safe_selector_table.tex` | LaTeX table of conservative Fashion-MNIST greedy-safe selector rows. |
| `src/cnn_channel_gauge.py` | No-BatchNorm small Fashion-MNIST CNN plus exact channel permutation and positive channel-scaling gauges. |
| `tests/test_cnn_channel_gauge.py` | Exactness tests for CNN channel permutation, positive scaling, combined gauges, parameter count, and inference-cost proxy. |
| `experiments/fashion_mnist_cnn_ladder.py` | Small CNN Fashion-MNIST channel-gauge ladder benchmark over channel permutation, positive scale, shrinkage/global scales, greedy soup, greedy-safe selector, and ensemble. |
| `reports/fashion_mnist_cnn_ladder_report.md` | CNN channel-gauge report with exactness status, method table, residual diagnostics, and claim decisions. |
| `reports/csv/fashion_mnist_cnn_ladder.csv` | Per-setting CNN channel-gauge benchmark rows. |
| `reports/csv/fashion_mnist_cnn_ladder_summary.csv` | CNN method summaries, paired channel-scale comparisons, residual diagnostics, and claim decisions. |
| `reports/plots/fashion_cnn_delta_vs_c2m3.pdf` | CNN method deltas versus channel-permutation C2M3. |
| `reports/plots/fashion_cnn_delta_vs_greedy_soup.pdf` | CNN method deltas versus greedy soup. |
| `reports/plots/fashion_cnn_channel_residual_taxonomy.pdf` | CNN residual taxonomy fractions showing no central/projective or finite-index candidate claims. |
| `reports/tables/fashion_cnn_ladder_table.tex` | LaTeX summary table for the CNN channel-gauge ladder benchmark. |
| `reports/configs/fashion_mnist_greedy_safe_selector_config.json` | Saved command, environment, and selector-grid metadata for the Fashion-MNIST greedy-safe selector replay. |
| `reports/configs/fashion_mnist_cnn_ladder_config.json` | Saved command, environment, and benchmark metadata for the Fashion-MNIST CNN ladder run. |
| `experiments/fashion_mnist_cnn_channel_gauge_confirmatory.py` | Confirmatory Fashion-MNIST CNN channel-gauge benchmark over N=3 and N=4 with optimized layer-gated scale grids and channel-gauge soup variants. |
| `reports/fashion_mnist_cnn_channel_gauge_confirmatory_report.md` | Confirmatory CNN report answering exactness, C2M3, greedy soup, greedy-safe selector, residual taxonomy, and paper-placement questions. |
| `reports/csv/fashion_mnist_cnn_channel_gauge_confirmatory.csv` | Per-setting confirmatory CNN channel-gauge benchmark rows. |
| `reports/csv/fashion_mnist_cnn_channel_gauge_confirmatory_summary.csv` | Confirmatory CNN method summaries, paired comparisons, selector behavior, diagnostics, and claim decisions. |
| `reports/plots/fashion_cnn_confirmatory_delta_vs_c2m3.pdf` | Confirmatory CNN method deltas versus channel-permutation C2M3. |
| `reports/plots/fashion_cnn_confirmatory_delta_vs_greedy_soup.pdf` | Confirmatory CNN method deltas versus greedy soup. |
| `reports/plots/fashion_cnn_confirmatory_channel_residuals.pdf` | Confirmatory CNN channel residual diagnostic scatter plot. |
| `reports/plots/fashion_cnn_confirmatory_selector_choices.pdf` | Confirmatory CNN greedy-safe selector choice counts. |
| `reports/tables/fashion_cnn_channel_gauge_confirmatory_table.tex` | LaTeX summary table for the confirmatory CNN channel-gauge benchmark. |
| `reports/configs/fashion_mnist_cnn_channel_gauge_confirmatory_config.json` | Saved command, environment, and grid metadata for the confirmatory CNN channel-gauge benchmark. |
| `external_baselines/README.md` | License-clean external-baseline documentation for Git Re-Basin, C2M3, Model Soups, internal controls, deviations, capacity matching, inference cost, and fairness boundaries. |
| `external_baselines/NSD_INTEGRATION.md` | Official Neural Sheaf Diffusion integration note with environment setup, blockers, smoke-run command, and optional-claim boundaries. |
| `experiments/external_baseline_comparison.py` | MNIST MLP external-baseline comparison generator using shared checkpoints/splits, faithful Git-ReBasin-style alignment, faithful C2M3-style synchronization, faithful greedy soup, monomial scaling, and validation-only selectors. |
| `external_baselines/OFFICIAL_INTEGRATION.md` | Official Git Re-Basin, C2M3, and Model Soups clone/install/run attempt log with commit hashes, licenses, environment requirements, checkpoint-interface blockers, and no-official-results boundary. |
| `reports/official_external_baseline_attempt.md` | Report summarizing the official external-code integration attempt and why no official baseline CSV/table was generated. |
| `reports/external_baseline_comparison.md` | Report for the external-baseline MNIST MLP comparison, integration status, method metadata, paired deltas, selector behavior, and negative boundaries. |
| `reports/csv/external_baseline_comparison.csv` | Per-setting, per-method external-baseline comparison rows with test/validation metrics, capacity/inference multipliers, exact-symmetry flags, and paired deltas versus C2M3, greedy soup, and weight average. |
| `reports/csv/external_baseline_comparison_summary.csv` | Overall and fixed-setting method summaries with bootstrap confidence intervals for paired accuracy deltas. |
| `reports/csv/external_baseline_individual_models.csv` | Per-checkpoint validation/test metrics for the trained MNIST MLP models used by the external-baseline comparison. |
| `reports/tables/external_baseline_comparison.tex` | LaTeX summary table for the external-baseline comparison. |
| `reports/plots/external_baseline_deltas.pdf` | Error-bar plot of paired accuracy deltas versus internal C2M3, greedy soup, and weight average. |
| `reports/configs/external_baseline_comparison_config.json` | Saved configuration, command, git state, and environment metadata for the external-baseline comparison. |
| `reports/sheaf_gnn_feasibility.md` | Feasibility report for Neural Sheaf Diffusion integration, supported datasets, dependency status, learned-map storage, and minimal local run. |
| `experiments/nsd_official_cycle_diagnostics.py` | Non-vendored wrapper that runs official NSD BundleSheaf in the separate PyG environment and computes triangle cycle diagnostics from cached learned transports. |
| `reports/nsd_official_integration_report.md` | Official NSD integration verification report with exact commands, environment, blockers, smoke-run result, cycle diagnostics, and unsupported boundaries. |
| `reports/csv/nsd_cycle_diagnostics.csv` | Tiny WebKB Texas official NSD diagnostic rows with test metrics, triangle counts, cycle scores, and cycle-regularizer status. |
| `reports/configs/nsd_official_integration_config.json` | Saved command and configuration for the official NSD diagnostic wrapper. |
| `experiments/sheaf_gnn_cycle_diagnostics.py` | Self-contained PyTorch synthetic heterophily benchmark with dense GCN, rotation-sheaf GNN, cycle diagnostics, and optional cycle regularizer. |
| `reports/sheaf_gnn_optional_report.md` | Optional sheaf/GNN diagnostic report with method summaries, heterophily slices, cycle correlations, and unsupported boundaries. |
| `reports/csv/sheaf_gnn_cycle_diagnostics.csv` | Per-seed synthetic GCN/sheaf rows with test accuracy, heterophily, triangle counts, cycle inconsistency, feature variance, and Dirichlet proxy. |
| `reports/plots/sheaf_gnn_cycle_vs_accuracy.pdf` | Scatter plot of learned sheaf cycle inconsistency versus test accuracy. |
| `reports/configs/sheaf_gnn_cycle_diagnostics_config.json` | Saved command, git state, runtime, and output metadata for the optional sheaf/GNN diagnostic run. |

| Rank-lift branch evidence is separated from branch-capacity matched non-obstruction controls. | Supported implementation | `src/rank_lift_baselines.py` adds random, validation-selected, and C2M3-cluster branch ensembles. `reports/csv/real_obstruction_paired_deltas.csv` marks rank-lift support only when observed paired CI lower bounds are positive against all three branch controls with at least 20 paired seeds. |

| The full stepwise greedy-soup empirical descent theorem is directly auditable for the checkpointed activation-matching fixed-setting MLP2 trajectory run. | Supported limited | `reports/greedy_soup_trajectory_report.md` and `reports/csv/greedy_soup_trajectory.csv` log directly observed candidate-soup validation accuracy/loss for every accepted and rejected candidate in the checkpointed activation settings; decision `directly_supported_for_checkpointed_activation_settings`; test metrics are final-selection evaluation only and are not used for selection. |

| Same-base task-vector baselines are evaluated separately from independent-seed rebasin baselines. | Supported descriptive | `reports/same_base_task_vector_report.md` records a same-base task-vector benchmark; 6 completed fixed settings with minimum `20` seeds; best mean method `task_arithmetic` on `mnist/mnist_digit_subsets/W128` with mean accuracy `0.8839`; the report preserves validation-only selection and no-broad-superiority boundaries. |

| SLERP is audited as a path-geometry baseline rather than a descent obstruction method. | Supported descriptive | `reports/slerp_barrier_geometry_report.md` compares SLERP against linear and aligned paths; mean validation max-loss barrier delta versus linear is `0.1901`; SLERP remains a path-geometry baseline, not a descent obstruction method. |

| Enriched generated-candidate descent envelopes are validation-only same-base selectors. | Supported exact-setting | `reports/descent_envelope_selector_report.md` records validation-only generated-candidate selectors; 16 exact-setting selector rows pass positive paired CIs; strongest row `greedy_soup_over_generated_candidates` on `mnist/mnist_digit_subsets/W128` has mean delta `0.0493`. No broad superiority claim is made. Overall multi-setting support is also flagged descriptively in the selector summary. |

<!-- prompt10-claim-audit:start -->
## Prompt 10 Verification Pipeline And Claim Boundary Audit

This section is generated by `experiments/generate_claim_audit.py`. It is deliberately conservative: fake-MNIST smoke rows are diagnostic only and never count as empirical support.

### Required Claim Statuses

| claim_id | status | evidence | safe_wording | forbidden_wording |
| --- | --- | --- | --- | --- |
| controlled_twisted_overlap_rank_lift | supported controlled | experiments/controlled_twisted_overlap_benchmark.py; reports/controlled_twisted_overlap_report.md; reports/csv/controlled_twisted_overlap_summary.csv | In the controlled central-twist benchmark, rank-lifted branches are supported as controlled obstruction-structured evidence. | This does not show that real neural residuals are Brauer classes or that rank lift is a capacity-matched single merged model. |
| training_quality_sweep | supported design choice | experiments/train_quality_sweep.py; reports/training_quality_sweep_report.md; reports/csv/training_quality_sweep.csv | The training-quality sweep supports choosing model-quality settings before the paper-grade verification run. | Do not use the sweep as evidence for obstruction prediction or merge-method superiority. |
| real_fixed_setting_obstruction_prediction | supported narrow single setting | experiments/model_merging_fixed_setting_verification.py; reports/fixed_setting_verification_report.md; reports/fixed_setting_full_run_interpretation.md; reports/csv/fixed_setting_verification_stats.csv | The fixed-setting script is the paper-grade real verification entry point. The Prompt 11 quality-gated MNIST/Fashion-MNIST run supports a narrow real obstruction-prediction claim for one Fashion-MNIST setting while keeping all unsupported settings visible. | Do not claim broad real-model prediction, MNIST support, monomial performance evidence from this Prompt 11 run, or general positive empirical support across settings. |
| monomial_gauge_functional_preservation | supported implementation | src/monomial_gauge_alignment.py; tests/test_monomial_gauge_alignment.py; reports/monomial_gauge_alignment_report.md | Positive monomial ReLU MLP gauges are implemented and tested as function-preserving transformations. | Do not turn exact functional preservation into a performance or generalization claim. |
| monomial_gauge_performance | not yet supported | reports/monomial_gauge_alignment_report.md is implementation/descriptive until full repeated-seed runs exist. | Monomial gauge performance remains an open empirical question in this audit layer. | Do not claim monomial gauges improve merge accuracy from implementation checks alone. |
| greedy_soup_win | not supported | reports/external_baseline_comparison.md and later audit reports treat greedy soup as a strong boundary baseline. | Greedy soup remains a strong boundary baseline that exact-gauge methods do not robustly beat under the current evidence. | Do not claim TwistedMerge beats greedy soup unless paired CIs directly support that exact comparison. |
| official_external_baseline_win | not supported | external_baselines/OFFICIAL_INTEGRATION.md; reports/official_external_baseline_attempt.md | Official external-code integration was attempted and documented, but no official baseline win is claimed. | Do not say TwistedMerge beats official Git-ReBasin, C2M3, Model Soups, or NSD baselines unless official-code runs produce those metrics. |
| real_brauer_projective_residual | not supported | reports/claims_audit.md; residual taxonomy reports; period-index detector reports | Real residuals remain non-Brauer under tested diagnostics; controlled period-index examples support the mathematics. | Do not call real MNIST, Fashion-MNIST, CIFAR, CNN, or block residuals Brauer/period-index classes under the current evidence. |

### Safe Abstract Wording

- We study model-merging residuals as descent defects and separate controlled obstruction evidence from real-model diagnostic evidence.
- Controlled central-twist experiments support rank-lifted branch constructions in a synthetic setting, while real fixed-setting obstruction-prediction claims remain gated by repeated-seed verification.
- Positive monomial ReLU gauges are implemented as exact function-preserving symmetries, but their performance advantage is not claimed without full repeated-seed support.
- Greedy soup and official external baselines remain claim boundaries; no official external-code win or robust greedy-soup win is claimed.
- Real residuals are reported as non-Brauer under tested diagnostics, while period-index and projective lifts are claimed only for controlled certified settings.

### Forbidden Wording

- TwistedMerge broadly beats greedy soup.
- TwistedMerge beats official external baselines.
- Fake-MNIST smoke runs support real empirical claims.
- Raw weight-average degradation is predicted unless the full fixed-setting gates pass.
- Monomial gauges improve performance based only on implementation checks.
- Real MNIST/Fashion-MNIST/CIFAR residuals are Brauer or period-index classes.
- The historical model_merging_benchmark.py verification mode is the paper-grade real run.

### Next Paper-Grade Real Run

Use `experiments/model_merging_fixed_setting_verification.py` for the next paper-grade real verification run. Keep `experiments/model_merging_benchmark.py --mode verification` as historical/descriptive context only.

<!-- prompt10-claim-audit:end -->

<!-- method-family-comparison:start -->
## Method Family Comparison Appendix

This section is generated by `experiments/method_family_comparison.py`. It is an appendix-level comparison across internal/fair-style method families, not an official external-baseline claim.

| claim_id | status | safe_wording | reason |
| --- | --- | --- | --- |
| twistedmerge_highest_accuracy | Not supported | Do not claim highest accuracy unless the exact paired comparison supports it. | The independent-seed MNIST MLP summary has greedy soup above the improved selector, and the shared-base task-vector benchmark is a separate regime. |
| twistedmerge_most_well_rounded | Supported | TwistedMerge is the most well-rounded framework among the methods studied, while greedy soup remains a very strong pure-accuracy baseline. | The coverage matrix marks TwistedMerge as the only compared family covering validation selection, gauge correction, cycle/holonomy diagnostics, central/projective obstruction detection, and conservative no-lift behavior. This is a qualitative structural statement, not an averaged metric leaderboard. |
| greedy_soup_strong_accuracy_baseline | Supported | Greedy soup remains a strong pure-accuracy boundary baseline. | Greedy soup is the top or near-top independent-seed pure-accuracy baseline, and task-vector methods are interpreted in a separate shared-base regime. |
| task_vector_methods_fixed_trivialization | Supported | Task Arithmetic/TIES/DARE are evaluated only in shared-base task-vector settings here. | The report keeps these methods out of independent-seed rebasin claims and uses the common-base task-vector artifact for their accuracy rows. |
| slerp_path_geometry_not_obstruction | Partially supported | SLERP is an internal path-geometry baseline in a fixed chart; it does not provide gauge synchronization or obstruction diagnostics. | The existing SLERP audit is path-geometry evidence and did not show an average barrier reduction; the no-gauge/no-obstruction part is structural. |
| c2m3_permutation_not_full_taxonomy | Supported | C2M3-style synchronization is the permutation cycle-consistency baseline, not a full residual taxonomy or period-index detector. | The coverage matrix gives C2M3 permutation and cycle/holonomy coverage but not monomial scaling, central/projective detection, or full conservative lift rejection. |

Claim boundary: the supported wording is `TwistedMerge is the most well-rounded framework among the methods studied, while greedy soup remains a very strong pure-accuracy baseline.` Do not claim broad highest-accuracy or official implementation wins from this artifact.
<!-- method-family-comparison:end -->
