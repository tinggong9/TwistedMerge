# Final experimental report

Execution commit: `4ab8022b5c7826c1584ba781a707fc491aa6ede6`.

## Stage status and protocol coverage

- A1: `completed`; runtime `279.3070809589699` seconds; trained Fashion-MNIST chart inference.
- A2: `completed`; runtime `77.19020387483761` seconds; end-to-end controlled systems audit.
- A3: `completed`; runtime `0.7053552500437945` seconds; nontrivial refinement invariance.
- B1: `completed`; runtime `556.3412893330678` seconds; full-model hidden-layer geometry.
- B2: `completed`; runtime `954.2150744590908` seconds; strong learned compositional baselines.
- B3: `completed`; runtime `65.96659733401611` seconds; genuine multiview coordinate retransport.
- B4: `completed`; runtime `0.6763165409211069` seconds; new realistic residual search.
- B5: `completed`; runtime `631.3642158329021` seconds; structure-preserving compression.
- B6: `completed`; runtime `2.3194605000317097` seconds; noncyclic central extensions.
- B7: `completed_gate_closed`; runtime `0.11381466686725616` seconds; conditional official baseline integration.
- C1_C2: `completed_gate_closed`; runtime `0.08790099993348122` seconds; conditional extended vision and adapter families.
- C3: `completed`; runtime `88.5313935840968` seconds; language checkpoint transition geometry.
- C4_C5: `completed`; runtime `0.28467991715297103` seconds; comparison-complex and alignment robustness.
- C6: `completed`; runtime `1.8229343749117106` seconds; selective activation diagnostics.
- C7: `completed`; runtime `2.4821250420063734` seconds; real runtime and memory scaling.
- C8: `completed`; runtime `8.778634292073548` seconds; reproducibility and release manifests.

All selected stages use separate training, transition/router, selector, calibration, and test roles where applicable. Candidate logits are saved before test-label metrics and checked after label permutation. Discovery gates control confirmation and conditional extensions.

## Numerical paired results

- `reports/next_program/iclr/composition_paired.csv`: 4 paired rows; first row `{"best_equivalent_information_method": "symbolic_word_reduction_oracle", "ci_high": "0.0", "ci_low": "0.0", "group": "A4", "mean_delta": "0.0", "reference": "exact_structured_retransport"}`.
- `reports/next_program/iclr/full_model_paired.csv`: 2 paired rows; first row `{"alternative": "strict_synchronization", "ci_high": "0.0125", "ci_low": "0.005099999999999971", "mean_delta": "0.008799999999999985", "reference": "structured_retransport_certified_only"}`.
- `reports/next_program/iclr/multiview_paired.csv`: 2 paired rows; first row `{"alternative": "generic_calibration_network", "ci_high": "0.011", "ci_low": "-0.0010000000000000009", "mean_delta": "0.004999999999999999", "reference": "inferred_view_structured_retransport"}`.
- `reports/next_program/immediate/chart_paired.csv`: 2 paired rows; first row `{"best_generic": "generic_low_rank_context_adapter", "ci_high": "0.38550000000000006", "ci_low": "0.3268", "criterion_a": "True", "criterion_b": "True", "criterion_c": "True", "gate_passed": "True", "mean_accuracy_delta": "0.35629999999999995", "phase": "discovery", "structured_method": "chart_abstaining_structured_retransport", "worst_condition_ci_high": "0.27920001149177553", "worst_condition_ci_low": "0.17840000092983246", "worst_condition_delta": "0.22000000774860382"}`.
- `reports/next_program/immediate/cost_paired.csv`: 6 paired rows; first row `{"alternative": "context_blind_synchronization", "ci_high": "0.08867309570312498", "ci_low": "0.048094482421875", "mean_accuracy_delta": "0.067626953125", "reference": "structured_group_retransport"}`.

## Actual implementations and boundaries

- A1, A2, B1, B2, B3, B5, C3, C6, and C7 execute trained models or measured end-to-end numerical paths.
- A3, B6, and C4 execute exact finite-algebra or comparison-complex calculations.
- B1 names internal activation-alignment implementations explicitly; B7 does not relabel them as official baselines.
- B4 reuses preregistered B1/B3 discovery artifacts without selecting on test accuracy.
- C1/C2 and B7 remain gated when their prerequisites fail; gated rows are not counted as executions.

## Failed attempts

- B3: `RuntimeError: Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead.`.
- B4: `FileNotFoundError: B4 requires completed B1 and B3 discovery artifacts: /Users/tinggong/Documents/Codex/2026-07-14/ru-3/work/TwistedMerge-next-evidence/reports/next_program/iclr/multiview_transitions.csv, /Users/tinggong/Documents/Codex/2026-07-14/ru-3/work/TwistedMerge-next-evidence/reports/next_program/iclr/multiview_stability.csv, /Users/tinggong/Documents/Codex/2026-07-14/ru-3/work/TwistedMerg`.

## Negative findings

- `reports/next_program/extended/language_claims.csv`: `residual_exceeds_nulls` = false.
- `reports/next_program/extended/language_claims.csv`: `rank_stable` = false.
- `reports/next_program/extended/language_claims.csv`: `correction_reduces_interference` = false.
- `reports/next_program/extended/language_claims.csv`: `adapter_subspace_available` = false.
- `reports/next_program/extended/language_claims.csv`: `complete_language_gate_passed` = false.
- `reports/next_program/iclr/composition_claims.csv`: `twistedmerge_specific_compositional_advantage` = false.
- `reports/next_program/iclr/composition_claims.csv`: `twistedmerge_specific_compositional_advantage` = false.
- `reports/next_program/iclr/composition_claims.csv`: `twistedmerge_specific_compositional_advantage` = false.
- `reports/next_program/iclr/composition_claims.csv`: `twistedmerge_specific_compositional_advantage` = false.
- `reports/next_program/iclr/compression_claims.csv`: `structured_compression_gate_passed` = false.
- `reports/next_program/iclr/compression_claims.csv`: `teacher_eligible` = false.
- `reports/next_program/iclr/compression_claims.csv`: `teacher_executed` = false.
- `reports/next_program/iclr/full_model_claims.csv`: `residual_exceeds_all_matched_nulls` = false.
- `reports/next_program/iclr/full_model_claims.csv`: `residual_rank_stable_five_resamples` = false.
- `reports/next_program/iclr/full_model_claims.csv`: `structured_correction_activated` = false.
- `reports/next_program/iclr/full_model_claims.csv`: `complete_realistic_gate_passed` = false.
- `reports/next_program/iclr/multiview_claims.csv`: `residual_survives_matched_nulls` = false.
- `reports/next_program/iclr/multiview_claims.csv`: `residual_rank_stable` = false.
- `reports/next_program/iclr/multiview_claims.csv`: `inferred_retransport_beats_generic_methods` = false.
- `reports/next_program/iclr/multiview_claims.csv`: `complete_multiview_gate_passed` = false.
- `reports/next_program/iclr/natural_claims.csv`: `selection_used_test_accuracy` = false.
- `reports/next_program/iclr/natural_claims.csv`: `complete_new_family_gate_passed` = false.
- `reports/next_program/immediate/chart_claims.csv`: `cifar_triggered` = false.

## Artifact paths

- `reports/next_program/experiment_manifest.csv` maps tables and plots to scripts, raw data, summaries, execution commit, and checksums.
- `reports/next_program/artifact_checksums.csv` contains artifact hashes.
- `reports/next_program/test_results.txt` contains the executed test command and output.
- `reports/next_program/commands.csv` and `reports/next_program/failures.csv` preserve command and failure provenance.
