# LoRA gauge practical follow-up: final assessment

Decision: **the practical gauge program passes its invariance and low-rank-native systems gates, but it does not support an accuracy-superiority claim**.

Source corpus: holonomy application commit `9c91bc707d1f44beb36fe0fdce43af9ce1be79ed`.

Experiment branch: `codex/lora-gauge-practical`.

No adapter was retrained. The program reused five independent training groups, each containing eight chart-specific rank-4 residual adapters over frozen ResNet-18 CIFAR-10 features.

## 1. Did the controlled gauge result reproduce on trained adapters?

Yes, for the three preregistered well-conditioned families. Across five independent groups and 20 scrambles per family, the maximum individual-adapter relative effective-update error was `1.447298e-15`, the maximum individual logit error was `2.131628e-14`, prediction disagreement was zero, and validation accuracy was unchanged. The pilot passed on groups 0-2 before groups 3-4 were evaluated.

## 2. Was naive factor averaging representation-dependent on trained adapters?

Yes, in every independent group. Under dense condition-at-most-30 gauges, naive factor averaging reached maximum relative merged-delta change `1.034210e+01`, maximum test-logit change `1.156529e+01`, and maximum prediction disagreement `0.579250`. The all-primary paired group-bootstrap difference in relative-delta change between global synchronization and naive averaging was `-3.218202`, with 95% CI `[-3.437573,-2.998831]`.

## 3. Was TwistedMerge stable across equivalent gauges?

Yes for the whitened factor-space methods under the accepted families. Global synchronization had maximum relative merged-delta change `2.709900e-13`, maximum test-logit change `9.865442e-13`, and zero prediction disagreement. Pairwise reference alignment was also stable, with maximum relative change `8.052867e-14`.

The cycle-aware result needs a narrower statement. Natural trained-adapter transition cycles exceeded the preregistered cycle gate in every one of the 300 primary rows, so cycle-aware alignment selected `fallback_full_delta_svd` rather than a factor-only cycle correction. This is a useful consistency diagnostic, not evidence of a natural holonomy-specific merge improvement.

## 4. Did full-delta SVD perform equally well?

It performed at least as well and is the main boundary baseline. Full-delta SVD had maximum relative merged-delta change `2.711270e-15`, maximum test-logit change `1.154632e-14`, and zero prediction disagreement. Its unscrambled mean test accuracy exceeded factor-space global synchronization by `0.051725` across the five groups. Cycle-aware fallback reproduced the full-delta SVD output on the primary trained-adapter rows.

Therefore the result does not support accuracy superiority, uniqueness, or a claim that full-delta SVD is inadequate. The TwistedMerge-specific benefit is low-rank-native execution plus global consistency diagnostics.

## 5. Did TwistedMerge offer a memory, runtime, or diagnostic advantage?

Yes for memory and diagnostics; runtime was favorable but not uniform.

- All 336 process-isolated shape/method cases completed, with 1,008 timed trials and zero failures or timeouts.
- Pairwise, global, and factor-only cycle-aware methods made zero dense effective-update allocations.
- At `4096 x 4096`, global synchronization used lower measured incremental peak RSS than deterministic dense SVD in all 12 rank/count configurations.
- Across the complete 48 shape/rank/count comparisons, global synchronization used at most half the measured incremental peak RSS in 24 cases. The minimum measured ratio was `0.0312705`.
- At `4096 x 4096`, global synchronization was faster in 10/12 configurations, with median runtime ratio `0.220642` versus deterministic dense SVD. Two configurations prevent a uniform runtime-superiority claim.
- The systems correctness fixture gave global synchronization mean relative error `1.43e-4` and maximum error `8.58e-4` from the exact effective-update mean, comparable to the deterministic dense method's mean `1.66e-4` and maximum `9.77e-4`.

## 6. At which dimensions did factor-space processing become advantageous?

Analytical temporary memory first fell below half the dense method at the smallest tested case, dimension `768`, rank `4`, four adapters. Because process-level RSS is noisy at small sizes, the stronger measured statement is that dimension `4096` is the first tested dimension where global synchronization used less incremental peak RSS in every rank/count configuration.

For the representative `4096 x 4096`, rank-8, eight-adapter case:

- global synchronization: `11,845,632` incremental peak-RSS bytes, `2,113,536` analytical temporary bytes, median `0.009687` seconds;
- deterministic dense SVD: `54,214,656` incremental peak-RSS bytes, `68,157,440` analytical temporary bytes, median `0.061481` seconds.

## 7. Did the method avoid dense update materialization?

Global synchronization and pairwise reference alignment did, in both the implementation sentinel and all scalability runs. The factor-only scalability version of cycle-aware alignment also did because the systems fixtures had closing cycles.

The natural trained-adapter cycle-aware path did not: its conservative diagnostic gate rejected the cycles and deliberately used dense SVD fallback. Claims must distinguish factor-space global synchronization from this safety fallback.

## 8. Is adapter fingerprinting useful and nonduplicative?

Potentially useful, but not established here. Exact gauge-equivalent detection can reuse the canonical and random-probe machinery, yet a defensible tool still needs separately validated rank-expanded, quantized, perturbed, and genuinely different adapter pairs plus abstention thresholds. That is a distinct experiment rather than negligible reuse. Optional Phase C was therefore skipped.

## 9. What is the strongest practical paper claim?

> Across five independently trained groups of rank-4 residual adapters, naive factor averaging changed under equivalent well-conditioned rank-space reparameterizations, while whitened TwistedMerge global synchronization and full-delta SVD remained stable. In the tested low-rank systems grid, global synchronization avoided dense effective-update materialization and used lower measured incremental peak memory than deterministic dense SVD in every `4096 x 4096` rank/count configuration.

This wording does not claim accuracy superiority, uniqueness, unrestricted ill-conditioned `GL(r)` robustness, or natural Brauer/holonomy structure.

## 10. Should the gauge experiment program stop?

Yes. The trained-adapter invariance question and low-rank-native systems question have been answered with positive but bounded evidence. Additional gauge scrambles, datasets, or generic baseline sweeps would not change the central boundary that full-delta SVD is invariant and had better task accuracy here. Resume only for a genuinely new question, such as full transformer-layer LoRA checkpoints or a separately preregistered fingerprint/deduplication utility.

## Negative results and limitations

- Full-delta SVD was numerically at least as invariant and had `5.1725` percentage points higher mean test accuracy than global synchronization across the trained groups.
- Every primary natural cycle-aware row used dense fallback; there was no natural factor-only cycle correction.
- The ill-conditioned `1e8` boundary produced 204 alignment failures: 68 each for pairwise, global, and cycle-aware methods. It is excluded from the primary claim.
- The trained adapters have one residual LoRA-form feature layer plus an averaged classification head. They are not a multi-layer PEFT transformer evaluation.
- The systems matrices use scale statistics from the trained adapters but remain numerical fixtures, not application-accuracy evidence.
- The local corpus artifacts do not embed a complete standalone license dossier for CIFAR-10 and the pretrained ImageNet weights.
- No broad baseline collection, biomedical experiment, holonomy classification, Brauer certification, period-index sweep, linter, or model training was repeated.

## Primary artifacts

- Deduplication and provenance: `reports/practical_twistedmerge/lora_practical_followup/deduplication_audit.md`; `reuse_manifest.csv`.
- Trained-adapter evidence: `real_adapter_gauge/runs.csv`; `layerwise_stability.csv`; `model_stability.csv`; `paired_group_stats.csv`; `capacity_cost.csv`; `report.md`.
- Scalability evidence: `scalability/runs.csv`; `memory.csv`; `timing.csv`; `correctness_probes.csv`; `complexity_audit.md`; `report.md`.
