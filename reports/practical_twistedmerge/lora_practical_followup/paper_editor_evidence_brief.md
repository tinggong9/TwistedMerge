# Paper-editor evidence brief: gauge-aware LoRA merging

## Recommended supported wording

> Across five independently trained groups of eight rank-4 residual adapters, equivalent well-conditioned rank-space reparameterizations produced substantial changes under naive factor averaging, while whitened TwistedMerge global synchronization and full-delta SVD were stable. Global synchronization operated directly on low-rank factors. In a process-isolated systems grid spanning dimensions 768-4096, ranks 4-32, and 4-16 adapters, it used no dense effective-update allocation and had lower measured incremental peak memory than deterministic dense SVD in every tested 4096-dimensional configuration.

The wording is suitable as a practical invariance/systems result. It should not be presented as an accuracy win.

## Exact quantitative support

- Trained groups: 5; adapters per group: 8; accepted primary scrambles: 300 group-family-scramble settings, with 20 scrambles per family.
- Individual preservation: maximum relative delta error `1.447298e-15`; maximum logit error `2.131628e-14`; zero prediction disagreement and zero validation-accuracy change.
- Dense-gauge naive factor average: maximum relative merged-delta change `1.034210e+01`; maximum logit change `1.156529e+01`; maximum prediction disagreement `0.579250`.
- Global synchronization: maximum relative merged-delta change `2.709900e-13`; maximum logit change `9.865442e-13`; zero prediction disagreement; zero dense allocations.
- Full-delta SVD: maximum relative merged-delta change `2.711270e-15`; maximum logit change `1.154632e-14`; zero disagreement.
- Group-bootstrap stability difference, global minus naive over all primary families: `-3.218202`, 95% CI `[-3.437573,-2.998831]`.
- Systems coverage: 336/336 method/shape cases, 1,008 timed trials after warmups, zero failures/timeouts.
- Measured memory: global synchronization below deterministic dense SVD in all 12 `4096 x 4096` rank/count cases; at least half the measured incremental peak RSS in 24/48 complete-grid comparisons.
- Representative `4096`, rank-8, eight-adapter case: `11.85 MB` versus `54.21 MB` incremental peak RSS and `0.009687 s` versus `0.061481 s` median time for global versus deterministic dense SVD.

## Required limitations

- Full-delta SVD is also gauge-invariant and was `0.051725` higher in mean trained-adapter test accuracy than global synchronization. Do not claim performance superiority or uniqueness.
- Cycle-aware alignment used full-delta SVD fallback in all 300 primary trained-adapter rows because the natural transition cycles exceeded the gate. Factor-only cycle-aware completion occurred only in the closing-cycle systems fixtures.
- The primary gauge claim covers orthogonal, positive-diagonal condition-at-most-8, and dense condition-at-most-30 transformations. The `1e8` boundary is excluded and produced 204 alignment failures.
- The trained corpus consists of one rank-4 residual feature layer and a classification head over frozen ResNet-18 features. It is not evidence across multi-layer transformer LoRA modules.
- Systems matrices are numerical fixtures scaled from trained factors. They support resource/numerical claims, not application accuracy.
- Runtime was not uniformly lower: global synchronization was faster in 10/12 `4096` cases, not all cases.
- No claim is supported about TIES, DARE, task arithmetic, soups, universal merge quality, natural Brauer classes, or unrestricted `GL(r)` robustness.

## Positive and negative result placement

- Main paper: one concise proposition/result paragraph for trained-adapter representation dependence and the invariant global/full-delta comparison; one systems sentence for dense-allocation avoidance and the uniform 4096-dimensional measured-memory result.
- Main or compact systems table: the representative `4096`, rank-8, eight-adapter memory/time row.
- Appendix: full five-group bootstrap table, per-family stability tables, all 336 systems cases, correctness probes, cycle fallback counts, and the ill-conditioned failure boundary.
- Negative-results paragraph: full-delta accuracy advantage, natural cycle-aware dense fallback, mixed runtime, and absence of a fingerprinting result.

## Claim-to-artifact map

| Claim | Artifact |
| --- | --- |
| Corpus reuse and no retraining | `reports/practical_twistedmerge/lora_practical_followup/reuse_manifest.csv` |
| Individual gauge preservation | `real_adapter_gauge/preservation.csv` |
| Naive representation dependence | `real_adapter_gauge/layerwise_stability.csv`; `model_stability.csv` |
| Five-group paired intervals | `real_adapter_gauge/paired_group_stats.csv` |
| Global/full-delta stability and cycle fallback | `real_adapter_gauge/runs.csv`; `report.md` |
| Dense-allocation sentinel and measured memory | `scalability/runs.csv`; `memory.csv` |
| Timing distribution | `scalability/timing.csv` |
| Comparable numerical quality and gauge probes | `scalability/correctness_probes.csv` |
| Implemented complexity | `scalability/complexity_audit.md` |
| Final claim decision | `reports/practical_twistedmerge/lora_practical_followup/final_assessment.md` |

## Forbidden wording

- TwistedMerge is the only gauge-invariant LoRA merge.
- TwistedMerge improves adapter accuracy or beats full-delta SVD.
- Cycle-aware TwistedMerge removed natural trained-adapter holonomy in factor space.
- The random systems grid proves application quality.
- The result covers arbitrary ill-conditioned rank-space transformations or general transformer LoRA stacks.
