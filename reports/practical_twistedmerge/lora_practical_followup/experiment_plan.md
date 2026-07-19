# LoRA practical follow-up experiment plan

## Frozen objective

Test whether harmless rank-space reparameterizations change merges of the already trained holonomy adapters, and whether factor-space TwistedMerge methods provide gauge stability and systems advantages without dense `m x n` update materialization. Accuracy superiority is not a target claim.

## Execution order and non-duplication

1. Complete the deduplication and corpus-reuse audit.
2. Load every reused checkpoint, feature cache, split manifest, and saved-logit artifact; require exact SHA-256 matches.
3. Run a preservation smoke that exercises artifact loading and one accepted gauge from each family without training.
4. Run the Phase A pilot on independent groups `0,1,2`, with 20 scrambles per accepted family.
5. Only if every pilot gate passes, run the confirmatory extension on the untouched independent groups `3,4`; aggregate pilot and extension rows without rerunning groups `0,1,2`.
6. Run the process-isolated factor-space scalability benchmark after Phase A is frozen.
7. Skip adapter fingerprinting unless Phase A/B outputs make it both nonduplicative and negligible-cost.
8. Write the final assessment and paper-editor evidence brief.

## Phase A protocol

Each independent training seed is one statistical unit and contains eight chart adapters. Every adapter supplies one trained residual layer `B A` with `B=up.weight` and `A=down.weight`. Heads are averaged identically across representations and are not gauge-transformed.

Accepted primary gauge families are orthogonal (condition number 1), positive diagonal (at most 8), and dense well-conditioned (at most 30). An ill-conditioned `1e8` family is reported separately and excluded from primary claims. A scramble is rejected before merge evaluation if its relative effective-delta error exceeds `1e-6`, maximum individual logit change exceeds `1e-5`, prediction disagreement is nonzero beyond floating tie ambiguity, or individual validation accuracy changes.

The frozen method set is:

- naive factor averaging;
- full effective-delta mean followed by deterministic rank-4 SVD;
- canonical-SVD factor averaging;
- TwistedMerge pairwise reference alignment;
- TwistedMerge global synchronization;
- TwistedMerge cycle-aware alignment;
- planted-scramble oracle alignment.

No TIES, DARE, soups, Git Re-Basin, C2M3, or broad merging baseline is rerun. For every method, the output rank is at most four. The layerwise table measures merged residual updates. End-to-end evaluation uses the merged residual adapter, the representation-independent averaged head, and the existing validation/test chart features. Scrambles remain dependent representations: metrics are first averaged within a training group, then paired group-bootstrap intervals are computed over independent groups only.

The pilot passes only if every accepted individual adapter is preserved, naive factor averaging is representation-dependent in at least one group, TwistedMerge global synchronization is materially more stable than naive averaging, the result repeats across the three pilot groups, ordinary validation performance is not materially worse than the unscrambled merge, and compared methods have the same output-rank cap. A failed pilot is reported and blocks the confirmatory extension.

## Phase B protocol

Benchmark dimensions `768`, `1024`, `2048`, and `4096`; ranks `4,8,16,32`; and adapter counts `4,8,16` in float32. `8192` is omitted unless a preliminary resource check shows safe headroom. Values use scale statistics derived from the trained holonomy factors. Random matrices are labeled systems/numerical fixtures, never application-performance evidence.

The method set is naive factor averaging, dense effective-delta averaging plus deterministic truncated SVD, randomized dense SVD, canonical factor-space compression, TwistedMerge pairwise alignment, global synchronization, and cycle-aware alignment. Factor-space methods operate on `B` and `A` plus rank-sized transition matrices. A materialization sentinel and tests must fail if those methods request an `m x n` buffer.

Measurements are process-isolated with one warmup and at least three timed trials where feasible. The reports separate median/p25/p75/min/max wall time, CPU time, peak RSS, analytical temporary bytes, stored bytes, dense-allocation count, success/failure/timeout, and correctness probes. Dense references use the same dimensions, adapter count, precision, and output rank. Larger cases may use reproducible probes when a dense reference fails or exceeds the resource boundary.

## Claim boundaries

A trained-adapter invariance claim requires the Phase A gates. A scalability claim requires measured lower peak memory, successful completion beyond a dense failure boundary, faster comparable-quality processing, or stable invariant output without dense materialization. Full-delta SVD remains a valid invariant baseline. Controlled gauge invariance does not imply natural holonomy, Brauer classes, universal accuracy gains, or uniqueness.
