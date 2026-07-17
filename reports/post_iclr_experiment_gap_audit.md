# Post-ICLR Experiment Gap Audit

Audit baseline: commit `7a0620bb19dffba97012350b6ffd20684bcbe220` on `main`, inspected from the isolated `codex/post-iclr-experiments` worktree. The authoritative checkout was clean and matched `origin/main` when the audit began. No manuscript, LaTeX, bibliography, or paper-draft file was modified.

## Audit method

The audit covered the README, claim ledgers, capacity and regime audits, official-baseline attempt, monomial and CNN gauge reports, CIFAR boundary reports, controlled-overlap and obstruction-prediction reports, primary/nonabelian/small-quotient holonomy reports, period-index reports, selector reports, recent next-program/compact/future-program manifests, tracked experiment and test entry points, and the local ignored data/checkpoint stores.

The repository currently contains 187 tracked experiment entry points, 64 tracked source modules, 150 tracked test files, 640 tracked report CSVs, 153 tracked plot files, and 215 tracked report Markdown files. The authoritative local checkout additionally contains 3.3 GB under `data/`, 1,798 files under `reports/checkpoints/` (1,440 NPZ and 358 PT files), 260 files under `reports/csv/`, and 114 files under `reports/plots/`. The external-baseline checkpoint store contains 20 MNIST MLP settings and 70 original `model_*.pt` checkpoints for `N in {3,4}`, widths 32/64, and seeds 1800--1804.

## Regime-level finding

The existing regime separation is scientifically sound and must remain intact:

- Independent-initialization/rebasin evidence uses independently trained checkpoints and compares weight averaging, pairwise/global synchronization, exact gauges, soups, and upper-bound ensembles.
- Common-base task-vector evidence uses one base checkpoint and compares Task Arithmetic, TIES, DARE, SLERP, soups, and validation-selected candidate envelopes.
- Controlled obstruction evidence supplies known cocycles or holonomy and is not evidence that natural image-model residuals are Brauer or projective classes.
- Greedy-soup descent evidence concerns validation-selected candidate trajectories and is not interchangeable with either of the preceding regimes.

## Gap map

| Proposed experiment | Existing evidence | Audit status | Distinct contribution still needed |
| --- | --- | --- | --- |
| Official Git Re-Basin on independent checkpoints | Faithful internal pairwise alignment and an older blocked official-source inspection | Partially complete | Execute official author code through a declared checkpoint adapter if compatible; otherwise preserve a fresh reproducible failure row. |
| Official C2M3 on independent checkpoints | Faithful internal cycle synchronization and an older blocked import/interface inspection | Partially complete | Test the pinned official matcher on the exact saved checkpoints; distinguish unmodified, adapter-assisted, and blocked outcomes. |
| Official Model Soups on independent checkpoints | Faithful in-repo greedy soup on the exact checkpoint pool | Partially complete | Run author-provided selection code if it can accept an explicit evaluator/checkpoint adapter; do not relabel the internal implementation. |
| Official common-base Task Arithmetic/TIES/DARE | Internal same-base benchmark and descent-envelope selector | Partially complete | Pin official repositories and execute compatible algorithms on the exact common-base checkpoints, separately from rebasin results. |
| CIFAR-10 ResNet-18 independent merge | No-BatchNorm small-CNN CIFAR boundary runs; frozen-backbone or proxy ResNet programs do not satisfy this protocol | Absent | Credible independently trained ResNet-18 groups with three/four-model merges, at least five confirmatory groups, full capacity and base-accuracy gates. |
| CIFAR-100 ResNet-18 independent merge | Frozen-feature/pretrained exploratory artifacts only | Absent | Same full-model protocol on CIFAR-100 with honest compute and base-quality accounting. |
| BatchNorm-aware channel permutation | Exact no-BatchNorm channel permutations are tested | Absent | Explicit adjacent-layer and BatchNorm parameter/running-stat transformation with many-batch functional-preservation tests. |
| Positive BatchNorm-aware channel scaling | Exact no-BatchNorm positive scaling is tested | Absent | Derive whether each treatment is exact or approximate under epsilon and running statistics; test frozen, transformed, recalibrated, and recomputed statistics. |
| Secondary modern architecture | Several frozen-backbone/proxy ResNet/DeiT explorations exist | Deferred by gate | Select exactly one architecture only after full ResNet-18 succeeds; test a missing gauge or synchronization mechanism. |
| Real-task planted synchronizable case | Controlled overlap and planted-obstruction MLP experiments exist | Partially complete | Rebuild in one clean benchmark with ground-truth transitions, detector blindness to labels, unified capacity controls, and untouched test evaluation. |
| Real-task exact permutation case | Exact MLP/CNN gauge tests exist across separate scripts | Partially complete | Integrate known transition maps and selector decisions in the unified planted benchmark. |
| Real-task exact positive monomial/channel case | Positive ReLU gauges and functional tests exist | Partially complete | Integrate into the unified planted benchmark with capacity-matched controls and detector recovery. |
| Real-task controlled central/projective obstruction | Controlled `H^2(mu_2)` overlap/rank-lift evidence exists | Partially complete | Bridge to a real classification task in one preregistered benchmark without claiming a natural Brauer class. |
| Real-task controlled nonabelian finite holonomy | Structural branch-lift machinery exists, but the old controlled nonabelian accuracy family is quarantined as invalid because logits were target-injected | Absent valid empirical result | Generate predictions from actual models, save logits before labels, use invariant pooling and branch/random/capacity controls. |
| Diagnostic harmful-merge prediction | Narrow alignment-conditioned prediction and newer diagnostic-prediction artifacts exist; raw-average prediction is negative | Partially complete | Freeze a target-complete feature model and report AUROC/AUPRC/calibration/coverage-risk on untouched test settings. |
| Conservative diagnostic selector | Greedy-safe and matched-selector experiments show safe fallback behavior in bounded settings | Partially complete | One preregistered selector spanning ordinary candidates and certified lifts, with false/missed-lift rates and regret against greedy/oracle. |
| Lightweight biomedical multi-site pilot | A Kvasir segmentation/spatial-output program and synthetic-domain expert work exist; it is not the requested classification-site protocol | Partially complete but distinct | Use a lightweight public biomedical classification dataset, explicit simulated sites, post-hoc merge baselines, AUROC/F1/calibration, and no clinical claim. |
| Confirmatory biomedical benchmark | No requested second classification dataset/site protocol | Absent and gated | Run only after the pilot passes data, base-quality, and selector-validity gates. |
| LoRA merging | Exploratory LoRA/adapter artifacts exist, but the request explicitly defers a campaign | Deferred | Produce design only after official, modern-vision, planted, and selector phases close. |
| Additional MNIST/Fashion/period-index/holonomy sweeps | Extensive completed evidence | Complete for current purpose | Regression-only use; no new large sweep without a missing artifact or code change. |

## Existing evidence that must not be duplicated

- Controlled central period/index rank thresholds and the `TwistedMerge rank lift` already have substantial controlled evidence. Real residual lift gates remain negative because no real torsion candidate is certified.
- Prime-primary, nonabelian, small-quotient, and residual-peeling programs already expose both structural and negative results. The target-injected controlled-nonabelian accuracy family remains invalid empirical evidence and must not be revived by copying its metrics.
- Exact positive monomial gauges for ReLU MLPs and exact positive channel gauges for no-BatchNorm CNNs already have focused preservation tests. Their limited practical results do not establish a BatchNorm-aware method.
- The bounded CIFAR-10 no-BatchNorm CNN run passed its base-accuracy gate but produced descriptive/negative gauge performance. It is not a substitute for independently trained ResNet-18 on CIFAR-10/100.
- Same-base task-vector and descent-envelope artifacts already demonstrate exact-setting validation descent. They must not be pooled with independent-seed rebasin results.
- Greedy-safe selectors can exactly fall back to greedy soup in the tested replay. They do not yet provide the requested unified false-lift/missed-lift analysis.
- The recent biomedical segmentation program is valuable spatial-output evidence but has synthetic domains and no real center/site metadata; it does not answer the requested biomedical classification-site questions.

## Highest-priority unfinished phase

Priority A remains unfinished. The older official-integration attempt was a well-documented negative result, but it did not create the requested post-ICLR run manifest or retry against today's environment. The next phase will therefore:

1. pin and clone the official sources outside the tracked tree;
2. probe current imports and exact checkpoint compatibility;
3. permit only explicit adapter-assisted use of official algorithm code;
4. run the exact available checkpoint set where technically valid;
5. record every success, failure, patch, license, commit, conversion, and output type;
6. keep official and internal results visibly separate.

A blocked official method is a completed integration attempt, not a performance result. It will produce a failure/status row and no fabricated metric row.
