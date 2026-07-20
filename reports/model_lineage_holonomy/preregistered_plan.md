# Preregistered model-lineage holonomy plan

Plan frozen: 2026-07-20, before new feature extraction or training.

## Objective and stopping rule

The sole objective is to test whether natural representation holonomy over sequential CIFAR-10 task lineages predicts task-order sensitivity or harmful same-base branch merges beyond pairwise drift, and whether one conservative cycle-aware synchronization rule improves merge regret or abstention.

If none of H1-H4 passes after the five-seed analysis, the experiment stops with the required negative interpretation. No additional dataset, corruption, backbone, transition family, predictor, or cycle estimator will be added.

## Frozen data, model, and seed design

- Dataset: the existing local CIFAR-10 archive only; `download=False` is mandatory.
- Base encoder: cached torchvision ResNet-18 `IMAGENET1K_V1`, fully frozen.
- Starting models: chart-0 adapters from the five completed holonomy-corpus seed bundles. Each is a rank-4 residual map in the existing train-only 64-dimensional projection plus a ten-class head.
- Independent training seeds: pilot `0,1,2`; confirmatory extension `3,4`. A five-seed result is confirmatory; three-seed output is pilot-only.
- Split seed: `7192026`. Train-side sizes are 2,500 adaptation identities, 384 transport-fit identities, 384 transport-validation identities, 384 transport-test identities, and 512 model-validation identities. The application test split contains 1,000 CIFAR-10 test identities.
- The transport anchor contains A, B, and C versions of every identity in identical domain and identity order. Transport splits are disjoint and transition fitting never reads labels.

## Frozen domains

Inputs remain in `[0,1]` before encoder normalization.

- A, Gaussian noise: add per-identity deterministic Gaussian noise with standard deviation `0.15`, seed `42101 + dataset_index`, then clamp.
- B, blur: depthwise `5 x 5` Gaussian blur with standard deviation `1.0` and reflect padding.
- C, color/contrast: multiply centered contrast by `1.35`, then apply RGB scales `(1.10, 0.90, 1.00)` and offsets `(0.04, -0.02, 0.00)`, then clamp.

No clean-domain or alternative-corruption result may be substituted after inspection.

## Frozen lineage graph and training budget

Each seed contains:

- `M0`;
- `M_A`, `M_B`, `M_C`;
- `M_AB`, `M_BA`, `M_AC`, `M_CA`, `M_BC`, `M_CB`;
- `M_ABC`, `M_ACB`, `M_BAC`, `M_BCA`, `M_CAB`, `M_CBA`.

Every child copies its named parent and receives exactly 12 epochs on the appended task, batch size 256, AdamW learning rate `0.003`, weight decay `0.0001`, with optimizer state reset at each edge. The task-specific shuffle stream is fixed per independent seed, task, and epoch and does not depend on the path. There is no early stopping. Trainable capacity is the existing rank-4 residual adapter plus classifier; the encoder and projection remain frozen.

Every checkpoint record includes parent, appended task, independent seed, examples and optimizer steps, validation metrics on A/B/C, first-task forgetting, state hash, trainable/total parameters, stored bytes, and exact command/source/config hashes.

## Frozen representation layers

The same examples and ordering are used at every node.

- `early`: pooled ResNet layer1, deterministically projected to 32 dimensions;
- `mid`: pooled ResNet layer3, deterministically projected to 32 dimensions;
- `late`: the existing train-only projected encoder representation, 64 dimensions;
- `adapter`: the rank-4 residual contribution, 64 dimensions;
- `penultimate`: late plus adapter contribution, 64 dimensions.

The first three are encoder-frozen controls and are stored once with hashes referenced by every checkpoint. Adapter and penultimate representations are saved for every checkpoint and anchor split.

## Frozen transition estimators

Four unlabeled estimators are fit on transport-fit anchors:

1. orthogonal Procrustes;
2. ridge linear transport with ridge `1e-3`;
3. rank-8 low-rank residual transport;
4. whitened CCA-style orthogonal transport with covariance ridge `1e-4`.

For each layer and directed graph relation, the estimator with the lowest transport-validation normalized residual is selected, with alphabetical method name as an exact-tie breaker. Transport-test anchors are used only after selection. Diagnostics include fit/validation/test residual, inverse consistency, condition number, effective rank, singular spread, layer, pair, path relation, and 100 within-anchor bootstrap replicates. Anchor bootstraps are stability diagnostics, not independent inferential units.

## Frozen natural loops

Two-task squares are:

- `M0 -> M_A -> M_AB -> M_B -> M0`;
- `M0 -> M_A -> M_AC -> M_C -> M0`;
- `M0 -> M_B -> M_BC -> M_C -> M0`.

Three-task adjacent-swap comparisons are `ABC/BAC`, `ABC/ACB`, `ACB/CAB`, `BAC/BCA`, `BCA/CBA`, and `CAB/CBA`. Each closes with a learned terminal-to-terminal map and the reverse lineage path. Every loop/layer records identity distance, spectral radius, singular-value spread, normalized trace, determinant sign/log magnitude where finite, nearest-orthogonal distance, bootstrap interval, validation stability, and conjugacy-invariant eigenvalue modulus/phase summaries. Independent-loop commutators are reported. These objects are representation holonomies only, never Brauer classes.

A loop is called stably nonidentity only when its transport-test identity-distance bootstrap lower bound exceeds `0.05`, its bootstrap interval width is at most `0.15`, and every selected edge has validation residual at most `0.35` and condition number at most `10^4`. A commutator is called stably nontrivial only when the corresponding lower bound exceeds `0.03` under the same edge-quality checks.

## Frozen application outcomes

Order-paired checkpoints are the three reversal pairs (`AB/BA`, `AC/CA`, `BC/CB`) and the six adjacent-swap terminal pairs. Report mean and worst-domain accuracy, first-task forgetting, prediction disagreement, penultimate-feature discrepancy, ECE, and normalized checkpoint distance `d/(1+d)`.

The order-sensitivity score is:

`0.30*|mean accuracy delta| + 0.20*|worst accuracy delta| + 0.20*|forgetting delta| + 0.10*prediction disagreement + 0.10*feature discrepancy/(1+feature discrepancy) + 0.05*|ECE delta| + 0.05*normalized checkpoint distance`.

For each same-base single-task pair (`A/B`, `A/C`, `B/C`), evaluate:

1. raw state-dict averaging;
2. pairwise reference alignment;
3. ordinary global synchronization over `M0` and the two branches;
4. cycle-aware synchronization over the corresponding two-order lineage subgraph;
5. validation-selected parameter interpolation on the fixed grid `0.0,0.1,...,1.0`;
6. two-branch prediction ensemble upper bound;
7. the better of the two sequential terminal checkpoints, labeled an additional-training nondeployable oracle.

All materialized merged adapters are deterministically compressed to rank at most four, so deployable methods have the same active capacity. A raw ordinary merge is harmful if its test mean accuracy is more than `0.01` below the average of the two branch means, or its worst-domain accuracy is more than `0.02` below the average of the two branch worst-domain accuracies. Test labels define outcomes only; no rule is changed from test results.

## Frozen cycle-aware rule

Ordinary synchronization estimates latent frames on the `M0`/branch triangle. Cycle-aware synchronization adds both ordered two-task terminals and all available directed lineage/closing edges for that task pair.

- Correct only if the validation loop is stably nonidentity by the frozen quality criteria and the synchronized connection residual is at most `0.20`.
- Use the validation-selected ordinary synchronization fallback when the loop is trivial or the correction precondition fails without severe instability.
- Recommend abstention when selected-edge condition number exceeds `10^4`, validation loop interval width exceeds `0.35`, or synchronized connection residual exceeds `0.35`.
- The test-time action is frozen from validation diagnostics. Abstention retains the better individual branch selected on model validation and incurs one branch at inference.

## Frozen predictors and held-out evaluation

Feature sets are exactly:

1. parameter distance only;
2. prediction disagreement only;
3. pairwise transport residuals only;
4. pairwise residuals plus inverse consistency;
5. loop-holonomy features;
6. pairwise plus holonomy features;
7. all features.

Use standardized logistic regression with L2 penalty and `C=1` for harmful-merge classification, and standardized ridge regression with `alpha=1` for continuous order sensitivity. If a training fold has one outcome class, emit its training prevalence rather than fit a different model.

Every out-of-fold prediction double-holds out the test row's entire independent seed and its entire task-pair/order family. No row from the same lineage loop enters training. Report AUROC, AUPRC, Brier, ten-bin ECE, Spearman correlation, harmful-merge avoidance, and merge/abstain regret. Uncertainty is a 2,000-replicate seed bootstrap; seeds, not rows, layers, loops, orders, or anchors, are the inferential unit.

## Success gates

- H1: the group-bootstrap interval for holonomy's incremental association with order sensitivity, after the three pairwise controls, excludes zero.
- H2: pairwise-plus-holonomy improves held-out AUROC or AUPRC over pairwise-only and the seed-bootstrap improvement interval is positive.
- H3: cycle-aware correction has a positive paired seed-bootstrap merge-performance interval over matched ordinary/pairwise synchronization without capacity increase.
- H4: across independent seeds, pairwise fits are acceptable, a loop is stably nontrivial, ordinary merging is harmful, and correction or abstention reduces regret.

No isolated row can pass H4. Three seeds permit pilot wording only. Five seeds permit the frozen confirmatory assessment.

## Leakage, hashing, and quarantine

Transition fitting is label-free. Estimators, interpolation, thresholds, and actions use training/model-validation/transport-validation data only. Test logits for every checkpoint and merge method are written and SHA-256 hashed before test labels are loaded for scoring. Exact commands, environment, source/config/split/checkpoint/logit hashes, validation-evaluation counts, failures, capacity, wall time, peak RSS, and latency are recorded. Any missing hash, split overlap, label access before logits, or method-specific target generation quarantines the affected seed and prevents a positive gate.
