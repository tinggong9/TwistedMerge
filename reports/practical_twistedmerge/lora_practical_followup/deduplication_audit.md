# LoRA practical follow-up deduplication audit

Audit date: 2026-07-19

Protected source commit: `9c91bc707d1f44beb36fe0fdce43af9ce1be79ed`

Experiment branch: `codex/lora-gauge-practical`

Isolated worktree: `/Users/tinggong/Documents/Codex/2026-07-19/lora-gauge-practical/work/TwistedMerge-lora-gauge-practical`

## Repository protection

Before this worktree was created, the required status, branch, recent-log, worktree, diff-stat, and diff-name checks were run. The manuscript checkout on `main`, the completed holonomy worktree, and the older `codex/practical-twistedmerge` worktree were clean. `main`, `origin/main`, and `codex/holonomy-applications` all identified the consumed holonomy program at `9c91bc7`. No collaborator changes were present, and no manuscript, bibliography, paper LaTeX, collaborator branch, or existing evidence ledger is in scope.

The current report namespace is additive:

`reports/practical_twistedmerge/lora_practical_followup/`

## Completed work that will not be repeated

- The controlled four-adapter LoRA smoke and its 20-scramble synthetic gauge families are already complete under `reports/practical_twistedmerge/lora_gauge/`. They establish a controlled representation-dependence result but are not trained-adapter evidence.
- The historical four-adapter algebra smoke in `experiments/lora_holonomy_merging.py` is synthetic and will not be rerun.
- The earlier tiny-BERT adapter run in `experiments/real_lora_adapter_near_term.py` is negative and its nominal alignment methods collapse to effective-delta merging. It will remain a boundary result rather than be rerun.
- BatchNorm gauge exactness, MNIST selector attribution, ordinary baseline sweeps, biomedical runs, holonomy classification, Brauer-like certification, period-index sweeps, invariant pooling, and the mergeability linter are out of scope.
- No new ResNet or adapter training will be launched.

## Reusable holonomy corpus

The completed holonomy program provides the required trained factorized adapters. Its confirmatory corpus contains five independent training seeds and eight D4 chart adapters per seed. Every adapter has one exact rank-4 residual layer with factors `up.weight` of shape `64 x 4` and `down.weight` of shape `4 x 64`, followed by a separately trained `10 x 64` classification head. Thus the residual update is a genuine LoRA-form update `B A`; no post-hoc or fabricated factorization is needed.

All five checkpoint bundles, all five saved test-logit files, and the shared projected-feature cache exist and match the committed SHA-256 values. The 40 manifest rows report zero failures, mean validation accuracy `0.623633`, and worst chart/seed validation accuracy `0.582031`.

The exact source paths and hashes are frozen in `reuse_manifest.csv`. Those source artifacts will be opened read-only and reused without modification. The consumed adapter-training source is `experiments/holonomy_shared_corpus.py` at source hash `618a4f5ce68c64035e766c05ada5796b7ce3bc2e5a101b79bed98dc830a68dde`; its training commit is `4f0a08c9b7b4b0ead2e1450a9fdf57b8149d41b2`, and the completed application/report commit consumed here is `9c91bc707d1f44beb36fe0fdce43af9ce1be79ed`.

## Corpus boundaries

- Frozen model: torchvision ResNet-18 with `IMAGENET1K_V1` weights, used only as a frozen feature encoder.
- Dataset: the existing local CIFAR-10 archive with committed SHA-256 `6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce`.
- Shared splits: 2,500 adapter-training, 512 overlap-fit, 512 overlap-validation, 512 validation, and 1,000 test identities.
- Adapter groups: seeds `0,1,2,3,4`; each group has eight chart adapters, satisfying the independent-group and adapter-count requirements.
- LoRA layers: one trained residual feature layer per chart adapter. Layerwise evaluation therefore has one applicable layer for every adapter and five independent group-level units.
- The test logits were saved before test labels were accessed. The follow-up will use validation for protocol checks and untouched test labels only after the method set is frozen.
- The local holonomy artifacts record provenance and file hashes but do not embed a complete standalone license dossier for the underlying CIFAR-10 data and pretrained ImageNet weights. This is retained as a reproducibility limitation; it does not trigger replacement training.

## Decision

Corpus reuse is technically valid and is required. The practical follow-up will consume the trained factors, heads, projected activations, split metadata, and saved logits. A second training campaign would duplicate completed work and is forbidden.
