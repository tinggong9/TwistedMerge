# Practical TwistedMerge current-gap audit

Audit date: 2026-07-19

Protected baseline commit: `85e8c17` (`Gate modern CIFAR ResNet experiments`)

Experiment branch: `codex/practical-twistedmerge`

Isolated worktree: `/Users/tinggong/Documents/Codex/2026-07-19/practical-twistedmerge/work/TwistedMerge-practical`

## Repository protection and scope

The authoritative checkout at `/Users/tinggong/Documents/GitHub/TwistedMerge` was clean on `main` and matched `origin/main` when this audit began. The required `git status`, branch, recent-log, worktree, diff-stat, and diff-name checks were run before creating the isolated worktree. Existing paper, bibliography, manuscript-figure, and LaTeX-table paths were inventoried and are out of scope. No collaborator branch, uncommitted work, or existing paper artifact has been modified.

Two local branches containing work not on `main` remain untouched:

- `codex/fix-evidence-checksum-20260715`
- `local/evidence-before-main-sync-20260715`

Only experimental code, supporting modules, tests, configs, and new files under `reports/practical_twistedmerge/` are authorized in this phase.

## Evidence reviewed

The audit covered the requested claim ledgers and post-ICLR reports, the underlying selector and BatchNorm CSVs, the existing LoRA implementations and result files, the same-base task-vector artifacts, and the official-baseline integration report.

The important boundaries are:

- Ordinary merge accuracy is not an open positive claim. In the fresh selector-attribution study, A5 trails the budget-matched ordinary B0 control by `-0.001865`, with group-bootstrap 95% CI `[-0.002578, -0.001215]`. All requested positive attribution gates failed.
- Greedy soup remains a strong boundary. The existing improved selector trails it by `-0.002415`, with 95% CI `[-0.004175, -0.000860]` in the audited exact-family comparison.
- The new BatchNorm work establishes narrow function-preserving identities for compatible ResNet-18 BasicBlock channel permutations and frozen-eval affine compensations. It is not trained-CIFAR merging evidence, and arbitrary positive channel scaling is not a static train-mode BatchNorm gauge.
- Adapter-assisted official Git Re-Basin, C2M3, and TIES cores have exact-family results. Official Model Soups remains interface-blocked; official Task Arithmetic and DARE remain license-blocked in their pinned author repositories. Internal implementations must remain labeled internal.
- Real residuals are not certified natural Brauer, projective, or period-index classes. Controlled obstructions cannot be presented as natural-data evidence.

## Existing LoRA evidence and the missing experiment

There are two relevant prior LoRA paths.

1. `experiments/lora_holonomy_merging.py` is a four-adapter synthetic smoke. It checks a factor transform and reports nearly closing cycles, but it does not repeat a fixed underlying adapter group across many gauge representations and does not measure output variability across scrambles. Its historical dependency blocker is now stale because the local environment contains `torch`, `transformers`, `datasets`, and `peft`.
2. `experiments/real_lora_adapter_near_term.py` trained four rank-4 adapters on a tiny BERT sentiment setup. The stored report is explicitly negative: the persistent-residual and held-out-gain gate failed. Its methods named as gauge-aligned operate on effective deltas rather than implementing a genuine common rank-space synchronization.

The central missing experiment is therefore narrower than another adapter benchmark:

> Hold the effective LoRA updates fixed, represent each adapter in at least 20 independently scrambled but equivalent rank-space gauges, and measure how much each merge output changes.

The experiment must include full-delta averaging followed by fixed-rank SVD as a mandatory gauge-invariant baseline. That baseline already removes factorization ambiguity, so any TwistedMerge claim must be about reliable factor-space synchronization, diagnostic value, or an explicit capacity/cost distinction—not uniqueness or superiority over effective-delta methods.

## Smallest controlled experiment

The first active experiment uses tiny float64 linear adapters with a planted shared rank subspace and four distinct effective updates. It will:

- verify `BA = (BQ)(Q^{-1}A)` at the delta, logit, and prediction levels;
- generate orthogonal, positive-diagonal, moderately conditioned dense, and separately labeled ill-conditioned gauges;
- use 20 scrambles per well-conditioned family;
- compare naive factor averaging, effective-delta methods, reference alignment, global synchronization, cycle-aware fallback, canonical SVD factors, and planted-gauge oracle alignment;
- keep one frozen synthetic protocol, with no model selection on test labels;
- report numerical stability and representation dependence, not real-model performance.

The controlled smoke has zero training compute and one fixed underlying adapter group. Gauge scrambles are dependent representations, not independent training seeds; they must not be bootstrapped as if they were independent experiments.

## Gates before any larger run

The next phase remains closed until all of the following pass:

1. unit tests prove gauge preservation, transition conventions, synchronization, and deterministic fallback behavior;
2. the smoke records at least 20 scrambles per well-conditioned family;
3. every underlying adapter is invariant within a preregistered float64 tolerance;
4. gauge-invariant baselines are invariant across scrambles;
5. naive factor averaging exhibits a measurable dependence in at least one non-orthogonal family, otherwise the smoke is reported as uninformative rather than positive;
6. all artifact hashes, commands, environment metadata, failures, costs, and output ranks are recorded;
7. the real dataset license gate is resolved or a replacement dataset/model suite is selected.

No Phase B linter, Phase C checkpoint canonicalization, federated, or biomedical experiment is active while these gates are open.

## Current claim boundary

Before the smoke executes, the only defensible statement is structural: LoRA factors admit invertible rank-space reparameterizations that preserve their effective update, and factorwise merging can therefore be representation-dependent in principle. No new empirical invariance, accuracy, natural-obstruction, mergeability-prediction, or practical-benefit claim has yet been established.
