# Post-ICLR Experiment Plan

This plan is frozen from the audit at commit `7a0620b`. It follows the requested phase order and opens only one large experiment family at a time.

## Universal protocol

Every runnable phase has three modes: focused smoke, preregistered pilot, and confirmatory repeated-seed run. Configuration is saved before confirmatory execution. Selection uses training/validation data only; test data are evaluated after rules and thresholds are frozen. Reports retain failed runs, exact-setting losses, capacity multipliers, inference multipliers, command lines, environment, source commit, dataset/checkpoint provenance, compute, and artifact paths.

Paired confirmatory output must include mean, median, standard deviation, bootstrap confidence interval, wins/ties/losses, seed count, and failed-run count. Lifts are named only as `TwistedMerge rank lift` or `TwistedMerge branch lift + invariant pooling`; same-capacity exact transformations are named `TwistedMerge gauge merge`; the conservative policy is named `TwistedMerge diagnostic selector`.

## Ordered phases and gates

1. **Audit and manifest -- complete.** The gap map, plan, and machine-readable manifest are the preregistration boundary for later work.
2. **Official baseline integration -- complete.** Pinned Git Re-Basin, C2M3, and TIES cores ran through declared adapters on exact checkpoint families. Model Soups was interface-blocked; Task Arithmetic and DARE were license-blocked. Independent-initialization and common-base runs remain separate, and blocked methods have status rows without metrics.
3. **ResNet-18 CIFAR-10/CIFAR-100 plus BatchNorm study -- pending.** Begin only after phase 2 has a complete status ledger. Smoke one group, pilot the declared recipe, then run at least five groups for both three- and four-model merges. Reject claims if the base-accuracy gate fails.
4. **Unified planted real-network obstruction -- pending.** Implement synchronizable, permutation, positive channel, controlled central/projective, and valid nonabelian cases. The old target-injected nonabelian accuracy artifacts are excluded.
5. **Diagnostic prediction and conservative selector -- pending.** Freeze prediction targets and selector rules from validation data, then evaluate on untouched test settings. Default to the best validation-selected ordinary candidate whenever no structured residual is certified.
6. **Biomedical multi-site classification pilot -- pending.** Resolve license and dataset first, use simulated-site language, and separate it from the existing spatial-output segmentation program.
7. **Biomedical confirmation -- gated.** Requires valid dataset provenance, credible individual-model quality, multiple site partitions, stable selector behavior, and no leakage.
8. **One secondary architecture or LoRA design -- deferred.** Select a single missing-mechanism test only after preceding gates close. No automatic LLM campaign.

## Phase 2 acceptance criteria

- Every official repository has URL, license, immutable commit, clone/install command, import probe, interface verdict, and exact status.
- Metric rows exist only when official source code actually participates in producing the merge or selection.
- Adapter-assisted runs name the adapter and patch; they are not called unmodified official end-to-end runs.
- Every run points to an exact checkpoint group and records output capacity.
- Internal faithful baselines remain in separate rows.
- Common-base Task Arithmetic/TIES/DARE rows never appear in independent-initialization comparisons.
- A complete negative integration ledger is acceptable; a silent substitution is not.

## Phase 3 preregistration boundary

Before any confirmatory ResNet execution, freeze dataset versions and splits, torchvision/model recipe, augmentation, optimizer, scheduler, epochs, batch size, weight decay, seeds/model groups, base-accuracy thresholds, BatchNorm treatment, calibration method, candidate set, selector rule, and compute budget. The no-BatchNorm small-CNN evidence is a control only.

## Large-run stop conditions

Stop rather than scale when the smoke test reveals leakage, target-derived logits, non-reproducible checkpoints, invalid functional-preservation claims, unacceptable base accuracy, missing dataset/license provenance, or an external baseline that cannot consume the checkpoint family without replacing its algorithm. Record the boundary as a negative result.
