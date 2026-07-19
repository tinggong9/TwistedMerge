# Holonomy Applications Deduplication Audit

Audit date: 2026-07-19

Frozen baseline: `2fa48e5` (`main` and `origin/main` at audit time)

Working branch: `codex/holonomy-applications`

External worktree: `/Users/tinggong/Documents/Codex/2026-07-19/holonomy-applications/work/TwistedMerge-holonomy-applications`

## Decision

The deduplication gate passes only for a restricted new program: build one shared CIFAR-10 corpus from a frozen ImageNet-pretrained ResNet-18 and eight independently fitted D4 chart adapters/heads, then consume the exact saved models, activations, transition estimates, splits, and pre-label-scoring logits in Applications A-D.

This is not authorization to rerun an existing chart, multiview, holonomy, period/index, or obstruction-predictor benchmark. The new scientific unit is the shared corpus plus the cross-application chain from measured nonabelian transition geometry to prediction, certification, controlled capacity, and a conservative linter.

No training or data download was started during this audit. No manuscript, LaTeX, bibliography, or collaborator-owned file was changed.

## Invalid empirical evidence quarantine

The old controlled nonabelian accuracy family is invalid as empirical performance evidence.

- `src/controlled_nonabelian_holonomy.py::logits_with_target_accuracy` constructs predictions from the supplied labels and a requested target accuracy.
- `src/controlled_nonabelian_holonomy.py::target_accuracy_for_method` prescribes method-dependent scores.
- `experiments/controlled_nonabelian_holonomy.py` obtains those target values and calls the label-dependent logit constructor for validation and test labels.
- Therefore the reported accuracies, losses, paired deltas, selector choices/regret, wins, sign tests, and any performance-based claim derived from those values are not model measurements.
- `reports/sequential_quotient_lift_correction_note.md` independently quarantines an older quotient-lift accuracy path for analogous method-dependent label injection and records the corrected actual-logit guard.

The following artifacts remain in the repository for provenance but must not be used as empirical support:

- `reports/csv/controlled_nonabelian_holonomy.csv`
- `reports/csv/controlled_nonabelian_holonomy_paired_stats.csv`
- `reports/csv/controlled_nonabelian_holonomy_summary.csv`
- `reports/csv/controlled_nonabelian_holonomy_selector.csv`
- performance rows in `reports/csv/controlled_nonabelian_holonomy_claims.csv`
- the accuracy and selector-regret plots generated from those rows

Permitted reuse from that family is structural only: finite-group operations, planted holonomy construction, regular actions, invariant-pooling identities, residual identities, and structural tests after they are separated from the label-derived accuracy path. No numeric performance result from the quarantined artifacts may enter an application table, plot, model-selection decision, or claim.

Every new empirical path must satisfy these guards:

1. Candidate logits are materialized and SHA-256 hashed before labels are loaded for scoring.
2. A label-permutation audit confirms that permuting evaluation labels changes only metrics, never logits, transitions, residuals, or method choice when selection is frozen.
3. Test labels are used once after all model, threshold, and selector choices are frozen.
4. Controlled structural labels such as group elements or planted cocycles are kept distinct from classification labels.

## Completed work that must not be repeated

### D4 CIFAR chart routing and retransport

`experiments/cifar10_chart_retransport.py` and `experiments/chart_followup_common.py` already implement D4 image actions, multiplication/inversion, order controls, D4 test-time augmentation, chart routing, and a CIFAR-10 chart-followup benchmark. That program trained four small CNN experts from scratch and did not construct eight independent adapters over one frozen pretrained encoder.

Its discovery gate failed and confirmation did not open. The worst-condition accuracy delta was `-0.043243` with CI `[-0.054955, -0.029730]`; structured retransport also trailed generic MoE, a low-rank adapter, and a direct D4 task model in the recorded paired comparisons. Those negative results are reference evidence, not a campaign to rerun. D4 utilities and controls may be reused; the benchmark, training schedule, and claim are closed.

### Genuine and realistic multiview retransport

The two ModelNet10 multiview families already fit four view experts, estimate transitions, measure loop/inverse residuals, and compare synchronized/retransported variants. Their complete gates failed. They use a different dataset, four views, and either view-specific CNNs or linear experts; they are not the proposed eight-chart frozen-encoder corpus. Generic transition fitting, Procrustes, graph, and residual utilities may be adapted, but no ModelNet rerun or result pooling is allowed.

### Controlled nonabelian and quotient-lift work

The invalid accuracy family is quarantined above. The corrected sequential quotient-lift smoke does produce executed-model logits and has an explicit label guard, but its D4 application gates did not yield a positive natural branch result. Its exact coset actions, quotient-chain checks, and label-independence pattern may be reused. It is not positive evidence for Application A.

### Central/projective and period-index work

The repository already contains extensive algebraic matrix, clock/shift, controlled central-extension, detector-calibration, rank-threshold, and period/index lift evidence. The natural finite-index residual mining and residual-peeling paths did not certify a real image-model torsion candidate. The time-frequency benchmark uses known operators and an orbit-invariant prototype task, not learned chart transitions from image models.

Application B must therefore be a conservative certificate on the new shared natural adapter transitions, not another controlled matrix grid. Application C may add only the smallest preregistered controlled projective layer on the same frozen real features and saved evaluation protocol; it cannot be presented as a naturally discovered Brauer class.

### Prior obstruction prediction and selectors

`reports/obstruction_predictor_target_report.md` supports selected alignment-conditioned targets but explicitly does not support raw weight-average degradation prediction. Existing selector and attribution studies are narrow or negative. Application D is distinct only as a grouped, held-out classifier trained on outcomes generated by Applications A-C in the one new corpus. Existing predictors supply comparison features and leakage lessons, not transferable performance.

### Other closed evidence families

- Post-ICLR selector attribution is negative and must stay separate.
- BatchNorm evidence establishes only narrow functional identities, not a trained-ResNet merge benefit.
- Controlled overlap, primary holonomy, small-quotient, residual-peeling, greedy-soup, task-vector, rebasin, and gauge-only LoRA results answer different questions and must remain in their original evidence regimes.
- No additional MNIST, Fashion-MNIST, ModelNet, gauge-only, or matrix-only sweep is authorized by this program.

## Approved shared corpus boundary

The sole approved corpus is:

- dataset: the existing local CIFAR-10 cache, with one frozen train/calibration/validation/test split manifest shared by every application;
- encoder: one frozen ImageNet-pretrained torchvision ResNet-18, with its exact weight checksum recorded;
- charts: all eight D4 image transforms, using the existing tested action convention;
- chart models: eight independently initialized and independently optimized low-rank feature adapters plus classification heads, one per chart, while the encoder remains frozen;
- overlaps: the same image identities evaluated in chart pairs, so activation correspondences and transitions are measured rather than invented;
- artifacts: model states, features or sufficient activation sketches, transitions, loop/commutator ledgers, logits, hashes, resource counts, and split identities saved once and reused without per-application retraining;
- seeds: a frozen multi-seed discovery design, with any confirmation phase gated before it is run;
- claims: bounded to this adapter corpus; no full-network, general vision, natural Brauer, or state-of-the-art claim.

The corpus is distinct from prior CIFAR chart work because it uses a frozen pretrained feature map, eight separately fitted chart adapters, actual overlap activations for all D4 charts, and a single immutable artifact set feeding all four applications. Merely rerunning `experiments/cifar10_chart_retransport.py` would fail this audit.

## Available immutable inputs

- Pretrained weight cache: `/Users/tinggong/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth`
- Pretrained weight SHA-256: `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`
- CIFAR-10 archive: `/Users/tinggong/Documents/GitHub/TwistedMerge/data/cifar-10-python.tar.gz`
- CIFAR-10 archive size: `170498071` bytes
- CIFAR-10 archive SHA-256: `6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce`
- Runtime observed at audit: PyTorch `2.12.1`, torchvision `0.27.1`, NumPy `2.5.0`, pandas `3.0.4`, SciPy `1.18.0`, scikit-learn `1.9.0`, matplotlib `3.11.0`, macOS arm64.

The new worktree intentionally has no duplicated ignored `data/` directory. Corpus commands must use the explicit existing data root and must not download another dataset.

## Application-specific opening gates

### Application A: nonabelian holonomy

Open only after corpus integrity tests pass. Compare ordinary pooling/merge baselines with D4-aware invariant or branch pooling, actual estimated transitions, noncommuting loop and commutator diagnostics, missing-edge controls, random/capacity-matched controls, and an oracle used only as an upper bound. A positive claim requires a preregistered paired performance gate and structural evidence that the relevant measured loops are noncommuting. A negative result closes extra chart/dataset expansion.

### Application B: conservative central projective certificate

Run on the unchanged natural transitions from Application A. Report scalarity/centrality residuals, cocycle consistency, root-of-unity margins, bootstrap stability, sensitivity to tolerances, and an explicit abstain/negative/positive verdict. Do not force a class. A positive natural Brauer-like claim requires all structural and stability gates; otherwise the defensible result is a negative certificate or abstention.

### Application C: period-index capacity prediction

Use the same frozen features and evaluation split, adding only a small controlled projective layer with preregistered ranks around the predicted index and matched parameter/inference controls. It must test both structural residual reduction and held-out task performance. It may establish a controlled capacity principle; it may not be relabeled as natural discovery.

### Application D: mergeability linter

Use only rows produced by the frozen A-C pipeline. Fit one interpretable logistic regression unless the sample or class balance is inadequate. Split by corpus seed and perturbation family, never by correlated row. Report AUROC, AUPRC, Brier/calibration, and coverage-risk only when the held-out groups contain both outcome classes. If there are fewer than three independent held-out seed groups or fewer than ten examples of either class, report descriptive scores and abstention behavior without a predictive-performance claim.

## Stopping rules

- No new chart family or dataset is added if Application A misses its gate.
- No repeated controlled matrix grid is added if Application B is negative or abstains.
- No capacity sweep beyond the preregistered ranks is added if Application C misses either structural or task gate.
- No more flexible linter is substituted if logistic regression is negative; the correct result is negative or inadequate-sample.
- Existing paper files remain untouched. Any eventual application snippet must be a new file under `reports/holonomy_applications/` and must state the evidence boundary.

## Audit conclusion

`deduplication_gate_passed = true` for the restricted shared-corpus construction above.

`existing_experiment_rerun_authorized = false`.

`training_during_audit = false`.

`next_phase = implement and validate the shared-corpus manifest, configuration, leakage guards, and smoke path; commit those before the first bounded training run`.
