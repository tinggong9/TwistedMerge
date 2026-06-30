# Paper-Level Decision After 35791f7

This decision pass read the current report and generated-LaTeX corpus under `reports/`, including 71 markdown/TeX artifacts, the post-35791f7 results synthesis, the CIFAR/bridge appendix, external integration appendix, capacity/claim audit, and the current claims audit. The conclusion is deliberately conservative.

## Actual Recommendation

Target now: arXiv plus an ICLR/NeurIPS workshop submission.

Secondary target: SIAGA / applied topology / applied algebraic geometry only if the paper is compressed around the descent/period-index mathematics and the ML benchmarks are framed as motivating diagnostics.

Do not target ICLR main, NeurIPS main, or ICML main with the current evidence. TMLR is possible only after further compression and stronger empirical hygiene; it is not the cleanest first target.

## Recommended Thesis

Model-merging residuals are descent defects. Central/projective torsion residuals obey period-index rank thresholds and can be conservatively detected in controlled and time-frequency chart settings. Practical ReLU-compatible monomial/channel gauges beat faithful in-repo C2M3-style alignment on MNIST, Fashion-MNIST, and bridge/CNN slices, but they do not robustly beat greedy soup. CIFAR passes a bounded base-accuracy gate but remains an appendix boundary with descriptive exact-gauge effects only. Real MNIST/Fashion/CIFAR residuals remain non-Brauer under tested diagnostics.

## Brutal Assessment

The paper is not a model-merging performance paper in its current form. It is a descent-defect taxonomy paper with controlled mathematics, conservative detectors, and a limited exact-gauge ML slice. That is good, but it must be sold as exactly that.

The strongest publishable unit is the combination of:

1. a finite-site obstruction witness;
2. a planted-obstruction-to-merge-degradation experiment;
3. a period-index rank-threshold theorem and detector stack;
4. time-frequency chart recovery as a natural controlled source of central/projective torsion;
5. a residual taxonomy that prevents false Brauer claims on real neural residuals;
6. limited exact ReLU-compatible monomial/channel gauge wins over faithful internal C2M3-style baselines;
7. a clear negative boundary: greedy soup and CIFAR remain hard.

The weakest part for ML main venues is the practical result: the method does not robustly beat greedy soup, official external code did not run on the exact checkpoint set, CIFAR exact-gauge deltas are descriptive only, and many results are controlled or diagnostic. The paper gets stronger if it stops pretending to be a broad empirical model-merging win.

## Venue Decision Matrix

| Target | Recommendation | Allowed claims | Forbidden claims | Missing experiments or proof | Likely reviewer objections | Sections to cut |
| --- | --- | --- | --- | --- | --- | --- |
| arXiv / general workshop | Submit now after compression | Full conservative thesis; limited C2M3-style wins; greedy-soup boundary; CIFAR appendix boundary; non-Brauer residual taxonomy; controlled period-index/time-frequency detectors | SOTA, official-baseline win, greedy-soup win, broad CIFAR, real Brauer residuals | None required for a preprint, but clarity and pruning are required | "Too sprawling"; "too many experiments"; "unclear main claim" | Cut duplicate prompt-by-prompt benchmark detail; keep only one main practical table and one claim-boundary table |
| SIAGA / applied topology / applied AG | Plausible if math-first | Descent-defect framing; finite-site `H^2(mu_2)` witness; period-index rank thresholds; noncentral versus central/projective taxonomy; time-frequency controlled chart examples | ML performance headline; claims about practical superiority; CIFAR or greedy-soup claims | More formal definitions, theorem statements, proof sketches, relation to projective representations/nonabelian cocycles/Brauer language | "ML section feels ad hoc"; "too much engineering"; "where is the theorem?" | Cut most MNIST/Fashion tables; move empirical section to motivation/diagnostics; cut external-baseline details from main |
| TMLR | Borderline, not recommended as first target | Honest diagnostic model-merging paper; exact ReLU gauge wins over faithful in-repo C2M3-style baselines; negative greedy/CIFAR results | Official-baseline win; broad architecture generalization; greedy-soup win; period-index explanation for real residuals | Cleaner official-baseline story, stronger reproducibility package, broader architectures, more standard baselines, ablations on validation selection | "Why not beat greedy soup?"; "official C2M3/Git-ReBasin missing"; "too synthetic"; "CIFAR weak" | Cut deep period-index/time-frequency details or move them to appendix; shorten block/sheaf diagnostics; keep empirical narrative central |
| ICLR/NeurIPS workshop | Best current venue | Complete recommended thesis; controlled math plus limited practical exact-gauge evidence; negative boundary as a feature | SOTA, robust CIFAR win, official external-baseline win, greedy-soup win unless a paired CI later supports it | Polished story, compressed figures, one clean table of claims, reproducibility appendix | "Hybrid paper"; "model-merging result is modest"; "too much appendix dependence" | Cut early smoke runs, duplicate selector variants, exhaustive CSV tables, and most optional sheaf/GNN detail from main |
| ICLR/NeurIPS main | No-go now | At most a diagnostic/theory contribution with small empirical support | Broad ML contribution, strong empirical win, robust generalization, official-baseline superiority | Official baselines, strong comparisons on modern architectures, CIFAR/ImageNet-like wins, clearer theorem-to-algorithm payoff | "No greedy-soup win"; "no official baselines"; "CIFAR appendix only"; "too many controlled experiments" | Would need a different paper; current compression is insufficient for main-track competitiveness |
| ICML main | No-go now | Same as ICLR/NeurIPS main, but even less forgiving on empirical framing | Any claim of practical model-merging competitiveness | Capacity-matched gains over strong baselines, official-code comparisons, larger-scale architectures, robust ablations, cleaner statistical design | "Practical contribution too weak"; "theory not tied tightly enough to real residuals"; "negative CIFAR" | Cut or redesign around one compelling empirical method, which current artifacts do not support |

## Final Paper Outline

Working title:

`Model-Merging Residuals as Descent Defects: Exact Gauges, Period-Index Diagnostics, and Non-Brauer Boundaries`

Compressed main paper:

1. Introduction
   - State the descent-defect view.
   - State the negative boundary up front: not a SOTA model-merging paper, not a greedy-soup win, not a CIFAR win.
   - Contributions in four bullets: obstruction, detectors, exact ReLU-compatible gauges, residual taxonomy.

2. Descent Defects And Exact Gauge Symmetries
   - Define local alignments, cycle residuals, central/projective residuals, noncentral residuals.
   - State which gauges are exact for ReLU MLPs and no-BatchNorm CNNs.
   - Include one compact capacity/symmetry table.

3. Central Torsion And Period-Index Rank Thresholds
   - Present the finite-index and k-pair finite-Heisenberg period-index theorem.
   - Explain period versus index and why period-divisible ranks can still fail.
   - Keep proofs or proof sketches tight; move full derivations to appendix.

4. Conservative Detectors And Residual Gates
   - Present central commutator-matrix detector, robust calibration, nearest-Heisenberg projection, and connection-residual gate.
   - Emphasize certified/uncertain/rejected outcomes.
   - Show false-lift rate zero on tested negative controls.

5. Controlled Obstruction-To-Merge Evidence
   - Finite-site `H^2(mu_2)` witness.
   - Planted obstruction experiment: pairwise degradation increases with cycle defect; C2M3-style synchronization fixes the tested planted inconsistency.
   - State that branch/projective lifts are extra capacity unless separately compressed.

6. Practical Exact-Gauge Model Merging
   - MNIST MLP: improved selector beats faithful internal C2M3-style baseline, not greedy soup.
   - Fashion-MNIST MLP/CNN: exact monomial/channel gauges generalize over C2M3-style baselines, greedy soup remains stronger.
   - Bridge datasets: boundary pattern persists.
   - CIFAR: base gate passes, exact-gauge effects are descriptive only, appendix boundary.

7. Residual Taxonomy On Real Neural Merges
   - Real MNIST/Fashion/CIFAR residuals remain noncentral/non-Brauer under tested diagnostics.
   - The taxonomy is useful because it refuses false period-index/Brauer claims.

8. Related Work
   - Git Re-Basin, C2M3, Model Soups, mode connectivity, neural sheaves, projective representations, period-index problems.
   - Keep official integration status to one paragraph plus appendix.

9. Limitations
   - No official external-baseline numbers.
   - No greedy-soup win.
   - No broad CIFAR/general-vision result.
   - No real-neural Brauer/period-index residual certification.
   - Extra-capacity lifts are not capacity-matched single-model merges.

Appendix:

1. Full claim audit and capacity/symmetry table.
2. Full benchmark tables and bootstrap intervals.
3. Robust detector calibration details.
4. Time-frequency learned/denoised/projection experiments.
5. Block-gauge phase diagram and projection-trap details.
6. Official external integration attempt and NSD smoke.
7. Reproducibility commands and configuration metadata.

## Compression Rules

Keep in the main text:

- one theorem-style result for finite-site/descent obstruction;
- one theorem-style result for period-index rank thresholds;
- one detector algorithm box;
- one practical benchmark table with MNIST, Fashion-MNIST MLP, Fashion-MNIST CNN, bridge, and CIFAR rows;
- one residual-taxonomy table;
- one venue/claim-boundary table only if submitting as workshop/arXiv.

Move to appendix:

- all prompt-by-prompt benchmark reports;
- early smoke MNIST/CIFAR results;
- greedy-aware selector variants not changing the final claim;
- most block-gauge phase-diagram tables;
- full time-frequency denoising/projection sweeps;
- official integration logs;
- NSD smoke details;
- full capacity audit.

Cut entirely from the compressed manuscript unless needed for a rebuttal:

- repeated environment dumps;
- per-seed raw tables where summary CIs suffice;
- duplicate LaTeX tables from intermediate prompts;
- claims about methods that are now superseded by stronger confirmatory reports;
- any language suggesting broad model-merging, broad CIFAR, official-baseline, or greedy-soup superiority.

## Decision In One Sentence

Submit a compressed arXiv plus ICLR/NeurIPS workshop paper now, with an optional math-venue variant later; do not spend the current manuscript on a main-track ML submission until there is an official-baseline story and a practical result that survives greedy soup beyond MNIST-derived slices.
