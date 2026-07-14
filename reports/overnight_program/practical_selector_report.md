# Stage 1: fresh practical selector rerun

## Decision

This is a **full** execution with 120 matched settings and fresh model training, merged checkpoints, inference, saved logits, validation-only selection, and held-out test evaluation. The saved-logit label-permutation regression passed for every method-setting row.

The TwistedMerge selector mean accuracy is 0.855847; ordinary greedy soup is 0.857245. The paired selector delta is -0.001398 (95% CI [-0.002066, -0.000731]), with 24/43/53 win/tie/loss.

No central lift was selected. No nonabelian lift was selected. Both activation rates are exactly 0 because no certificate passed and these candidates were not invented.

## Protocol

- MNIST, one-hidden-layer ReLU MLP.
- Model counts: [3, 4]; widths: [16, 32, 64]; seeds: 1800--1819.
- Checkpoints and splits are matched across all methods within each setting.
- Alignment uses the training partition, selection uses a disjoint validation partition, and the test set is evaluation-only.
- Every reported prediction comes from an executed merged model, soup, selector choice, or ensemble.
- Selector frequencies: `{"c2m3_greedy_soup": 1, "global_monomial": 2, "global_monomial_greedy_soup": 2, "monomial_greedy_soup": 3, "optimized_monomial": 5, "optimized_monomial_greedy_soup": 8, "randomly_augmented_candidate_union": 20, "shrinkage_monomial": 1, "shrinkage_monomial_greedy_soup": 1, "union_candidate_soup": 77}`.

## Evidence boundary

Selector regret is reported only as a post-hoc audit and never influences selection. The central and nonabelian activation columns are retained as explicit negative findings. Runtime, peak process memory, parameter counts, stored parameters, branch count, candidate count, validation budget, and measured inference time are in the run and capacity CSV files.
