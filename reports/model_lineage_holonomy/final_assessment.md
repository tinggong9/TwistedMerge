# Final assessment: model-lineage holonomy

Mode: **confirmatory**. Gate status: `{'H1': False, 'H2': False, 'H3': False, 'H4': False}`.

1. **Were stable nonidentity loop holonomies observed?** No; 0 loop/layer rows passed the frozen stability definition.
2. **Were any independent loop holonomies noncommuting?** No; 0 commutator rows passed the frozen interval threshold.
3. **Did holonomy correlate with task-order dependence?** No admissible H1 association passed. The standardized numerical coefficient is retained, but the relevant loop distances were only 1.933e-14 to 6.229e-14 and no stable nonidentity order loop was observed.
4. **Did holonomy add information beyond pairwise drift?** No.
5. **Did holonomy predict harmful branch merges?** No held-out H2 improvement was established; 0 harmful raw merge rows were observed.
6. **Did cycle-aware correction improve merging?** No paired seed-level H3 improvement was established.
7. **Did conservative abstention reduce regret?** No repeated H4 conflict class was established.
8. **Which layers carried the strongest stable signal?** No layer passed the stability gate; `adapter` had the largest raw distance (0.9734) but is not a stable signal.
9. **What is the strongest main-paper claim?** Different learning orders produced measurable terminal differences, but holonomy supplied no reliable incremental predictive or corrective value beyond pairwise diagnostics.
10. **Should holonomy application experiments stop?** Yes. The preregistered stopping rule applies.

## Integrity

- Test-logit-before-label flag: `True`.
- Double seed/family/loop holdout flags: `True`.
- Failure rows: `0`.
- No manuscript, LaTeX, bibliography, protected worktree, or prior evidence artifact was modified.
