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

## Final repository verification

- Confirmatory evidence commit: `1e23e83`.
- Required root deliverables: `21 / 21` present.
- Artifact manifest: `320` rows and `315,089,617` referenced bytes verified for path, size, and SHA-256.
- Focused lineage and reused-module boundary: `22 passed`.
- Repository-wide nonhanging suite: `544 passed, 1 deselected, 5 subtests passed` in `29.57s`.
- The deselected closure test is the pre-existing `tests/test_block_gauge_branch_closure.py::BlockGaugeBranchClosureTests::test_smoke_closure_outputs_and_threshold_columns`, previously observed to hang in its experiment subprocess; it was not changed by this work.
- The main checkout and the completed holonomy, protected LoRA-gauge, and earlier practical worktrees remained clean.
