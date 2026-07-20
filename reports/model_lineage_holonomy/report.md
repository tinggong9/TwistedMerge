# Model-lineage holonomy report

Mode: **confirmatory**. Decision: **no preregistered gate passed**.

## Execution

`/Users/tinggong/Documents/GitHub/TwistedMerge/.venv/bin/python experiments/model_lineage_holonomy.py --mode confirmatory --data-dir /Users/tinggong/Documents/GitHub/TwistedMerge/data --device auto`

- Independent seeds: [0, 1, 2, 3, 4]
- Checkpoints: 80
- Stable nonidentity loop/layer rows: 0 / 225
- Stable noncommutator rows: 0 / 900
- Harmful raw branch merges: 0 / 15
- Cycle-policy actions: `{'fallback:validation_selected_interpolation': 13, 'fallback:pairwise_reference_alignment': 2}`
- Failures: 0
- Gates: `{'H1': False, 'H2': False, 'H3': False, 'H4': False}`
- H1 numerical diagnostic: penultimate loop distances range from `1.933e-14` to `6.229e-14`; stable order-loop seeds `0`.

All transition estimators were selected by unlabeled transport-validation residual. Application test logits were saved and hashed before labels were loaded in each execution phase. Seeds are the inferential unit.

## Boundary

These are natural learning-path representation loops, not Brauer classes or topological certificates. Three seeds permit pilot wording only; only the five-seed aggregate is assessed against the confirmatory gates.
