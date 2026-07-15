# Evidence-based manuscript revision packet

The manuscript source is not stored in this repository, so this packet identifies replacement targets by claim and artifact rather than inventing line numbers.

## Remove or replace

1. Remove any table or paragraph that treats one-dataset natural, adapter, transformer, pose, or frozen-backbone smoke results as full evidence.
2. Remove any statement of practical cross-domain superiority unless Level 3 is mechanically true.
3. Replace selector-only comparisons that omit generic context-conditioned methods.
4. Retain the negative practical-selector result (ordinary greedy soup approximately 0.8572 versus selector approximately 0.8558) and label it as a negative ordinary-regime result.

## Insert

1. Insert `tables/context_main.tex` and `tables/context_efficiency.tex` in the controlled context section.
2. Insert `tables/hodge_ablation.tex` after the algorithm description.
3. Insert `tables/natural_main.tex` in the natural-checkpoint section.
4. Insert `tables/vision_main.tex` only if `vision_claims.json` records an executed benchmark.
5. Insert `tables/federated_main.tex` and `tables/systems.tex` in the practical evidence section.

## Supported wording

The strongest supported claim is Level 2. The controlled context gate passes with narrow confirmation. The Hodge component is positive in controlled families but not in the real-image frame ablation. A natural residual is not promoted. The compact pretrained gate does not pass or is resource-blocked. A persistent federated lift gain is not found.

Use the exact replacement text in the adjacent supported-text files. Do not strengthen it beyond `claim_ladder.json`.
