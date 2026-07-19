# Practical TwistedMerge paper-editor evidence brief

Status: audit and controlled LoRA smoke complete; real-adapter pilot remains blocked.

## New controlled evidence

One fixed synthetic four-adapter rank-3 group was represented in 20 independent gauges from each of four families. The three well-conditioned families contribute 60 dependent representations; they are not independent training groups and have no bootstrap confidence interval.

- The maximum relative change in an individual effective update was `1.797e-15`, with zero prediction disagreement.
- Under 20 moderately conditioned dense scrambles, naive factor averaging had accuracy range `0.252930`, maximum relative merged-delta change `5.967`, and maximum absolute logit change `9.367`.
- Global synchronization had zero accuracy variation, maximum relative merged-delta change `1.463e-14`, and maximum absolute logit change `1.910e-14`.
- Full-delta SVD was also invariant, with maximum relative merged-delta change `1.603e-15` and maximum logit change `2.220e-15`.
- An injected inconsistent transition graph produced normalized cycle defect `0.2021`; the cycle-aware method abstained and returned the full-delta SVD fallback.
- Forty alignment failures were recorded for the ill-conditioned boundary family and zero for the primary well-conditioned families. The failed boundary rows were retained and safely fell back.

All eight controlled smoke gates and all seven targeted unit tests passed.

## Existing boundaries retained

- The selector-attribution phase is negative; it does not support a TwistedMerge-specific ordinary-merge gain.
- Greedy soup remains a strong accuracy boundary.
- Existing BatchNorm results are narrow functional identities, not trained-ResNet merging performance.
- The previous four-task BERT LoRA gate failed and cannot support a positive adapter-merging claim.
- Effective-delta averaging plus fixed-rank SVD is an essential gauge-invariant baseline.
- The smoke has one planted shared-B group, so it does not support a real-adapter performance claim or confidence interval.
- The ill-conditioned family is a numerical boundary, not part of the primary invariance claim.
- No natural Brauer, period-index, clinical, or broad SOTA statement is supported.

## Editorial action

If this smoke is mentioned, acceptable wording is limited to: in one controlled planted shared-rank-space example, TwistedMerge alignment removed representation dependence under the tested well-conditioned LoRA gauges, while naive factor averaging did not. Full-delta SVD and other effective-delta controls were also invariant. Any natural-model, performance, prediction, or broad practical wording remains gated on licensed, independently trained adapter groups and a frozen confirmatory protocol.
