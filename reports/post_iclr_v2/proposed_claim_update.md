# Post-ICLR v2 Proposed Claim Update

No paper claim is promoted by this audit alone.

| Exact wording | Status | Evidence | Limitation |
| --- | --- | --- | --- |
| On the exact 20-setting MNIST one-hidden-layer MLP family, the existing validation-only selector exceeded adapter-assisted official Git Re-Basin by `0.0143` (95% CI `[0.0074, 0.0221]`) and official C2M3 by `0.0081` (95% CI `[0.0030, 0.0136]`). | supported-narrow | Official baseline summary CSV | Attribution is unresolved and selections are soup-dominated. |
| The pure TwistedMerge monomial gauge beats official C2M3. | negative | Official C2M3 leads by `0.0087` with positive CI. | Wording must not appear as a positive result. |
| TwistedMerge beats greedy soup. | forbidden until new evidence | Existing selector delta is `-0.0024` with a negative CI. | Requires the fresh budget-matched selector-attribution phase. |
| TwistedMerge works on ResNet-18 with exact BatchNorm-aware gauges. | pending | No current artifact. | Requires derivation, exactness tests, base-quality gate, and confirmatory groups. |
