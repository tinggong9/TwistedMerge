# Post-ICLR v2 Proposed Claim Update

The completed selector-attribution phase promotes no positive algorithmic claim. It adds a confirmatory negative result that closes the previous attribution ambiguity on this MNIST MLP protocol.

| Exact wording | Status | Evidence | Limitation |
| --- | --- | --- | --- |
| On the exact historical 20-setting MNIST one-hidden-layer MLP family, the existing validation-only selector exceeded adapter-assisted official Git Re-Basin by `0.0143` (95% CI `[0.0074, 0.0221]`) and official C2M3 by `0.0081` (95% CI `[0.0030, 0.0136]`). | supported-narrow | Official baseline summary CSV | This is a historical exact-family comparison, not evidence of a TwistedMerge-specific gain; the fresh attribution study is negative. |
| The pure TwistedMerge monomial gauge beats official C2M3. | negative | Official C2M3 leads by `0.0087` with positive CI. | Wording must not appear as a positive result. |
| On 10 new MNIST MLP training groups (40 exact settings), the full A5 selector trailed an ordinary B0 soup selector with exactly matched candidate count and selector validation evaluations by `0.001865` accuracy (95% group-bootstrap CI `[-0.002578, -0.001215]`). | negative | `selector_attribution/paired.csv` | Candidate-generation kernels differ and their compute is reported; all outputs are same-capacity single models or materialized soups. |
| TwistedMerge-specific candidates were selected in `65%` of A5 settings, but their conditional mean gain over A0 was `-0.001758`. | negative | `selector_attribution/selection_counts.csv`; `claim_status_update.json` | Selection frequency is not benefit; the conditional result has the wrong sign. |
| Residual diagnostics improve choice quality with the same candidate pool. | negative | A4 and A1 tied on all 40 settings; paired delta and CI are exactly zero. | The frozen pilot-median rule did not change test outcomes. |
| TwistedMerge beats greedy soup. | forbidden | Existing selector delta is negative, and the fresh A5 attribution is also below A0/B0. | Do not describe enriched-pool selection as a TwistedMerge-specific advantage. |
| TwistedMerge works on ResNet-18 with exact BatchNorm-aware gauges. | pending | No current artifact. | Requires derivation, exactness tests, base-quality gate, and confirmatory groups. |
