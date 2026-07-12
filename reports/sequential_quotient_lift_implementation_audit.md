# Sequential Quotient Lift Implementation Audit

## Audit Result

The old primary-depth sweep is not a genuine consecutive quotient-lift experiment.  It estimated primary factors from observed orders and relation residues, then reused the existing real `twisted_rank_lift_2` row for q=2.  Real q=4, q=8, and deeper branch lifts were not constructed as routed prediction tensors or parameter-level models.

## Distinctions

| Category | Current status |
| --- | --- |
| Diagnostics only | Existing primary candidates, order divisibility, pooling residuals, and large/truncated holonomy summaries. |
| Actual quotient construction | Implemented here only when an exact homomorphism `Gamma -> C2/C3` is verified, or for the ambient permutation sign character. |
| Actual branch/model construction | Controlled prediction-level branch lifts are implemented here; no new real parameter-level model is built. |
| Prediction-level lift | Implemented for controlled quotient chains using measured logits and validation-only routing. |
| Parameter-level lift | Not implemented. |
| Uniform invariant pooling | Implemented as a control; zero residual alone is not success because uniform pooling is invariant by construction. |
| Fourier/equivariant pooling | Implemented at prediction level; C2 keeps both plus and minus components before validation readout. |
| Learned or validation routing | Implemented at prediction level using validation labels only. |
| Depth > 1 | Implemented and tested in controlled groups C2xC2, C4, D4, and S3. Not implemented on natural MNIST. |

## Old Primary Sweep Boundary

The old sweep can be used as a diagnostic inventory and a q=2 disagreement-cluster baseline.  It must not be described as a certified cohomological, Brauer, or genuine quotient-driven sequential lift.  Its controlled sanity table included hard-coded target accuracies and therefore is not evidence of measured quotient-lift performance.
