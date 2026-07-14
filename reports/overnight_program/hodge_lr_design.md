# TwistedMerge-Hodge/LR design

The implementation estimates functional transition maps in permutation, positive-monomial, orthogonal, whitened-linear/CCA, low-rank, LoRA-basis, and block-orthogonal families. Diagnostics separate calibration fit, held-out fit, rank, conditioning, inverse consistency, cycle distance, centrality distance, and residual spectrum.

Weighted edge cochains are decomposed as a removable gradient component, a harmonic component, and the weighted-adjoint face/coexact component. The code verifies boundary-of-boundary, weighted orthogonality, and exact reconstruction. A harmonic numerical component is called persistent cycle structure, not an H^2 obstruction, unless external closure and coefficient assumptions are supplied.

Only the SVD-selected residual subspace is eligible for a lift. The dispatcher defaults to strict synchronization for removable structure and the validated ordinary family for uncertified structure. Central and noncentral lifts require structural certificates, representation-rank checks where applicable, and a positive lower confidence bound on validation gain. Routing uses inference-available features; distillation consumes teacher probabilities and not labels.
