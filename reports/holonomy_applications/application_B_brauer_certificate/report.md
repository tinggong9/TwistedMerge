# Application B: Conservative Brauer-Like Certificate

Decision: **no natural Brauer-like candidate certified**.

## Commands

Smoke:

```bash
/Users/tinggong/Documents/GitHub/TwistedMerge/.venv/bin/python experiments/holonomy_application_B.py --mode smoke
```

Confirmatory:

```bash
/Users/tinggong/Documents/GitHub/TwistedMerge/.venv/bin/python experiments/holonomy_application_B.py --mode confirmatory
```

Executed command: `/Users/tinggong/Documents/GitHub/TwistedMerge/.venv/bin/python experiments/holonomy_application_B.py --mode confirmatory`

## Result

- Natural adapter corpus only; no adapter training and no test-label access.
- Medium-threshold classification counts: `{'noncentral_holonomy': 15, 'trivial_coboundary': 5}`.
- Application-A-selected connection classifications: `{'trivial_coboundary': 5}`.
- Natural central finite-order candidates: `0`.
- Settings whose label is unchanged across strict/medium/loose thresholds: `20` / `20`.

The weight-derived transitions selected by held-out overlap residual in Application A are classified as trivial/coboundary. Their near-identity triangle defects follow from the construction `Q_ij = A_j pinv(A_i)` and are not evidence of a nontrivial projective class. Activation-Procrustes, low-rank, and joint connections produce larger defects, but they fail centrality and/or stability before torsion or cohomological language is admissible.

## Certificate boundary

A nonzero cycle defect was never treated as sufficient. Each triangle records scalarity, normalized residual, eigenvalue dispersion, commutation with local transitions, finite-root fits through order 6, bootstrap intervals, and gauge sensitivity. All 70 tetrahedra per seed/method are checked, followed by a scalar edge-rephasing fit. No natural row passes centrality, torsion, cocycle, nontriviality modulo coboundaries, gauge invariance, bootstrap stability, and predicted lift behavior together.

The only defensible natural conclusion is negative: selected residuals are trivial/coboundary, while alternative activation-derived residuals are predominantly noncentral or unstable. The phrase `Brauer class` is not used for them.
