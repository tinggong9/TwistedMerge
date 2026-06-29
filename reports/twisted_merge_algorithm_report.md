# TwistedMerge Algorithm Report

## Pseudocode

```text
TwistedMerge(M_i, g_ij=None, alpha_ijk=None, q=2):
  1. If g_ij is absent, estimate pairwise alignments from local models.
  2. Compute c_ijk = g_ij g_jk g_ki on each triangle.
  3. Try to find gauges h_i with g_ij ~= h_j h_i^{-1}.
  4. If the residual is small, align all M_i by h_i^{-1} and average.
  5. Otherwise compare c_ijk with the supplied finite central twist alpha_ijk.
  6. If c_ijk ~= alpha_ijk I and q >= 2 for mu_2, form a doubled branch
     representation with branches (w, -w) and central action rho(-1) =
     [[0, 1], [1, 0]].
  7. Evaluate ordinary merge, cycle-consistent merge, twisted merge, and
     ensemble; report the cycle score and twist residual.
```

## Prototype Implementation

- Main class: `src.twisted_merge_algorithm.TwistedMerge`
- Generic matrix defect routine: `compute_triangle_defects(g)`
- Gauge routine: `try_global_gauge_synchronization(g)`
- Finite central check: `finite_central_twist_close(defects, alpha)`
- mu_2 doubled representation: stores two branches `(w, -w)` and represents
  the nontrivial central sign by the 2x2 branch-swap matrix
  `[[0, 1], [1, 0]]`.

The lifted transition for a base alignment `G_ij` and a central sign is
`kron(rho(sign), G_ij)`, so its matrix size is `2r x 2r` for q=2.

## Commands

```bash
.venv/bin/python experiments/twisted_merge_algorithm_demo.py
```

## Numerical Results

| case | rank | q | status | cycle_score | twist_residual | ordinary_loss | cycle_consistent_loss | twisted_loss | ensemble_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| trivial | 1 | 1 | ordinary | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| nontrivial | 1 | 1 | failed | 1.0000 | 0.0000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 |
| trivial | 1 | 2 | ordinary | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| nontrivial | 1 | 2 | twisted_rank_lifted | 1.0000 | 0.0000 | 0.5000 | 0.5000 | 0.0000 | 0.5000 |
| trivial | 2 | 1 | ordinary | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| nontrivial | 2 | 1 | failed | 1.0000 | 0.0000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 |
| trivial | 2 | 2 | ordinary | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| nontrivial | 2 | 2 | twisted_rank_lifted | 1.0000 | 0.0000 | 0.5000 | 0.5000 | 0.0000 | 0.5000 |
| trivial | 4 | 1 | ordinary | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| nontrivial | 4 | 1 | failed | 1.0000 | 0.0000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 |
| trivial | 4 | 2 | ordinary | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| nontrivial | 4 | 2 | twisted_rank_lifted | 1.0000 | 0.0000 | 0.5000 | 0.5000 | 0.0000 | 0.5000 |
| trivial | 8 | 1 | ordinary | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| nontrivial | 8 | 1 | failed | 1.0000 | 0.0000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 |
| trivial | 8 | 2 | ordinary | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| nontrivial | 8 | 2 | twisted_rank_lifted | 1.0000 | 0.0000 | 0.5000 | 0.5000 | 0.0000 | 0.5000 |

## Numerical Stability

- Gauge synchronization uses Moore-Penrose pseudoinverses and normalized
  Frobenius residuals.
- Central-twist matching uses normalized Frobenius distance to `alpha_ijk I`.
- The default tolerances are `tolerance=1e-06` and
  `central_tolerance=1e-05`.
- For near-singular dense alignments, use orthogonal projection or polar
  cleanup before defect computation; this prototype does not silently project
  arbitrary matrices.

## When It Works

- The ordinary branch works when triangle defects are close to identity.
- The q=2 mu_2 branch works in this controlled construction when the observed
  central defects match the supplied sign twist. In the nontrivial case, q=1
  twisted loss is `0.5000` while q=2 twisted loss
  is `0.0000`.
- The doubled branch improves downstream prediction when the task labels
  actually depend on the same sign sector as the twist.

## When It Fails

- If no twist is supplied and gauge trivialization fails, the algorithm reports
  failure instead of inventing a correction.
- If q < 2 for a nontrivial mu_2 twist, the doubled representation is not
  available.
- If pairwise alignments are noisy and defects are not close to either identity
  or a finite central twist, the prototype should be treated as diagnostic only.
- The doubled representation absorbs the central sign at the branch/prediction
  level. It does not prove that a nonzero H^2 class became an ordinary untwisted
  vector bundle on the same cover.

## Output Files

- CSV: `reports/csv/twisted_merge_algorithm_demo.csv`
- Config: `reports/configs/twisted_merge_algorithm_demo_config.json`
- Report: `reports/twisted_merge_algorithm_report.md`
