# Synthetic H^2(mu_2) Obstruction Report

This experiment uses the boundary of a tetrahedron as a triangulated 2-sphere.
The face signs are a `mu_2` 2-cocycle.  The trivial case assigns `+1` to every
face.  The nontrivial case assigns `-1` to one face and `+1` to the other three
faces, so the product over all four faces is `-1`; on this complex that class is
not a coboundary.

## Important construction note

An ordinary scalar edge cochain on a closed 2-complex cannot generate a
nonzero `H^2` class: its triangle signs are, by definition, a coboundary.  The
experiment therefore uses twisted descent data: pairwise edge alignments are
locally exact, while the triple-overlap defect includes a prescribed central
2-cocycle.  The code checks this with `is_coboundary_mu2`.

## Commands

```bash
.venv/bin/python experiments/synthetic_h2_mu2_obstruction.py
```

## Outputs

- CSV: `reports/csv/synthetic_h2_mu2_obstruction.csv`
- Obstruction plot: `reports/plots/synthetic_h2_mu2_obstruction_vs_failure.png`
- Rank plot: `reports/plots/synthetic_h2_mu2_rank_success.png`
- Config: `reports/configs/synthetic_h2_mu2_obstruction_config.json`

## Results

| case | rank | local_loss | pairwise_alignment_loss | global_merge_loss | twisted_merge_loss | obstruction_score | is_coboundary | global_sync_success | can_absorb_twist |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| trivial | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | True | True | True |
| nontrivial | 1 | 0.000 | 0.000 | 0.250 | 0.250 | 0.250 | False | False | False |
| trivial | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | True | True | True |
| nontrivial | 2 | 0.000 | 0.000 | 0.250 | 0.000 | 0.250 | False | False | True |
| trivial | 4 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | True | True | True |
| nontrivial | 4 | 0.000 | 0.000 | 0.250 | 0.000 | 0.250 | False | False | True |
| trivial | 8 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | True | True | True |
| nontrivial | 8 | 0.000 | 0.000 | 0.250 | 0.000 | 0.250 | False | False | True |

## Claim status

- Local models have near-zero local loss: supported in this construction.
- Pairwise edge alignment loss is zero: supported.
- Ordinary global merging fails in the nontrivial class: supported here; rank 1
  nontrivial global merge loss is `0.250`.
- Rank 2/doubled representation absorbs the sign twist: supported here; rank 2
  nontrivial twisted merge loss is `0.000`.
- This is a synthetic obstruction witness, not evidence yet for MNIST/CIFAR or
  external model-merging baselines.

## Parameters

- Ranks: `1,2,4,8`
- Samples per face: `512`
- Samples per vertex: `512`
- Seed: `5209`
