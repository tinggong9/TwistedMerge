# TwistedMerge Algorithm Verification

Generated from the `TwistedMerge` working tree after hardening `src/twisted_merge_algorithm.py`.

## Commit Hash

- Source base commit at verification time: `49d3001c97fb78f416b7dc0f87554f71093c26bd`
- Note: this report is part of the hardening changes made after that base commit. Use `git log` for the final commit containing this report.

## Exact Commands Run

```bash
sed -n '1,520p' src/twisted_merge_algorithm.py
sed -n '1,360p' experiments/twisted_merge_algorithm_demo.py
sed -n '1,360p' src/simplicial_mu2.py
sed -n '1,260p' reports/twisted_merge_algorithm_report.md
sed -n '1,260p' reports/synthetic_obstruction_report.md
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python experiments/twisted_merge_algorithm_demo.py
.venv/bin/python -m compileall src experiments tests
git diff --check
git rev-parse --short HEAD
git rev-parse HEAD
```

## Regression Test Results

| Test | Expected Behavior | Result |
| --- | --- | --- |
| `test_trivial_twist_q1_is_ordinary` | trivial twist, `q=1`: status `ordinary`, gauge succeeds, ordinary loss is `0` | PASS |
| `test_nontrivial_finite_central_twist_q1_fails` | finite central twist, `q=1`: status `failed`, gauge fails, no twisted model | PASS |
| `test_nontrivial_finite_central_twist_q2_builds_branch_lift` | finite central twist, `q=2`: status `twisted_rank_lifted`, gauge fails, twist residual is `0`, twisted loss is `0` | PASS |
| `test_nontrivial_finite_central_twist_without_alpha_fails` | finite central twist with no supplied alpha: fail instead of inventing a twist | PASS |
| `test_wrong_alpha_fails_with_nonzero_residual` | wrong alpha: fail and report nonzero twist residual | PASS |
| `test_noisy_central_defects_respect_tolerance` | small controlled perturbation: loose tolerance accepts, strict tolerance rejects | PASS |
| `test_h2_nontrivial_tetrahedral_twist_is_not_absorbed_by_algorithm` | nonzero `H^2(mu_2)` tetrahedral twist: current algorithm does not absorb it | PASS |
| `test_lifted_transition_maps_encode_nontrivial_edge_sign` | lifted transition maps encode a nontrivial `mu_2` edge sign, not the old placeholder | PASS |

Command output:

```text
Ran 8 tests in 0.041s
OK
```

## Lifted-Transition Check

Before this hardening pass, `lifted_transition_maps` were placeholders: every lifted edge used the trivial central sign. The prototype now solves for a small `mu_2` edge cochain `beta_ij` when the supplied finite central twist is a coboundary and builds

```text
G'_ij = rho(beta_ij) tensor G_ij
```

For the controlled finite central example, the solver finds a nontrivial edge sign on `(0, 2)`, and the regression test checks that the lifted map is `rho(-1) tensor G_02`, not `rho(+1) tensor G_02`.

This is still not a proof that nonzero `H^2` data has become an ordinary untwisted vector bundle. For non-coboundary face data, there is no `mu_2` edge cochain for the current lifted-map construction.

## What The Prototype Proves

The hardened prototype supports the following controlled sanity claims:

- It detects failed gauge synchronization in a finite central `mu_2` twist example.
- It refuses to invent a twist when no `alpha_ijk` is supplied.
- It rejects a wrong supplied `alpha_ijk`.
- It respects numerical tolerances for near-central noisy defects.
- For a coboundary finite central sign twist, `q=2` branch lifting recovers perfect prediction on the synthetic branch-labeled task.

## What The Prototype Does Not Prove

The prototype does not yet prove:

- that TwistedMerge beats Git Re-Basin, C2M3, Model Soups, or other external baselines;
- that TwistedMerge solves natural MNIST/CIFAR model merging;
- that the q=2 branch predictor fully trivializes a nonzero `H^2(mu_2)` class as an ordinary untwisted vector bundle;
- that noisy real neural-network alignments are close enough to a finite central twist for the rank lift to be valid.

## Separating The Three Evidence Tracks

| Track | Status | Interpretation |
| --- | --- | --- |
| (A) Finite central coboundary twist absorbed by q=2 branch lift | Supported by regression tests | This is the algorithmic sanity check in `tests/test_twisted_merge_algorithm.py`. It uses edge-realizable central defects and a supplied alpha. |
| (B) Nonzero `H^2(mu_2)` obstruction witness | Supported as a separate obstruction witness, not solved by `TwistedMerge` | `reports/synthetic_obstruction_report.md` shows the non-coboundary tetrahedral face class. The current `TwistedMerge` prototype does not absorb it through pairwise edge maps. |
| (C) Real model-merging benchmark | Smoke-scale only | `reports/model_merging_report.md` records MNIST/CIFAR plumbing and weak diagnostic signal. It does not establish a natural-image merging claim. |

## Bottom Line

The prototype now fails in the right cases and succeeds for the right controlled reason: a supplied, edge-realizable finite central `mu_2` twist with `q=2`. The nonzero `H^2(mu_2)` experiment remains a separate obstruction witness and is not silently counted as solved by the algorithm.
