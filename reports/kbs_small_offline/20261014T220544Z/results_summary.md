# Small offline digits: REDIRECT

Actual completed pilot groups: 2; independently trained
post-pilot groups scored on sealed heldout IDs: 4. The post-pilot
REDIRECT audit used the first four predeclared seeds, rather than all eight,
because structural applicability was absent. These are training-randomness
replicates conditional on one shared split, not independent datasets or writers.

| Seed | P0 loss | P1 loss | P4 loss | P4-P0 | P4-P1 |
| --- | --- | --- | --- | --- | --- |
| 1201 | 0.185668 | 0.209283 | 0.185668 | 0.000000 | -0.023616 |
| 1202 | 0.205212 | 0.205212 | 0.205212 | 0.000000 | 0.000000 |
| 1203 | 0.179153 | 0.184853 | 0.179153 | 0.000000 | -0.005700 |
| 1204 | 0.184039 | 0.184039 | 0.184039 | 0.000000 | 0.000000 |

P4_minus_P0: mean 0.00000000, descriptive group-bootstrap 95% interval [0.0, 0.0].
P4_minus_P1: mean -0.00732899, descriptive group-bootstrap 95% interval [-0.01771172638436485, 0.0].

Loss is 1-accuracy averaged equally over four domains of the same original
heldout digit IDs. Lower is better. The four domains and two gauge controls
within each seed are dependent. Small-n bootstrap intervals are descriptive;
no powered significance or formal noninferiority claim is made. All action
choices were hash-frozen before one test unseal. Pilot groups are excluded
from heldout estimates. See per_group.csv and per_domain.csv for exact rows.

P4 changed 0/4 actions versus P0 and
2/4 versus P1; changed and lower-loss
decisions: 0 and 2 respectively.
Different heldout classification predictions versus P1 occurred in
2/4 groups; per_group.csv
separates action names from actual classification disagreement rates.
P4 was inapplicable in every group because no justified neural central
coefficient/closure/uncertainty model was identified. Formal full-simplex
constant-real H2=0 is not a learned neural certificate; no Hodge projection,
finite class, ROC or risk/coverage curve was invented.

Actual stand-alone sequential times (seconds, means across heldout groups):

| Policy | Invocation | Mean total seconds | Groups |
| --- | --- | --- | --- |
| P0 | cold | 0.007139 | 4 |
| P0 | warm | 0.005065 | 4 |
| P1 | cold | 0.017083 | 4 |
| P1 | warm | 0.015572 | 4 |
| P4 | cold | 0.006993 | 4 |
| P4 | warm | 0.004917 | 4 |

Cold denotes fresh local model/factor state, with process/library/OS caches
potentially warm; it is not a process-startup or disk-cache cold claim. Warm
retains only loaded base/factors, rebuilding all policy-specific candidates.
Times include actual loading/preparation/diagnostics/generation/validation/
selection and heldout inference. No candidate panel is shared across policies;
P4 exits on applicability before edges or alternative candidates. Microsecond
overheads and invocation ordering make tiny P0/P4 time differences noisy.
Full stage rows and per-policy sampled RSS are in stage_costs.csv.

Every base has 13,130 parameters. Each specialist adds 1,024 parameters across
two rank-4 hidden layers; each selected predictor has 14,154 factorized
parameters. Base/head stayed frozen during specialist training. No rank-16
concatenation or ensemble capacity advantage is included. Dense SVD is parity
only, not a distinct leaderboard method. Numerical reload and compact/dense
errors, actual quality and training times are in the model/group manifests.

Gauge controls are deliberately imposed equivalent representations, not natural
trained holonomy. Their individual delta/logit invariance and merge instability
are measured on D_align in gauge_controls.csv; they add no independent groups.
Noise is independently seeded by original ID then held fixed across roles,
groups and repeated invocations; domain views share originals within a role.
Original IDs never cross roles, and exact identical source images were audited.
Transforms are intended label-preserving; translation clipping/occlusion may
remove digit information, and no writer-disjoint generalization is established.

This is a small real-data/from-scratch neural experiment. It does not establish
modern pretrained-model efficacy, a natural Brauer class, or useful structural
certification. The practical-method KBS advantage remains unsupported; retain
the narrower theory/diagnostic framing and negative result.
