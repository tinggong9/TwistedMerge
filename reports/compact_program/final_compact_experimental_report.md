# Final compact experimental report

## Execution summary

- Execution commit at report generation: `c4fa3a6653becbffddcbee6e03c7c99b1bdd30e4`.
- Existing test suite: `389 passed, 5 subtests passed in 21.11s`.
- Runtime by runner stage: `{"0": 23.536, "1": 17.377, "2": 7.326, "3": 47.594, "4": 58.572, "5": 7.78, "6": 0.912, "7": 0.524}`.
- Natural checkpoint collections: 48 mandatory discovery collections, 6 optional CIFAR collections, and 0 confirmation collections.
- Pretrained checkpoint collections: 3; federated frame collections: 18.
- Fresh reusable natural checkpoint pool: 120 local models.
- Pretrained checkpoint status: executed.

## Decisions

- Context fairness: **pass**; best generic method `generic_mixture_of_experts`; structured mean 0.6378, generic mean 0.5636 over the discovery aggregate.
- Hodge and low-rank contribution: **promoted in controlled families**; real-image frame negatives are retained.
- Natural residual: **not promoted**.
- Pretrained vision: **gate not passed**.
- Federated frame: **no persistent lift gain found**.
- Strongest supported scientific claim: **Level 2**.

## Reproducibility and public-release policy

The output contains numerical evidence, negative findings, commands, checksums, and paste-ready tables. The next justified expansion is the exact positive family identified by a passed conditional gate; failed discovery families are not expanded merely to consume compute.

## Paste-ready tables

- `tables/context_main.tex`
- `tables/context_efficiency.tex`
- `tables/hodge_ablation.tex`
- `tables/natural_main.tex`
- `tables/vision_main.tex`
- `tables/federated_main.tex`
- `tables/systems.tex`
