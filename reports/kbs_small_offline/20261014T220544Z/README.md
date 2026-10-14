# Reproduce the offline small digits audit

Source commit: `c0c637b1a755277852fa01aef05e092c51cd5137`; exact module/test/helper hashes in
protocol_frozen.json (or pilot protocol if STOP). CPU-only, two Torch threads.
Hardware: Apple M1 MacBookPro17,1,8 GiB shared system RAM (hardware_inventory.json).
Uses installed sklearn load_digits, all weights initialized/trained locally.
Zero-network attestation: no network/data fetch/model service used; socket
connect, connect_ex, create_connection and DNS are blocked during execution.
No externally pretrained model, AI inference service or extra agent is an
experimental dependency. Codex wrote/reviewed software only.

```
.venv/bin/python -m pytest -q tests/test_kbs_small_offline.py
.venv/bin/python experiments/kbs_small_offline.py --out reports/kbs_small_offline/<fresh-UTC-run>
```

Limits: incremental peak RSS <=1 GiB over explicitly measured post-import
startup high-water baseline;150 MB newly written run artifacts;20 min pilot;
90 min total. Conservative 128 MiB incremental admission plus256 MiB host
safety reserve. A5ms monitor tracks RSS/time, high-water checks run every
batch/stage, with disk/time/memory checkpoints after training stages. Native
kernels are not preempted; any cap event stops the run as BLOCKED, not a
scientific negative. Actual receipt is resource_checkpoints.jsonl.

Do not use the old unavailable data/checkpoints mounts. Weights are small
safe numeric NPZ files used locally with allow_pickle=False; only hashes are
committed, and weights are retained locally under the ignored weights/ directory. Data IDs/hash/split
and model hashes are in models_and_splits_manifest.json. Stratification uses
labels to assign original IDs; sealed test labels are not scored/selected on
until all source, protocol and policy actions pass frozen-hash checks.

D_K declares common comparison contexts before edges; D_align supplies
pairwise-agreement/gauge controls; D_cert is reserved and not inspected because
no neural certificate model is applicable. D_select drives ordinary selection;
pilot supplies quality/admission only. D_test is unsealed once after all
post-pilot actions are frozen. No final-test tuning. Review results_summary.md,
decision_utility.md and arxiv_v2_claim_map.md with the raw CSVs.
