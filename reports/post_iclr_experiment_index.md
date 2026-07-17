# Post-ICLR Experiment Index

| Phase | Status | Primary artifacts | Claim boundary |
| --- | --- | --- | --- |
| Repository and artifact audit | Complete | `post_iclr_experiment_gap_audit.md`; `post_iclr_experiment_plan.md`; `csv/post_iclr_experiment_manifest.csv` | Inventory and prioritization only; no performance claim. |
| Official baseline integration | Complete | `external_baselines/POST_ICLR_INTEGRATION.md`; `post_iclr_official_baseline_report.md`; `csv/post_iclr_official_baseline_runs.csv`; `csv/post_iclr_official_baseline_summary.csv`; `csv/post_iclr_baseline_regime_audit.csv`; `configs/post_iclr_official_baseline_config.json`; `plots/post_iclr_official_baseline_deltas.pdf`; `tables/post_iclr_official_baseline.tex` | Adapter-assisted official cores only. Narrow exact-setting paired claims are allowed; no broad official-baseline or SOTA claim. |
| ResNet-18 and BatchNorm-aware gauges | Pending | Planned under a dedicated config/report namespace | Existing no-BatchNorm CIFAR and frozen-feature artifacts do not satisfy this phase. |
| Planted real-network obstruction | Pending | Planned `experiments/planted_real_network_obstruction.py` and matching report/config/CSVs | Old target-injected nonabelian accuracy artifacts are excluded. |
| Diagnostic prediction and conservative selection | Pending | Existing predictor and matched-selector reports are inputs only | Fallback to the best validation-selected ordinary method unless a residual is certified. |
| Biomedical multi-site classification | Pending | Separate future `experiments/biomedical_multisite/` and report namespace | Existing Kvasir segmentation has synthetic domains and no clinical/site claim. |
| Secondary architecture / LoRA | Deferred | Design only until earlier gates close | No large language-model campaign. |
