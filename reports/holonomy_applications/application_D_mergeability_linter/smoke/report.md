# Application D: Holonomy-Aware Mergeability Linter

Decision: **bounded smoke; sample inadequate**.

## Commands

Smoke: `/Users/tinggong/Documents/GitHub/TwistedMerge/.venv/bin/python experiments/holonomy_application_D.py --mode smoke`

Confirmatory: `/Users/tinggong/Documents/GitHub/TwistedMerge/.venv/bin/python experiments/holonomy_application_D.py --mode confirmatory`

Executed: `/Users/tinggong/Documents/GitHub/TwistedMerge/.venv/bin/python experiments/holonomy_application_D.py --mode smoke`

## Leakage and model boundary

The linter uses only accumulated A-C outputs plus shared adapter checkpoint metadata. It creates no new image/model corpus. Every prediction is double held out: the test observation's corpus seed and its entire setting family are both absent from training. The model class is logistic regression for every feature set; no tree or boosted fallback was tried after seeing results.

## Data and outcomes

- Observation rows: 136.
- Independent corpus seeds: 1.
- Seed-family cells: 4.
- Setting families: ['natural_application_A', 'period2_index2', 'period2_index4', 'period3_index3'].
- Outcome counts: `{'ordinary_fusion_harmful': 68.0, 'gauge_sync_sufficient': 16.0, 'branch_lift_beneficial': 80.0, 'projective_rank_expansion_required': 72.0, 'abstention_recommended': 69.0}`.
- Recommended-capacity accuracy inherited from controlled Application C: `1.000`.

## Primary result

- Baseline: `pairwise_plus_ordinary_sync`.
- Full diagnostic: `full_holonomy_projective`.
- Incremental discrimination/calibration gate: `False`.

evidence_label,mode,outcome,feature_set,status,rows,positive_examples,negative_examples,independent_seeds
diagnostic_only,smoke,ordinary_fusion_harmful,pairwise_residual_only,inadequate_sample,136,68,68,1
diagnostic_only,smoke,ordinary_fusion_harmful,parameter_distance_only,inadequate_sample,136,68,68,1
diagnostic_only,smoke,ordinary_fusion_harmful,prediction_disagreement_only,inadequate_sample,136,68,68,1
diagnostic_only,smoke,ordinary_fusion_harmful,pairwise_plus_ordinary_sync,inadequate_sample,136,68,68,1
diagnostic_only,smoke,ordinary_fusion_harmful,holonomy_features,inadequate_sample,136,68,68,1
diagnostic_only,smoke,ordinary_fusion_harmful,full_holonomy_projective,inadequate_sample,136,68,68,1
diagnostic_only,smoke,ordinary_fusion_harmful,oracle_structural_labels,inadequate_sample,136,68,68,1


The only allowed positive claim would require holonomy/projective features to improve double-held-out discrimination without worsening Brier score. That gate did not pass. Controlled capacity labels remain separate from natural mergeability evidence.
