# Spatial-output claim ladder

| Level | Claim | Passed | Evidence |
|---|---|---:|---|
| S1 | output_action_correctness | true | sanity/mask_claims.csv and sanity/output_action_runs.csv |
| S2 | controlled_spatial_retransport | true | exact asymmetric-mask retransport and negative controls |
| S3 | inferred_spatial_retransport | false | biomedical/discovery/claims.csv retransport_gate |
| S4 | twistedmerge_specific_spatial_benefit | false | B1 paired accuracy gates and B4 matched-cost gate |
| S5 | multi_domain_benefit | false | multidomain/claims.csv; domains labeled synthetic |
| S6 | realistic_residual_correction | false | transitions/residuals.csv and correction_claims.csv |
