# CIFAR-10 chart retransport

Execution commit: `11c8fb838cfa616d934dfc724777b3cd3fa6c05e`. Discovery used seeds [0, 1, 2, 3, 4]; confirmation was not triggered by the preregistered discovery gate.

All methods used identical CIFAR-10 splits and transformations within each seed. Candidate logits were persisted before final labels were evaluated, and the saved hashes were unchanged by the label-permutation audit.

The discovery criteria and worst-condition interval are recorded in `gate.csv`; failed criteria remain negative findings.
