# Paper-editor evidence brief: model-lineage holonomy

## Recommended wording

Different learning orders produced measurable terminal differences, but holonomy supplied no reliable incremental predictive or corrective value beyond pairwise diagnostics.

## Required limitations

- Five independent training seeds are the inferential units; lineage rows and layers are not replicates.
- The model is one frozen ResNet-18 encoder with rank-4 feature adapters and a classifier on three deterministic CIFAR-10 corruptions.
- Prediction is double held out by seed and task-order family.
- Ensemble and sequential-adaptation rows are upper-bound or additional-training controls.
- No natural Brauer, topology, universal continual-learning, or broad model-merging claim is permitted.

## Gate status

`{'H1': False, 'H2': False, 'H3': False, 'H4': False}`
