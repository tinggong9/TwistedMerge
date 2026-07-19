# Application C: Period-Index Capacity Planner

Decision: **controlled structural threshold without practical superiority**.

## Commands

Smoke: `/Users/tinggong/Documents/GitHub/TwistedMerge/.venv/bin/python experiments/holonomy_application_C.py --mode smoke`

Confirmatory: `/Users/tinggong/Documents/GitHub/TwistedMerge/.venv/bin/python experiments/holonomy_application_C.py --mode confirmatory`

Executed: `/Users/tinggong/Documents/GitHub/TwistedMerge/.venv/bin/python experiments/holonomy_application_C.py --mode confirmatory`

## Scope

Evidence label: `controlled_on_real_features`. The experiment uses the same frozen ResNet features, the same 8 chart adapters, and actual local-model logits. It adds no dataset and retrains no adapter. The finite-Heisenberg carrier is planted and exact at capacities divisible by its index; this is not a naturally discovered class.

## Result

- Cases: period 2/index 2, period 2/index 4, and period 3/index 3.
- Independent adapter seeds: 5.
- Structural-plus-task success matched divisibility: `True`.
- Predicted index equaled every minimum successful controlled capacity: `True`.
- Predicted-index lift beat parameter-matched generic unitary capacity: `False`.
- Practical capacity-planner gate: `False`.

The controlled layer verifies that complete projective blocks restore exact relations and preserve actual classifier logits. However, matched generic unitary carriers recover the same real-task predictions without waiting for the projective index. Therefore the structural threshold does not translate into a uniquely useful capacity recommendation on this task.

Several index-insufficient truncations also match or exceed the ensemble's classification accuracy while failing the exact relation/unitarity gate. Accuracy alone therefore does not identify the period-index threshold; the observed minimum is a controlled structural definition, not an independently emerging task threshold.

## Boundary

The positive part, if any, is controlled algebra tied to real frozen features and actual logits. It is not evidence of a natural Brauer-like obstruction. A practical period-index claim requires superiority over matched generic capacity; that gate failed.
