# Stage 9: LoRA/adapter holonomy smoke

Four rank-3 adapters over one fixed linear base were executed on four synthetic domains. Gauge-equivalent factor transformations preserve every delta matrix, pairwise basis maps and cycle residuals are measured, and all saved-logit leakage checks pass. This is an algebra/prediction smoke, not an open-pretrained-model result.

Exact blocker: `transformers`, `datasets`, and `peft` are absent, and no small open pretrained checkpoint or four real adapters are installed. Full mode refuses to substitute simulation. Install and pin those dependencies/checkpoints, then run `python experiments/lora_holonomy_merging.py --mode full`.
