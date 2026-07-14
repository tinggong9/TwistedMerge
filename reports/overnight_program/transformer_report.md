# Stage 11: shared-base Transformer merging smoke

Four checkpoints of one local tiny PyTorch Transformer were fine-tuned on four fixed synthetic sequence tasks and merged with averaging, greedy selection, Task Arithmetic, TIES, DARE, SLERP, a low-rank control, and conservative TwistedMerge fallbacks. All predictions and saved logits were executed; label permutation leaves saved logits unchanged. No obstruction certificate passed and lift frequency is zero.

Exact blocker: this is not an open pretrained transformer. `transformers` and `datasets` are not installed, no pretrained checkpoint/tokenizer is cached or pinned, and no real sentiment/topic/NLI/domain datasets are available. Full mode refuses to substitute these synthetic scores. After installing and pinning the model, tokenizer, license, datasets, and four fine-tuned checkpoints, run `python experiments/shared_base_transformer_merging.py --mode full`.
