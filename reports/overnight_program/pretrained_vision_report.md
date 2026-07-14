# Stage 8: pretrained vision merging smoke

One frozen-backbone ResNet-18/CIFAR-10 smoke completed with four specialized heads, a separate validation split, saved-logit leakage regression, Task Arithmetic, TIES, DARE, SLERP, greedy soup, weight averaging, and the validation-only TwistedMerge selector. It is feasibility evidence only.

Exact blockers: one seed rather than five; CIFAR-100 absent; the backbone is frozen rather than partially/fully fine-tuned; only class-group specialization is run; official Git Re-Basin, C2M3, RegMean, representation-alignment, and low-rank implementations are not integrated; official external code is pinned in metadata but internal implementations are used for Task Arithmetic/TIES/DARE. Full command is deliberately refused until those conditions are supplied. No obstruction certificate passed and no branch candidate activated.
