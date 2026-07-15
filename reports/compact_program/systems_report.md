# Compact systems and distillation audit

The targeted comparison measured `generic_mixture_of_experts` as the strongest generic context method and `strict_synchronization` as the strongest non-ensemble natural baseline in the completed discovery artifacts. Latency was measured at batch sizes 1, 32, and 128. The most accurate distilled student was `low_rank_adapter_student` at 0.1833 accuracy with KL 0.4380 to the executed lifted teacher.
