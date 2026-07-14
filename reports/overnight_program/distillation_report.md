# Distillation audit

The component smoke reduced teacher/student KL from 0.29736 to 6.84134e-09. In controlled mu2, the supplied-context teacher and distilled single-model accuracies are recorded in `distillation_summary.csv`; the distilled model is not relabeled as a successful lift. No pretrained vision or language branch teacher was sufficiently supported for publication-grade distillation.
