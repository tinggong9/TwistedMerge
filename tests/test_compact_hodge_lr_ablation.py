from __future__ import annotations

from experiments.compact_hodge_lr_ablation import context_family, mu2_family


def test_mu2_ablation_has_all_methods() -> None:
    rows = mu2_family(32, 0)
    assert len(rows) == 9
    assert all(row["leakage_hash_passed"] for row in rows)


def test_context_ablation_has_all_methods() -> None:
    rows = context_family("S3", 0)
    assert len(rows) == 9
