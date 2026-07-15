from experiments.compression_claim_reaudit import storage_gate


def test_storage_claim_requires_reduction_and_retained_gain():
    assert storage_gate(0.25, 0.95)
    assert not storage_gate(0.249, 1.0)
    assert not storage_gate(0.50, 0.949)
    assert not storage_gate(0.50, None)
