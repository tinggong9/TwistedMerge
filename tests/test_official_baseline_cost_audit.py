from experiments.official_baseline_cost_audit import estimate_flops, measured_proxy_latency


def test_cost_accounting_and_measured_proxy_latency():
    assert estimate_flops(16, 8, 4, 32) == 2 * 16 * 8 * 4 * 32
    latency, memory = measured_proxy_latency(16, 8, 4, 8, 1)
    assert latency >= 0
    assert memory > 0
