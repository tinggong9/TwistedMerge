from experiments.cifar10_chart_retransport import CONFIRMATION_SEEDS, DISCOVERY_SEEDS


def test_cifar_discovery_and_confirmation_seeds_are_preregistered_and_disjoint():
    assert tuple(DISCOVERY_SEEDS) == tuple(range(5))
    assert tuple(CONFIRMATION_SEEDS) == tuple(range(5, 10))
    assert set(DISCOVERY_SEEDS).isdisjoint(CONFIRMATION_SEEDS)
