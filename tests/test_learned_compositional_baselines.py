from experiments.learned_compositional_baselines import (
    EquivariantTransformer,
    find_generators,
    make_word_data,
    run_group,
)
from experiments.next_program_common import symmetric_group_3


def test_real_learned_compositional_models_execute_without_proxies():
    table = symmetric_group_3()
    assert len(set(find_generators(table))) == 2
    data = make_word_data(table, seed=1, train_size=32, test_per_length=2)
    assert len(data[2]) == 14
    model = EquivariantTransformer(table, data[7], data[8], width=8)
    assert model.base.encoder is not None
    rows, efficiency = run_group("S3", table, seed=0, epochs=1, train_size=64, test_per_length=2)
    assert len(rows) == 12
    assert efficiency
    implementations = {row["method"]: row["implementation"] for row in rows}
    assert implementations["ordinary_sequence_transformer"] == "SequenceTransformer"
    assert implementations["differentiable_finite_state_automaton"] == "DifferentiableAutomaton"
    assert all(row["label_permutation_hash_passed"] for row in rows)
