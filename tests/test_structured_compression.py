from experiments.next_program_common import symmetric_group_3
from experiments.structured_compression import run_group


def test_structured_compression_executes_models_and_measures_artifacts():
    rows = run_group("S3", symmetric_group_3(), seed=0, targets=(0.25,), objectives=("supervised",), students=("finite_state_chart_module", "quantized_structured_student"), student_epochs=1)
    assert len(rows) == 2
    assert all(row["student_storage_bytes"] > 0 for row in rows)
    assert all(row["student_artifact_sha256"] for row in rows)
    assert all(row["label_permutation_hash_passed"] for row in rows)
