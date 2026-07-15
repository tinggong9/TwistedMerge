from experiments.next_program_common import symmetric_group_3
import torch

from experiments.structured_compression import FashionCompressedStudent, run_group


def test_structured_compression_executes_models_and_measures_artifacts():
    rows = run_group("S3", symmetric_group_3(), seed=0, targets=(0.25,), objectives=("supervised",), students=("finite_state_chart_module", "quantized_structured_student"), student_epochs=1)
    assert len(rows) == 2
    assert all(row["student_storage_bytes"] > 0 for row in rows)
    assert all(row["student_artifact_sha256"] for row in rows)
    assert all(row["label_permutation_hash_passed"] for row in rows)


def test_fashion_students_execute_mode_specific_chart_paths():
    images = torch.rand(3, 1, 28, 28)
    for mode in ("chart_token_student", "low_rank_group_generators", "finite_state_chart_module", "tensor_factorized_equivariant_head", "ordinary_single_model_control"):
        model = FashionCompressedStudent(8, mode)
        logits, chart_logits = model(images)
        assert logits.shape == (3, 10)
        assert chart_logits.shape == (3, 8)
