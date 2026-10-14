from pathlib import Path

from src.kbs_revision.neural_driver import software_smoke


def test_four_task_trained_pipeline_and_resume_without_research_claim(tmp_path):
    root = Path(__file__).resolve().parents[1]
    result = software_smoke(tmp_path, root)
    assert result["status"] == "software_only"
    assert result["independent_research_training_groups"] == 0
    assert not result["final_test_evaluated"]
    assert result["resume_reused_without_reexecution"]
    episode = result["episodes"][0]["result"]
    assert len(episode["candidate_metrics"]) == 11
    assert set(episode["base_metrics"]["per_task_loss"]) == {"BoolQ", "PIQA", "ARC-Challenge", "GSM8K"}
    assert episode["actions_frozen_before_candidate_evaluation"]
    stages = [row["stage"] for row in episode["stage_costs"]]
    assert max(i for i, stage in enumerate(stages) if stage == "policy_rule") < min(i for i, stage in enumerate(stages) if stage == "evaluation_inference")
    assert episode["policies"]["P3"]["selected_action"] == episode["policies"]["P4"]["selected_action"]
    for value in episode["policies"].values():
        assert value["parameter_count"] > 0
        assert value["end_to_end_with_loading_seconds"] >= value["merging_selection_seconds"]
        assert not value["candidate_generation_savings_claim"]
