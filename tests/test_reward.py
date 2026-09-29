import pytest

from edge_agent_workflow_scheduling.rl import MultiObjectiveReward
from edge_agent_workflow_scheduling.scheduler.objectives import (
    ObjectiveNormalization,
    ObjectiveWeights,
)


def test_reward_reports_weighted_terms_and_negative_cost():
    reward = MultiObjectiveReward(
        weights=ObjectiveWeights(
            latency=0.5, energy=0.0, deadline_miss=0.0, load_imbalance=0.0, quality=0.5
        ),
        normalization=ObjectiveNormalization(latency_ref_sec=2.0, energy_ref_joules=10.0),
    )
    result = reward.calculate_from_dict(
        {
            "latency_sec": 1.0,
            "energy_joules": None,
            "quality": 0.8,
            "deadline_miss": 0,
            "load_imbalance": 0.0,
        }
    )
    assert result.reward == pytest.approx(-0.35)
    assert result.normalized_terms["latency"] == pytest.approx(0.25)
    assert result.normalized_terms["quality"] == pytest.approx(0.1)


def test_reward_rejects_missing_weighted_metric():
    reward = MultiObjectiveReward(
        weights=ObjectiveWeights(
            latency=0.5, energy=0.5, deadline_miss=0.0, load_imbalance=0.0, quality=0.0
        )
    )
    with pytest.raises(ValueError, match="energy_joules"):
        reward.calculate_from_dict(
            {
                "latency_sec": 1.0,
                "energy_joules": None,
                "quality": 1.0,
                "deadline_miss": 0,
                "load_imbalance": 0.0,
            }
        )
