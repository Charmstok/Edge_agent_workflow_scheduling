import pytest

from edge_agent_workflow_scheduling.common import LLMCall
from edge_agent_workflow_scheduling.resources import (
    LLMInstanceProfile,
    ResourceRegistry,
)
from edge_agent_workflow_scheduling.rl import SchedulingEnv


def _resources() -> ResourceRegistry:
    registry = ResourceRegistry()
    for llm_id, rate in (("llm-a", 100.0), ("llm-b", 200.0)):
        registry.register_llm(
            LLMInstanceProfile(
                llm_id=llm_id, provider="profile", model=llm_id, node_id="local",
                platform="linux", executor_type="profile", model_size_b=9,
                context_window_tokens=4096, capabilities=["function_calling"],
                token_profile={"tokens_per_sec": rate}, max_concurrency=1,
            )
        )
    return registry


def test_environment_reset_step_and_mask_are_reproducible():
    call = LLMCall(
        llm_call_id="call-1", run_id="run-1", agent_id="agent-1",
        input_tokens=10, estimated_output_tokens=10, context_length=20,
    )
    env = SchedulingEnv([call], _resources())
    observation, info = env.reset(seed=7)
    assert env.observation_space.contains(observation)
    assert info["action_mask"].tolist() == [1, 1]
    with pytest.raises(ValueError, match="outside action_space"):
        env.step(2)
    next_observation, _, terminated, truncated, step_info = env.step(0)
    assert terminated and not truncated
    assert next_observation.shape == observation.shape
    assert step_info["decision"]["target_id"] == "llm-a"
