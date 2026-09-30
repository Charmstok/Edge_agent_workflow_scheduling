import numpy as np

from edge_agent_workflow_scheduling.rl import DoubleDQNAgent, DQNConfig, Transition


def test_double_dqn_respects_action_mask_and_checkpoint(tmp_path):
    agent = DoubleDQNAgent(
        3,
        2,
        config=DQNConfig(episodes=1, replay_capacity=8, batch_size=1, warmup_steps=1),
        seed=4,
    )
    observation = np.asarray([0.1, 0.2, 0.3], dtype=np.float32)
    assert agent.select_action(observation, np.asarray([0, 1], dtype=np.int8), epsilon=0.0) == 1
    for _ in range(3):
        agent.observe(Transition(observation, 1, -0.2, observation, np.asarray([1, 0]), True))
    path = tmp_path / "checkpoint.json"
    agent.save(path, {"resource_version": "test-v1", "reward_weights": {"latency": 1.0}})
    restored, metadata = DoubleDQNAgent.load(path)
    assert metadata["resource_version"] == "test-v1"
    assert restored.total_steps == agent.total_steps
    assert restored.select_action(observation, np.asarray([1, 0], dtype=np.int8), epsilon=0.0) == 0
