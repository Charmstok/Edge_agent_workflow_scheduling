"""Training and evaluation helpers for replay-call scheduling episodes."""

from __future__ import annotations

from typing import Any

import numpy as np

from edge_agent_workflow_scheduling.rl.dqn import DoubleDQNAgent, Transition


def train_agent(env, agent: DoubleDQNAgent, episodes: int, *, seed: int) -> list[dict[str, Any]]:
    history = []
    for episode in range(episodes):
        observation, info = env.reset(seed=seed + episode)
        action_mask = np.asarray(info["action_mask"], dtype=np.int8)
        episode_reward = 0.0
        losses = []
        terminated = False
        while not terminated:
            action = agent.select_action(observation, action_mask, epsilon=agent.epsilon())
            next_observation, reward, terminated, truncated, step_info = env.step(action)
            if truncated:
                raise RuntimeError("SchedulingEnv unexpectedly truncated an episode")
            next_mask = np.asarray(step_info["action_mask"], dtype=np.int8)
            loss = agent.observe(Transition(
                observation=observation.copy(), action=action, reward=float(reward),
                next_observation=next_observation.copy(), next_action_mask=next_mask.copy(),
                terminated=terminated,
            ))
            if loss is not None:
                losses.append(loss)
            episode_reward += float(reward)
            observation, action_mask = next_observation, next_mask
        history.append({
            "episode": episode,
            "seed": seed + episode,
            "reward": episode_reward,
            "epsilon": agent.epsilon(),
            "mean_loss": sum(losses) / len(losses) if losses else None,
            "decisions": len(env.decisions),
        })
    return history


def evaluate_agent(env, agent: DoubleDQNAgent, episodes: int, *, seed: int) -> list[dict[str, Any]]:
    results = []
    for episode in range(episodes):
        observation, info = env.reset(seed=seed + episode)
        mask = np.asarray(info["action_mask"], dtype=np.int8)
        total_reward = 0.0
        terminated = False
        while not terminated:
            action = agent.select_action(observation, mask, epsilon=0.0)
            observation, reward, terminated, truncated, info = env.step(action)
            if truncated:
                raise RuntimeError("SchedulingEnv unexpectedly truncated an episode")
            mask = np.asarray(info["action_mask"], dtype=np.int8)
            total_reward += float(reward)
        results.append({
            "episode": episode,
            "seed": seed + episode,
            "reward": total_reward,
            "episode_objectives": info.get("episode_objectives", {}),
            "decisions": env.decisions,
        })
    return results
