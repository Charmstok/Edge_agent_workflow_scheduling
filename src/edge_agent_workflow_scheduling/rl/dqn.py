"""Small NumPy Double-DQN implementation for the local scheduling prototype."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class Transition:
    observation: np.ndarray
    action: int
    reward: float
    next_observation: np.ndarray
    next_action_mask: np.ndarray
    terminated: bool


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int = 0) -> None:
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._items: list[Transition] = []
        self._rng = random.Random(seed)

    def add(self, transition: Transition) -> None:
        if len(self._items) >= self.capacity:
            self._items.pop(0)
        self._items.append(transition)

    def sample(self, batch_size: int) -> list[Transition]:
        if len(self._items) < batch_size:
            raise ValueError("not enough transitions in replay buffer")
        return self._rng.sample(self._items, batch_size)

    def __len__(self) -> int:
        return len(self._items)


class MLPQNetwork:
    """Two-layer Q network with deterministic NumPy SGD/Adam-like updates."""

    def __init__(
        self, observation_size: int, action_size: int, *, hidden_size: int = 64, seed: int = 0
    ) -> None:
        if observation_size < 1 or action_size < 1:
            raise ValueError("network dimensions must be positive")
        rng = np.random.default_rng(seed)
        scale = np.sqrt(2.0 / observation_size)
        self.weights = {
            "w1": (rng.normal(0.0, scale, (observation_size, hidden_size))).astype(np.float32),
            "b1": np.zeros(hidden_size, dtype=np.float32),
            "w2": (rng.normal(0.0, np.sqrt(2.0 / hidden_size), (hidden_size, action_size))).astype(
                np.float32
            ),
            "b2": np.zeros(action_size, dtype=np.float32),
        }

    def copy(self) -> MLPQNetwork:
        other = object.__new__(MLPQNetwork)
        other.weights = {key: value.copy() for key, value in self.weights.items()}
        return other

    def predict(self, observations: np.ndarray) -> np.ndarray:
        hidden = np.maximum(observations @ self.weights["w1"] + self.weights["b1"], 0.0)
        return hidden @ self.weights["w2"] + self.weights["b2"]

    def train_batch(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        targets: np.ndarray,
        learning_rate: float,
        gradient_clip: float,
    ) -> float:
        hidden_pre = observations @ self.weights["w1"] + self.weights["b1"]
        hidden = np.maximum(hidden_pre, 0.0)
        predictions = hidden @ self.weights["w2"] + self.weights["b2"]
        indices = np.arange(len(actions))
        errors = predictions[indices, actions] - targets
        loss = float(np.mean(errors**2))
        grad_out = np.zeros_like(predictions)
        grad_out[indices, actions] = (2.0 / len(actions)) * errors
        grad_w2 = hidden.T @ grad_out
        grad_b2 = grad_out.sum(axis=0)
        grad_hidden = (grad_out @ self.weights["w2"].T) * (hidden_pre > 0)
        grad_w1 = observations.T @ grad_hidden
        grad_b1 = grad_hidden.sum(axis=0)
        for name, gradient in (("w1", grad_w1), ("b1", grad_b1), ("w2", grad_w2), ("b2", grad_b2)):
            np.clip(gradient, -gradient_clip, gradient_clip, out=gradient)
            self.weights[name] -= learning_rate * gradient.astype(np.float32)
        return loss

    def to_dict(self) -> dict[str, Any]:
        return {key: value.tolist() for key, value in self.weights.items()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MLPQNetwork:
        other = object.__new__(cls)
        other.weights = {key: np.asarray(value, dtype=np.float32) for key, value in data.items()}
        return other


@dataclass(frozen=True, slots=True)
class DQNConfig:
    episodes: int = 100
    gamma: float = 0.99
    learning_rate: float = 0.001
    hidden_size: int = 64
    replay_capacity: int = 10000
    batch_size: int = 32
    warmup_steps: int = 32
    target_update_interval: int = 20
    train_interval: int = 1
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 5000
    gradient_clip: float = 1.0

    def __post_init__(self) -> None:
        if self.episodes < 1 or self.batch_size < 1 or self.replay_capacity < self.batch_size:
            raise ValueError("invalid DQN episode or replay settings")
        if not 0.0 <= self.epsilon_end <= self.epsilon_start <= 1.0:
            raise ValueError("epsilon values must be ordered in [0, 1]")


class DoubleDQNAgent:
    def __init__(
        self,
        observation_size: int,
        action_size: int,
        *,
        config: DQNConfig | None = None,
        seed: int = 0,
    ) -> None:
        self.config = config or DQNConfig()
        self.seed = seed
        self.rng = random.Random(seed)
        self.online = MLPQNetwork(
            observation_size, action_size, hidden_size=self.config.hidden_size, seed=seed
        )
        self.target = self.online.copy()
        self.buffer = ReplayBuffer(self.config.replay_capacity, seed=seed)
        self.total_steps = 0
        self.losses: list[float] = []

    def select_action(
        self, observation: np.ndarray, action_mask: np.ndarray, *, epsilon: float = 0.0
    ) -> int:
        valid = np.flatnonzero(action_mask.astype(bool))
        if not len(valid):
            raise ValueError("cannot select an action with an empty action mask")
        if self.rng.random() < epsilon:
            return int(self.rng.choice(valid.tolist()))
        q_values = self.online.predict(np.asarray(observation, dtype=np.float32)[None, :])[0]
        return int(valid[np.argmax(q_values[valid])])

    def epsilon(self) -> float:
        progress = min(self.total_steps / max(self.config.epsilon_decay_steps, 1), 1.0)
        return self.config.epsilon_start + progress * (
            self.config.epsilon_end - self.config.epsilon_start
        )

    def observe(self, transition: Transition) -> float | None:
        self.buffer.add(transition)
        self.total_steps += 1
        if self.total_steps % self.config.train_interval or len(self.buffer) < max(
            self.config.batch_size, self.config.warmup_steps
        ):
            return None
        batch = self.buffer.sample(self.config.batch_size)
        observations = np.stack([item.observation for item in batch]).astype(np.float32)
        next_observations = np.stack([item.next_observation for item in batch]).astype(np.float32)
        actions = np.asarray([item.action for item in batch], dtype=np.int64)
        rewards = np.asarray([item.reward for item in batch], dtype=np.float32)
        nonterminal = np.asarray([not item.terminated for item in batch])
        next_online = self.online.predict(next_observations)
        next_target = self.target.predict(next_observations)
        next_values = np.zeros(len(batch), dtype=np.float32)
        for index, item in enumerate(batch):
            valid = np.flatnonzero(item.next_action_mask.astype(bool))
            if nonterminal[index] and len(valid):
                best = valid[np.argmax(next_online[index, valid])]
                next_values[index] = next_target[index, best]
        targets = rewards + self.config.gamma * next_values * nonterminal.astype(np.float32)
        loss = self.online.train_batch(
            observations, actions, targets, self.config.learning_rate, self.config.gradient_clip
        )
        self.losses.append(loss)
        if self.total_steps % self.config.target_update_interval == 0:
            self.target = self.online.copy()
        return loss

    def checkpoint(self, metadata: dict[str, Any]) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "algorithm": "double_dqn",
            "seed": self.seed,
            "config": asdict(self.config),
            "total_steps": self.total_steps,
            "metadata": metadata,
            "online": self.online.to_dict(),
            "target": self.target.to_dict(),
        }

    def save(self, path: str | Path, metadata: dict[str, Any]) -> None:
        Path(path).write_text(
            json.dumps(self.checkpoint(metadata), indent=2) + "\n", encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> tuple[DoubleDQNAgent, dict[str, Any]]:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1 or payload.get("algorithm") != "double_dqn":
            raise ValueError("unsupported Double DQN checkpoint")
        online = MLPQNetwork.from_dict(payload["online"])
        agent = cls(
            online.weights["w1"].shape[0],
            online.weights["w2"].shape[1],
            config=DQNConfig(**payload["config"]),
            seed=payload["seed"],
        )
        agent.online = online
        agent.target = MLPQNetwork.from_dict(payload["target"])
        agent.total_steps = payload["total_steps"]
        return agent, payload["metadata"]
