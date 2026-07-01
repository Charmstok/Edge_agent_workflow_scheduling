"""Queue ordering policies for workflow steps."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

QueueOrdering = str


class _PrioritizedStep(Protocol):
    priority: int


class _QueuedItem(Protocol):
    sequence_id: int
    step: _PrioritizedStep


class QueuePolicy(Protocol):
    """Policy interface for selecting the next workflow queue item."""

    name: str

    def select_index(self, items: Sequence[_QueuedItem]) -> int:
        """Select the next item index from non-empty queued items."""


# FIFO 策略
# 结构完全匹配 QueuePolicy，自动被视为 QueuePolicy 类型
class FifoQueuePolicy():
    """Select items by insertion order."""

    name = "fifo"

    def select_index(self, items: Sequence[_QueuedItem]) -> int:
        _require_non_empty(items)
        return 0


# 优先级策略
class PriorityQueuePolicy():
    """Select the highest-priority item, preserving FIFO order for ties."""

    name = "priority"

    def select_index(self, items: Sequence[_QueuedItem]) -> int:
        _require_non_empty(items)
        return max(
            range(len(items)),
            key=lambda index: (
                items[index].step.priority,
                -items[index].sequence_id,
            ),
        )


@dataclass(slots=True)
class QueuePolicyFactory:
    """Registry-backed factory for queue ordering policies."""

    _policies: dict[str, QueuePolicy] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self._policies:
            self.register(FifoQueuePolicy())
            self.register(PriorityQueuePolicy())

    def register(self, policy: QueuePolicy) -> None:
        if not policy.name:
            msg = "queue policy name must be non-empty"
            raise ValueError(msg)
        self._policies[policy.name] = policy

    def create(self, ordering: QueueOrdering) -> QueuePolicy:
        try:
            return self._policies[ordering]
        except KeyError as exc:
            msg = f"ordering must be one of {self.available_orderings()}"
            raise ValueError(msg) from exc

    def available_orderings(self) -> list[str]:
        return sorted(self._policies)


DEFAULT_QUEUE_POLICY_FACTORY = QueuePolicyFactory()


def create_queue_policy(ordering: QueueOrdering) -> QueuePolicy:
    """Create a queue policy from the default policy registry."""

    return DEFAULT_QUEUE_POLICY_FACTORY.create(ordering)


def _require_non_empty(items: Sequence[_QueuedItem]) -> None:
    if not items:
        msg = "cannot select an item from an empty queue"
        raise IndexError(msg)
