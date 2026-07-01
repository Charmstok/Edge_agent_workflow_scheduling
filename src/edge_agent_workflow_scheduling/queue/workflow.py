"""In-memory workflow queue for mixed LLM and tool steps."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from edge_agent_workflow_scheduling.common import LLMCall, ToolCall
from edge_agent_workflow_scheduling.queue.policy import (
    DEFAULT_QUEUE_POLICY_FACTORY,
    QueueOrdering,
    QueuePolicyFactory,
)

WorkflowQueueItem = LLMCall | ToolCall


@dataclass(frozen=True, slots=True)
class _QueuedWorkflowItem:
    sequence_id: int
    step: WorkflowQueueItem


class InMemoryWorkflowQueue:
    """A small in-memory queue for the first local scheduling prototype.

    FIFO ordering returns steps in insertion order. Priority ordering returns
    the largest numeric ``priority`` first and preserves FIFO order for ties.
    """

    def __init__(
        self,
        initial_steps: Iterable[WorkflowQueueItem] | None = None,
        *,
        ordering: QueueOrdering = "fifo",
        policy_factory: QueuePolicyFactory | None = None,
    ) -> None:
        self._policy_factory = policy_factory or DEFAULT_QUEUE_POLICY_FACTORY
        self._policy = self._policy_factory.create(ordering)
        self._items: list[_QueuedWorkflowItem] = []
        self._next_sequence_id = 0

        if initial_steps is not None:
            self.push_many(initial_steps)

    @property
    def ordering(self) -> QueueOrdering:
        """Default ordering used by ``pop`` and ``peek``."""

        return self._policy.name

    def push(self, step: WorkflowQueueItem) -> None:
        """Append one workflow step to the queue."""

        _validate_step(step)
        self._items.append(_QueuedWorkflowItem(sequence_id=self._next_sequence_id, step=step))
        self._next_sequence_id += 1

    def push_many(self, steps: Iterable[WorkflowQueueItem]) -> None:
        """Append multiple workflow steps in iteration order."""

        for step in steps:
            self.push(step)

    def pop(self, *, ordering: QueueOrdering | None = None) -> WorkflowQueueItem | None:
        """Remove and return the next workflow step, or ``None`` if empty."""

        if not self._items:
            return None

        selected_index = self._select_index(ordering)
        return self._items.pop(selected_index).step

    def peek(self, *, ordering: QueueOrdering | None = None) -> WorkflowQueueItem | None:
        """Return the next workflow step without removing it."""

        if not self._items:
            return None

        selected_index = self._select_index(ordering)
        return self._items[selected_index].step

    def size(self) -> int:
        """Return the number of queued workflow steps."""

        return len(self._items)

    def is_empty(self) -> bool:
        """Return whether the queue has no pending workflow steps."""

        return not self._items

    def clear(self) -> None:
        """Remove all queued workflow steps."""

        self._items.clear()

    def _select_index(self, ordering: QueueOrdering | None) -> int:
        policy = self._policy if ordering is None else self._policy_factory.create(ordering)
        return policy.select_index(self._items)


def _validate_step(step: WorkflowQueueItem) -> None:
    if not isinstance(step, LLMCall | ToolCall):
        msg = "step must be an LLMCall or ToolCall"
        raise TypeError(msg)
