"""In-memory queue for mixed LLM and Tool steps."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from edge_agent_workflow_scheduling.common import LLMCall, ToolCall

QueueOrdering = Literal["fifo", "priority"]
WorkflowQueueItem = LLMCall | ToolCall


@dataclass(frozen=True, slots=True)
class _QueuedWorkflowItem:
    sequence_id: int
    step: WorkflowQueueItem


class InMemoryWorkflowQueue:
    """FIFO or priority queue used by the local research prototype."""

    def __init__(
        self,
        initial_steps: Iterable[WorkflowQueueItem] | None = None,
        *,
        ordering: QueueOrdering = "fifo",
    ) -> None:
        if ordering not in {"fifo", "priority"}:
            raise ValueError("ordering must be 'fifo' or 'priority'")
        self._ordering = ordering
        self._items: list[_QueuedWorkflowItem] = []
        self._next_sequence_id = 0
        if initial_steps is not None:
            self.push_many(initial_steps)

    @property
    def ordering(self) -> QueueOrdering:
        return self._ordering

    def push(self, step: WorkflowQueueItem) -> None:
        if not isinstance(step, LLMCall | ToolCall):
            raise TypeError("step must be an LLMCall or ToolCall")
        self._items.append(_QueuedWorkflowItem(self._next_sequence_id, step))
        self._next_sequence_id += 1

    def push_many(self, steps: Iterable[WorkflowQueueItem]) -> None:
        for step in steps:
            self.push(step)

    def pop(self, *, ordering: QueueOrdering | None = None) -> WorkflowQueueItem | None:
        if not self._items:
            return None
        return self._items.pop(self._select_index(ordering)).step

    def peek(self, *, ordering: QueueOrdering | None = None) -> WorkflowQueueItem | None:
        if not self._items:
            return None
        return self._items[self._select_index(ordering)].step

    def size(self) -> int:
        return len(self._items)

    def is_empty(self) -> bool:
        return not self._items

    def clear(self) -> None:
        self._items.clear()

    def _select_index(self, ordering: QueueOrdering | None) -> int:
        selected_ordering = ordering or self._ordering
        if selected_ordering == "fifo":
            return 0
        if selected_ordering == "priority":
            return max(
                range(len(self._items)),
                key=lambda index: (
                    self._items[index].step.priority,
                    -self._items[index].sequence_id,
                ),
            )
        raise ValueError("ordering must be 'fifo' or 'priority'")
