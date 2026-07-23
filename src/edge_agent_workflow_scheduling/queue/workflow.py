"""In-memory queue for mixed LLM and Tool calls."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from edge_agent_workflow_scheduling.common import (
    CallStatus,
    LLMCall,
    SchedulableCall,
    ToolCall,
)

QueueOrdering = Literal["fifo", "priority"]
CallQueueItem = SchedulableCall


@dataclass(frozen=True, slots=True)
class _QueuedCall:
    sequence_id: int
    call: CallQueueItem


class InMemoryCallQueue:
    """FIFO or priority queue used by the local research prototype."""

    def __init__(
        self,
        initial_calls: Iterable[CallQueueItem] | None = None,
        *,
        ordering: QueueOrdering = "fifo",
    ) -> None:
        if ordering not in {"fifo", "priority"}:
            raise ValueError("ordering must be 'fifo' or 'priority'")
        self._ordering = ordering
        self._items: list[_QueuedCall] = []
        self._next_sequence_id = 0
        if initial_calls is not None:
            self.push_many(initial_calls)

    @property
    def ordering(self) -> QueueOrdering:
        return self._ordering

    def push(self, call: CallQueueItem) -> None:
        if not isinstance(call, LLMCall | ToolCall):
            raise TypeError("call must be an LLMCall or ToolCall")
        call.transition_to(CallStatus.QUEUED)
        self._items.append(_QueuedCall(self._next_sequence_id, call))
        self._next_sequence_id += 1

    def push_many(self, calls: Iterable[CallQueueItem]) -> None:
        for call in calls:
            self.push(call)

    def pop(self, *, ordering: QueueOrdering | None = None) -> CallQueueItem | None:
        if not self._items:
            return None
        return self._items.pop(self._select_index(ordering)).call

    def peek(self, *, ordering: QueueOrdering | None = None) -> CallQueueItem | None:
        if not self._items:
            return None
        return self._items[self._select_index(ordering)].call

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
                    self._items[index].call.priority,
                    -self._items[index].sequence_id,
                ),
            )
        raise ValueError("ordering must be 'fifo' or 'priority'")
