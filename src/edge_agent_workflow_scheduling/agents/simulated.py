"""Simulated agents for generating deterministic LLMCall and ToolCall workloads."""

from __future__ import annotations

from dataclasses import dataclass, field

from edge_agent_workflow_scheduling.common import LLMCall, SchedulableCall, ToolCall


@dataclass(frozen=True, slots=True)
class ToolCallTemplate:
    """Template used by a simulated agent to generate tool calls."""

    tool_name: str
    input_uri_prefix: str
    input_size_mb: float
    page_count: int = 0
    image_count: int = 0
    deadline_sec: float | None = None
    priority: int = 0
    file_extension: str = "dat"


@dataclass(frozen=True, slots=True)
class LLMCallTemplate:
    """Template used by a simulated agent to generate LLM inference calls."""

    model_name: str | None = None
    input_tokens: int = 0
    estimated_output_tokens: int = 0
    context_length: int = 0
    deadline_sec: float | None = None
    priority: int = 0


@dataclass(slots=True)
class SimulatedAgent:
    """A lightweight agent that generates deterministic call workloads.

    This is not a full LLM-backed agent. It emits preconfigured LLMCall and
    ToolCall objects so the queue, scheduler, worker/runtime, and profiler can
    be developed before the real agent loop is introduced.
    """

    agent_id: str
    template: ToolCallTemplate
    llm_template: LLMCallTemplate = field(default_factory=LLMCallTemplate)

    def generate_tool_call(self, sequence_id: int) -> ToolCall:
        if sequence_id < 0:
            msg = "sequence_id must be non-negative"
            raise ValueError(msg)

        return ToolCall(
            tool_call_id=f"{self.agent_id}-tc-{sequence_id:04d}",
            run_id=self._build_run_id(sequence_id),
            agent_id=self.agent_id,
            call_id=f"call_{self.agent_id}_{sequence_id:04d}",
            tool_name=self.template.tool_name,
            arguments={"input_uri": self._build_input_uri(sequence_id)},
            metadata={
                "input_size_mb": self.template.input_size_mb,
                "page_count": self.template.page_count,
                "image_count": self.template.image_count,
            },
            deadline_sec=self.template.deadline_sec,
            priority=self.template.priority,
        )

    def generate_tool_calls(self, count: int, *, start_sequence_id: int = 0) -> list[ToolCall]:
        if count < 0:
            msg = "count must be non-negative"
            raise ValueError(msg)
        if start_sequence_id < 0:
            msg = "start_sequence_id must be non-negative"
            raise ValueError(msg)

        return [
            self.generate_tool_call(sequence_id)
            for sequence_id in range(start_sequence_id, start_sequence_id + count)
        ]

    def generate_llm_call(self, sequence_id: int) -> LLMCall:
        if sequence_id < 0:
            msg = "sequence_id must be non-negative"
            raise ValueError(msg)

        return LLMCall(
            llm_call_id=f"{self.agent_id}-lc-{sequence_id:04d}",
            run_id=self._build_run_id(sequence_id),
            agent_id=self.agent_id,
            input_items=[
                {
                    "role": "user",
                    "content": f"Simulated request {self.agent_id}/{sequence_id}",
                }
            ],
            input_tokens=self.llm_template.input_tokens,
            estimated_output_tokens=self.llm_template.estimated_output_tokens,
            context_length=self.llm_template.context_length,
            model_name=self.llm_template.model_name,
            deadline_sec=self.llm_template.deadline_sec,
            priority=self.llm_template.priority,
        )

    def generate_llm_calls(self, count: int, *, start_sequence_id: int = 0) -> list[LLMCall]:
        if count < 0:
            msg = "count must be non-negative"
            raise ValueError(msg)
        if start_sequence_id < 0:
            msg = "start_sequence_id must be non-negative"
            raise ValueError(msg)

        return [
            self.generate_llm_call(sequence_id)
            for sequence_id in range(start_sequence_id, start_sequence_id + count)
        ]

    def generate_calls(
        self,
        count: int,
        *,
        start_sequence_id: int = 0,
    ) -> list[SchedulableCall]:
        """Generate paired LLM and Tool calls for a deterministic workload."""

        if count < 0:
            msg = "count must be non-negative"
            raise ValueError(msg)
        if start_sequence_id < 0:
            msg = "start_sequence_id must be non-negative"
            raise ValueError(msg)

        calls: list[SchedulableCall] = []
        for sequence_id in range(start_sequence_id, start_sequence_id + count):
            calls.append(self.generate_llm_call(sequence_id))
            calls.append(self.generate_tool_call(sequence_id))
        return calls

    def _build_input_uri(self, sequence_id: int) -> str:
        filename = f"{self.agent_id}_{sequence_id:04d}.{self.template.file_extension}"
        return f"{self.template.input_uri_prefix.rstrip('/')}/{filename}"

    def _build_run_id(self, sequence_id: int) -> str:
        return f"{self.agent_id}-run-{sequence_id:04d}"
