"""Provider-neutral dynamic Function Calling agent loop."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass, field
from math import isfinite
from time import monotonic
from typing import Any, Protocol
from uuid import uuid4

from edge_agent_workflow_scheduling.common import (
    AgentRun,
    AgentRunStatus,
    CallStatus,
    LLMCall,
    ToolCall,
)
from edge_agent_workflow_scheduling.tools import (
    ToolExecution,
    ToolRegistry,
    ToolSpec,
    build_function_call_output,
)


@dataclass(frozen=True, slots=True)
class LLMResponse:
    """Normalized Responses-style output returned by an LLM backend."""

    output_items: list[dict[str, Any]]
    output_text: str = ""
    response_id: str | None = None
    model: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.output_text, str):
            raise ValueError("output_text must be a string")
        if self.response_id is not None:
            _validate_non_empty(self.response_id, "response_id")
        if self.model is not None:
            _validate_non_empty(self.model, "model")
        _validate_json_object_list(self.output_items, "output_items")
        _validate_json_object(self.metadata, "metadata")
        object.__setattr__(self, "output_items", deepcopy(self.output_items))
        object.__setattr__(self, "metadata", deepcopy(self.metadata))


class LLMBackend(Protocol):
    """Backend that turns one normalized LLMCall into Responses-style output."""

    def create_response(
        self,
        llm_call: LLMCall,
        *,
        system_instruction: str,
        tools: list[ToolSpec],
        timeout_sec: float,
    ) -> LLMResponse:
        """Create one model response."""


class ToolRunner(Protocol):
    """Injectable execution boundary for one schedulable ToolCall."""

    def run(self, tool_call: ToolCall, *, timeout_sec: float) -> ToolExecution:
        """Execute one ToolCall and return its provider-neutral result."""


@dataclass(frozen=True, slots=True)
class RegistryToolRunner:
    """Run Tool calls locally through a ToolRegistry."""

    registry: ToolRegistry

    def run(self, tool_call: ToolCall, *, timeout_sec: float) -> ToolExecution:
        del timeout_sec
        return self.registry.execute(
            tool_call.tool_name,
            tool_call.arguments,
            invocation_id=tool_call.tool_call_id,
        )


@dataclass(frozen=True, slots=True)
class AgentExecution:
    """AgentRun plus the dynamic calls and backend results that produced it."""

    agent_run: AgentRun
    llm_calls: tuple[LLMCall, ...]
    tool_calls: tuple[ToolCall, ...]
    llm_responses: tuple[LLMResponse, ...]
    tool_executions: tuple[ToolExecution, ...]


@dataclass(slots=True)
class FunctionCallingAgent:
    """Run a bounded LLM -> Tool -> LLM loop over normalized conversation state."""

    agent_id: str
    system_instruction: str
    tool_registry: ToolRegistry
    llm_backend: LLMBackend
    tool_runner: ToolRunner | None = None
    max_rounds: int = 8
    max_tool_calls: int = 16
    timeout_sec: float = 120.0

    def __post_init__(self) -> None:
        _validate_non_empty(self.agent_id, "agent_id")
        _validate_non_empty(self.system_instruction, "system_instruction")
        if (
            isinstance(self.max_rounds, bool)
            or not isinstance(self.max_rounds, int)
            or self.max_rounds < 1
        ):
            raise ValueError("max_rounds must be a positive integer")
        if (
            isinstance(self.max_tool_calls, bool)
            or not isinstance(self.max_tool_calls, int)
            or self.max_tool_calls < 0
        ):
            raise ValueError("max_tool_calls must be a non-negative integer")
        if (
            isinstance(self.timeout_sec, bool)
            or not isinstance(self.timeout_sec, int | float)
            or not isfinite(self.timeout_sec)
            or self.timeout_sec <= 0
        ):
            raise ValueError("timeout_sec must be finite and positive")
        if self.tool_runner is None:
            self.tool_runner = RegistryToolRunner(self.tool_registry)

    def run(
        self,
        user_task: str,
        *,
        task_id: str,
        run_id: str | None = None,
    ) -> AgentExecution:
        """Execute one bounded AgentRun and always return its terminal state."""

        _validate_non_empty(user_task, "user_task")
        if run_id is not None:
            _validate_non_empty(run_id, "run_id")
        resolved_run_id = run_id if run_id is not None else f"run_{uuid4().hex}"
        agent_run = AgentRun(
            run_id=resolved_run_id,
            agent_id=self.agent_id,
            task_id=task_id,
            conversation_items=[
                {"content": self.system_instruction, "role": "system"},
                {"content": user_task, "role": "user"},
            ],
        )
        agent_run.transition_to(AgentRunStatus.READY_FOR_LLM)

        llm_calls: list[LLMCall] = []
        tool_calls: list[ToolCall] = []
        llm_responses: list[LLMResponse] = []
        tool_executions: list[ToolExecution] = []
        seen_function_call_ids: set[str] = set()
        deadline = monotonic() + self.timeout_sec
        tool_runner = self.tool_runner
        if tool_runner is None:
            raise RuntimeError("tool_runner was not initialized")

        def finish() -> AgentExecution:
            return AgentExecution(
                agent_run=agent_run,
                llm_calls=tuple(llm_calls),
                tool_calls=tuple(tool_calls),
                llm_responses=tuple(llm_responses),
                tool_executions=tuple(tool_executions),
            )

        while True:
            remaining = deadline - monotonic()
            if remaining <= 0:
                _fail(agent_run, "timeout", "AgentRun exceeded timeout_sec")
                return finish()
            if agent_run.turn_index >= self.max_rounds:
                _fail(agent_run, "max_rounds_exceeded", "AgentRun reached max_rounds")
                return finish()

            llm_call = LLMCall(
                llm_call_id=f"{resolved_run_id}-llm-{agent_run.turn_index:04d}",
                run_id=resolved_run_id,
                agent_id=self.agent_id,
                turn_index=agent_run.turn_index,
                input_items=deepcopy(agent_run.conversation_items),
                required_capabilities=["function_calling"],
            )
            llm_calls.append(llm_call)
            llm_call.transition_to(CallStatus.QUEUED)
            llm_call.transition_to(CallStatus.RUNNING)
            agent_run.transition_to(AgentRunStatus.WAITING_FOR_LLM)

            try:
                response = self.llm_backend.create_response(
                    llm_call,
                    system_instruction=self.system_instruction,
                    tools=self.tool_registry.tools(),
                    timeout_sec=remaining,
                )
                if not isinstance(response, LLMResponse):
                    raise TypeError("LLMBackend.create_response() must return LLMResponse")
            except TimeoutError as exc:
                llm_call.transition_to(CallStatus.FAILED)
                _fail(agent_run, "timeout", str(exc) or "LLM backend timed out")
                return finish()
            except Exception as exc:
                llm_call.transition_to(CallStatus.FAILED)
                _fail(agent_run, "llm_error", str(exc) or exc.__class__.__name__)
                return finish()

            if monotonic() >= deadline:
                llm_call.transition_to(CallStatus.FAILED)
                _fail(agent_run, "timeout", "LLM backend exceeded AgentRun timeout")
                return finish()

            llm_call.transition_to(CallStatus.SUCCEEDED)
            llm_responses.append(response)
            agent_run.conversation_items.extend(deepcopy(response.output_items))
            agent_run.turn_index += 1
            function_items = [
                item for item in response.output_items if item.get("type") == "function_call"
            ]

            if not function_items:
                final_output = response.output_text.strip() or extract_output_text(
                    response.output_items
                )
                if not final_output:
                    _fail(
                        agent_run,
                        "invalid_llm_output",
                        "LLM response contained neither function_call nor final text",
                    )
                    return finish()
                agent_run.final_output = final_output
                agent_run.transition_to(AgentRunStatus.COMPLETED)
                return finish()

            agent_run.transition_to(AgentRunStatus.WAITING_FOR_TOOLS)
            if len(tool_calls) + len(function_items) > self.max_tool_calls:
                _fail(
                    agent_run,
                    "max_tool_calls_exceeded",
                    "AgentRun would exceed max_tool_calls",
                )
                return finish()

            round_tool_calls: list[ToolCall] = []
            for item in function_items:
                try:
                    tool_call = tool_call_from_response_item(
                        item,
                        run_id=resolved_run_id,
                        agent_id=self.agent_id,
                        turn_index=agent_run.turn_index - 1,
                        sequence_id=len(tool_calls) + len(round_tool_calls),
                    )
                except (TypeError, ValueError) as exc:
                    _fail(agent_run, "invalid_llm_output", str(exc))
                    return finish()
                if tool_call.call_id in seen_function_call_ids:
                    _fail(
                        agent_run,
                        "duplicate_call_id",
                        f"duplicate function call_id {tool_call.call_id!r}",
                    )
                    return finish()
                seen_function_call_ids.add(tool_call.call_id)
                round_tool_calls.append(tool_call)

            tool_calls.extend(round_tool_calls)
            round_executions: list[ToolExecution] = []
            timed_out = False
            for tool_call in round_tool_calls:
                tool_call.transition_to(CallStatus.QUEUED)
                tool_call.transition_to(CallStatus.RUNNING)
                remaining = deadline - monotonic()
                if remaining <= 0:
                    execution = _failed_execution(
                        "timeout",
                        "ToolCall was not started before AgentRun timeout",
                    )
                    timed_out = True
                else:
                    try:
                        execution = tool_runner.run(tool_call, timeout_sec=remaining)
                        if not isinstance(execution, ToolExecution):
                            raise TypeError("ToolRunner.run() must return ToolExecution")
                    except TimeoutError as exc:
                        execution = _failed_execution(
                            "timeout",
                            str(exc) or "Tool runner timed out",
                        )
                        timed_out = True
                    except Exception as exc:
                        execution = _failed_execution(
                            "tool_runner_error",
                            str(exc) or exc.__class__.__name__,
                        )
                    if monotonic() >= deadline:
                        execution = _failed_execution(
                            "timeout",
                            "Tool runner exceeded AgentRun timeout",
                        )
                        timed_out = True

                tool_call.transition_to(
                    CallStatus.SUCCEEDED if execution.success else CallStatus.FAILED
                )
                round_executions.append(execution)

            tool_executions.extend(round_executions)
            agent_run.conversation_items.extend(
                build_function_call_output(tool_call.call_id, execution)
                for tool_call, execution in zip(
                    round_tool_calls,
                    round_executions,
                    strict=True,
                )
            )

            if timed_out:
                _fail(agent_run, "timeout", "AgentRun timed out while executing Tool calls")
                return finish()
            failed_pair = next(
                (
                    (tool_call, execution)
                    for tool_call, execution in zip(
                        round_tool_calls,
                        round_executions,
                        strict=True,
                    )
                    if not execution.success
                ),
                None,
            )
            if failed_pair is not None:
                failed_call, failed_execution = failed_pair
                _fail(
                    agent_run,
                    "tool_error",
                    (f"Tool {failed_call.tool_name!r} failed: {failed_execution.error_message}"),
                )
                return finish()

            agent_run.transition_to(AgentRunStatus.READY_FOR_LLM)


def tool_call_from_response_item(
    item: dict[str, Any],
    *,
    run_id: str,
    agent_id: str,
    turn_index: int,
    sequence_id: int,
) -> ToolCall:
    call_id = item.get("call_id")
    tool_name = item.get("name")
    raw_arguments = item.get("arguments")
    if not isinstance(call_id, str) or not call_id.strip():
        raise ValueError("function_call.call_id must be a non-empty string")
    if not isinstance(tool_name, str) or not tool_name.strip():
        raise ValueError("function_call.name must be a non-empty string")
    if isinstance(raw_arguments, str):
        arguments = json.loads(raw_arguments)
    elif isinstance(raw_arguments, dict):
        arguments = deepcopy(raw_arguments)
    else:
        raise TypeError("function_call.arguments must be a JSON string or object")
    if not isinstance(arguments, dict):
        raise ValueError("function_call.arguments must contain a JSON object")
    return ToolCall(
        tool_call_id=f"{run_id}-tool-{sequence_id:04d}",
        run_id=run_id,
        agent_id=agent_id,
        call_id=call_id,
        tool_name=tool_name,
        arguments=arguments,
        turn_index=turn_index,
    )


def extract_output_text(items: list[dict[str, Any]]) -> str:
    texts: list[str] = []
    for item in items:
        if item.get("type") != "message":
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") == "output_text":
                text = part.get("text")
                if isinstance(text, str):
                    texts.append(text)
    return "\n".join(texts).strip()


def _failed_execution(code: str, message: str) -> ToolExecution:
    return ToolExecution(
        success=False,
        error_code=code,
        error_message=message,
    )


def _fail(agent_run: AgentRun, code: str, message: str) -> None:
    agent_run.error_code = code
    agent_run.error_message = message
    agent_run.transition_to(AgentRunStatus.FAILED)


def _validate_non_empty(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _validate_json_object(value: Any, field_name: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a JSON object")
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be JSON serializable") from exc


def _validate_json_object_list(value: Any, field_name: str) -> None:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list")
    for index, item in enumerate(value):
        _validate_json_object(item, f"{field_name}[{index}]")
