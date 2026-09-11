"""Chat Completions adapter for vLLM and Ark; conversation stays provider neutral."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

from edge_agent_workflow_scheduling.common import LLMCall, LLMResult
from edge_agent_workflow_scheduling.executors.base import llm_call_error, validate_timeout
from edge_agent_workflow_scheduling.resources import LLMInstanceProfile
from edge_agent_workflow_scheduling.tools import ToolSpec


def chat_messages(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for item in items:
        kind = item.get("type")
        if kind == "function_call":
            if not messages or messages[-1].get("role") != "assistant":
                messages.append({"role": "assistant", "content": None})
            messages[-1].setdefault("tool_calls", []).append(
                {
                    "id": item["call_id"],
                    "type": "function",
                    "function": {"name": item["name"], "arguments": item["arguments"]},
                }
            )
        elif kind == "function_call_output":
            messages.append(
                {"role": "tool", "tool_call_id": item["call_id"], "content": item["output"]}
            )
        elif item.get("role") in {"system", "developer", "user", "assistant"}:
            content = item.get("content", "")
            if isinstance(content, list):
                if any(
                    part.get("type") not in {"input_text", "output_text", "text"}
                    for part in content
                ):
                    raise ValueError("chat adapter currently supports text content only")
                content = "\n".join(part["text"] for part in content)
            messages.append({"role": item["role"], "content": content})
        elif kind == "reasoning":
            # Responses reasoning items are provider-specific, not portable chat messages.
            continue
        else:
            raise ValueError("unsupported conversation item")
    return messages


@dataclass(slots=True)
class OpenAIChatExecutor:
    profile: LLMInstanceProfile
    client: Any
    model_parameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        protected = {"messages", "model", "tools", "stream", "timeout", "n"}
        if protected & self.model_parameters.keys():
            raise ValueError("model_parameters overrides protocol fields")
        extra = self.model_parameters.get("extra_body", {})
        if not isinstance(extra, dict) or protected & extra.keys():
            raise ValueError("extra_body overrides protocol fields")

    def execute(
        self,
        llm_call: LLMCall,
        *,
        tools: list[ToolSpec] | None = None,
        timeout_sec: float | None = None,
    ) -> LLMResult:
        validate_timeout(timeout_sec)
        invalid = llm_call_error(self.profile, llm_call)
        if invalid:
            return LLMResult(
                llm_call_id=llm_call.llm_call_id,
                llm_id=self.profile.llm_id,
                success=False,
                error_code=invalid[0],
                error_message=invalid[1],
            )
        start = perf_counter()
        metadata: dict[str, Any] = {
            "executor_type": "openai_chat",
            "energy_status": "unavailable",
            "timing_scope": "client_request_including_network_and_server_wait",
        }
        try:
            options = deepcopy(self.model_parameters)
            if tools:
                options["tools"] = [
                    {"type": "function", "function": {k: v for k, v in tool.items() if k != "type"}}
                    for tool in tools
                ]
            response = self.client.chat.completions.create(
                model=self.profile.model,
                messages=chat_messages(llm_call.input_items),
                timeout=timeout_sec or 120.0,
                stream=False,
                **options,
            )
            raw = response.model_dump(mode="json", exclude_none=True)
            metadata["raw_response"] = raw
            metadata["usage"] = raw.get("usage")
            choice = raw["choices"][0]
            reason = choice.get("finish_reason")
            message = choice["message"]
            output = []
            content = message.get("content") or ""
            if content:
                output.append(
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": content, "annotations": []}],
                    }
                )
            for call in message.get("tool_calls", []):
                output.append(
                    {
                        "type": "function_call",
                        "call_id": call["id"],
                        "name": call["function"]["name"],
                        "arguments": call["function"]["arguments"],
                    }
                )
            success = reason in {"stop", "tool_calls"}
            elapsed = perf_counter() - start
            code = None if success else "incomplete_response"
            if timeout_sec is not None and elapsed > timeout_sec:
                success, code = False, "timeout"
            return LLMResult(
                llm_call_id=llm_call.llm_call_id,
                llm_id=self.profile.llm_id,
                success=success,
                output_items=output,
                output_text=content,
                response_id=raw.get("id"),
                response_model=raw.get("model"),
                output_tokens=(raw.get("usage") or {}).get("completion_tokens", 0),
                inference_time_sec=elapsed,
                metadata=metadata,
                error_code=code,
                error_message=None if success else f"request ended with {reason}",
            )
        except Exception as exc:
            # Provider exception bodies can echo Authorization headers. Persist only type.
            code = "timeout" if "timeout" in type(exc).__name__.lower() else "llm_execution_failed"
            return LLMResult(
                llm_call_id=llm_call.llm_call_id,
                llm_id=self.profile.llm_id,
                success=False,
                inference_time_sec=perf_counter() - start,
                error_code=code,
                error_message=type(exc).__name__,
                metadata=metadata,
            )


def create_openai_chat_executor(profile: LLMInstanceProfile) -> OpenAIChatExecutor:
    from edge_agent_workflow_scheduling.executors.openai import create_openai_client

    return OpenAIChatExecutor(
        profile,
        create_openai_client(profile),
        profile.deployment_config.get("model_parameters", {}),
    )
