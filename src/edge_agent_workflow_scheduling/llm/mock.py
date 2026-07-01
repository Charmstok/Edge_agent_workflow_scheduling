"""Mock LLM runtime for local scheduling prototypes."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import sleep
from typing import Any

from edge_agent_workflow_scheduling.common import (
    LLMCall,
    LLMInstanceInfo,
    LLMInstanceState,
    LLMResult,
)


@dataclass(slots=True)
class MockLLMRuntime:
    """Deterministic in-process LLM runtime used before real model services exist."""

    llm_id: str
    model_name: str
    device_id: str
    tokens_per_sec: float
    max_concurrency: int = 1
    queue_wait_time_sec: float = 0.0
    fixed_inference_overhead_sec: float = 0.0
    default_output_tokens: int = 128
    output_uri_prefix: str = "local://outputs/llm"
    endpoint_url: str | None = None
    model_size_b: float | None = None
    accelerator: str | None = None
    sleep_scale: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.llm_id:
            msg = "llm_id must be non-empty"
            raise ValueError(msg)
        if not self.model_name:
            msg = "model_name must be non-empty"
            raise ValueError(msg)
        if not self.device_id:
            msg = "device_id must be non-empty"
            raise ValueError(msg)
        if self.tokens_per_sec <= 0:
            msg = "tokens_per_sec must be positive"
            raise ValueError(msg)
        if self.max_concurrency < 1:
            msg = "max_concurrency must be at least 1"
            raise ValueError(msg)
        if self.queue_wait_time_sec < 0:
            msg = "queue_wait_time_sec must be non-negative"
            raise ValueError(msg)
        if self.fixed_inference_overhead_sec < 0:
            msg = "fixed_inference_overhead_sec must be non-negative"
            raise ValueError(msg)
        if self.default_output_tokens < 0:
            msg = "default_output_tokens must be non-negative"
            raise ValueError(msg)
        if self.sleep_scale < 0:
            msg = "sleep_scale must be non-negative"
            raise ValueError(msg)

    def generate(self, llm_call: LLMCall) -> LLMResult:
        """Generate a deterministic mock result for an LLM call."""

        if llm_call.model_name is not None and llm_call.model_name != self.model_name:
            return LLMResult(
                llm_call_id=llm_call.llm_call_id,
                llm_id=self.llm_id,
                success=False,
                error_message=(
                    f"model_name {llm_call.model_name!r} does not match runtime model "
                    f"{self.model_name!r}"
                ),
            )

        output_tokens = self._estimate_output_tokens(llm_call)
        inference_time_sec = self._estimate_inference_time_sec(llm_call, output_tokens)
        if self.sleep_scale > 0:
            sleep((self.queue_wait_time_sec + inference_time_sec) * self.sleep_scale)

        return LLMResult(
            llm_call_id=llm_call.llm_call_id,
            llm_id=self.llm_id,
            success=True,
            output_uri=self._default_output_uri(llm_call),
            output_tokens=output_tokens,
            queue_wait_time_sec=self.queue_wait_time_sec,
            inference_time_sec=inference_time_sec,
        )

    def to_info(self) -> LLMInstanceInfo:
        """Return static runtime metadata."""

        return LLMInstanceInfo(
            llm_id=self.llm_id,
            model_name=self.model_name,
            endpoint_url=self.endpoint_url or f"mock://{self.llm_id}",
            device_id=self.device_id,
            model_size_b=self.model_size_b,
            accelerator=self.accelerator,
            max_concurrency=self.max_concurrency,
            metadata=dict(self.metadata),
        )

    def get_state(
        self,
        *,
        queue_len: int = 0,
        running_requests: int = 0,
        gpu_util: float = 0.0,
        memory_util: float = 0.0,
        avg_latency_sec: float = 0.0,
        is_online: bool = True,
    ) -> LLMInstanceState:
        """Return a lightweight runtime state snapshot."""

        return LLMInstanceState(
            llm_id=self.llm_id,
            model_name=self.model_name,
            device_id=self.device_id,
            queue_len=queue_len,
            running_requests=running_requests,
            gpu_util=gpu_util,
            memory_util=memory_util,
            tokens_per_sec=self.tokens_per_sec,
            avg_latency_sec=avg_latency_sec,
            is_online=is_online,
        )

    def _estimate_output_tokens(self, llm_call: LLMCall) -> int:
        if llm_call.estimated_output_tokens > 0:
            return llm_call.estimated_output_tokens
        return self.default_output_tokens

    def _estimate_inference_time_sec(self, llm_call: LLMCall, output_tokens: int) -> float:
        total_tokens = llm_call.input_tokens + output_tokens
        return self.fixed_inference_overhead_sec + total_tokens / self.tokens_per_sec

    def _default_output_uri(self, llm_call: LLMCall) -> str:
        prefix = self.output_uri_prefix.rstrip("/")
        return f"{prefix}/{self.llm_id}/{llm_call.llm_call_id}.json"
