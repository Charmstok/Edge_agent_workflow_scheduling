"""Scheduler-only replay of recorded Agent calls."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path

from edge_agent_workflow_scheduling.common import CallStatus, LLMCall, SchedulableCall, ToolCall
from edge_agent_workflow_scheduling.profiler.models import CallTrace, TraceBundle
from edge_agent_workflow_scheduling.profiler.privacy import content_digest
from edge_agent_workflow_scheduling.resources import (
    LLMInstanceProfile,
    LLMInstanceState,
    ResourceRegistry,
    ToolReplicaProfile,
    ToolReplicaState,
)
from edge_agent_workflow_scheduling.scheduler import BaselineScheduler


@dataclass(frozen=True, slots=True)
class ReplayDecision:
    """A policy decision for one immutable replay input."""

    sequence_id: int
    call_id: str
    call_kind: str
    call_digest: str
    selected_target: str
    policy_name: str


@dataclass(frozen=True, slots=True)
class ReplayResult:
    """Ordered decisions made by one policy over a recorded call stream."""

    source_run_id: str
    policy_name: str
    input_fingerprint: str
    decisions: tuple[ReplayDecision, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "source_run_id": self.source_run_id,
            "policy_name": self.policy_name,
            "input_fingerprint": self.input_fingerprint,
            "decisions": [asdict(decision) for decision in self.decisions],
        }


def load_trace_bundle(path: str | Path) -> TraceBundle:
    """Load a versioned trace bundle from JSON."""

    return TraceBundle.from_json(Path(path).read_text(encoding="utf-8"))


def replay_calls(
    trace: TraceBundle,
    *,
    scheduler: BaselineScheduler,
    resources: ResourceRegistry,
) -> ReplayResult:
    """Schedule recorded calls in order without invoking an LLM or executor."""

    decisions: list[ReplayDecision] = []
    call_digests: list[str] = []
    for call_trace in trace.calls:
        call = reconstruct_call(call_trace)
        decision = scheduler.schedule(call, resources=resources)
        if decision.call_id != call_trace.call_id or decision.call_kind != call_trace.call_kind:
            raise ValueError("scheduler decision identity does not match replay call")
        decisions.append(
            ReplayDecision(
                sequence_id=call_trace.sequence_id,
                call_id=call_trace.call_id,
                call_kind=call_trace.call_kind,
                call_digest=call_trace.call_digest,
                selected_target=decision.selected_target,
                policy_name=decision.policy_name,
            )
        )
        call_digests.append(call_trace.call_digest)

    policy_name = decisions[0].policy_name if decisions else scheduler.policy_name
    return ReplayResult(
        source_run_id=trace.run.run_id,
        policy_name=policy_name,
        input_fingerprint=content_digest(call_digests),
        decisions=tuple(decisions),
    )


def resources_from_manifest(trace: TraceBundle) -> ResourceRegistry:
    """Rebuild the static profiles and initial states saved with a trace."""

    profiles = trace.manifest.resource_profiles
    llm_instances = profiles.get("llm_instances")
    tool_replicas = profiles.get("tool_replicas")
    if not isinstance(llm_instances, list) or not isinstance(tool_replicas, list):
        raise ValueError("manifest resource_profiles must contain resource lists")

    resources = ResourceRegistry()
    for item in llm_instances:
        if not isinstance(item, dict):
            raise ValueError("each LLM resource snapshot must be an object")
        resources.register_llm(
            LLMInstanceProfile.from_dict(item.get("profile")),
            LLMInstanceState.from_dict(item.get("state")),
        )
    for item in tool_replicas:
        if not isinstance(item, dict):
            raise ValueError("each Tool resource snapshot must be an object")
        resources.register_tool_replica(
            ToolReplicaProfile.from_dict(item.get("profile")),
            ToolReplicaState.from_dict(item.get("state")),
        )
    return resources


def reconstruct_call(call_trace: CallTrace) -> SchedulableCall:
    """Validate a trace record and rebuild its call in CREATED state."""

    payload = deepcopy(call_trace.call_payload)
    if content_digest(payload) != call_trace.call_digest:
        raise ValueError(f"call digest mismatch for {call_trace.call_id!r}")

    if call_trace.call_kind == "llm":
        payload_id = payload.get("llm_call_id")
        if payload_id != call_trace.call_id:
            raise ValueError(f"LLM call_id mismatch for {call_trace.call_id!r}")
        expected_input_digest = call_trace.parameter_summary.get("input_digest")
        actual_input_digest = content_digest(payload.get("input_items"))
        if expected_input_digest != actual_input_digest:
            raise ValueError(f"LLM input summary mismatch for {call_trace.call_id!r}")
        payload["status"] = CallStatus.CREATED
        return LLMCall.from_dict(payload)

    payload_id = payload.get("tool_call_id")
    if payload_id != call_trace.call_id:
        raise ValueError(f"Tool call_id mismatch for {call_trace.call_id!r}")
    if payload.get("tool_name") != call_trace.tool_name:
        raise ValueError(f"Tool name mismatch for {call_trace.call_id!r}")
    expected_arguments_digest = call_trace.parameter_summary.get("arguments_digest")
    actual_arguments_digest = content_digest(payload.get("arguments"))
    if expected_arguments_digest != actual_arguments_digest:
        raise ValueError(f"Tool arguments summary mismatch for {call_trace.call_id!r}")
    payload["status"] = CallStatus.CREATED
    return ToolCall.from_dict(payload)
