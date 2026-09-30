"""Private stdio worker process; one isolated persistent process per execution slot."""

from __future__ import annotations

import json
import os
import resource
import sys
from dataclasses import asdict, replace
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from uuid import uuid4


def _reply(value):
    print(json.dumps(value, ensure_ascii=False), flush=True)


def _usage():
    own = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    return own, children


def main():
    try:
        request = json.loads(sys.stdin.readline())
        from edge_agent_workflow_scheduling.common import ToolCall
        from edge_agent_workflow_scheduling.executors.adapters import LocalToolExecutor
        from edge_agent_workflow_scheduling.resources import ToolReplicaProfile
        from edge_agent_workflow_scheduling.tools import ToolRegistry
        from edge_agent_workflow_scheduling.tools.deployment import (
            canonical_tool_output,
            create_local_tool,
            dependency_report,
        )
        from edge_agent_workflow_scheduling.workers.local import LocalWorker

        profile = ToolReplicaProfile.from_dict(request["profile"])
        tool = create_local_tool(
            profile.tool_name,
            input_root=request["input_root"],
            output_dir=request["output_dir"],
            timeout_sec=request["timeout_sec"],
            options=profile.deployment_config.get("tool_options", {}),
        )
        dependencies = dependency_report(tool, request["timeout_sec"])
        registry = ToolRegistry()
        registry.register(tool)
        executor = LocalToolExecutor(LocalWorker(profile, registry))
        _reply(
            {
                "kind": "ready",
                "pid": os.getpid(),
                "cwd": str(Path.cwd()),
                "dependencies": dependencies,
                "tool_schema": tool.spec,
                "implementation_version": getattr(tool, "implementation_version", "pillow-v1"),
            }
        )
    except Exception as exc:
        _reply(
            {
                "kind": "startup_failure",
                "error_code": "dependency_unavailable"
                if isinstance(exc, (ModuleNotFoundError, ImportError))
                else "worker_startup_failed",
                "error_message": str(exc) or type(exc).__name__,
            }
        )
        return
    for line in sys.stdin:
        try:
            request = json.loads(line)
            call = ToolCall.from_dict(request["call"])
            # Repeated call IDs must not overwrite or reuse a prior invocation's
            # files (PDFRender detects only newly produced page artifacts).
            invocation_dir = (
                Path(tool.config.output_dir)
                / sha256(call.tool_call_id.encode()).hexdigest()[:16]
                / uuid4().hex
            )
            invocation_tool = replace(tool, config=replace(tool.config, output_dir=invocation_dir))
            invocation_registry = ToolRegistry()
            invocation_registry.register(invocation_tool)
            executor = LocalToolExecutor(LocalWorker(profile, invocation_registry))
            before, before_children = _usage()
            started = perf_counter()
            result = executor.execute(call, timeout_sec=request["timeout_sec"])
            if result.success:
                try:
                    canonical_tool_output(
                        profile.tool_name,
                        result.output,
                        artifact_root=Path(request["artifact_root"]),
                    )
                except (ValueError, KeyError, TypeError, OSError) as exc:
                    result = replace(
                        result,
                        success=False,
                        output=None,
                        error_code="invalid_output",
                        error_message=str(exc),
                    )
            after, after_children = _usage()
            cpu_sec = (after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime) + (
                after_children.ru_utime
                + after_children.ru_stime
                - before_children.ru_utime
                - before_children.ru_stime
            )
            elapsed = perf_counter() - started
            result = replace(
                result,
                metadata={
                    **result.metadata,
                    "worker_pid": os.getpid(),
                    "worker_cwd": str(Path.cwd()),
                    "configured_implementation_version": profile.implementation_version,
                    "implementation_version": getattr(tool, "implementation_version", "pillow-v1"),
                    "dependency_versions": dependencies,
                    "backend_configuration": result.metadata.get(
                        "backend_configuration",
                        {
                            key: value
                            for key, value in asdict(tool.config).items()
                            if key
                            not in {"output_dir", "local_root", "timeout_sec", "inline_text_chars"}
                        },
                    ),
                    "backend_version": result.metadata.get(
                        "backend_version",
                        (
                            dependencies["executables"]
                            .get(getattr(tool.config, "executable", ""), {})
                            .get("version", dependencies["packages"].get("Pillow"))
                        ),
                    ),
                    "deployment_kind": "same_host_logical_replica",
                    "resource_sampling": {
                        "source": "getrusage_self_and_reaped_children",
                        "cpu_time_sec": max(cpu_sec, 0.0),
                        "max_rss_kib": max(after.ru_maxrss, after_children.ru_maxrss),
                        "rss_scope": "worker_lifetime_peak",
                        "cpu_util_one_core": min(max(cpu_sec / max(elapsed, 1e-9), 0), 1),
                    },
                },
            )
            _reply({"kind": "result", "result": result.to_dict()})
        except Exception as exc:
            _reply({"kind": "protocol_failure", "error_message": str(exc) or type(exc).__name__})
            return


if __name__ == "__main__":
    main()
