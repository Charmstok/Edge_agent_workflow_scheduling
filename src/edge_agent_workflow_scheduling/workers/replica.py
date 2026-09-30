"""Persistent local replica workers with isolated slots, bounded queues and deadlines."""

from __future__ import annotations

import json
import os
import queue
import signal
import subprocess
import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock, Thread
from time import perf_counter

from edge_agent_workflow_scheduling.common import CallStatus, ToolCall, ToolResult
from edge_agent_workflow_scheduling.resources import (
    ResourceRegistry,
    ToolReplicaProfile,
)


class WorkerSlot:
    """One owned process group and one serial stdio request stream."""

    def __init__(self, profile, *, input_root, output_dir, work_dir, timeout_sec, on_exit):
        self.responses = queue.Queue()
        self.closed = False
        self.on_exit = on_exit
        work_dir.mkdir(parents=True, exist_ok=True)
        scratch = work_dir / "tmp"
        scratch.mkdir(exist_ok=True)
        environment = {**os.environ, "TMPDIR": str(scratch)}
        source_root = Path(__file__).resolve().parents[2]
        # The parent may have a relative PYTHONPATH; workers use a different cwd.
        paths = [str(source_root)] + [
            str(Path(p).resolve()) for p in environment.get("PYTHONPATH", "").split(os.pathsep) if p
        ]
        environment["PYTHONPATH"] = os.pathsep.join(paths)
        self.stderr = (work_dir / "stderr.log").open("a", encoding="utf-8")
        self.process = subprocess.Popen(
            [sys.executable, "-m", "edge_agent_workflow_scheduling.workers.replica_process"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.stderr,
            text=True,
            encoding="utf-8",
            bufsize=1,
            cwd=work_dir,
            env=environment,
            start_new_session=True,
        )
        self.reader = Thread(target=self._read, daemon=True)
        self.reader.start()
        self.ready = self.request(
            {
                "profile": profile.to_dict(),
                "input_root": str(input_root),
                "output_dir": str(output_dir),
                "timeout_sec": timeout_sec,
            },
            timeout_sec,
        )

    def _read(self):
        try:
            for line in self.process.stdout:
                try:
                    self.responses.put(json.loads(line))
                except ValueError:
                    self.responses.put(
                        {"kind": "protocol_failure", "error_message": "invalid worker JSON"}
                    )
        finally:
            self.responses.put({"kind": "worker_exited", "error_message": "worker stdout closed"})
            self.on_exit()

    def request(self, payload, timeout_sec):
        try:
            self.process.stdin.write(json.dumps(payload) + "\n")
            self.process.stdin.flush()
            return self.responses.get(timeout=timeout_sec)
        except queue.Empty:
            return {"kind": "timeout", "error_message": "local worker deadline exceeded"}
        except (BrokenPipeError, OSError, ValueError) as exc:
            return {"kind": "worker_exited", "error_message": str(exc)}

    def close(self):
        if self.closed:
            return
        self.closed = True
        # Kill only this worker's session, including its OCR/Poppler descendants.
        try:
            os.killpg(self.process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        self.process.wait(timeout=5)
        self.reader.join(timeout=1)
        self.process.stdin.close()
        self.process.stdout.close()
        self.stderr.close()


class ReplicaWorker:
    """LocalWorker-compatible manager, owning max_concurrency process slots."""

    def __init__(
        self,
        profile: ToolReplicaProfile,
        *,
        resources: ResourceRegistry,
        input_root: Path,
        output_dir: Path,
        work_dir: Path,
        timeout_sec: float = 30.0,
        manage_load: bool = True,
        lock=None,
        events=None,
    ):
        self.profile = profile
        self.resources = resources
        self.input_root = input_root.resolve()
        self.output_dir = output_dir.resolve()
        self.work_dir = work_dir.resolve()
        self.timeout_sec = timeout_sec
        self.manage_load = manage_load
        self.lock = lock or RLock()
        self.events = events if events is not None else []
        self.slots: list[WorkerSlot] = []
        self.available = queue.Queue()
        self.online = False
        self.offline_error = ("replica_offline", "local replica is offline")
        self.pending: dict[str, float] = {}
        self.queue_len = 0
        self.running = 0
        self.completed = 0
        self.failures = 0
        self.peak_running = 0
        self._publish("created")

    def start(self):
        self.close()
        self.available = queue.Queue()
        self.slots = []
        self.output_dir.mkdir(parents=True, exist_ok=True)
        failure = None
        for index in range(self.profile.max_concurrency):
            slot = WorkerSlot(
                self.profile,
                input_root=self.input_root,
                output_dir=self.output_dir,
                work_dir=self.work_dir / f"slot-{index}",
                timeout_sec=self.timeout_sec,
                on_exit=self._worker_exited,
            )
            self.slots.append(slot)
            if slot.ready.get("kind") != "ready":
                failure = slot.ready
                break
            self.available.put(slot)
        with self.lock:
            self.online = failure is None
            self.offline_error = (
                (
                    failure.get("error_code", failure.get("kind", "worker_startup_failed")),
                    failure.get("error_message", "worker failed to start"),
                )
                if failure
                else ("replica_offline", "local replica is offline")
            )
            self._publish("started" if self.online else "startup_failed")
        report = {
            "replica_id": self.profile.replica_id,
            "online": self.online,
            "max_concurrency": self.profile.max_concurrency,
            "deployment_kind": "same_host_logical_replica",
            "work_dir": str(self.work_dir),
            "artifact_root": str(self.output_dir),
            "slots": [slot.ready for slot in self.slots],
            "failure": failure,
        }
        if failure:
            self.close()
        return report

    def _publish(self, stage, call_id=None, result=None):
        previous = self.resources.tool_snapshot(self.profile.replica_id).state
        sampling = result.metadata.get("resource_sampling", {}) if result else {}
        state = replace(
            previous,
            is_online=self.online,
            network_latency_ms=0.0,
            queue_len=self.queue_len if self.manage_load else previous.queue_len,
            running_tasks=self.running if self.manage_load else previous.running_tasks,
            cpu_util=sampling.get("cpu_util_one_core", previous.cpu_util),
            avg_execution_time_sec=result.execution_time_sec
            if result
            else previous.avg_execution_time_sec,
            recent_failure_rate=self.failures / self.completed if self.completed else 0.0,
            updated_at=datetime.now(UTC).isoformat(),
        )
        self.resources.update_tool_state(state)
        self.events.append(
            {
                "replica_id": self.profile.replica_id,
                "call_id": call_id,
                "stage": stage,
                "state": state.to_dict(),
            }
        )

    def _worker_exited(self):
        with self.lock:
            if self.online:
                self.online = False
                self.offline_error = ("worker_exited", "local replica worker exited")
                self._publish("worker_exited")

    def reserve(self, call):
        with self.lock:
            if call.tool_call_id in self.pending:
                raise ValueError("call is already reserved on this replica")
            self.pending[call.tool_call_id] = perf_counter()
            self.queue_len += 1
            self._publish("queued", call.tool_call_id)

    def run_tool(self, call: ToolCall, *, timeout_sec=None) -> ToolResult:
        budget = min(timeout_sec, self.timeout_sec) if timeout_sec is not None else self.timeout_sec
        with self.lock:
            if call.tool_call_id not in self.pending:
                self.reserve(call)
            submitted = self.pending.pop(call.tool_call_id)
        deadline = submitted + budget
        slot = None
        running_started = None
        failure_code = None
        message = None
        while slot is None:
            with self.lock:
                if not self.online:
                    failure_code, message = self.offline_error
                    break
                if any(item.process.poll() is not None for item in self.slots):
                    self.online = False
                    self._publish("worker_exited", call.tool_call_id)
                    failure_code, message = "worker_exited", "local replica worker exited"
                    break
            remaining = deadline - perf_counter()
            if remaining <= 0:
                failure_code, message = "timeout", "local replica queue budget exhausted"
                break
            try:
                slot = self.available.get(timeout=min(remaining, 0.05))
            except queue.Empty:
                continue
        if slot is not None:
            running_started = perf_counter()
            with self.lock:
                self.queue_len -= 1
                self.running += 1
                self.peak_running = max(self.peak_running, self.running)
                if call.status == CallStatus.QUEUED:
                    call.transition_to(CallStatus.RUNNING)
                self._publish("running", call.tool_call_id)
            reply = slot.request(
                {
                    "call": call.to_dict(),
                    "timeout_sec": max(deadline - perf_counter(), 1e-9),
                    "artifact_root": str(self.output_dir),
                },
                max(deadline - perf_counter(), 1e-9),
            )
            if reply.get("kind") == "result":
                try:
                    result = ToolResult.from_dict(reply["result"])
                    if (
                        result.tool_call_id != call.tool_call_id
                        or result.replica_id != self.profile.replica_id
                    ):
                        raise ValueError("worker result identity mismatch")
                except (ValueError, TypeError, KeyError) as exc:
                    failure_code, message = "worker_protocol_error", str(exc)
            else:
                failure_code = reply.get("kind", "worker_protocol_error")
                message = reply.get("error_message", "local worker failed")
            if failure_code is not None:
                slot.close()
                with self.lock:
                    self.online = False
        if failure_code is not None:
            result = ToolResult(
                tool_call_id=call.tool_call_id,
                replica_id=self.profile.replica_id,
                success=False,
                error_code=failure_code,
                error_message=message,
            )
        now = perf_counter()
        waited = (running_started if running_started is not None else now) - submitted
        execution = now - running_started if running_started is not None else 0.0
        result = replace(
            result,
            queue_wait_time_sec=max(waited, 0.0),
            execution_time_sec=max(execution, result.execution_time_sec),
            input_transfer_time_sec=0.0,
            output_transfer_time_sec=0.0,
            metadata={
                **result.metadata,
                "deployment_kind": "same_host_logical_replica",
                "replica_work_dir": str(self.work_dir),
                "artifact_root": str(self.output_dir),
                "executor_type": "local_tool",
                "energy_source": result.metadata.get("energy_source", "unavailable"),
                "execution_time_source": "measured_local_worker_round_trip",
                "network_latency_semantics": "same_host_zero",
                "worker_pid": slot.process.pid if slot else None,
                "resource_sampling": result.metadata.get(
                    "resource_sampling",
                    {
                        "source": "unavailable_after_worker_failure",
                        "status": "unavailable",
                    },
                ),
            },
        )
        with self.lock:
            if running_started is None:
                self.queue_len -= 1
            else:
                self.running -= 1
            self.completed += 1
            self.failures += int(not result.success)
            self._publish("completed", call.tool_call_id, result)
        if slot is not None and failure_code is None:
            self.available.put(slot)
        return result

    def close(self):
        with self.lock:
            self.online = False
            self._publish("stopped")
        for slot in self.slots:
            slot.close()
