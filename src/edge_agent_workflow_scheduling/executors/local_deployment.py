"""Load, self-check and share all real local Tool replica executors."""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from statistics import fmean
from threading import Event, RLock

from edge_agent_workflow_scheduling.config import load_tool_profiles
from edge_agent_workflow_scheduling.executors.adapters import LocalToolExecutor
from edge_agent_workflow_scheduling.resources import ResourceRegistry, ToolReplicaState
from edge_agent_workflow_scheduling.workers.replica import ReplicaWorker


class LocalToolDeployment:
    """Context-managed workers; use these same executors in CLI or AgentRunner."""

    def __init__(
        self,
        profiles,
        *,
        input_root,
        output_dir,
        timeout_sec=30.0,
        resources=None,
        manage_load=True,
    ):
        self.resources = resources or ResourceRegistry()
        self.input_root = Path(input_root).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.timeout_sec = timeout_sec
        self.lock = RLock()
        self.events = []
        self.workers = {}
        self.executors = {}
        self.startup = {}
        for profile in profiles:
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", profile.replica_id):
                raise ValueError("replica_id must be safe for a local deployment directory")
            if profile.executor_type != "local_tool":
                raise ValueError("local deployment requires executor_type=local_tool")
            relative = profile.deployment_config.get("work_dir", f"replicas/{profile.replica_id}")
            work_dir = (self.output_dir / relative).resolve()
            if not work_dir.is_relative_to(self.output_dir) or work_dir == self.output_dir:
                raise ValueError("replica work_dir must be inside the deployment output directory")
            if any(worker.work_dir == work_dir for worker in self.workers.values()):
                raise ValueError("replicas must use independent work directories")
            artifact_dir = work_dir / "artifacts"
            self.resources.register_tool_replica(
                profile, ToolReplicaState(replica_id=profile.replica_id, is_online=False)
            )
            worker = ReplicaWorker(
                profile,
                resources=self.resources,
                input_root=self.input_root,
                output_dir=artifact_dir,
                work_dir=work_dir,
                timeout_sec=timeout_sec,
                manage_load=manage_load,
                lock=self.lock,
                events=self.events,
            )
            self.workers[profile.replica_id] = worker
            self.executors[profile.replica_id] = LocalToolExecutor(worker)
        self.threads = ThreadPoolExecutor(
            max_workers=max(sum(profile.max_concurrency for profile in profiles), 1)
        )

    @classmethod
    def from_catalog(cls, path, **kwargs):
        return cls(load_tool_profiles(path), **kwargs)

    def __enter__(self):
        try:
            for replica_id, worker in self.workers.items():
                self.startup[replica_id] = worker.start()
        except BaseException:
            self.close()
            raise
        return self

    def close(self):
        for worker in self.workers.values():
            worker.close()
        self.threads.shutdown(wait=True)

    def __exit__(self, *_):
        self.close()

    def register_factories(self, factories):
        factories.register_tool("local_tool", lambda profile: self.executors[profile.replica_id])

    def measure_profile(self, replica_id, results):
        """Publish one-fixture measured latency without inventing energy or quality."""
        if not results or not all(result.success for result in results):
            with self.lock:
                self.workers[replica_id].online = False
                self.workers[replica_id]._publish("self_check_failed")
            return
        with self.lock:
            worker = self.workers[replica_id]
            profile = replace(
                worker.profile,
                latency_profile={
                    "execution_time_sec": fmean(result.execution_time_sec for result in results)
                },
                metadata={
                    **worker.profile.metadata,
                    "profile_version": "local-tool-startup-fixture-v1",
                    "profile_source": "measured_startup_fixture",
                    "latency_scope": "one_fixture_not_general_calibration",
                    "energy_status": "unavailable",
                    "quality_status": "consistency_only_not_task_quality",
                },
            )
            state = self.resources.tool_snapshot(replica_id).state
            self.resources.register_tool_replica(profile, state, replace=True)
            worker.profile = profile

    def submit_batch(self, calls, scheduler):
        """Reserve all arrivals before launching a same-round Tool batch."""
        gate = Event()
        submissions = []

        def execute(call, worker, executor):
            gate.wait()
            return executor.execute(call, timeout_sec=self.timeout_sec)

        try:
            for call in calls:
                with self.lock:
                    decision = scheduler.schedule(call, resources=self.resources)
                    worker = self.workers[decision.selected_target]
                    worker.reserve(call)
                    future = self.threads.submit(
                        execute, call, worker, self.executors[decision.selected_target]
                    )
                    submissions.append((call, decision, future))
        finally:
            gate.set()
        return submissions
