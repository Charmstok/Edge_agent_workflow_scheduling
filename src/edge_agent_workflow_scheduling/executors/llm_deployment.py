"""Load four LLM resources with explicit offline/live activation boundaries."""

from __future__ import annotations

from dataclasses import replace

from edge_agent_workflow_scheduling.config import load_llm_profiles
from edge_agent_workflow_scheduling.executors.activation import llm_activation_error
from edge_agent_workflow_scheduling.executors.chat import OpenAIChatExecutor
from edge_agent_workflow_scheduling.executors.openai import (
    OpenAIResponsesExecutor,
    create_openai_client,
)
from edge_agent_workflow_scheduling.executors.profile import ProfileLLMExecutor
from edge_agent_workflow_scheduling.resources import LLMInstanceState, ResourceRegistry


class LLMDeployment:
    """Own clients and publish schedulable resources, including masked cloud profiles."""

    def __init__(
        self,
        profiles,
        *,
        mode="offline",
        allow_cloud=False,
        offline_rates=None,
        health_timeout_sec=3.0,
        client_factory=None,
    ):
        if mode not in {"offline", "live"}:
            raise ValueError("LLM deployment mode must be offline or live")
        self.profiles = list(profiles)
        self.mode = mode
        self.allow_cloud = allow_cloud
        self.offline_rates = offline_rates or {}
        self.health_timeout_sec = health_timeout_sec
        self.client_factory = client_factory or create_openai_client
        self.resources = ResourceRegistry()
        self.executors = {}
        self.reports = {}
        self.clients = []

    @classmethod
    def from_catalog(cls, path, **kwargs):
        return cls(load_llm_profiles(path), **kwargs)

    def __enter__(self):
        try:
            for original in self.profiles:
                profile = original
                cloud = profile.platform == "cloud"
                report = {
                    "llm_id": profile.llm_id,
                    "provider": profile.provider,
                    "model": profile.model,
                    "endpoint": profile.base_url,
                    "mode": self.mode,
                    "status": "skipped",
                    "online": False,
                    "client_created": False,
                    "function_calling_currently_verified": False,
                }
                if cloud and (self.mode == "offline" or not self.allow_cloud):
                    report["reason"] = (
                        "offline_mode" if self.mode == "offline" else "cloud_not_enabled"
                    )
                else:
                    if cloud:
                        profile = replace(
                            profile,
                            deployment_config={
                                **profile.deployment_config,
                                "enabled": True,
                            },
                        )
                    invalid = llm_activation_error(profile)
                    if invalid:
                        report["reason"] = invalid[0]
                    elif self.mode == "offline":
                        rate = self.offline_rates.get(profile.llm_id)
                        if rate is None or rate <= 0:
                            raise ValueError(
                                "offline contract validation requires explicit synthetic rates"
                            )
                        profile = replace(
                            profile,
                            executor_type="profile",
                            token_profile={"tokens_per_sec": rate},
                            metadata={
                                **profile.metadata,
                                "profile_source": "synthetic_offline_contract_fixture",
                                "hardware_performance_claim": False,
                            },
                        )
                        self.executors[profile.llm_id] = ProfileLLMExecutor(profile)
                        report.update(
                            status="profile_ready",
                            online=True,
                            reason="synthetic_offline_contract_fixture",
                        )
                    else:
                        try:
                            client = self.client_factory(profile)
                            self.clients.append(client)
                            report["client_created"] = True
                            if not cloud:
                                models = client.models.list(timeout=self.health_timeout_sec)
                                model_ids = [item.id for item in models.data]
                                report["available_models"] = model_ids
                                if profile.model not in model_ids:
                                    raise ValueError("configured model not advertised by endpoint")
                            factory = {
                                "openai_chat": OpenAIChatExecutor,
                                "openai_responses": OpenAIResponsesExecutor,
                            }[profile.executor_type]
                            self.executors[profile.llm_id] = factory(
                                profile,
                                client,
                                profile.deployment_config.get("model_parameters", {}),
                            )
                            report.update(
                                status="ready",
                                online=True,
                                reason="client_ready_not_function_calling_verification",
                            )
                        except Exception as exc:
                            report.update(
                                status="offline",
                                reason="endpoint_or_executor_unavailable",
                                error_type=type(exc).__name__,
                            )
                self.resources.register_llm(
                    profile,
                    LLMInstanceState(llm_id=profile.llm_id, is_online=report["online"]),
                )
                self.reports[profile.llm_id] = report
        except BaseException:
            self.close()
            raise
        return self

    def register_factories(self, factories):
        def factory(profile):
            if profile.llm_id not in self.executors:
                raise RuntimeError("LLM resource is offline or skipped")
            return self.executors[profile.llm_id]

        for kind in {profile.executor_type for profile in self.profiles} | {"profile"}:
            factories.register_llm(kind, factory)

    def close(self):
        for client in self.clients:
            client.close()
        self.clients.clear()

    def __exit__(self, *_):
        self.close()
