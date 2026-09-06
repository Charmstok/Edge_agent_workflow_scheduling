"""Compare real Tool replicas against fixed expectations and complete text artifacts."""

from __future__ import annotations

import hashlib
import json
from math import isclose
from typing import Any

from edge_agent_workflow_scheduling.common import ToolResult
from edge_agent_workflow_scheduling.resources import ToolConsistencySample
from edge_agent_workflow_scheduling.tools import resolve_local_path


def compare_tool_results(
    sample: ToolConsistencySample,
    results: list[ToolResult],
) -> dict[str, Any]:
    """Ignore output paths and timings; compare normalized content and numeric features."""

    from pathlib import Path

    if len(results) < 2 or len({result.replica_id for result in results}) != len(results):
        raise ValueError("consistency comparison requires at least two distinct replicas")
    reports = []
    signatures = []
    digests = []
    for result in results:
        differences = []
        metadata = result.metadata
        signatures.append(
            json.dumps(
                {
                    key: metadata.get(key)
                    for key in (
                        "backend",
                        "backend_version",
                        "implementation_version",
                        "backend_configuration",
                    )
                },
                sort_keys=True,
            )
        )
        if not result.success:
            differences.append(f"execution failed: {result.error_code}")
        else:
            try:
                output = result.output
                text_bytes = resolve_local_path(output["text_uri"], Path(".")).read_bytes()
                if hashlib.sha256(text_bytes).hexdigest() != output["text_sha256"]:
                    raise ValueError("text artifact digest mismatch")
                normalized = " ".join(text_bytes.decode("utf-8").split())
                digests.append(hashlib.sha256(normalized.encode()).hexdigest())
                observations = {**metadata.get("work_features", {}), **output}
                for name, expected in sample.expected.items():
                    if name == "text_contains":
                        for fragment in expected:
                            if " ".join(fragment.split()) not in normalized:
                                differences.append(f"missing text: {fragment}")
                    elif name in sample.numeric_tolerances:
                        actual = observations.get(name)
                        if (
                            isinstance(actual, bool)
                            or not isinstance(actual, int | float)
                            or not isclose(
                                actual, expected, rel_tol=0, abs_tol=sample.numeric_tolerances[name]
                            )
                        ):
                            differences.append(f"{name}: expected {expected}, observed {actual}")
                    elif observations.get(name) != expected:
                        differences.append(f"{name}: expected {expected}")
            except (OSError, ValueError, KeyError, TypeError) as exc:
                differences.append(f"invalid artifact: {exc}")
        reports.append(
            {
                "replica_id": result.replica_id,
                "passed": not differences,
                "differences": differences,
                "implementation": json.loads(signatures[-1]),
            }
        )
    matching_content = len(digests) == len(results) and len(set(digests)) == 1
    same_configuration = len(set(signatures)) == 1
    return {
        "sample_id": sample.sample_id,
        "tool_name": sample.tool_name,
        "passed": all(report["passed"] for report in reports) and matching_content,
        "matching_normalized_text": matching_content,
        "same_implementation_configuration": same_configuration,
        "requires_separate_quality_profile": not same_configuration or not matching_content,
        "replicas": reports,
    }
