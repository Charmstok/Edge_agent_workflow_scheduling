# Scheduler

The scheduler selects an execution target for each workflow step.

Current supported step types:

- `LLMCall`: scheduled to an online LLM runtime.
- `ToolCall`: scheduled to an online worker that supports the requested `tool_type`.

The scheduler returns a `ScheduleDecision` with the selected target, policy name, score, and reason.

## Supported Policies

### `random`

Randomly selects one valid target from the candidate list.

For `ToolCall`, candidates are workers that are online and support the requested tool.
For `LLMCall`, candidates are online LLM runtimes with a matching `model_name` when the call specifies one.

This policy is useful as a simple baseline for trace collection.

### `round_robin`

Cycles through valid targets in a stable order.

For tool steps, rotation is tracked per `tool_type`.
For LLM steps, rotation is tracked per requested `model_name`.

This policy is useful for basic load spreading without using runtime load metrics.

### `least_queue`

Selects the valid target with the smallest reported `queue_len`.

This policy uses current runtime state and is a stronger heuristic baseline than `random` or `round_robin`.

### `earliest_finish_time`

Selects the valid target with the smallest estimated finish time.

For LLM runtimes, the estimate uses queue length, average latency, token count, and `tokens_per_sec`.
For workers, the estimate uses queue length, network latency, CPU utilization, and memory utilization.

This policy is intended as the main heuristic comparison point for later multi-objective optimization.

## Multi-Objective Optimization Context

These baseline policies provide comparison points for future RL-based scheduling.

Future reward functions can compare against these policies using metrics such as:

- total latency
- queue waiting time
- execution time
- timeout rate
- failure rate
- load balance
- privacy or placement constraints
