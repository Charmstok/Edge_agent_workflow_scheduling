# Scheduler Policies

This directory contains the scheduler policies registered by `SchedulerPolicyRegistry`. Each policy implements the same small selection interface, so a policy can be compared on the same replay call stream without changing the Agent or executor code.

## Baseline policies

### `random`

Selects one feasible target using a local `random.Random` instance. The `random_seed` in `SchedulerPolicyConfig` controls the sequence. It is useful as a stochastic reference point and should normally be evaluated with multiple seeds.

### `round_robin`

Cycles through feasible targets in stable target order. It does not use queue length, profile quality, latency, or energy estimates. The policy is deterministic when the input call order and candidate set are fixed.

### `least_queue`

Chooses the feasible target with the smallest current queue length. Ties are resolved by stable target order. This is a lightweight load-oriented baseline, not a prediction of end-to-end completion time.

### `earliest_finish_time`

Chooses the feasible target with the smallest estimated finish time. The estimate uses the target state and the latency/throughput profile. Missing latency information is rejected instead of silently assigning a default.

## Objective-aware policies

### `quality_aware`

Chooses the feasible target with the highest profiled quality for the call's `task_type`. Quality here means the configured resource profile value; it is not a measurement of final task accuracy.

### `energy_aware`

Chooses the feasible target with the lowest estimated energy. LLM energy is derived from token count and `joules_per_token`; Tool energy is derived from `joules_per_call`.

### `weighted_objective`

Computes a fixed, configurable cost from latency, energy, deadline miss, capacity-normalized load imbalance, and `1 - quality`. The weights must sum to one, and latency and energy use positive experiment-wide normalization scales. The selected objective vector is recorded when `record_objectives` is enabled.

### `quality_constrained_earliest_finish_time`

First applies the call's hard minimum-quality constraint, then chooses the earliest-finishing feasible target. A missing `min_quality` or missing required quality profile is an error; it is not treated as a zero-quality candidate.

## Configuration and extension

`BaselineScheduler` applies action masks and hard constraints before invoking a policy. `SchedulerPolicyConfig` carries the random seed, objective settings, and whether estimated objectives should be recorded. The registry is the single place that maps policy names to configured factories.

Future reinforcement-learning policies should live in this same `policies/` directory and be added to the registry. They should implement the existing `SchedulerPolicy` protocol, consume the same action mask, and preserve the same decision and trace fields so that baseline and RL experiments remain comparable.
