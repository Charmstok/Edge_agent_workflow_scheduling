# Edge Agent Workflow Scheduling

## Project

This project studies multi-objective scheduling for Agent workflows on
heterogeneous edge resources. A workflow contains both LLM inference steps and
Tool steps. The scheduler selects:

- one of several LLM instances with different model sizes, throughput, quality,
  energy profiles, and queue states;
- one of several replicas of the same Tool deployed on different edge nodes.

The optimization targets are Agent end-to-end latency, deadline misses, model
quality, energy, and load balance. The current prototype combines mock LLM
runtimes with a real local image preprocessing Tool. Additional real Tools,
models, and remote devices are introduced as experiment adapters rather than as
requirements for algorithm development.

The research roadmap is documented in
[`docs/project_plan.md`](docs/project_plan.md).

## Install

The project requires Python 3.11+ and uses `uv`:

```bash
uv sync --dev
```

## Run

Run the local end-to-end prototype:

```bash
uv run python scripts/run_first_demo.py --policy round_robin
```

Available policies:

```text
random
round_robin
least_queue
earliest_finish_time
```

Use a smaller workload while developing:

```bash
uv run python scripts/run_first_demo.py \
  --policy earliest_finish_time \
  --steps-per-agent 2 \
  --trace-path data/traces/first_demo.jsonl
```

The demo creates two simulated Agents, two heterogeneous mock LLM runtimes, and
two local Worker replicas. LLM steps use deterministic mock inference; Tool
steps execute `ImagePreprocessTool` with Pillow and record real execution time.
Results are written as JSONL traces under `data/traces/`.

Run static checks:

```bash
uv run ruff check .
```

## Layout

```text
src/edge_agent_workflow_scheduling/
├── agents/       # workload generation
├── common/       # calls, results, target state, trace schemas
├── llm/          # mock LLM runtime
├── profiler/     # JSONL trace logging
├── queue/        # mixed LLM/Tool queue
├── scheduler/    # baseline policies
├── tools/        # real Tool wrappers
└── workers/      # local real-Tool execution

scripts/
└── run_first_demo.py
```
