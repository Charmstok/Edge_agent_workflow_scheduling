# Edge Agent Workflow Scheduling

## Project

This project studies multi-objective scheduling for dynamic Agent calls on
heterogeneous edge resources. The scheduler selects:

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

The project requires Python 3.11+ and uses `uv` to create `.venv`. Dependencies
are installed from requirements files; the project does not use `uv.lock`. Use
the same Python minor version on the main device and edge nodes when collecting
comparable experiment results.

For development on the main experiment device:

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -r requirements-dev.txt
uv pip install --no-deps -e .
```

For a Raspberry Pi or another Tool-only edge node:

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -r requirements-edge.txt
uv pip install --no-deps -e .
```

Use a 64-bit Raspberry Pi OS when possible so Pillow and its dependencies can
use prebuilt wheels. If Pillow must build from source, install the platform's
JPEG, zlib, and FreeType development packages first. Edge nodes do not install
main-device development or future LLM-provider dependencies.

## Run

Run the local end-to-end prototype:

```bash
python scripts/run_first_demo.py --policy round_robin
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
python scripts/run_first_demo.py \
  --policy earliest_finish_time \
  --runs-per-agent 2 \
  --trace-path data/traces/first_demo.jsonl
```

The demo creates two simulated Agents, two heterogeneous mock LLM runtimes, and
two local Worker replicas. LLM calls use deterministic mock inference; Tool
calls execute `ImagePreprocessTool` with Pillow and record real execution time.
Results are written as JSONL traces under `data/traces/`.

Run static checks:

```bash
ruff check .
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
