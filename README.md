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

Run all Milestone 2 demos. Without `ARK_API_KEY`, the online demo is skipped
while offline, multi-Tool, and replay verification still complete:

```bash
python scripts/run_agent_demos.py --mode all
```

Artifacts are written under `data/milestone_2_8/`. Each executed demo writes a
public experiment manifest, a complete call trace, and an AgentRun or replay
summary.

Run one mode at a time:

```bash
python scripts/run_agent_demos.py --mode offline
python scripts/run_agent_demos.py --mode multi-tool
python scripts/run_agent_demos.py --mode replay
```

The replay mode reads `data/milestone_2_8/offline/trace.json` by default and
compares `round_robin` with `least_queue`. Another trace or policy set can be
selected explicitly:

```bash
python scripts/run_agent_demos.py \
  --mode replay \
  --replay-trace path/to/trace.json \
  --replay-policies least_queue earliest_finish_time
```

### Online LLM configuration

The online profile uses Volcengine Ark and lives in
`configs/llm_profiles.toml`. API key values must not be added to that file.
Export the Ark API key, then run the online demo:

```bash
export ARK_API_KEY="your-api-key"
python scripts/run_agent_demos.py --mode online
```

Use repeated live runs to observe variation in Tool selection and latency:

```bash
python scripts/run_agent_demos.py --mode online --online-runs 5
```

The program creates and caches the OpenAI-compatible SDK client only when the
Doubao instance is selected. The SDK sends requests to Volcengine's configured
`base_url`; it does not use the OpenAI platform.

The earlier mixed-call JSONL prototype remains available:

```bash
python scripts/run_first_demo.py --policy round_robin
```

Run static checks:

```bash
ruff check .
```

Run the versioned Milestone 3 Pareto experiment on the fixed offline replay trace:

```bash
python scripts/run_pareto.py data/milestone_2_8/offline/trace.json
```

This scans representative objective weights, runs the reference policies, and
writes traceable CSV/JSON points under `data/milestone_3_7/`.

## Layout

```text
src/edge_agent_workflow_scheduling/
├── agents/       # workload generation
├── common/       # calls, results, target state, trace schemas
├── executors/    # provider-neutral real and profile execution adapters
├── llm/          # mock LLM runtime
├── profiler/     # experiment traces, manifests, and replay
├── queue/        # mixed LLM/Tool queue
├── scheduler/    # baseline policies
├── tools/        # real Tool wrappers
└── workers/      # local real-Tool execution

scripts/
├── run_agent_demos.py
├── run_baselines.py
├── run_first_demo.py
└── run_pareto.py
```
