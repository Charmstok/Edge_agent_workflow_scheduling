# Edge Agent Workflow Scheduling

## Project

This project studies multi-objective scheduling for dynamic Agent calls on heterogeneous edge resources. The scheduler selects:

- one of several LLM instances with different model sizes, throughput, quality, energy profiles, and queue states;
- one of several replicas of the same Tool deployed on different edge nodes.

The optimization targets are Agent end-to-end latency, deadline misses, model quality, energy, and load balance. The current prototype combines mock LLM runtimes with a real local image preprocessing Tool. Additional real Tools, models, and remote devices are introduced as experiment adapters rather than as requirements for algorithm development.

The research roadmap is documented in [`docs/project_plan.md`](docs/project_plan.md).

## Install

The project requires Python 3.11+ and uses `uv` to create `.venv`. Dependencies are installed from requirements files; the project does not use `uv.lock`. Use the same Python minor version on the main device and edge nodes when collecting comparable experiment results.

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

Use a 64-bit Raspberry Pi OS when possible so Pillow and its dependencies can use prebuilt wheels. If Pillow must build from source, install the platform's JPEG, zlib, and FreeType development packages first. Edge nodes do not install main-device development or future LLM-provider dependencies.

## Run

### Real document Tools

OCR, PDF extraction, and PDF page rendering are optional local adapters with measured execution time, explicit pixel/page work units, bounded artifacts, and batch-wide timeouts. See `src/edge_agent_workflow_scheduling/tools/README.md` for the measurement protocol, dependency installation, and multi-objective interpretation.

```bash
uv pip install --python .venv/bin/python -r requirements-tools.txt
python scripts/run_tool_demos.py
python scripts/run_tool_demos.py --tools ocr --batch-size 80 --scale large
```

OCR additionally requires a local Tesseract installation. Missing dependencies are reported as skips, not successful validation. Existing offline/profile demos remain usable.

Collect a versioned local Tool sampling matrix with cold-start, warm-up, and measurement phases:

```bash
python scripts/sample_tools.py
```

The sampler records timing distributions, throughput, success/failure counts, queue depth, host inventory, process-tree CPU/RSS samples, and explicit unavailable GPU metrics.

Version-controlled experiment inputs and assumptions live in `configs/`. All generated
samples, imported benchmarks, traces, fitted profiles, reports, and demo artifacts live in
`data/`, which is intentionally excluded from Git.

### Latency and energy profile fitting

Collect the bounded local calibration matrix, then fit bucketed Tool and LLM profiles from
the saved observations:

```bash
PYTHONPATH=src python scripts/sample_tools.py \
  --config configs/tool_profile_sampling_v1.json \
  --output-dir data/tool_sampling \
  --experiment-id arch-linux-document-tools-profile-v2-20260923-r2

PYTHONPATH=src python scripts/fit_profiles.py \
  --tool-sampling-run data/tool_sampling/arch-linux-document-tools-profile-v2-20260923-r2 \
  --llm-benchmark data/llm_sampling/qwen38-27b-local-20260911/benchmark.json \
  --profile-version arch-linux-document-tools-calibrated-v2 \
  --synthetic-energy-config configs/synthetic_energy_profiles_v1.json \
  --output data/profile_calibration/arch-linux-document-tools-calibrated-v2/profiles.json \
  --overwrite
```

The complete calibration-and-validation chain can also be run through the traceable
orchestrator. It records input paths, SHA-256 references, the fitted catalog, and the
holdout report in the generated profile directory:

```bash
PYTHONPATH=src python scripts/calibrate_profiles.py \
  --tool-sampling-run data/tool_sampling/arch-linux-document-tools-profile-v2-20260923-r2 \
  --llm-benchmark data/llm_sampling/qwen38-27b-local-20260911/benchmark.json \
  --profile-version arch-linux-document-tools-calibrated-v2 \
  --synthetic-energy-config configs/synthetic_energy_profiles_v1.json \
  --profile-output data/profile_calibration/arch-linux-document-tools-calibrated-v2/profiles.json \
  --tool-holdout-run data/tool_validation_sampling/arch-linux-document-tools-holdout-v2-20260923-r2 \
  --validation-config configs/profile_validation_v1.json \
  --validation-output data/profile_validation/arch-linux-document-tools-validation-v2 \
  --overwrite --require-pass
```

The generated profile catalog is directly loadable by the existing replay/baseline resource
loader. Calibrated profiles use exact task/input-size/concurrency buckets and reject calls
outside their declared scope unless the fit command explicitly selects aggregate fallback.
Local energy is currently unavailable, so measured profiles leave `energy_profile` empty and
energy-dependent policies reject them. Synthetic energy is confined to the labeled logical
replica and records its assumptions separately.
For a new device or profile version, repeat `--tool-sampling-run` and choose a new output;
the host digest is part of every Tool profile ID. Existing catalogs are not overwritten unless
`--overwrite` is explicitly supplied.

Validate the fitted profile against the independent `beta-*` holdout fixtures using the
predeclared thresholds in `configs/profile_validation_v1.json`:

```bash
PYTHONPATH=src python scripts/sample_tools.py \
  --config configs/tool_profile_validation_v1.json \
  --output-dir data/tool_validation_sampling \
  --experiment-id arch-linux-document-tools-holdout-v2-20260923-r2

PYTHONPATH=src python scripts/validate_profiles.py \
  --tool-holdout-run \
    data/tool_validation_sampling/arch-linux-document-tools-holdout-v2-20260923-r2 \
  --output-dir data/profile_validation/arch-linux-document-tools-validation-v2 \
  --require-pass
```

The validator writes per-sample errors and grouped summaries by Tool, input size, and
concurrency. It compares an aggregate-only baseline with exact fitted buckets, checks input
hashes for calibration/holdout leakage, and separately reports distribution mismatch across
multiple profile seeds. Missing LLM holdout and measured energy observations remain explicitly
unvalidated rather than receiving synthetic scores.

### Agent demos

Run all Milestone 2 demos. Without `ARK_API_KEY`, the online demo is skipped while offline, multi-Tool, and replay verification still complete:

```bash
python scripts/run_agent_demos.py --mode all
```

Artifacts are written under `data/milestone_2_8/`. Each executed demo writes a public experiment manifest, a complete call trace, and an AgentRun or replay summary.

Run one mode at a time:

```bash
python scripts/run_agent_demos.py --mode offline
python scripts/run_agent_demos.py --mode multi-tool
python scripts/run_agent_demos.py --mode replay
```

The replay mode reads `data/milestone_2_8/offline/trace.json` by default and compares `round_robin` with `least_queue`. Another trace or policy set can be selected explicitly:

```bash
python scripts/run_agent_demos.py \
  --mode replay \
  --replay-trace path/to/trace.json \
  --replay-policies least_queue earliest_finish_time
```

### Online LLM configuration

The online profile uses Volcengine Ark and lives in `configs/llm_profiles.toml`. API key values must not be added to that file. Export the Ark API key, then run the online demo:

```bash
export ARK_API_KEY="your-api-key"
python scripts/run_agent_demos.py --mode online
```

Use repeated live runs to observe variation in Tool selection and latency:

```bash
python scripts/run_agent_demos.py --mode online --online-runs 5
```

The program creates and caches the OpenAI-compatible SDK client only when the Doubao instance is selected. The SDK sends requests to Volcengine's configured `base_url`; it does not use the OpenAI platform.

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

This scans representative objective weights, runs the reference policies, and writes traceable CSV/JSON points under `data/milestone_3_7/`.

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
├── fit_profiles.py
├── run_agent_demos.py
├── run_baselines.py
├── run_workload.py
├── run_first_demo.py
├── run_pareto.py
└── validate_profiles.py
```

### LLM measurement

vLLM/Ark deployment inputs are defined in `configs/llm_profiles.toml`; bounded real
sampling and credential-free benchmark import are provided by `scripts/sample_llms.py`.
Both local vLLM deployments are enabled: Qwen3.5-9B at port 8000 and
Qwen3.8-27B-FP8 at port 8001. The local experiment data currently includes 18 historical
Qwen3.8-27B-FP8 observations and explicitly labeled measured/synthetic profiles under
`data/`; current-deployment 9B/27B calibration and cloud measurements remain incomplete.
These generated artifacts are not committed. The current 9B and 27B endpoints have both
passed the repository's real end-to-end Function Calling verifier with `tool_choice=auto`,
including a Tool-needed and a no-Tool scenario. A minimal standalone 27B curl can still
expose XML-format content, so the repository verifier, with the full Tool schema and
Runner prompt, is the acceptance path. Re-run `python scripts/verify_function_calling.py`
after any parser or chat-template change.

### Task quality calibration

Task scoring is defined by the versioned rules in
`configs/workload_milestone_4_1_v1.json`. The quality sampler runs the same Agent prompt,
Tool set, and budget over calibration and validation samples, then writes raw scores,
confidence intervals, a holdout report, and loadable LLM profiles:

```bash
export PYTHONPATH="$PWD/src"
python scripts/sample_quality.py --output-dir data/quality_sampling
```

For existing traces, run only the deterministic scoring and aggregation step:

```bash
python scripts/score_quality.py \
  --trace-root data/quality_sampling/<run-timestamp> \
  --output-dir data/quality_scoring
```

`data/quality_sampling/<run-timestamp>/quality/quality_report.json` keeps calibration
quality separate from validation quality. The generated
`data/quality_scoring/document-agent-quality-v1/profiles.json` contains the calibrated
Qwen3.8-27B task profile;
models without measured coverage retain an empty `quality_profile` and are not given a
silent default score. The evaluator reports both selected-profile quality and the final
task score of each AgentRun.

### Milestone 4.8 workload, replay, and live status

Use one entry point for the versioned workload. The scripted mode needs no API key and
writes complete, deterministic call traces for all three document task types:

```bash
PYTHONPATH=src python scripts/run_workload.py \
  --mode scripted \
  --workload configs/workload_milestone_4_1_v1.json \
  --scenario low_load --split validation \
  --output-dir data/milestone_4_8
```

Replay never asks an LLM to choose a Tool. It reuses the saved call stream and compares
at least two policies over the same inputs; profile jitter/failure injection is explicit:

```bash
PYTHONPATH=src python scripts/run_workload.py \
  --mode replay --output-dir data/milestone_4_8 \
  --policies round_robin least_queue --seeds 0 1
```

Live mode records the selected deployment, repeat count, and sampling parameters. It
returns `not_validated` when credentials or verified Function Calling are unavailable.
The local vLLM profiles use automatic Tool parsing and have been verified with the
project Runner. A complete 27B validation run is recorded under
`data/milestone_4_8_live_27b_verified/` (generated and ignored). Scripted scores and
replay evaluations must not be reported as replacement live LLM quality or latency
measurements:

```bash
PYTHONPATH=src python scripts/run_workload.py \
  --mode live --llm-config configs/llm_profiles.toml \
  --llm-id local-qwen35-9b --repeats 3 \
  --output-dir data/milestone_4_8
```
