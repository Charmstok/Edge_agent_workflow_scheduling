# Edge Agent Workflow Scheduling

This project is a research prototype for multi-objective scheduling of dynamic
Agent workflows. Each workflow alternates between LLM calls and Tool calls. The
scheduler chooses among heterogeneous LLM instances and multiple replicas of the
same Tool while tracking latency, deadline misses, quality, energy, and load balance.

The repository supports local/profile execution for deterministic development and
RL experiments, local real Tools and vLLM models for live validation, and optional
Volcengine Ark access through OpenAI-compatible APIs.

## Install

Python 3.11 or newer is recommended. The project uses a virtual environment.

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -r requirements-dev.txt
uv pip install --no-deps -e .
```

Optional document Tools require:

```bash
uv pip install -r requirements-tools.txt
```

OCR also requires Tesseract. On Arch Linux:

```bash
sudo pacman -S --needed tesseract tesseract-data-eng poppler
```

Generated traces, profiles, reports, and training artifacts are written under
`data/`, which is ignored by Git. Versioned experiment inputs live in `configs/`.

## Resource Configuration

- `configs/llm_profiles.toml`: local Qwen 9B/27B and Volcengine GLM profiles;
- `configs/tool_profiles.toml`: two logical local replicas for each document Tool;
- `configs/rl_resources_arch_local_v1.toml`: local RL resource catalog;
- `configs/document_agent_workload_v1.json`: document tasks, fixtures, and arrival plans.

Validate the resource catalog without network access:

```bash
PYTHONPATH=src .venv/bin/python scripts/validate_resource_catalog.py
```

The cloud profiles use `glm-5-3-flash-260828`, endpoint
`https://ark.cn-beijing.volces.com/api/v3`, and `ARK_API_KEY`. The resource IDs are
`online-glm-1` and `online-glm-2`.

Set the key in the current zsh terminal without displaying it:

```zsh
read -rs 'ARK_API_KEY?Volcengine Ark API key: '
print
export ARK_API_KEY
```

For repeated local use, persist it in the ignored `.env.ark` file:

```zsh
(umask 077; printf 'export ARK_API_KEY=%q' "$ARK_API_KEY" > .env.ark)
source .env.ark
```

Never commit or share `.env.ark`. Fixed profile/replay experiments do not require
the key; live cloud calls do.

## Validate Real Local Tool Replicas

Run the offline deployment and acceptance checks:

```bash
PYTHONPATH=src .venv/bin/python scripts/deploy_local_tools.py
```

The command starts two replicas each of `image_preprocess`, `ocr`, `pdf_parse`,
and `pdf_render`. Every replica has its own ID, work directory, artifact root,
profile, and concurrency limit of two persistent process slots (16 slots total).
These are labeled `same_host_logical_replica`; measured network latency is zero.
The workers reuse the existing Tool implementations and `LocalToolExecutor`.

Startup checks record dependency versions and executables, validate fixture
outputs across both slots and replicas, and mask failed replicas. Acceptance
checks exercise round-robin, least-queue, restored Double DQN, individual RL
action masks, and structured missing-dependency, invalid-input, timeout, and
worker-exit failures. Worker deadlines include queue waiting and terminate owned
process groups, including subprocesses. Repeated calls keep separate artifacts.

`data/local_tool_deployment/` contains `summary.json`, `startup.json`, unified
`trace.jsonl`, full `execution_records.jsonl`, state events, measured startup
profiles, and the RL smoke checkpoint/history. Traces include queue/execution
times, failure codes, CPU time, and process lifetime peak RSS. This command stops
its workers after validation. Live `scripts/run_workload.py` runs start and
self-check the same eight replicas and share them throughout the workload.

The RL check uses one-fixture latency profiles and proves scheduling integration;
it does not measure a performance improvement. Energy is unavailable and output
consistency is not a task-quality measurement. The synthetic RL comparison uses two
synthetic image Tool profiles, so its results do not cover this deployment.

## Validate LLM Deployments

Load the two local Qwen and two Ark profiles and validate their offline paths
without creating HTTP clients or requiring credentials:

```bash
PYTHONPATH=src .venv/bin/python scripts/validate_llm_deployment.py
```

The offline contract fixture supplies explicitly synthetic throughput values
for the local profiles. It proves both targets are schedulable and masks both
cloud profiles; it does not calibrate model performance. The public catalog
preserves missing quality, throughput, and energy measurements with their source
notes, context limits, capability declarations, and historical verification.

Check currently running local endpoints and require real automatic Function
Calling, Tool execution, result feedback, and direct arithmetic responses:

```bash
PYTHONPATH=src .venv/bin/python scripts/validate_llm_deployment.py \
  --live-local --require-local-function-calling
```

Local endpoints use `QWEN9B_BASE_URL` and `QWEN27B_BASE_URL` overrides. Real Tool
execution shares the eight independent local replicas. The validation closes
its clients and Tool workers after use; the existing vLLM services remain running.
Outputs are in `data/llm_deployment_validation/`, including `summary.json`,
offline traces, per-model live Function Calling traces, and Tool self-checks.
Unavailable endpoints are recorded as offline and masked. Without the strict
flag, unavailable local endpoints leave real validation explicitly skipped.

Cloud clients require both explicit enablement and `ARK_API_KEY`; a key alone
does not enable them. Both Ark catalog entries default to `enabled=false`.
For an optional real cloud smoke, use `--cloud-smoke`; it enables at most one
request per cloud profile, capped at 256 output tokens by the validation config.
Missing credentials produce skipped/offline entries. `ARK_PRIMARY_MODEL`,
`ARK_SECONDARY_MODEL`, and `ARK_BASE_URL` override cloud routing independently.
The two cloud IDs share a model/endpoint by default and do not establish two
independent remote servers. Other live entry points require explicitly setting
the selected cloud profile's `deployment_config.enabled=true`.

Provider executors record provider/model, a public parameter summary and digest,
and measured client request time including network and server wait. Pure network
time is unavailable; transfers stay zero to avoid counting that time twice.
Secret values and authorization fields are removed from provider responses and
error artifacts. These checks do not measure energy or task quality.

## Train RL

The RL prototype uses a Gymnasium environment, configurable multi-objective reward,
and a NumPy Double DQN agent. Training uses profile executors by default and does
not require network access or cloud credentials.

Train from a fixed replay trace:

```bash
PYTHONPATH=src .venv/bin/python scripts/train_rl.py \
  path/to/trace.json \
  --output-dir data/rl_training \
  --episodes 100 \
  --eval-episodes 10 \
  --seed 0 \
  --profile-seed 0
```

To create a deterministic validation trace before training:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_workload.py \
  --mode scripted \
  --workload configs/document_agent_workload_v1.json \
  --scenario low_load --split validation \
  --output-dir data/workload
```

Training writes `checkpoint.json`, `training_history.json`, and `evaluation.json`.
The checkpoint contains network weights, seeds, resource/workload metadata, the
environment schema, and stable action IDs. The environment applies the same action
mask as the baseline scheduler and rejects invalid targets before executor dispatch.

## Compare RL with Baselines

Run the offline RL comparison with its versioned replay fixture,
weights, normalization, training settings, and statistical protocol:

```bash
PYTHONPATH=src .venv/bin/python scripts/compare_rl.py
```

`configs/rl_comparison_v1.json` declares two scheduler/training seeds and four
profile seeds. Each of the eight baselines and greedy Double DQN uses the same
replay engine, resource initial states, calls, dependency batches, profile noise,
and evaluator. Profile noise is keyed by call ID, target ID, and profile seed;
it is independent of scheduler seed. Quality-constrained EFT additionally applies
its declared minimum quality threshold.

Outputs in `data/rl_comparison/` include checkpoints, training histories,
per-run manifests/decisions/traces, raw objectives and metrics in JSON/CSV,
policy statistics, Pareto flags, and RL comparison deltas with every regression.
The paired sign-flip test averages scheduler seeds within each profile seed and
corrects eight baseline comparisons with Bonferroni. P95/P99 describe repeated
profile runs of one fixed synthetic AgentRun; energy and quality are profile
proxies. This experiment makes no hardware or unseen-workload performance claim.

Evaluate the saved checkpoints in a fresh process without training:

```bash
PYTHONPATH=src .venv/bin/python scripts/compare_rl.py \
  --checkpoint-dir data/rl_comparison/training \
  --output-dir data/rl_comparison_reload
```

Checkpoint loading rejects mismatched source code, workload/arrival fingerprints,
resource profiles, observation schema, action IDs, weights, or normalization.
Missing objective profiles and infeasible calls fail preflight with a saved reason;
executor failures remain in the seven-call trace and success statistics.

## Tests

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q
.venv/bin/ruff check .
```

The detailed research plan is in [`docs/project_plan.md`](docs/project_plan.md).
