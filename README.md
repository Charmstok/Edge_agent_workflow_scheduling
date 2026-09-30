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

## Tests

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q
.venv/bin/ruff check .
```

The detailed research plan is in [`docs/project_plan.md`](docs/project_plan.md).
