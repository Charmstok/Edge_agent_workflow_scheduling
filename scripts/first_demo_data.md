# First Demo Data

`scripts/run_first_demo.py` writes local generated data under `data/` by default. The
directory is ignored by git because it contains reproducible runtime artifacts.

Run the demo with:

```bash
uv run python scripts/run_first_demo.py --policy round_robin
```

## Default Layout

```text
data/
├── inputs/
│   └── first_demo/
│       ├── images/
│       │   ├── agent_image_7b_0000.ppm
│       │   └── agent_image_27b_0010.ppm
│       └── outputs/
│           ├── agent_image_7b-tc-0000.pgm
│           └── agent_image_27b-tc-0010.pgm
└── traces/
    └── first_demo.jsonl
```

## Generated Input Images

`data/inputs/first_demo/images/` contains synthetic PPM images created by the demo.
They are not external datasets. They exist only to give the `image_preprocess` tool
local files to process.

Filename pattern:

```text
<agent_id>_<sequence_id>.ppm
```

Examples:

- `agent_image_7b_0000.ppm`: input image for the simulated 7B agent workflow.
- `agent_image_27b_0010.ppm`: input image for the simulated 27B agent workflow.

The default demo creates 20 input images: 10 for `agent_image_7b` and 10 for
`agent_image_27b`.

## Tool Outputs

`data/inputs/first_demo/outputs/` contains processed PGM images written by the
`image_preprocess` tool. These files are the outputs of `ToolCall` execution.

Filename pattern:

```text
<agent_id>-tc-<sequence_id>.pgm
```

Examples:

- `agent_image_7b-tc-0000.pgm`: output from tool call `agent_image_7b-tc-0000`.
- `agent_image_27b-tc-0010.pgm`: output from tool call `agent_image_27b-tc-0010`.

The default demo creates 20 output images, one for each `image_preprocess` tool
call.

## Trace File

`data/traces/first_demo.jsonl` is the profiler trace. It is a JSONL file: each
line is one completed workflow step.

The default demo generates 40 trace records:

- 20 records for `LLMCall` execution.
- 20 records for `ToolCall` execution.

Important fields:

- `task_id`: workflow step id, such as `agent_image_7b-lc-0000` or
  `agent_image_7b-tc-0000`.
- `task_kind`: `llm` for LLM inference steps, `tool` for tool execution steps.
- `agent_id`: simulated agent that produced the step.
- `selected_target`: selected LLM runtime id or worker id.
- `policy_name`: scheduler policy used for the decision.
- `queue_wait_time_sec`: queue wait time included in the trace.
- `execution_time_sec`: LLM inference time or tool execution time.
- `total_latency_sec`: total measured or simulated step latency.
- `success`: whether the step completed successfully.
- `timeout`: whether the step timed out.
- `reward`: first-stage reward, currently based on negative latency plus penalties.

LLM records additionally use:

- `model_name`
- `input_tokens`
- `output_tokens`

Tool records additionally use:

- `tool_type`

## Runtime Options

The data locations can be changed:

```bash
uv run python scripts/run_first_demo.py \
  --trace-path data/traces/custom_demo.jsonl \
  --data-dir data/inputs/custom_demo
```

By default, an existing trace file is replaced. Use `--append-trace` to append new
records to an existing trace file.
