# Workload Fixture Dataset

## 1. Scope and Purpose

This directory contains the fixed input artifacts used by the workload configuration
`configs/workload_milestone_4_1_v1.json`. The dataset consists of synthetic PNG images
and PDF documents for evaluating dynamic Agent workloads that combine LLM inference and
external Tool calls.

The fixtures are designed for research prototyping, regression testing, and profile
calibration. They are not production data, a public benchmark, or a representative
sample of real business documents.

The artifacts are intentionally model- and scheduler-independent. They define task
inputs only; they do not prescribe a Tool-call sequence, an LLM target, or a Tool replica.

## 2. Data Generation

The fixtures were generated locally from deterministic templates:

- PNG files were rendered with Pillow as synthetic invoice images.
- PDF files were rendered with ReportLab as synthetic project reports.
- Invoice identifiers, project names, monetary values, and budgets are artificial.
- No real customer, financial, or operational data is included.

The original development-time generation logic was implemented in a temporary local
script. The checked-in fixture files, rather than that temporary script, are the
canonical inputs for replay and regression experiments.

## 3. Naming and Dataset Splits

The filename format is:

```text
<source-group>-<scale>.<extension>
```

The current dataset uses two source groups:

| Source group | Dataset split | Role |
| --- | --- | --- |
| `alpha` | `calibration` | Used for initial profile and scoring development |
| `beta` | `validation` | Held out from calibration and used for validation |

Each source group has three scale variants:

| Scale | PNG resolution | PDF page count |
| --- | ---: | ---: |
| `small` | 640 × 480 | 1 |
| `medium` | 1280 × 960 | 3 |
| `large` | 2560 × 1920 | 6 |

The scale variants within a source group are related inputs, not independent observations.
All variants of a source remain in the same split. Composite tasks that reference both an
image and a PDF must also remain within one split. This prevents source-level leakage
between calibration and validation.

## 4. Artifact Semantics

### 4.1 Invoice Images

The PNG artifacts contain synthetic invoice-like fields:

- invoice identifier
- project name
- total amount
- currency

For example, `beta-small.png` contains:

```text
Invoice: INV-202
Project: BETA
Total: 270.00
Currency: USD
```

The image is used as input to image OCR tasks and to composite document-reconciliation
tasks. The different resolutions provide controlled variation in image size and pixel
workload.

### 4.2 Project Reports

The PDF artifacts contain synthetic project-report fields:

- project name
- page number and total page count
- currency
- repeated observation text
- approved budget
- approval status

For example, `beta-small.pdf` defines a synthetic report with a budget of `250.00`.
Together with `beta-small.png`, it forms a reconciliation case in which the invoice total
is `270.00` and therefore exceeds the stated budget.

## 5. Workload Task Mapping

The workload configuration defines 18 tasks:

```text
3 task types × 3 input scales × 2 source groups = 18 tasks
```

| Task type | Input artifacts | Intended operation |
| --- | --- | --- |
| `image_ocr` | PNG | Extract invoice fields |
| `pdf_extract` | PDF | Extract project, budget, currency, and status |
| `document_reconcile` | PNG + PDF | Reconcile invoice and project-report fields |

Each task specifies a reference answer and an `exact_fields` scoring rule. The score is
normalized to `[0, 1]` as the fraction of reference fields that are matched correctly.
Invalid JSON, missing fields, failed execution, empty output, and timeout are scored as
zero. Extra output fields do not affect the score.

The task definition, Tool schema, reference answer, scale metadata, and artifact references
are stored in `workload_milestone_4_1_v1.json`; they are not encoded in the image or PDF
files themselves.

## 6. Role in the Scheduling Study

These fixtures support controlled experiments in which an Agent may repeatedly perform
LLM inference and invoke external Tools. The same task inputs can be evaluated under
multiple resource configurations, including:

- larger local LLM instances, such as approximately 30B-parameter models on Ubuntu;
- smaller LLM instances, such as approximately 7B-parameter models on other edge boards;
- multiple replicas of image, OCR, or PDF Tools;
- profile-based execution and later real execution adapters.

The scheduler may choose both the LLM execution target and the Tool replica. Consequently,
the fixtures provide stable inputs for measuring queueing delay, inference latency, Tool
execution latency, energy proxies, task quality, and end-to-end Agent latency.

Image resolution, PDF page count, and task composition provide controlled workload
variation. They do not imply that Tool-call count, LLM output length, execution time, or
semantic difficulty must increase monotonically with scale. Those properties must be
measured rather than assumed.

## 7. Reproducible Workload Preparation

From the repository root, prepare a deterministic request plan with:

```bash
python scripts/prepare_workload.py \
  configs/workload_milestone_4_1_v1.json \
  --scenario burst \
  --split validation \
  --output data/workload_v1/plan.json
```

The preparation command:

1. validates that all referenced artifacts exist;
2. computes SHA-256 hashes for the artifact files;
3. selects task samples using the configured input seed;
4. generates deterministic Agent request IDs and arrival offsets; and
5. writes a workload plan without executing an Agent, LLM, OCR Tool, or PDF Tool.

The `artifact_refs` paths are resolved relative to the workload configuration file. For
example, `workload_fixtures_v1/beta-small.png` refers to the PNG file in this directory.
Identical configuration, artifact contents, and seeds produce the same request plan.

## 8. Limitations

This is a small synthetic fixture set with two source groups. It is suitable for verifying
interfaces, replay determinism, and controlled profile experiments, but it is not
sufficient by itself to support claims about real-world document understanding, model
quality, or heterogeneous hardware performance.

Measured execution results, generated traces, profile estimates, and scheduler summaries
must be stored separately from these immutable input artifacts. If the fixture semantics
change, update the workload version and reference answers so that prior experiments remain
interpretable.
