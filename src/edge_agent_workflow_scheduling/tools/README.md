# Local Tools for Joint LLM and Tool Scheduling

## 1. Research Scope

This module provides real local computation for document-oriented Agent workloads.
Its purpose is to support scheduling experiments, not to introduce new image-processing
or document-recognition algorithms. The implementations reuse `ToolRegistry`, `LocalWorker`,
`LocalToolExecutor`, `ToolCall`, and `ToolResult`; no separate scheduler or service is added.

An Agent decides which operation to request, while the scheduler selects its execution
target. An illustrative task is to read an invoice image, extract a project budget from
a PDF, and determine whether the invoice exceeds that budget. The Agent may request image
preprocessing, OCR, and PDF extraction before reasoning over the returned evidence.
This example describes task semantics, not a fixed Tool-call sequence or enforced DAG.

The scheduling problem includes both the Tool calls and the LLM calls that produce or
consume them. The intended deployment includes two approximately 30B LLM instances on an
Ubuntu server and approximately 7B instances on other boards. These are deployment targets,
not devices evaluated by the local Tool demo.

## 2. Tool Descriptions

Each Tool is described below in terms of its task purpose, implementation, workload
characteristics, and relevance to multi-objective scheduling.

| Tool | Backend | Input → output | Primary workload controls |
| --- | --- | --- | --- |
| `image_preprocess` | Pillow | Image → transformed image artifact | Resolution, operation sequence, repetition |
| `ocr` | Tesseract | Image or image batch → extracted text | Pixel count, batch size, recognition configuration |
| `pdf_parse` | pypdf | PDF or PDF batch → page-ordered text | Page count, file size, batch size |

### 2.1 Image Preprocessing

**Purpose and implementation.** `ImagePreprocessTool` applies a configured sequence of
grayscale conversion, resizing, Gaussian blur, thresholding, and edge detection. It returns
an image artifact rather than recognized text. An Agent may submit the resulting image
to OCR, but preprocessing is not mandatory. Deskewing, page segmentation, and learned
image enhancement are outside the implemented scope.

**Workload characterization.** The implementation records input/output dimensions,
operation count, and the following estimated work proxy:

```text
estimated_work_units = input_pixels * operation_count
```

This is not an exact count of processed pixels or CPU instructions. Operations have
different costs, and resizing changes the dimensions used by subsequent operations.
Repeated transformations can change the output content; increasing repetition is not
necessarily an output-preserving way to increase load.

**Scheduling relevance.** This Tool provides configurable image computation for studying
processing latency and downstream recognition quality. Whether a transform improves OCR
accuracy must be measured. For placement-only comparisons, keep operations and parameters
fixed; changed preprocessing settings are additional workload or quality variables.

### 2.2 Optical Character Recognition

**Purpose and implementation.** `OCRTool` invokes the Tesseract command-line engine to
extract text from single-frame images. An invocation accepts one image or an explicit
image batch. The returned text supplies evidence for subsequent LLM field extraction,
interpretation, or cross-document comparison. The wrapper does not itself infer structured
invoice fields or determine whether an invoice is valid.

The default configuration uses language `eng`, page segmentation mode 6, and
`OMP_THREAD_LIMIT=1`. The engine version, language, segmentation mode, and thread setting
are recorded with successful results. Other languages require the corresponding language
data. Multi-frame images are rejected; use separate image entries for multiple pages.

**Workload characterization.** Recorded features include input count, input bytes,
completed image count, and total completed image pixels. A descriptive normalization is
seconds per pixel. A candidate model for subsequent calibration is:

```text
T_ocr ≈ intercept + a * input_count + b * image_pixels
```

Batch size increases actual recognition work without introducing artificial waits.
Duplicate inputs are permitted for labeled stress tests, not as independent quality samples.

**Scheduling relevance.** OCR supplies a candidate compute-intensive operation for
studying replica placement, queueing, and Agent completion time. Recognition errors may
propagate into the final answer, so execution success is not a quality score. Hardware
speed and energy differences require measurement; changing the engine or recognition
settings requires separate quality evaluation.

### 2.3 PDF Text Extraction

**Purpose and implementation.** `PDFParseTool` runs pypdf in an isolated Python subprocess
to extract embedded text from every page, preserving page order with separators. This
supports subsequent LLM information extraction, summarization, or comparison with OCR
results. It does not reconstruct tables, export images, or recover a complete layout.

The Tool does not perform implicit OCR or render scanned pages for a later OCR call.
Image-only pages may yield empty text, recorded in `empty_text_pages`; empty extraction
is not successful document understanding. Encrypted PDFs are rejected.

**Workload characterization.** Recorded features include input count, input bytes,
completed page count, empty-text page count, and `top_level_image_count`. The latter counts
image XObjects directly referenced by page resources, not images recursively nested in
form XObjects. It is a measurement feature, not an image-export capability.

The primary normalization is seconds per page. A candidate latency model is:

```text
T_pdf ≈ intercept + c * input_count + d * page_count + e * input_bytes
```

**Scheduling relevance.** PDF extraction introduces a document workload distinct from
OCR, allowing experiments with mixed short and long calls. Simple text PDFs may remain
fast at larger page counts; no minimum duration is assumed. Larger extracted documents
increase subsequent LLM context only when that text is actually included in the request.
Page count is not a substitute for measured LLM token usage, especially with truncated
inline Tool output.

## 3. Shared Execution Contracts

The input, text-output, and subprocess timeout conventions in this section apply to
`OCRTool` and `PDFParseTool`. `ImagePreprocessTool` retains its existing operation-specific
arguments, image-artifact output, and in-process execution.

### 3.1 Single-file and Batch Inputs

Both document Tools accept one required argument:

```json
{"input_uri": "configs/workload_fixtures_v1/alpha-small.pdf"}
```

Plain paths are relative to the configured `local_root`; `file://` and `local://` references
use the existing local path resolver. Remote URLs are not downloaded. A JSON input denotes
a batch manifest:

```json
{
  "schema_version": 1,
  "input_uris": ["page-001.png", "page-002.png", "page-003.png"]
}
```

Entries are resolved relative to the manifest directory and executed once in order.
OCR batches contain images, PDF batches contain PDFs, and nested manifests are unsupported.
Input-byte totals count every entry, including duplicates, rather than unique file storage.

### 3.2 Text Outputs and Artifacts

| Output field | Meaning |
| --- | --- |
| `schema_version` | Output contract version, currently 1 |
| `text` | Complete short text, or bounded first/last excerpts |
| `text_truncated` | Whether inline text omits any content |
| `text_uri` | Local UTF-8 artifact containing the complete text |
| `text_sha256` | SHA-256 of the complete artifact bytes |
| `normalized_text_sha256` | Hash after whitespace normalization |
| `text_chars` | Character count before truncation |
| `input_count` | Number of inputs in the invocation |

The default inline limit is 4096 characters. Truncated text concatenates the first and
last portions within this limit; it is not a continuous excerpt. Set `inline_text_chars=0`
for artifact-only text delivery. Full-text consumers must resolve `text_uri`; an Agent
without artifact-reading support receives only the inline excerpt.

Artifacts are written under the configured output directory using invocation and content
hashes. Results and traces contain no input image or PDF binary payloads. Keep full-text
artifacts available for subsequent consistency checks.

### 3.3 Timeouts and Failures

The optional `TimeoutTool.execute_with_timeout` interface preserves compatibility with
existing `Tool.execute` implementations. `LocalToolExecutor` forwards the remaining call
budget through the worker and registry. OCR/PDF use the smaller of that budget and their
configured timeout, sharing one deadline across the entire batch.

Extraction subprocesses are killed and reaped on timeout; each input does not receive a
fresh budget. Input inspection and filesystem operations are checked between stages but
are not independently preempted. This is not a hard real-time guarantee. Existing
in-process image execution retains its post-execution timeout check.

| Error code | Condition |
| --- | --- |
| `invalid_arguments` | Arguments rejected by ToolRegistry schema validation |
| `invalid_input` | Malformed manifest, unsupported URI, invalid image, or local I/O issue |
| `input_not_found` | Missing referenced local input |
| `dependency_unavailable` | Missing Tesseract executable or pypdf package |
| `backend_execution_failed` | Unsuccessful backend exit, including corrupt/encrypted PDFs |
| `timeout` | Invocation budget expires |

Failed calls preserve elapsed time and available metadata, but do not return a partial
successful document result. Missing dependencies are reported separately from passes.

## 4. Measurement and Multi-objective Interpretation

### 4.1 Timing Boundaries

`LocalWorker` measures `ToolResult.execution_time_sec` with a monotonic performance clock.
For the document Tools, this includes registry validation, input inspection, subprocess
startup, backend execution, and output handling. Queueing and transfer times remain
separate result fields. No artificial delay is enabled in the document Tool demos.

OCR/PDF metadata additionally records:

| Metadata field | Interpretation |
| --- | --- |
| `tool_wall_time_sec` | Elapsed time within the wrapper |
| `backend_time_sec` | Sum of completed extraction subprocess wall times |
| `completed_inputs` | Inputs completed before success or failure |
| `work_features` | Requested batch size and completed-input workload measurements |
| `work_units`, `work_unit`, `sec_per_work_unit` | Successful-call normalization described in Section 2 |
| `backend_version`, `backend_configuration`, `implementation_version` | Execution provenance |

Backend time includes process launch overhead, not just algorithm CPU time. It excludes
failed or interrupted extraction attempts, whose time remains included in the whole-call
measurement. On failure, `input_count` and `input_bytes` describe the requested batch,
while page/pixel totals describe completed inputs only.

The models in Section 2 are calibration candidates, not implemented predictors with
assigned coefficients. Fit and validate them under fixed device, backend, language, and
concurrency settings. Unit-time ratios do not establish exact linear scaling. CPU frequency,
text density, and layout complexity are not inferred from nominal input sizes.

### 4.2 Energy, Quality, and Load

- **Energy:** the wrappers do not measure energy. Configured `joules_per_call` is labeled
  `energy_source="profiled"`; missing energy is labeled `energy_source="unavailable"`.
  The legacy numeric field then contains 0 for schema compatibility, not a physical
  zero-energy observation. It must not be used as measured energy or as calibration data.
- **Quality:** execution completion, replica consistency, and final-task correctness are
  different properties. The demo leaves quality profiles uncalibrated. Backend or setting
  changes require separately calibrated quality profiles before quality-aware use.
- **Load:** batch entries execute sequentially. Concurrent Agent load and resource capacity
  belong to the existing scheduling/execution layer, not a hidden worker pool within a Tool.

The wrappers do not change the scheduler's profile estimator. Recorded observations can
support subsequent profile calibration; they do not automatically change decisions.

## 5. Experimental Validation

### 5.1 Replica Consistency

`configs/tool_consistency_v1.json` defines `ToolConsistencySample` inputs, expected text
fragments, numeric features, and absolute tolerances. The demo registers these samples in
`ResourceRegistry` and schedules two same-configuration logical replicas per Tool through
the round-robin scheduler and `LocalToolExecutor`.

Checks cover artifact integrity, reference fragments, numeric expectations, and equality
of whitespace-normalized complete text. Output paths and timings are excluded from
equivalence. A pass applies only to the tested inputs. Configuration changes require a
separate quality profile even when sample outputs still match.

### 5.2 Workload Scaling

Use real input batches rather than sleeps to construct longer calls. For example, 80
references to a 2560 × 1920 image constitute 393,216,000 input pixels per OCR invocation.
Such repeated-input stress tests measure computational load, not independent task quality.

A ten-second duration is not guaranteed across devices. The demo records `over_10_sec`
for each call without padding fast executions. PDF page count, image resolution, and
operation repetition likewise do not imply monotonically increasing latency or difficulty.

### 5.3 Reporting Boundaries

For paper writing, Section 2 describes workload rationale and implemented operations;
Sections 3–5 define the measurement and validation protocol. Report empirical results
separately with input sizes, backend versions, device configuration, concurrency, and
metric provenance. In particular:

- Same-host logical replicas are not measurements on heterogeneous physical devices.
- Repeated fixtures are not independent task samples.
- Matching consistency samples do not establish universal functional equivalence.
- Changed Tool settings must not be treated as placement-only scheduler effects.
- Standalone Tool time does not replace Agent end-to-end latency or LLM queueing time.
- Timing metadata alone does not establish energy consumption or final-answer quality.

## 6. Setup and Execution

### 6.1 Dependencies

Install optional Python dependencies in the project environment:

```bash
uv pip install --python .venv/bin/python -r requirements-tools.txt
```

Install Tesseract separately on the execution node:

```bash
# macOS
brew install tesseract

# Ubuntu / Debian-based edge nodes
sudo apt-get install tesseract-ocr tesseract-ocr-eng
```

Other OCR languages require their language data. OCR/PDF dependencies are optional for
the existing image-only and profile-based execution paths.

### 6.2 Validation Commands

Run from the repository root:

```bash
python scripts/run_tool_demos.py
python scripts/run_tool_demos.py --batch-size 20 --scale large
python scripts/run_tool_demos.py --tools ocr --batch-size 80 --scale large
```

These commands run small-sample validation, larger batches, and an OCR stress workload,
respectively. They do not execute a live LLM Agent or compare multiple physical devices.

### 6.3 Experiment Artifacts

Each run creates a distinct directory under `data/tool_demos/` containing:

| Artifact | Contents |
| --- | --- |
| `trace.jsonl` | Existing timing trace records |
| `results.jsonl` | Calls, decisions, complete ToolResult metadata, and artifact references |
| `summary.json` | Exact inputs, consistency checks, elapsed times, and dependency skips |
| Per-replica text files | Complete extraction outputs |
| Batch manifests, when used | Ordered input references for each batch |

Retain raw results and full-text artifacts with their corresponding summaries.
