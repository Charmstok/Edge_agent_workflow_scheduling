# Milestone 4.4：真实 LLM 测量与离线导入

此流程复用 LLMExecutor，不部署新的模型服务。Linux、macOS 均使用同一 Python
采样入口；vLLM GPU 容器运行在 Linux 服务节点。Scheduler 接口保持不变。

## 已完成的验收

2026-09-11 对本机 `http://127.0.0.1:8000/v1` 的 `Qwen3.8-27B-FP8`
执行 18 次真实请求：文本字段提取、文本摘要、文本记录核对，各覆盖 2/16/48 条记录，
每种组合重复 2 次。18 次均正常结束。总输入 8400 token，总输出 952 token，
单次延迟约 0.15–8.31 秒。这里的成功指请求正常结束，不是答案正确率。
这些是文本探针，不等同于 4.1 的 image_ocr/pdf_extract/document_reconcile 工具工作流。

可追溯原始数据已保存到 `configs/llm_benchmarks/qwen38_27b_local_20260911.json`，
包含逐次提示词、参数、usage、原始响应、结果快照、模型与 endpoint 标识，以及运行 manifest。
实际调用日志还在 `data/llm_sampling/20260911T075826-ffa65d7b/`。
后续 Function Calling 验证取得了实际 Hugging Face revision
`017b9c7af6b5689d5dd426a76e0bc077eb5ca20a` 和 vLLM 镜像 digest；GPU 型号未写入这组
早先的吞吐测量 manifest。一次沙箱网络不可达的失败运行也保留在 data 中，不混入这组
成功测量数据。

`configs/llm_measured_and_synthetic_v1.json` 含两种可被现有 `load_llm_profiles`、
`ResourceRegistry` 和 `ProfileLLMExecutor` 加载的 profile：

- 27B 实测 profile：317.9515 **总 token/秒**，为 `sum(input+output)/sum(request time)`。
- 独立 synthetic 对照：64 总 token/秒，只是明确指定的逻辑对照，**不代表 9B 或任何云模型**。

请求耗时包含客户端、网络与服务端等待；不能把该 rate 当作 decode tokens/s，也不能再叠加
一次相同网络耗时。整体 rate 仅适用于已记录探针组合与串行负载；按任务、规模分组的延迟
保存在 summary 中。分桶预测、外推控制与留出误差验收属于 4.6/4.7。
质量 profile 和能耗 profile 为空，禁止直接用于要求这些指标的多目标实验。
测量 JSON 中未知 energy/cost/token 为 null；`result` 是观测快照，未知 energy 被置为 null，
因此不是可直接反序列化的旧版 LLMResult（旧 schema 的 energy 默认值为 0）。

## Linux 环境

在仓库根目录执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt -r requirements-tools.txt
export PYTHONPATH="$PWD/src"
python scripts/sample_llms.py --llm-id local-qwen38-27b
```

当前工作已创建 `.venv` 并安装这些依赖。主机的旧 macOS 虚拟环境不能直接搬到 Linux。
LLM 文本采样不依赖 Tesseract、Poppler、CUDA Python 包；只需能访问模型 HTTP 服务。
本地 loopback endpoint 自动绕过系统代理，避免工作站 SOCKS/HTTP 代理影响本地请求。
云端仍沿用系统代理配置；若使用 SOCKS，需要自行安装 httpx 的 socks 可选依赖。

## 四个部署入口

编辑 `configs/llm_profiles.toml`：

| llm_id | 请求模型名称 | 默认地址/状态 |
|---|---|---|
| local-qwen38-27b | Qwen3.8-27B-FP8 | 127.0.0.1:8000/v1；实测及自动 Function Calling 已验证 |
| local-qwen35-9b | Qwen3.5-9B | 127.0.0.1:8001/v1；enabled=false，待部署 |
| online-doubao | doubao-seed-2-0-lite-260428 | 火山引擎 Ark；需 ARK_API_KEY |
| online-ark-secondary | 由 ARK_SECONDARY_MODEL 指定 | 火山引擎 Ark；需 ARK_API_KEY |

可通过 `QWEN27B_BASE_URL`、`QWEN9B_BASE_URL` 覆盖本地地址，完整包含 `/v1`。
SDK 请求的 model 必须对应 `--served-model-name`，不是容器内模型缓存路径。
第二个云模型尚未指定，不编造可用模型 ID。配置中的云端上下文 8192 是保守实验上限，
不是服务商声明的最大窗口。

9B 部署示例（需先选择实际可用 GPU、确认显存与镜像支持；此命令未自动执行）：

```bash
# 先下载到实际的宿主机缓存；HF_HUB_OFFLINE=1 要求缓存完整。
hf download Qwen/Qwen3.5-9B --cache-dir /data/huggingface/hub

# GPU 1 仅为示例；不能默认与占用 0.92 显存比例的 27B 共用 GPU 0。
docker run -d \
  --name qwen35-9b \
  --restart unless-stopped \
  --gpus '"device=1"' \
  --ipc=host \
  -p 8001:8000 \
  -v /data/huggingface:/root/.cache/huggingface:ro \
  -e HF_HUB_OFFLINE=1 \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen3.5-9B \
  --served-model-name Qwen3.5-9B \
  --tensor-parallel-size 1 \
  --max-model-len 32768 \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.90
```

端口 8001 可用后将对应 `enabled` 改为 true。正式复现实验应记录镜像 digest 和权重 revision。
两条本地配置通过 chat_template_kwargs 禁用 thinking，并将该参数写入采样数据，避免隐藏推理
耗尽短输出预算。

27B 当前实际运行命令在原配置后增加了 Qwen 官方建议的解析参数，并锁定镜像 digest：

```bash
docker run -d \
  --name qwen38-27b-fp8 \
  --restart unless-stopped \
  --gpus '"device=0"' \
  --ipc=host \
  -p 8000:8000 \
  -v /data/huggingface:/root/.cache/huggingface:ro \
  -e HF_HUB_OFFLINE=1 \
  vllm/vllm-openai@sha256:c2914767605584b6d8f45686b82de173ecc99e781897aa3d0a66dacd72c51ae1 \
  --model Qwen/Qwen3.8-27B-FP8 \
  --served-model-name Qwen3.8-27B-FP8 \
  --tensor-parallel-size 1 \
  --max-model-len 65536 \
  --max-num-seqs 4 \
  --gpu-memory-utilization 0.92 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder
```

当前服务是 vLLM 0.29.0，模型 revision 为
`017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`。原容器以停止状态保留为
`qwen38-27b-fp8-no-tools-backup`，restart policy 为 `no`；新容器使用原名称并保持
`unless-stopped`。官方参数来源见
[Qwen3.8 README](https://github.com/QwenLM/Qwen3.8/blob/main/README.md) 和
[vLLM Tool Calling](https://docs.vllm.ai/en/stable/features/tool_calling/)。

### 27B 自动 Function Calling 验证

```bash
export PYTHONPATH="$PWD/src"
python scripts/verify_function_calling.py
```

脚本通过现有 `ToolRegistry.tools()` 将 `image_preprocess`、`ocr`、`pdf_parse`、
`pdf_render` 四个仓库 Tool schema 传给 Chat Completions API，并显式使用
`tool_choice=auto`。2026-09-11 的真实结果为：

- 图像任务：模型从四个 Tool 中选择 `image_preprocess`，生成合法参数；Scheduler 选择本地
  replica，Pillow 执行成功；结果按原 `call_id` 回传，第二轮 LLM 输出真实 artifact URI。
- 算术任务：同样提供四个 Tool，模型没有调用任何 Tool，直接输出 `42`。

精简证据位于
`configs/llm_function_calling/qwen38_27b_auto_tools_20260911.json`，完整 trace 位于
`data/function_calling_verification/20260911T082239Z/`。这验证的是自动结构化解析、选择一个
正确 Tool、真实 Tool 执行、结果回传和不调用 Tool 的决定；没有验证 OCR/PDF 实际执行、
并行 ToolCall、9B 或火山引擎 Function Calling。

`src/edge_agent_workflow_scheduling/tools` 提供 Tool 功能和 schema；它不会自动部署或自动注册
全部工具。Agent 只能在本次 `ToolRegistry` 已注册、资源表中有可调度 replica 的 Tool 之间选择。
验证脚本注册了全部四个 Tool 和各自本地 replica。现有 `run_agent_demos.py --mode online`
只注册 `image_preprocess`，用于单 Tool demo。

## 在线采样与预算

API key 只通过环境变量提供，不写入 TOML、命令行参数或测量文件。

```bash
# 在当前 shell 安全设置 ARK_API_KEY；此处省略凭据值。
export ARK_SECONDARY_MODEL='填写实际开通的模型ID'
python scripts/sample_llms.py --llm-id online-doubao --llm-id online-ark-secondary
```

`configs/llm_sampling_v1.json` 定义完整提示词矩阵。默认串行，每组合重复 2 次，
全局最多 72 请求、每请求最多 512 输出 token、全局输出 token 预留预算 36864、每请求 120 秒。
每次发送前全额预留输出上限，失败不退款，SDK 不自动重试；输入是固定有限提示词集合。
这是输出 token 加请求数预算，不是货币预算或输入加输出的硬 token 上限；收费数据缺失时
保持 null。更改任务长度/重复次数前同步检查预算。默认 72 次足够覆盖四模型完整矩阵。

未部署、缺少 key 或第二模型 ID 时，summary 明确 skipped，调用数为 0；有请求但全部失败
则为 failed。逐条 JSONL 立即 flush，失败、截断与 usage 缺失均保留。生成 rate 时排除失败、
缺失 token 或非正耗时，记录排除规则和数量；其余统计保留失败样本。
请求超时无响应时无法获得服务端实际 token 用量，记录 null。

## 无凭据离线导入

```bash
python scripts/sample_llms.py \
  --import-benchmark configs/llm_benchmarks/qwen38_27b_local_20260911.json \
  --output-dir data/llm_import_new
```

目标目录必须不存在，禁止覆盖原始数据。导入不联网、不请求模型，输出原始 benchmark、
manifest、summary 和 profiles.json。新的外部数据使用同一 JSON 交换格式：

- `provenance`：source、version、task_definition、scoring_rule、timing_scope 必填；
  hardware、runtime_config 必须存在，未知可为 null；并发必须为 1。
- `llm_instances`：现有 LLMInstanceProfile 列表。
- `samples`：llm_id、sample_id、task_type、input_size、success；input_tokens、output_tokens、
  request_time_sec 缺失保持未知，其余原始字段原样保留。

目前只有 `client_request_including_network_and_server_wait` 耗时口径可转换为现有 rate；
只有解码速度、TTFT 或评分的数据不可冒充整个请求耗时。评分数据不自动映射 quality_profile，
文本 benchmark 不会赋予 function_calling 能力。不同任务、硬件、采样参数和评分方法的数据
不能直接比较质量；当前 synthetic 对照仅用于执行器联通验收。

## 验证范围

新增测试覆盖协议转换、多工具 call_id、截断 usage、超时、异常脱敏、失败预算、缺失凭据、
缺失 benchmark 字段、吞吐量口径及两 profile 的加载与执行。运行：

```bash
python -m pytest -q
```

未验证：9B 实际部署、两个云模型调用、OCR/PDF 的真实自动选择与多步调用、并行 ToolCall、
答案质量、能耗、独立留出预测误差。27B 的自动 Function Calling 与本地
`image_preprocess` 完整闭环已通过。
