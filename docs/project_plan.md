# 项目规划：Edge Agent Workflow Scheduling

## 1. 研究目标

本项目研究 Agent 工作流背景下的异构边缘资源调度问题：

```text
多个 Agent 持续产生由 LLM 推理和 Tool 调用构成的动态调用轨迹；
系统中存在多个不同模型、不同参数量、不同运行平台的 LLM 实例；
同一种 Tool 可以在多个边缘节点上部署为不同副本；
调度器需要为 Agent 运行过程中产生的每个 LLMCall/ToolCall 选择执行目标；
最终联合优化 Agent 端到端延迟、模型质量、能耗和负载均衡。
```

项目重点是调度建模、算法和实验，不是构建生产级 Agent 平台、通用边缘服务框架，也不是从零实现 OCR、PDF 解析、PDF 页面渲染或图像处理算法。

## 2. 核心研究问题

### 2.1 联合调度对象

调度器同时处理两类调用：

```text
LLMCall:
  在多个 LLMInstance 中选择一个实例；
  不同实例具有不同模型规模、吞吐量、质量、能耗和队列状态。

ToolCall:
  在同一 tool_name 的多个 ToolReplica 中选择一个副本；
  不同副本运行在不同节点上，具有不同延迟、能耗、并发能力和队列状态。
```

### 2.2 AgentRun 与动态 Function Calling

本项目使用 OpenAI Python SDK 和 Responses API Function Calling 实现真实 Agent。 Agent 不预先声明完整 Workflow DAG；下一次 Tool 调用由当前 LLM 响应动态决定：

```text
用户输入
  -> LLMCall
  -> 普通回答：AgentRun 完成
  -> 一个或多个 function_call
       -> 转换为 ToolCall
       -> Scheduler 为每个 ToolCall 选择 ToolReplica
       -> 收集 function_call_output
       -> 下一次 LLMCall
```

核心运行对象：

- `AgentRun`：一次用户请求从输入到最终回答的完整运行，保存规范化对话状态和端到端指标。
- `LLMCall`：Agent 的一次模型请求，由 Scheduler 选择 `LLMInstance`。
- `ToolCall`：由模型 `function_call` 产生的一次工具请求，由 Scheduler 选择同类型的 `ToolReplica`。
- `SchedulableCall = LLMCall | ToolCall`：Scheduler 接收的统一类型别名，不额外定义复杂的 `WorkflowStep` 实体。

`AgentRun` 使用最小运行状态：

```text
CREATED
  -> READY_FOR_LLM
  -> WAITING_FOR_LLM
       -> WAITING_FOR_TOOLS
       -> READY_FOR_LLM
       -> COMPLETED

任意未结束状态 -> FAILED
```

单个调用使用 `CREATED -> QUEUED -> RUNNING -> SUCCEEDED/FAILED`。一个 LLM 响应可以产生多个 ToolCall；这些 ToolCall 可以并行调度，但必须收集本轮全部 `function_call_output` 后才能开始下一次 LLMCall。由此形成的依赖关系是运行时动态调用轨迹，不要求预先进行 DAG 构造或完整 DAG 校验。

### 2.3 异构 LLM

每个 LLMInstance 至少包含：

```text
llm_id
model_name
model_size_b
platform
supported_task_types
tokens_per_sec
quality_profile
energy_per_token
queue_len
running_requests
max_concurrency
is_online
```

模型质量不应只用参数量代替。实验中使用任务类型相关的经验质量：

```text
quality(model, task_type)
```

该质量可以来自离线 benchmark、真实任务成功率或预先给定的实验 profile。

### 2.4 Tool 与 ToolReplica

`Tool` 表示功能定义，`ToolReplica` 表示该 Tool 在某个节点上的一个部署副本：

```text
tool_name = ocr

replicas:
  ocr@macbook
  ocr@ubuntu
  ocr@rpi_1
```

每个 ToolReplica 至少包含：

```text
replica_id
tool_name
node_id
platform
latency_profile
energy_profile
queue_len
running_tasks
max_concurrency
is_online
```

Scheduler 根据 `tool_name` 过滤可执行副本，再根据目标函数选择具体副本。

### 2.5 真实执行与 Profile 模拟

项目采用混合实验方式，而不是完全模拟：

```text
Real execution:
  在当前机器或真实边缘设备上实际执行 Tool/LLM；
  记录延迟、资源利用率、能耗或代理指标；
  用于校准 profile 和最终实验。

Profile execution:
  对暂时没有的设备、模型或 Tool 副本使用 profile 模拟；
  用于算法开发、RL 训练和大规模可重复实验。
```

真实执行和 profile 模拟必须通过相同的 Executor Protocol 返回结构一致的 `LLMResult` 或 `ToolResult`。

## 3. 多目标优化模型

### 3.1 优化目标

第一版保留以下目标：

```text
minimize:
  Agent 端到端延迟
  deadline miss
  能耗
  负载不均衡

maximize:
  LLM 任务质量
```

训练阶段可以先使用可配置的加权标量化：

```text
cost =
  α * latency
  + β * energy
  + γ * deadline_miss
  + δ * load_imbalance
  - η * quality
```

实验阶段必须报告原始目标向量，并通过多组权重或约束设置绘制 Pareto frontier，不能只报告一个加权总分。

### 3.2 硬约束

以下条件优先建模为约束，而不是软惩罚：

- 当前调用由有效的 AgentRun 产生且处于可调度状态
- 下一次 LLMCall 只能在本轮所需 ToolCall 全部返回后产生
- 目标支持当前 call 类型
- ToolReplica 的 `tool_name` 匹配
- LLM/Tool 内存和并发容量满足要求
- 目标在线
- 敏感数据不离开允许的节点域
- 需要时满足最低模型质量

### 3.3 调度动作

```text
LLMCall:
  action = selected_llm_instance_id

ToolCall:
  action = selected_tool_replica_id
```

不可执行目标使用 action mask 排除。

## 4. 最小系统架构

```text
AgentRunner
  -> 产生 LLMCall
  -> Call Queue -> Scheduler -> LLMExecutor
  -> 解析模型响应
       ├── 最终回答 -> AgentRun 完成
       └── function_call -> 产生一个或多个 ToolCall
                              -> Call Queue -> Scheduler -> ToolExecutor
                              -> function_call_output -> AgentRunner

Executor:
  Local Real Executor / Profile Executor / Remote Executor（获得真实设备后）

所有调用与结果 -> Trace Logger
```

核心模块：

```text
1. AgentRunner 与 Function Calling loop
2. AgentRun、LLMCall、ToolCall schema
3. LLMInstance / ToolReplica profile
4. Call queue
5. Baseline / RL Scheduler
6. LLM/Tool Executor adapter
7. Trace、replay 与 evaluator
```

第一阶段不引入 Redis、RabbitMQ、数据库、服务发现或生产级容错。

## 5. 实验方法

### 5.1 开发阶段

在一台笔记本上完成：

- 多个逻辑 LLMInstance
- 同一 Tool 的多个逻辑 ToolReplica
- 至少一个真实本地 Tool
- 其他目标的 profile 模拟
- Baseline 和 RL 策略开发
- 可重复 workload 与 trace

### 5.2 校准阶段

对当前可用机器执行真实 Tool/LLM，采集：

- 输入规模
- 排队时间
- 执行时间
- CPU/GPU/内存利用率
- token throughput
- 能耗或功耗代理指标
- 成功率和质量指标

用采集结果拟合不同任务类型和执行目标的 profile。

### 5.3 真实设备阶段

获得 Ubuntu、Raspberry Pi 或其他边缘设备后：

- 部署已有 Tool 或 LLM adapter
- 复用同一 AgentRun、Call、Scheduler 和 Trace 协议
- 更新对应 ToolReplica/LLMInstance profile
- 进行真实异构设备实验

没有真实设备不阻塞算法开发，但不能将单机逻辑副本实验描述为真实异构硬件结果。

### 5.4 Agent 随机性与实验可重复性

Function Calling Agent 可能在相同任务上产生不同数量、不同顺序的 ToolCall。固定模型名称和提示词也不能保证云端模型逐 token 确定性，因此项目区分两种实验模式：

```text
Trace Replay:
  固定 Agent 产生的 LLMCall/ToolCall 轨迹；
  不重新决定调用哪些 Tool；
  用相同调用集合比较 Scheduler 和副本选择策略。

Live Agent:
  允许 LLM 动态选择 Tool；
  对相同任务重复运行多次；
  将 Tool 调用数量、调用成功率和最终任务质量作为实验结果的一部分。
```

实验配置必须保存为 `ExperimentManifest`，至少记录：

- task 数据集与样本 ID
- system prompt、用户输入模板及版本
- Tool schema、顺序及实现版本
- 模型和 endpoint 标识
- API 暴露的采样参数，以及提供方支持时使用的随机种子
- Agent 的最大轮数、最大 Tool 调用数和超时
- Scheduler 名称、参数及随机种子
- LLMInstance/ToolReplica profile 版本
- 代码版本、运行时间和执行模式（`replay` 或 `live`）

公平比较规则：

1. 比较纯调度策略时使用同一份 replay trace，使所有策略面对完全相同的调用。
2. 比较不同 LLM 选择策略或完整 Agent 系统时使用 live 模式；Tool 选择差异属于策略效果，不能强行抹平。
3. live 模式对每个任务和策略运行多次，报告均值、分位数、方差或置信区间，不能只报告单次结果。
4. 保存每次模型原始输出、ToolCall、参数、执行结果和调度决策，以便复查异常运行。
5. 同一 `tool_name` 的副本必须通过相同输入的功能一致性检查；如果实现版本造成输出质量差异，则将质量作为显式 profile 和实验变量，不能只把它视为部署副本差异。

## 6. 实现里程碑

### Milestone 1：最小混合执行闭环 [已完成]

目标：

```text
在单机上跑通 LLMCall 和 ToolCall 的统一调度；
LLM 使用 mock runtime；
至少一个 Tool 在 LocalWorker 中真实执行；
所有结果写入 trace。
```

执行流程：

```text
SimulatedAgent
  -> LLMCall / ToolCall
  -> InMemoryCallQueue
  -> BaselineScheduler
  -> MockLLMRuntime / LocalWorker
  -> LLMResult / ToolResult
  -> JSONL Trace
```

保留的实现：

- 核心 call、result、state 和 trace schema
- `SimulatedAgent`
- 统一内存队列
- random、round-robin、least-queue 和 earliest-finish-time baseline
- `MockLLMRuntime`
- `LocalWorker`
- `Tool` Protocol 和 `ToolRegistry`
- 使用成熟图像库的真实 `ImagePreprocessTool`
- 本地端到端 demo

验收标准：

```text
1. 一个命令可以运行本地端到端 demo
2. 同时产生 LLM trace 和真实 Tool trace
3. 至少两个逻辑 LLM 实例和两个逻辑 Worker 参与调度
4. ToolResult 记录真实执行时间
5. Trace 可以作为后续 AgentRun、replay 和 RL 开发输入
```

### Milestone 2：真实 Function Calling Agent 与异构资源模型

目标：

```text
使用 OpenAI Python SDK 和 Responses API 跑通真实 Function Calling Agent；
将 Agent 动态产生的每次 LLMCall 和 ToolCall 接入 Scheduler；
建立可在笔记本上运行的 LLMInstance、ToolReplica 和 real/profile Executor；
同时支持固定 trace 的可重复调度实验和动态 Agent 的端到端实验。
```

本里程碑不预先构造 Workflow DAG，不实现通用 Agent 框架，也不要求已有 Raspberry Pi 或 Ubuntu 节点。

#### 2.1 定义 AgentRun 与通用 Call schema [已完成]

实现：

1. 定义 `AgentRunStatus`：`CREATED`、`READY_FOR_LLM`、 `WAITING_FOR_LLM`、`WAITING_FOR_TOOLS`、`COMPLETED`、`FAILED`。
2. 定义 `CallStatus`：`CREATED`、`QUEUED`、`RUNNING`、 `SUCCEEDED`、`FAILED`。
3. 定义 `AgentRun`，至少包含 `run_id`、`agent_id`、`task_id`、 `status`、`turn_index`、开始/结束时间、最终输出和错误信息。
4. 为 `LLMCall` 增加 `run_id`、`turn_index` 和模型能力约束；模型名称只在实验要求固定模型时作为硬约束。
5. 将 `ToolCall` 改为通用 Function Calling 请求，至少包含 `run_id`、 OpenAI `call_id`、`tool_name` 和 JSON `arguments`。
6. 将 `input_uri`、`page_count`、`image_count` 等图像专用字段移入对应 Tool 的 arguments 或 profile metadata。
7. 将 Scheduler 使用的类型别名统一为 `SchedulableCall = LLMCall | ToolCall`。
8. 直接迁移当前 schema 和调用方，不增加旧版 JSON 兼容层。

完成检查：

- 可以构造、校验和 JSON 序列化/反序列化 `AgentRun`、`LLMCall` 和 `ToolCall`。
- 非法状态转换、空 `run_id`、空 `call_id` 和非对象 arguments 会被拒绝。
- 现有 ImagePreprocessTool 通过新的通用 arguments 接口仍能真实执行。

#### 2.2 对齐 OpenAI Function Tool 与 ToolRegistry [已完成]

实现：

1. 保留 `ToolSpec` 的 `type`、`name`、`description`、 `parameters` 和 `strict` 字段。
2. 为 Tool 定义通用执行入口：接收解析后的 arguments，返回可 JSON 序列化的结果和执行 metadata。
3. 让 `ToolRegistry.tools()` 直接生成传给 Responses API 的 tools 列表。
4. 增加 function name 到本地 Tool 实现的查找，并明确“Tool 功能类型”和 “Tool 部署副本”不是同一个对象。
5. 在执行前校验 function arguments；未知 Tool、非法 JSON 和参数不匹配返回结构化失败结果。
6. 定义 Tool 结果到 `function_call_output` 字符串的唯一序列化规则。

完成检查：

- Registry 中的 Tool schema 可以直接作为 Responses API 的 `tools` 参数。
- 给定 `name + arguments` 可以找到功能实现，但不会提前绑定执行副本。
- 成功和失败结果都可以转换为带原始 `call_id` 的 `function_call_output`。

#### 2.3 实现可注入执行后端的 FunctionCallingAgent [已完成]

实现：

1. 新增 `FunctionCallingAgent`，输入包括 system instruction、用户任务、 ToolRegistry、最大轮数、最大 ToolCall 数和超时。
2. Agent 自己维护平台无关的规范化对话记录，不把 `previous_response_id` 作为唯一状态来源，以支持不同 LLM 平台切换。
3. 每轮先创建 `LLMCall`，通过注入的 LLM 调用接口获得 Responses API 风格的输出。
4. 将本轮模型输出完整加入对话状态，并识别全部 `function_call` item。
5. 没有 function_call 时保存最终文本并将 AgentRun 标记为 `COMPLETED`。
6. 有 function_call 时，为每个 item 创建保留原始 `call_id` 的 `ToolCall`。
7. 收齐本轮全部 Tool 结果后，按模型输出中的稳定顺序追加 `function_call_output`，再发起下一次 LLMCall。
8. 对达到轮数/调用数限制、LLM 错误、Tool 错误和超时定义明确的停止策略。
9. 实现一个 scripted LLM backend，确定性地产生“单 Tool”“多 Tool”和 “最终回答”，用于无 API key 的离线验证。

完成检查：

- 离线运行可以完成 `LLM -> Tool -> LLM -> final answer`。
- 一轮返回多个 function_call 时，每个 call_id 都得到且只得到一个 output。
- 最大轮数和错误路径不会形成无限循环。

#### 2.4 建立异构 LLMInstance 与 ToolReplica profile [已完成]

实现：

1. 定义静态 `LLMInstanceProfile`，至少包含 `llm_id`、provider、 model、base URL、platform、模型规模、能力、上下文限制、质量 profile、 token/能耗 profile、并发上限和 executor 类型。
2. 保留 `LLMInstanceState` 表示 queue、running、利用率、实测吞吐量和 online 状态，不把动态状态重复写入静态 profile。
3. 定义静态 `ToolReplicaProfile`，至少包含 `replica_id`、 `tool_name`、`node_id`、platform、实现版本、延迟/能耗 profile、并发上限和 executor 类型。
4. 定义 `ToolReplicaState` 表示 queue、running、资源利用率、失败率和 online 状态。
5. 允许同一 `tool_name` 注册多个 replica，并根据 Tool 和平台能力生成 action mask。
6. 平台特定依赖保存在 replica 的部署配置中；核心 Tool arguments 和结果 schema 在 macOS、Ubuntu 和 Raspberry Pi 间保持一致。
7. 为同一 Tool 的副本准备共享一致性样例；实现存在精度差异时，在 profile 中记录实现版本和质量指标。
8. API key 等秘密只通过环境变量读取，不写入 profile、trace 或仓库。

完成检查：

- 单机配置中可以声明至少两个逻辑 LLMInstance 和同一 Tool 的两个 ToolReplica。
- 一个 replica 离线或能力不匹配时会被 action mask 排除。
- 新增平台只需增加 profile/executor 配置，不修改 Scheduler 接口。

#### 2.5 实现 LLM 与 Tool Executor adapter [已完成]

实现：

1. 定义 `LLMExecutor` 和 `ToolExecutor` Protocol；两者分别返回 `LLMResult` 和 `ToolResult`，共同携带排队、传输、执行时间及错误。
2. 用 adapter 保留现有 `MockLLMRuntime` 和 `LocalWorker`。
3. 实现 `OpenAIResponsesExecutor`，使用注入的 OpenAI client、 instance profile 和模型参数执行真实 LLMCall；不固定模型提供商，兼容 Responses API 的在线控制台通过各自的 base URL 和环境变量密钥接入。
4. 实现 `LocalToolExecutor`，在当前进程真实运行已有 Tool。
5. 实现 `ProfileLLMExecutor` 和 `ProfileToolExecutor`，按固定 profile 或带种子的分布生成延迟、能耗和结果。
6. executor 选择由目标 profile 的 `executor_type` 决定，Scheduler 不感知模型提供商、操作系统和真实/profile 差异；非 Responses API 后端只需新增一个实现相同 Protocol 的 adapter。

完成检查：

- scripted/profile LLM、真实 OpenAI LLM、本地真实 Tool 和 profile Tool 通过各自统一接口调用。
- 没有 OpenAI API key 时，离线路径仍可完整运行；在线 demo 明确跳过并说明原因。
- 相同结果 schema 足以供 Scheduler、AgentRunner 和 trace 使用。

#### 2.6 接通 Agent、Queue、Scheduler 与 Executor [已完成]

实现：

1. AgentRunner 创建 LLMCall 后，将其放入 Call Queue。
2. Scheduler 根据能力约束和 action mask 选择 LLMInstance，再调用对应 LLMExecutor。
3. AgentRunner 将模型 function_call 转换为 ToolCall 并放入 Call Queue。
4. Scheduler 先按 `tool_name` 过滤副本，再选择具体 ToolReplica。
5. 同一轮多个 ToolCall 可以并发执行；完成顺序不改变回传给模型的稳定顺序。
6. 每次入队、开始和完成时更新 instance/replica 动态状态。
7. 每个调用保存选中目标和调度理由，并关联同一个 `run_id`。
8. 当策略选择不同 LLM 平台时，由 executor 将规范化对话转换为对应请求格式。

完成检查：

- 一次 AgentRun 中至少发生一次 LLM 调度、一次 ToolReplica 调度和一次后续 LLM 调度。
- 两个同功能 ToolReplica 可以被独立选择。
- Scheduler 不会收到模型尚未产生的“未来步骤”。

#### 2.7 实现 Trace、ExperimentManifest 与 Replay [已完成]

实现：

1. 扩展 trace，记录 `run_id`、`turn_index`、`call_id`、调用类型、选中目标、调度策略、时间、成功状态、模型/Tool 标识及参数摘要。
2. 记录 AgentRun 开始、状态转换、最终输出、总轮数、ToolCall 总数和端到端延迟。
3. 保存模型原始 response item 和 function_call_output；敏感参数使用脱敏值或内容哈希加受控 artifact 引用。
4. 定义 `ExperimentManifest` 并记录第 5.4 节列出的复现信息。
5. 实现 trace replay loader，将已记录的 SchedulableCall 按原顺序重新送入 Scheduler，而不重新调用 LLM 决定 Tool。
6. replay 时校验调用 ID、Tool 名称和 arguments 摘要与原 trace 一致。

完成检查：

- 同一 replay trace 使用不同调度策略时，输入调用集合完全相同，但选择的 LLMInstance/ToolReplica 可以不同。
- 可以从 trace 计算单次调用指标和 AgentRun 端到端指标。
- 同一 manifest 和带种子的 profile 配置可以复现实验输入。

#### 2.8 提供离线、在线与 Replay demo [已完成]

实现：

1. 离线 demo：scripted LLM 产生 ToolCall，至少一个 ToolReplica 使用真实 ImagePreprocessTool，其他 LLM/Tool 目标使用 profile。
2. 多 Tool demo：单轮产生至少两个 function_call，验证并行调度和结果汇合。
3. 在线 demo：存在 `OPENAI_API_KEY` 时使用 OpenAI Responses API 完成至少一次真实 Function Calling；不存在时不影响离线验收。
4. replay demo：读取一次固定 trace，分别运行至少两种 baseline policy。
5. 所有 demo 输出 manifest、call trace 和 AgentRun 汇总，不引入服务端、数据库或远程节点。

里程碑验收标准：

```text
1. 项目实现真实的动态 Function Calling Agent loop，而不是预声明 DAG
2. Agent 可以完成 LLM -> Tool -> LLM -> 最终回答
3. 每次 LLMCall 都可以在多个 LLMInstance 中选择执行目标
4. 每次 ToolCall 都可以在同一 Tool 的多个 ToolReplica 中选择副本
5. 多 ToolCall 可以并发执行并正确回传原始 call_id
6. 真实执行和 profile 执行使用相同的调用/结果协议
7. 无边缘设备、无 API key 时可以完成全部离线调度开发和验证
8. 有 API key 时可以完成至少一次真实 OpenAI Function Calling
9. replay 模式可以公平、可重复地比较调度策略
10. live 模式通过重复实验报告 Agent 动态 Tool 选择带来的分布
```

### Milestone 3：多目标 Baseline Scheduler

目标：

```text
在 Milestone 2 的统一 Call、Resource、Executor 和 Replay 接口上，
建立可解释、可重复的多目标 baseline；
明确区分“调度时的 profile 估计值”和“执行后的 trace 实测值”；
为 Milestone 5 的 RL 环境提供一致的约束、目标函数和实验对照。
```

本里程碑只实现研究实验需要的轻量策略、指标聚合和批量运行入口。复用现有 `BaselineScheduler`、`ResourceRegistry`、profile executor 和 replay，不引入通用优化求解器、数据库、分布式控制面或生产级监控系统。Milestone 4 才负责扩大 workload 并用更多真实测量校准 profile；本阶段可以使用版本固定的合成 profile 验证算法和实验流程。

#### 3.1 统一多目标指标与估计口径 [已完成]

实现：

1. 为每个“调用-候选目标”计算轻量目标向量：预计完成时间 `latency_sec`、预计能耗 `energy_joules`、预计质量 `quality`、预计 `deadline_miss` 和分配后的 `load_imbalance`。
2. LLM 完成时间继续使用输入/预计输出 token、`tokens_per_sec`、队列长度和实测平均延迟估计；Tool 完成时间使用网络延迟、执行时间 profile 和队列长度估计，不另建复杂性能模型。
3. LLM 能耗按预计总 token 数乘 `joules_per_token` 估计；Tool 能耗第一版使用 `joules_per_call`。后续校准只更新 profile，不改变策略接口。
4. 任务类型从 call metadata 中的 `task_type` 读取，未提供时使用 `default`； LLM 质量读取 `quality_profile[task_type]`。仅当 Tool 副本存在显式质量差异时使用其 quality profile，否则通过 Milestone 2 的功能一致性检查后视为等价副本。
5. 有 deadline 时，预计完成时间超过 deadline 则预计 `deadline_miss = 1`，否则为 0；未设置 deadline 时该项为 0。决策阶段先按并发容量归一化候选分配后的 queue/running 负载，再在同类可行目标间计算 `load_imbalance`。
6. 调度分数使用 profile 和当前资源状态的估计值；实验报告使用执行结果和 trace 的实测延迟、实测能耗及选中目标的质量 profile，不能用预测值冒充实测结果。
7. 采用固定、可配置的归一化尺度计算加权 cost：

   ```text
   normalized_cost =
     alpha * latency_sec / latency_ref_sec
     + beta * energy_joules / energy_ref_joules
     + gamma * deadline_miss
     + delta * load_imbalance
     + eta * (1 - quality)
   ```

权重必须非负并归一化为和为 1；`latency_ref_sec` 和 `energy_ref_joules` 必须为正，并在比较所有策略前由同一 workload/profile 确定并写入 manifest，禁止按单个策略的结果分别归一化。
8. 若某策略需要的 LLM 质量、非等价 Tool 质量或能耗 profile 缺失，则在实验配置检查阶段明确失败；不把缺失值静默当作 0，也不临时根据模型参数量推断质量。

完成检查：

- 同一个 call 和资源快照总能得到字段、单位和方向一致的目标向量。
- 构造速度、能耗和质量互有优劣的候选时，各指标估计符合手工计算结果。
- 改变策略不会改变同一批实验使用的归一化尺度。
- trace 汇总明确标注 estimated/profiled 与 measured，二者不会混为一个字段。

#### 3.2 集中处理硬约束与 Action Mask [已完成]

实现：

1. 在策略执行前集中生成可行候选和 action mask，继续复用 `ResourceRegistry` 已有的调用类型、`tool_name`、模型名称、能力、上下文长度、在线状态和并发容量检查。
2. 增加 baseline 实验需要的最低质量约束；需要限制数据位置时，从实验配置或 call metadata 读取允许的 `node_id` 集合，不为单个实验场景扩展新的权限子系统。
3. AgentRun 状态和 Tool 依赖仍由 `AgentRunner` 在产生调用前检查；action mask 只负责“当前调用能否在当前目标执行”，不重复实现 Agent 状态机。
4. 所有 baseline 接收相同的、已经过滤的 `SchedulingCandidate` 列表；各策略不得自行绕过 action mask，也不得重复一套能力匹配逻辑。
5. 使用稳定的目标 ID 顺序输出布尔 action mask，保证 baseline、后续 RL 环境和 trace 中的动作索引含义一致。
6. 没有可行候选时返回结构化调度失败并记录失败原因，不自动放宽最低质量、数据域或容量约束。

完成检查：

- 离线、容量已满、能力不匹配、Tool 名称不匹配和低于质量阈值的目标均被排除。
- 任意 baseline 的选择结果都属于 action mask 标记为可行的候选。
- 相同 call、资源快照和约束配置在不同策略下得到完全相同的 action mask。
- 无可行候选时实验可终止并定位具体约束原因，而不是产生非法执行动作。

#### 3.3 完成单目标启发式 Baseline [已完成]

实现：

1. 保留并复用已经实现的 `random`、`round_robin`、`least_queue` 和 `earliest_finish_time`，只补充统一配置、种子和指标记录，不重写现有策略。
2. 实现 `quality_aware`：选择预计质量最高的候选；质量相同时依次按预计完成时间和目标 ID 打破平局。
3. 实现 `energy_aware`：选择预计能耗最低的候选；能耗相同时依次按预计完成时间和目标 ID 打破平局。
4. `random` 使用 manifest 中记录的 scheduler seed；其余策略使用确定性的稳定 tie-break，避免候选注册顺序改变实验结果。
5. 每次决策记录策略名、选中目标、主要 score、目标向量和简短 reason，便于解释 baseline 行为，但不保存冗长的逐候选调试日志。

完成检查：

- 构造明确的候选 profile 时，quality-aware 和 energy-aware 分别选择预期目标。
- 两个候选主指标相同时，重复运行得到相同结果。
- 固定 scheduler seed 后 random 的完整选择序列可以复现。
- 既有四种 baseline 的行为和 Milestone 2 demo 保持兼容。

#### 3.4 实现加权与质量约束策略 [已完成]

实现：

1. 实现 `weighted_objective`，对每个可行候选计算第 3.1 节的 `normalized_cost`，选择 cost 最小的目标。
2. 权重、归一化尺度和使用的 profile 版本通过实验配置注入并写入 `ExperimentManifest.scheduler_parameters`，不硬编码在策略类中。
3. 实现 `quality_constrained_earliest_finish_time`：将最低质量阈值交给第 3.2 节的统一候选过滤，再对返回的可行候选使用 earliest-finish-time 选择。
4. 最低质量属于硬约束；全部候选不满足时返回“无可行目标”，不退化成选择质量最高但仍不达标的目标。
5. LLMCall 和 ToolCall 共用目标向量与策略接口，但各自通过对应 profile 估计指标；不为两类调用复制两套 Scheduler。

完成检查：

- 单独提高 latency、energy 或 quality 权重时，策略选择会朝对应目标变化。
- 将加权策略设为单一非零权重时，其选择与对应单目标策略在无平局时一致。
- quality-constrained EFT 的所有决策均满足阈值，且在可行集合内具有最小预计完成时间。
- 非法权重、非正归一化尺度和缺失 profile 在运行前被拒绝。

#### 3.5 实现 Trace 指标聚合与实验 Evaluator [已完成]

实现：

1. 基于现有 `TraceBundle` 增加轻量 evaluator，输出 call、AgentRun 和整个实验三个层级的指标；原始 trace 保持为事实来源，不建立额外数据库。
2. call 层记录实际总延迟、排队/传输/执行时间、能耗、deadline miss、成功状态、选中目标及该决策对应的 profiled quality。
3. AgentRun 层记录端到端延迟、总能耗、LLM/Tool 调用数、成功状态和质量摘要；在 Milestone 4 引入真实任务评分前，质量明确报告为 selected-profile quality，不声称是最终回答的真实准确率。
4. 实验层至少报告平均、P95 和 P99 Agent 延迟、deadline miss rate、总能耗、平均质量、成功率、吞吐量、各目标选择次数和负载不均衡。
5. 负载不均衡使用容量归一化工作量计算：目标 `i` 的负载为其累计执行时间除以 `max_concurrency_i`；分别在 LLM 集合及各 `tool_name` 的副本集合内计算变异系数，再对存在至少两个候选的集合取平均。空集合或所有负载均为 0 时记为 0。
6. 同时输出原始目标向量和使用固定尺度计算的 weighted cost；JSON 用于保存完整配置与结果，CSV 用于后续统计和绘图。

完成检查：

- evaluator 可以直接读取 Milestone 2 生成的 trace，不重新执行 Agent 或修改 trace。
- 一个小型手算 trace 的延迟、能耗、deadline miss、质量和负载不均衡聚合结果正确。
- 报告保留原始单位，不只输出一个加权总分。
- profile 质量与后续真实任务质量在字段名和报告说明中可以明确区分。

#### 3.6 提供可重复的 Baseline 比较实验 [已完成]

实现：

1. 提供一个批量实验入口，接收固定 replay trace、资源 profile、策略列表、权重、归一化尺度和随机种子；不为每个策略各写一个脚本。
2. 比较纯调度策略时复用完全相同的 Call 集合、顺序和到达信息，并在每个策略开始前重置资源状态和带随机性的 profile executor。
3. 每个实验生成 manifest、决策/执行 trace、逐策略 summary 和合并后的 CSV；输出目录包含 workload/profile 版本与实验 ID，避免不同实验结果互相覆盖。
4. 固定 profile 模式作为主要的快速回归路径；live Agent 只用于确认策略可接入真实 Function Calling loop，不能与固定 replay 的结果混在同一组公平性结论中。
5. 对确定性策略至少运行一次；对 random 或带随机分布的 profile 使用多组 seed，报告均值和离散程度。
6. 记录 Scheduler 决策耗时，确认 baseline 本身的计算成本相对执行时间可忽略。

完成检查：

- 一个命令可以在同一固定 workload/profile 上运行全部 baseline。
- 同一 manifest 和 seed 重复执行时，输入、决策序列和 profile 结果一致。
- 所有策略处理的 call ID 集合一致；失败调用不会被静默删除。
- live 与 replay/profile 结果写入不同实验目录并带有明确 mode 标识。

#### 3.7 扫描权重并生成 Pareto Frontier [已完成]

实现：

1. 准备一组版本化的代表性权重，包括各目标的单目标端点、等权重点，以及少量 latency-quality、latency-energy 和 quality-energy 折中点；第一版使用显式列表或粗粒度 simplex 网格，不引入超参数优化框架。
2. 对每组权重运行 `weighted_objective`，并把 random、round-robin、least-queue、 EFT、quality-aware、energy-aware 和 quality-constrained EFT 作为参照点。
3. 使用原始实验指标进行非支配判断：最小化 latency、energy、deadline miss 和 load imbalance，最大化 quality；weighted cost 只用于策略决策和辅助排序，不用于判定 Pareto 支配关系。
4. 输出全部实验点和 `is_pareto` 标记的 CSV/JSON；至少生成 latency-energy、 latency-quality 和 energy-quality 三组二维数据，其他目标作为列保留。
5. Pareto 计算使用简单的两两非支配比较即可；当前实验规模较小，不实现复杂的多目标优化库或可视化服务。

完成检查：

- 人工构造的支配点会被排除，互有优劣的点会被保留。
- 每个 Pareto 点都能追溯到唯一 experiment ID、manifest、策略参数和原始 summary。
- 至少一组固定 replay/profile workload 产生可重复的 baseline Pareto frontier 数据。
- 改变权重能够产生至少两个不同的原始目标向量；若没有变化，实验报告需说明是 profile 缺少目标冲突，而不能把重复点解释为有效权衡。

里程碑验收标准：

```text
1. 既有四种 baseline 保持可用，并新增 quality-aware、energy-aware、
   weighted-objective 和 quality-constrained earliest-finish-time
2. 所有策略共用同一可行候选过滤和 action mask，非法目标不会进入 Executor
3. 同一 workload、profile、manifest 和 seed 下的实验可以重复
4. 调度使用的估计指标与 trace 汇总的实测指标被明确区分
5. 结果同时报告 latency、deadline miss、energy、quality、load imbalance
   原始向量和 weighted cost
6. quality profile 缺失或最低质量约束不可满足时明确失败，不静默降级
7. random 和带随机 profile 的结果通过多 seed 报告统计量
8. 至少生成一组可追溯、可重复的 baseline Pareto frontier CSV/JSON 数据
9. 整个里程碑可在单机和 profile 模式完成，不依赖真实边缘设备或在线 LLM
```

### Milestone 4：工作负载与真实 Profile 校准

目标：

```text
构造具有代表性的 Agent 任务集和动态调用轨迹，并用真实执行数据校准异构资源模型。
```

本里程碑复用 Milestone 2 的 AgentRunner、ToolRegistry、Executor、Trace 和 ExperimentManifest，以及 Milestone 3 的指标口径、baseline 批量实验和 evaluator。只增加任务数据、真实 Tool wrapper、采样与校准流程，不重新实现 Agent 框架、 Scheduler 或实验平台；远程部署和真实多设备实验仍留给 Milestone 6。

无在线 LLM 时，可以使用 scripted Agent、固定 trace 和注明来源的已有 benchmark 完成离线开发；真实 live Agent 的统计验收必须单独执行，未执行时明确标为未验证。单机逻辑副本、外部 benchmark 和合成 profile 均需注明来源，不能统称为真实异构硬件测量结果。

#### 4.1 定义版本化任务集与 Workload 配置 [已完成]

实现：

1. 定义轻量任务样本格式，至少包含 `task_id`、`task_type`、输入或 artifact 引用、参考答案或评分规则、允许使用的 Tool、输入规模和数据版本。
2. 第一版提供至少三类任务模板：图像预处理与 OCR、文本型或扫描型 PDF 的解析与信息抽取，以及需要多次或多个 Tool 调用的综合任务。扫描型 PDF 允许采用 `pdf_parse -> pdf_render -> ocr` 的条件回退路径。模板描述任务目标和可用 Tool，不预声明完整 Workflow DAG，实际调用由 AgentRunner 动态产生。
3. 每类任务准备小、中、大至少三个输入规模档位，明确图像分辨率、PDF 页数、文本长度或预计 token 数等规模字段；提交少量可离线运行的固定样本。
4. 配置 Agent 的 system prompt、Tool schema 及顺序、最大轮数、最大 Tool 调用数和超时，并复用 ExperimentManifest 保存版本和参数。
5. 配置 Agent 请求数量、到达间隔或到达分布、并发上限和随机种子，至少覆盖低负载、持续并发和突发到达场景；输入生成与策略执行使用独立种子。
6. 按原始样本 ID 划分校准集与留出验证集，同一原始文档或图像的变体不跨集合，避免用同一数据拟合 profile 并报告验证误差。

完成检查：

- 一个 workload 配置可以确定任务样本、Agent 配置、到达计划和数据划分。
- 相同配置和种子生成相同的样本 ID、输入摘要和到达计划。
- 三类任务及不同规模档位均有明确评分规则，不只检查 Agent 是否正常退出。
- 缺失 artifact、重复 task ID 或校准集与验证集重叠在运行前被拒绝。

#### 4.2 扩展真实 Tool 与副本一致性样本 [已完成]

实现：

1. 保留 ImagePreprocessTool，使用成熟库或本地命令封装 OCRTool、PDFParseTool 和 PDFRenderTool；通过已有 Tool Protocol、ToolRegistry 和 LocalToolExecutor 接入，不从零实现识别、解析或 PDF 页面渲染算法。
2. 为新增 Tool 定义稳定的输入参数、输出 schema、错误类型和实现版本；长文本或大文件使用受控 artifact 引用，避免将完整二进制内容写入 trace。
3. 使用大图像、多页 PDF 或批量处理构造至少一种长耗时 Tool 场景，优先复用现有 Tool，不为增加耗时而单独实现无研究价值的工具。
4. 提供小型固定输入、期望输出和必要的比较容差；同一 tool_name 的副本使用同一套功能一致性样本，沿用 Milestone 2 的一致性检查。
5. 如果 OCR 引擎、配置或实现版本改变输出质量，记录为显式实验变量并保存对应 quality profile，不把不同精度的实现当作完全等价的副本。
6. 将额外依赖和本地安装条件写入运行说明；缺失依赖时明确跳过对应 demo 并说明原因，不影响既有离线路径，但不能把跳过视为真实 Tool 验收通过。

完成检查：

- OCR、PDF parse 和 PDF render 均至少完成一次真实本地调用，返回统一 ToolResult 并写入 trace；其中 PDF render 的页面 PNG artifact 可作为后续 OCR 的输入。
- 正常输入、无效输入和执行失败具有可检查结果，长耗时场景遵循已有超时约定。
- 同功能副本的一致性检查可重复；不一致时可以定位到版本、参数或输出差异。

实现与验收记录（2026-09-06）：

- `OCRTool` 使用 Tesseract；`PDFParseTool` 使用独立子进程中的 pypdf。复用现有 Registry、Worker、Executor 和调度器，不增加服务端组件。
- 记录实测执行时间、输入字节数、像素/页数、单位工作量耗时及后端版本配置；使用真实批处理放大负载，不通过人工 sleep 制造慢任务。
- `scripts/run_tool_demos.py` 保存 ToolResult、trace 和一致性报告。小样本 OCR/PDF 均通过两个同机逻辑副本的一致性检查。
- 80 张 2560 × 1920 图像的重复输入压力场景中，两次真实 OCR 调用分别耗时 12.69 秒和 12.70 秒，输出一致。该结果仅证明存在十秒级本地真实工作负载，不构成异构硬件性能比较或独立样本质量评估。
- 统一预算传递至整个 batch，超时中止并回收后端子进程；原有纯进程内 Tool 保持执行后超时检查。失败、缺失依赖、输出篡改及配置变化均有针对性测试。
- `PDFRenderTool` 使用 Poppler `pdftoppm`，支持单个 PDF 和 JSON batch manifest，记录页数、DPI、渲染耗时和输出页面 URI；其输出不包含 OCR 文本，不应与 `PDFParseTool` 的文本抽取结果混淆。
- 能耗未实测，quality profile 未校准；不同后端配置需要独立质量校准，不将小样本一致性通过等同于质量为 1。采样与 profile 拟合仍由后续子任务完成。
- 英文技术说明位于 `src/edge_agent_workflow_scheduling/tools/README.md`。

#### 4.3 建立 MacBook 真实执行采样流程 [已完成]

实现：

1. 提供统一采样入口，按 Tool、输入规模和执行配置批量调用真实 Executor； warm-up 与正式测量分开保存，每个采样组合的重复次数可配置。
2. 保存设备型号、CPU/GPU、内存、操作系统、Tool/依赖版本、并发设置、采样时间和测量方式；区分冷启动与热运行，不将二者直接混成一个执行时间常数。
3. 分别采集排队、传输和执行时间；本地不存在传输时明确记为本地无传输，不能用人工添加的网络延迟冒充实测值。使用单调时钟测量耗时。
4. 在可用条件下采集 CPU、内存和 GPU 利用率及峰值内存；记录采样间隔和统计方式，无权限或无硬件支持的指标标记为 unavailable，不用 0 代替。
5. 分别测量单请求和受控并发下的延迟、吞吐量、成功率；记录实际排队与运行数量，区分资源竞争造成的退化和单次调用执行开销。
6. 保存逐次调用原始 trace、失败和超时样本及聚合统计；不只保留均值，也不静默删除异常点，若排除样本需记录规则和数量。

完成检查：

- 在 MacBook 上为图像预处理、OCR 和 PDF parse 生成可追溯的真实采样记录；PDF render 需要单独纳入采样矩阵后，才能宣称具有同等覆盖。
- 每个已纳入统一采样矩阵的 Tool 覆盖不同输入规模及至少两档并发设置，重复次数与样本数可核查；尚未纳入的 Tool 必须单独标注覆盖缺口。
- 每条聚合结果都能追溯到设备配置、输入样本和原始 trace。
- 采样结果包含延迟分布、成功率及资源指标可用性，而不只是一张平均耗时表。

实现与验收记录（2026-09-07）：

- 新增 `ToolSamplingConfig` 和 `scripts/sample_tools.py`，以 Tool × 输入规模 × 并发度组织统一采样矩阵；配置保存 cold-start、warm-up、正式测量次数和采样间隔。
- 正式测量保留每个终止调用，输出延迟分布、吞吐量、成功率、失败码、timeout 数量、队列等待、执行和总延迟的汇总，不静默删除异常值。
- 每个调用记录提交、开始、完成时刻、队列深度、运行数量、输入哈希和原始 `ToolResult`；本地输入输出传输明确为 `local_same_host_no_transfer`。
- 使用 `psutil` 轮询 profiler 进程及后端子进程，采集 process-tree CPU、RSS 峰值、进程数和线程数；GPU 利用率及不可用指标统一记录为 `unavailable`，不填 0。
- 主机清单记录操作系统、架构、CPU/芯片、逻辑/物理 CPU 数、内存、GPU inventory、 Python 与依赖版本；主机名仅保存哈希，不保存序列号、平台 UUID 等直接标识。
- cold-start 定义为新建 Tool/Executor 后的首次调用，不清理操作系统缓存；warm-up 单独保留且不混入正式统计。采样过程不插入人工等待，也不宣称物理异构节点结果。
- `image_preprocess`、`ocr`、`pdf_parse` 均覆盖 small/medium/large 与并发 1/2，组合级 summary、逐次 JSONL、资源采样和规范 trace 均可追溯。当前 `pdf_render` 已有独立 Tool 测试，但尚未接入 `configs/tool_sampling_v1.json` 和 `scripts/sample_tools.py` 的统一采样矩阵。
- 默认矩阵使用真实本地 Tool；能耗仍为 unavailable，质量仍为 uncalibrated，采样结果只作为后续 latency/energy profile 拟合输入，不直接改变 Scheduler。

- 单元测试和采样协议说明位于 `tests/test_tool_sampling.py` 与 `src/edge_agent_workflow_scheduling/tools/README.md`。

#### 4.4 接入真实 LLM 测量或已有 Benchmark [已完成]

实现：

1. 复用已有 LLMExecutor 接入至少一个可用的真实 LLM，或实现轻量导入流程读取已有 benchmark；不为本阶段另建模型服务框架。
2. 真实调用记录模型和 endpoint 标识、模型版本、提示词、采样参数、输入/输出 token、请求耗时、成功状态及提供方返回的 usage；凭据不写入数据和 manifest。
3. 覆盖不同 task_type 和输入规模，真实调用采用重复测量；在线请求保存费用或 token 预算限制，避免批量采样无限扩大。
4. 导入 benchmark 时保存来源、版本、任务定义、评分规则、硬件和运行配置；缺失的 token、能耗或设备信息保持缺失，不从模型参数量推断。
5. 为 LLM profile 提供吞吐量及适用范围；沿用现有 tokens_per_sec 估计口径时，明确 token 计数定义与请求耗时是否包含网络和服务端等待，避免重复计入延迟。
6. 准备至少两种存在速度、质量或能耗差异的 LLM profile，可组合真实测量、外部 benchmark 和显式合成配置；只有测量条件与评分口径可比的数据才能直接比较。

完成检查：

- 至少一组 LLM 数据来自真实调用或可追溯的已有 benchmark，而非全部手工设定。
- 两种 LLM profile 可以被现有 ResourceRegistry 和 Executor 加载，来源标识明确。
- 只有文本生成 benchmark 的模型不被标记为已通过真实 Function Calling 验证。
- 无在线凭据时离线导入仍可运行，在线采样明确跳过且不宣称获得了真实调用结果。

实现与验收记录（2026-09-11）：

- 新增 Chat Completions LLMExecutor，接入 vLLM / 火山引擎 Ark，统一转换消息、多个 ToolCall 和 Tool 输出；保留原有 Responses Executor，并保留不完整响应的 usage。
- `scripts/sample_llms.py` 支持有预算的真实重复采样及无需凭据的 benchmark JSON 导入，保存来源、任务与评分定义、原始响应、usage、失败、参数和 manifest。
- 配置两个本地、两个云端入口：27B 已实测；9B 未部署而默认禁用；Ark key 缺失则跳过，第二云模型由 `ARK_SECONDARY_MODEL` 指定，不伪造模型 ID。
- 本机 Qwen3.8-27B-FP8 完成 3 种文本任务 × 3 档规模 × 2 次重复，共 18 次真实成功请求；累计输入 8400、输出 952 token。原始数据作为 `configs/llm_benchmarks/qwen38_27b_local_20260911.json` 保存并通过离线导入验收。
- `configs/llm_measured_and_synthetic_v1.json` 提供 317.9515 总 token/秒的实测聚合 profile 和 64 总 token/秒的显式 synthetic 对照，均可由 ResourceRegistry / ProfileLLMExecutor 加载。synthetic 对照不冒充 9B 实测。
- rate 口径为 `(输入+输出 token)/客户端完整请求耗时`，包含网络和服务端等待；适用范围、样本数、排除规则与来源摘要保存在 metadata。不把该 rate 当 decode-only 速度，不自动补质量或能耗。
- 27B 服务更新为 vLLM 0.29.0 的 `qwen3` reasoning parser 与 `qwen3_coder` tool parser，并锁定容器镜像 digest、Hugging Face revision。更新前的真实 API 探测返回 HTTP 400，明确证明原启动参数未启用自动 Tool 解析。
- 更新后以 `tool_choice=auto` 同时暴露仓库中的 image preprocess、OCR、PDF parse、PDF render 四个 Tool。图像任务由模型自行选择并真实执行 `image_preprocess`，Tool 输出进入第二次 LLMCall 后生成最终回答；普通算术任务自行选择不调用 Tool。两个 AgentRun 均完成，27B profile 已标记真实 Function Calling 验证通过。
- 验证脚本为 `scripts/verify_function_calling.py`，精简验收产物位于 `configs/llm_function_calling/qwen38_27b_auto_tools_20260911.json`。本次没有验证 OCR/PDF 实际选择、并行 ToolCall、9B 或云模型 Function Calling。
- 质量评分及留出误差仍未验证；文本采样吞吐 profile 与 Function Calling 能力验证继续作为不同证据记录。
- Linux 创建独立 Python 环境；本地 loopback 模型请求绕过工作站代理。操作说明、部署模板、离线格式与验收范围见 `docs/llm_measurement.md`。

#### 4.5 建立任务评分与 Quality Profile

实现：

1. 为各 task_type 定义版本化评分函数，优先使用可重复的规则评分，例如 OCR 字符准确率、结构化字段匹配率和信息抽取 F1；明确空答案、解析失败和超时的计分规则。
2. 将任务评分映射到统一的 [0, 1] 区间，保存原始分数和映射规则；如需模型评分，额外保存评分模型、提示词和重复测量配置，不将其当作无噪声真值。
3. 在相同任务、提示词、Tool 配置和运行预算下测量不同模型，按 task_type 聚合 `quality(model, task_type)`；记录样本量、均值和离散程度或置信区间。
4. 用校准集建立 quality profile，用留出集报告泛化表现；外部 benchmark 仅在任务定义和评分口径匹配时映射到对应 task_type，不跨任务直接复制分数。
5. 将质量结果写入已有 quality_profile；没有覆盖的 task_type 明确标记缺失，是否允许使用 default 必须由实验配置声明，继续遵守 Milestone 3 的缺失值检查。
6. 扩展 evaluator 区分 selected-profile quality 与真实最终任务评分；多轮 Agent 混用模型时，最终任务得分归属于整个 AgentRun，不能简单平均成各模型独立质量。

完成检查：

- 固定输出的规则评分可以复现，正确、部分正确、错误和失败样本符合预期。
- 每项 quality profile 可追溯到任务集、模型、评分版本及样本量。
- 报告同时保留预测质量与真实任务评分，不再把 profile 质量当作答案准确率。
- 最低质量约束可以使用校准后的 profile，未覆盖任务不会被静默赋予高质量。

#### 4.6 拟合 Latency 与 Energy Profile

实现：

1. 先采用按任务类型、输入规模和并发档位分桶的统计量或简单回归，保存样本量、均值、分位数及适用范围；不在第一版引入复杂性能预测模型。
2. 将校准结果转换为现有 latency_profile、token_profile 和 energy_profile 可消费的字段。若引入分桶选择，统一封装 profile 查询逻辑供目标估计与 profile executor 复用，不在不同策略中各实现一套预测规则。
3. 能测量能耗时记录功率数据来源、采样区间、积分方式及是否扣除空闲基线；并发测量需说明能耗归属规则，不能将整个采样窗口的设备能耗重复算给每个调用。
4. 无直接能耗测量时可使用 CPU 时间或功率乘耗时等代理，明确标为 proxy/estimated 并记录单位和假设；CPU 时间不能直接写成 joules，缺失能耗不能填成 0。
5. 导出 LLM 的 joules_per_token 与 Tool 的 joules_per_call 时保留推导依据；若缺少所需能耗数据，则禁用依赖它的实验或明确使用单独版本的合成能耗配置，不绕过 Milestone 3 的配置检查。
6. 同一 Tool 至少准备两个不同执行 profile：优先使用真实可区分的运行配置，也允许一个本地实测 profile 加一个明确标注的逻辑模拟副本；记录配置差异，不把同一设备上的两个逻辑副本描述为两台真实设备。
7. 保存 profile ID、版本、来源类别、测量配置、原始数据引用、拟合方法和随机分布参数；后续新增设备时增加 profile 条目，不覆盖既有版本。

完成检查：

- 导出的 profile 可直接用于现有 baseline 和 profile executor，接口保持兼容。
- 同一 Tool 的两个 profile 有可解释的性能差异，质量差异按第 4.2 节显式处理。
- measured、benchmark、synthetic 等来源及能耗代理在数据与报告中可区分。
- profile 超出适用范围时明确拒绝或记录配置指定的回退，不静默外推。

#### 4.7 使用留出样本验证 Profile 误差

实现：

1. 使用第 4.1 节的留出样本，在相同输入和执行配置下比较 profile 预测与真实测量；外部 benchmark 只有存在对应留出观测时才参与误差验证。
2. 分别报告 Tool 执行时间、LLM 请求耗时或吞吐量的 MAE、相对误差及样本数，并按 task_type、输入规模和并发档位分组，避免总体均值掩盖局部偏差。
3. 仅对有真实能耗观测的样本报告能耗预测误差；代理量独立验证，缺失观测的指标标为未验证。相对误差对零值或近零分母使用预先声明的处理规则。
4. 对带随机性的 profile 比较多 seed 模拟与实测的均值、P95 及离散程度，区分单点拟合误差和分布差异，不要求模拟逐次复现实测噪声。
5. 在查看留出结果前配置可接受误差阈值；不达标时报告失败及适用范围限制，若据此调整模型，则另留独立测试集，不能反复用同一验证集调参后宣称无偏验证。
6. 保存校准前后在同一验证集上的误差对比及逐样本预测/观测数据；不预设校准必然改善所有指标，也不把未测量的合成节点标为已校准。

完成检查：

- 一条校准验证命令输出逐样本误差、分组汇总及配置阈值是否通过。
- 至少一组真实 Tool profile 的留出误差被量化，LLM 和能耗验证范围如实列明。
- profile、拟合样本和留出样本的对应关系可追溯，验证过程没有数据泄漏。

#### 4.8 生成固定 Replay Trace 与重复 Live 实验

实现：

1. 提供统一的 `scripts/run_workload.py` 入口，加载任务集、Agent 配置、到达计划、资源 profile 和策略；复用现有 Runner 和批量实验能力，不为每种任务复制脚本。
2. 保存动态 Agent 的完整调用轨迹，包括 run_id、turn_index、call_id、Tool 参数摘要、结果、最终任务评分和原始 response item；敏感内容遵守已有脱敏规则。
3. 从已保存轨迹生成固定 replay 输入；比较策略时保持相同任务、调用集合、外部到达计划和依赖关系，每轮 Tool 汇合后的就绪时间按当前回放执行结果推导，不能机械沿用原策略的开始/完成时间而破坏公平比较。
4. replay 不重新请求模型决定 Tool，也不将固定轨迹的任务评分解释为更换 LLM 后的真实质量；比较 LLM 选择对调用轨迹和最终任务质量的影响时使用 live 模式。
5. 对每个 live 任务和策略运行多次，保存重复编号与实际采样参数，报告端到端延迟、 Tool 调用数量、成功率和最终任务评分的均值、分位数及离散程度或置信区间。
6. scripted 离线路径用于回归检查并明确标识，不能替代真实 live 统计；固定 seed 只保证可控输入与 profile 随机性，不保证云端模型输出完全一致。

完成检查：

- 三类任务均可生成可加载的固定 trace，同一 trace 的调用 ID 和依赖关系保持一致。
- 同一 manifest 和 seed 可复现输入及 profile 模式结果，真实墙钟耗时单独记录。
- 至少两种 baseline 可以比较同一 replay workload，失败调用仍保留在统计中。
- 至少一组真实 live 实验完成重复运行并报告分布；未具备条件时明确列为未验证。

#### 4.9 打包校准产物与端到端验收

实现：

1. 提供 `scripts/calibrate_profiles.py` 入口，串联原始采样或导入、拟合、留出验证和 profile 导出；允许复用已保存样本，不强制每次重新执行昂贵测量。
2. 统一保存 workload、任务数据划分、Agent 配置、原始 trace、评分结果、版本化 profile、误差报告和 ExperimentManifest，使用内容摘要或版本引用建立关联。
3. 为新增任务解析、Tool wrapper、评分、profile 导出和误差计算增加单元测试，为离线 workload -> trace -> replay -> evaluator 增加小规模集成测试。
4. 使用校准后的 workload/profile 重跑 Milestone 3 的 baseline 比较与 Pareto 数据生成；各策略使用同一组预先固定的归一化尺度，并在 manifest 中保存。
5. 提供最小运行说明，区分无需 API key 的离线流程、本地 Tool 依赖、在线采样条件和能耗采集权限；说明如何增量加入新设备 profile，而不提前实现远程部署。

完成检查：

- 通过文档中的命令可以生成采样数据、校准 profile、误差报告和 baseline 汇总。
- 保存的产物足以离线重跑拟合与评估，原始测量不会被后续校准覆盖。
- 校准 profile 可以替换合成 profile 而不修改 Scheduler 核心接口。
- 验收报告列出已完成、缺失依赖和未验证项目，不将跳过项目计为通过。

里程碑验收标准：

```text
1. 至少三类任务模板覆盖不同输入规模，具有版本化样本、评分规则和到达配置
2. OCR、PDF parse、PDF render 和既有图像 Tool 完成真实本地执行；已纳入采样矩阵的 Tool 保存 MacBook 采样 trace，PDF render 在纳入统一采样矩阵前单独报告测试结果
3. 同一种 Tool 至少具有两个不同执行 profile，实测配置与逻辑模拟副本明确区分
4. 至少两种 LLM profile 存在速度、质量或能耗差异，其中至少一组来自真实执行
   或可追溯的已有 benchmark
5. 建立按 task_type 区分的 quality profile，并与真实 AgentRun 任务评分分别报告
6. latency/energy profile 保留来源、单位、适用范围和版本，能耗代理不冒充实测能耗
7. 至少量化真实 Tool profile 在独立留出样本上的误差，其他指标的未验证范围明确列出
8. 固定 replay trace 和带种子的 profile workload 可以复现，并可重跑 baseline 比较
9. 至少一组真实 live Agent 实验通过重复运行报告统计分布，缺少条件时不得宣称完整验收
10. workload、profile、原始 trace、manifest、评分和误差报告可追溯，后续设备可增量加入
```

### Milestone 5：多目标 RL 环境与策略

目标：

```text
在统一 AgentRun/Call/Resource 模型上训练多目标调度策略。
```

任务：

- 实现 Gymnasium-compatible 环境
- observation 包含当前 SchedulableCall、AgentRun 摘要和资源状态
- action 使用目标 ID 和 action mask
- reward 支持可配置权重和约束
- 第一版实现 Double DQN
- 支持 replay buffer、target network 和 checkpoint
- 使用不同权重训练或评估策略
- 与全部 baseline 对比

验收标准：

```text
环境通过 API 检查；
非法动作不会进入 Executor；
训练曲线和 checkpoint 可复现；
RL 在至少一组目标权重下优于对应 baseline；
同时报告退化的其他目标，避免只展示单一收益。
```

### Milestone 6：远程执行与真实边缘节点

目标：

```text
在不改变调度核心的前提下，将 Local/Profile Executor 替换为最小 Remote Executor。
```

任务：

- 定义最小 HTTP Executor adapter
- 远程查询执行目标状态
- 远程执行 LLMCall/ToolCall
- 支持超时、离线和恢复
- 在 Ubuntu、Raspberry Pi 或其他节点部署已有 Tool/LLM adapter
- 采集真实网络、排队和执行开销
- 更新设备 profile

本阶段不要求：

- 通用服务发现
- 复杂消息队列
- 生产级认证和容错
- Kubernetes 或大规模集群

验收标准：

```text
至少一个远程节点可以完成真实 LLMCall 或 ToolCall；
Scheduler 无需感知节点操作系统；
Remote Executor 返回与 Local/Profile Executor 结构一致的 LLMResult/ToolResult；
真实硬件结果可以复现实验配置。
```

### Milestone 7：系统评估

目标：

```text
回答联合调度、多目标权衡和异构资源利用是否有效。
```

核心实验：

1. 联合调度与 LLM/Tool 分离调度对比
2. 不同 LLM 规模和质量差异实验
3. 同一 Tool 多副本异构实验
4. 负载变化和节点失效实验
5. 不同目标权重与约束实验
6. Profile 模拟与真实设备结果对比
7. Baseline 与 RL 对比

报告指标：

- Agent 平均、P95、P99 端到端延迟
- deadline miss rate
- LLM 质量或任务成功率
- 总能耗或能耗代理指标
- 吞吐量
- LLM/ToolReplica 利用率
- 负载不均衡程度
- Scheduler 决策开销
- Pareto frontier

## 7. 精简后的仓库边界

```text
src/edge_agent_workflow_scheduling/
├── agents/        # Function Calling loop 与 AgentRun
├── common/        # AgentRun、Call、Result、Profile schema
├── queue/         # LLMCall/ToolCall queue
├── scheduler/     # baseline 与后续 RL policy
├── executors/     # local、profile、remote adapter
├── tools/         # 真实 Tool wrapper
└── profiler/      # trace 与指标

scripts/
├── run_first_demo.py
├── run_workload.py
├── calibrate_profiles.py
├── train_rl.py
└── evaluate.py
```

任何新增模块都需要直接服务于以下至少一个问题：

- AgentRun 与动态调用轨迹建模
- 异构 LLM/ToolReplica 建模
- 多目标调度
- 真实执行校准
- RL 训练
- 实验评估

否则不进入当前项目范围。
