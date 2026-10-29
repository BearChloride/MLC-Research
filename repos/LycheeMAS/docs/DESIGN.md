# LycheeMAS 架构设计

> 本文件是 LycheeMAS 的**唯一架构设计文档**：框架目标、五接缝架构、各层职责与接口契约、组件注册全景、评测体系与验证策略。面向编码代理的操作规范（环境、命令、代码规范、检查清单）见仓库根 `CLAUDE.md`。
>
> 版本范围：LycheeMAS v0.3（五模块全插件化重构后）。

---

## 1. 核心目标

**LycheeMAS 是一个多智能体系统（MAS）研究框架**。核心思想：把整个 MAS 统一表示成一张带时序与记忆状态的有向图 **G=(V,E,W,T,M)**（V=智能体节点、E=通信边、W=边权、T=多轮时序、M=记忆状态）；**五个研究模块 = 五个挂载式接缝**，每一次系统演化都是接缝上的一次挂载/变换。由此得到四条设计原则：

1. **统一表示**：图的载体是 LangGraph 的 `StateGraph` + 节点契约（`AgentSpec` 元数据，§3.3）；公共类型只在 `core/types.py`。
2. **可插拔、可消融**：每个算法 = 注册到 REGISTRY 的一个类，由 `method` 参数按名挂载；对照实验只换方法名。未实现的方法以桩占名，让消融矩阵在代码里可见。
3. **接口与实现分层**：`plugins/` 定义接缝（协议 + 统一入口 + 分发 + 校验，薄），`methods/` 存方法（论文复现 / 训练循环 / 重机器，厚），按接缝互相镜像。
4. **离线可跑、可复现**：核心代码与示例在无 GPU / 无 API key 的环境必须能跑（脚本化 LLM + 零重依赖注册）；实验固定种子、落 config 快照与 git SHA，同时报告 accuracy / token / latency。

新增代码遵循**显式错误原则**：组件遇到不支持的输入/配置显式报错，不静默降级、不写死兜底数据。

---

## 2. 总体架构：五接缝生命周期

一次任务的完整生命周期（任何一环不挂载即零回归直通）：

```
build_langgraph(method)        构建：team 模板/选队器 → 契约 StateGraph
      │
optimize_langgraph(sg, method) 运行前：剪枝/提示优化改写图（apply 即插即用；optimize 离线产物化）
      │
attach_memory(sg, method)      运行时：记忆注入六步包裹进 agent 节点（P3 实现中）
      │
sg.compile() ──► runner        编译执行：每题跑图、组装 Trajectory
      │
run_processed(runner, method)  执行：serial 1 次 / parallel 并发 K 次 + 归约（pass@K 承载点）
      │
analyze_run(trajectory, ...)   运行后：归因 → 信用（读侧）；train_from_runs(...)（写侧离线训练）
```

训练的统一形态：**训练素材自给的方法（MASPO/GEPA/AgentPrune），训练循环跟方法走**（`methods/prerun/`，即 prerun 的 `mode="optimize"`）；**素材来自执行轨迹的训练（RL）走 postrun 写侧**（`train_from_runs`）。所有训练产物统一经 prerun 的 `mode="apply"` 挂载回图。

---

## 3. 共享基座

### 3.1 `core/` — 类型与注册表

- **`core/types.py`**（纯 dataclass，零重依赖）：`AgentSpec`（图节点画像，**节点契约的载体**）、`Message`、`Answer`、`Trajectory`（一次执行 τ）、`TaskQuery`（含 `gold`）、`Budget/BudgetUnit`。
- **`core/registry.py`**：`@REGISTRY.register(category, name)` / `REGISTRY.create` / `snapshot`。**CATEGORIES 按接缝分组**：
  - build：`graph_builder, agent_selector`
  - prerun：`pre_run_optimizer, graph_pruner, vocab_adapter`
  - memory：`memory_manager, memory_router`
  - processing：`processor, aggregator`
  - postrun：`attributor, credit_assigner, trainer`
  - 其他：`benchmark`（评测）、`optimizer`（GEPA 离线 compile）

### 3.2 `backends/` — 生成原语层

回答「**怎么调一个 LLM**」（plugins/methods 回答「用 LLM 做什么」）：`hf_backend`（本地 HF：`generate_chat` / `encode_hidden` / prefix-KV 原语）、`openai_api_backend`（OpenAI 兼容 API）、`spans`（`JsonlSpanLogger` 运行事件落盘）。methods 里的算法不直接 import transformers/openai——由脚本用本层构造 async 回调注入（如 MASPO 的 `agent_llm`/`evaluator_llm`/`proposer_llm`）。重依赖只在此层（与 methods 内部）惰性导入，`make selfcheck` 必须 `HEAVY LOADED: NONE`。

### 3.3 节点契约（graphview，五接缝互操作的唯一约定）

```python
sg.add_node(name, node_fn, metadata={"agent_spec": spec})   # spec: core.types.AgentSpec
# spec.system_prompt        可变异提示模板（{question}/{context} 占位）
# spec.meta["predecessors"] 通信前驱节点名列表（节点函数据此从 state 选 context）
# node_fn 运行时从 spec 读模板/前驱——闭包与元数据共享同一 spec 对象，改 spec 即改行为
```

读写唯一通道 `plugins/prerun/graphview.py`：`extract_view(sg)`（视图提取；缺元数据/非 DAG/多终端显式报错）、`rebuild(sg, prompts=…/adjacency=…)`（提示原地写 / 邻接产新图并回写元数据）。build 接缝产契约图，prerun/memory 接缝消费与改写契约图。

---

## 4. 五接缝详解（plugins/ 接口 × methods/ 实现）

### 4.1 build（构建，`plugins/build/` × `methods/build/`）

回答「由谁组队、怎么连」。`build_langgraph(method, node_factory, state_schema, ...) -> StateGraph`：节点函数语义（生成后端、状态形状）由实验方以 `node_factory(spec, is_terminal)` 注入，构建器负责 AgentSpec 链、通信结构、元数据挂载与 START→…→END 执行边。已实现 `graph_builder/static`（team 模板链）；选队器 `agent_selector/agentinit`（EMNLP'25，多样性×相关性 Pareto 选队）产出 AgentSpec 列表经 `agents=` 传入。

### 4.2 prerun（运行前优化，`plugins/prerun/` × `methods/prerun/`）

回答「执行前对图做什么优化」。`optimize_langgraph(sg, method, **kw) -> sg`，图进图出；两段式：`mode="optimize"`（离线重活，产物落 JSON）/ `mode="apply"`（加载注入，即插即用）。已接：

- `pre_run_optimizer/maspo`——MASPO 联合提示优化（ICML 2026）：多粒度成对评估（Local/Lookahead/Global 0.4/0.4/0.2，免 gold）+ 错位驱动采样 + 进化 beam search + fixed-rounds 坐标上升 + Beam Refresh；断点续跑（`*_ckpt.json`）。MATH-500 复现：归一化口径 0.78→0.85（+7pt）。
- `pre_run_optimizer/agentprune`——AgentPrune 时空掩码剪枝（ICLR 2025）：REINFORCE 训练逐边 logit + one-shot 剪枝（训练脚本 `run_agentprune_gsm8k.py`），threshold 确定性实现剪图。
- `pre_run_optimizer/agentdropout`——AgentDropout 动态节点/边淘汰（ACL 2025）：两阶段——①逐轮加权度 softmax 采样跳过节点 + skip-REINFORCE 训度权重，每轮淘汰最小归一化加权度节点（空间行列 + 跨轮时间边清零）；②逐轮独立参数的边 REINFORCE + one-shot 剪边（日程 = 5 agents × 2 轮，2 batch × 20 题 → node_dropout，4 batch × 10 题 + batch 1/3 各剪一次）。训练循环跟方法走：`mode="optimize"` 由 `methods/prerun/agentdropout/trainer.py::AgentDropoutTrainer` 跑（`trainset` / `rollout` / `reward` / `predict` 由脚本注入，产物落 `state_file` + `*_train_log.json`），`scripts/run_agentdropout_gsm8k.py` 经 `optimize_langgraph(mode="optimize")` 驱动、只提供题池与生成后端（与 MASPO 同一接口面）。拓扑逐轮不同：`mode="apply"` 按 `round=r` 挂载该轮实现，被淘汰节点写 `meta["dropped"]`。接缝实现类（`AgentDropoutLG`：mode 校验 / apply 挂载 / optimize 驱动）与 `maspo` 同款放在 `methods/prerun/agentdropout/optimizer.py`，`plugins/prerun/` 不留实现（`agentdropout_lg.py` 只剩兼容 shim）。
- `optimizer/gepa`——GEPA 反思式提示演化（`methods/prerun/gepa/`，MASProgram 表示；graph-native 适配待接）。

### 4.3 memory（运行时记忆，`plugins/memory/` × `methods/memory/`）

回答「智能体之间记住什么、以什么表征传递」。`attach_memory(sg, method, backend, **kw) -> sg`：把注入六步（observe → route → recall → system 段注入 → 生成 → 记账）**重包进每个 agent 节点**（P3 全新实现中，当前显式桩）。算法库：`channels/`（NL / 隐空间 / C2C）+ `managers/`（`memory_manager/cdm` 已实现）+ `routing/`（`static`/`fixed` 已实现）+ `store.py` + `context.py`（RoutingContext 决策日志/span 落盘）。注入六步的旧引擎实现已随 runtime 兼容层删除（git 历史 `runtime/injection.py` 可作 P3 语义参照）。

### 4.4 processing（执行，`plugins/processing/` × `methods/processing/`）

回答「一个任务跑几次、多次结果如何归约」。`run_processed(runner, method, **kw) -> ProcessingResult`（`runner: async () -> Trajectory` 由调用方提供）：`processor/serial`（1 次）/ `processor/parallel`（并发 K 次 + aggregator 归约，构造超参走 `aggregator_kwargs` 透传）。归约策略：`self_consistency` 多数投票 / `aggagent`（AggAgent 移植，COLM 2026：聚合本身 agentic 化——检索工具跨轨迹「数证据不数轨迹数」，需 tool-calling 端点）。pass@K 的承载点。

### 4.5 postrun（归因训练，`plugins/postrun/` × `methods/postrun/`）

回答「一条轨迹里谁该为成败负责、如何用信号改进系统」。三入口：

- **读侧** `analyze_run(trajectory, score, method, ...)`：attributor（`all_at_once/step_by_step/binary_search` 桩）→ credit_assigner（`attribution_guided` 桩）→ 写回 `trajectory.meta` 与 `TraceStore`。
- **图闭环** `optimize_postrun(sg, trajectories, method, ...)`：与 prerun 对称的批量离线运行后优化——图 + 轨迹批 → 优化 → 图（`post_run_optimizer` 类别；`attribution`/`train` 桩占名）。
- **写侧** `train_from_runs(method, ...)`：离线消费轨迹与信用产训练产物（`trainer` 类别占名待接 RL 线）；产物经 prerun apply 挂载。

---

## 5. 评测体系（`eval/`，顶层独立）

- **benchmark（20 个注册名）**：文本推理/知识 `gsm8k, aime_2024, math500, medqa, arc_easy, openbookqa, locomo10`；代码/通用助理 `human_eval, gaia_validation(_level_1..3)`；MAS 轨迹分析 `aftraj_audit(_test), agent_collab_{idr,rtd,cpr,clc}, mast_failure, open_agent_traces`。统一记录格式 `{task, kind, question, gold, context}`；数据加载惰性；入口 `benchmarks.load(task, n)` / `prepare(task)`。
- **推理 / 打分分离**：实验脚本只推理落盘（samples + config 快照）；`scripts/analyze_benchmark_run.py --score-predictions` 事后打分。
- **pass@K**：`run_processed(method="parallel", k=K)` 采样 K 次；打分侧 `metrics.aggregate_samples` 按 case 聚合（pass@1 = 各 case K 份均分再平均；pass@K = best-of-K 再平均）。
- **指标**：`kind` 覆盖 mc / exact / aime（数值+符号等价）/ f1 / human_eval / gaia / MAS 专用族；评测同时报告 accuracy / token / latency。注意：免 gold 的判官偏好优化会漂移答案表面形式（Unicode 极简写法等），评测端需写法鲁棒（MASPO 复现的方法学发现）。

---

## 6. 组件注册全景（按接缝分组）

| 接缝 | 类别 | 已实现/可跑 | 桩（占名待接） |
|---|---|---|---|
| build | `graph_builder` | `static` | — |
| build | `agent_selector` | `agentinit` | — |
| prerun | `pre_run_optimizer` | `maspo`, `agentprune`, `agentdropout` | — |
| prerun | `graph_pruner` | `agentprune`, `agentdropout` | `agentdropout_v2` |
| prerun | `vocab_adapter` | — | `agentvocab` |
| prerun | `optimizer` | `gepa`（graph-native 适配待接） | — |
| memory | `memory_manager` | `cdm` | `mem0`, `ama` |
| memory | `memory_router` | `static`, `fixed` | `learned`, `soft_gate` |
| processing | `processor` | `serial`, `parallel` | — |
| processing | `aggregator` | `self_consistency`, `aggagent`（AggAgent agentic 聚合，COLM 2026） | `dynamicagg` |
| postrun | `attributor` | — | `all_at_once`, `step_by_step`, `binary_search` |
| postrun | `credit_assigner` | — | `attribution_guided` |
| postrun | `trainer` | — | —（RL 线待接） |
| postrun | `post_run_optimizer` | — | `attribution`, `train`（图闭环桩） |
| — | `benchmark` | 20 个（§5） | — |

> 桩能被 `REGISTRY.list` 看到是有意为之：占好名字、让消融矩阵可见。组件状态变化时同步更新本表。

---

## 7. 验证策略

1. **离线三件套**（每次改动收尾必跑）：`make lint`、`make test`（全部离线，LLM 脚本化）、`make selfcheck`（`HEAVY LOADED: NONE`）。
2. **端到端 demo**：`make demo` 跑五接缝离线链路（build → prerun apply → compile → processing 归约）。
3. **黄金基线对拍**：重构涉及执行链路时，deterministic 设置下与改前基线逐字比对 `predictions.jsonl`、滤易变字段后比对 `spans.jsonl`。
4. **可复现**：固定 `--seed`；每次实验落 config 快照 + git SHA 到 `runs/`。
