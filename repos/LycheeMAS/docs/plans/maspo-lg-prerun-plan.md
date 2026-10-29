# 代码修改方案 —— MASPO 接入「LangGraph 原生运行前优化」统一接口

> 状态：**已实施**（2026-08-31）。用户决策：黄金法则 1 已修订（prerun 例外）；
> 旧 `pre_run_plugin` 接缝保留并存；实验先跑 **MATH-500**（已注册 `benchmark/math500`）；
> 执行/评估/反思用 MASPO 同款模型（执行=Qwen3-8B 本地，评估反思=gemini-2.5-pro API 端点）。
> 与本文差异：文本工具落在 `maspo/textops.py`（未并入 prompts.py）；`trainer/maspo` 桩已删除。
> 对标仓库：https://github.com/wangzx1219/MASPO（ICML 2026, arXiv:2605.06623），已克隆通读。
> 需求来源（用户四条设计要求）：
> 1. 统一接口，用一个 `method` 参数确定调用哪个算法；
> 2. 接口传入一个 **LangGraph 图**，在 LangGraph 图上优化并返回一个图——不再使用隔离的接口（MASGraph），直接接入 LangGraph；
> 3. 代码简单易读、注释完整、即插即用；
> 4. 拉取 MASPO 仓库，根据其实验写出测试代码（纯 LangGraph 实验 + 运行前优化实验）。

---

## 1. MASPO 读仓结论（决定怎么接的事实基础）

MASPO 是**免标注的 MAS 联合提示优化**（同族于 GEPA，不是 RL trainer——DESIGN.md §7 里 `trainer/maspo` 桩位放错了，本次一并修正）。核心机制：

| 环节 | 原版实现（文件:入口） | 机制 |
|---|---|---|
| 系统表示 | `agent.py: MAS` | agents 列表 + `edges: dict[int, list[int]]` DAG；每个 agent 一个**提示模板**（含 `{question}`/`{context}` 占位）；终端节点 = 无出边节点 |
| 执行 | `MAS.arun_with_cache` | 按 DAG 分层并发执行；`InferenceCache` 缓存每节点 (输入 context, raw 输出, 压缩短输出)；context = 前驱 short 输出以 `\n---\n` 拼接 |
| 局部重执行 | `MAS.arun_from_node` | 换某节点的 prompt 后，只重跑「该节点 + 全部后继」，上游读缓存——这是评估省钱的关键 |
| 提议变异 | `optimizers.py: _propose_new_prompt` | 反思 LLM 读（角色描述 + 采样执行轨迹 + 旧 prompt + 可选下游反馈）→ `<prompt>...</prompt>` 抽取 → `_sanitize_prompt` 校验占位符（非法即回退旧 prompt） |
| 多粒度评分 | `_evaluate_candidate` | 三路 LLM **成对比较**（候选 vs 基线，免 gold）：Local（本节点中间输出）、Lookahead（直接后继输出）、Global（终端答案）；win_rate = 0.4/0.4/0.2 加权（无 lookahead 时 0.7 local + 0.3 global）；score = win_rate − 0.5 |
| 错位挖掘 | 同上 | Local-Win 但 Global/Next-Lose 的样本 = misleading cases，按优先级收集，注入后续采样池（`use_misleading_sampling`）作 hard negatives |
| 搜索 | `process_single_node` + `_optimize_agent_fixed_rounds` | 每节点每层提 2 个候选、进化 beam search（beam_width=2，score>0 才入 beam，累计分排序）；调度 = 坐标上升式 fixed-rounds（每 agent 轮流做 `rounds_per_turn=3` 层，直到 `max_total_depth=9`）；`use_beam_refresh` 在重访 agent 时按队友新 prompt 重打 beam 分 |
| 产物 | `run_maspo.py` | **`prompt_map: dict[agent_id, prompt]` 落 JSON**；评测时 `mas.inject_prompt_map(prompt_map)` 注入后跑分 |
| 实验 | `run_maspo.py` | 数据集 aqua/math-500/aime-2025/gpqa/mbpp/humaneval-et（jsonl 随仓库发布）；拓扑主用 `reflect`（predictor→reflector 链，Nr 轮）；对照 = 原始 prompts vs 优化 prompts 的 accuracy；MASPO 全量 = `--optimize --fixed-rounds --beam-refresh --lookahead-score --misleading-sampling` |
| LLM | `config.py`/`utils.py` | 执行 LLM（Qwen3-8B）与评估/反思 LLM（gemini-2.5-pro）**分离**，OpenAI 兼容 API、异步并发 |

关键结论：**MASPO 的产物是一份 prompt_map，天然就是「运行前把图的提示替换掉」的插件**；重活（联合优化）是离线阶段，运行前挂载只是加载 + 注入——与 AgentPrune 的 train（落 state 文件）/ eval（before_run 产剪枝图）两段式完全同构。

版权注意：MASPO 仓库**无 LICENSE 文件**（匿名发布）。按 CLAUDE.md 规则 9：只迁移逻辑 + vendor 必要提示词资产，逐文件保留出处引用（repo URL + arXiv 引用）；不整包搬运。若未来对外发布需与作者确认许可。

---

## 2. 设计决策

### 2.1 统一接口（要求 1 + 2）

新增 REGISTRY 类别 **`pre_run_optimizer`**（第 18 个类别），协议与统一入口都放在新包 `plugins/prerun/`：

```python
# plugins/prerun/base.py
@runtime_checkable
class PreRunOptimizer(Protocol):
    """运行前优化器：传入一个（未编译的）LangGraph StateGraph，优化后返回一个 StateGraph。"""
    def optimize(self, graph: "StateGraph") -> "StateGraph": ...

def optimize_langgraph(graph: "StateGraph", method: str, **kwargs) -> "StateGraph":
    """统一入口（要求 1）：method 按名分发到 REGISTRY 的 pre_run_optimizer 类别。

    optimize_langgraph(sg, method="maspo",      mode="apply", prompt_file=...)
    optimize_langgraph(sg, method="maspo",      mode="optimize", trainset=..., backend=..., ...)
    optimize_langgraph(sg, method="agentprune", state_file=...)
    未知 method → REGISTRY 显式 KeyError（列出可用名）。
    """
    opt = REGISTRY.create("pre_run_optimizer", method, **kwargs)
    out = opt.optimize(graph)
    # 与 pre_run_plugin 同款校验：返回值必须是 StateGraph，否则显式 TypeError
    return out
```

要点：
- **图进图出**：接口只认 LangGraph 的 `StateGraph`（**未编译** builder，不是 `CompiledGraph`——编译后节点/边不可改；调用方在 `optimize_langgraph` 之后再 `.compile()`）。
- **两种模式统一在一个入口**（与 AgentPrune train/eval 两段式对齐）：
  - `mode="apply"`（默认，轻量、每 query 可调）：加载离线产物（MASPO 的 prompt JSON / AgentPrune 的 state JSON）→ 改写图 → 返回；
  - `mode="optimize"`（重活、脚本驱动一次）：在图上跑完整联合优化（需 trainset + backend），落产物文件，再 apply 后返回。
- 缺必需参数（apply 无 prompt_file、optimize 无 trainset/backend）→ 显式 `ValueError`，不静默兜底。

### 2.2 直接接入 LangGraph 的图约定（要求 2 的落地机制）

`StateGraph` 的节点是闭包函数，提示词与角色对优化器不可见。解决：**一条节点元数据约定 + 一个读写工具**，这是本方案唯一的新增契约：

```python
# 建图方（脚本/构建层封装）按约定挂元数据；节点函数运行时从 spec 读模板（闭包持有可变 AgentSpec）
sg.add_node(name, node_fn, metadata={"agent_spec": spec})   # spec: core.types.AgentSpec
# spec.system_prompt = 可变异提示模板（MASPO 语义：含 {question} / {context} 占位）

# plugins/prerun/graphview.py —— StateGraph ↔ 优化器 的唯一读写通道
extract_view(sg) -> GraphView      # 节点(名字+AgentSpec) + 邻接(来自 sg.edges) + 终端节点
                                   # metadata 缺 agent_spec / 图非 DAG / 无唯一终端 → 显式报错
rebuild(sg, *, prompts=None, adjacency=None) -> StateGraph
                                   # 产新图：替换 system_prompt（深拷贝 spec）和/或按新邻接重连边
                                   # 节点函数原样复用（它读 spec，换 spec 即换行为）
```

优化器内部循环在 `GraphView` 上跑（MASPO 原版评估也是自建 MAS 实例跑缓存局部重执行，`app.ainvoke` 做不了「从某节点重跑读上游缓存」）；**接口边界上进出都是 StateGraph**。真实评测执行始终走编译后的 LangGraph 图。

### 2.3 与 CLAUDE.md 黄金法则 1 的冲突与修订（需确认）

现行规则：`langgraph` 只允许出现在 `runtime/backends/langgraph_runtime.py` 的 `_build_app` 内。要求 2 明确要"直接接入 LangGraph"，与之冲突。修订提案：

> 黄金法则 1 增补："`langgraph` 还允许出现在 `plugins/prerun/`（LangGraph 原生运行前优化接缝）。"
> 黄金法则 2 不变：`prerun` 内所有 `langgraph` import 一律惰性（函数内），`make selfcheck` 仍须 `HEAVY LOADED: NONE`；类型注解用字符串前向引用。

旧接缝 **`pre_run_plugin`（MASGraph 版）保留不动**：Orchestrator、`run_langgraph_preplug.py`、AgentPrune GSM8K 脚本零回归；`prerun` 是 LangGraph 主线的新接缝，两者并存（AutoGen 退役后再考虑收敛）。

### 2.4 AgentPrune 并入统一接口（要求 1 的第二个算法）

`pre_run_optimizer/agentprune`：薄适配器，复用既有 `graph_pruner/agentprune` 的全部状态与 threshold 实现——`extract_view` 取邻接 → `AgentPrunePruner(state_file=...).realized_matrices("threshold")` → `rebuild(sg, adjacency=剪枝后)`。算法零改动，一个 `method` 参数即可在 maspo / agentprune 间切换。

---

## 3. 文件级修改清单

### 新增

```
src/lychee_mas/plugins/prerun/
├── __init__.py          # 触发注册（maspo / agentprune）+ 导出 optimize_langgraph（纯标准库 import）
├── base.py              # PreRunOptimizer 协议 + optimize_langgraph 统一入口 + 返回类型校验
├── graphview.py         # GraphView / extract_view / rebuild（langgraph 惰性 import 的唯一位置之一）
├── agentprune_lg.py     # @REGISTRY.register("pre_run_optimizer", "agentprune")
└── maspo/
    ├── __init__.py
    ├── optimizer.py     # @REGISTRY.register("pre_run_optimizer", "maspo")：
    │                    #   apply（加载 prompt JSON → rebuild(prompts=...)）
    │                    #   optimize（fixed-rounds 坐标上升主线 + beam search + 多粒度评分
    │                    #            + beam_refresh / lookahead / misleading_sampling / feedback 开关；
    │                    #            简单拓扑序模式 variant="topo" 一并提供；round-robin 不移植）
    ├── executor.py      # CachedExecutor：移植 InferenceCache + arun_with_cache + arun_from_node 语义；
    │                    #   LLM 经注入的 async 回调触达（backend / evaluator_backend 两路，
    │                    #   分别对应原版 Qwen3-8B 执行端与 gemini 评估/反思端）
    └── prompts.py       # vendored 提示资产（PROMPT_OPTIMIZE / ANSWER_EVALUATE / INTERMEDIATE_COMPARE /
                         #   FINAL_ANSWER_COMPARE / COMPRESS / ROLE_DESCRIPTIONS / AGENT_TEMPLATES(reflect 用到的) /
                         #   OPTIMIZATION_REQUIREMENTS）+ 文本工具（_sanitize_prompt、extract_answer/code、
                         #   parse_comparison_result、majority_vote）——逐字迁移，文件头保留出处引用

configs/prerun/maspo.yaml        # 超参默认值 = 原版论文模式：max_total_depth=9, rounds_per_turn=3,
                                    #   beam_width=2, lookahead_weights=[0.4,0.4,0.2], eval_batch=10,
                                    #   use_beam_refresh/lookahead/misleading=true, use_feedback=false
configs/prerun/agentprune.yaml   # state_file / eval 实现模式

tests/test_maspo.py                 # 纯离线（零 langgraph）：算法核心单测（§5.1）
tests/test_prerun.py             # importorskip("langgraph")：图接口 + 端到端脚本化测试（§5.1）

scripts/run_maspo_langgraph.py      # 复现实验脚本（§5.2）：纯 LangGraph 基线 / 运行前优化 两条腿
```

### 修改

| 文件 | 改动 |
|---|---|
| `src/lychee_mas/core/registry.py` | `CATEGORIES` 加 `"pre_run_optimizer"`（表本身不强制，纯文档性） |
| `src/lychee_mas/plugins/__init__.py` | import `prerun` 触发注册 |
| `src/lychee_mas/plugins/README.md` | 新增 `pre_run_optimizer` 一节（接口 + method 表） |
| `CLAUDE.md` | 黄金法则 1 增补 `plugins/prerun/` 例外（§2.3 措辞） |
| `docs/DESIGN.md` | §5 插件系统加 prerun 接缝；§7 表加 `pre_run_optimizer` 行（`maspo`, `agentprune`）；`trainer` 行删去 `maspo` 桩（归类错误，迁至此处） |
| `pyproject.toml` | 无新依赖（复用 `[langgraph]` extra；MASPO 逻辑纯标准库化，tqdm/openai 依赖全部去掉） |

### 不改

`pipeline.py`（黄金法则 3）、`plugins/adapters.py`、`layers/prune/` 全部、三个既有 langgraph 脚本、`run_agentprune_gsm8k.py`。

---

## 4. MASPO 移植忠实度声明

逐项对应（机制 1:1）：提示模板与评估/反思/压缩提示词**逐字 vendored**；`_sanitize_prompt` 占位符校验逐行移植；三路成对比较与 win_rate 加权公式、score=win_rate−0.5、score>0 才入 beam、累计分排序 top-beam_width、misleading 优先级（both-lose > next-lose > global-lose，取前 3）、fixed-rounds 调度与 beam refresh（top-1 overlap 统计）全部保持原语义；候选数=2/层、eval 采样 10 题/层、misleading 注入上限 5 题保持默认。

已声明偏差（docstring 级）：
1. **LLM 后端**：经本框架 backend 回调（本地 HF 或 API），非原版 Qwen3-8B/gemini 双端点；绝对准确率不可比，对比对象是同模型下「原始 prompts vs MASPO 优化 prompts」。
2. **系统表示**：原版 `MAS` 类 → 本方案 `GraphView`（从 StateGraph 元数据提取）；`inject_prompt_map` → `rebuild(sg, prompts=...)`。
3. **并发**：保留 async 结构；本地 HF backend 天然串行（原版靠 API 并发提速，语义不变）。
4. **不移植**：round-robin 调度模式（论文主实验用 fixed-rounds）、AGGREGATOR 投票节点（reflect 拓扑用不到；遇到显式报错）、LLM judge 打分（我们的评测走 eval/metrics 的 gold 打分——注意：**优化过程本身免 gold**，与原版一致；gold 只用于事后评测）。

---

## 5. 测试与实验（要求 4）

### 5.1 离线测试（pytest，进 `make test`）

`tests/test_maspo.py`（零重依赖，LLM 全部脚本化 fake）：
- `_sanitize_prompt`：合法占位保留 / 未知占位回退旧 prompt / 缺 `{question}` 自动补 / 重复 `{question}` 回退 / 花括号归一化；
- `parse_comparison_result`：`<choose>A</choose>`、尾字母回退、默认 True；
- 评分聚合：构造脚本化比较结果，断言 lookahead 加权 win_rate、无 lookahead 的 0.7/0.3 退化、misalignment（Local-Win/Global-Lose）计数与 misleading 优先级排序；
- beam 一步：两候选一好一坏 → 好者入 beam、best_overall 更新、score≤0 者原地保留；
- 显式报错路径：apply 无 prompt_file、optimize 无 trainset/backend、反思器返回空文本。

`tests/test_prerun.py`（`pytest.importorskip("langgraph")`，与 `test_langgraph_runtime.py` 同模式）:
- `extract_view` / `rebuild` roundtrip：reflect 小图（2 节点）元数据齐全 → 视图正确；缺 metadata / 有环 → 显式报错；
- 统一入口分发：`optimize_langgraph(sg, method="maspo", mode="apply", prompt_file=...)` 后编译执行，fake LLM 记录到的 formatted prompt 已换新（即插即用验证）；未知 method → KeyError；
- `method="agentprune"`：同一份 state 文件，统一接口剪出的邻接 == 既有 `graph_pruner/agentprune.realized_matrices("threshold")`（两条路径对拍）；
- 脚本化端到端 optimize：3 题迷你 trainset + 脚本化反思器（固定返回改进 prompt）+ 脚本化比较器（新 prompt 恒赢）→ 跑通 fixed-rounds 全循环，产物 prompt JSON 可 roundtrip 加载。

### 5.2 真实实验（对标 `run_maspo.py`，GPU/API）

`scripts/run_maspo_langgraph.py`，reflect 拓扑（predictor→reflector，`--nr` 轮），节点函数按 §2.2 约定建图，评测/落盘与 `run_agentprune_gsm8k.py` 同口径（samples + summary + config 快照，报 accuracy/token/latency）：

```bash
# 实验一：纯 LangGraph 基线（原版 prompts，无优化）—— 对应 run_maspo.py 不带 --optimize
python scripts/run_maspo_langgraph.py --phase baseline --task gsm8k \
    --model-path /data/mxy/Models/Qwen/Qwen3-4B --eval-n 100

# 实验二：运行前优化（MASPO 全量：fixed-rounds + beam-refresh + lookahead + misleading）
#   optimize：统一接口 mode="optimize" 在图上联合优化 → 落 prompt JSON
#   eval：    统一接口 mode="apply" 挂载 → 同图跑分
python scripts/run_maspo_langgraph.py --phase both --task gsm8k \
    --model-path ... --train-n 50 --eval-n 100 [--evaluator-api ...]
```

数据集：默认 `gsm8k`（本地已备，MATH 任务族语义一致，声明偏差）；`--task aqua` 作可选项——P2 增加 `benchmark/aqua` loader（`prepare` 从 MASPO raw URL 拉 jsonl 到 `CDM_DATA_ROOT`，遵守 .gitignore 不入库），严格对齐原版第一实验。

---

## 6. 实施顺序与验收

- **P0 图接缝**：`prerun/base.py` + `graphview.py` + `agentprune_lg.py` + registry/`__init__` + `test_prerun.py` 的接口部分。验收：三件套全绿、`agentprune` 双路径对拍一致、`make selfcheck` 仍 `HEAVY LOADED: NONE`。
- **P1 MASPO 核心**：`maspo/`（prompts vendored + executor + optimizer）+ `test_maspo.py` + 端到端脚本化测试 + configs。验收：三件套全绿、脚本化 optimize 全循环跑通。
- **P2 实验与文档**：`run_maspo_langgraph.py` 小样本真跑（`--train-n 8 --eval-n 8` 冒烟）→ 全量两实验对比；可选 `benchmark/aqua`；CLAUDE.md / DESIGN.md / plugins README 同步。验收：baseline 与 optimized 两份 run 目录落盘可对比。

## 7. 待确认问题

1. **黄金法则 1 修订**（§2.3）：同意 `plugins/prerun/` 作为 langgraph 的第二个合法出现地（惰性 import）？
2. **旧接缝去留**：`pre_run_plugin`（MASGraph 版）按本方案保留并存；若你想彻底替换（Orchestrator 也改吃 StateGraph），是另一个量级的重构，需明示。
3. **实验数据集**：默认 gsm8k、aqua 作 P2 可选，还是必须严格用 MASPO 的 aqua/math-500 起步？
4. **评估/反思 LLM**：原版用 gemini-2.5-pro 独立端点；本地复现默认与执行 LLM 同一个模型（可 `--evaluator-api` 指到 API）。可接受？
