# 代码修改记录 —— AggAgent 接入（aggregator/aggagent）

> 状态：**已实施**。范围：将 princeton-pli/AggAgent移植为聚合器类别组件 `aggregator/aggagent`，与 `self_consistency` 同类协议可直接对拍；
> 纯标准库实现 + 脚本化客户端 mock 测试全绿（离线零重依赖）。

---

## 1. 背景

AggAgent 把「K 条并行 rollout 轨迹 → 一个答案」的归约本身做成 agentic 任务：一个聚合 agent
把 K 条轨迹当作**可检索环境**，用四个轻量工具跨轨迹推理（核心原则写进 system prompt：
**数证据不数轨迹数**——单条带工具观测支撑的证据强于多条仅靠推理的多数一致，冲突时信工具观测）：

- `get_solution(trajectory_id?)`：取轨迹最后一步的最终解（不传 = 全部 K 个解）；
- `search_trajectory(traj, q)`：单条轨迹内关键词检索 top-k 步（ROUGE-L 排序，可 `role='tool'`
  过滤到真实工具观测）；
- `get_segment(traj, s, e)`：细读连续段（≤5 步）验证上下文；
- `finish(solution, reason)`：提交 `<explanation>/<answer>` 或长报告格式终答。

## 2. 设计

1. **落点：`aggregator` 类别**（与 `self_consistency` 并列），不是 plugins——它是 `processor/parallel`
   的可插拔归约策略（协议 `aggregate(list[Trajectory]) -> Answer`，见 `processing/base.py`），
   不是执行图变换。
2. **同步引擎**：`aggregate` 是同步协议（由 async `ParallelProcessor.run` 在事件循环内调用），
   移植的聚合循环保持同步 + `AggClient.complete` 同步接缝，无 asyncio/线程纠缠。
3. **Client 接缝替代 litellm**：原版经 litellm 路由 hosted_vllm/gemini/openai 多后端；本框架不引
   litellm（黄金法则 2），收窄为 **OpenAI 兼容 chat/completions tool-calling 端点**（vLLM 开
   `--enable-auto-tool-choice`，与论文 rollout 启动方式一致）。请求体/重试/错误语义复刻原版
   call_server 默认值（temperature 1.0、top_p 0.95、max_tokens 10000、parallel_tool_calls False、
   指数退避），仅去掉重试 jitter（确定性），错误归一成两个哨兵字符串
   `"ContextLengthError"` / `"Server error"`——与原版 call_server 返回类型一致，便于逐字对照。
   mock 测试注入脚本化客户端即离线跑通。
4. **ROUGE-L 纯 stdlib 化**：原版依赖 rouge_score；移植版用 LCS 实现 `_rouge_l_recall`，
   分词近似（空白切分 + 小写）≈ rouge_score SimpleTokenizer，检索排序语义对齐。
5. **轨迹适配（唯一新增适配层）**：原版消费「消息步 dict 列表」；`trajectory_to_steps` 把
   `Trajectory` 按 (round, sender) 稳定排序折成步列表——role 直通 `Message.role`（工具观测消息
   标 `role="tool"`，对齐检索过滤语义）；`meta.reasoning/reasoning_content` → reasoning_content；
   `meta.tool_calls`（OpenAI 形状）透传。终答落在最后一条 message（对齐 get_solution last-step）。
6. **显式错误，无静默兜底**（黄金法则 6）：空输入 ValueError、非 Trajectory TypeError、缺题面
   ValueError、未配置聚合模型 ValueError（注入 client 或 model+api_base 二选一）、注入 client 缺
   `complete` TypeError、聚合失败/迭代预算耗尽/服务端错误 → RuntimeError（带 stats）。原版
   return-error-dict 的路径改为 raise，不吞错。
7. **License / 归因**：Apache-2.0。`NOTICE.md` 记录逐文件移植映射与行为差异，`LICENSE-APACHE-2.0.txt`
   随包放置；prompt / 工具 schema 文案**逐字保留**（改词会破坏与论文 prompt 的可审计一致性），
   pyproject per-file 豁免 E501（先例：`maspo/prompts.py`）。

## 3. 改动文件清单

### 新增（8 个文件）

| 文件                                                                                   | 职责                                                                                                                      |
| -------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| `src/lychee_mas/methods/processing/aggagent/prompts.py`             | system/user prompt ×3 + FINAL_MESSAGE，原版 verbatim（附出处头）                                                          |
| `src/lychee_mas/methods/processing/aggagent/tools.py`               | 四工具纯 stdlib 移植：schema（strict:True）、clamp 语义、ROUGE-L 检索、finish 变体校验                                    |
| `src/lychee_mas/methods/processing/aggagent/client.py`              | `AggClient` 协议 + `HttpAggClient`（urllib，OpenAI 兼容端点）+ `sanitize_tool_name`                                       |
| `src/lychee_mas/methods/processing/aggagent/engine.py`              | 原版 `_run` 循环移植：`run_aggregation(question, trajectories, client, ...) -> {result, error, stats, messages}`          |
| `src/lychee_mas/methods/processing/aggagent/__init__.py`            | 适配层：`trajectory_to_steps` / `extract_answer_text` / `@REGISTRY.register("aggregator", "aggagent") AggAgentAggregator` |
| `src/lychee_mas/methods/processing/aggagent/NOTICE.md`              | 移植映射表 + 行为差异 + 复现/对齐说明                                                                                     |
| `src/lychee_mas/methods/processing/aggagent/LICENSE-APACHE-2.0.txt` | Apache-2.0 原文（自原仓库拷贝）                                                                                           |
| `configs/aggregator/aggagent.yaml`                                                     | 真实实验配置（model/api_base/api_key/question/task/预算/重试，含中文注释）                                                |
| `tests/test_aggagent.py`                                                               | 35 个离线测试（LLM 全脚本化，见 §5）                                                                                      |

### 修改（4 处）

| 文件                                                    | 改动                                                                                   |
| ------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| `src/lychee_mas/layers/processing/parallel/__init__.py` | import `aggagent` 触发注册；docstring + `__all__` 补 `AggAgentAggregator`              |
| `src/lychee_mas/layers/processing/__init__.py`          | 链式导入（包 `__init__` 链：`lychee_mas/__init__` → processing → parallel → aggagent） |
| `pyproject.toml`                                        | `aggagent/prompts.py`、`aggagent/tools.py` per-file E501 豁免（逐字文本先例）          |
| `docs/DESIGN.md` + `src/lychee_mas/layers/README.md`    | §7 全景表 aggregator 行 + §4.4 文档补 aggagent                                         |

## 4. 接口形态

```python
# methods/processing/aggagent/__init__.py（注册组件）
@REGISTRY.register("aggregator", "aggagent")
class AggAgentAggregator:
    # LLM 来源二选一：
    #  A. 注入 client（AggClient 协议，mock/脚本化用）——满足 complete(messages, tools)->dict|str
    #  B. model + api_base（内置 HttpAggClient，真实实验；api_key 留空 = vLLM 惯例 Bearer EMPTY）
    def __init__(self, client=None, question=None, task="", model="", api_base=None,
                 api_key="", max_context_tokens=100 * 1024, max_iterations=100,
                 max_tries=5, timeout=60.0): ...

    def aggregate(self, trajectories: list[Trajectory]) -> Answer: ...
    #   校验输入/题面/模型配置 → 显式报错；trajectory_to_steps 适配 → run_aggregation；
    #   成功 → Answer(source="aggagent", content=..., meta={variant, reason, solution, stats})
    #   失败/超预算/服务端错误 → RuntimeError（带 stats），不静默兜底
```

```python
# mock（tests/test_aggagent.py 的 ScriptedEvidenceClient 状态机用法）
agg = AggAgentAggregator(question="题目...", client=scripted)
ans = agg.aggregate(trajectories)

# 真实实验（OpenAI 兼容 tool-calling 端点；等价 configs/aggregator/aggagent.yaml）
agg = AggAgentAggregator(question="题目...", model="Qwen3-8B",
                         api_base="http://localhost:6000/v1")
ans = agg.aggregate(trajectories)
```

同实验对拍：`processor/parallel` 的聚合器名换 `self_consistency` ↔ `aggagent`，并把聚合器
构造超参放 `aggregator_kwargs` 透传（`task`/`model` 决定 finish 变体；`task ∈ {healthbench,
researchrubrics}` → 长报告变体，model 含 qwen → "Exact Answer:" 变体）。

```python
# 离线 mock（processor 全链路接线；examples/02_aggagent_e2e.py 另有组件直调版演示）
proc = REGISTRY.create("processor", "parallel", k=5, aggregator="aggagent",
                       aggregator_kwargs={"client": scripted_client})
# 真实实验：把 client 换成 model + api_base（HttpAggClient）
proc = REGISTRY.create("processor", "parallel", k=5, aggregator="aggagent",
                       aggregator_kwargs={"model": "...", "api_base": "http://.../v1"})
```

## 5. 验证

- 新测试 `tests/test_aggagent.py` **35/35 通过**，LLM 全脚本化覆盖：
  - 注册/类别可见性；错误路径（空输入 / 非 Trajectory / 缺题面 / 未配置聚合模型 / client 缺 complete）；
  - 适配层（排序稳定性、role 直通、reasoning meta、tool_calls 透传）与四工具单元测试
    （get_solution 全/单/越界，get_segment clamp ≤5 与 start>end，search 的 role 过滤 +
    ROUGE-L 弱匹配排序 + k 上限 + 无命中串，finish XML/qwen/long_form 校验，schema）；
  - 端到端流（ScriptedEvidenceClient 状态机：survey → search(role="tool") → 只信 content
    含证据的命中 → get_segment → finish）：**少数正确反超**（7362 正确 vs 多数 7342——
    验证 evidence-over-count 核心主张）、多数正确无回归、finish 格式错重试、qwen 变体、
    long_form 报告变体、context-limit 强制 finish（max_context_tokens=1）、迭代预算耗尽、
    服务端/上下文超限哨兵、未知工具名；
  - HttpAggClient（mock urlopen：请求体断言、context-length HTTPError → 哨兵、500 重试后
    "Server error"、缺 model/api_base ValueError）。
- 全量 `pytest`：197 passed + **1 failed 预先存在且与本次无关**（
  `test_eval_benchmark_extensions::test_benchmark_roots_prefer_new_names_and_keep_legacy_fallback`，
  Windows 路径分隔符断言）+ 1 skipped。
- `ruff check src` 全绿；selfcheck `HEAVY LOADED: NONE`（纯标准库实现）；`make demo` 跑通，
  registry 快照 aggregator 列表 = `['aggagent', 'dynamicagg', 'self_consistency']`。
- **追加（同日）**：`ParallelProcessor` 增加 `aggregator_kwargs` 透传（`__init__.py`），
  让「配置换名字 + 超参」成为真实对拍路径；`tests/test_processing.py` 与
  `tests/test_aggagent.py` 各 +1 转发测试（合计 37 个 aggagent/processing 相关用例通过）；
  新增 `examples/02_aggagent_e2e.py` 离线 demo——3 条合成轨迹（真值只在 role="tool" 观测里），
  多数投票取 7342（错）vs aggagent 检索工具观测取 7362（对），打印完整决策日志与 stats。

## 6. 后续接入点

1. **真实端点回归**：接线已就绪——`ParallelProcessor` 支持 `aggregator_kwargs` 透传
   （同日追加，test_processing / test_aggagent 各补 1 个测试），`configs/processor/parallel.yaml`
   带 aggagent 用法注释，`examples/02_aggagent_e2e.py` 是离线直调版演示。剩下的步骤：
   vLLM（`--enable-auto-tool-choice` + `--tool-call-parser`）起 OpenAI 兼容端点 →
   `configs/aggregator/aggagent.yaml` 填 model/api_base → 经 `processor/parallel` 跑小样本
   真实回归；与 self_consistency 同设置对拍，报 accuracy / token / latency。
2. **task/model 覆盖**：qwen 变体与 long_form（healthbench / researchrubrics）真实触发确认。
3. **约定提示**：运行时录制轨迹时把工具观测消息标 `role="tool"`（检索过滤语义依赖），文档已注明。
4. 若日后做上下文预算 / 检索行为实验，引擎内 `count_tokens` 是近似计数（字符/4 折算工具 schema），
   超限阈值按需调 `max_context_tokens`。
