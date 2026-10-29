# 开发实例走读：以 MASPO 为教材

> 新人最快的上手方式是看一个**真实方法**如何从论文变成本框架里可挂载、可测试、可跑分的组件。
> MASPO（联合提示优化，ICML 2026）是最好的教材——它踩过框架的每一个部件：methods/plugins 分层、
> optimize/apply 两段式、LLM 回调注入、断点续跑、脚本化测试、真实实验与打分。
> 读本文时请对照源码；配方总纲见 [`ONBOARDING.md`](ONBOARDING.md) §5，接缝契约见 [`DESIGN.md`](DESIGN.md)。

---

## 0. 方法本身，30 秒

MASPO 把「联合优化 MAS 里每个 agent 的提示词」做成免 gold 标注的搜索：反思 LLM 提议新提示 →
三路 LLM **成对比较**打分（本节点输出 Local / 直接后继 Lookahead / 终端答案 Global，加权 0.4/0.4/0.2）→
每个 agent 进化 beam search → 坐标上升轮转 + Beam Refresh 重锚定。产物 = 一份
`{"prompts": {节点名: 优化后提示}}` 的 JSON。

**关键观察**：产物是 prompt map ⇒ 天然是「运行前把图的提示换掉」的 prerun 方法；
训练素材自给（自己造 rollout）⇒ 训练循环跟方法走，放 `methods/prerun/`。

## 1. 代码落位：一张表看懂分层

| 文件 | 行数 | 装什么 | 为什么在这一层 |
|---|---|---|---|
| `methods/prerun/maspo/optimizer.py` | ~635 | 注册类 `MASPOOptimizer`：apply/optimize 两模式、fixed-rounds 主循环、beam search、Beam Refresh、checkpoint | **算法本体 = methods**；`@REGISTRY.register("pre_run_optimizer", "maspo")` 打在这里 |
| `methods/prerun/maspo/executor.py` | ~129 | `CachedExecutor`：带缓存图执行 + **局部重执行**（换某节点提示只重跑它和后继） | 优化内循环的省钱机器，属方法私有 |
| `methods/prerun/maspo/prompts.py` | ~166 | 原版提示词**逐字 vendored**（反思/三路比较/压缩模板）+ 出处声明 | 外部资产隔离一个文件，改动可审计 |
| `methods/prerun/maspo/textops.py` | ~108 | 占位符 sanitize、`<answer>`/`\boxed` 抽取、比较结果解析 | 纯文本工具，独立可测 |
| `plugins/prerun/`（base + graphview） | — | 统一入口 `optimize_langgraph` + 节点契约读写 | **接缝 = plugins**：MASPO 没有往这里加过一行方法逻辑 |
| `configs/prerun/maspo.yaml` | — | 论文模式超参默认值 | 对照实验只改配置 |
| `tests/test_maspo.py` + `test_prerun.py` | — | 离线测试（LLM 全脚本化） | 见 §4 |
| `scripts/run_maspo_langgraph.py` | — | 三条腿实验：baseline / optimize / eval | 见 §5 |

方向纪律：methods **不在顶层 import plugins**（optimizer 里对 graphview 的引用是函数内惰性导入，
避免包环）；plugins 不含算法。

## 2. 走读一：统一入口如何吃掉一个方法

用户侧永远只有一句话（换方法 = 换 `method` 字符串）：

```python
from lychee_mas.plugins import optimize_langgraph

sg = optimize_langgraph(sg, method="maspo", mode="apply", prompt_file="p.json")
```

入口做三件事（`plugins/prerun/base.py`）：校验传入的是**未编译** StateGraph →
`REGISTRY.create("pre_run_optimizer", "maspo", **kwargs)` 构造方法实例 → 调 `optimize(graph)`
并强校验返回值仍是 StateGraph。未知 method 由 REGISTRY 显式 KeyError 并列出可用名。

方法侧的 `optimize()` 是 apply/optimize 的分岔口（`optimizer.py:161`，节选）：

```python
def optimize(self, graph):
    from ....plugins.prerun.graphview import extract_view, rebuild  # 函数内导入避免包内环
    view = extract_view(graph)                       # 图 → 视图（名字/spec/邻接/终端）
    if self.mode == "apply":
        prompt_map = self._load_prompt_file()        # 轻量：读产物 JSON
    else:
        if not self.trainset:
            raise ValueError("maspo mode=optimize 需要非空 trainset（训练问题列表）")
        if self._raw_agent_llm is None or self._raw_evaluator_llm is None:
            raise ValueError("maspo mode=optimize 需要 agent_llm 与 evaluator_llm 回调")
        prompt_map, statistics = asyncio.run(self._optimize_with_limits(view))  # 重活
        if self.prompt_file:
            self._save(prompt_map, statistics)       # 产物落盘（供 apply / 复现）
    return rebuild(graph, prompts=prompt_map)        # 写回图（原地写 spec，即插即用）
```

三个值得模仿的点：**① 显式报错**（缺 trainset/回调/种子提示直接 raise，绝不静默）；
**② 产物文件是两段式的桥**（optimize 写、apply 读，格式在 docstring 写死）；
**③ 写回走 graphview.rebuild**，不自己碰 StateGraph 内部。

## 3. 走读二：方法怎么用 LLM（回调注入，不 import 模型库）

`methods/` 里全程见不到 `import transformers/openai`。MASPO 声明三路回调（类型就是
`async (prompt: str) -> str`）：

```python
MASPOOptimizer(...,
    agent_llm=...,       # 执行端（跑 rollout 的模型，实验里是本地 Qwen3-8B）
    evaluator_llm=...,   # 比较端（temperature=0）
    proposer_llm=...)    # 反思提议端（temperature=0.7，缺省回退 evaluator）
```

由实验脚本用 `backends/` 构造后注入（`scripts/run_maspo_langgraph.py` 的
`make_agent_llm` / `make_evaluator_llms`：本地 HF 直连或 OpenAI 兼容 API + 重试，二选一）。
好处：换模型/换端点/换成测试里的脚本化假 LLM，方法代码一行不动。

顺带看两个工程细节（都是踩过坑后加的，你的方法大概率也需要）：

- **断点续跑**：每个 agent 轮次后把优化现场原子落盘 `*_ckpt.json`，重启自动恢复
  （`_load_or_seed_states` / `_save_ckpt`）——17 小时的优化曾因 API 额度耗尽整场丢失，从此有了它。
- **并发限流**：`asyncio.Semaphore` 在每次 `optimize()` 的事件循环内包装回调
  （`_optimize_with_limits`）——Semaphore 不能跨 loop 复用，建在构造器里会炸。

## 4. 走读三：测试怎么写（LLM 一律脚本化）

`tests/test_maspo.py` 的核心道具是两个假 LLM——按提示模板的**特征串**分派回复：

```python
class ScriptedEvaluatorLLM:
    """评估/反思端脚本：按模板特征分派——反思回 <prompt>，比较按预设 A/B 回答。"""
    async def __call__(self, prompt):
        if "optimizing a prompt" in prompt:      # 反思模板 → 回一个候选提示
            return "<analyse>ok</analyse><prompt>Improved. {question} {context}</prompt>"
        if "more conducive" in prompt:           # Local/Lookahead 比较 → "A" 或 "B"
            return self.replies["local"]
        ...
```

有了它就能离线断言**算法性质**而不是 mock 调用次数：

- 评分公式：三路全赢 → score=+0.5；全输 → −0.5；Local 赢 Global 输 → 错位率=1 且收 misleading cases；
- beam 一步：好候选带累计分入 beam，坏候选原节点占位；
- 局部重执行：换 reflector 提示只多 1 次 LLM 调用、predictor 读缓存；
- 断点续跑：第 8 次调用后故意抛错 → checkpoint 落盘 → 新实例恢复跑完 → 现场文件自动删除；
- 显式报错路径**必测**（缺 prompt_file、空 trainset、反思返回空文本…）。

需要 langgraph 的整图测试放 `tests/test_prerun.py`，文件顶部
`pytest.importorskip("langgraph")`——环境没装时整文件跳过而不是红。

## 5. 走读四：实验与结果（推理/打分分离）

`scripts/run_maspo_langgraph.py` 三条腿，全部落统一目录结构（samples + summary + config 快照）：

```bash
# ① baseline：种子提示直接跑分（不需要评估端）
CUDA_VISIBLE_DEVICES=0 python scripts/run_maspo_langgraph.py --phase baseline --eval-n 100
# ② optimize + ③ eval：联合优化落 prompt JSON → apply 挂载跑分
CUDA_VISIBLE_DEVICES=0,1 python scripts/run_maspo_langgraph.py --phase both \
    --train-n 50 --eval-n 100 --evaluator-model-path /path/to/Qwen3-32B
```

MATH-500 复现结果（100 题、同口径 @4096）：

| | 原始打分 | 归一化打分 | prompt tok/题 |
|---|---|---|---|
| 基线（种子提示） | 0.76 | 0.78 | 981 |
| MASPO（gemini 评估端） | 0.77 | 0.85 | 1892 |
| MASPO（本地 Qwen3-32B 评估端，修正配置） | **0.80** | **0.85** | 1428 |

两个方法学教训（写进了 DESIGN §5，也应写进你的实验设计）：
**免 gold 的判官优化会漂移答案表面形式**（Unicode 极简写法骗过了字面打分器——归一化重打分才见真差距）；
**保真核查要对照参考实现逐参数做**（我们曾漏了"反思端 temperature=0.7"与"生成上限 4096"两处，重跑才拿到干净数字）。

## 6. 把这套复制到你的方法：核对清单

| 你需要 | 抄 MASPO 的哪里 |
|---|---|
| 注册 + apply/optimize 分岔 + 显式报错 | `optimizer.py` 的 `__init__` / `optimize()` |
| 外部提示词/资产 vendored 规范 | `prompts.py` 文件头（出处 + 许可声明 + 逐字原则）；许可证范例另见 `methods/processing/aggagent/NOTICE.md` |
| LLM 回调注入 + 本地/API 双后端 | 脚本的 `make_agent_llm` / `make_evaluator_llms` |
| 长时训练断点续跑 | `_save_ckpt` / `_load_or_seed_states` |
| 脚本化 LLM 测试 | `tests/test_maspo.py` 的 Scripted* 两个类 |
| 图级挂载测试 + 与协议路径对拍 | `tests/test_prerun.py` |
| 实验脚本骨架（多 phase + 统一落盘） | `scripts/run_maspo_langgraph.py` |
| 与原版的声明差异怎么写 | `optimizer.py` 模块 docstring 的「与原版的声明差异」段 |

最后一步永远相同：`make lint && make test && make selfcheck && make demo` 四绿，才算完。
