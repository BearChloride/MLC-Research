# `core` — 公共类型 + 组件注册表（共享基座）

## 功能

一切的地基：所有模块共享的**统一图抽象类型**（`types.py`）与**插件注册机制**（`registry.py`）。纯标准库 dataclass，零重依赖；模块之间只通过这里的类型交互（层内专用契约留在各层 `base.py`，不进 core）。

## 接口

### `types.py` — 公共类型（纯 dataclass）

| 类型 | 含义 | 关键字段/属性 |
|---|---|---|
| `AgentSpec` | 图节点 = 智能体画像 | `id/name/role/system_prompt/model/tools/profile/meta` |
| `Message` | 一条通信消息（边上的一次传输） | `sender/content/receiver/round/role/prompt_tokens/completion_tokens`；`.tokens` 汇总开销 |
| `Answer` | 候选答案（聚合单元） | `content/source/confidence/score/meta` |
| `Trajectory` | 一次执行 τ | `task_id/messages/candidates/final_answer`；`add(msg)`、`.total_tokens`、`.num_rounds` |
| `TaskQuery` | 输入查询 | `question/context/gold`（gold 供可验证奖励/评测） |
| `Budget` / `BudgetUnit` | 预算约束 | `limit` + 单位（`TOKENS/CALLS/USD`） |

### `registry.py` — 组件注册表

```python
from lychee_mas.core.registry import REGISTRY, CATEGORIES

@REGISTRY.register("aggregator", "my_agg")   # 注册（重名报错）
class MyAgg: ...

REGISTRY.get("aggregator", "my_agg")         # 取类（未找到时报错并列出可用项）
REGISTRY.create("aggregator", "my_agg", k=5) # 实例化（kwargs 透传构造器）
REGISTRY.list("aggregator")                  # 该类别所有已注册名
REGISTRY.snapshot()                          # {category: [names]} 全量快照
```

已登记 `CATEGORIES`（**17 个**，新增类别须在此同步登记）：
`runtime, model_client, agent_selector, topology_generator, graph_pruner, vocab_adapter, memory_manager, memory_router, aggregator, processor, attributor, credit_assigner, trainer, benchmark, pre_run_plugin, post_run_plugin, optimizer`。

## 约定

- 本包被任何模块 import 都必须零开销、零重依赖。
- 新增公共类型前先确认它确实被 ≥2 个模块共享，否则放到该模块的 `base.py`。
