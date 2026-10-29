# `plugins` — 五模块接缝的接口层

## 定位

**plugins/ 定义接缝，methods/ 存方法**：本包只放协议、统一入口、`method` 按名分发、返回校验与薄适配（目标 <300 行/文件）；论文级算法（复现、训练循环、缓存执行器）一律在姊妹包 `methods/`（按接缝镜像分组），注册装饰器随实现走。

## 五个接缝

| 接缝 | 统一入口 | REGISTRY 类别 | 说明 |
|---|---|---|---|
| 构建 | `build_langgraph(method, node_factory, state_schema, ...) -> sg` | `graph_builder`(+`agent_selector`/`topology_generator`) | 产出契约 StateGraph；节点语义由 `node_factory(spec, is_terminal)` 注入 |
| 运行前 | `optimize_langgraph(sg, method, **kw) -> sg` | `pre_run_optimizer`(+`graph_pruner`/`vocab_adapter`) | 图进图出；`mode="optimize"` 离线产物化 / `mode="apply"` 即插即用 |
| 记忆 | `attach_memory(sg, method, backend, **kw) -> sg` | `memory_manager`+`memory_router` | 注入六步重包进 agent 节点（P3 实现中，显式桩） |
| 执行 | `run_processed(runner, method, **kw) -> ProcessingResult` | `processor`+`aggregator` | serial 1 次 / parallel×K + 归约（pass@K 承载点） |
| 运行后 | `analyze_run(...)` / `optimize_postrun(sg, taus, method)` / `train_from_runs(...)` | `attributor`+`credit_assigner` / `post_run_optimizer` / `trainer` | 读侧归因；图+轨迹→图闭环；写侧离线训练 |

## 目录结构

五接缝一缝一子目录（与 `methods/` 镜像）：`build/`、`prerun/`、`memory/`、`processing/`、`postrun/`——各自 `base.py` 装协议与统一入口，`__init__.py` 再导出并触发注册。

## 节点契约（`prerun/graphview.py`，五接缝互操作的唯一约定）

```python
sg.add_node(name, node_fn, metadata={"agent_spec": spec})   # spec: core.types.AgentSpec
# spec.system_prompt        = 可变异提示模板（{question}/{context} 占位）
# spec.meta["predecessors"] = 通信前驱（节点函数据此从 state 选 context）
# 节点函数运行时从 spec 读——闭包与元数据共享同一对象，优化器改 spec 即改行为
```

`extract_view(sg)` 提取视图（缺元数据 / 非 DAG / 多终端显式报错）；`rebuild(sg, prompts=…/adjacency=…)` 写回。只支持静态 DAG + 唯一终端。

## 用法

```python
from lychee_mas.plugins import (build_langgraph, optimize_langgraph,
                                run_processed, analyze_run)

sg = build_langgraph(method="static", node_factory=make_node,
                     state_schema=MyState, team="default")
sg = optimize_langgraph(sg, method="maspo", mode="apply", prompt_file="p.json")
sg = optimize_langgraph(sg, method="agentprune", state_file="state.json")  # 换算法=换 method
app = sg.compile()
result = await run_processed(runner, method="parallel", k=8,
                             aggregator="self_consistency")   # 或 aggregator="aggagent"
```

端到端示例见 `examples/01_five_seams_demo.py`（`make demo`）。

## 约定

- 纯标准库注册；langgraph 一律惰性导入（`make selfcheck` 须 HEAVY LOADED: NONE）。
- 接缝入口对非法输入显式 raise（未知 method → KeyError 列可用名；返回类型强校验）；桩组件的 `NotImplementedError` 如实上抛。
- 新增方法：在 `methods/<接缝>/` 实现 + 注册 + 该包 `__init__` 触发 + config + test（六步配方见 CLAUDE.md §4.1），**不改本包任何文件**。
