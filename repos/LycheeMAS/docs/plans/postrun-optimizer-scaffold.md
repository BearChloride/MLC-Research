# 代码修改记录 —— 运行后优化（postrun（optimize_postrun））框架搭建

> 状态：**已实施**。范围：只搭框架接缝，算法全部留桩（`NotImplementedError`），
> 具体算法实现待接。
> （归因 → 信用 → 优化 → 反哺的闭环）；接缝形态仿照已有运行前优化 `plugins/prerun/`。

---

## 1. 背景

与「运行后」对应的是**归因与训练层**（Layer 5）：执行轨迹 τ + 结果 →错误归因（MAST 分类）→ 个体级信用 → 强化学习 / 提示优化 → 反哺构建 / 剪枝 / 记忆层，完成闭环。
其中 §5.5 明确 MASPO 应作为该闭环里的 `PromptOptimizer` 直接消费归因产出的错位案例。

代码库已有「在线」运行后接缝（`post_run_plugin/attribution` 适配器：每次执行后消费单条轨迹），
但缺少与 `pre_run_optimizer` 对称的**批量离线**运行后优化接缝。本次搭建该接缝的框架骨架。

## 2. 设计

1. **与 prerun 完全对称**：新 REGISTRY 类别 `post_run_optimizer`（第 19 个类别）；
   协议 `optimize(graph, trajectories) -> graph`，**图进图出** + 多一份输入 = 执行轨迹批 τ
   （运行后独有语义：消费 τ 反哺图）。返回值必须是未编译 StateGraph，非法显式 TypeError。
2. **协议签名最小化**：超参与运行素材（LLM 回调、trainset、产物文件路径等）全部走构造参数，
   与 prerun 的 `optimize(graph)` 约定一致；轨迹校验只查形状（`list/tuple` 且全为
   `Trajectory`，**可空**——apply 模式可能不消费轨迹），语义要求（如 optimize 模式需非空）由实现决定。
3. **读写通道复用**：图的提示/邻接写回直接复用 `prerun.graphview`（`extract_view` / `rebuild`，
   节点契约一致），不重复造轮子；`_require_state_graph` 校验也从 `prerun.base` 导入共用。
4. **桩占位约定**（DESIGN.md §7）：先注册 `attribution`与
   `train`两个方法名，调用时显式 `NotImplementedError`，让消融矩阵可见、
   不静默兜底。后续填实现时接缝零改动。

## 3. 改动文件清单

### 新增（4 个文件）

| 文件                                            | 职责                                                                                                                            |
| ----------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| `src/lychee_mas/plugins/postrun（optimize_postrun）/base.py`     | `PostRunOptimizer` 协议 + `optimize_postrun(graph, trajectories, method, **kwargs)` 统一入口 + `_require_trajectories` 轨迹校验 |
| `src/lychee_mas/plugins/postrun（optimize_postrun）/stubs.py`    | `post_run_optimizer/attribution`、`post_run_optimizer/train` 注册占位（`NotImplementedError`）                                  |
| `src/lychee_mas/plugins/postrun（optimize_postrun）/__init__.py` | import 触发注册；导出 `PostRunOptimizer` / `optimize_postrun`                                                                   |
| `tests/test_postrun（optimize_postrun）.py`                      | 8 个离线测试（需要 langgraph，未装则整文件跳过）                                                                                |

### 修改（5 处）

| 文件                                 | 改动                                                                                           |
| ------------------------------------ | ---------------------------------------------------------------------------------------------- |
| `src/lychee_mas/core/registry.py`    | CATEGORIES 18 → 19 个，追加 `post_run_optimizer`                                               |
| `src/lychee_mas/plugins/__init__.py` | 链式 import `postrun（optimize_postrun）`（触发注册）+ 包说明 + `__all__`                                       |
| `docs/DESIGN.md`                     | §3.1 类别清单（19 个）、§5 新增 postrun（optimize_postrun） 接缝说明、§7 组件全景表新增 `post_run_optimizer` 行 |
| `src/lychee_mas/plugins/README.md`   | 新增「postrun（optimize_postrun） — LangGraph 原生运行后优化」章节 + 已注册组件表两行                           |
| `README.md`                          | 目录树 plugins 段补 postrun（optimize_postrun） 一行                                                            |

## 4. 接口形态

```python
# plugins/postrun（optimize_postrun）/base.py
@runtime_checkable
class PostRunOptimizer(Protocol):
    """运行后优化器协议：传入未编译 StateGraph + 轨迹批，返回优化后的 StateGraph。"""
    def optimize(self, graph: Any, trajectories: Any) -> Any: ...

def optimize_postrun(graph, trajectories, method, **kwargs) -> Any:
    """统一入口：method 按名分发到 REGISTRY 的 post_run_optimizer 类别。

    sg = optimize_postrun(sg, taus, method="attribution",
                          mode="apply", prompt_file="p.json")      # 即插即用（待接）
    sg = optimize_postrun(sg, taus, method="attribution", mode="optimize",
                          attributor="all_at_once", evaluator_llm=...)  # 离线闭环（待接）
    app = sg.compile()
    """
    _require_state_graph(graph, "optimize_postrun")       # 复用 prerun.base
    taus = _require_trajectories(trajectories, "optimize_postrun")
    optimizer = REGISTRY.create("post_run_optimizer", method, **kwargs)
    out = optimizer.optimize(graph, taus)
    _require_state_graph(out, f"post_run_optimizer/{method}.optimize 的返回值")
    return out
```

## 5. 验证

- 新测试 `tests/test_postrun（optimize_postrun）.py`：8/8 通过（类别可见性、桩显式报错、未知 method KeyError、
  图/轨迹/返回值三处 TypeError 校验、空轨迹合法、注册类清理不污染快照）。
- 相关测试（registry / prerun / postrun（optimize_postrun））：23/23 通过。
- 全量 `pytest`：162 passed，**1 failed 预先存在且与本次无关**——
  `test_eval_benchmark_extensions::test_benchmark_roots_prefer_new_names_and_keep_legacy_fallback`
  （Windows 路径分隔符断言 `'data\\benchmarks\\raw'` vs `'data/benchmarks/raw'`，位于 eval/ 层）。
- `ruff check src` 全绿；selfcheck `HEAVY LOADED: NONE`（langgraph 惰性导入不破）；demo 跑通。

## 6. 后续接入点（填实现时的固定动作）

1. **P0 数据源头**：`attributor` / `credit_assigner` 类别目前全为桩，先实现
   `attributor/all_at_once`（可验证规则 + LLM judge，MAST 标签，测试语料见
   `eval/benchmarks/mast_data.py`）与 `credit_assigner/attribution_guided`——`post_run_plugin/attribution`
   适配器随之从桩变真。
2. **P1 填 `post_run_optimizer/attribution` 桩**：`extract_view` → 归因坏案例 → 反思出提示
   （复用 `prerun/maspo` 的提示模板与 sanitize）→ `rebuild` 写回 `spec.system_prompt`；
   optimize 落 prompt JSON / apply 即插即用，双模式与 prerun/maspo 同构。
3. **P2 填 `post_run_optimizer/train` 桩**：归因信用（trace）→ `Trainer`（train/）→ 参数/产物落文件。
4. 每个实现按 CLAUDE.md 六步配方收尾：config + 测试（LLM 全脚本化离线）+ 三件套全绿；
   组件状态变化时同步 DESIGN.md §7 全景表与 plugins/README.md。
