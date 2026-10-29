"""LangGraph 原生「运行前优化」统一接口（REGISTRY 类别 ``pre_run_optimizer``）。

设计要求（docs/plans/maspo-lg-prerun-plan.md）：
1. **统一接口**：一个 ``method`` 参数确定调用哪个算法（maspo / agentprune / ...）；
2. **图进图出**：传入**未编译**的 LangGraph ``StateGraph``，在图上优化并返回一个 ``StateGraph``；
3. **即插即用**：``mode="apply"`` 加载离线产物直接改图（轻量、每 query 可调）；
   ``mode="optimize"`` 在图上跑完整优化（重活、脚本驱动一次）。

与 ``pre_run_plugin``（MASGraph 接缝，挂在 Orchestrator 上）**并存**：本接缝是 LangGraph
主线的运行前优化入口，旧接缝零回归保留。

黄金法则约定：langgraph 只在函数内部惰性导入（``make selfcheck`` 必须 HEAVY LOADED: NONE）；
优化器对不支持的输入/配置显式 raise，不静默降级。
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from ...core.registry import REGISTRY


@runtime_checkable
class PreRunOptimizer(Protocol):
    """运行前优化器协议：传入未编译的 StateGraph，返回优化后的 StateGraph。

    实现类经 ``@REGISTRY.register("pre_run_optimizer", <method>)`` 注册；
    超参与运行素材（LLM 回调、trainset、产物文件路径等）全部走构造参数，
    ``optimize`` 保持「图进图出」的最小签名。
    """

    def optimize(self, graph: Any) -> Any: ...


def _require_state_graph(obj: Any, where: str) -> None:
    """校验 obj 是未编译的 langgraph.graph.StateGraph（惰性导入，显式报错）。"""
    from langgraph.graph import StateGraph  # 惰性：仅在真正走图接口时触达

    if isinstance(obj, StateGraph):
        return
    hint = ""
    if hasattr(obj, "builder"):  # CompiledStateGraph 持有 .builder
        hint = "（看起来是已编译的图：请传编译前的 builder，优化后再 .compile()）"
    raise TypeError(f"{where} 需要未编译的 langgraph StateGraph，得到 {type(obj).__name__}{hint}")


def optimize_langgraph(graph: Any, method: str, **kwargs: Any) -> Any:
    """统一入口：按 ``method`` 从 REGISTRY 取运行前优化器，在 LangGraph 图上优化。

    用法示例::

        sg = optimize_langgraph(sg, method="maspo", mode="apply", prompt_file="p.json")
        sg = optimize_langgraph(sg, method="maspo", mode="optimize",
                                trainset=[...], agent_llm=..., evaluator_llm=...)
        sg = optimize_langgraph(sg, method="agentprune", state_file="state.json")
        app = sg.compile()

    未知 method 由 REGISTRY 显式 KeyError（并列出可用名）；返回值必须仍是
    StateGraph，否则显式 TypeError（与 pre_run_plugin 的 before_run 校验同款约定）。
    """
    _require_state_graph(graph, "optimize_langgraph")
    optimizer = REGISTRY.create("pre_run_optimizer", method, **kwargs)
    out = optimizer.optimize(graph)
    _require_state_graph(out, f"pre_run_optimizer/{method}.optimize 的返回值")
    return out
