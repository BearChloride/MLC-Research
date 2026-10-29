"""构建接缝（build）—— 统一入口 ``build_langgraph``：按 method 产出契约 StateGraph。

契约（graphview 节点契约）：每个 agent 节点 ``add_node(name, fn,
metadata={"agent_spec": spec})``；``spec.system_prompt`` 为提示模板、
``spec.meta["predecessors"]`` 为通信前驱；节点函数运行时从 spec 读。
产出的图可直接进 prerun / memory 接缝或 compile 执行。

实现（graph_builder 类别）在 ``methods/build/``：``static``（team 模板链）；
选队器（agent_selector/agentinit）产出 AgentSpec 列表后经 ``agents=`` 传入。
"""
from __future__ import annotations

from typing import Any

from ...core.registry import REGISTRY
from ...methods.build.base import AgentSelector, GraphBuilder  # noqa: F401  协议 re-export


def build_langgraph(method: str = "static", **kwargs: Any) -> Any:
    """统一入口：``REGISTRY.create("graph_builder", method, **kwargs).build()``。

    返回值必须是未编译 StateGraph 且满足节点契约（经 extract_view 校验，非法显式报错）。
    """
    from ..prerun.base import _require_state_graph
    from ..prerun.graphview import extract_view

    builder = REGISTRY.create("graph_builder", method, **kwargs)
    sg = builder.build()
    _require_state_graph(sg, f"graph_builder/{method}.build 的返回值")
    extract_view(sg)  # 契约校验：缺 agent_spec / 非 DAG / 多终端在此显式暴露
    return sg
