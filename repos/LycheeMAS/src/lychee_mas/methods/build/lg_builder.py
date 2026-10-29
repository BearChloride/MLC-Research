"""graph_builder/static —— 按 team 模板产出契约 StateGraph（build 接缝的默认实现）。

节点函数由调用方以 ``node_factory(spec, is_terminal) -> 节点函数`` 注入（生成后端、
状态读写属实验语义，构建层不越权）；本类负责：AgentSpec 链（team 模板或显式 agents）、
``meta["predecessors"]`` 链式通信结构、``metadata={"agent_spec": spec}`` 元数据挂载、
START→…→END 线性执行边。langgraph 惰性导入。
"""
from __future__ import annotations

from typing import Any, Callable, List, Optional

from ...core.registry import REGISTRY
from ...core.types import AgentSpec
from .templates import team_to_agentspecs

NodeFactory = Callable[[AgentSpec, bool], Any]  # (spec, is_terminal) -> LangGraph 节点函数


@REGISTRY.register("graph_builder", "static")
class StaticGraphBuilder:
    """静态链构建器：team 模板（或显式 AgentSpec 列表）→ 契约 StateGraph。"""

    name = "static"

    def __init__(self, node_factory: NodeFactory, state_schema: Any,
                 team: str = "default", model: Optional[str] = None,
                 agents: Optional[List[AgentSpec]] = None) -> None:
        if node_factory is None or state_schema is None:
            raise ValueError("graph_builder/static 需要 node_factory 与 state_schema"
                             "（节点语义与状态形状由实验方定义）")
        self.node_factory = node_factory
        self.state_schema = state_schema
        self.team = team
        self.model = model
        self.agents = list(agents) if agents else None  # 选队器（agentinit）产物可直接传入

    def build(self) -> Any:
        from langgraph.graph import END, START, StateGraph  # 惰性

        specs = self.agents or team_to_agentspecs(self.team, model=self.model)
        if not specs:
            raise ValueError(f"team {self.team!r} 产出空团队：无法建图")
        names = [s.name for s in specs]
        sg = StateGraph(self.state_schema)
        for i, spec in enumerate(specs):
            spec.meta.setdefault("predecessors", names[i - 1:i])  # 链式通信结构
            sg.add_node(spec.name, self.node_factory(spec, i == len(specs) - 1),
                        metadata={"agent_spec": spec})
        sg.add_edge(START, names[0])
        for a, b in zip(names, names[1:]):
            sg.add_edge(a, b)
        sg.add_edge(names[-1], END)
        return sg
