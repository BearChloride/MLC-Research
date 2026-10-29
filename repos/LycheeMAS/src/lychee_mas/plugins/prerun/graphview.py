"""GraphView —— StateGraph ↔ 运行前优化器 的唯一读写通道。

**节点契约**（建图方遵守，即插即用的机制所在）：

- 每个 agent 节点以 ``sg.add_node(name, node_fn, metadata={"agent_spec": spec})`` 注册，
  ``spec`` 为 ``core.types.AgentSpec``；
- ``spec.system_prompt`` = 可变异提示模板（MASPO 语义：含 ``{question}`` / ``{context}`` 占位）；
- ``spec.meta["predecessors"]`` = 该节点的**通信前驱**节点名列表（节点函数据此从 state 里选
  上游输出拼 context）；缺省时通信结构 = StateGraph 的边（去掉 START/END）；
- ``node_fn`` 运行时从 ``spec`` 读模板/前驱——**节点闭包与 metadata 持有同一 spec 对象**，
  因此 ``rebuild(prompts=...)`` 改 ``spec.system_prompt`` 即改执行行为，无需重建节点函数。

约束（显式报错，不静默降级）：只支持**静态 DAG**（无条件分支 branches）、恰好一个终端节点
（唯一有边指向 END 的节点）；metadata 缺 ``agent_spec`` 即报错。

langgraph 一律惰性导入（黄金法则 2）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ...core.types import AgentSpec
from ...methods.prerun.graphops import topological_order


@dataclass
class GraphView:
    """StateGraph 的 agent 视图：名字（拓扑序）+ spec + 通信前驱 + 终端节点名。"""

    names: List[str]                     # 全部 agent 节点名，按执行拓扑序
    specs: Dict[str, AgentSpec]          # name -> AgentSpec（与节点闭包共享同一对象）
    predecessors: Dict[str, List[str]]   # name -> 通信前驱名列表（决定 context 来源）
    terminal: str                        # 终端节点名（其输出为最终答案）

    def successors(self, name: str) -> List[str]:
        return [n for n in self.names if name in self.predecessors[n]]

    def all_successors(self, name: str) -> set[str]:
        """name 的全部传递后继（沿通信边）。"""
        out: set[str] = set()
        frontier = [name]
        while frontier:
            cur = frontier.pop()
            for nxt in self.successors(cur):
                if nxt not in out:
                    out.add(nxt)
                    frontier.append(nxt)
        return out


def _exec_edges(graph: Any) -> tuple[Dict[str, List[str]], List[str], str]:
    """从 StateGraph 边表提取（执行邻接, 入口列表, 终端名）；非法结构显式报错。"""
    from langgraph.graph import END, START  # 惰性

    adj: Dict[str, List[str]] = {name: [] for name in graph.nodes}
    entries: List[str] = []
    terminals: List[str] = []
    for src, dst in sorted(graph.edges):
        if src == START:
            entries.append(dst)
        elif dst == END:
            terminals.append(src)
        else:
            adj[src].append(dst)
    if len(terminals) != 1:
        raise ValueError(
            f"prerun 需要恰好一个终端节点（有边指向 END），"
            f"得到 {len(terminals)} 个: {terminals}")
    return adj, entries, terminals[0]


def extract_view(graph: Any) -> GraphView:
    """从未编译 StateGraph 提取 GraphView（只读；spec 对象与图共享）。"""
    from .base import _require_state_graph

    _require_state_graph(graph, "extract_view")
    if dict(graph.branches):
        raise ValueError("prerun 只支持静态 DAG 边：图里存在条件分支（branches），无法优化")
    if not graph.nodes:
        raise ValueError("extract_view: 图里没有任何节点")

    specs: Dict[str, AgentSpec] = {}
    for name, node in graph.nodes.items():
        spec = (node.metadata or {}).get("agent_spec")
        if not isinstance(spec, AgentSpec):
            raise ValueError(
                f"节点 {name!r} 缺 metadata['agent_spec']（AgentSpec）——建图方需按 graphview "
                "节点契约挂元数据，prerun 才能读写提示/邻接")
        specs[name] = spec

    exec_adj, _entries, terminal = _exec_edges(graph)

    # 执行拓扑序：把名字映射为下标，复用 agentprune 的确定性 Kahn（成环显式在此暴露为破环，
    # 但 StateGraph 的静态边成环会在 compile 时死循环，这里直接显式拒绝）
    names_decl = list(graph.nodes)
    idx = {n: i for i, n in enumerate(names_decl)}
    edge_set = {(idx[s], idx[d]) for s, ds in exec_adj.items() for d in ds}
    if _has_cycle(len(names_decl), edge_set):
        raise ValueError("prerun 只支持 DAG：StateGraph 的执行边成环")
    order_idx, _preds = topological_order(len(names_decl), edge_set)
    names = [names_decl[i] for i in order_idx]

    # 通信前驱：优先 spec.meta["predecessors"]（剪枝等已写回的结构），否则用执行边
    predecessors: Dict[str, List[str]] = {}
    for name in names:
        meta_preds = specs[name].meta.get("predecessors")
        if meta_preds is not None:
            unknown = [p for p in meta_preds if p not in specs]
            if unknown:
                raise ValueError(f"节点 {name!r} 的 meta.predecessors 含未知节点 {unknown}")
            predecessors[name] = list(meta_preds)
        else:
            predecessors[name] = [s for s, ds in exec_adj.items() if name in ds]
    return GraphView(names=names, specs=specs, predecessors=predecessors, terminal=terminal)


def _has_cycle(n: int, edges: set[tuple[int, int]]) -> bool:
    indeg = {i: 0 for i in range(n)}
    for _s, d in edges:
        indeg[d] += 1
    stack = [i for i in range(n) if indeg[i] == 0]
    seen = 0
    while stack:
        cur = stack.pop()
        seen += 1
        for s, d in edges:
            if s == cur:
                indeg[d] -= 1
                if indeg[d] == 0:
                    stack.append(d)
    return seen != n


def rebuild(graph: Any, *, prompts: Optional[Dict[str, str]] = None,
            adjacency: Optional[Dict[str, List[str]]] = None) -> Any:
    """把优化结果写回图。

    - ``prompts``：name -> 新提示模板。**原地写** ``spec.system_prompt``（节点闭包与
      metadata 持有同一 spec 对象，这正是不重建节点函数就能换提示的机制）；未知节点名
      显式 KeyError。
    - ``adjacency``：name -> 通信后继名列表（新的通信结构）。产**新** StateGraph：
      按确定性拓扑序（methods/prerun/graphops.topological_order：Kahn + 破环）
      把节点连成线性执行链
      （START → ... → 终端 → END），并把每个节点实际保留的通信前驱写回
      ``spec.meta["predecessors"]``。终端节点保持终端：其出边被忽略。
    - 两者都给：先写提示，再重连边。都不给：原图返回。
    """
    view = extract_view(graph)

    if prompts is not None:
        unknown = [n for n in prompts if n not in view.specs]
        if unknown:
            raise KeyError(f"prompts 含图中不存在的节点 {unknown}；图节点: {view.names}")
        for name, text in prompts.items():
            if not isinstance(text, str) or not text.strip():
                raise ValueError(
                    f"节点 {name!r} 的新提示为空：拒绝注入（显式失败，不静默保留旧提示）")
            view.specs[name].system_prompt = text

    if adjacency is None:
        return graph

    from langgraph.graph import END, START, StateGraph  # 惰性

    unknown = [n for n in adjacency if n not in view.specs]
    if unknown:
        raise KeyError(f"adjacency 含图中不存在的节点 {unknown}；图节点: {view.names}")

    # 终端保持终端（出边忽略）；名字 ↔ 下标映射时把终端放到最大下标，确保线性化后位于链尾
    names = [n for n in view.names if n != view.terminal] + [view.terminal]
    idx = {n: i for i, n in enumerate(names)}
    edge_set = {(idx[s], idx[d]) for s, ds in adjacency.items() for d in ds
                if s != view.terminal and s != d}
    order_idx, final_preds = topological_order(len(names), edge_set)
    order = [names[i] for i in order_idx]
    if order[-1] != view.terminal:  # 破环 Kahn 以小下标优先，终端下标最大 ⇒ 必然收尾；防御性校验
        raise RuntimeError(f"内部错误：线性化后终端 {view.terminal!r} 不在链尾: {order}")

    new_sg = StateGraph(graph.state_schema)
    for name in order:
        node = graph.nodes[name]
        view.specs[name].meta["predecessors"] = sorted(names[p] for p in final_preds[idx[name]])
        new_sg.add_node(name, node.runnable, metadata=dict(node.metadata or {}))
    new_sg.add_edge(START, order[0])
    for a, b in zip(order, order[1:]):
        new_sg.add_edge(a, b)
    new_sg.add_edge(order[-1], END)
    return new_sg
