"""pre_run_optimizer/agentprune —— 把既有 graph_pruner/agentprune 接上 LangGraph 统一接口。

薄适配器：算法（logits/masks/threshold 实现）全部复用 ``methods/prerun/agentprune.py``，
本类只做「视图取邻接 → threshold 实现矩阵 → rebuild 写回」。与 MASGraph 路径
（``pre_run_plugin/prune`` + ``AgentPrunePruner.prune``）语义一致，测试对拍见
``tests/test_prerun.py``。

agent 下标映射约定：``i = extract_view(graph).names 的第 i 个``（执行拓扑序），与训练脚本
建图顺序一致时可直接加载其 state_file。
"""
from __future__ import annotations

from typing import Any, Optional

from ...core.registry import REGISTRY
from ...methods.prerun.agentprune import AgentPrunePruner


@REGISTRY.register("pre_run_optimizer", "agentprune")
class AgentPruneLG:
    """AgentPrune 时空掩码剪枝（LangGraph 统一接口版，apply 语义：确定性 threshold 实现）。"""

    name = "agentprune"

    def __init__(self, state_file: Optional[str] = None, **pruner_kwargs: Any) -> None:
        # n_agents 在 optimize() 时从图推断；其余超参与 state_file 透传给 AgentPrunePruner
        self.state_file = state_file
        self.pruner_kwargs = dict(pruner_kwargs)
        self.last_meta: Optional[dict] = None  # 最近一次剪枝的矩阵/存活统计（脚本落盘用）

    def optimize(self, graph: Any) -> Any:
        from .graphview import extract_view, rebuild  # 函数内导入避免包内环

        view = extract_view(graph)
        pruner = AgentPrunePruner(n_agents=len(view.names), state_file=self.state_file,
                                  **self.pruner_kwargs)
        sm, tm = pruner.realized_matrices("threshold")
        names = view.names
        adjacency = {names[i]: [names[j] for j in range(len(names)) if sm[i][j]]
                     for i in range(len(names))}
        self.last_meta = {
            "spatial": sm, "temporal": tm,
            "alive_spatial": sum(v for row in sm for v in row),
            "alive_temporal": sum(v for row in tm for v in row),
            "names": names,
        }
        return rebuild(graph, adjacency=adjacency)
