"""图剪枝（GraphPruner 接缝）。

已实现：
  graph_pruner/agentprune        AgentPrune 时空掩码剪枝（ICLR 2025，见 agentprune.py）
  graph_pruner/agentdropout      AgentDropout 动态节点/边淘汰（ACL 2025，
                                 见 agentdropout/optimizer.py）

占位桩（统一报错文案 `<name>: not wired yet (TODO)`，能被 REGISTRY.list 看到）：
  graph_pruner/agentdropout_v2   运行时在线淘汰（AgentDropoutV2，在研）

迁移逻辑时：实现 prune(graph, context) -> 剪枝后图，并保留原始引用/许可证说明。
"""
from __future__ import annotations

from typing import Any

from ...core.registry import REGISTRY
from .agentprune import AgentPrunePruner


class _StubPruner:
    name = "pruner"

    def __init__(self, **kwargs):
        self.cfg = kwargs

    def prune(self, graph: Any, context: Any = None) -> Any:
        raise NotImplementedError(f"{self.name}: not wired yet (TODO)")


@REGISTRY.register("graph_pruner", "agentdropout_v2")
class AgentDropoutV2(_StubPruner):
    name = "agentdropout_v2"  # 运行时在线淘汰（在研）


__all__ = ["AgentDropoutV2", "AgentPrunePruner"]
