"""剪枝层协议（CLAUDE.md §5）。

- GraphPruner   网络剪枝（逐轮邻接淘汰等）：输入一张图 + 上下文，输出剪枝后的图（或 mask）。
- VocabAdapter  模型级词表降本（结构感知词表适配）：把 agent 模型词表收窄以降成本。

实现一个新组件 = 实现协议 + `@REGISTRY.register(category, name)` + 加 config + 加 test。
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class GraphPruner(Protocol):
    """网络剪枝：给定图（MASGraph）与可选轨迹/上下文，产出剪枝后的图。"""

    def prune(self, graph: Any, context: Any = None) -> Any: ...


@runtime_checkable
class VocabAdapter(Protocol):
    """词表适配：对给定 agent/模型产出一个降本的词表子集或映射。"""

    def adapt(self, agent: Any, context: Any = None) -> Any: ...
