"""外部基线记忆管理器（CLAUDE.md §5 消融矩阵：mem0 | ama）。

这些是「占位槽」：各自把一个外部记忆系统包到同一个 MemoryManager 接口背后
（observe/recall -> MemoryBundle，走 NL 通道）。接外部对比时再实现。
保留为显式桩，让消融矩阵在代码里可见。统一报错文案：`<name>: not wired yet (TODO)`。

注册为 `memory_manager/mem0`、`memory_manager/ama`。
"""
from __future__ import annotations

from typing import Any, List

from ....core.registry import REGISTRY
from ..base import MemoryBundle, MemoryManager
from ..routing.base import RouteDecision


class _NotImplementedManager(MemoryManager):
    """桩基类：observe 空操作，recall 直接抛错（提醒尚未接线）。"""

    def observe(self, messages: List[Any]) -> None:
        return

    def recall(self, decision: RouteDecision, query: str) -> MemoryBundle:
        raise NotImplementedError(f"{self.name}: not wired yet (TODO)")


@REGISTRY.register("memory_manager", "mem0")
class Mem0Manager(_NotImplementedManager):
    name = "mem0"  # TODO: 包 mem0 存储；NL 召回 -> bundle.NL_Channel


@REGISTRY.register("memory_manager", "ama")
class AMAManager(_NotImplementedManager):
    name = "ama"  # TODO: 包 AMA
