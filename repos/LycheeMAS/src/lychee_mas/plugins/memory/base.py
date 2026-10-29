"""记忆接缝（memory）—— 统一入口 ``attach_memory``：把记忆注入挂载进图的每个 agent 节点。

设计已定（five-module-plugin-refactor.md §4-1）：``attach_memory(sg, method, backend, **kw)``
**重包节点 runnable**，在每次 agent 发言前后执行注入六步（observe → route → recall →
system 段注入 → 生成 → 记账），对建图方零侵入。算法库在 ``methods/memory/``
（channels / managers / routing），方法接缝 memory_manager 与触发接缝 memory_router 沿用。

当前状态：挂载语义属 P3 全新实现项（旧 runtime 注入路径已随重构删除，git 历史可考）；
本入口先立接缝、显式报错，不静默假实现。
"""
from __future__ import annotations

from typing import Any

from ...methods.memory.base import MemoryBundle, MemoryManager  # noqa: F401  协议 re-export
from ...methods.memory.routing.base import MemoryRouter, RouteDecision  # noqa: F401


def attach_memory(sg: Any, method: str = "cdm", **kwargs: Any) -> Any:
    """把 memory_manager/<method> + memory_router 包裹进契约图的每个 agent 节点（P3）。"""
    raise NotImplementedError(
        "attach_memory: 记忆挂载（注入六步的节点包裹）在 P3 全新实现——设计已定为重包节点 "
        "runnable（旧注入引擎实现见 git 历史 runtime/injection.py）")
