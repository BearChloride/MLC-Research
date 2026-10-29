"""路由（触发接缝）：导入各路由器以触发 REGISTRY 注册。

`memory_router` 类别：static / fixed / learned / soft_gate（CLAUDE.md §5）。
（跨 agent 共享状态 RoutingContext 在 `memory/context.py`，不属于本子包。）
"""
from __future__ import annotations

from .base import CHANNELS, Channel, MemoryRouter, RouteDecision, RouterInputs
from .learned import LearnedRouter
from .soft_gate import SoftGateRouter
from .static import FixedChannelRouter, StaticRouter, fixed_channel_router

__all__ = [
    "MemoryRouter",
    "RouterInputs",
    "RouteDecision",
    "Channel",
    "CHANNELS",
    "StaticRouter",
    "FixedChannelRouter",
    "fixed_channel_router",
    "LearnedRouter",
    "SoftGateRouter",
]
