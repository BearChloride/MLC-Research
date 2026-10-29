"""运行时多维度多表征记忆管理（Memory，CLAUDE.md §0）。

把记忆拆成两个对称的可替换接缝：
- 方法接缝（manager）：记忆如何存储/召回/物化各通道——`memory_manager` 类别（cdm/mem0/ama）。
- 触发接缝（router）：当前 agent 本轮用哪个通道——`memory_router` 类别（
static/fixed/learned/soft_gate）。

通道实现见 channels/（NL + Latent）；跨 agent 共享状态见 context.py（RoutingContext，包顶层）。
import 本包会触发所有 manager/router 的注册，但不触发 torch（惰性导入，黄金法则 4）。
"""
from __future__ import annotations

from .base import MemoryBundle, MemoryManager
from .channels import PREV_OUTPUT_HEADER, LatentMemory, NLMemory
from .context import RoutingContext
from .managers import AMAManager, DualChannelMemoryManager, Mem0Manager
from .routing import (
    CHANNELS,
    Channel,
    FixedChannelRouter,
    LearnedRouter,
    MemoryRouter,
    RouteDecision,
    RouterInputs,
    SoftGateRouter,
    StaticRouter,
    fixed_channel_router,
)
from .store import MemoryStore  # 从原 stores/ 迁入：记忆缓存接缝

__all__ = [
    "MemoryManager",
    "MemoryBundle",
    "MemoryStore",
    "NLMemory",
    "LatentMemory",
    "PREV_OUTPUT_HEADER",
    "DualChannelMemoryManager",
    "Mem0Manager",
    "AMAManager",
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
    "RoutingContext",
]
