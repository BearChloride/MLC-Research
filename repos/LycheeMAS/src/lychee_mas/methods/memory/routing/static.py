"""L0 静态路由器 —— 固定的 (role × task) -> channel 策略表。必做 baseline。

同时提供固定通道路由（always none/nl/latent/both）：它们本质就是「策略恒定」的 StaticRouter，
注册为 `memory_router/fixed`（用 `channel=` 选具体通道）。

注册：
  - memory_router/static   按 (role,task) 查表
  - memory_router/fixed    恒定某通道（always-<channel> 消融）
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

from ....core.registry import REGISTRY
from .base import Channel, MemoryRouter, RouteDecision, RouterInputs


@REGISTRY.register("memory_router", "static")
class StaticRouter(MemoryRouter):
    name = "static"

    def __init__(self, table: Optional[Dict[Tuple[str, str], Channel]] = None,
                 default: Channel = "nl"):
        # table 以 (role, task) 为键；查不到回退 (role, "*")，再查不到用 default
        self.table = table or {}
        self.default = default

    def decide(self, x: RouterInputs) -> RouteDecision:
        # 三级查表：精确 (role,task) -> 通配 (role,"*") -> 全局 default
        ch = (self.table.get((x.role, x.task))
              or self.table.get((x.role, "*"))
              or self.default)
        d = RouteDecision(channel=ch, reason=f"static[{x.role},{x.task}]")
        return self._enforce_availability(x, d)  # 末尾兜底硬约束 #2（latent 不可用则回退 NL）


@REGISTRY.register("memory_router", "fixed")
class FixedChannelRouter(StaticRouter):
    """always-<channel> 消融：无表、default 恒为某通道，于是所有 agent/轮次都走该通道。

    用 channel= 选具体通道（none/nl/latent/both）；name 反映所选通道，便于结果区分。
    """

    def __init__(self, channel: Channel = "nl"):
        super().__init__(default=channel)
        self.name = f"always_{channel}"


def fixed_channel_router(channel: Channel) -> FixedChannelRouter:
    """便捷构造 always-<channel> 路由器（内部走 FixedChannelRouter）。"""
    return FixedChannelRouter(channel=channel)
