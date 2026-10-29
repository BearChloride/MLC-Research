"""MemoryManager = 记忆「方法」接缝（CLAUDE.md §5，主战场）。

一个 manager 负责：存储 + 召回 + 各通道如何被物化：
  - observe(messages): 从目前的对话历史更新内部记忆库
  - recall(decision, query): 按路由器选的通道 + 成本旋钮，产出一个 MemoryBundle
    （NL_Channel / Latent_Channel 及各自的 strategy）
  - reset(): 清空单次对话的状态

要加一种新记忆方法（mem0 / AMA / LatentMem），继承 MemoryManager +
`@REGISTRY.register("memory_manager", name)`。
注入 client 与 MAS 消费的是 MemoryBundle，不需任何改动（接缝隔离）。

⚠️ 惰性导入：本文件不在 import 时触发 torch；`Latent_Channel` 只携带运行期对象（torch 张量 /
projector 栈），核心骨架零运行依赖（黄金法则 4）。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from .routing.base import RouteDecision


@dataclass
class MemoryBundle:
    """本轮要注入的东西：每个通道一对 (Channel=内容, strategy=方法)；空通道则该对为 None。

    - NL_Channel / NL_strategy         : NL 通道注入文本 + 方法（prev_output / simplemem）
    - Latent_Channel / Latent_strategy : latent 载荷（soft prefix 张量 或 projector 栈）+ 方法
      （soft_token / c2c）；注入 client 据 Latent_strategy 决定注入方式（prefix 拼接 vs KV 融合）。
    - meta                             : 附带元信息（所选 channel、latent_kind 等）
    """

    NL_Channel: Optional[str] = None  # NL 通道要注入的文本
    NL_strategy: Optional[str] = None  # 产出它的 NL 方法（prev_output / simplemem …）
    Latent_Channel: Optional[Any] = None  # latent 载荷：soft prefix 张量 或 projector 栈
    Latent_strategy: Optional[str] = None  # 产出它的 latent 方法（soft_token / c2c），决定注入方式
    meta: dict = field(default_factory=dict)  # 附带元信息（如所选 channel、latent_kind）


class MemoryManager(ABC):
    name: str = "memory"  # 子类覆盖，用于日志/落盘目录命名

    @abstractmethod
    def observe(self, messages: list[Any]) -> None:
        """从对话历史（role/content 字典列表）更新记忆库。每轮 create() 调一次。"""

    @abstractmethod
    def recall(self, decision: RouteDecision, query: str) -> MemoryBundle:
        """把路由器选定的通道物化成 MemoryBundle（真正决定注入什么）。"""

    def reset(self) -> None:
        pass  # 默认无状态；有状态的子类（如 DualChannel）覆盖它
