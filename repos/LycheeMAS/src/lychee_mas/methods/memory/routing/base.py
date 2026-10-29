"""MemoryRouter 抽象基类 + 决策/输入契约。这就是 「触发算法接缝」（CLAUDE.md §5）。

要加新触发算法：继承 MemoryRouter 实现 decide() + `@REGISTRY.register("memory_router", name)`。
其余一切（MAS / 注入 client / 评测）消费的是 RouteDecision，不需任何改动。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Literal, Optional

Channel = Literal["none", "nl", "latent", "both"]  # 四种通道（核心决策空间）
CHANNELS = ("none", "nl", "latent", "both")


@dataclass
class RouterInputs:
    """路由器可以条件化的全部信息（决策空间）。"""

    role: str  # 当前 agent 角色：manager | worker | verifier
    task: str  # 任务名：gsm8k | locomo10 | ...
    turn: int  # 该 agent 在本对话中的轮次（0 起）
    sender: Optional[str] = None  # 产出「待消费消息」的角色（收发对的「发」）
    receiver: Optional[str] = None  # 将消费本 agent 输出的角色（若已知；收发对的「收」）
    query: str = ""  # 当前 query / 最近一条消息内容
    availability: Dict[str, bool] = field(default_factory=dict)  # 通道 -> 是否可用
    # 收发双方是否共享 hidden 空间？（约束 #2：latent 能否跨 agent 传）
    same_model_pair: bool = True


@dataclass
class RouteDecision:
    channel: Channel = "nl"  # 选定的通道（路由器只决定用哪个通道）
    reason: str = ""  # 决策理由（落盘日志 / 可解释性用）

    def uses_latent(self) -> bool:
        return self.channel in ("latent", "both")

    def uses_nl(self) -> bool:
        return self.channel in ("nl", "both")


class MemoryRouter(ABC):
    """把 RouterInputs 映射成 RouteDecision。默认无状态；子类可带学习参数。"""

    name: str = "router"  # 子类覆盖，用于日志/落盘命名

    @abstractmethod
    def decide(self, x: RouterInputs) -> RouteDecision: ...

    def _enforce_availability(self, x: RouterInputs, d: RouteDecision) -> RouteDecision:
        """硬约束（CLAUDE.md §4）：latent 不可用、或收发是异构（未对齐）agent 对时，回退到 NL。

        子类在 decide() 末尾应调用本方法兜底，保证不会把 latent 发给读不了它的 agent。
        """
        if d.uses_latent():
            # latent 可用 = 该通道被标为可用 且 收发为同模型对（hidden 对齐）
            latent_ok = x.availability.get("latent", True) and x.same_model_pair
            if not latent_ok:
                # latent / both 一律降级为 nl（both 丢掉 latent 分量，保留 NL）
                d.channel = "nl" if d.uses_nl() or d.channel == "latent" else d.channel
                d.reason = (d.reason + " | latent->nl (unavailable/heterogeneous pair)").strip(" |")
        return d
