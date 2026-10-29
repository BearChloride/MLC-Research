"""Trace —— 执行轨迹的归因/信用协议（从 attribute_train 拆出；CLAUDE.md §5/§10）。

本包 = 「读」执行轨迹的一侧：错误归因 + 信用分配 + TraceStore（轨迹/决策落点，见 store.py）。
训练（Trainer）拆到姊妹包 `lychee_mas.train`（「写」参数/提示的一侧）。

- FailureAttributor  错误归因：把一条失败轨迹定位到「哪个 agent / 哪一步」出错（Attribution 列表）。
- CreditAssigner     信用分配：把归因转成 per-agent 稠密信用（核心贡献：attribution_guided）。

接口先稳定，产出的信用喂给 `lychee_mas.train` 的 Trainer。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ...core.types import Trajectory


@dataclass
class Attribution:
    """一次归因结果：定位到某 agent / 某步，附理由与置信度。"""

    agent: str = ""
    step: int = -1
    is_fault: bool = False
    reason: str = ""
    confidence: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class FailureAttributor(Protocol):
    """错误归因：给定一条（失败）轨迹，产出 Attribution 列表。"""

    def attribute(self, trajectory: Trajectory, context: Any = None) -> list[Attribution]: ...


@runtime_checkable
class CreditAssigner(Protocol):
    """信用分配：把归因 + 标量奖励转成 per-agent 信用（agent -> float）。"""

    def credits(self, attributions: list[Attribution], reward: float) -> dict[str, float]: ...
