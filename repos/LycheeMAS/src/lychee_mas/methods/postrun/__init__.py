"""Trace —— 执行轨迹的归因/信用 + 落点（从 attribute_train 拆出，CLAUDE.md §0）。

「读」执行轨迹的一侧：
  attributor/{all_at_once, step_by_step, binary_search}   错误归因（Who&When / MAST 对标）
  credit_assigner/attribution_guided                       归因引导信用分配（核心贡献）
  TraceStore                      轨迹/决策落点（消息级 + JSONL，见 store.py）

训练（trainer）拆到姊妹包 `lychee_mas.train`。注册占位实现（`<name>: not wired yet (TODO)`），
但能被 REGISTRY.list 看到。import 本包触发注册（不触发 torch/RL 库）。
"""
from __future__ import annotations

from typing import Any

from ...core.registry import REGISTRY
from ...core.types import Trajectory
from . import optimizers  # noqa: F401  触发 post_run_optimizer/{attribution,train} 桩注册
from .base import (
    Attribution,
    CreditAssigner,
    FailureAttributor,
)
from .store import TraceStore


class _StubAttributor:
    name = "attributor"

    def __init__(self, **kwargs):
        self.cfg = kwargs

    def attribute(self, trajectory: Trajectory, context: Any = None) -> list[Attribution]:
        raise NotImplementedError(f"{self.name}: not wired yet (TODO)")


@REGISTRY.register("attributor", "all_at_once")
class AllAtOnceAttributor(_StubAttributor):
    name = "all_at_once"


@REGISTRY.register("attributor", "step_by_step")
class StepByStepAttributor(_StubAttributor):
    name = "step_by_step"


@REGISTRY.register("attributor", "binary_search")
class BinarySearchAttributor(_StubAttributor):
    name = "binary_search"


@REGISTRY.register("credit_assigner", "attribution_guided")
class AttributionGuidedCredit:
    """归因引导信用分配（核心贡献，桩）：把归因转为 per-agent 稠密信用喂给 train。"""

    name = "attribution_guided"

    def __init__(self, **kwargs):
        self.cfg = kwargs

    def credits(self, attributions: list[Attribution], reward: float) -> dict[str, float]:
        raise NotImplementedError("attribution_guided: not wired yet (TODO)")


__all__ = [
    "FailureAttributor",
    "CreditAssigner",
    "Attribution",
    "TraceStore",
    "AllAtOnceAttributor",
    "StepByStepAttributor",
    "BinarySearchAttributor",
    "AttributionGuidedCredit",
]
