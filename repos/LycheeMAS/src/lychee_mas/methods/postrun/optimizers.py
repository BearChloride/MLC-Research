"""post_run_optimizer 的注册占位实现（算法未接，调用时显式 NotImplementedError）。

占位约定（DESIGN.md §6）：桩能被 REGISTRY.list 看到是有意为之——占好名字、
让消融矩阵在代码里可见；调用时显式报错，不静默兜底、不写死假结果。

已占的两个方法（运行后图闭环，经 plugins/postrun.optimize_postrun 挂载）：
- ``attribution``：归因引导提示精炼（轻量闭环：归因坏案例 → 反思出新提示写回图）；
- ``train``：训练闭环（重闭环：归因信用 → RL/提示训练 → 产物经 prerun apply 挂载）。
"""
from __future__ import annotations

from typing import Any, Sequence

from ...core.registry import REGISTRY
from ...core.types import Trajectory


class _StubPostRunOptimizer:
    name = "post_run_optimizer"

    def __init__(self, **kwargs: Any) -> None:
        self.cfg = kwargs

    def optimize(self, graph: Any, trajectories: Sequence[Trajectory]) -> Any:
        raise NotImplementedError(f"{self.name}: not wired yet (TODO)")


@REGISTRY.register("post_run_optimizer", "attribution")
class AttributionPostRunOptimizer(_StubPostRunOptimizer):
    """归因引导的运行后优化（桩）：归因 → 坏案例 → 提示精炼写回图。"""

    name = "attribution"


@REGISTRY.register("post_run_optimizer", "train")
class TrainPostRunOptimizer(_StubPostRunOptimizer):
    """训练闭环（桩）：归因信用 → Trainer → 产物经 prerun apply 挂载回图。"""

    name = "train"
