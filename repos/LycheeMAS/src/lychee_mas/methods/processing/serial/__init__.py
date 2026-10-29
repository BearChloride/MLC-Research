"""处理层 · **串行子模块**（Processor 接缝）——只执行一次，产出一条轨迹。

- processor/serial  串行：跑 1 次 MAS → 1 条轨迹 → 直接返回该轨迹的 final_answer（无聚合）。

与并行子模块（`processor/parallel`）对称：串行=单次；并行=多次并发 + 聚合。
import 本包触发 `processor` 类别（serial）注册（纯标准库，不触发重依赖）。
"""
from __future__ import annotations

from ....core.registry import REGISTRY
from ....core.types import Answer
from ..base import ProcessingResult, Runner


@REGISTRY.register("processor", "serial")
class SerialProcessor:
    """串行处理：只调用 runner 一次，产出一条轨迹，返回其 final_answer（无候选则空 Answer）。"""

    name = "serial"

    def __init__(self, **kwargs):
        self.cfg = kwargs

    async def run(self, runner: Runner) -> ProcessingResult:
        traj = await runner()
        ans = traj.final_answer or (traj.candidates[-1] if traj.candidates else Answer(content=""))
        return ProcessingResult(answer=ans, trajectories=[traj])


__all__ = ["SerialProcessor"]
