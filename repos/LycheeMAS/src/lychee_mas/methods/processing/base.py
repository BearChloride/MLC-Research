"""处理层协议（Processing，CLAUDE.md §5）—— 决定「跑几次 MAS + 如何得到最终 Answer」。

给定一个 **runner**（`async () -> Trajectory`：把 MAS 在当前 query 上跑一次、产一条轨迹），
`Processor` 决定调用它几次、以及如何把结果归约成一个 `Answer`。两种模式 = processing 的两个子模块：

- **serial（串行）**：只执行**一次**，产出**一条**轨迹，直接返回该轨迹的 `final_answer`（无聚合）。
- **parallel（并行）**：并发执行 **K 次**，产出 **K 条**轨迹，再用 `aggregator`（多轨迹聚合）
  归约成一个 `Answer`。

`aggregator` 类别（`TrajectoryAggregator`：多轨迹→一个 Answer）是**并行**处理器的可插拔归约策略
（self_consistency / dynamicagg），供 `ParallelProcessor` 调用。

实现新组件 = 实现对应协议 + `@REGISTRY.register(<category>, name)` + 加 config + 加 test。
"""
from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from ...core.types import Answer, Trajectory

# runner：把 MAS 在当前 query 上跑一次、产一条轨迹（调用方负责每次产出独立轨迹，如 ctx.reset）
Runner = Callable[[], Awaitable[Trajectory]]


@dataclass
class ProcessingResult:
    """处理结果：最终 Answer + 本次产出的所有轨迹（serial=1 条；parallel=K 条）。"""

    answer: Answer
    trajectories: list[Trajectory] = field(default_factory=list)


@runtime_checkable
class Processor(Protocol):
    """处理策略：给定 runner，决定跑几次 + 如何归约（serial 跑 1 次 / parallel 跑 K 次）。"""

    async def run(self, runner: Runner) -> ProcessingResult: ...


@runtime_checkable
class TrajectoryAggregator(Protocol):
    """多轨迹聚合：list[Trajectory] -> Answer（并行处理器的归约策略）。子类可另接受 Answer 列表。"""

    def aggregate(self, trajectories: list[Trajectory]) -> Answer: ...
