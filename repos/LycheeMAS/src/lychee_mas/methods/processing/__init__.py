"""处理层（Processing，CLAUDE.md §0）—— 决定「跑几次 MAS + 如何得到最终 Answer」，两个子模块：

- **serial（串行）**：`processor/serial` —— 只执行**一次**，产出一条轨迹，返回其 final_answer。
- **parallel（并行）**：`processor/parallel` —— 并发执行 **K 次**，产出 K 条轨迹，用 `aggregator`
  （`self_consistency` / `aggagent` / `dynamicagg`）聚合成一个 Answer。

（本层由原 `aggregate` 层重命名而来；`aggregator` 类别是并行处理器的可插拔归约策略，保持不变。）
import 本包触发两个子模块的注册（纯标准库，不触发重依赖；aggagent 子包同理）。
"""
from __future__ import annotations

from .base import ProcessingResult, Processor, Runner, TrajectoryAggregator
from .parallel import AggAgentAggregator, DynamicAggregator, ParallelProcessor, SelfConsistencyVote
from .serial import SerialProcessor

__all__ = [
    "Processor",
    "ProcessingResult",
    "Runner",
    "TrajectoryAggregator",
    "SerialProcessor",
    "ParallelProcessor",
    "SelfConsistencyVote",
    "DynamicAggregator",
    "AggAgentAggregator",
]
