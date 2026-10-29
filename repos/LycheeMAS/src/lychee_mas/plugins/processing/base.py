"""执行接缝（processing）—— 统一入口 ``run_processed``：决定一个任务跑几次、如何归约。

``runner``（``async () -> Trajectory``）由调用方提供：把编译后的图在当前 query 上跑一次、
组装一条轨迹。processor 决定调用次数与归约（serial 1 次 / parallel 并发 K 次 +
aggregator 投票）——pass@K 的承载点。实现在 ``methods/processing/``。
"""
from __future__ import annotations

from typing import Any

from ...core.registry import REGISTRY
from ...methods.processing.base import (  # noqa: F401  协议 re-export
    ProcessingResult,
    Processor,
    Runner,
    TrajectoryAggregator,
)


async def run_processed(runner: Runner, method: str = "serial",
                        **kwargs: Any) -> ProcessingResult:
    """统一入口：``REGISTRY.create("processor", method, **kwargs).run(runner)``。

    parallel 的可插拔归约经 kwargs 透传（如 ``aggregator="self_consistency", k=8``）。
    """
    processor = REGISTRY.create("processor", method, **kwargs)
    result = await processor.run(runner)
    if not isinstance(result, ProcessingResult):
        raise TypeError(
            f"processor/{method}.run 必须返回 ProcessingResult，得到 {type(result).__name__}")
    return result
