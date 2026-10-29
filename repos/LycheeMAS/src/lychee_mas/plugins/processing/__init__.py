"""processing —— 执行接缝（REGISTRY 类别 ``processor`` + ``aggregator``）。

- base.py  run_processed 统一入口 + Processor/TrajectoryAggregator 协议 re-export
"""
from __future__ import annotations

from .base import (
    ProcessingResult,
    Processor,
    Runner,
    TrajectoryAggregator,
    run_processed,
)

__all__ = ["run_processed", "Processor", "TrajectoryAggregator", "ProcessingResult", "Runner"]
