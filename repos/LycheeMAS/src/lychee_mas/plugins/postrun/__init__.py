"""postrun —— 运行后接缝
（类别 ``attributor`` / ``credit_assigner`` / ``post_run_optimizer`` / ``trainer``）。

- base.py  三入口：analyze_run（读侧归因）+ optimize_postrun（图+轨迹→图闭环）
           + train_from_runs（写侧训练）；PostRunOptimizer/Trainer 协议
"""
from __future__ import annotations

from .base import (
    Attribution,
    CreditAssigner,
    FailureAttributor,
    PostRunOptimizer,
    TraceStore,
    Trainer,
    analyze_run,
    optimize_postrun,
    train_from_runs,
)

__all__ = [
    "analyze_run",
    "optimize_postrun",
    "train_from_runs",
    "PostRunOptimizer",
    "Trainer",
    "Attribution",
    "FailureAttributor",
    "CreditAssigner",
    "TraceStore",
]
