"""GEPA 反思式提示演化：pareto（纯函数选择核心）+ optimizer（compile 主循环）。

import 本包触发 `optimizer/gepa` 注册（纯标准库，不触发重依赖）。
"""
from __future__ import annotations

from .optimizer import GEPAOptimizer, Reflector, Rollout
from .pareto import dominated, pareto_pool, per_instance_best, sample_candidate

__all__ = [
    "GEPAOptimizer",
    "Rollout",
    "Reflector",
    "per_instance_best",
    "pareto_pool",
    "dominated",
    "sample_candidate",
]
