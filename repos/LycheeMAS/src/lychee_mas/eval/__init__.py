"""eval：评测 / 基准 harness（CLAUDE.md §9）。

- benchmarks/   数据 loaders + benchmark 注册（benchmark/<task>，惰性加载）
- metrics.py    accuracy/token/latency 等评分 + 结果落盘（math/yaml 惰性导入）
- task_config.py 每个 task 的默认队伍 + 答案提取策略

import 本包会触发 benchmark 注册（不读盘、不导入 datasets/sympy）。
"""
from __future__ import annotations

from . import (
    benchmarks,  # noqa: F401  触发 benchmark/<task> 注册
    metrics,
    task_config,
)

__all__ = ["benchmarks", "metrics", "task_config"]
