"""MASPO 联合提示优化（pre_run_optimizer/maspo）。

- prompts.py    vendored MATH 提示资产（保留原始引用；上游无 LICENSE，见文件头声明）
- textops.py    sanitize / extract_answer / parse_comparison / <prompt> 抽取
- executor.py   带缓存图执行器（InferenceCache + 局部重执行，优化内循环用）
- optimizer.py  MASPOOptimizer（apply / optimize 两模式，fixed-rounds 论文主线）

import 本包触发 `pre_run_optimizer/maspo` 注册（纯标准库）。
"""
from __future__ import annotations

from .executor import AsyncLLM, CachedExecutor, InferenceCache
from .optimizer import AgentOptState, MASPOOptimizer

__all__ = [
    "MASPOOptimizer",
    "AgentOptState",
    "CachedExecutor",
    "InferenceCache",
    "AsyncLLM",
]
