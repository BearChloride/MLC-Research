"""共享基座：统一图抽象类型 + 组件注册表（CLAUDE.md §4）。"""
from __future__ import annotations

from .registry import CATEGORIES, REGISTRY, Registry
from .types import (
    AgentSpec,
    Answer,
    Budget,
    BudgetUnit,
    Message,
    TaskQuery,
    Trajectory,
)

__all__ = [
    "REGISTRY",
    "Registry",
    "CATEGORIES",
    "AgentSpec",
    "Message",
    "Answer",
    "Trajectory",
    "TaskQuery",
    "Budget",
    "BudgetUnit",
]
