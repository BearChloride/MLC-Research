"""memory —— 运行时记忆接缝（REGISTRY 类别 ``memory_manager`` + ``memory_router``）。

- base.py  attach_memory 统一入口（P3 全新实现中，显式桩）+ 协议 re-export
"""
from __future__ import annotations

from .base import MemoryBundle, MemoryManager, MemoryRouter, RouteDecision, attach_memory

__all__ = ["attach_memory", "MemoryManager", "MemoryRouter", "MemoryBundle", "RouteDecision"]
