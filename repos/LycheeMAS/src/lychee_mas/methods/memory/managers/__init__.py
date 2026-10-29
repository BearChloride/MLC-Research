"""记忆方法（manager 接缝）：导入各 manager 以触发 REGISTRY 注册。

`memory_manager` 类别：cdm（CDM 默认，在研）/ mem0 / ama（外部基线桩）。
"""
from __future__ import annotations

from .DualChannelMemory import DualChannelMemoryManager
from .external import AMAManager, Mem0Manager

__all__ = ["DualChannelMemoryManager", "Mem0Manager", "AMAManager"]
