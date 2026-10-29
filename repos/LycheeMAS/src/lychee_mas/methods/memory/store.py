"""MemoryStore —— 记忆方法的简单持久化/缓存接缝（原 stores/ 迁入 memory 包）。

P0 阶段提供一个纯标准库的 key->value 内存存储，供记忆管理器缓存物化结果（如 latent prefix 缓存的
框架级落点）。具体记忆方法（DualChannel 等）目前自带缓存；本 store 是统一接口，便于后续替换为
向量库 / 磁盘缓存而不动业务层。
"""
from __future__ import annotations

from typing import Any


class MemoryStore:
    """最简 key->value 存储（带可选条目上限的 FIFO 淘汰）。"""

    def __init__(self, capacity: int | None = None) -> None:
        self.capacity = capacity
        self._d: dict[Any, Any] = {}
        self._order: list[Any] = []

    def put(self, key: Any, value: Any) -> None:
        if key not in self._d:
            self._order.append(key)
            if self.capacity is not None and len(self._order) > self.capacity:
                old = self._order.pop(0)
                self._d.pop(old, None)
        self._d[key] = value

    def get(self, key: Any, default: Any = None) -> Any:
        return self._d.get(key, default)

    def has(self, key: Any) -> bool:
        return key in self._d

    def clear(self) -> None:
        self._d.clear()
        self._order.clear()

    def __len__(self) -> int:
        return len(self._d)
