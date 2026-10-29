"""组件注册表（插件机制核心，CLAUDE.md §4）。

每个算法 = 注册一个类 + 由配置选择，绝不硬编码实现。新增方法即 `@REGISTRY.register(cat, name)`，
不改编排器。`memory_router` 类别承接「触发接缝」（路由决策），与 `memory_manager`（方法接缝
）对称。
"""
from __future__ import annotations

from typing import Any, Callable, TypeVar

T = TypeVar("T")

CATEGORIES = (
    "graph_builder", "agent_selector",                         # build 接缝
    "pre_run_optimizer", "graph_pruner", "vocab_adapter",      # prerun 接缝
    "memory_manager", "memory_router",                         # memory 接缝
    "processor", "aggregator",                                 # processing 接缝
    "attributor", "credit_assigner", "trainer",
    "post_run_optimizer",                                      # postrun 接缝
    "benchmark", "optimizer",                                  # eval / 离线 compile（gepa）
)


class Registry:
    def __init__(self) -> None:
        self._items: dict[str, dict[str, type]] = {}

    def register(self, category: str, name: str) -> Callable[[type[T]], type[T]]:
        def deco(cls: type[T]) -> type[T]:
            bucket = self._items.setdefault(category, {})
            if name in bucket:
                raise KeyError(f"组件已存在: {category}/{name}")
            bucket[name] = cls
            return cls

        return deco

    def get(self, category: str, name: str) -> type:
        try:
            return self._items[category][name]
        except KeyError as e:
            avail = ", ".join(self.list(category)) or "(空)"
            raise KeyError(f"未找到组件 {category}/{name};该类别可用: {avail}") from e

    def create(self, category: str, name: str, **kwargs: Any) -> Any:
        return self.get(category, name)(**kwargs)

    def list(self, category: str) -> list[str]:
        return sorted(self._items.get(category, {}).keys())

    def snapshot(self) -> dict[str, list[str]]:
        return {c: self.list(c) for c in sorted(self._items)}


REGISTRY = Registry()
