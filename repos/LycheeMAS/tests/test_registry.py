"""注册表：注册/查询/重复报错/snapshot 含关键类别。"""
from __future__ import annotations

import pytest
from lychee_mas.core.registry import REGISTRY, Registry


def test_register_and_get():
    r = Registry()

    @r.register("aggregator", "foo")
    class Foo:
        def __init__(self, k=1):
            self.k = k

    assert r.get("aggregator", "foo") is Foo
    inst = r.create("aggregator", "foo", k=5)
    assert inst.k == 5
    assert r.list("aggregator") == ["foo"]


def test_duplicate_raises():
    r = Registry()

    @r.register("runtime", "a")
    class A:
        pass

    with pytest.raises(KeyError):
        @r.register("runtime", "a")
        class B:
            pass


def test_get_missing_raises():
    r = Registry()
    with pytest.raises(KeyError):
        r.get("runtime", "nope")


def test_global_snapshot_has_key_categories():
    snap = REGISTRY.snapshot()
    # memory_router 是本次重构新增的关键类别（CLAUDE.md §5）
    assert "memory_router" in snap
    assert "static" in snap["memory_router"]
    # cdm 记忆方法 + 五接缝关键类别 + 可跑的 self_consistency 聚合器
    assert "cdm" in snap["memory_manager"]
    assert "static" in snap["graph_builder"]          # build 接缝
    assert "maspo" in snap["pre_run_optimizer"]       # prerun 接缝
    assert "self_consistency" in snap["aggregator"]   # processing 接缝
