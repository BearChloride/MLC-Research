"""TraceStore：消息 hook 累积 + 决策日志 + reset。"""
from __future__ import annotations

from lychee_mas.core.types import Message
from lychee_mas.methods.memory import MemoryStore
from lychee_mas.methods.postrun import TraceStore


def test_trace_store_collects_messages():
    ts = TraceStore()
    ts.hook(Message(sender="a", content="x"))
    ts.hook(Message(sender="b", content="y"))
    assert len(ts) == 2
    assert ts.messages[0].sender == "a"


def test_trace_store_decisions_and_reset():
    ts = TraceStore()
    ts.log_decision({"role": "worker", "channel": "nl"})
    assert len(ts.decisions) == 1
    ts.reset()
    assert len(ts) == 0 and len(ts.decisions) == 0


def test_memory_store_put_get_capacity():
    ms = MemoryStore(capacity=2)
    ms.put("a", 1)
    ms.put("b", 2)
    ms.put("c", 3)  # 触发 FIFO 淘汰最早的 "a"
    assert ms.get("a") is None
    assert ms.get("b") == 2 and ms.get("c") == 3
    assert len(ms) == 2
