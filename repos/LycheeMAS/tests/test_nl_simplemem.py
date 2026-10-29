"""NL 通道：prev_output 不变 + simplemem 策略（用 stub 的 SimpleMem，无需真装重依赖）。"""
from __future__ import annotations

import sys
import types

import pytest
from lychee_mas.methods.memory.channels.nl import MEMORY_HEADER, PREV_OUTPUT_HEADER, NLMemory


class _FakeSimpleMem:
    """替身 SimpleMem：记录 add_dialogue / finalize，ask 返回可断言的固定串。"""

    instances: list = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.dialogues: list = []
        self.finalized = 0
        _FakeSimpleMem.instances.append(self)

    def add_dialogue(self, speaker, content, timestamp=None):
        self.dialogues.append((speaker, content, timestamp))

    def finalize(self):
        self.finalized += 1

    def ask(self, question):
        return f"ANSWER[{len(self.dialogues)} turns]: {question}"


@pytest.fixture
def fake_simplemem(monkeypatch):
    _FakeSimpleMem.instances = []
    mod = types.ModuleType("simplemem")
    mod.SimpleMem = _FakeSimpleMem
    monkeypatch.setitem(sys.modules, "simplemem", mod)
    return mod


def test_prev_output_unchanged():
    nl = NLMemory(strategy="prev_output")
    assert nl.recall("hello") == f"{PREV_OUTPUT_HEADER}\nhello"
    assert nl.recall("") == ""
    nl.observe("a", "x")  # prev_output 下 observe 为 no-op
    assert nl.recall("y") == f"{PREV_OUTPUT_HEADER}\ny"


def test_simplemem_observe_finalize_ask(fake_simplemem):
    nl = NLMemory(strategy="simplemem", simplemem_kwargs={"model": "m"})
    assert nl.recall("anything") == ""  # 无记忆 -> 不注入
    nl.observe("Alice", "Meet at Starbucks 2pm", "2025-11-15T14:30:00")
    nl.observe("Bob", "I'll bring the report")
    out = nl.recall("When and where?")
    assert out.startswith(MEMORY_HEADER)
    assert "ANSWER[2 turns]: When and where?" in out
    inst = _FakeSimpleMem.instances[-1]
    assert inst.kwargs == {"model": "m"}          # 构造 kwargs 透传
    assert inst.finalized >= 1                     # recall 前 finalize
    assert inst.dialogues[0] == ("Alice", "Meet at Starbucks 2pm", "2025-11-15T14:30:00")


def test_simplemem_reset_starts_new_memory(fake_simplemem):
    nl = NLMemory(strategy="simplemem")
    nl.observe("A", "hi")
    first = _FakeSimpleMem.instances[-1]
    nl.reset()
    nl.observe("B", "yo")
    assert _FakeSimpleMem.instances[-1] is not first  # 新对话 -> 新 SimpleMem 实例


def test_manager_feeds_nl_simplemem(fake_simplemem):
    from lychee_mas.methods.memory.managers.DualChannelMemory import DualChannelMemoryManager
    from lychee_mas.methods.memory.routing.base import RouteDecision

    class FakeBackend:  # 无 tok / 无 torch
        pass

    m = DualChannelMemoryManager(FakeBackend(), nl_strategy="simplemem",
                                 include_transcript=False)
    m.observe([{"role": "analyst", "content": "Alice: meet at 2pm"}])
    m.observe([{"role": "solver", "content": "ok noted"}])
    bundle = m.recall(RouteDecision(channel="nl"), "when?")
    assert bundle.NL_Channel and MEMORY_HEADER in bundle.NL_Channel
    assert len(_FakeSimpleMem.instances[-1].dialogues) == 2


def test_simplemem_missing_dependency_raises_clearly(monkeypatch):
    # 未安装 simplemem 时：只有真正用到才报清晰错误，import / prev_output 不受影响
    monkeypatch.setitem(sys.modules, "simplemem", None)  # 触发 ImportError
    nl = NLMemory(strategy="simplemem")
    with pytest.raises(ImportError, match="SimpleMem"):
        nl.observe("a", "hi")
