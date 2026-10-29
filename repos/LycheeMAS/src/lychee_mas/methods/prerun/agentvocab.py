"""词表适配（VocabAdapter 接缝，模型级降本）。

注册 `vocab_adapter/agentvocab`（结构感知词表适配·自有，桩）。统一报错文案：not wired yet (TODO)。
迁移时实现 adapt(agent, context) -> 词表子集/映射；对标 AdaptiVocab / TokAlign。
"""
from __future__ import annotations

from typing import Any

from ...core.registry import REGISTRY


@REGISTRY.register("vocab_adapter", "agentvocab")
class AgentVocab:
    name = "agentvocab"

    def __init__(self, **kwargs):
        self.cfg = kwargs

    def adapt(self, agent: Any, context: Any = None) -> Any:
        raise NotImplementedError("agentvocab: not wired yet (TODO)")


__all__ = ["AgentVocab"]
