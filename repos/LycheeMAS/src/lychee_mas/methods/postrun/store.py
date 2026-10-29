"""TraceStore —— 执行轨迹的消息级落点（CLAUDE.md §8）。

Runtime 的 `intercept` 把每条 Message 写入 TraceStore，并喂给 做记忆抽取——这是 记忆/处理/训练 的数据
来源。
本实现纯标准库：内存累积 + 可选 JSONL 落盘。决策日志（RoutingContext）也可选地写到这里。
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass
from typing import Any

from ...core.types import Message


def _to_jsonable(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: _to_jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


class TraceStore:
    """消息 + 决策日志的内存累积器（可选写 JSONL）。"""

    def __init__(self, path: str | None = None) -> None:
        self.path = path
        self.messages: list[Message] = []
        self.decisions: list[dict] = []

    def hook(self, message: Message) -> None:
        """供 Runtime.intercept 注册：每条 Message 调用一次。"""
        self.messages.append(message)
        if self.path:
            self._append_jsonl({"kind": "message", **_to_jsonable(message)})

    def log_decision(self, record: dict) -> None:
        """记录一次路由决策（RoutingContext 可选写入）。"""
        self.decisions.append(record)
        if self.path:
            self._append_jsonl({"kind": "decision", **_to_jsonable(record)})

    def reset(self) -> None:
        self.messages.clear()
        self.decisions.clear()

    def _append_jsonl(self, record: dict) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def __len__(self) -> int:
        return len(self.messages)
