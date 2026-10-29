"""统一图抽象的公共类型（CLAUDE.md §4，所有层共享）。

把整个 MAS 表示为带时序/记忆状态的有向图 G=(V,E,W,T,M)：
- AgentSpec  = 图节点（智能体画像）
- Message    = 一条通信消息（执行轨迹与记忆抽取的基本单元）
- Answer     = 候选答案（融合单元）
- Trajectory = 一次执行 τ（有序 messages + candidates + final_answer）
- TaskQuery  = 输入查询（gold 供可验证奖励/评测）
- Budget     = 预算约束（tokens/calls/usd，供预算感知拓扑（construct）与剪枝（prune））

纯标准库 dataclass，零重依赖（黄金法则 4：核心骨架零运行依赖）。
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


@dataclass
class AgentSpec:
    """图节点 = 智能体画像。`profile` 放专长向量/多样性特征，供 construct 的 AgentInit 用。"""

    id: str = field(default_factory=_new_id)
    name: str = ""
    role: str = ""
    system_prompt: str = ""
    model: Optional[str] = None
    tools: list[str] = field(default_factory=list)
    profile: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Message:
    """一条通信消息（边上的一次传输）。`.tokens` 汇总开销。"""

    sender: str = ""
    content: str = ""
    receiver: Optional[str] = None
    round: int = 0
    role: str = "assistant"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    id: str = field(default_factory=_new_id)
    ts: float = field(default_factory=time.time)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def tokens(self) -> int:
        return int(self.prompt_tokens) + int(self.completion_tokens)


@dataclass
class Answer:
    """候选答案（融合单元）。"""

    content: str = ""
    source: Optional[str] = None
    confidence: float = 0.0
    score: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Trajectory:
    """一次执行 τ = 有序 messages + candidates + final_answer。"""

    task_id: str = ""
    messages: list[Message] = field(default_factory=list)
    final_answer: Optional[Answer] = None
    candidates: list[Answer] = field(default_factory=list)
    id: str = field(default_factory=_new_id)
    meta: dict[str, Any] = field(default_factory=dict)

    def add(self, message: Message) -> Message:
        """追加一条消息并返回它（便于链式记账）。"""
        self.messages.append(message)
        return message

    @property
    def total_tokens(self) -> int:
        return sum(m.tokens for m in self.messages)

    @property
    def num_rounds(self) -> int:
        # 轮次数 = 出现过的 round 编号的个数（空轨迹为 0）
        return len({m.round for m in self.messages}) if self.messages else 0


@dataclass
class TaskQuery:
    """输入查询。`gold` 供可验证奖励/评测。"""

    question: str = ""
    context: Optional[str] = None
    gold: Any = None
    id: str = field(default_factory=_new_id)
    meta: dict[str, Any] = field(default_factory=dict)


class BudgetUnit(str, Enum):
    TOKENS = "tokens"
    CALLS = "calls"
    USD = "usd"


@dataclass
class Budget:
    """预算约束（供预算感知拓扑（construct）与剪枝（prune））。"""

    limit: float = 0.0
    unit: BudgetUnit = BudgetUnit.TOKENS
