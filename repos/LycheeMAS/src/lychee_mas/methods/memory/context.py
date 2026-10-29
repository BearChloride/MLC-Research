"""RoutingContext —— 把路由信号从 GroupChat 层下传到各 agent 的 InjectionClient 的共享状态
（CLAUDE.md §8，「AutoGen 版唯一要接仔细的链路」）。

GroupChat 里 agent 串行发言，所以单个共享 context 是安全的：每个 agent 的 client 从这里读
（自己的 role、当前 task/turn、上一个发言者），问 router（记忆通道决策）、向 memory 召回，再把决策
写回。
router（记忆通道决策）与 memory（记忆方法）是两个可替换接缝；其余固定。

决策日志默认存在 `self.decisions`（可解释性 + 落盘）；若构造时传入 `trace_store`，则同时
写入 `trace.TraceStore`（CLAUDE.md §8：统一落点）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .routing.base import MemoryRouter, RouteDecision

if TYPE_CHECKING:  # 仅类型检查期需要；避免与 memory.base 形成 import 环
    from .base import MemoryManager


@dataclass
class RoutingContext:
    task: str  # 当前任务名（下传给 RouterInputs）
    router: MemoryRouter  # 记忆通道决策（所有 agent 共用一个）
    memory: MemoryManager  # 记忆方法（所有 agent 共用一个）
    team: Optional[str] = None  # 强制用的队伍 profile；None=按 task 自动选
    turns: Dict[str, int] = field(default_factory=dict)  # role -> 该角色已发言轮次
    availability: Dict[str, bool] = field(default_factory=lambda: {
        "none": True, "nl": True, "latent": True, "both": True})  # 各通道是否可用（约束 #2 用）
    model_of_role: Dict[str, str] = field(default_factory=dict)  # role -> model id（判同模型对）
    decisions: List[dict] = field(default_factory=list)  # 决策日志（可解释性 + 落盘 routing_trace）
    trace_store: Any = None  # 可选 trace.TraceStore：把决策同时写入统一落点
    span_logger: Any = None  # 可选 JsonlSpanLogger：运行中实时写 spans.jsonl
    current_case_id: Optional[str] = None
    current_sample_index: Optional[int] = None
    trace_model_calls: bool = True  # 是否在终端打印每次 LLM 调用的 start/done 摘要

    def turn_of(self, role: str) -> int:
        return self.turns.get(role, 0)  # 读取某角色当前轮次（默认 0）

    def bump_turn(self, role: str) -> None:
        self.turns[role] = self.turns.get(role, 0) + 1  # 该角色发言后 +1

    def same_model_pair(self, sender: Optional[str], receiver: Optional[str]) -> bool:
        """约束 #2：latent 只能在对齐（同模型）的 agent 对之间传递。

        缺信息时（无 sender/receiver 或未登记模型）保守返回 True（交给路由器后续判断）。
        """
        if not sender or not receiver:
            return True
        ms, mr = self.model_of_role.get(sender), self.model_of_role.get(receiver)
        if ms is None or mr is None:
            return True
        return ms == mr  # 同模型才允许 latent 跨 agent

    def log_decision(self, role: str, turn: int, sender: Optional[str],
                     decision: RouteDecision, extra: dict) -> None:
        # 把一次路由决策 + 额外信息（成本记账 + 本 agent 的输入 input_messages / 输出 output）
        # 追加进日志；落盘为每样本的 routing_trace（供核查 / MAST judge / 反事实蒸馏复用）
        record = {
            # Backward-compatible short names.
            "role": role, "turn": turn, "sender": sender,
            "channel": decision.channel,  # P 已移出 RouteDecision（归 latent 通道）
            "reason": decision.reason,
            # Clear names for downstream metric/trace consumers.
            "turn_index": turn, "sender_role": sender,
            "memory_channel": decision.channel,
            "routing_reason": decision.reason,
            **extra}
        self.decisions.append(record)
        if self.trace_store is not None:
            self.trace_store.log_decision(record)

    def set_case(self, case_id: str, sample_index: int) -> None:
        self.current_case_id = str(case_id)
        self.current_sample_index = int(sample_index)
        if self.span_logger is not None:
            self.span_logger.set_case(self.current_case_id, self.current_sample_index)

    def log_span(self, span_type: str, **fields: Any) -> Optional[str]:
        if self.span_logger is None:
            return None
        return self.span_logger.log(
            span_type,
            case_id=self.current_case_id,
            sample_index=self.current_sample_index,
            **fields,
        )

    def reset(self) -> None:
        # 每个样本开始前清空轮次/日志，并重置记忆库（清 transcript/缓存）
        self.turns.clear()
        self.decisions.clear()
        self.memory.reset()
