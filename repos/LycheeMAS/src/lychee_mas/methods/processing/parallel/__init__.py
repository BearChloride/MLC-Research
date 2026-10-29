"""处理层 · **并行子模块**——并发跑 K 次 MAS，再用 aggregator 聚合成一个 Answer。

- processor/parallel           并行处理器：并发 K 次 → K 条轨迹 → aggregator 聚合（子模块入口）
- aggregator/self_consistency  纯标准库多数投票（真实可测组件，CLAUDE.md §5 要求至少一个可跑）
- aggregator/aggagent          agentic 聚合（AggAgent 移植，见 ../aggagent/）：检索工具跨轨迹
                               「数证据不数轨迹数」，mock 可测、真实跑分走 tool-calling 端点
- aggregator/dynamicagg        动态聚合（在研，桩）

`aggregator`（TrajectoryAggregator）是并行处理器的可插拔归约策略：对 list[Answer]（或从
list[Trajectory] 取 final_answer）按归一化内容取众数（self_consistency）——aggagent 在
同样的并行采集上做检索式 agentic 归约，是它的对拍对象。
"""
from __future__ import annotations

import asyncio
import re
from collections import Counter

from ....core.registry import REGISTRY
from ....core.types import Answer, Trajectory
from ..aggagent import AggAgentAggregator
from ..base import ProcessingResult, Runner


def _norm(text: str) -> str:
    # 轻量归一化：小写 + 压空白，便于把同义答案归并到同一票
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _as_answers(items: list) -> list[Answer]:
    """把输入统一成 list[Answer]：Trajectory 取其 final_answer/最末候选；Answer 原样；str 包成
    Answer。"""
    out: list[Answer] = []
    for it in items:
        if isinstance(it, Answer):
            out.append(it)
        elif isinstance(it, Trajectory):
            if it.final_answer is not None:
                out.append(it.final_answer)
            elif it.candidates:
                out.append(it.candidates[-1])
        elif isinstance(it, str):
            out.append(Answer(content=it))
    return out


@REGISTRY.register("aggregator", "self_consistency")
class SelfConsistencyVote:
    """多数投票（Self-Consistency）：对候选答案按归一化内容取众数。纯标准库，可跑可测。"""

    name = "self_consistency"

    def __init__(self, k: int | None = None):
        self.k = k  # 可选：只取前 k 个候选参与投票（None=全用）

    def aggregate(self, trajectories: list[Trajectory]) -> Answer:
        answers = _as_answers(list(trajectories))
        if self.k is not None:
            answers = answers[: self.k]
        if not answers:
            return Answer(content="", source=self.name, meta={"votes": 0})
        # 按归一化文本计票，保留每个归一化键的一个原始代表（首次出现）
        counter: Counter[str] = Counter()
        rep: dict[str, Answer] = {}
        for a in answers:
            key = _norm(a.content)
            counter[key] += 1
            rep.setdefault(key, a)
        best_key, votes = counter.most_common(1)[0]
        chosen = rep[best_key]
        return Answer(
            content=chosen.content,
            source=self.name,
            confidence=votes / len(answers),
            score=chosen.score,
            meta={"votes": votes, "total": len(answers),
                  "distribution": dict(counter)},
        )


@REGISTRY.register("aggregator", "dynamicagg")
class DynamicAggregator:
    """动态聚合（在研，桩）：按过程信号自适应选择融合策略。统一报错文案：not wired yet (TODO)。"""

    name = "dynamicagg"

    def __init__(self, **kwargs):
        self.cfg = kwargs

    def aggregate(self, trajectories: list[Trajectory]) -> Answer:
        raise NotImplementedError("dynamicagg: not wired yet (TODO)")


@REGISTRY.register("processor", "parallel")
class ParallelProcessor:
    """并行处理：并发调用 runner **K 次**产出 K 条轨迹，再用 `aggregator` 聚合成一个 Answer。

    K=`k`；聚合策略 = `aggregator`（默认 self_consistency 多数投票）；聚合器构造超参走
    `aggregator_kwargs` 透传（如 aggagent 的 client / model+api_base——多数投票无参可省）。
    runner 需每次产出独立轨迹（调用方负责隔离，如 ctx.reset）。"""

    name = "parallel"

    def __init__(
        self,
        k: int = 5,
        aggregator: str = "self_consistency",
        aggregator_kwargs: dict | None = None,
        **kwargs,
    ):
        self.k = max(1, int(k))
        self.aggregator = aggregator
        self.aggregator_kwargs = dict(aggregator_kwargs or {})
        self.cfg = kwargs

    async def run(self, runner: Runner) -> ProcessingResult:
        trajs = list(await asyncio.gather(*[runner() for _ in range(self.k)]))
        agg = REGISTRY.create("aggregator", self.aggregator, **self.aggregator_kwargs)
        return ProcessingResult(answer=agg.aggregate(trajs), trajectories=trajs)


__all__ = ["SelfConsistencyVote", "DynamicAggregator", "ParallelProcessor", "AggAgentAggregator"]
