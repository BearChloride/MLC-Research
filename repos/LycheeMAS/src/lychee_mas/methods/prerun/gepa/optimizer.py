"""GEPAOptimizer —— 反思式提示演化（注册 `optimizer/gepa`）。

compile 循环：
  候选池（program + per-instance 分数向量）
  → Pareto 采样母本（pareto.sample_candidate）
  → 轮换选一个可变组件（round-robin over mutable_keys）
  → minibatch rollout 收集反馈（question / score / final answer / 路由决策摘要）
  → 反思变异（reflector：旧组件文本 + 反馈 → 新文本；默认用 backend.generate_chat）
  → minibatch 得分提升才上全量评估入池
  → 预算（max_metric_calls，含 rollout+metric 的每次调用）耗尽返回均分最优。

rollout 与 reflector 都可注入（离线测试用脚本化实现）；缺 rollout、或缺 reflector 且缺
backend 时**显式报错**，不静默降级。纯标准库；LLM 只经由 backend.generate_chat 触达。
"""
from __future__ import annotations

import json
import random
from typing import Any, Callable, Dict, List, Optional, Sequence

from ....core.registry import REGISTRY
from ....core.types import TaskQuery, Trajectory
from .pareto import sample_candidate
from .program import MASProgram

# 评分函数：一条轨迹在一条 query 上的得分（越大越好；原 plugins/base.py 定义迁此）
Metric = Callable[[Trajectory, TaskQuery], float]

# rollout：把 program 物化成系统并在一条 query 上跑一次，返回轨迹
Rollout = Callable[[MASProgram, TaskQuery], Trajectory]
# reflector：(旧组件文本, 反馈记录) -> 新组件文本
Reflector = Callable[[str, List[Dict[str, Any]]], str]

_REFLECT_SYSTEM = (
    "You improve one component of a multi-agent system based on execution feedback. "
    "Reply with ONLY the improved component text, no commentary, no markdown fences."
)


class _BudgetExhausted(Exception):
    """内部控制流：metric 调用预算耗尽（对外表现为返回当前最优）。"""


@REGISTRY.register("optimizer", "gepa")
class GEPAOptimizer:
    name = "gepa"

    def __init__(self, rollout: Optional[Rollout] = None,
                 reflector: Optional[Reflector] = None, backend: Any = None,
                 minibatch_size: int = 3, max_metric_calls: int = 100,
                 reflect_max_new_tokens: int = 512, seed: int = 0) -> None:
        self.rollout = rollout
        self.reflector = reflector
        self.backend = backend
        self.minibatch_size = int(minibatch_size)
        self.max_metric_calls = int(max_metric_calls)
        self.reflect_max_new_tokens = int(reflect_max_new_tokens)
        self.seed = int(seed)
        self._calls = 0

    # ---------------- 反思 ----------------

    def _reflect(self, component_text: str, feedback: List[Dict[str, Any]]) -> str:
        if self.reflector is not None:
            new_text = self.reflector(component_text, feedback)
        else:
            if self.backend is None:
                raise ValueError(
                    "optimizer/gepa 需要 reflector 或 backend（默认反思器经 "
                    "backend.generate_chat 产出变异文本）")
            user = (
                "Current component text:\n---\n" + component_text + "\n---\n"
                "Execution feedback (JSON):\n" + json.dumps(feedback, ensure_ascii=False)
                + "\nRewrite the component to fix the observed failures while keeping "
                  "what already works."
            )
            g = self.backend.generate_chat(
                [{"role": "system", "content": _REFLECT_SYSTEM},
                 {"role": "user", "content": user}],
                max_new_tokens=self.reflect_max_new_tokens)
            new_text = g.text.strip()
        if not isinstance(new_text, str) or not new_text.strip():
            raise ValueError("反思器返回空文本：无法变异组件（显式失败，不静默保留旧文本）")
        return new_text

    # ---------------- 评估（预算记账的唯一入口） ----------------

    def _run_scored(self, program: MASProgram, query: TaskQuery,
                    metric: Metric) -> tuple[Trajectory, float]:
        if self._calls >= self.max_metric_calls:
            raise _BudgetExhausted()
        assert self.rollout is not None  # optimize() 入口已校验
        traj = self.rollout(program, query)
        score = float(metric(traj, query))
        self._calls += 1
        return traj, score

    def _evaluate(self, program: MASProgram, items: Sequence[TaskQuery],
                  metric: Metric) -> List[float]:
        return [self._run_scored(program, q, metric)[1] for q in items]

    # ---------------- compile 主循环 ----------------

    def optimize(self, system: MASProgram, trainset: Sequence[TaskQuery],
                 metric: Metric) -> MASProgram:
        if self.rollout is None:
            raise ValueError("optimizer/gepa 需要 rollout（把 MASProgram 跑成 Trajectory）")
        if not trainset:
            raise ValueError("trainset 为空")
        if not system.mutable_keys:
            raise ValueError("MASProgram.mutable_keys 为空：没有可变异组件")

        rng = random.Random(self.seed)
        self._calls = 0
        pool: Dict[str, MASProgram] = {}
        scores: Dict[str, List[float]] = {}
        mutation_round = 0

        try:
            pool["seed"] = system
            scores["seed"] = self._evaluate(system, trainset, metric)
            child_index = 0
            while self._calls < self.max_metric_calls:
                parent_id = sample_candidate(scores, rng)
                parent = pool[parent_id]
                key = parent.mutable_keys[mutation_round % len(parent.mutable_keys)]
                mutation_round += 1

                batch = list(trainset)
                if len(batch) > self.minibatch_size:
                    batch = rng.sample(batch, self.minibatch_size)

                feedback: List[Dict[str, Any]] = []
                parent_batch_scores: List[float] = []
                for q in batch:
                    traj, s = self._run_scored(parent, q, metric)
                    parent_batch_scores.append(s)
                    feedback.append({
                        "question": q.question,
                        "score": s,
                        "final_answer": (traj.final_answer.content
                                         if traj.final_answer else ""),
                        "decisions": traj.meta.get("decisions", [])[-3:],
                    })

                child = parent.mutated(key, self._reflect(parent.components[key], feedback))
                child_batch_scores = self._evaluate(child, batch, metric)
                if sum(child_batch_scores) <= sum(parent_batch_scores):
                    continue  # minibatch 未提升：丢弃变异，不花全量预算
                child_id = f"child_{child_index}"
                child_index += 1
                pool[child_id] = child
                scores[child_id] = self._evaluate(child, trainset, metric)
        except _BudgetExhausted:
            pass

        best_id = max(scores, key=lambda cid: (sum(scores[cid]) / len(scores[cid]), cid))
        best = pool[best_id]
        return MASProgram(components=dict(best.components), mutable_keys=best.mutable_keys,
                          meta={**best.meta, "gepa": {
                              "selected": best_id,
                              "pool_size": len(pool),
                              "metric_calls": self._calls,
                              "mean_score": sum(scores[best_id]) / len(scores[best_id]),
                          }})
