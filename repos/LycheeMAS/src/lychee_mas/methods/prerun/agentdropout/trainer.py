"""AgentDropout 两阶段训练日程（自 scripts/run_agentdropout_gsm8k.py 平移而来）。

阶段一 Node Dropout：``phase1_batches × phase1_batch_size`` 题；每题**逐轮**
``opt.sample_skip(r)`` 采样一个跳过节点（rollout 里该节点输出 'None.'），批末
``opt.skip_reinforce``（批内 mean over batch）；全部批次后 ``opt.node_dropout()``
一次性淘汰（原版 update_masks_dec）。
阶段二 Edge Dropout：``phase2_batches × phase2_batch_size`` 题（与阶段一同池）；
每题逐轮 ``opt.sample_round(r)`` 伯努利实现，批末 ``opt.edge_reinforce``；
``batch idx ∈ prune_batch_idx`` 后各 ``opt.edge_dropout(pruning_rate)`` 一次
（原版 update_masks_diff）。

**RNG 消费序即复现口径**：本模块是实验脚本循环的逐行转写——优化器的每次
``sample_skip / sample_round / reinforce / node_dropout / edge_dropout`` 的调用次数与
先后顺序不得增删改（reinforce/dropout 不抽 RNG，但顺序变了训练轨迹就变）。采样
一律「先把该题所有轮的实现采完 → 再整题 rollout」。LLM 只经注入的回调触达
（rollout 跑图、reward 打分、predict 抽取预测），本模块纯标准库、无重依赖。

数据（题池）由调用方按 core.types.TaskQuery 提供；本类只做日程、落盘与打印。
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Dict, List, Optional, Sequence, Tuple

from ....core.types import TaskQuery
from ..graphops import Edge

if TYPE_CHECKING:  # 仅类型检查期需要；optimizer.py 运行时导入本模块（AgentDropoutLG 需 Trainer），
    from .optimizer import AgentDropoutOptimizer  # 若此处也运行时导入即形成 import 环

# ---- 日程常量（脚本与 config 快照的唯一来源；原版硬编码 20/10/2/4 与 %2 剪枝） ----
PHASE1_BATCHES = 2          # 阶段一批次数（原版 dec loop）
PHASE1_BATCH = 20           # 阶段一每批题数
PHASE2_BATCHES = 4          # 阶段二批次数（原版 diff loop）
PHASE2_BATCH = 10           # 阶段二每批题数
PRUNE_BATCHES = (1, 3)      # 阶段二在哪几个 batch 后剪边（原版 (i_batch+1)%2==0 and i_batch<4）


@dataclass
class RolloutStats:
    """一次 rollout 的记账（字段名同 eval 落盘与 train_log 记账口径）。"""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    model_calls: int = 0
    latency_s: float = 0.0


@dataclass
class RoundPlan:
    """一轮执行的图：空间边（拓扑链）+ 时间边（读上一轮）+ 本轮被淘汰的节点（'None.'）。"""

    spatial_edges: set[Edge]
    temporal_edges: set[Edge]
    skip_idx: Optional[int]


def full_temporal_edges(n: int) -> set[Edge]:
    """固定全时间掩码的确定性无环实现（原版 optimized=False 的构造结果 = i<=j 含对角）。"""
    return {(a, b) for a in range(n) for b in range(n) if a <= b}


# rollout：(question, 逐轮 RoundPlan) -> (final 输出, 记账)；reward：(final, gold) -> 效用；
# predict：final -> 抽取的预测值（train_log 的 pred 字段，口径由调用方给定）
Rollout = Callable[[str, List[RoundPlan]], Awaitable[Tuple[str, RolloutStats]]]
Reward = Callable[[str, str], float]
Predict = Callable[[str], str]


@dataclass
class TrainReport:
    """训练产物摘要（供脚本/插件读数，不进任何落盘 schema）。"""

    state_path: str
    train_log_path: str
    skip_nodes: Dict[int, int]
    phase1_accuracy: float                    # 阶段一末批的 running_acc
    phase2_accuracy: float                    # 阶段二末批的 running_acc
    prune_events: List[Dict[str, Any]]        # [{"batch": i, "pruned": {...}, "alive_spatial": n}]
    log: List[Dict[str, Any]]                 # 与 *_train_log.json 逐字一致


class AgentDropoutTrainer:
    """AgentDropout 两阶段日程（阶段一 node dropout → 阶段二 edge dropout）。"""

    def __init__(self, opt: AgentDropoutOptimizer, *,
                 phase1_batches: int = PHASE1_BATCHES,
                 phase1_batch_size: int = PHASE1_BATCH,
                 phase2_batches: int = PHASE2_BATCHES,
                 phase2_batch_size: int = PHASE2_BATCH,
                 prune_batch_idx: Sequence[int] = PRUNE_BATCHES,
                 pruning_rate: float = 0.10,
                 verbose: bool = True) -> None:
        if phase1_batches < 1 or phase2_batches < 1:
            raise ValueError(f"阶段批次数需 ≥1，得到 {phase1_batches}/{phase2_batches}")
        if phase1_batch_size < 1 or phase2_batch_size < 1:
            raise ValueError(f"每批题数需 ≥1，得到 {phase1_batch_size}/{phase2_batch_size}")
        if not 0.0 <= pruning_rate <= 1.0:
            raise ValueError(f"pruning_rate 需在 [0,1]，得到 {pruning_rate}")
        bad = [i for i in prune_batch_idx if not 0 <= i < phase2_batches]
        if bad:
            raise ValueError(
                f"prune_batch_idx {sorted(bad)} 越界（阶段二 batch 下标 [0,{phase2_batches})）")
        self.opt = opt
        self.phase1_batches = int(phase1_batches)
        self.phase1_batch_size = int(phase1_batch_size)
        self.phase2_batches = int(phase2_batches)
        self.phase2_batch_size = int(phase2_batch_size)
        self.prune_batch_idx = tuple(int(i) for i in prune_batch_idx)
        self.pruning_rate = float(pruning_rate)
        self.verbose = bool(verbose)

    @property
    def needed_queries(self) -> int:
        """两阶段同池所需题数 = max(阶段一用量, 阶段二用量)。"""
        return max(self.phase1_batches * self.phase1_batch_size,
                   self.phase2_batches * self.phase2_batch_size)

    # ---------------------------------- 日程 ----------------------------------

    async def run(self, trainset: Sequence[TaskQuery], rollout: Rollout, reward: Reward,
                  predict: Predict, *, state_file: str,
                  train_log_file: Optional[str] = None) -> TrainReport:
        """跑完两阶段日程，落 state 与 train_log，返回 TrainReport。"""
        if state_file is None:
            raise ValueError("trainer 需要 state_file（两阶段训练产物落盘路径）")
        if len(trainset) < self.needed_queries:
            raise ValueError(
                f"训练需要 ≥{self.needed_queries} 题（阶段一 "
                f"{self.phase1_batches}×{self.phase1_batch_size} + 阶段二 "
                f"{self.phase2_batches}×{self.phase2_batch_size} 同池），得到 {len(trainset)}")
        for i, query in enumerate(trainset):
            if not isinstance(query, TaskQuery):
                raise TypeError(f"trainset[{i}] 需为 core.types.TaskQuery，得到 {type(query)!r}")
            if not isinstance(query.question, str) or not query.question:
                raise ValueError(f"trainset[{i}].question 需为非空字符串")
        if train_log_file is None:
            # 与实验脚本逐字同款推导：<…>_state.json → <…>_train_log.json
            train_log_file = state_file.replace("_state.json", "_train_log.json")
            if train_log_file == state_file:
                raise ValueError(
                    f"state_file 名不含 '_state.json'，无法推导 train_log 路径：{state_file}"
                    "（请显式传 train_log_file）")
        elif os.path.abspath(train_log_file) == os.path.abspath(state_file):
            raise ValueError(f"train_log_file 不能与 state_file 同路径：{state_file}")

        opt = self.opt
        log: List[Dict[str, Any]] = []
        prune_events: List[Dict[str, Any]] = []

        # ---- 阶段一：phase1_batches × phase1_batch_size skip-REINFORCE → node_dropout ----
        solved_1, acc_1 = 0, 0.0
        for i_batch in range(self.phase1_batches):
            batch = list(trainset[i_batch * self.phase1_batch_size:
                                  (i_batch + 1) * self.phase1_batch_size])
            grad_batch, entries, solved = await self._run_phase1_batch(
                batch, i_batch, rollout, reward, predict)
            opt.skip_reinforce(grad_batch)  # 原版：批末 mean over batch 的 Adam 步
            solved_1 += solved
            log.extend(entries)
            acc_1 = solved_1 / ((i_batch + 1) * len(batch))
            if self.verbose:
                print(f"[node_dropout] batch {i_batch + 1}/{self.phase1_batches} "
                      f"running_acc={acc_1:.3f}")
        skip_nodes = opt.node_dropout()  # 原版 update_masks_dec（全部批次后一次）
        if self.verbose:
            print(f"[node_dropout] done: skip_nodes={skip_nodes}")

        # ---- 阶段二：phase2_batches × phase2_batch_size 边 REINFORCE，prune idx 后各剪一次 ----
        solved_2, acc_2 = 0, 0.0
        for i_batch in range(self.phase2_batches):
            batch = list(trainset[i_batch * self.phase2_batch_size:
                                  (i_batch + 1) * self.phase2_batch_size])
            grad_batch, entries, solved = await self._run_phase2_batch(
                batch, i_batch, rollout, reward, predict)
            opt.edge_reinforce(grad_batch)
            solved_2 += solved
            log.extend(entries)
            if i_batch in self.prune_batch_idx:
                pruned = opt.edge_dropout(self.pruning_rate)
                alive = sum(1 for r in range(opt.rounds)
                            for e, m in opt.spatial_masks[r].items()
                            if m == 1 and e[0] != e[1])
                prune_events.append({"batch": i_batch, "pruned": pruned,
                                     "alive_spatial": alive})
                if self.verbose:
                    print(f"[edge_dropout] prune @batch {i_batch + 1}: {pruned} "
                          f"alive_spatial={alive}")
            acc_2 = solved_2 / ((i_batch + 1) * len(batch))
            if self.verbose:
                print(f"[edge_dropout] batch {i_batch + 1}/{self.phase2_batches} "
                      f"running_acc={acc_2:.3f}")

        # ---- 落盘（默认编码与 indent 与实验脚本逐字一致） ----
        os.makedirs(os.path.dirname(state_file) or ".", exist_ok=True)
        opt.save(state_file)
        os.makedirs(os.path.dirname(train_log_file) or ".", exist_ok=True)
        with open(train_log_file, "w") as f:
            json.dump(log, f, indent=2)
        if self.verbose:
            print(f"[train] done: state -> {state_file}  skip_nodes={opt.skip_nodes}")
        return TrainReport(state_path=state_file, train_log_path=train_log_file,
                           skip_nodes=dict(opt.skip_nodes), phase1_accuracy=acc_1,
                           phase2_accuracy=acc_2, prune_events=prune_events, log=log)

    # ------------------------------ 单批执行（逐行转写） ------------------------------

    async def _run_phase1_batch(self, batch: Sequence[TaskQuery], batch_idx: int,
                                rollout: Rollout, reward: Reward, predict: Predict) -> tuple:
        """阶段一一个 batch：逐题逐轮 sample_skip 后整题 rollout（被跳节点 'None.'）。"""
        grad_batch: List[Tuple[List[Tuple[int, set]], float]] = []
        log: List[Dict[str, Any]] = []
        solved = 0
        for query in batch:
            per_round: List[Tuple[int, set]] = []
            plans: List[RoundPlan] = []
            for r in range(self.opt.rounds):
                skip, edges = self.opt.sample_skip(r)  # 固定掩码全图的确定性无环实现上采样
                per_round.append((skip, edges))
                # 时间边同 ref dec 期构造：optimized=False + 全时间掩码 → i<=j（含对角）
                plans.append(RoundPlan(spatial_edges=edges,
                                       temporal_edges=(full_temporal_edges(self.opt.n)
                                                       if r >= 1 else set()),
                                       skip_idx=skip))
            final, stats = await rollout(query.question, plans)
            u = reward(final, query.gold)
            solved += int(u)
            grad_batch.append((per_round, u))
            log.append({"phase": "node_dropout", "batch": batch_idx,
                        "skip": {r: s for r, (s, _e) in enumerate(per_round)},
                        "utility": u, "pred": predict(final), "gold": query.gold,
                        "prompt_tokens": stats.prompt_tokens,
                        "completion_tokens": stats.completion_tokens,
                        "model_calls": stats.model_calls})
        return grad_batch, log, solved

    async def _run_phase2_batch(self, batch: Sequence[TaskQuery], batch_idx: int,
                                rollout: Rollout, reward: Reward, predict: Predict) -> tuple:
        """阶段二一个 batch：逐轮 sample_round 实现后整题 rollout（skip_nodes 节点 'None.'）。"""
        grad_batch: List[Tuple[List[Any], float]] = []
        log: List[Dict[str, Any]] = []
        solved = 0
        for query in batch:
            reals = [self.opt.sample_round(r) for r in range(self.opt.rounds)]
            plans = [RoundPlan(spatial_edges=re.spatial_edges,
                               temporal_edges=re.temporal_edges,
                               skip_idx=self.opt.skip_nodes.get(r))
                     for r, re in enumerate(reals)]
            final, stats = await rollout(query.question, plans)
            u = reward(final, query.gold)
            solved += int(u)
            grad_batch.append((reals, u))
            log.append({"phase": "edge_dropout", "batch": batch_idx,
                        "alive_edges": {r: len(re.spatial_edges)
                                        for r, re in enumerate(reals)},
                        "utility": u, "pred": predict(final), "gold": query.gold,
                        "prompt_tokens": stats.prompt_tokens,
                        "completion_tokens": stats.completion_tokens,
                        "model_calls": stats.model_calls})
        return grad_batch, log, solved
