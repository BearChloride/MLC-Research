"""AgentPrune —— 时空通信图剪枝（graph_pruner/agentprune 的真实现）。

复现自：Cut the Crap: An Economical Communication Pipeline for LLM-based
Multi-Agent Systems（ICLR 2025）；参考实现 https://github.com/yanweiyue/AgentPrune
（AgentPrune/graph/graph.py 与 experiments/run_gsm8k.py 的机制逐一对应）：

- 每条潜在**空间边**（同轮 i→j）与**时间边**（上一轮 i→本轮 j）持有一个可训练 logit；
  fixed mask 为 0 的边永久禁用（如 FullConnected：空间对角线为 0）。
- 每次执行**伯努利采样**一张实现图：p = sigmoid(logit / temperature)，
  并累计 log P(采样)（原版 construct_*_connection）。
- **REINFORCE**：loss = -log_prob × utility（utility = 该题是否做对），Adam 更新 logits
  （原版 torch.optim.Adam(lr=0.1)；此处纯 python 实现同款 Adam，端到端语义一致）。
- **one-shot 剪枝** `update_masks(pruning_rate)`：对仍存活的边按 logit 升序，
  置零最低的 round(alive × rate) 条（空间/时间各自独立，原版逐张量执行）。

与原版的两处已知偏差（docstring 级声明，不影响方法语义）：
1. 原版对采样出的空间图做拓扑排序执行；采样图可能成环，此处以固定 agent 顺序做
   Kahn 拓扑排序 + 确定性破环（丢弃指向已阻塞最小序号节点的入边），见
   `graphops.topological_order`。
2. logits 为逐边独立参数（原版同为逐边 Parameter 向量，无低秩参数化）。

纯标准库、零重依赖；训练循环（rollout 谁来跑）由实验脚本驱动，本类只负责
「采样 / 更新 / 剪枝 / 产出实现矩阵」。图级挂载经统一接口
`optimize_langgraph(sg, method="agentprune")`（plugins/prerun/agentprune_lg.py）。

共享图原语（`Edge` / `full_connected_masks` / `topological_order` / `Realization` /
`Adam`）定义在 `graphops.py`（同族 AgentDropout 与 graphview 同用）；本模块 re-export
同名符号，旧 import 路径（含私有名 `_Adam`）继续可用。
"""
from __future__ import annotations

import json
import math
import random
from typing import Any, Dict, List, Optional, Tuple

from ...core.registry import REGISTRY
from .graphops import (
    Adam,
    Edge,
    Realization,
    full_connected_masks,
    topological_order,
)

_Adam = Adam  # 旧私有名继续可用（本文件下方实例化点未改）

__all__ = ["Edge", "Realization", "Adam", "_Adam", "full_connected_masks",
           "topological_order", "AgentPrunePruner"]


@REGISTRY.register("graph_pruner", "agentprune")
class AgentPrunePruner:
    """AgentPrune 时空掩码剪枝器（GraphPruner 协议 + 训练接口）。"""

    name = "agentprune"

    def __init__(self, n_agents: int = 4,
                 initial_spatial_probability: float = 0.5,
                 initial_temporal_probability: float = 0.5,
                 fixed_spatial_masks: Optional[List[List[int]]] = None,
                 fixed_temporal_masks: Optional[List[List[int]]] = None,
                 lr: float = 0.1, temperature: float = 1.0, seed: int = 0,
                 optimized_spatial: bool = True, optimized_temporal: bool = True,
                 state_file: Optional[str] = None) -> None:
        self.n = int(n_agents)
        if self.n < 1:
            raise ValueError(f"n_agents 必须 >= 1，得到 {self.n}")
        if fixed_spatial_masks is None or fixed_temporal_masks is None:
            fs, ft = full_connected_masks(self.n)
            fixed_spatial_masks = fixed_spatial_masks or fs
            fixed_temporal_masks = fixed_temporal_masks or ft
        self.temperature = float(temperature)
        self.optimized_spatial = bool(optimized_spatial)
        self.optimized_temporal = bool(optimized_temporal)
        self.rng = random.Random(seed)

        def init_logit(p: float, optimized: bool) -> float:
            # 原版：optimized 时 logit=log(p/(1-p))；否则 10.0（σ≈1，边恒在）
            return math.log(p / (1.0 - p)) if optimized else 10.0

        s_init = init_logit(initial_spatial_probability, self.optimized_spatial)
        t_init = init_logit(initial_temporal_probability, self.optimized_temporal)
        # 潜在边 = 全体有序对（空间边 i==j 无意义，由 FullConnected 固定掩码禁用；
        # 时间边允许 i==j = 看到自己上一轮输出）
        self.spatial_logits: Dict[Edge, float] = {}
        self.spatial_masks: Dict[Edge, int] = {}
        self.temporal_logits: Dict[Edge, float] = {}
        self.temporal_masks: Dict[Edge, int] = {}
        for i in range(self.n):
            for j in range(self.n):
                self.spatial_logits[(i, j)] = s_init
                self.spatial_masks[(i, j)] = int(fixed_spatial_masks[i][j])
                self.temporal_logits[(i, j)] = t_init
                self.temporal_masks[(i, j)] = int(fixed_temporal_masks[i][j])
        self._adam_s = _Adam(lr)
        self._adam_t = _Adam(lr)
        if state_file:
            self.load(state_file)

    # ---------------- 概率 / 采样 ----------------

    def _prob(self, logit: float) -> float:
        return 1.0 / (1.0 + math.exp(-logit / self.temperature))

    def sample_realization(self, rng: Optional[random.Random] = None,
                           include_temporal: bool = True) -> Realization:
        """伯努利采样一张实现图（只对 mask==1 的存活边采样，原版语义）。

        include_temporal=False 用于首轮（round 0 没有上一轮，时间边既不传消息也不计
        log_prob）。
        """
        rng = rng or self.rng
        real = Realization(spatial_edges=set(), temporal_edges=set())
        for edge, logit in self.spatial_logits.items():
            if self.spatial_masks[edge] == 0 or edge[0] == edge[1]:
                continue
            p = self._prob(logit)
            b = 1 if rng.random() < p else 0
            real.spatial_samples[edge] = (b, p)
            if b:
                real.spatial_edges.add(edge)
        if include_temporal:
            for edge, logit in self.temporal_logits.items():
                if self.temporal_masks[edge] == 0:
                    continue
                p = self._prob(logit)
                b = 1 if rng.random() < p else 0
                real.temporal_samples[edge] = (b, p)
                if b:
                    real.temporal_edges.add(edge)
        return real

    # ---------------- REINFORCE 更新 ----------------

    def reinforce(self, batch: List[Tuple[List[Realization], float]]) -> None:
        """REINFORCE 的 Adam 步：loss = mean_query(-Σ_rounds log_prob × utility)。

        每个 batch 项 = (该 query 各轮的实现图列表, utility)；多轮的 log_prob 相加
        （∂loss/∂logit = -u·Σ_rounds(b-p)/T），除数 = query 数（原版 batch mean）。
        """
        if not batch:
            return
        gs: Dict[Edge, float] = {}
        gt: Dict[Edge, float] = {}
        for reals, utility in batch:
            for real in reals:
                if self.optimized_spatial:
                    for edge, (b, p) in real.spatial_samples.items():
                        gs[edge] = gs.get(edge, 0.0) - utility * (b - p) / self.temperature
                if self.optimized_temporal:
                    for edge, (b, p) in real.temporal_samples.items():
                        gt[edge] = gt.get(edge, 0.0) - utility * (b - p) / self.temperature
        k = float(len(batch))
        if gs:
            self._adam_s.step(self.spatial_logits, {e: g / k for e, g in gs.items()})
        if gt:
            self._adam_t.step(self.temporal_logits, {e: g / k for e, g in gt.items()})

    # ---------------- one-shot 剪枝 ----------------

    def update_masks(self, pruning_rate: float) -> tuple[int, int]:
        """对存活边按 logit 升序置零最低的 round(alive × rate) 条；返回 (剪空间数, 剪时间数)。"""
        def prune_one(logits: Dict[Edge, float], masks: Dict[Edge, int],
                      skip_diag: bool) -> int:
            alive = [e for e, m in masks.items() if m == 1 and not (skip_diag and e[0] == e[1])]
            k = int(round(len(alive) * pruning_rate))
            if k <= 0:
                return 0
            for edge in sorted(alive, key=lambda e: logits[e])[:k]:
                masks[edge] = 0
            return k

        ks = prune_one(self.spatial_logits, self.spatial_masks, True) \
            if self.optimized_spatial else 0
        kt = prune_one(self.temporal_logits, self.temporal_masks, False) \
            if self.optimized_temporal else 0
        return ks, kt

    # ---------------- 确定性实现（评测 / 统一接口挂载用） ----------------

    def realized_matrices(self, mode: str = "threshold",
                          rng: Optional[random.Random] = None
                          ) -> tuple[List[List[int]], List[List[int]]]:
        """产出 (spatial, temporal) 邻接矩阵。threshold: p>=0.5 且未被剪；sample: 伯努利。"""
        if mode == "sample":
            real = self.sample_realization(rng)
            spatial, temporal = real.spatial_edges, real.temporal_edges
        elif mode == "threshold":
            spatial = {e for e, m in self.spatial_masks.items()
                       if m == 1 and e[0] != e[1] and self._prob(self.spatial_logits[e]) >= 0.5}
            temporal = {e for e, m in self.temporal_masks.items()
                        if m == 1 and self._prob(self.temporal_logits[e]) >= 0.5}
        else:
            raise ValueError(f"未知 realized 模式 {mode!r}（threshold|sample）")
        sm = [[1 if (i, j) in spatial else 0 for j in range(self.n)] for i in range(self.n)]
        tm = [[1 if (i, j) in temporal else 0 for j in range(self.n)] for i in range(self.n)]
        return sm, tm

    # ---------------- 状态持久化 ----------------

    def state_dict(self) -> dict:
        def pack(d: Dict[Edge, Any]) -> Dict[str, Any]:
            return {f"{i},{j}": v for (i, j), v in d.items()}

        return {"n_agents": self.n, "temperature": self.temperature,
                "optimized_spatial": self.optimized_spatial,
                "optimized_temporal": self.optimized_temporal,
                "spatial_logits": pack(self.spatial_logits),
                "spatial_masks": pack(self.spatial_masks),
                "temporal_logits": pack(self.temporal_logits),
                "temporal_masks": pack(self.temporal_masks)}

    def load_state_dict(self, state: dict) -> None:
        if int(state["n_agents"]) != self.n:
            raise ValueError(f"状态 n_agents={state['n_agents']} 与当前 {self.n} 不符")

        def unpack(d: Dict[str, Any], cast) -> Dict[Edge, Any]:
            out = {}
            for key, v in d.items():
                i, j = key.split(",")
                out[(int(i), int(j))] = cast(v)
            return out

        self.spatial_logits = unpack(state["spatial_logits"], float)
        self.spatial_masks = unpack(state["spatial_masks"], int)
        self.temporal_logits = unpack(state["temporal_logits"], float)
        self.temporal_masks = unpack(state["temporal_masks"], int)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.state_dict(), f, indent=2)

    def load(self, path: str) -> None:
        with open(path, encoding="utf-8") as f:
            self.load_state_dict(json.load(f))
