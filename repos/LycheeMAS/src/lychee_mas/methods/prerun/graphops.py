"""methods/prerun 共享图原语：时空边 / 掩码 / 拓扑序 / 实现集 / 逐参数 Adam。

同族方法（AgentPrune / AgentDropout，同出 AutoAgents/AgentPrune 代码基）共用的
最小图与优化原语。原先定义在 ``agentprune.py``，被 ``agentdropout/optimizer.py`` 与
``plugins/prerun/graphview.py`` 跨模块借用（含私有名 ``_Adam``）——现集中于此，
``agentprune.py`` 保留同名 re-export 以不破坏既有 import 路径。

纯标准库、零重依赖；本模块不注册任何组件（注册在各自的算法模块里）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

Edge = Tuple[int, int]  # (src_agent_index, dst_agent_index)


def full_connected_masks(n: int) -> tuple[List[List[int]], List[List[int]]]:
    """原版 get_kwargs(mode='FullConnected')：空间 i≠j 全 1（对角 0），时间全 1。"""
    spatial = [[1 if i != j else 0 for j in range(n)] for i in range(n)]
    temporal = [[1 for _ in range(n)] for _ in range(n)]
    return spatial, temporal


def topological_order(n: int, edges: set[Edge]) -> tuple[List[int], Dict[int, set[int]]]:
    """按固定优先级（agent 序号小者优先）的 Kahn 拓扑排序。

    返回 (执行顺序, 每个节点最终保留的前驱集合)。采样图成环时确定性破环：
    无零入度节点时，强制执行剩余节点中序号最小者——其尚未执行的前驱边被丢弃
    （消息只能从先执行的节点流向后执行的节点）。
    """
    indeg = {i: 0 for i in range(n)}
    preds: Dict[int, set[int]] = {i: set() for i in range(n)}
    for src, dst in edges:
        if src != dst and src not in preds[dst]:
            preds[dst].add(src)
            indeg[dst] += 1
    order: List[int] = []
    executed: set[int] = set()
    final_preds: Dict[int, set[int]] = {i: set() for i in range(n)}
    remaining = set(range(n))
    while remaining:
        ready = sorted(i for i in remaining if indeg[i] == 0)
        cur = ready[0] if ready else min(remaining)  # 无零入度即破环：强制最小序号
        final_preds[cur] = preds[cur] & executed  # 只保留已执行的前驱（破环边自然被丢弃）
        order.append(cur)
        executed.add(cur)
        remaining.discard(cur)
        for j in remaining:
            if cur in preds[j]:
                indeg[j] -= 1
    return order, final_preds


@dataclass
class Realization:
    """一次伯努利采样的实现图 + REINFORCE 所需的逐边 (采样值, 概率)。"""

    spatial_edges: set[Edge]
    temporal_edges: set[Edge]
    spatial_samples: Dict[Edge, Tuple[int, float]] = field(default_factory=dict)
    temporal_samples: Dict[Edge, Tuple[int, float]] = field(default_factory=dict)

    def log_prob(self) -> float:
        total = 0.0
        for b, p in list(self.spatial_samples.values()) + list(self.temporal_samples.values()):
            total += math.log(p if b else (1.0 - p))
        return total


class Adam:
    """逐参数 Adam（与 torch.optim.Adam 默认超参一致：betas=(0.9,0.999), eps=1e-8）。

    原 ``agentprune._Adam``：迁入本模块时去私有化；实例化点仍用原名（见
    ``agentprune.py`` 的 re-export）。
    """

    def __init__(self, lr: float):
        self.lr = lr
        self.m: Dict[Any, float] = {}
        self.v: Dict[Any, float] = {}
        self.t = 0

    def step(self, params: Dict[Any, float], grads: Dict[Any, float]) -> None:
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for key, g in grads.items():
            self.m[key] = b1 * self.m.get(key, 0.0) + (1 - b1) * g
            self.v[key] = b2 * self.v.get(key, 0.0) + (1 - b2) * g * g
            mhat = self.m[key] / (1 - b1 ** self.t)
            vhat = self.v[key] / (1 - b2 ** self.t)
            params[key] = params[key] - self.lr * mhat / (math.sqrt(vhat) + eps)
