"""L1 团队组建：AgentInit 选择器（`agent_selector/agentinit`）。

论文：AgentInit (EMNLP 2025 Findings, arXiv:2509.19236)。把「组建一支高效 MAS 团队」建模为
**多目标平衡选择**：兼顾团队多样性与任务相关性，选出小而互补的团队（降冗余、降 token、保性能）。

两种模式（同一 `select()`）：
  - `pool`（本文件 M1 实现）：固定候选池 + 确定性嵌入 + 非支配排序 + 确定性挑选。
    零 GPU/API，可复现，兼作「AgentInit 去掉生成」的离线消融基线。
  - `generate`（M2，见 `_generate.py`）：忠实 AgentInit —— LLM 现场生成角色 + HF
    embedder + LLM 挑组。

选择数学两模式共享 `_pareto.py`（余弦 + Vendi 多样性 + 非支配排序），与官方源码一致。
numpy/vendi_score/torch 全部惰性（`selectors/__init__` 被 import 时零重依赖）。
"""
from __future__ import annotations

from itertools import combinations
from typing import Optional

from ....core.registry import REGISTRY
from ....core.types import AgentSpec, Budget, TaskQuery


@REGISTRY.register("agent_selector", "agentinit")
class AgentInitSelector:
    """多样性×相关性 Pareto 选择器。构造参数来自 `configs/agents/agentinit.yaml`（构造器注入）。"""

    name = "agentinit"

    def __init__(self, mode: str = "pool", min_roles: int = 1, max_roles: int = 5,
                 objectives: Optional[list[str]] = None, seed: int = 0, **cfg):
        self.mode = mode
        self.min_roles = int(min_roles)
        self.max_roles = int(max_roles)
        self.objectives = objectives or ["relevance", "diversity"]
        self.seed = seed
        self.cfg = cfg  # generate 模式用（model_client / embedder_model / critique_rounds 等）
        # generate 模式一次 select() 的总生成 token 数（角色生成+批判+挑组），供实验记账取用。
        # pool 模式不调 LLM，恒为 0。见 `_generate.generate_and_select`（#2：不逐 spec 重复记账）。
        self.last_gen_tokens = 0

    # ---- 协议入口 ----
    def select(self, query: TaskQuery, budget: Optional[Budget] = None) -> list[AgentSpec]:
        """从候选/生成的角色中选出一支团队。

        **budget 语义（本 selector 专属约定）**：仅 `generate` 模式使用，且**只认 token 预算**
        （`Budget(unit=TOKENS, limit=X)`）——给「角色生成的多轮迭代」封顶：累计生成 token 超过 `X`
        时停止继续批判/精炼，用当前已生成的角色定案。`calls`/`usd` 单位与 `pool` 模式**一律忽略**
        （pool 不调 LLM，无预算可言）。budget=None 时不设上限，仅受 `critique_rounds` 约束。
        """
        if self.mode == "pool":
            return self._select_pool(query, budget)
        if self.mode == "generate":
            return self._select_generate(query, budget)
        raise ValueError(f"agentinit: unknown mode {self.mode!r} (expected 'pool' | 'generate')")

    # ---- pool 模式（M1）：不调 LLM，budget 不适用（见 select() docstring）----
    def _select_pool(self, query: TaskQuery, budget: Optional[Budget]) -> list[AgentSpec]:
        from . import pool as _pool
        from ._pareto import (
            cosine_matrix,
            cosine_to_query,
            fast_non_dominated_sort,
            objective_diversity,
            objective_relevance,
        )

        cands = _pool.candidate_pool(query)
        n = len(cands)
        if n == 0:
            return []

        embeddings = [c.profile["vec"] for c in cands]
        sim_matrix = cosine_matrix(embeddings)
        query_sims = cosine_to_query(_pool.embed_query(query), embeddings)

        # 规模纯由 min/max_roles + 前沿涌现决定（忠实官方，不用 budget）
        lo = max(1, min(self.min_roles, n))
        hi = min(self.max_roles, n)

        # 枚举所有规模 lo..hi 的角色子集当作「种群」（与源码 Init_Population 一致）
        groups: list[tuple[int, ...]] = []
        for k in range(lo, hi + 1):
            groups.extend(combinations(range(n), k))
        if not groups:
            return []

        objectives = [
            (objective_relevance(g, query_sims), objective_diversity(g, sim_matrix))
            for g in groups
        ]
        front = fast_non_dominated_sort(objectives)[0]

        # 确定性替身（代替 LLM SelectGroup）：前沿里选 relevance+diversity 之和最优者；
        # 目标是负值 → 取 obj1+obj2 最小；平手时偏好更小团队、再按下标字典序。
        best = min(front, key=lambda idx: (
            objectives[idx][0] + objectives[idx][1], len(groups[idx]), groups[idx]))
        chosen = groups[best]
        self.last_gen_tokens = 0  # pool 不调 LLM
        return [cands[i] for i in chosen]

    # ---- generate 模式（M2）----
    def _select_generate(self, query: TaskQuery, budget: Optional[Budget]) -> list[AgentSpec]:
        from ._generate import generate_and_select

        return generate_and_select(self, query, budget)


__all__ = ["AgentInitSelector"]
