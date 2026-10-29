"""AgentInit 多目标选择核心（从官方仓库逐行移植，黄金法则 §8 保留原始引用）。

来源：github.com/1737423697/AgentInit
  - `AgentInit/agentinit/Optimizer.py` → `fast_non_dominated_sort` / `dominates`
  - `AgentInit/agentinit/manager.py`   → `calculate_objective_1/2`（relevance / diversity）
  - `AgentInit/agentinit/embedder.py`  → `cosine_similarity` / `cosine_similarity_query`

方法（**非 NSGA-II 进化**，与源码一致）：把候选角色的所有子集当作「种群」，逐个在两目标
（relevance=与 query 的平均余弦；diversity=角色间相似度子矩阵的 Vendi 分数）上打分，再做
**非支配排序**取第一前沿（Pareto front）。两目标都取负号 → 最小化 = 同时最大化相关性与多样性。

⚠️ numpy / vendi_score 一律**函数内惰性导入**（黄金法则 §2：本模块被 import 时不得加载重库，
`make selfcheck` 须保持 HEAVY LOADED: NONE）。diversity 用官方同款 `vendi_score`（不自实现）。
"""
from __future__ import annotations

from typing import Sequence


# ---- 余弦相似度（移植自 embedder.py 的两个 staticmethod）----
def cosine_matrix(embeddings, normalize: bool = True):
    """角色×角色余弦相似度矩阵 (n, n)。"""
    import numpy as np

    e = np.asarray(embeddings, dtype=float)
    if normalize:
        e = e / np.clip(np.linalg.norm(e, axis=1, keepdims=True), 1e-12, None)
    return np.dot(e, e.T)


def cosine_to_query(query_vec, embeddings, normalize: bool = True):
    """每个角色与 query 的余弦相似度，展平成 (n,)。"""
    import numpy as np

    e = np.asarray(embeddings, dtype=float)
    q = np.asarray(query_vec, dtype=float)
    if normalize:
        e = e / np.clip(np.linalg.norm(e, axis=1, keepdims=True), 1e-12, None)
        q = q / np.clip(np.linalg.norm(q), 1e-12, None)
    return np.dot(e, q.T).flatten()


# ---- 两目标（移植自 manager.py，均取负号 → 最小化问题）----
def objective_relevance(group: Sequence[int], query_sims) -> float:
    """calculate_objective_1：组内角色与 query 的平均相关性（负）。"""
    import numpy as np

    return float(-np.mean([query_sims[i] for i in group]))


def objective_diversity(group: Sequence[int], sim_matrix) -> float:
    """calculate_objective_2：组内角色相似度子矩阵的 Vendi 多样性分数（负）。

    用官方同款 `vendi_score`（不自实现）。单元素组的 Vendi=1.0。
    """
    import numpy as np
    from vendi_score import vendi

    g = list(group)
    submatrix = np.asarray(sim_matrix)[np.ix_(g, g)]
    return float(-vendi.score_K(submatrix))


# ---- 非支配排序（逐行移植自 Optimizer.py，maximize=False）----
def _dominates(a, b) -> bool:
    """最小化语义：a 支配 b 当 a<=b 全部且 a<b 至少一项。"""
    import numpy as np

    a = np.asarray(a)
    b = np.asarray(b)
    return bool(np.all(a <= b) and np.any(a < b))


def fast_non_dominated_sort(objectives: Sequence[Sequence[float]]) -> list[list[int]]:
    """返回按前沿分层的下标列表；fronts[0] = Pareto 最优集。"""
    from collections import defaultdict

    import numpy as np

    S: dict[int, list[int]] = defaultdict(list)
    n = np.zeros(len(objectives), dtype=int)
    rank = np.zeros(len(objectives), dtype=int)
    fronts: list[list[int]] = [[]]

    for i, a in enumerate(objectives):
        for j, b in enumerate(objectives):
            if i == j:
                continue
            if _dominates(a, b):
                S[i].append(j)
            elif _dominates(b, a):
                n[i] += 1
        if n[i] == 0:
            rank[i] = 0
            fronts[0].append(i)

    k = 0
    while fronts[k]:
        next_front: list[int] = []
        for i in fronts[k]:
            for j in S[i]:
                n[j] -= 1
                if n[j] == 0:
                    rank[j] = k + 1
                    next_front.append(j)
        k += 1
        fronts.append(next_front)

    return fronts[:-1]  # 末尾空前沿丢弃
