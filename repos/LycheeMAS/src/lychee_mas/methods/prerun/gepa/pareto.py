"""GEPA 的候选选择核心 —— per-instance Pareto（纯标准库，离线可测）。

candidates 的分数表示：``scores[candidate_id] = [该候选在训练集每条 instance 上的得分]``，
所有候选的分数向量长度必须一致（同一训练集）。
"""
from __future__ import annotations

import random
from typing import Mapping, Sequence


def _check(scores: Mapping[str, Sequence[float]]) -> None:
    if not scores:
        raise ValueError("scores 为空：至少需要一个候选")
    lengths = {len(v) for v in scores.values()}
    if len(lengths) != 1:
        raise ValueError(f"候选分数向量长度不一致：{lengths}")
    if 0 in lengths:
        raise ValueError("分数向量为空：训练集不能为空")


def per_instance_best(scores: Mapping[str, Sequence[float]]) -> list[set[str]]:
    """每条 instance 上取得最高分的候选集合（GEPA 的『实例级最优前沿』）。"""
    _check(scores)
    n = len(next(iter(scores.values())))
    fronts: list[set[str]] = []
    for i in range(n):
        best = max(v[i] for v in scores.values())
        fronts.append({cid for cid, v in scores.items() if v[i] == best})
    return fronts


def dominated(a: Sequence[float], b: Sequence[float]) -> bool:
    """a 被 b 支配：b 在所有 instance 上不差于 a，且至少一处严格更好。"""
    return all(y >= x for x, y in zip(a, b)) and any(y > x for x, y in zip(a, b))


def pareto_pool(scores: Mapping[str, Sequence[float]]) -> set[str]:
    """剔除被支配候选，返回 Pareto 前沿上的候选 id 集合。"""
    _check(scores)
    ids = list(scores)
    return {
        cid for cid in ids
        if not any(dominated(scores[cid], scores[other]) for other in ids if other != cid)
    }


def sample_candidate(scores: Mapping[str, Sequence[float]], rng: random.Random) -> str:
    """从 Pareto 前沿按『取得实例级最优的次数』加权采样一个母本候选。"""
    pool = pareto_pool(scores)
    fronts = per_instance_best(scores)
    weights = {cid: sum(1 for front in fronts if cid in front) for cid in pool}
    total = sum(weights.values())
    if total <= 0:  # 全零权重（理论上仅当前沿候选从未拿过实例级最优；均匀退化）
        return rng.choice(sorted(pool))
    pick = rng.uniform(0.0, total)
    acc = 0.0
    for cid in sorted(pool):
        acc += weights[cid]
        if pick <= acc:
            return cid
    return sorted(pool)[-1]
