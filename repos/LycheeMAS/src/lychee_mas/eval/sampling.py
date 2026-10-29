"""固定种子的评测子集抽样（对齐论文「每数据集随机抽 N 题」口径）。

框架的 benchmark loader 支持 ``load(task, n)`` 取**前** n 条；但 MLC（ACL 2025）等论文
的口径是「每个数据集**随机**抽 100 题」。本模块提供可复现的随机抽样，并把抽样元数据
（种子、池大小、被抽中的下标）原样返回，供实验脚本写进 config 快照，
使任何人事后都能精确复原同一子集。

**只决定「评哪些题」，不影响「怎么判对错」**——评分口径完全不变。

用法::

    pool = benchmarks.load("gsm8k")          # 全量
    subset, meta = sample_records(pool, n=100, seed=0)
    # meta = {"sample_seed": 0, "pool_size": 1319, "sampled": 100, "sampled_indices": [...]}
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Sequence, Tuple


def sample_records(
    records: Sequence[Any],
    n: Optional[int] = None,
    seed: int = 0,
) -> Tuple[List[Any], Dict[str, Any]]:
    """从 records 中固定种子随机抽 n 条，返回 (子集, 元数据)。

    - ``n`` 为空或 ``n >= len(records)``：返回全集（不抽样），元数据里如实记录。
    - 抽样下标排序后返回，保证**顺序稳定**（选择随机、排列确定）。
    - 空输入显式报错，不做静默兜底。
    """
    pool_size = len(records)
    if pool_size == 0:
        raise ValueError("sample_records: 输入为空，无法抽样")

    if not n or n >= pool_size:
        meta = {
            "sample_seed": seed,
            "pool_size": pool_size,
            "sampled": pool_size,
            "sampled_indices": None,
        }
        return list(records), meta

    rng = random.Random(seed)
    indices = sorted(rng.sample(range(pool_size), n))
    meta = {
        "sample_seed": seed,
        "pool_size": pool_size,
        "sampled": len(indices),
        "sampled_indices": indices,
    }
    return [records[i] for i in indices], meta
