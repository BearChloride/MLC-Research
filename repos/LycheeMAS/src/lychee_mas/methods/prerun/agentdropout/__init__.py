"""AgentDropout 动态节点/边淘汰（ACL 2025）——本包注册两个类别。

  optimizer.py  AgentDropoutOptimizer（采样 / 更新 / 淘汰 / 实现矩阵）+ acyclic_realization
                + ``@REGISTRY.register("graph_pruner", "agentdropout")``
                + AgentDropoutLG——统一接口 ``optimize``（``mode="apply"`` 逐轮挂载 /
                ``mode="optimize"`` 驱动 trainer）+ ``@REGISTRY.register("pre_run_optimizer",
                "agentdropout")``；图级读写只经 plugins/prerun/graphview.py（函数内导入）
  trainer.py    两阶段训练日程（RoundPlan / RolloutStats / AgentDropoutTrainer / TrainReport），
                LLM 只经注入的 rollout / reward / predict 回调触达

接缝实现与 ``maspo/`` 同款放在本包（plugins/prerun/agentdropout_lg.py 只剩兼容 shim）。
import 本包触发注册（纯标准库，无重依赖）。
"""
from __future__ import annotations

from .optimizer import AgentDropoutLG, AgentDropoutOptimizer, acyclic_realization
from .trainer import (
    PHASE1_BATCH,
    PHASE1_BATCHES,
    PHASE2_BATCH,
    PHASE2_BATCHES,
    PRUNE_BATCHES,
    AgentDropoutTrainer,
    Predict,
    Reward,
    Rollout,
    RolloutStats,
    RoundPlan,
    TrainReport,
    full_temporal_edges,
)

__all__ = [
    "AgentDropoutLG",
    "AgentDropoutOptimizer",
    "acyclic_realization",
    "AgentDropoutTrainer",
    "TrainReport",
    "RoundPlan",
    "RolloutStats",
    "Rollout",
    "Reward",
    "Predict",
    "full_temporal_edges",
    "PHASE1_BATCHES",
    "PHASE1_BATCH",
    "PHASE2_BATCHES",
    "PHASE2_BATCH",
    "PRUNE_BATCHES",
]
