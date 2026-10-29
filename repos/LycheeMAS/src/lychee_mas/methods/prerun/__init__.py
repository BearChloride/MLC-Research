"""methods/prerun —— 运行前优化的方法实现（被 plugins/prerun 挂载）。

  agentprune.py   AgentPrune 时空掩码剪枝（ICLR 2025；训练=REINFORCE 自带）
  agentdropout/   AgentDropout 动态节点/边淘汰（ACL 2025；统一接口类 AgentDropoutLG +
                  两阶段训练日程在同包 trainer.py）
  maspo/          MASPO 联合提示优化（ICML 2026；训练=fixed-rounds beam search 自带）
  gepa/           GEPA 反思式提示演化（optimizer/gepa；graph-native 适配待接）
  graphops.py     共享图原语（Edge / 掩码 / 拓扑序 / Realization / Adam，同族方法+graphview 共用）
  prune_base.py   GraphPruner / VocabAdapter 协议（方法族内部契约）
  prune_stubs.py  agentdropout_v2 桩（+ agentprune 兼容 re-export）；agentvocab.py 词表适配桩

import 本包触发全部注册。
"""
from __future__ import annotations

from . import agentvocab, prune_stubs  # noqa: F401  触发桩注册（含 agentprune）
from .agentdropout import AgentDropoutLG, AgentDropoutOptimizer, AgentDropoutTrainer
from .agentprune import AgentPrunePruner
from .gepa import GEPAOptimizer  # noqa: F401  触发 optimizer/gepa 注册
from .maspo import MASPOOptimizer  # noqa: F401  触发 pre_run_optimizer/maspo 注册
from .prune_base import GraphPruner, VocabAdapter

__all__ = ["AgentPrunePruner", "AgentDropoutLG", "AgentDropoutOptimizer",
           "AgentDropoutTrainer", "GraphPruner", "VocabAdapter", "MASPOOptimizer",
           "GEPAOptimizer"]
