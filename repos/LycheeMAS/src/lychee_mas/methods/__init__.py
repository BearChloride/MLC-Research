"""methods —— 实现层（厚）：论文复现 / 训练循环 / 重机器，按五接缝镜像分组。

  build/       static 模板、agentinit 选队
  prerun/      agentprune、agentdropout（接缝类 + 两阶段训练）、maspo/、gepa/
               （+ 桩：agentdropout_v2、agentvocab）
  memory/      channels / managers / routing / store / context（CDM 记忆线算法库）
  processing/  serial / parallel + self_consistency / dynamicagg 归约器
  postrun/     attributor / credit_assigner 桩 + TraceStore

注册装饰器随实现走；import 本包触发全部注册（重依赖一律惰性导入）。
"""
from __future__ import annotations

from . import build, memory, postrun, prerun, processing  # noqa: F401

__all__ = ["build", "prerun", "memory", "processing", "postrun"]
