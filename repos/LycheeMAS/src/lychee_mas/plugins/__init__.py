"""plugins —— 五模块接缝的接口层（薄：协议、统一入口、method 分发、校验、薄适配）。

  build/       构建   build_langgraph(method, ...) -> StateGraph
  prerun/      运行前 optimize_langgraph(sg, method, ...) -> sg（base + graphview 节点契约）
  memory/      运行时 attach_memory(sg, method, ...) -> sg（P3 语义实现中，显式桩）
  processing/  执行   run_processed(runner, method, ...) -> ProcessingResult
  postrun/     运行后 analyze_run(读侧归因) + optimize_postrun(图+轨迹→图闭环)
               + train_from_runs(写侧训练)

论文级算法一律在姊妹包 ``methods/``（按接缝镜像分组），注册装饰器随实现走；
本包 import 各接缝模块即触发全部注册。langgraph 惰性导入（selfcheck 零重依赖）。
"""
from __future__ import annotations

from .build import AgentSelector, GraphBuilder, build_langgraph
from .memory import attach_memory
from .postrun import PostRunOptimizer, Trainer, analyze_run, optimize_postrun, train_from_runs
from .prerun import optimize_langgraph
from .processing import run_processed

__all__ = [
    "build_langgraph",
    "optimize_langgraph",
    "attach_memory",
    "run_processed",
    "analyze_run",
    "optimize_postrun",
    "train_from_runs",
    "AgentSelector",
    "GraphBuilder",
    "Trainer",
    "PostRunOptimizer",
]
