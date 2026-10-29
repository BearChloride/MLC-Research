"""prerun —— 运行前优化接缝（REGISTRY 类别 ``pre_run_optimizer``）。

- base.py         PreRunOptimizer 协议 + optimize_langgraph 统一入口（method 按名分发）
- graphview.py    GraphView / extract_view / rebuild（StateGraph 读写的唯一通道 + 节点契约）
- agentprune_lg.py  pre_run_optimizer/agentprune（薄适配，算法在 methods/prerun/agentprune）
- agentdropout_lg.py  兼容 shim：pre_run_optimizer/agentdropout 的实现已下沉到
                    ``methods/prerun/agentdropout/optimizer.py``（与 maspo 同款：注册类在
                    methods，本包不留实现），此处只留旧 import 路径

方法实现（agentdropout 的接缝类 + 两阶段训练日程、maspo/ gepa/ agentprune 训练机器）在
``methods/prerun/``；import methods 包触发其注册。langgraph 惰性导入。
"""
from __future__ import annotations

# agentprune 薄适配在此触发自身注册；agentdropout 已由 methods 侧注册，此处仅为旧路径兼容
from . import agentdropout_lg, agentprune_lg  # noqa: F401
from .base import PreRunOptimizer, optimize_langgraph
from .graphview import GraphView, extract_view, rebuild

__all__ = [
    "PreRunOptimizer",
    "optimize_langgraph",
    "GraphView",
    "extract_view",
    "rebuild",
]
