"""兼容 shim：``pre_run_optimizer/agentdropout`` 的实现已下沉到 methods 侧。

实现（统一接口 ``optimize``：``mode="apply"`` 逐轮挂载 / ``mode="optimize"`` 驱动训练）
现在位于 ``methods/prerun/agentdropout/optimizer.py``——与 ``pre_run_optimizer/maspo``
同款：**注册类在 methods，``plugins/`` 不留实现**。本文件仅为旧 import 路径
（``lychee_mas.plugins.prerun.agentdropout_lg``）留一行 re-export，导入的是同一个类对象
（非拷贝），注册副作用由 methods 模块的装饰器触发，此处**不得再写一份 register**。
"""
from __future__ import annotations

from ...methods.prerun.agentdropout.optimizer import AgentDropoutLG

__all__ = ["AgentDropoutLG"]
