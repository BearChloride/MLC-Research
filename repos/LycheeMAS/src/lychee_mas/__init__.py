"""LycheeMAS —— 运行时可插拔的多智能体系统研究框架（架构设计见 docs/DESIGN.md）。

把整个 MAS 统一表示为带时序/记忆状态的有向图 G=(V,E,W,T,M)；五个模块 = 五个挂载式接缝，
一切算法经 REGISTRY 按名挂载在 LangGraph 图上：

  build → prerun → memory → compile → processing 包裹执行 → postrun

两层结构：``plugins/`` 定义接缝（协议 + 统一入口 + method 分发），``methods/`` 存方法
（论文复现/训练循环/重机器）。``eval/`` 评测、``core/`` 类型+注册表、``backends/`` 生成原语。

import 本包会触发所有可插拔组件的注册，但**不触发** torch / transformers / langgraph
（重依赖全部惰性导入）：`import lychee_mas` 与 `REGISTRY.snapshot()` 纯离线可用。
"""
from __future__ import annotations

__version__ = "0.3.0"

# 依次 import 触发组件注册（side-effect import；放进 __all__ 以避免被判为未使用）。
from . import backends, eval, methods, plugins
from .core.registry import REGISTRY

__all__ = ["__version__", "REGISTRY", "plugins", "methods", "eval", "backends"]
