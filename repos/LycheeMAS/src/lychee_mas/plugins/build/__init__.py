"""build —— 构建接缝（REGISTRY 类别 ``graph_builder`` + ``agent_selector``）。

- base.py  AgentSelector/GraphBuilder 协议 re-export + build_langgraph 统一入口
"""
from __future__ import annotations

from .base import AgentSelector, GraphBuilder, build_langgraph

__all__ = ["AgentSelector", "GraphBuilder", "build_langgraph"]
