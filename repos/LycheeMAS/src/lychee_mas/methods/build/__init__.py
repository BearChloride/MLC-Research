"""多智能体网络构建（Construct，CLAUDE.md §0）。

- AgentSelector       团队组建（agent_selector/agentinit，桩）
- StaticGraphBuilder  graph_builder/static（team 模板 → 契约 StateGraph）

import 本包触发上述组件注册（不触发 torch/autogen）。
"""
from __future__ import annotations

from .base import AgentSelector, GraphBuilder
from .lg_builder import StaticGraphBuilder  # noqa: F401  触发 graph_builder/static 注册
from .selectors import AgentInitSelector
from .templates import (
    ROLE_SYSTEM,
    TEAMS,
    Role,
    team_to_agentspecs,
)

__all__ = [
    "AgentSelector",
    "GraphBuilder",
    "AgentInitSelector",
    "Role",
    "ROLE_SYSTEM",
    "TEAMS",
    "team_to_agentspecs",
]
