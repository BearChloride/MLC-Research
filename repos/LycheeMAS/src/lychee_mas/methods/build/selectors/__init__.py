"""团队组建（AgentSelector 接缝）。

注册 `agent_selector/agentinit`：AgentInit（EMNLP 2025 Findings）的多样性×相关性 Pareto 选择。
- `mode="pool"`   已实现（M1）：固定候选池 + 确定性嵌入 + 非支配排序（离线/消融，零 GPU/API）。
- `mode="generate"` 待实现（M2）：忠实 AgentInit，LLM 现场生成角色 + HF embedder + LLM 挑组。

本包被 import 时只登记类，不加载 numpy/vendi/torch（重库均在 select() 内惰性导入）。
"""
from __future__ import annotations

from .agentinit import AgentInitSelector

__all__ = ["AgentInitSelector"]
