"""MASProgram —— 把一个 MAS 系统表示为「可变异文本组件」的集合（Optimizer 的操作对象）。

组件键约定：
  - ``agent:<name>:system_prompt``   某个 agent 的 system prompt（MVP 唯一可变异组件）
  - ``topology:description``         拓扑描述（暴露给反思器做上下文，当前冻结不变异）

纯标准库、零重依赖。变异产生新实例（`mutated`），不原地修改。
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from ....core.types import AgentSpec

AGENT_PROMPT_PREFIX = "agent:"
AGENT_PROMPT_SUFFIX = ":system_prompt"
TOPOLOGY_KEY = "topology:description"


def _agent_key(name: str) -> str:
    return f"{AGENT_PROMPT_PREFIX}{name}{AGENT_PROMPT_SUFFIX}"


def _agent_name_of(key: str) -> str | None:
    if key.startswith(AGENT_PROMPT_PREFIX) and key.endswith(AGENT_PROMPT_SUFFIX):
        return key[len(AGENT_PROMPT_PREFIX):-len(AGENT_PROMPT_SUFFIX)]
    return None


@dataclass(frozen=True)
class MASProgram:
    """系统的文本组件视图。`mutable_keys` 之外的组件只作反思上下文，不参与变异。"""

    components: dict[str, str]
    mutable_keys: tuple[str, ...]
    meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_agents(cls, agents: list[AgentSpec]) -> "MASProgram":
        """从 AgentSpec 列表提取组件：每个 agent 的 system prompt + 拓扑描述。"""
        components: dict[str, str] = {}
        for spec in agents:
            components[_agent_key(spec.name)] = spec.system_prompt
        components[TOPOLOGY_KEY] = " -> ".join(spec.name for spec in agents)
        mutable = tuple(_agent_key(spec.name) for spec in agents)
        return cls(components=components, mutable_keys=mutable, meta={})

    def apply_to(self, agents: list[AgentSpec]) -> list[AgentSpec]:
        """把组件写回：产出**新** AgentSpec 列表（深拷贝节点，替换 system prompt）。

        列表里存在但 program 未携带对应组件的节点原样保留；program 里多出的 agent 组件
        （列表中无此节点）说明系统不匹配，显式报错。
        """
        known = {spec.name for spec in agents}
        for key in self.components:
            name = _agent_name_of(key)
            if name is not None and name not in known:
                raise KeyError(f"MASProgram 组件 {key!r} 找不到对应 agent")
        out: list[AgentSpec] = []
        for spec in agents:
            prompt = self.components.get(_agent_key(spec.name))
            if prompt is not None and prompt != spec.system_prompt:
                out.append(replace(spec, system_prompt=prompt))
            else:
                out.append(replace(spec))
        return out

    def mutated(self, key: str, text: str) -> "MASProgram":
        """返回把组件 `key` 替换为 `text` 的新 program（key 必须在 mutable_keys 内）。"""
        if key not in self.mutable_keys:
            raise KeyError(f"组件 {key!r} 不可变异（mutable_keys={self.mutable_keys!r}）")
        components = dict(self.components)
        components[key] = text
        return MASProgram(components=components, mutable_keys=self.mutable_keys,
                          meta=dict(self.meta))
