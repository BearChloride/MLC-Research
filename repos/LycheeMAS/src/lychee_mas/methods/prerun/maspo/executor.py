"""MASPO 的带缓存图执行器（移植自 https://github.com/wangzx1219/MASPO 的 agent.py）。

优化内循环需要「换某节点的提示后，只重跑该节点 + 全部后继，上游读缓存」——LangGraph 编译图
的 ``ainvoke`` 做不到局部重执行，因此优化期在 ``GraphView`` 上用本执行器跑（对应原版
``MAS.arun_with_cache`` / ``MAS.arun_from_node`` + ``InferenceCache``）；接口边界上进出仍是
StateGraph，真实评测始终走编译后的 LangGraph 图。

LLM 只经注入的 ``AsyncLLM``（``async (prompt) -> text``）触达；纯标准库。
"""
from __future__ import annotations

import asyncio
from typing import Any as GraphView  # 视图仅作注解（避免 methods→plugins 顶层环）
from typing import Awaitable, Callable, Dict, Optional, Set, Tuple

from .prompts import COMPRESS_PROMPT
from .textops import extract_answer

AsyncLLM = Callable[[str], Awaitable[str]]

CONTEXT_JOINER = "\n---\n"  # 原版 InferenceCache.get_context_for_node 的拼接符


class InferenceCache:
    """一条 query 的逐节点执行缓存（原版同名类，键改为节点名）。"""

    def __init__(self) -> None:
        self.question: str = ""
        self.node_inputs: Dict[str, str] = {}        # name -> 该节点看到的 context
        self.node_outputs_raw: Dict[str, str] = {}   # name -> 原始输出
        self.node_outputs_short: Dict[str, str] = {} # name -> 压缩短输出（下游 context 用）

    def set_node_data(self, name: str, context: str, raw: str, short: str) -> None:
        self.node_inputs[name] = context
        self.node_outputs_raw[name] = raw
        self.node_outputs_short[name] = short

    def context_for(self, predecessors: list[str], use_short: bool = True) -> str:
        outputs = self.node_outputs_short if use_short else self.node_outputs_raw
        return CONTEXT_JOINER.join(outputs[p] for p in predecessors if p in outputs)

    def clone_without(self, drop: Set[str]) -> "InferenceCache":
        """克隆缓存但丢掉 drop 中节点的数据（原版 clone_up_to 的补集写法）。"""
        new = InferenceCache()
        new.question = self.question
        new.node_inputs = {k: v for k, v in self.node_inputs.items() if k not in drop}
        new.node_outputs_raw = {k: v for k, v in self.node_outputs_raw.items() if k not in drop}
        new.node_outputs_short = {k: v for k, v in self.node_outputs_short.items()
                                  if k not in drop}
        return new


def format_agent_prompt(template: str, question: str, context: str) -> str:
    """模板 → 单条 user 消息（原版 Agent.arun_full 的 format + 异常回退 replace）。"""
    try:
        return template.format(question=question, context=context)
    except (KeyError, IndexError):
        return template.replace("{question}", str(question)).replace("{context}", context)


class CachedExecutor:
    """在 GraphView 上按通信 DAG 执行一条 query，产出 InferenceCache。

    ``prompt_map`` 覆盖各节点提示模板（原版 inject_prompt_map 的临时注入语义），
    不改 view 里的 spec。
    """

    def __init__(self, view: GraphView, agent_llm: AsyncLLM,
                 prompt_map: Optional[Dict[str, str]] = None) -> None:
        self.view = view
        self.agent_llm = agent_llm
        self.prompt_map = dict(prompt_map or {})

    def _template(self, name: str) -> str:
        return self.prompt_map.get(name, self.view.specs[name].system_prompt)

    async def _run_node(self, name: str, question: str, cache: InferenceCache) -> None:
        """跑一个节点：格式化 → 生成 raw → 压缩 short（终端改为抽答案），写入缓存。"""
        context = cache.context_for(self.view.predecessors[name])
        raw = await self.agent_llm(
            format_agent_prompt(self._template(name), question, context))
        if name == self.view.terminal:
            short = extract_answer(raw)  # 原版：终端节点 short = 抽取的最终答案
        else:
            short = (await self.agent_llm(COMPRESS_PROMPT.format(raw=raw))).strip()
        cache.set_node_data(name, context, raw, short)

    async def run(self, question: str) -> Tuple[str, InferenceCache]:
        """全图执行（原版 arun_with_cache：按依赖分层，层内并发）。返回 (最终答案, 缓存)。"""
        cache = InferenceCache()
        cache.question = question
        pending = list(self.view.names)
        done: Set[str] = set()
        while pending:
            level = [n for n in pending
                     if all(p in done for p in self.view.predecessors[n])]
            if not level:  # extract_view 已拒绝成环；防御性显式失败
                raise RuntimeError(f"通信图无法推进（疑似成环）：剩余 {pending}")
            await asyncio.gather(*[self._run_node(n, question, cache) for n in level])
            done.update(level)
            pending = [n for n in pending if n not in done]
        return extract_answer(cache.node_outputs_raw[self.view.terminal]), cache

    async def run_from_node(self, start: str, base_cache: InferenceCache,
                            new_prompt: Optional[str] = None
                            ) -> Tuple[str, InferenceCache]:
        """局部重执行（原版 arun_from_node）：只重跑 start + 全部后继，上游读 base_cache。

        ``new_prompt`` 只对 start 节点临时生效，不污染 executor 的 prompt_map。
        """
        if start not in self.view.specs:
            raise KeyError(f"run_from_node: 未知节点 {start!r}；图节点: {self.view.names}")
        rerun = {start} | self.view.all_successors(start)
        cache = base_cache.clone_without(rerun)
        saved = self.prompt_map.get(start)
        if new_prompt is not None:
            self.prompt_map[start] = new_prompt
        try:
            for name in self.view.names:  # names 已是拓扑序，逐个串行（原版同为串行）
                if name in rerun:
                    await self._run_node(name, cache.question or base_cache.question, cache)
        finally:
            if new_prompt is not None:
                if saved is None:
                    self.prompt_map.pop(start, None)
                else:
                    self.prompt_map[start] = saved
        cache.question = base_cache.question
        return extract_answer(cache.node_outputs_raw[self.view.terminal]), cache
