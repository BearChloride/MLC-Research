"""DualChannelMemoryManager —— CDM 默认记忆方法（在研，CLAUDE.md §5：memory_manager/cdm）。

维护**单一**记忆 source（seed 进来的基础上下文，如 LoCoMo 对话历史 + 运行中的 MAS transcript），
并从这同一个 source 物化出路由器选定的任意通道——保证 apples-to-apples（对比时只变「注入/路由」）。

  channel none   -> 空 bundle（什么都不注入）
  channel nl     -> bundle.NL_Channel = NLMemory.recall(query)（prev_output / simplemem）
  channel latent -> LatentMemory.materialize -> bundle.Latent_Channel（prefix / c2c projector）
  channel both   -> 上面两个都产

**两个主接口 = `nl` 与 `latent`**：latent 的所有子策略（soft_token / c2c 等）都由 `LatentMemory`
内部处理（c2c 时它内部调 `c2c_channel`）；manager 不直接 import c2c，只依赖 nl / latent。

注册为 `memory_manager/cdm`（自然语言/隐空间动态通道选择，CDM 在研重点）。import 本模块不触发 torch
（NLMemory/LatentMemory 的 torch 已惰性化），满足离线可注册（黄金法则 4）。
"""
from __future__ import annotations

from typing import Any, List

from ....core.registry import REGISTRY
from ..base import MemoryBundle, MemoryManager
from ..channels.latent import LatentMemory
from ..channels.nl import NLMemory
from ..routing.base import RouteDecision


@REGISTRY.register("memory_manager", "cdm")
class DualChannelMemoryManager(MemoryManager):
    name = "cdm"

    def __init__(self, backend, latent_strategy: str = "soft_token",
                 nl_strategy: str = "prev_output",
                 max_encode_tokens: int = 4096, include_transcript: bool = True,
                 P: int = 16, c2c_ckpt: str | None = None, c2c_gate: str = "soft",
                 nl_simplemem: dict | None = None):
        self.backend = backend
        self.latent_strategy = latent_strategy
        # 两个主接口：NL 通道 + latent 通道（latent 统管 soft_token / c2c 子策略；P 归 latent）
        self.nl = NLMemory(getattr(backend, "tok", None), strategy=nl_strategy,
                           simplemem_kwargs=nl_simplemem)
        self.latent = LatentMemory(backend, latent_strategy, max_encode_tokens,
                                   P=P, c2c_ckpt=c2c_ckpt, c2c_gate=c2c_gate)
        self.include_transcript = include_transcript  # 是否把运行中的对话也并入 source
        self._seed = ""  # 基础上下文（任务历史），seed() 预载
        self._transcript: List[str] = []  # 运行中累积的对话行
        self._last_observed: str | None = None  # 去重：上一条已观察的对话行

    # ---- 生命周期 ----
    def reset(self) -> None:
        # 每个样本开始前清空（由 run_one_task 的 ctx.reset 间接驱动 / 显式调用）
        self._seed = ""
        self._transcript = []
        self._last_observed = None
        self.latent.reset()  # 清 latent prefix 缓存（source 变了）
        self.nl.reset()  # 清 NL 有状态记忆（如 simplemem：新对话 = 新记忆库）

    def seed(self, context_text: str) -> None:
        """预载基础上下文（任务自带的历史），manager 全程持有。长程记忆任务在此灌入对话历史。"""
        self._seed = context_text or ""
        self.latent.reset()  # source 变了，缓存失效

    def observe(self, messages: List[Any]) -> None:
        """把最新一条消息喂给两条通道：latent 的 transcript（source）+ NL 有状态记忆（simplemem）。

        messages: {role, content} 字典列表（最新在末尾）。只取最后一条，避免每轮把整段历史重复追加；
        内容与上一条相同则去重。nl.observe 对 prev_output no-op、对 simplemem 为 add_dialogue。
        """
        if not messages:
            return
        last = messages[-1]
        # 兼容 dict 与对象两种消息形态
        is_dict = isinstance(last, dict)
        role = last.get("role", "?") if is_dict else getattr(last, "source", "?")
        content = last.get("content", "") if is_dict else getattr(last, "content", "")
        if not (isinstance(content, str) and content.strip()):
            return
        line = f"[{role}]: {content.strip()}"
        if line == self._last_observed:  # 与上一条相同 -> 去重（不重复喂两条通道）
            return
        self._last_observed = line
        self.nl.observe(role, content.strip())  # NL 通道观察新对话（simplemem→add_dialogue）
        if self.include_transcript:
            self._transcript.append(line)
            self.latent.reset()  # source 变了，缓存失效

    # ---- 各通道召回所依据的统一 source ----
    def _source(self) -> str:
        # seed 基础上下文 + 运行 transcript 拼成单一来源，NL 与 latent 都从它产出（
        # apples-to-apples）
        parts = []
        if self._seed:
            parts.append(self._seed)
        if self.include_transcript and self._transcript:
            parts.append("\n".join(self._transcript))
        return "\n\n".join(parts)

    # ---- 物化通道 ----
    def recall(self, decision: RouteDecision, query: str) -> MemoryBundle:
        if decision.channel == "none":
            return MemoryBundle(meta={"channel": "none"})  # 什么都不注入
        bundle = MemoryBundle(meta={"channel": decision.channel})
        if decision.uses_nl():  # nl / both
            # NL 通道：prev_output 原样转发上一个 agent 输出 / simplemem 检索问答
            bundle.NL_Channel = self.nl.recall(query)
            bundle.NL_strategy = self.nl.strategy
        if decision.uses_latent():  # latent / both
            # 全权交给 latent 主接口：内部按 strategy 决定 soft_token(→prefix) 或 c2c(→projector 栈)
            self.latent.materialize(bundle, self._source())
        return bundle
