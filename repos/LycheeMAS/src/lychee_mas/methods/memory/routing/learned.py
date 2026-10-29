"""学习式路由器 —— 在 RouterInputs 特征上做轻量分类 -> channel。

TODO：把 RouterInputs 特征化（role/task one-hot、turn、query embedding、收发对），
训一个小 MLP，用【反事实蒸馏】监督（离线两通道各跑一遍，按下游结果标更优者）。
目前是占位：委托给 fallback 路由器，让框架在分类器就绪前能端到端跑通（维持占位 fallback 行为）。
"""
from __future__ import annotations

from ....core.registry import REGISTRY
from .base import MemoryRouter, RouteDecision, RouterInputs
from .static import StaticRouter


@REGISTRY.register("memory_router", "learned")
class LearnedRouter(MemoryRouter):
    name = "learned"

    def __init__(self, fallback: MemoryRouter | None = None):
        self.fallback = fallback or StaticRouter(default="nl")  # 桩阶段的兜底策略
        self.model = None  # TODO: 训练好的分类器；为 None 时走 fallback

    def decide(self, x: RouterInputs) -> RouteDecision:
        if self.model is None:
            # 还没训分类器：直接用 fallback 决策，并在 reason 里标明是桩
            d = self.fallback.decide(x)
            d.reason = f"learned(stub->{self.fallback.name}): {d.reason}"
            return d
        raise NotImplementedError  # TODO: 特征化 x -> model 预测 channel/P
