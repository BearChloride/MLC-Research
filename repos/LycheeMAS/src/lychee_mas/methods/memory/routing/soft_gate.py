"""软门控 —— MoE 式的通道混合权重（论文主菜）。

TODO：产出连续门控权重 g = softmax(W·features)，在 {none,nl,latent,both} 上分布；
注入端（runtime/backends/autogen_injection_client.py）按 g 混合通道（缩放 latent prefix 和/或 NL
强度）。
初始化用【反事实蒸馏】，再用 RL 微调（CLAUDE.md §10）。
目前是占位：委托给 fallback，让框架在门控训练好之前能跑通（维持占位 fallback 行为）。
"""
from __future__ import annotations

from ....core.registry import REGISTRY
from .base import MemoryRouter, RouteDecision, RouterInputs
from .static import StaticRouter


@REGISTRY.register("memory_router", "soft_gate")
class SoftGateRouter(MemoryRouter):
    name = "soft_gate"

    def __init__(self, fallback: MemoryRouter | None = None):
        # 桩阶段兜底用 both（目标是软混合，default=both 最接近其行为）
        self.fallback = fallback or StaticRouter(default="both")
        self.gate = None  # TODO: 训练好的门控；为 None 时走 fallback

    def decide(self, x: RouterInputs) -> RouteDecision:
        if self.gate is None:
            # 还没训门控：用 fallback 决策，reason 标明是桩
            d = self.fallback.decide(x)
            d.reason = f"soft_gate(stub->{self.fallback.name}): {d.reason}"
            return d
        raise NotImplementedError  # TODO: 门控权重 -> 混合注入（需 Composer 配合做软混合）
