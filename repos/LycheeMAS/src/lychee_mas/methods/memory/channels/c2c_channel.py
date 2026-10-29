"""C2CLatentChannel —— 用训练好的 Cache-to-Cache 逐层 KV 融合器作为 latent 通道。

替代免训练的 soft_token（`latent.py`）：不再压成 (1,P,H) prefix，而是把「上一个 agent 的
KV-cache」经逐层 C2CProjector 融进本 agent 生成时的 KV（见 hf_backend.generate_chat_with_c2c）。

本类只负责**懒加载 + 持有 projector 栈**（"记忆方法" 这一侧的产物）；source 的组装与对齐由注入
client 完成（它握有 ctx：上一个 agent 的输入/输出）。

⚠️ 惰性导入（黄金法则 4）：import 本模块不触发 torch —— c2c_projector / torch 只在 _load() 内
import，保证离线环境可 import / 注册 `memory_manager/cdm`（它在 __init__ 持有本类引用）。
"""
from __future__ import annotations

import os
from typing import Any, Optional


class C2CLatentChannel:
    """持有一份训练好的 C2C projector 栈（首次 recall 时从 ckpt 懒加载到 backend.device）。"""

    def __init__(self, backend, ckpt_path: str, gate: str = "soft"):
        self.backend = backend
        self.ckpt_path = ckpt_path  # projectors.pt 文件或其所在目录
        self.gate = gate  # "soft"=σ(logit) 连续门；"hard"=logit>0 硬阈值
        self._projectors: Optional[Any] = None  # 懒加载缓存

    def get_projectors(self) -> Any:
        if self._projectors is None:
            self._projectors = self._load()
        return self._projectors

    def _load(self) -> Any:
        import torch  # 惰性导入

        from .c2c_projector import build_projector_stack  # 惰性：其模块顶层 import torch

        path = self.ckpt_path
        if path and os.path.isdir(path):
            path = os.path.join(path, "projectors.pt")
        if not path or not os.path.isfile(path):
            raise FileNotFoundError(f"C2C ckpt 不存在：{self.ckpt_path!r}")
        blob = torch.load(path, map_location=self.backend.device)
        b = blob["build"]
        projectors = build_projector_stack(
            b["num_hidden_layers"], b["head_dim"], b["num_kv_heads"],
            hidden_dim=b["hidden_dim"], intermediate_dim=b["intermediate_dim"],
            num_layers=b["num_layers"], dtype=torch.float32, zero_init=False,
            src_num_kv_heads=b.get("src_num_kv_heads"),
            src_head_dim=b.get("src_head_dim")).to(self.backend.device)
        for proj, sd in zip(projectors, blob["state_dicts"]):
            proj.load_state_dict(sd)
        projectors.eval()
        for proj in projectors:
            proj.hard_gate = (self.gate == "hard")
        return projectors
