"""通道实现——**两个主接口**：`nl`（自然语言）与 `latent`（隐空间）。manager 只依赖这两个。

- nl.py            NL 通道：prev_output（原样转发）/ simplemem（SimpleMem 长时记忆问答，重依赖惰性）
- latent.py        Latent 通道**统一接口**：soft_token 压 prefix，或 c2c KV 融合（torch 惰性）
- c2c_channel.py   （latent 的子实现，非主接口）训练好的 C2C 融合器；仅被 latent.py 内部调用
- c2c_projector.py （latent 的子实现，非主接口）逐层 C2CProjector；被 c2c_channel 内部调用

⚠️ c2c_* / simplemem 都是主接口内部按需 import 的「方法」，manager 不直接 import 它们。
"""
from __future__ import annotations

from .latent import LatentMemory, compress_hidden, soft_token_prefix
from .nl import MEMORY_HEADER, PREV_OUTPUT_HEADER, NLMemory

__all__ = [
    "NLMemory",
    "PREV_OUTPUT_HEADER",
    "MEMORY_HEADER",
    "LatentMemory",
    "compress_hidden",
    "soft_token_prefix",
]
