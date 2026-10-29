"""backends —— 生成原语层：「怎么调一个 LLM」（plugins/methods 回答「用 LLM 做什么」）。

  hf_backend.py         本地 HF 模型（generate_chat / encode_hidden / prefix-KV 原语）
  openai_api_backend.py OpenAI 兼容 API（MASPO 的 API 评估端等）
  spans.py              运行事件落盘（JsonlSpanLogger，回归对拍原始数据）

methods 里的算法不直接 import transformers/openai——由脚本用本层构造回调注入。
重依赖全部惰性导入（selfcheck 零依赖纪律的唯一豁免点之一）。
"""
from __future__ import annotations

__all__ = ["hf_backend", "openai_api_backend", "spans"]
