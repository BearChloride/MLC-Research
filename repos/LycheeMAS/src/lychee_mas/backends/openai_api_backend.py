"""OpenAI-compatible chat backend for API-only experiments.

This backend implements the small synchronous interface consumed by
``autogen_injection_client``: ``generate_chat`` returns text plus usage/latency
metadata. It is intended for none/NL-channel runs when local GPUs are busy.
Latent prefix methods require local hidden states and are therefore not
available for API providers.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


@dataclass
class APIGenResult:
    text: str
    n_prompt_pos: int
    n_gen_tokens: int
    latency_s: float
    prefix_len: int = 0


class OpenAICompatibleBackend:
    def __init__(
        self,
        model: str,
        *,
        base_url: Optional[str] = None,
        api_key_env: str = "DASHSCOPE_API_KEY",
        api_key: Optional[str] = None,
        timeout: float = 120.0,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        seed: Optional[int] = None,
        model_info: Optional[Dict[str, Any]] = None,
    ):
        from openai import OpenAI

        resolved_key = api_key or os.environ.get(api_key_env)
        if not resolved_key:
            raise ValueError(f"API key not found; set {api_key_env} or pass api_key explicitly.")
        self.model_name = model
        self.base_url = base_url or os.environ.get("DASHSCOPE_BASE_URL") or DEFAULT_BASE_URL
        self.api_key_env = api_key_env
        self.timeout = timeout
        self.temperature = temperature
        self.top_p = top_p
        self.seed = seed
        self.model_info = model_info or {}
        self.tok = None
        self.client = OpenAI(api_key=resolved_key, base_url=self.base_url, timeout=timeout)

    def generate_chat(self, messages: List[Dict], max_new_tokens: int = 256) -> APIGenResult:
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": [{"role": m["role"], "content": m["content"]} for m in messages],
            "max_tokens": max_new_tokens,
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.top_p is not None:
            payload["top_p"] = self.top_p
        if self.seed is not None:
            payload["seed"] = self.seed

        t0 = time.time()
        response = self.client.chat.completions.create(**payload)
        latency = time.time() - t0
        choice = response.choices[0] if response.choices else None
        text = choice.message.content if choice and choice.message else ""
        usage = getattr(response, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        return APIGenResult(text or "", prompt_tokens, completion_tokens, latency)

    def generate_chat_with_prefix(self, messages: List[Dict], prefix, max_new_tokens: int = 256):
        raise NotImplementedError(
            "OpenAICompatibleBackend does not support latent prefix generation. "
            "Use method=none or method=nl_only for API runs."
        )

    def encode_hidden(self, text: str, max_tokens: int = 4096):
        raise NotImplementedError(
            "OpenAICompatibleBackend cannot expose hidden states. "
            "Use the local HF backend for latent-channel experiments."
        )
