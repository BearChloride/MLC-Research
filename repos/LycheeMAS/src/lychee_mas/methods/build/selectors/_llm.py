"""AgentInit generate 模式的 LLM 调用助手（决策 B：轻量 OpenAI 兼容 chat，与选择逻辑解耦）。

对应官方 `AgentInit/llm/gpt_chat.py::achat`，但：
  - 用**同步** OpenAI 客户端（框架 `AgentSelector.select` 是同步协议；角色生成是顺序状态机，
    无需异步。避免在 Orchestrator 的异步上下文里 `asyncio.run` 触发 "event loop is running"）。
  - endpoint/key/model 走 env（`LYCHEE_LLM_BASE_URL/API_KEY/MODEL`），绝不硬编码（黄金法则 §7）。
  - openai **惰性导入**（本模块 import 期不加载 openai；`make selfcheck` 保持 NONE）。

`chat()` 返回 (text, usage_tokens)；usage 供 token 记账（AgentInit 卖点是「降 token」，选择本身的
生成开销必须计入，否则消融不诚实）。测试时用 fake chat_fn 注入,不走真实网络（见 tests）。
"""
from __future__ import annotations

import os
from dataclasses import dataclass

# tenacity 是轻量纯 python 库；本模块只在 generate 模式被 import（惰性），不影响 selfcheck。
from tenacity import retry, stop_after_attempt, wait_random_exponential


@dataclass
class ChatUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total(self) -> int:
        return int(self.prompt_tokens) + int(self.completion_tokens)


def chat(
    messages: list[dict],
    *,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    timeout: float = 600.0,
) -> tuple[str, ChatUsage]:
    """同步一次 chat completion。返回 (回复文本, ChatUsage)。

    endpoint/key/model 缺省从 env 读；**缺配置立即抛 RuntimeError（不重试——非瞬时错误）**。
    真正的网络调用在 `_chat_call` 里，瞬时错误自动重试 3 次、指数退避（对齐官方 achat）。
    """
    model = model or os.getenv("LYCHEE_LLM_MODEL")
    base_url = base_url or os.getenv("LYCHEE_LLM_BASE_URL")
    api_key = api_key or os.getenv("LYCHEE_LLM_API_KEY")
    if not model or not api_key:
        raise RuntimeError(
            "agentinit generate mode needs an LLM endpoint: set env LYCHEE_LLM_MODEL, "
            "LYCHEE_LLM_API_KEY (and LYCHEE_LLM_BASE_URL for an OpenAI-compatible server), "
            "or inject a chat_fn (tests use a fake one)."
        )
    return _chat_call(messages, model, base_url, api_key, timeout)


@retry(wait=wait_random_exponential(max=60), stop=stop_after_attempt(3), reraise=True)
def _chat_call(messages, model, base_url, api_key, timeout) -> tuple[str, ChatUsage]:
    """真正的 API 调用；只有瞬时/网络错误在此重试（配置错误已在上游 chat() 拦截）。"""
    from openai import OpenAI  # 惰性导入（黄金法则 §2）

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
    resp = client.chat.completions.create(model=model, messages=messages)
    text = resp.choices[0].message.content or ""
    usage = ChatUsage()
    if getattr(resp, "usage", None) is not None:
        usage.prompt_tokens = int(getattr(resp.usage, "prompt_tokens", 0) or 0)
        usage.completion_tokens = int(getattr(resp.usage, "completion_tokens", 0) or 0)
    return text, usage
