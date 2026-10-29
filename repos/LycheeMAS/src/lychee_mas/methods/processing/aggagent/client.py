"""聚合模型客户端接缝 —— 对齐原版 call_server / litellm 语义的纯标准库替代。

出处: https://github.com/princeton-pli/AggAgent（commit 9638f7d, 2026-04-29）
原文件: aggagent/agent.py 的 ``call_server``。原版经 litellm 路由 hosted_vllm /
gemini / openai 多后端；本框架不引 litellm（黄金法则 2），统一收窄为 **OpenAI
兼容 chat/completions 端点**（vLLM 等本地服务或任意 OpenAI 兼容网关）：

- 端点要求服务端开 tool-calling（vLLM: --enable-auto-tool-choice + --tool-call-parser），
  与论文 rollout 的启动方式一致；
- 请求体/重试/错误语义复刻原版 call_server 默认值（temperature 1.0、top_p 0.95、
  max_tokens 10000、parallel_tool_calls False、指数退避重试）——仅去掉重试间随机
  jitter（确定性），并把错误归一成两个哨兵字符串交给 engine 处理：
  ``"ContextLengthError"``（上下文超限，engine 走强制 finish 路径）与 ``"Server error"``。

协议：``complete(messages, tools=None) -> dict | str``——返回 assistant 步字典
{role, content, reasoning_content, tool_calls, [usage]}，或上面任一错误哨兵
字符串（与原版 call_server 返回类型一致，便于逐字对照移植）。

mock 测试注入满足同协议的脚本化客户端即可离线跑通（见 tests/test_aggagent.py）。
许可证与修改说明见 NOTICE.md。
"""
from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from typing import Any, Protocol, runtime_checkable

# 原版 call_server 的默认采样参数（litellm kwargs，逐字保留）
_TEMPERATURE = 1.0
_TOP_P = 0.95
_MAX_TOKENS = 10000
_MAX_TRIES = 5


@runtime_checkable
class AggClient(Protocol):
    """聚合模型客户端：一次带工具定义的 chat 完成。

    ``tools`` 为 None 时请求体不带 tools（原版 only_finish 的 finish-only 简化等价，
    兼容不支持 tools 的服务端）。返回 assistant 步 dict 或错误哨兵字符串。
    """

    def complete(self, messages: list[dict], tools: list[dict] | None = None) -> dict | str: ...


class HttpAggClient:
    """OpenAI 兼容 tool-calling 客户端（urllib 实现，零第三方依赖）。

    参数:
        model: 模型名（vLLM served name 或网关认可的模型标识）。
        api_base: 端点基址，如 ``http://localhost:6000/v1``（拼 ``/chat/completions``）。
        api_key: 密钥；留空按 vLLM 惯例发 ``Bearer EMPTY``（原版 hosted_vllm 行为）。
        max_tries: 重试次数（指数退避 1,2,4,...，封顶 30s，无随机 jitter）。
        timeout: 单次 HTTP 超时秒数。
    """

    def __init__(
        self,
        model: str,
        api_base: str,
        api_key: str = "",
        max_tries: int = _MAX_TRIES,
        timeout: float = 60.0,
    ):
        if not model or not api_base:
            raise ValueError(
                f"HttpAggClient 需要 model 与 api_base，得到 {model=!r} {api_base=!r}")
        self.model = model
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key or "EMPTY"  # vLLM 惯例：无 key 时用占位 EMPTY
        self.max_tries = max(1, int(max_tries))
        self.timeout = timeout
        self._url = f"{self.api_base}/chat/completions"

    # -- 请求/响应 ----------------------------------------------------------

    def _request_body(self, messages: list[dict], tools: list[dict] | None) -> dict:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": _TEMPERATURE,
            "top_p": _TOP_P,
            "max_tokens": _MAX_TOKENS,
            "parallel_tool_calls": False,
        }
        if tools:
            body["tools"] = tools
        return body

    @staticmethod
    def _assistant_from_response(message: dict, usage: Any) -> dict:
        """把服务端 assistant 消息归一成 engine 消费的步 dict（≈原版 _message_to_dict）。

        只保留首个 tool_call（原版 ``tool_calls[:1]``）；arguments 保持 JSON 字符串，
        由 engine 侧 json.loads（与原版一致）。
        """
        tool_calls_list = []
        for tc in (message.get("tool_calls") or [])[:1]:
            fn = tc.get("function", {})
            tool_calls_list.append(
                {
                    "id": tc.get("id"),
                    "type": "function",
                    "function": {"name": fn.get("name", ""), "arguments": fn.get("arguments", "")},
                }
            )
        out = {
            "role": "assistant",
            "content": message.get("content") or "",
            "reasoning_content": message.get("reasoning_content") or "",
            "tool_calls": tool_calls_list,
        }
        if usage:
            out["usage"] = usage
        return out

    def complete(self, messages: list[dict], tools: list[dict] | None = None) -> dict | str:
        body = json.dumps(self._request_body(messages, tools), ensure_ascii=False).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        for attempt in range(self.max_tries):
            try:
                req = urllib.request.Request(self._url, data=body, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                choice = payload["choices"][0]["message"]
                return self._assistant_from_response(choice, payload.get("usage"))
            except urllib.error.HTTPError as e:
                text = ""
                try:
                    text = e.read().decode("utf-8", errors="replace")
                except Exception:
                    pass
                if "context length" in text.lower() or "input tokens" in text.lower():
                    return "ContextLengthError"  # 服务端拒绝：上下文超限哨兵
                # 其余 HTTP 错误与网络类错误同路：退避重试
            except Exception:
                pass  # 网络/超时/解析错误：退避重试
            if attempt < self.max_tries - 1:
                time.sleep(min(1 * (2**attempt), 30))  # 指数退避（原版无 jitter 版本）
        return "Server error"


# 导出工具调用名清洗（原版 sanitize_tool_name，供 engine 用）
_SANITIZE_RE = re.compile(r"[^a-zA-Z0-9_-]")


def sanitize_tool_name(name: str) -> str:
    """工具名清洗：只留 [a-zA-Z0-9_-]（原版逐字保留）。"""
    return _SANITIZE_RE.sub("", name)


__all__ = ["AggClient", "HttpAggClient", "sanitize_tool_name"]
