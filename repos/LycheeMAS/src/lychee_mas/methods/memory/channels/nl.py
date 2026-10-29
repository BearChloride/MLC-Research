"""NLMemory —— 自然语言上下文通道（按 strategy 选方法）。

可用 strategy：
- "prev_output"（默认）：把上一个 agent 的输出**原样**向下传递（不截断），加来源标志
  PREV_OUTPUT_HEADER，便于接收 agent 指认「这是上一个 agent 的输入」。无状态。
- "simplemem"：接 **SimpleMem**（长时对话记忆 + 检索问答，`from simplemem import SimpleMem`）。
  有状态：observe() 把每条对话喂 `add_dialogue`；recall() 用 `finalize`+`ask` 检索出与当前 query
  相关的记忆，作为要注入的 NL 文本（加 MEMORY_HEADER）。

⚠️ 惰性导入（黄金法则 2/4）：`simplemem` 是重外部依赖（LLM + 向量库 + 嵌入模型，且需 API/模型），
只在真正用到时（`_sm()` 内）import；import 本模块不触发它，selfcheck 仍 `HEAVY LOADED: NONE`。
未安装时给出清晰报错（`pip install -e ../SimpleMem`），不影响 prev_output。
"""
from __future__ import annotations

from typing import Any, Optional

# 注入「上一个 agent 输出」时加的来源标志：接收方在 prompt 里据此指认这是上一个 agent 的输入。
# ⚠️ 本常量在此【集中定义】，是全包唯一来源；roles/templates 的 # INPUT 段引用它（消除重复字符串）。
PREV_OUTPUT_HEADER = "## Input from the previous agent"
# simplemem 检索出的相关记忆注入时的来源标志（区别于「上一个 agent 输出」）。
MEMORY_HEADER = "## Relevant memory (retrieved)"


class NLMemory:
    """NL 通道：按 strategy 选择 NL 记忆方法（prev_output 无状态 / simplemem 有状态）。"""

    # 需要「有状态记忆后端」的 NL 策略（observe() 逐条喂对话、recall() 检索问答）。
    # 后续新增此类方法（mem0 / contriever …）在此登记即可，无需改 observe/reset 的分支逻辑。
    STATEFUL_STRATEGIES = frozenset({"simplemem"})

    def __init__(self, tokenizer=None, strategy: str = "prev_output",
                 simplemem_kwargs: Optional[dict] = None):
        self.tok = tokenizer  # 预留（如需按 token 处理）；当前不使用
        self.strategy = strategy  # 选用哪种 NL 记忆方法
        self._sm_kwargs = simplemem_kwargs or {}  # 传给 SimpleMem(...) 的构造参数
        self._mem: Any = None  # 懒建的 SimpleMem 实例（simplemem 策略）
        self._added = 0  # 本对话已喂入 SimpleMem 的对话条数

    # ---- 生命周期（有状态策略用；prev_output 下为 no-op）----
    def reset(self) -> None:
        # 新对话 -> 丢弃旧 SimpleMem 实例（清空其记忆库），重新开始
        self._mem = None
        self._added = 0

    def observe(self, speaker: str, content: str, timestamp: Optional[str] = None) -> None:
        """把一条新对话喂进 NL 记忆。有状态策略 → add_dialogue；其它（如 prev_output）→ no-op。"""
        if self.strategy not in self.STATEFUL_STRATEGIES:
            return
        if not (content or "").strip():
            return
        self._sm().add_dialogue(speaker or "agent", content, timestamp)
        self._added += 1

    # ---- 召回：产出要注入的 NL 文本 ----
    def recall(self, text: str) -> str:
        """按 strategy 产出要注入的 NL 文本。

        prev_output：text = 上一个 agent 的输出，原样（加 PREV_OUTPUT_HEADER）。
        simplemem  ：text = 当前 query，用 SimpleMem 检索相关记忆的答案（加 MEMORY_HEADER）。
        """
        if self.strategy == "prev_output":
            text = text or ""
            return f"{PREV_OUTPUT_HEADER}\n{text}" if text else ""  # 加来源标志，空内容不注入
        if self.strategy == "simplemem":
            return self._recall_simplemem(text)
        raise ValueError(f"unknown NL strategy {self.strategy}")

    # ---- simplemem 内部实现（重依赖惰性）----
    def _sm(self):
        """懒建 SimpleMem 实例；未安装则清晰报错（不影响 import / 其它策略）。"""
        if self._mem is None:
            try:
                from simplemem import SimpleMem  # 重外部依赖，仅此处按需 import
            except ImportError as e:  # pragma: no cover - 取决于环境是否装了 simplemem
                raise ImportError(
                    "nl_strategy='simplemem' 需要安装 SimpleMem："
                    "`pip install -e ../SimpleMem`（并配置 LLM API/模型）。"
                ) from e
            self._mem = SimpleMem(**self._sm_kwargs)  # auto 模式：首个方法调用选定 text 后端
        return self._mem

    def _recall_simplemem(self, query: str) -> str:
        if self._added == 0:  # 还没有任何记忆 -> 不注入
            return ""
        mem = self._sm()
        mem.finalize()  # 处理残余缓冲（安全检查；可重复调用）
        answer = mem.ask(query or "")  # 意图感知检索 + 生成答案
        answer = (answer or "").strip()
        return f"{MEMORY_HEADER}\n{answer}" if answer else ""
