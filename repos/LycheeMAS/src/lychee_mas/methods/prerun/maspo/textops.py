"""MASPO 文本工具（逐行移植自 https://github.com/wangzx1219/MASPO 的 utils.py / optimizers.py）。

- ``sanitize_prompt``          候选提示占位符校验/修复（非法即回退旧提示，原版 _sanitize_prompt）
- ``extract_answer``           从 raw 输出抽最终答案（<answer> → \\boxed → 末句截断，
                               原版 utils.extract_answer）
- ``parse_comparison_result``  解析成对比较结果（<choose>A/B</choose> → 尾字母回退，
                               原版 utils.parse_comparison_result）
- ``extract_prompt_tag``       抽 <prompt>...</prompt>（原版 _propose_new_prompt 的抽取步）

纯标准库。
"""
from __future__ import annotations

import html
import re
from typing import Optional


def sanitize_prompt(prompt: str, old_p: str) -> str:
    """校验/修复候选提示的占位符；无法修复即返回 ``old_p``（原版语义：回退不炸）。

    规则（与原版 _sanitize_prompt 逐条一致）：
    - 连续花括号归一化为单花括号；
    - 允许的占位符只有 ``{question}`` / ``{context}``（纯数字与空花括号被转义），
      出现其他名字的占位符 → 回退旧提示；
    - ``{question}`` 缺失自动补一行，出现多于一次 → 回退；
    - 旧提示带 ``{context}`` 时新提示缺失自动补一行，多于一次 → 回退；
      旧提示不带时新提示多于一次 → 回退。
    """

    def normalize_braces(text: str) -> str:
        text = re.sub(r"\{+", "{", text)
        text = re.sub(r"\}+", "}", text)
        return text

    prompt = normalize_braces(prompt)
    old_p = normalize_braces(old_p)

    for ph in re.findall(r"\{([^}]*)\}", prompt):
        if ph in ("question", "context") or ph.isdigit() or ph == "":
            continue
        return old_p

    escaped = prompt.replace("{", "{{").replace("}", "}}")
    final_prompt = escaped.replace("{{question}}", "{question}")
    final_prompt = final_prompt.replace("{{context}}", "{context}")

    question_count = final_prompt.count("{question}")
    if question_count == 0:
        final_prompt += "\nQuestion: {question}"
    elif question_count > 1:
        return old_p

    context_count = final_prompt.count("{context}")
    if "{context}" in old_p:
        if context_count == 0:
            final_prompt += "\nContext: {context}"
        elif context_count > 1:
            return old_p
    elif context_count > 1:
        return old_p

    return final_prompt


def extract_answer(raw: str) -> str:
    """从 raw 输出抽最终答案：<answer> 标签 → \\boxed{} → 末句截断（原版 extract_answer）。"""
    raw = raw.strip()
    for _ in range(3):
        new_raw = html.unescape(raw)
        if new_raw == raw:
            break
        raw = new_raw

    matches = re.findall(r"<answer>(.*?)</answer>", raw, re.S)
    if matches:
        return matches[-1].strip()

    box_pat = re.compile(r"\\boxed\s*\{((?:[^{}]|\{[^{}]*\})*)\}", re.S)
    m = box_pat.search(raw)
    if m:
        return m.group(1).strip()

    sentences = re.split(r"[。\n;]+", raw)
    last = sentences[-1].strip()
    return last[-30:] if len(last) > 30 else last


def parse_comparison_result(raw: str) -> bool:
    """True = A（候选）更好。<choose> 标签优先，其后从末尾找首个字母，默认 True（原版语义）。"""
    if "<choose>" in raw and "</choose>" in raw:
        try:
            choice = raw.split("<choose>")[1].split("</choose>")[0]
            return "A" in choice
        except IndexError:  # pragma: no cover - split 保证有值，防御性保留
            pass
    for c in reversed(raw.strip().upper()):
        if c.isalpha():
            return c == "A"
    return True


def extract_prompt_tag(raw: str) -> Optional[str]:
    """抽 <prompt>...</prompt> 内容；抽不到返回 None（调用方决定回退，原版打 WARN 回退旧提示）。"""
    try:
        return raw.split("<prompt>")[1].split("</prompt>")[0].strip()
    except IndexError:
        return None
