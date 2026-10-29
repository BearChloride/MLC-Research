"""评测口径的**审计工具**（只读，不改变任何官方打分逻辑）。

背景：``metrics.score_exact`` 在两边最后一个数字不相等时，会退化成「规整字符串包含」
判断，可能把"错误答案恰好包含正确答案数字子串"判成正确
（实例：gold=64、预测 ``\\boxed{10.648}``，``64`` 是 ``10648`` 的子串）。

评分口径不能改（要与团队/历史结果可比），因此审计按**第二口径**复算：

* **官方口径**：``metrics.score``，结果已写在 ``outputs.jsonl`` 的 ``score`` 字段；
* **严格数值口径**：抽取双方最后一个数字后按**数值**比较（``43`` 与 ``43.0`` 相等）。

两个数都报，结论以哪个为准由实验方决定。本模块只提供纯函数，不写任何产物。
"""

from __future__ import annotations

from typing import Any, Optional


def _to_float(text: Optional[str]) -> Optional[float]:
    if text is None:
        return None
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def strict_numeric_score(pred: str, gold: Any) -> Optional[float]:
    """严格数值口径：双方最后一个数字按数值比较。

    返回 ``None`` 表示**不适用**（gold 里抽不出数字，例如 ASDiv 的 Yes/No 类文本答案），
    调用方应把这些样本排除在严格口径之外单独统计，而不是当作错。
    """
    from .metrics import _last_number

    g = _to_float(_last_number(str(gold)))
    if g is None:
        return None
    p = _to_float(_last_number(str(pred)))
    if p is None:
        return 0.0
    return 1.0 if p == g else 0.0


def strict_correct(pred: str, gold: Any) -> Optional[bool]:
    """严格口径的布尔形式（不适用返回 None），便于配对统计。"""
    value = strict_numeric_score(pred, gold)
    return None if value is None else value == 1.0
