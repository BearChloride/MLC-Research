"""MATH-500 loader and data preparation.

MASPO（https://github.com/wangzx1219/MASPO）主实验数据集之一：HuggingFaceH4/MATH-500
的 500 题子集（jsonl 随 MASPO 仓库发布）。评分走 kind="aime"（数值 + sympy 符号等价），
gold 多为 LaTeX 表达式（如 ``\\frac{14}{3}``）。
"""

from __future__ import annotations

import json
import os
import shutil
import urllib.request
from typing import Dict, List, Optional

from .common import prepared_root

# MASPO 仓库随发的 jsonl（字段：problem / solution / answer / subject / level / unique_id）
_RAW_URL = ("https://raw.githubusercontent.com/wangzx1219/MASPO/main/"
            "dataset/math-500/math_500.jsonl")
_LOCAL_ENV = "LYCHEE_MATH500_JSONL"  # 已有本地文件（如 MASPO 克隆）时直接复制，免下载


def _jsonl_path() -> str:
    return os.path.join(prepared_root(), "math500", "math_500.jsonl")


def prepare_math500(force: bool = False, source: Optional[str] = None) -> str:
    path = _jsonl_path()
    if not force and os.path.isfile(path) and os.path.getsize(path) > 0:
        return os.path.dirname(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    local = os.environ.get(_LOCAL_ENV)
    if local:
        if not os.path.isfile(local):
            raise FileNotFoundError(f"{_LOCAL_ENV}={local!r} 指向的文件不存在")
        shutil.copyfile(local, path)
        return os.path.dirname(path)
    try:
        urllib.request.urlretrieve(_RAW_URL, path)  # noqa: S310 - 固定 https 源
    except Exception as exc:  # pragma: no cover - 网络相关
        raise RuntimeError(
            f"MATH-500 下载失败（{_RAW_URL}）：{exc}；"
            f"可设 {_LOCAL_ENV}=<MASPO 克隆>/dataset/math-500/math_500.jsonl 走本地复制"
        ) from exc
    return os.path.dirname(path)


def load_math500(n: Optional[int] = None) -> List[Dict]:
    prepare_math500()
    out: List[Dict] = []
    with open(_jsonl_path(), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            it = json.loads(line)
            out.append(
                {
                    "task": "math500",
                    "kind": "aime",
                    "question": it["problem"].strip(),
                    "gold": str(it["answer"]).strip(),
                    "context": None,
                }
            )
            if n and len(out) >= n:
                break
    if not out:
        raise RuntimeError(f"MATH-500 加载为空：{_jsonl_path()}")
    return out
