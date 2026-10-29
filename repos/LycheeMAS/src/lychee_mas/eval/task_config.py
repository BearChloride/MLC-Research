"""每个 task 的默认配置：用哪支队伍 + 用哪种答案提取策略（task 级默认，可被覆盖）。

迁移自 benchs/task_config.py。集中管理「对应 task 的默认配置」：
  - team:      该 task 默认用的队伍 profile 名（在 construct/templates.py 的 TEAMS 里解析成角色
  列表）
  - extractor: 从 MAS 对话里抽最终答案的策略名（见下方 EXTRACTORS）

未登记的 task 用 DEFAULT_TEAM / DEFAULT_EXTRACTOR 兜底。纯标准库（仅 re）。
"""

from __future__ import annotations

import re
from typing import Callable, Optional


# ====================== 答案提取策略（按任务分发） ======================
def _extract_default(messages) -> str:
    """默认策略：优先取最后一条含 'APPROVE:' 的内容；没有则回退最后一条消息。"""
    for m in reversed(messages):  # 从后往前找第一条 APPROVE
        c = getattr(m, "content", "") if not isinstance(m, dict) else m.get("content", "")
        if isinstance(c, str) and "APPROVE" in c:
            mt = re.search(r"APPROVE:\s*(.*)", c, re.DOTALL)  # 抓 APPROVE: 后的全部文本
            if mt:
                return mt.group(1).strip()
    last = messages[-1] if messages else None  # 没 APPROVE（如达轮数上限）就用末条
    if last is None:
        return ""
    c = getattr(last, "content", "") if not isinstance(last, dict) else last.get("content", "")
    return (c or "").strip()


def _extract_boxed(messages) -> str:
    r"""数学/AIME 策略：在 default 基础上，抽出 APPROVE 行里 \boxed{...} 内的答案。

    答案可能是整数（含负数）或带符号/分式/根式的式子，故取 \boxed{} 里的原始内容、不强转整数。
    """
    ans = _extract_default(messages)
    m = re.search(r"\\boxed\{(.*)\}", ans, re.DOTALL)
    return m.group(1).strip() if m else ans  # 没有 \boxed 就退回原文本


# 命名策略表
EXTRACTORS: dict[str, Callable[[list], str]] = {
    "default": _extract_default,
    "boxed": _extract_boxed,
}


# ====================== 每个 task 的默认配置 ======================
DEFAULT_TEAM = "default"  # 未登记 task 的默认队伍 profile
DEFAULT_EXTRACTOR = "default"  # 未登记 task 的默认答案提取策略

# task 名 -> {team: 队伍 profile 名, extractor: 提取策略名}
TASK_CONFIG: dict[str, dict] = {
    "gsm8k": {"team": "reason", "extractor": "default"},
    "aime_2024": {"team": "aime", "extractor": "boxed"},
    "medqa": {"team": "fact", "extractor": "default"},
    "arc_easy": {"team": "fact", "extractor": "default"},
    "openbookqa": {"team": "fact", "extractor": "default"},
    "locomo10": {"team": "memory", "extractor": "default"},
    "human_eval": {"team": "human_eval", "extractor": "default"},
    "gaia_validation": {"team": "gaia", "extractor": "default"},
    "gaia_validation_level_1": {"team": "gaia", "extractor": "default"},
    "gaia_validation_level_2": {"team": "gaia", "extractor": "default"},
    "gaia_validation_level_3": {"team": "gaia", "extractor": "default"},
    # MLC（ACL 2025）数学数据集族：数值答案用 default 提取器；MATH 是 LaTeX 答案用 boxed
    "addsub": {"team": "reason", "extractor": "default"},
    "singleeq": {"team": "reason", "extractor": "default"},
    "multiarith": {"team": "reason", "extractor": "default"},
    "asdiv": {"team": "reason", "extractor": "default"},
    "svamp": {"team": "reason", "extractor": "default"},
    "math": {"team": "aime", "extractor": "boxed"},
    "aftraj_audit": {"team": "reason", "extractor": "default"},
    "aftraj_audit_test": {"team": "reason", "extractor": "default"},
    "agent_collab_idr": {"team": "reason", "extractor": "default"},
    "agent_collab_rtd": {"team": "reason", "extractor": "default"},
    "agent_collab_cpr": {"team": "reason", "extractor": "default"},
    "agent_collab_clc": {"team": "reason", "extractor": "default"},
    "mast_failure": {"team": "reason", "extractor": "default"},
    "open_agent_traces": {"team": "reason", "extractor": "default"},
}


def team_name_for_task(task: Optional[str]) -> str:
    """按任务名选出默认队伍 profile 名；未登记回退 DEFAULT_TEAM。"""
    return TASK_CONFIG.get(task or "", {}).get("team", DEFAULT_TEAM)


def extractor_for_task(task: Optional[str]) -> Callable[[list], str]:
    """按任务名选出答案提取策略；未登记回退 DEFAULT_EXTRACTOR。"""
    name = TASK_CONFIG.get(task or "", {}).get("extractor", DEFAULT_EXTRACTOR)
    return EXTRACTORS[name]
