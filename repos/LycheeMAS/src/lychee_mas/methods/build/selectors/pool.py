"""预设角色候选池（`mode="pool"` 用）。

官方 AgentInit 是**按 query 用 LLM 现场生成角色**（见 `_generate.py`，M2）。`pool` 模式是它的
**离线/消融替身**：从一个固定候选池里选，用一个**确定性 hash 嵌入**代替 HF embedder、用
**确定性 argmax** 代替 LLM 的 SelectGroup。好处：零 GPU/API、可复现、CI 可跑，且天然是
「AgentInit 去掉生成」的对照组。

选择数学与 `_generate.py` 完全一致（`_pareto.py` 的余弦 + Vendi + 非支配排序），只是嵌入来源不同：
这里用角色描述 / query 文本的 char-trigram hash 向量（纯 stdlib，`select()` 时才惰性用 numpy）。
"""
from __future__ import annotations

from ....core.types import AgentSpec, TaskQuery

# ---- 候选角色（描述里放区分性关键词，供 hash 嵌入算 query 相关性）----
# 每项：(name, description, system_prompt)。name 需为合法标识符、池内唯一。
CANDIDATE_ROLES: list[tuple[str, str, str]] = [
    ("math_solver",
     "mathematics arithmetic algebra calculus number theory step by step derivation proof",
     "You are the MATH SOLVER. Solve the problem with explicit step-by-step derivation and "
     "state the final numeric answer clearly."),
    ("code_writer",
     "programming code python function algorithm implementation debugging software",
     "You are the CODE WRITER. Produce correct, runnable code for the task and briefly explain "
     "the key logic."),
    ("logician",
     "logic reasoning deduction inference constraints consistency formal argument",
     "You are the LOGICIAN. Analyze the problem's logical structure, list constraints, and "
     "reason deductively toward the answer."),
    ("fact_checker",
     "factual knowledge encyclopedic facts multiple choice question answering verification",
     "You are the FACT CHECKER. Answer the factual/multiple-choice question and justify the "
     "chosen option in one line."),
    ("planner",
     "planning decomposition strategy subtasks orchestration high level plan steps",
     "You are the PLANNER. Decompose the task into a short concrete plan; do not solve it "
     "yourself."),
    ("critic",
     "critique verification error checking review independent recompute validation",
     "You are the CRITIC. Independently re-check the candidate answer and point out any "
     "concrete error, or approve it."),
    ("retriever",
     "retrieval memory long context evidence quoting grounding search relevant facts",
     "You are the RETRIEVER. Extract and quote only the facts from the provided context that "
     "are relevant to the question."),
    ("creative_writer",
     "creative writing narrative storytelling open ended language expression trivia",
     "You are the CREATIVE WRITER. Produce a fluent, imaginative response that satisfies the "
     "open-ended prompt."),
    ("data_analyst",
     "data analysis statistics tables quantitative estimation reasoning over numbers",
     "You are the DATA ANALYST. Reason quantitatively over the given data and report the "
     "supported conclusion."),
    ("science_expert",
     "science physics chemistry biology commonsense scientific reasoning experiment",
     "You are the SCIENCE EXPERT. Apply scientific and commonsense reasoning to reach the "
     "answer."),
]


def _hash_embed(text: str, dim: int = 256):
    """确定性 char-trigram hash 嵌入（纯 stdlib + numpy）→ L2 归一化 (dim,) 向量。

    用 hashlib（Python 内置 hash() 有随机盐，不可复现，故不用）。相同文本 → 相同向量。
    """
    import hashlib

    import numpy as np

    vec = np.zeros(dim, dtype=float)
    s = f"  {text.lower()}  "
    for i in range(len(s) - 2):
        tri = s[i:i + 3]
        h = int(hashlib.md5(tri.encode("utf-8")).hexdigest(), 16)
        vec[h % dim] += 1.0
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 0 else vec


def candidate_pool(query: TaskQuery | None = None) -> list[AgentSpec]:
    """把候选池物化成 AgentSpec 列表。`profile["vec"]` 缓存该角色描述的确定性嵌入。

    query 暂不改变候选集合（固定池）；相关性在 selector 里用 query 的同款嵌入现算。
    保留参数是为与 `_generate.py::candidate_pool(query)` 的签名对齐（M2 可换成按 query 生成）。
    """
    specs: list[AgentSpec] = []
    for name, desc, system in CANDIDATE_ROLES:
        specs.append(AgentSpec(
            name=name, role=name, system_prompt=system,
            profile={"vec": _hash_embed(desc).tolist(), "description": desc},
            meta={"description": desc, "source": "agentinit_pool"}))
    return specs


def embed_query(query: TaskQuery):
    """query 文本的同款确定性嵌入（与候选描述同一空间，供算相关性）。"""
    text = (query.question or "")
    if query.context:
        text = f"{text} {query.context}"
    return _hash_embed(text)
