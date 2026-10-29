"""L1 agent_selector/agentinit —— generate 模式（M2），离线、确定性。

不调真实 LLM / 不下载模型：注入 **fake chat_fn**（按 prompt 特征返回罐装响应）+ **mock embed_fn**
（确定性 hash 向量）。测的是**状态机编排 / 解析正则 / 前沿→SelectGroup / 降级路径 / token 记账**，
不是模型本身。需 numpy + vendi_score（select() 时惰性加载）；缺则跳过。

真实端到端冒烟测试见文件末尾，默认 skip，需配 env（LYCHEE_LLM_* + LYCHEE_EMBED_MODEL）。
"""
from __future__ import annotations

import os

import pytest

pytest.importorskip("numpy")
pytest.importorskip("vendi_score")

from lychee_mas.core.registry import REGISTRY  # noqa: E402
from lychee_mas.core.types import AgentSpec, TaskQuery  # noqa: E402
from lychee_mas.methods.build.selectors._llm import ChatUsage  # noqa: E402
from lychee_mas.methods.build.selectors.pool import _hash_embed  # noqa: E402

# 一个合法的格式化响应：两个角色 + 执行计划（供解析）。
_FORMATTED = """\
## Selected Roles List:
```
```

## Created Roles List:
```
{
  "name": "Math Solver",
  "description": "solves math problems step by step",
  "suggestions": "show every step",
  "prompt": "You are a math expert, named Math Solver. Your goal is to solve."
},
{
  "name": "Answer Verifier",
  "description": "checks the candidate answer independently",
  "suggestions": "recompute",
  "prompt": "You are a verifier, named Answer Verifier. Your goal is to check."
}
```

## Execution Plan:
1. [Math Solver]: solve the problem
2. [Answer Verifier]: verify the result

## RoleFeedback
none

## PlanFeedback
none
"""


def _fake_chat_factory(formatted: str = _FORMATTED, checks: str = "No Suggestions"):
    """按 user prompt 里的特征返回罐装响应；返回 (text, ChatUsage) 以顺带测 token 记账。"""
    def fake_chat(messages):
        prompt = messages[-1]["content"]
        if "formatting expert" in prompt:
            text = formatted
        elif "created Expert Roles" in prompt:              # CheckRoles
            text = f"## Suggestions\n{checks}"
        elif "Execution Plan follows" in prompt:            # CheckPlans
            text = f"## Suggestions\n{checks}"
        elif "selecting the best group" in prompt:          # SelectGroup
            text = "Reason...\nChoice: Group 1"
        else:                                               # CreateRoles 生成
            text = "some raw role brainstorming with roles and a plan"
        return text, ChatUsage(prompt_tokens=10, completion_tokens=5)
    return fake_chat


def _fake_embed(sentences):
    import numpy as np
    if isinstance(sentences, str):
        sentences = [sentences]
    return np.stack([_hash_embed(s) for s in sentences])


def _sel(**cfg):
    base = dict(mode="generate", embed_fn=_fake_embed, chat_fn=_fake_chat_factory())
    base.update(cfg)
    return REGISTRY.create("agent_selector", "agentinit", **base)


def _q(text="Solve this algebra problem for x"):
    return TaskQuery(question=text)


# ---- 正常路径 ----
def test_generate_returns_specs_from_generated_roles():
    out = _sel(critique_rounds=1).select(_q())
    assert 1 <= len(out) <= 2
    assert all(isinstance(a, AgentSpec) for a in out)
    # 角色名来自生成结果，且被 sanitize 成合法标识符（空格→_）
    names = {a.name for a in out}
    assert names <= {"Math_Solver", "Answer_Verifier"}
    assert all(a.system_prompt.startswith("You are") for a in out)


def test_names_sanitized_original_kept_in_meta():
    out = _sel(critique_rounds=1).select(_q())
    spec = next(a for a in out if a.name == "Math_Solver")
    assert spec.meta["role_name"] == "Math Solver"  # 原名保留
    assert spec.role.isidentifier()


def test_token_accounting_on_selector_not_specs():
    # #2：整轮生成开销记在 selector.last_gen_tokens，不再逐 spec 塞 gen_tokens。
    sel = _sel(critique_rounds=1)
    out = sel.select(_q())
    assert sel.last_gen_tokens > 0
    assert all("gen_tokens" not in a.meta for a in out)


def test_critique_loop_reaches_consensus():
    # critique_rounds>=2 会真正走 CheckRoles/CheckPlans；均返回 No Suggestions → 共识 → 不报错
    out = _sel(critique_rounds=3).select(_q())
    assert len(out) >= 1


def test_deterministic():
    a = [x.name for x in _sel(critique_rounds=1).select(_q())]
    b = [x.name for x in _sel(critique_rounds=1).select(_q())]
    assert a == b


# ---- #1 token 预算封顶迭代 ----
def test_token_budget_caps_iteration():
    from lychee_mas.core.types import Budget, BudgetUnit

    # checks 返回非 "No Suggestions" → 不早停共识 → 多轮迭代会真的发生。
    calls = {"n": 0}

    def counting(messages):
        calls["n"] += 1
        prompt = messages[-1]["content"]
        if "formatting expert" in prompt:
            return _FORMATTED, ChatUsage(10, 5)
        if "created Expert Roles" in prompt or "Execution Plan follows" in prompt:
            return "## Suggestions\n1. tweak something", ChatUsage(10, 5)
        if "selecting the best group" in prompt:
            return "Choice: Group 1", ChatUsage(10, 5)
        return "raw brainstorm", ChatUsage(10, 5)

    def run(budget):
        calls["n"] = 0
        REGISTRY.create("agent_selector", "agentinit", mode="generate", critique_rounds=3,
                        embed_fn=_fake_embed, chat_fn=counting).select(_q(), budget=budget)
        return calls["n"]

    n_free = run(None)
    n_budget = run(Budget(limit=1, unit=BudgetUnit.TOKENS))
    assert n_budget < n_free  # 极小 token 预算 → 更早停迭代 → 更少 LLM 调用


def test_calls_budget_ignored_in_generate():
    # #1：generate 只认 token 预算；calls 单位应被忽略（不影响迭代）。
    from lychee_mas.core.types import Budget, BudgetUnit
    out = _sel(critique_rounds=1).select(_q(), budget=Budget(limit=1, unit=BudgetUnit.CALLS))
    assert len(out) >= 1


# ---- #4 重名去重 ----
def test_duplicate_sanitized_names_deduped():
    dup = """\
## Created Roles List:
```
{"name": "Math Solver", "description": "d1", "suggestions": "s", "prompt": "You are A."},
{"name": "Math-Solver", "description": "d2", "suggestions": "s", "prompt": "You are B."}
```
## Execution Plan:
1. [Math Solver]: a
2. [Math-Solver]: b
## RoleFeedback
none
## PlanFeedback
none
"""
    sel = REGISTRY.create("agent_selector", "agentinit", mode="generate", critique_rounds=1,
                          min_roles=2, max_roles=2, embed_fn=_fake_embed,
                          chat_fn=_fake_chat_factory(formatted=dup))
    out = sel.select(_q())
    names = [a.name for a in out]
    assert len(names) == len(set(names))          # 无重复 name
    assert "Math_Solver" in names and "Math_Solver_2" in names


# ---- #7 RoleFeedback/PlanFeedback 注入 + 正确路由 ----
def test_feedback_injected_and_routed():
    fb_formatted = _FORMATTED.replace("## RoleFeedback\nnone", "## RoleFeedback\nROLE_FB_MARKER") \
                             .replace("## PlanFeedback\nnone", "## PlanFeedback\nPLAN_FB_MARKER")
    seen = {"roles": None, "plans": None}

    def recording(messages):
        prompt = messages[-1]["content"]
        if "formatting expert" in prompt:
            return fb_formatted, ChatUsage(5, 5)
        if "created Expert Roles" in prompt:            # CheckRoles
            seen["roles"] = prompt
            return "## Suggestions\n1. tweak", ChatUsage(5, 5)
        if "Execution Plan follows" in prompt:          # CheckPlans
            seen["plans"] = prompt
            return "## Suggestions\n1. tweak", ChatUsage(5, 5)
        if "selecting the best group" in prompt:
            return "Choice: Group 1", ChatUsage(5, 5)
        return "raw", ChatUsage(5, 5)

    REGISTRY.create("agent_selector", "agentinit", mode="generate", critique_rounds=2,
                    embed_fn=_fake_embed, chat_fn=recording).select(_q())
    # 创建者反馈被注入到对应批判者，且路由正确（角色反馈进 CheckRoles，计划反馈进 CheckPlans）
    assert "ROLE_FB_MARKER" in seen["roles"]
    assert "PLAN_FB_MARKER" in seen["plans"] and "ROLE_FB_MARKER" not in seen["plans"]


# ---- 降级路径 ----
def test_invalid_format_response_falls_back_to_normal():
    # 格式化步骤返回无换行 → roles_plan 无 '\n' → 降级到 Normal
    sel = REGISTRY.create("agent_selector", "agentinit", mode="generate",
                          critique_rounds=1, embed_fn=_fake_embed,
                          chat_fn=_fake_chat_factory(formatted="INVALID"))
    out = sel.select(_q())
    assert len(out) == 1 and out[0].name == "Normal"


def test_no_roles_parsed_falls_back_to_normal():
    # 合法 '##' 结构但没有任何 JSON 角色块 → 解析出空 → 降级
    empty = "## Created Roles List:\n```\n```\n## Execution Plan:\n1. nothing\n## RoleFeedback\nx"
    sel = REGISTRY.create("agent_selector", "agentinit", mode="generate",
                          critique_rounds=1, embed_fn=_fake_embed,
                          chat_fn=_fake_chat_factory(formatted=empty))
    out = sel.select(_q())
    assert len(out) == 1 and out[0].name == "Normal"


def test_missing_llm_env_raises_without_injection():
    # 不注入 chat_fn 且没配 env → 明确报错（不静默）
    for k in ("LYCHEE_LLM_MODEL", "LYCHEE_LLM_API_KEY"):
        if k in os.environ:
            pytest.skip("LLM env configured; skip negative test")
    sel = REGISTRY.create("agent_selector", "agentinit", mode="generate", embed_fn=_fake_embed)
    with pytest.raises(RuntimeError):
        sel.select(_q())


# ---- 真实端到端冒烟（默认跳过；配 env 才跑）----
@pytest.mark.skipif(
    not (os.getenv("LYCHEE_LLM_API_KEY") and os.getenv("LYCHEE_EMBED_MODEL")),
    reason="needs real LLM + embedder env (LYCHEE_LLM_* / LYCHEE_EMBED_MODEL)",
)
def test_generate_real_smoke():
    out = REGISTRY.create("agent_selector", "agentinit", mode="generate").select(_q())
    assert len(out) >= 1 and all(isinstance(a, AgentSpec) for a in out)
