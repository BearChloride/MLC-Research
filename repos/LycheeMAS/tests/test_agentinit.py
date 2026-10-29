"""L1 agent_selector/agentinit（pool 模式）—— 离线、确定性。

需 numpy + vendi_score（官方同款多样性）；缺任一则跳过（对齐 test_c2c_projector 的 importorskip
约定）。这些是 select() 调用时才惰性加载的重库，不违反「import 期零重依赖」（另见 selfcheck 测试）。
"""
from __future__ import annotations

import pytest

pytest.importorskip("numpy")
pytest.importorskip("vendi_score")

from lychee_mas.core.registry import REGISTRY  # noqa: E402
from lychee_mas.core.types import AgentSpec, Budget, BudgetUnit, TaskQuery  # noqa: E402
from lychee_mas.methods.build.selectors import _pareto  # noqa: E402


def _sel(**cfg):
    return REGISTRY.create("agent_selector", "agentinit", mode="pool", **cfg)


def _q(text="Solve this algebra equation for x step by step"):
    return TaskQuery(question=text)


# ---- select() 契约 ----
def test_returns_agentspecs_within_bounds():
    out = _sel(min_roles=1, max_roles=4).select(_q())
    assert 1 <= len(out) <= 4
    assert all(isinstance(a, AgentSpec) for a in out)


def test_selected_come_from_candidate_pool():
    from lychee_mas.methods.build.selectors.pool import CANDIDATE_ROLES

    pool_names = {r[0] for r in CANDIDATE_ROLES}
    out = _sel().select(_q())
    assert {a.name for a in out} <= pool_names


def test_deterministic_same_seed():
    a = [x.name for x in _sel().select(_q())]
    b = [x.name for x in _sel().select(_q())]
    assert a == b


def test_relevance_picks_topical_role():
    # 数学 query 下，被选团队应包含与数学强相关的角色（相关性目标生效）。
    out = {a.name for a in _sel(min_roles=1, max_roles=3).select(_q("compute this arithmetic sum"))}
    assert "math_solver" in out


def test_pool_ignores_budget():
    # pool 不调 LLM，budget 一律不影响结果（budget 仅对 generate 的 token 迭代生效）。
    q = _q()
    sel = _sel(max_roles=4)
    a = [x.name for x in sel.select(q)]
    b = [x.name for x in sel.select(q, budget=Budget(limit=2, unit=BudgetUnit.CALLS))]
    c = [x.name for x in sel.select(q, budget=Budget(limit=100, unit=BudgetUnit.TOKENS))]
    assert a == b == c
    assert sel.last_gen_tokens == 0  # pool 不调 LLM


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        REGISTRY.create("agent_selector", "agentinit", mode="bogus").select(_q())


def test_generate_mode_wired_but_needs_llm():
    # M2 起 generate 已实现；无 env/注入时应明确报 RuntimeError（而非 NotImplementedError）。
    import os
    if os.getenv("LYCHEE_LLM_API_KEY"):
        pytest.skip("LLM env configured; skip negative test")
    with pytest.raises(RuntimeError):
        REGISTRY.create("agent_selector", "agentinit", mode="generate").select(_q())


# ---- Pareto 核心正确性（直接测 _pareto，构造已知支配关系）----
def test_fast_non_dominated_sort_front():
    # 最小化两目标：A=(0,0) 支配所有；B=(0,1),C=(1,0) 互不支配；D=(1,1) 被支配。
    objs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
    fronts = _pareto.fast_non_dominated_sort(objs)
    assert fronts[0] == [0]          # 全局最优独占第一前沿
    assert set(fronts[1]) == {1, 2}  # 次前沿 = 互不支配的 B,C
    assert fronts[2] == [3]


def test_diversity_penalises_duplicates():
    # 两个相同向量的 Vendi 多样性 < 两个不同向量的 —— 多样性目标能区分冗余团队。
    same = _pareto.cosine_matrix([[1.0, 0.0], [1.0, 0.0]])
    diff = _pareto.cosine_matrix([[1.0, 0.0], [0.0, 1.0]])
    div_same = _pareto.objective_diversity([0, 1], same)
    div_diff = _pareto.objective_diversity([0, 1], diff)
    assert div_diff < div_same  # 取负号后，更高多样性 = 更小（更优）
