"""GEPA（plugins/gepa）的离线测试：Pareto 选择纯函数 + 脚本化 rollout/reflector 的 compile 收敛。"""
from __future__ import annotations

import random

import pytest
from lychee_mas.core.registry import REGISTRY
from lychee_mas.core.types import AgentSpec, Answer, TaskQuery, Trajectory
from lychee_mas.methods.prerun.gepa.pareto import (
    dominated,
    pareto_pool,
    per_instance_best,
    sample_candidate,
)
from lychee_mas.methods.prerun.gepa.program import MASProgram

# ---------------- pareto 纯函数 ----------------

SCORES = {
    "a": [1.0, 0.0, 1.0],
    "b": [0.0, 1.0, 1.0],
    "c": [0.0, 0.0, 1.0],   # 被 a、b 支配
}


def test_dominated():
    assert dominated(SCORES["c"], SCORES["a"])
    assert not dominated(SCORES["a"], SCORES["b"])  # 互不支配


def test_pareto_pool_removes_dominated():
    assert pareto_pool(SCORES) == {"a", "b"}


def test_per_instance_best():
    fronts = per_instance_best(SCORES)
    assert fronts == [{"a"}, {"b"}, {"a", "b", "c"}]


def test_sample_candidate_deterministic_and_within_pool():
    rng = random.Random(0)
    picks = {sample_candidate(SCORES, rng) for _ in range(20)}
    assert picks <= {"a", "b"}  # 被支配的 c 永不入选
    assert len(picks) == 2  # 两个前沿候选都有机会被采到


def test_pareto_input_validation():
    with pytest.raises(ValueError):
        pareto_pool({})
    with pytest.raises(ValueError):
        pareto_pool({"a": [1.0], "b": [1.0, 0.0]})


# ---------------- MASProgram ----------------

def make_agents():
    return [
        AgentSpec(name="planner", role="planner", system_prompt="plan it"),
        AgentSpec(name="solver", role="solver", system_prompt="solve it"),
    ]


def test_program_roundtrip_and_mutation():
    agents = make_agents()
    prog = MASProgram.from_agents(agents)
    assert prog.components["agent:planner:system_prompt"] == "plan it"
    assert prog.components["topology:description"] == "planner -> solver"
    assert prog.mutable_keys == ("agent:planner:system_prompt", "agent:solver:system_prompt")

    child = prog.mutated("agent:solver:system_prompt", "solve harder")
    assert child.components["agent:solver:system_prompt"] == "solve harder"
    assert prog.components["agent:solver:system_prompt"] == "solve it"  # 不原地改

    new_agents = child.apply_to(agents)
    assert new_agents[1].system_prompt == "solve harder"
    assert agents[1].system_prompt == "solve it"  # 原列表不动

    with pytest.raises(KeyError):
        prog.mutated("topology:description", "x")  # 冻结组件不可变异


def test_program_apply_to_unknown_agent_raises():
    prog = MASProgram(components={"agent:ghost:system_prompt": "boo"},
                      mutable_keys=("agent:ghost:system_prompt",))
    with pytest.raises(KeyError):
        prog.apply_to(make_agents())


# ---------------- GEPA compile（脚本化 rollout/reflector） ----------------

def scripted_rollout(program: MASProgram, query: TaskQuery) -> Trajectory:
    """轨迹的 final answer = 两个组件文本拼接（让 metric 能看见组件内容）。"""
    text = " | ".join(program.components[k] for k in program.mutable_keys)
    traj = Trajectory(task_id=query.id)
    traj.final_answer = Answer(content=text)
    return traj


def keyword_metric(traj: Trajectory, query: TaskQuery) -> float:
    return 1.0 if "MAGIC" in (traj.final_answer.content if traj.final_answer else "") else 0.0


def appending_reflector(component_text: str, feedback: list) -> str:
    return component_text + " MAGIC"


def test_gepa_optimizes_to_keyword():
    prog = MASProgram.from_agents(make_agents())
    opt = REGISTRY.create("optimizer", "gepa", rollout=scripted_rollout,
                          reflector=appending_reflector,
                          minibatch_size=2, max_metric_calls=40, seed=0)
    trainset = [TaskQuery(question=f"q{i}") for i in range(3)]
    best = opt.optimize(prog, trainset, keyword_metric)
    assert any("MAGIC" in best.components[k] for k in best.mutable_keys)
    assert best.meta["gepa"]["mean_score"] == 1.0
    assert best.meta["gepa"]["metric_calls"] <= 40  # 硬预算


def test_gepa_budget_is_hard_ceiling():
    prog = MASProgram.from_agents(make_agents())
    calls = {"n": 0}

    def counting_rollout(program, query):
        calls["n"] += 1
        return scripted_rollout(program, query)

    opt = REGISTRY.create("optimizer", "gepa", rollout=counting_rollout,
                          reflector=appending_reflector,
                          minibatch_size=2, max_metric_calls=10, seed=1)
    trainset = [TaskQuery(question=f"q{i}") for i in range(4)]
    best = opt.optimize(prog, trainset, keyword_metric)
    assert calls["n"] <= 10
    assert best is not None  # 预算耗尽仍返回当前最优


def test_gepa_explicit_errors():
    prog = MASProgram.from_agents(make_agents())
    trainset = [TaskQuery(question="q")]
    with pytest.raises(ValueError, match="rollout"):
        REGISTRY.create("optimizer", "gepa").optimize(prog, trainset, keyword_metric)
    # 无 reflector 且无 backend：反思阶段显式报错（种子全量评估后第一轮变异触发）
    opt = REGISTRY.create("optimizer", "gepa", rollout=scripted_rollout,
                          minibatch_size=1, max_metric_calls=50, seed=0)
    with pytest.raises(ValueError, match="reflector"):
        opt.optimize(prog, trainset, keyword_metric)
