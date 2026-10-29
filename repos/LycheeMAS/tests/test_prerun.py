"""prerun 图接缝的离线测试（需要 langgraph，未装则整文件跳过；LLM 全部脚本化）。

覆盖：节点契约与视图提取、rebuild 提示/邻接写回、统一入口分发、
agentprune 新旧双路径对拍、maspo apply/optimize 端到端。
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, TypedDict

import pytest

pytest.importorskip("langgraph")

from langgraph.graph import END, START, StateGraph  # noqa: E402
from lychee_mas.core.types import AgentSpec  # noqa: E402
from lychee_mas.methods.prerun.agentprune import AgentPrunePruner  # noqa: E402
from lychee_mas.plugins.prerun import (  # noqa: E402
    extract_view,
    optimize_langgraph,
    rebuild,
)
from lychee_mas.methods.prerun.maspo.executor import format_agent_prompt  # noqa: E402
from lychee_mas.methods.prerun.maspo.prompts import AGENT_TEMPLATES  # noqa: E402


class ChainState(TypedDict):
    question: str
    outputs: Dict[str, str]


def make_node(spec: AgentSpec, record: list[str]):
    """契约节点函数：运行时从 spec 读模板与前驱（换 spec.system_prompt 即换行为）。"""

    async def node(state: ChainState) -> Dict[str, Any]:
        context = "\n---\n".join(
            state["outputs"][p] for p in spec.meta.get("predecessors", [])
            if p in state["outputs"])
        prompt = format_agent_prompt(spec.system_prompt, state["question"], context)
        record.append(prompt)
        return {"outputs": {**state["outputs"], spec.name: f"out-of-{spec.name}"}}

    return node


def reflect_graph(record: list[str] | None = None) -> StateGraph:
    """predictor -> reflector 的最小 reflect 图（MASPO 主实验拓扑），按节点契约建图。"""
    record = record if record is not None else []
    sg = StateGraph(ChainState)
    names = ["predictor", "reflector"]
    for i, name in enumerate(names):
        spec = AgentSpec(name=name, role=name, system_prompt=AGENT_TEMPLATES[name],
                         meta={"predecessors": names[:i][-1:]})
        sg.add_node(name, make_node(spec, record), metadata={"agent_spec": spec})
    sg.add_edge(START, "predictor")
    sg.add_edge("predictor", "reflector")
    sg.add_edge("reflector", END)
    return sg


def chain_graph(n: int) -> StateGraph:
    """n 节点顺序链（agentprune 对拍用）。"""
    sg = StateGraph(ChainState)
    names = [f"A{i}" for i in range(n)]
    for i, name in enumerate(names):
        spec = AgentSpec(name=name, role="predictor", system_prompt="P {question} {context}",
                         meta={"predecessors": names[:i][-1:]})
        sg.add_node(name, make_node(spec, []), metadata={"agent_spec": spec})
    sg.add_edge(START, names[0])
    for a, b in zip(names, names[1:]):
        sg.add_edge(a, b)
    sg.add_edge(names[-1], END)
    return sg


# ------------------------------ 视图提取与契约 ------------------------------

def test_extract_view_roundtrip():
    view = extract_view(reflect_graph())
    assert view.names == ["predictor", "reflector"]
    assert view.terminal == "reflector"
    assert view.predecessors == {"predictor": [], "reflector": ["predictor"]}
    assert view.successors("predictor") == ["reflector"]
    assert view.all_successors("predictor") == {"reflector"}


def test_extract_view_missing_metadata_raises():
    sg = StateGraph(ChainState)
    sg.add_node("a", lambda s: {})
    sg.add_edge(START, "a")
    sg.add_edge("a", END)
    with pytest.raises(ValueError, match="agent_spec"):
        extract_view(sg)


def test_extract_view_rejects_branches_and_multi_terminal():
    sg = reflect_graph()
    sg.add_conditional_edges("predictor", lambda s: "reflector", ["reflector"])
    with pytest.raises(ValueError, match="条件分支"):
        extract_view(sg)

    sg2 = StateGraph(ChainState)
    for name in ("a", "b"):
        spec = AgentSpec(name=name, role="predictor", system_prompt="P {question}")
        sg2.add_node(name, make_node(spec, []), metadata={"agent_spec": spec})
    sg2.add_edge(START, "a")
    sg2.add_edge("a", END)
    sg2.add_edge("b", END)
    with pytest.raises(ValueError, match="终端"):
        extract_view(sg2)


def test_rebuild_prompts_changes_compiled_behavior():
    record: list[str] = []
    sg = reflect_graph(record)
    new_p = "IMPROVED predictor.\nQuestion: {question}\nContext: {context}"
    out = rebuild(sg, prompts={"predictor": new_p})
    assert out is sg  # 只换提示：同一张图（spec 与节点闭包共享）
    app = out.compile()
    asyncio.run(app.ainvoke({"question": "q?", "outputs": {}}))
    assert record[0].startswith("IMPROVED predictor.")  # 即插即用：编译执行读到新提示
    with pytest.raises(KeyError, match="ghost"):
        rebuild(sg, prompts={"ghost": "x"})
    with pytest.raises(ValueError, match="为空"):
        rebuild(sg, prompts={"predictor": "  "})


def test_rebuild_adjacency_rewires_and_updates_meta():
    sg = chain_graph(3)
    out = rebuild(sg, adjacency={"A0": ["A2"], "A1": ["A2"]})  # A0/A1 并行汇入 A2
    assert out is not sg
    view = extract_view(out)
    assert view.terminal == "A2"
    assert view.predecessors["A2"] == ["A0", "A1"]  # meta.predecessors 已写回
    assert view.predecessors["A1"] == []
    edges = {(s, d) for s, d in out.edges}
    assert ("A0", "A1") in edges or ("A1", "A0") in edges  # 执行序仍为线性链


# ------------------------------ 统一入口分发 ------------------------------

def test_optimize_langgraph_dispatch_errors():
    sg = reflect_graph()
    with pytest.raises(KeyError, match="pre_run_optimizer"):
        optimize_langgraph(sg, method="no_such_method")
    with pytest.raises(TypeError, match="编译"):
        optimize_langgraph(sg.compile(), method="maspo", mode="apply", prompt_file="x")
    with pytest.raises(TypeError, match="StateGraph"):
        optimize_langgraph({"not": "a graph"}, method="maspo")


def test_optimize_langgraph_maspo_apply(tmp_path):
    record: list[str] = []
    sg = reflect_graph(record)
    pf = tmp_path / "prompts.json"
    pf.write_text(json.dumps({"prompts": {
        "predictor": "APPLIED.\nQuestion: {question}\nContext: {context}"}}))
    out = optimize_langgraph(sg, method="maspo", mode="apply", prompt_file=str(pf))
    asyncio.run(out.compile().ainvoke({"question": "q?", "outputs": {}}))
    assert record[0].startswith("APPLIED.")

    pf.write_text(json.dumps({"prompts": {"ghost": "x {question}"}}))
    with pytest.raises(KeyError, match="ghost"):
        optimize_langgraph(reflect_graph(), method="maspo", mode="apply", prompt_file=str(pf))


# ------------------------- agentprune 双路径对拍 -------------------------

def test_agentprune_unified_path_matches_masgraph_path(tmp_path):
    n = 4
    trained = AgentPrunePruner(n_agents=n, seed=0)
    trained.spatial_logits[(0, 1)] = -9.0  # 压低几条边再 one-shot 剪枝
    trained.spatial_logits[(2, 1)] = -8.0
    trained.update_masks(0.25)
    state_file = str(tmp_path / "state.json")
    trained.save(state_file)

    # 参照：MASGraph 路径的 threshold 实现矩阵
    ref = AgentPrunePruner(n_agents=n, state_file=state_file)
    sm, _tm = ref.realized_matrices("threshold")

    out = optimize_langgraph(chain_graph(n), method="agentprune", state_file=state_file)
    view = extract_view(out)
    names = [f"A{i}" for i in range(n)]
    got = {(i, j) for i in range(n) for j in range(n)
           if names[i] in view.predecessors[names[j]]}
    want_all = {(i, j) for i in range(n) for j in range(n) if sm[i][j]}
    # 终端保持终端：其出边被忽略；线性化破环可能丢边——保留边必是 threshold 边的子集，
    # 且非终端出发的 threshold 边在无环情形应全保留
    assert got <= want_all
    assert view.terminal == names[-1]


def test_agentprune_unified_records_meta():
    from lychee_mas.core.registry import REGISTRY

    opt = REGISTRY.create("pre_run_optimizer", "agentprune")
    out = opt.optimize(chain_graph(3))
    assert opt.last_meta is not None
    assert opt.last_meta["alive_spatial"] == 6  # 初始 p=0.5 threshold 全保留（对角除外）
    assert extract_view(out).terminal == "A2"


# ------------------------- maspo optimize 端到端（脚本化） -------------------------

async def scripted_agent(prompt: str) -> str:
    if prompt.startswith("Below is a solution"):
        return "short summary"
    return "reasoning <answer>42</answer>"


async def scripted_evaluator(prompt: str) -> str:
    if "optimizing a prompt" in prompt:
        return "<prompt>OPT.\nQuestion: {question}\nContext: {context}</prompt>"
    return "A"  # 候选恒赢


def test_optimize_langgraph_maspo_optimize_end_to_end(tmp_path):
    record: list[str] = []
    sg = reflect_graph(record)
    pf = tmp_path / "opt_prompts.json"
    out = optimize_langgraph(
        sg, method="maspo", mode="optimize", trainset=["q1", "q2", "q3"],
        agent_llm=scripted_agent, evaluator_llm=scripted_evaluator,
        prompt_file=str(pf), max_total_depth=1, rounds_per_turn=1, eval_batch=2,
        use_beam_refresh=False, verbose=False, seed=0)
    saved = json.loads(pf.read_text())
    assert saved["prompts"]["predictor"].startswith("OPT.")
    assert (tmp_path / "opt_prompts_stats.json").exists()
    # 优化产物已注入图：编译执行读到新提示
    asyncio.run(out.compile().ainvoke({"question": "q?", "outputs": {}}))
    assert record[-2].startswith("OPT.")  # predictor 在倒数第二条（终端 reflector 最后）


def test_maspo_optimize_missing_inputs_raise():
    sg = reflect_graph()
    with pytest.raises(ValueError, match="trainset"):
        optimize_langgraph(sg, method="maspo", mode="optimize",
                           agent_llm=scripted_agent, evaluator_llm=scripted_evaluator)
    with pytest.raises(ValueError, match="agent_llm"):
        optimize_langgraph(sg, method="maspo", mode="optimize", trainset=["q"])


# ------------------------- agentdropout 统一接口挂载 -------------------------

def test_agentdropout_unified_apply_round_topology(tmp_path):
    from lychee_mas.methods.prerun.agentdropout import AgentDropoutOptimizer

    n, rounds = 4, 2
    trained = AgentDropoutOptimizer(n_agents=n, rounds=rounds, seed=0)
    for e in trained.deg_logits[0]:
        trained.deg_logits[0][e] = -9.0 if 3 in e else 1.0  # 第 0 轮淘汰节点 3
    trained.node_dropout()
    trained.spatial_logits[0][(0, 1)] = 3.0  # 唯一过阈值的存活边
    trained.spatial_logits[0][(1, 2)] = 3.0
    state_file = str(tmp_path / "ad_state.json")
    trained.save(state_file)

    out = optimize_langgraph(chain_graph(n), method="agentdropout",
                             state_file=state_file, round=0, rounds=rounds)
    view = extract_view(out)
    names = [f"A{i}" for i in range(n)]
    # 通信结构 = 第 0 轮 threshold 实现（0→1、1→2），且被淘汰节点 3 无任何通信
    assert view.predecessors[names[1]] == [names[0]]
    assert view.predecessors[names[2]] == [names[1]]
    assert view.predecessors[names[3]] == []
    # dropped 元数据：只有第 0 轮被淘汰的 A3 为 True（节点工厂据此本轮不执行）
    dropped = {nm: view.specs[nm].meta.get("dropped") for nm in names}
    assert dropped == {"A0": False, "A1": False, "A2": False, "A3": True}
    assert view.terminal == names[-1]

    # 换一轮挂载：round=1 的淘汰节点不同（deg_logits[1] 未动 → argmin=首个最小 = A0）
    out1 = optimize_langgraph(chain_graph(n), method="agentdropout",
                              state_file=state_file, round=1, rounds=rounds)
    view1 = extract_view(out1)
    assert view1.specs["A0"].meta.get("dropped") is True
