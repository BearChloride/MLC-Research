"""`pre_run_optimizer/agentdropout` 的 optimize 模式（统一接口跑两阶段训练）离线测试。

覆盖：经 `optimize_langgraph(mode="optimize")` 与直建适配器两条路径跑训练 → 产物落盘、
`last_meta` 摘要、训练不消费图（返回原图）；`mode` 与训练素材的显式报错路径；
apply 模式行为不受本次改动影响（含 lr/seed 透传与 round 挂载）；实现类已下沉 methods
（与 maspo 同款）——规范路径 = methods，plugins 旧路径取到同一对象。
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, TypedDict

import pytest

pytest.importorskip("langgraph")

from langgraph.graph import END, START, StateGraph  # noqa: E402
from lychee_mas.core.registry import REGISTRY  # noqa: E402
from lychee_mas.core.types import AgentSpec, TaskQuery  # noqa: E402
from lychee_mas.methods.prerun.agentdropout import (  # noqa: E402
    AgentDropoutOptimizer,
    RolloutStats,
)
from lychee_mas.methods.prerun.agentdropout.optimizer import AgentDropoutLG  # noqa: E402
from lychee_mas.plugins.prerun import extract_view, optimize_langgraph  # noqa: E402


def test_import_paths_resolve_to_methods_class() -> None:
    """实现类在 methods：plugins 旧路径是同一对象的兼容 shim，注册也解析到它。"""
    from lychee_mas.plugins.prerun import agentdropout_lg as shim

    assert shim.AgentDropoutLG is AgentDropoutLG
    opt = REGISTRY.create("pre_run_optimizer", "agentdropout", mode="apply")
    assert type(opt) is AgentDropoutLG


class ChainState(TypedDict):
    outputs: Dict[str, str]


def chain_graph(n: int) -> StateGraph:
    """n 节点顺序链（契约节点：metadata["agent_spec"]），optimize 模式只用其节点数。"""
    sg = StateGraph(ChainState)
    names = [f"agent_{i}" for i in range(n)]
    for i, name in enumerate(names):
        spec = AgentSpec(name=name, role="predictor", system_prompt="P",
                         meta={"predecessors": names[:i][-1:], "dropped": False})
        sg.add_node(name, _node(spec), metadata={"agent_spec": spec})
    sg.add_edge(START, names[0])
    for a, b in zip(names, names[1:]):
        sg.add_edge(a, b)
    sg.add_edge(names[-1], END)
    return sg


def _node(spec: AgentSpec):
    async def node(state: ChainState) -> Dict[str, Any]:
        return {"outputs": {**state["outputs"], spec.name: "x"}}

    return node


def queries(n: int) -> List[TaskQuery]:
    return [TaskQuery(question=f"q{i}", gold="a") for i in range(n)]


async def rollout(question: str, plans: list):
    return "a", RolloutStats(prompt_tokens=3, completion_tokens=2, model_calls=1)


def reward(final: str, gold: str) -> float:
    return 1.0 if final == gold else 0.0


def predict(final: str) -> str:
    return final


TRAIN_KW = dict(phase1_batches=1, phase1_batch_size=2, phase2_batches=1, phase2_batch_size=1,
                prune_batch_idx=(0,), pruning_rate=0.5)


# ------------------------------ optimize 模式 ------------------------------

def test_optimize_via_unified_entry(tmp_path):
    """统一入口驱动训练：产物落盘、图原样返回（训练不消费图）。"""
    state_file = str(tmp_path / "ad_state.json")
    sg = chain_graph(4)
    out = optimize_langgraph(sg, method="agentdropout", mode="optimize",
                             state_file=state_file, trainset=queries(2),
                             rollout=rollout, reward=reward, predict=predict,
                             seed=0, rounds=2, **TRAIN_KW)
    assert out is sg                                     # 图进图出，未被改建
    state = json.loads(open(state_file, encoding="utf-8").read())
    assert state["n_agents"] == 4 and state["rounds"] == 2
    assert len(state["skip_nodes"]) == 2                  # node_dropout 逐轮淘汰
    log = json.loads(open(str(tmp_path / "ad_train_log.json"), encoding="utf-8").read())
    assert [e["phase"] for e in log] == ["node_dropout"] * 2 + ["edge_dropout"]


def test_optimize_last_meta_summary(tmp_path):
    state_file = str(tmp_path / "ad_state.json")
    lg = AgentDropoutLG(mode="optimize", state_file=state_file, trainset=queries(2),
                        rollout=rollout, reward=reward, predict=predict, seed=0, rounds=2,
                        **TRAIN_KW)
    lg.optimize(chain_graph(3))
    meta = lg.last_meta
    assert meta["mode"] == "optimize" and meta["n_agents"] == 3 and meta["rounds"] == 2
    assert meta["state_file"] == state_file
    assert meta["train_log_file"] == str(tmp_path / "ad_train_log.json")
    assert sorted(map(int, meta["skip_nodes"])) == [0, 1]
    assert [e["batch"] for e in meta["prune_events"]] == [0]
    assert meta["phase1_accuracy"] == pytest.approx(1.0)   # 脚本化 rollout 全对
    assert meta["phase2_accuracy"] == pytest.approx(1.0)


def test_optimize_train_log_file_override(tmp_path):
    state_file = str(tmp_path / "ad_state.json")
    log_file = str(tmp_path / "logs" / "other.json")
    lg = AgentDropoutLG(mode="optimize", state_file=state_file, train_log_file=log_file,
                        trainset=queries(2), rollout=rollout, reward=reward,
                        predict=predict, seed=0, rounds=2, **TRAIN_KW)
    lg.optimize(chain_graph(3))
    assert lg.last_meta["train_log_file"] == log_file
    assert (tmp_path / "logs" / "other.json").exists()


def test_optimize_does_not_load_state_as_input(tmp_path):
    """训练必须从零起：即便 state_file 已存在（含旧 skip_nodes），也不当输入加载。"""
    state_file = str(tmp_path / "ad_state.json")
    seed = AgentDropoutOptimizer(n_agents=3, rounds=2, seed=0)
    seed.deg_logits[0][(2, 0)] = -20.0
    seed.node_dropout()
    seed.save(state_file)                                  # 旧的 skip_nodes={0: 2,...}
    lg = AgentDropoutLG(mode="optimize", state_file=state_file, trainset=queries(2),
                        rollout=rollout, reward=reward, predict=predict, seed=0, rounds=2,
                        **TRAIN_KW)
    lg.optimize(chain_graph(3))
    # 若把产物当输入加载，skip_nodes 会被 seed 状态污染（跳的还是节点 2）
    assert lg.last_meta["skip_nodes"] != {} and lg.last_meta["skip_nodes"] != {"0": 2, "1": 2}


# -------------------------------- 显式报错 --------------------------------

def test_unknown_mode_raises():
    with pytest.raises(ValueError, match="未知 mode"):
        AgentDropoutLG(mode="train")


def test_optimize_rejects_round_and_missing_materials():
    with pytest.raises(ValueError, match="round"):
        AgentDropoutLG(mode="optimize", state_file="s.json", round=1, trainset=queries(2),
                       rollout=rollout, reward=reward, predict=predict)
    with pytest.raises(ValueError, match="state_file"):
        AgentDropoutLG(mode="optimize", trainset=queries(2), rollout=rollout,
                       reward=reward, predict=predict)
    with pytest.raises(ValueError, match="rollout/reward/predict"):
        AgentDropoutLG(mode="optimize", state_file="s.json", trainset=queries(2))
    with pytest.raises(ValueError, match="reward/predict"):
        AgentDropoutLG(mode="optimize", state_file="s.json", trainset=queries(2),
                       rollout=rollout)                    # 只缺一半就只报缺的那些
    with pytest.raises(ValueError, match="为空"):
        AgentDropoutLG(mode="optimize", state_file="s.json", trainset=[],
                       rollout=rollout, reward=reward, predict=predict)


def test_apply_rejects_training_materials():
    with pytest.raises(ValueError, match="不接受训练素材"):
        AgentDropoutLG(state_file="s.json", trainset=queries(2), rollout=rollout)
    with pytest.raises(ValueError, match="不接受训练素材"):
        AgentDropoutLG(state_file="s.json", reward=reward)


def test_optimize_rejects_short_trainset(tmp_path):
    lg = AgentDropoutLG(mode="optimize", state_file=str(tmp_path / "ad_state.json"),
                        trainset=queries(1), rollout=rollout, reward=reward,
                        predict=predict, seed=0, rounds=2, **TRAIN_KW)
    with pytest.raises(ValueError, match="训练需要"):
        lg.optimize(chain_graph(3))


# ------------------------------ apply 零回归 ------------------------------

def test_apply_mode_unchanged(tmp_path):
    """apply：threshold 实现挂载 + dropped 元数据 + last_meta 记 mode/round。"""
    trained = AgentDropoutOptimizer(n_agents=3, rounds=2, seed=0)
    for e in trained.deg_logits[0]:
        trained.deg_logits[0][e] = -9.0 if 2 in e else 1.0   # 第 0 轮淘汰节点 2
    trained.node_dropout()
    trained.spatial_logits[0][(0, 1)] = 3.0                  # 唯一过阈值的存活边
    state_file = str(tmp_path / "ad_state.json")
    trained.save(state_file)

    out = optimize_langgraph(chain_graph(3), method="agentdropout", state_file=state_file,
                             round=0, rounds=2)
    view = extract_view(out)
    names = ["agent_0", "agent_1", "agent_2"]
    assert view.predecessors[names[1]] == [names[0]]          # 0 → 1，1 无前驱
    assert view.predecessors[names[2]] == []
    assert view.specs[names[2]].meta["dropped"] is True       # 被淘汰节点本轮不执行
    assert view.specs[names[1]].meta["dropped"] is False

    lg = AgentDropoutLG(state_file=state_file, round=0, rounds=2, lr=0.25, seed=3)
    lg.optimize(chain_graph(3))
    assert lg.last_meta["mode"] == "apply" and lg.last_meta["round"] == 0
    assert lg.last_meta["dropped"] == names[2]
