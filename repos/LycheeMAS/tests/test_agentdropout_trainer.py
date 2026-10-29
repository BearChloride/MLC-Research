"""AgentDropout 两阶段训练日程（methods/prerun/agentdropout/trainer.py）离线测试。

LLM/执行全为脚本化注入（rollout 不跑真图，只按调用序记账），因此可离线断言：

- **日程组成**：阶段一 2 批 × 20 题、阶段二 4 批 × 10 题（同池，共需 40 题）；
- **调用次序 = RNG 消费序**（复现口径，见 trainer.py 模块 docstring）：每题先把全部轮次的
  ``sample_skip`` / ``sample_round`` 采完，再整题 rollout；批末 reinforce；
  ``node_dropout`` 阶段一结束后恰一次；``edge_dropout`` 仅在 batch idx {1,3} 后各一次；
- **train_log 契约**：条目键集合与键序、batch 编号、记账字段；
- **落盘**：state / train_log 写出，train_log 默认路径由 state 名推导；
- **确定性**：同 seed 两次 run 的 state 逐位相等；不同 seed 采样序列不同（RNG 真被消费）；
- **显式报错**：非法超参 / 题数不足 / 非 TaskQuery / 空 question / 路径冲突。
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, List, Optional, Tuple

import pytest
from lychee_mas.core.types import TaskQuery
from lychee_mas.methods.prerun.agentdropout import (
    AgentDropoutOptimizer,
    AgentDropoutTrainer,
    RolloutStats,
)

N_AGENTS, ROUNDS = 3, 2
LOG_KEYS_1 = ["phase", "batch", "skip", "utility", "pred", "gold",
              "prompt_tokens", "completion_tokens", "model_calls"]
LOG_KEYS_2 = ["phase", "batch", "alive_edges", "utility", "pred", "gold",
              "prompt_tokens", "completion_tokens", "model_calls"]


def queries(n: int) -> List[TaskQuery]:
    return [TaskQuery(question=f"q{i}", gold=str(i)) for i in range(n)]


class SpyOptimizer(AgentDropoutOptimizer):
    """真优化器 + 调用记账：timeline 与 rollout 共用一个列表，可断言交错次序。"""

    def __init__(self, **kw: Any) -> None:
        super().__init__(**kw)
        self.timeline: List[Tuple[Any, ...]] = []

    def sample_skip(self, r: int, edges: Optional[set] = None):
        out = super().sample_skip(r, edges)
        self.timeline.append(("sample_skip", r, out[0]))
        return out

    def skip_reinforce(self, batch):
        self.timeline.append(("skip_reinforce", len(batch)))
        super().skip_reinforce(batch)

    def node_dropout(self):
        out = super().node_dropout()
        self.timeline.append(("node_dropout", dict(out)))
        return out

    def sample_round(self, r: int):
        out = super().sample_round(r)
        self.timeline.append(("sample_round", r))
        return out

    def edge_reinforce(self, batch):
        self.timeline.append(("edge_reinforce", len(batch)))
        super().edge_reinforce(batch)

    def edge_dropout(self, rate: float):
        out = super().edge_dropout(rate)
        self.timeline.append(("edge_dropout", dict(out)))
        return out


class ScriptedRollout:
    """脚本化 rollout：记账后返回 (final, stats)；final="<i>" 对 gold="<i>" 命中。"""

    def __init__(self, spy: SpyOptimizer) -> None:
        self.spy = spy
        self.calls: List[Tuple[str, list]] = []

    async def __call__(self, question: str, plans: list):
        self.spy.timeline.append(("rollout", question))
        self.calls.append((question, plans))
        return question[1:], RolloutStats(prompt_tokens=11, completion_tokens=7,
                                          model_calls=5)


def reward(final: str, gold: str) -> float:
    return 1.0 if final == gold else 0.0


def predict(final: str) -> str:
    return f"<{final}>"


def train_once(opt: SpyOptimizer, trainset: List[TaskQuery], state_file: str, **kw: Any):
    trainer = AgentDropoutTrainer(opt, verbose=False, **kw)
    return asyncio.run(trainer.run(trainset, ScriptedRollout(opt), reward, predict,
                                   state_file=state_file))


def expected_kinds(p1_batches: int, p1_size: int, p2_batches: int, p2_size: int,
                   prune_idx) -> List[str]:
    """日程的期望调用序（只记种类）：采样先于 rollout，reinforce 在批末，dropout 按时机。"""
    kinds: List[str] = []
    for _ in range(p1_batches):
        for _ in range(p1_size):
            kinds += ["sample_skip"] * ROUNDS + ["rollout"]
        kinds.append("skip_reinforce")
    kinds.append("node_dropout")
    for b in range(p2_batches):
        for _ in range(p2_size):
            kinds += ["sample_round"] * ROUNDS + ["rollout"]
        kinds.append("edge_reinforce")
        if b in prune_idx:
            kinds.append("edge_dropout")
    return kinds


# ------------------------------- 日程与调用序 -------------------------------

def test_full_schedule_call_order(tmp_path):
    """默认日程（2×20 + 4×10，40 题同池）：调用序逐项等于转写口径。"""
    opt = SpyOptimizer(n_agents=N_AGENTS, rounds=ROUNDS, seed=0)
    report = train_once(opt, queries(40), str(tmp_path / "s_state.json"))
    roll = ScriptedRollout(opt)  # 只为类型清晰；实际 rollout 由 train_once 内部建

    assert [e[0] for e in opt.timeline] == expected_kinds(2, 20, 4, 10, (1, 3))
    # 阶段一 40 + 阶段二 40 次 rollout，每次含 rounds 轮计划
    n_rollouts = sum(1 for e in opt.timeline if e[0] == "rollout")
    assert n_rollouts == 80
    # node_dropout 恰一次、edge_dropout 恰两次（batch idx 1/3 后）
    assert sum(1 for e in opt.timeline if e[0] == "node_dropout") == 1
    prune = [e for e in opt.timeline if e[0] == "edge_dropout"]
    assert len(prune) == 2
    assert [p["batch"] for p in report.prune_events] == [1, 3]
    # 采样先于 rollout：每题的第一条 rollout 之前已有 ROUNDS 条采样
    idx = [i for i, e in enumerate(opt.timeline) if e[0] == "rollout"]
    assert idx[0] == ROUNDS
    assert report.skip_nodes  # node_dropout 落下的逐轮淘汰节点
    assert set(report.skip_nodes) == set(range(ROUNDS))
    del roll


def test_small_schedule_honors_hyperparams(tmp_path):
    """超参可换：1×2 + 2×1，prune idx {0} → 调用序与需求量随之变。"""
    opt = SpyOptimizer(n_agents=N_AGENTS, rounds=ROUNDS, seed=0)
    kw = dict(phase1_batches=1, phase1_batch_size=2, phase2_batches=2, phase2_batch_size=1,
              prune_batch_idx=(0,), pruning_rate=0.5)
    trainer = AgentDropoutTrainer(opt, verbose=False, **kw)
    assert trainer.needed_queries == 2
    report = asyncio.run(trainer.run(queries(2), ScriptedRollout(opt), reward, predict,
                                     state_file=str(tmp_path / "s_state.json")))
    assert [e[0] for e in opt.timeline] == expected_kinds(1, 2, 2, 1, (0,))
    assert [p["batch"] for p in report.prune_events] == [0]
    assert len(report.log) == 4


# -------------------------------- 落盘契约 --------------------------------

def test_train_log_schema_and_batch_numbering(tmp_path):
    opt = SpyOptimizer(n_agents=N_AGENTS, rounds=ROUNDS, seed=0)
    state_file = str(tmp_path / "ad_state.json")
    report = train_once(opt, queries(40), state_file)

    assert len(report.log) == 80
    keys = [list(e) for e in report.log]
    assert keys[:40] == [LOG_KEYS_1] * 40
    assert keys[40:] == [LOG_KEYS_2] * 40
    assert [e["batch"] for e in report.log[:40]] == [0] * 20 + [1] * 20
    assert [e["batch"] for e in report.log[40:]] == [0] * 10 + [1] * 10 + [2] * 10 + [3] * 10
    for e in report.log:
        assert e["utility"] in (0.0, 1.0)
        assert e["pred"] == f"<{e['gold']}>"          # predict 收 final、写进 pred 字段
        assert (e["prompt_tokens"], e["completion_tokens"], e["model_calls"]) == (11, 7, 5)
    assert set(report.log[0]["skip"]) == set(range(ROUNDS))   # 轮次 → 被跳节点下标
    # 阶段二记活着空间边数：轮次 → 计数
    assert set(report.log[40]["alive_edges"]) == set(range(ROUNDS))
    assert all(isinstance(v, int) for v in report.log[40]["alive_edges"].values())


def test_state_and_train_log_paths(tmp_path):
    opt = SpyOptimizer(n_agents=N_AGENTS, rounds=ROUNDS, seed=0)
    state_file = str(tmp_path / "ad_state.json")
    report = train_once(opt, queries(40), state_file)

    assert report.state_path == state_file
    assert report.train_log_path == str(tmp_path / "ad_train_log.json")  # 默认推导
    state = json.loads(open(state_file, encoding="utf-8").read())
    assert state["n_agents"] == N_AGENTS and state["rounds"] == ROUNDS
    assert state["skip_nodes"] == {str(r): v for r, v in report.skip_nodes.items()}
    # 文件内容 = 内存 log 的 JSON 往返（轮次键 0/1 落盘后成 "0"/"1"，同原脚本口径）
    assert json.loads(open(report.train_log_path, encoding="utf-8").read()) \
        == json.loads(json.dumps(report.log))

    # 显式 train_log_file 生效，且目录按需创建
    other = tmp_path / "nested" / "log.json"
    r2 = asyncio.run(AgentDropoutTrainer(
        SpyOptimizer(n_agents=N_AGENTS, rounds=ROUNDS, seed=0),
        phase1_batches=1, phase1_batch_size=2, phase2_batches=1, phase2_batch_size=2,
        prune_batch_idx=(), verbose=False).run(
            queries(2), ScriptedRollout(SpyOptimizer(n_agents=N_AGENTS, rounds=ROUNDS)),
            reward, predict, state_file=str(tmp_path / "s2_state.json"),
            train_log_file=str(other)))
    assert r2.train_log_path == str(other) and other.exists()


# --------------------------------- 确定性 ---------------------------------

def test_same_seed_is_deterministic(tmp_path):
    def once(tag: str):
        opt = SpyOptimizer(n_agents=N_AGENTS, rounds=ROUNDS, seed=7)
        before = json.dumps(opt.state_dict(), sort_keys=True)
        report = train_once(opt, queries(40), str(tmp_path / f"{tag}_state.json"))
        return before, json.dumps(opt.state_dict(), sort_keys=True), json.dumps(report.log)

    before, s1, l1 = once("a")
    _, s2, l2 = once("b")
    assert s1 == s2 and l1 == l2          # 同 seed → 逐位相同
    assert s1 != before                   # 训练确实改动了参数（非空转）


def test_different_seed_changes_samples(tmp_path):
    """不同 seed → 采样序列不同（防「RNG 压根没被消费」的假绿）。"""
    def skips(seed: int):
        opt = SpyOptimizer(n_agents=N_AGENTS, rounds=ROUNDS, seed=seed)
        report = train_once(opt, queries(40), str(tmp_path / f"s{seed}_state.json"))
        return [json.dumps(e["skip"]) for e in report.log[:40]]

    assert skips(0) != skips(1)


# ------------------------------- 显式报错路径 -------------------------------

def test_trainer_ctor_rejects_bad_schedule():
    opt = AgentDropoutOptimizer(n_agents=N_AGENTS, rounds=ROUNDS, seed=0)
    with pytest.raises(ValueError, match="批次数"):
        AgentDropoutTrainer(opt, phase1_batches=0)
    with pytest.raises(ValueError, match="每批题数"):
        AgentDropoutTrainer(opt, phase2_batch_size=0)
    with pytest.raises(ValueError, match="pruning_rate"):
        AgentDropoutTrainer(opt, pruning_rate=1.5)
    with pytest.raises(ValueError, match="越界"):
        AgentDropoutTrainer(opt, phase2_batches=4, prune_batch_idx=(4,))


def test_run_rejects_bad_inputs(tmp_path):
    opt = SpyOptimizer(n_agents=N_AGENTS, rounds=ROUNDS, seed=0)
    roll = ScriptedRollout(opt)
    state_file = str(tmp_path / "s_state.json")

    def run(trainset, **kw):
        trainer = AgentDropoutTrainer(opt, verbose=False)
        return asyncio.run(trainer.run(trainset, roll, reward, predict,
                                       state_file=kw.pop("state_file", state_file), **kw))

    with pytest.raises(ValueError, match="训练需要"):
        run(queries(39))
    with pytest.raises(TypeError, match="TaskQuery"):
        run(queries(39) + ["not-a-query"])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="question"):
        run([TaskQuery(question="", gold="0")] * 40)
    with pytest.raises(ValueError, match="state_file"):
        run(queries(40), state_file=None)
    with pytest.raises(ValueError, match="同路径"):
        run(queries(40), train_log_file=state_file)


def test_run_rejects_malformed_rollout(tmp_path):
    """rollout 返回形状不符 → 显式报错（解包失败，不静默兜底）。"""
    opt = SpyOptimizer(n_agents=N_AGENTS, rounds=ROUNDS, seed=0)

    async def bad_rollout(question: str, plans: list):
        return "final", RolloutStats(), "extra"

    trainer = AgentDropoutTrainer(opt, phase1_batches=1, phase1_batch_size=1,
                                  phase2_batches=1, phase2_batch_size=1,
                                  prune_batch_idx=(), verbose=False)
    with pytest.raises((ValueError, TypeError)):
        asyncio.run(trainer.run(queries(1), bad_rollout, reward, predict,
                                state_file=str(tmp_path / "s_state.json")))
