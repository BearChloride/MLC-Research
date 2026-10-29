"""graph_pruner/agentprune（AgentPrune 时空掩码剪枝）的离线测试：纯标准库，零重依赖。"""
from __future__ import annotations

import random

import pytest
from lychee_mas.core.registry import REGISTRY
from lychee_mas.core.types import AgentSpec, TaskQuery
from lychee_mas.methods.prerun.agentprune import (
    AgentPrunePruner,
    full_connected_masks,
    topological_order,
)


def make_pruner(**kw) -> AgentPrunePruner:
    return AgentPrunePruner(n_agents=4, seed=0, **kw)





def test_full_connected_masks():
    sm, tm = full_connected_masks(3)
    assert [row[i] for i, row in enumerate(sm)] == [0, 0, 0]  # 空间对角线 0
    assert sum(sum(r) for r in sm) == 6 and sum(sum(r) for r in tm) == 9


def test_initial_logits_and_sampling_respect_masks():
    p = make_pruner()
    assert p.spatial_logits[(0, 1)] == pytest.approx(0.0)  # p=0.5 -> logit 0
    real = p.sample_realization(random.Random(1))
    assert all(i != j for (i, j) in real.spatial_edges)  # 对角线永不采
    assert set(real.spatial_samples) == {(i, j) for i in range(4) for j in range(4) if i != j}
    # include_temporal=False：时间边不采样、不计 log_prob
    real0 = p.sample_realization(random.Random(1), include_temporal=False)
    assert real0.temporal_samples == {} and real0.temporal_edges == set()


def test_reinforce_moves_logits_toward_utility():
    p = make_pruner()
    real = p.sample_realization(random.Random(2))
    on = next(iter(real.spatial_edges))
    off = next(e for e, (b, _) in real.spatial_samples.items() if b == 0)
    p.reinforce([([real], 1.0)])
    assert p.spatial_logits[on] > 0.0   # 被采中且做对 -> 提升
    assert p.spatial_logits[off] < 0.0  # 未采中且做对 -> 压低
    # utility=0：零梯度，logit 不动
    p2 = make_pruner()
    r2 = p2.sample_realization(random.Random(2))
    p2.reinforce([([r2], 0.0)])
    assert all(v == pytest.approx(0.0) for v in p2.spatial_logits.values())


def test_update_masks_prunes_lowest_logits():
    p = make_pruner()
    p.spatial_logits[(0, 1)] = -5.0
    p.spatial_logits[(1, 0)] = -4.0
    p.spatial_logits[(2, 3)] = -3.0
    ks, kt = p.update_masks(0.25)  # 12 条存活空间边 × 0.25 = 3 条
    assert ks == 3
    assert p.spatial_masks[(0, 1)] == p.spatial_masks[(1, 0)] == p.spatial_masks[(2, 3)] == 0
    assert kt == round(16 * 0.25)  # 时间边 16 条全存活
    # 再剪一次按剩余存活数计
    ks2, _ = p.update_masks(0.25)
    assert ks2 == round(9 * 0.25)


def test_topological_order_dag_and_cycle_break():
    order, preds = topological_order(3, {(0, 1), (1, 2)})
    assert order == [0, 1, 2] and preds[1] == {0} and preds[2] == {1}
    # 环 0->1->2->0：破环后仍全执行，且消息只向后流
    order2, preds2 = topological_order(3, {(0, 1), (1, 2), (2, 0)})
    assert sorted(order2) == [0, 1, 2]
    pos = {n: k for k, n in enumerate(order2)}
    for dst, ps in preds2.items():
        assert all(pos[src] < pos[dst] for src in ps)
    assert preds2[order2[0]] == set()  # 首节点无前驱（破环边被丢弃）


def test_threshold_realization_keeps_all_at_init():
    p = make_pruner()
    sm, tm = p.realized_matrices("threshold")
    assert sum(v for row in sm for v in row) == 12  # 初始 p=0.5，threshold(>=0.5) 全保留
    assert sum(v for row in tm for v in row) == 16
    with pytest.raises(ValueError):
        p.realized_matrices("bogus")  # 未知模式显式报错
    # 图级挂载（optimize_langgraph(method="agentprune")）的对拍见 tests/test_prerun.py

def test_state_roundtrip(tmp_path):
    p = make_pruner()
    p.spatial_logits[(0, 1)] = -9.0
    p.update_masks(0.25)
    path = str(tmp_path / "state.json")
    p.save(path)
    q = AgentPrunePruner(n_agents=4, state_file=path)
    assert q.spatial_logits[(0, 1)] == -9.0
    assert q.spatial_masks == p.spatial_masks and q.temporal_masks == p.temporal_masks
    with pytest.raises(ValueError):
        AgentPrunePruner(n_agents=3, state_file=path)  # n 不符显式报错


def test_registered():
    assert "agentprune" in REGISTRY.list("graph_pruner")
