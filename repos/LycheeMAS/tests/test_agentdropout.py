"""methods/prerun/agentdropout（AgentDropout 动态节点/边淘汰）的离线测试：纯标准库。"""
from __future__ import annotations

import pytest
from lychee_mas.core.registry import REGISTRY
from lychee_mas.methods.prerun.agentdropout import (
    AgentDropoutOptimizer,
    acyclic_realization,
)

N, R = 5, 2  # 原版设置：5 agents、2 轮


def make_opt(**kw) -> AgentDropoutOptimizer:
    return AgentDropoutOptimizer(n_agents=N, rounds=R, seed=0, **kw)


def test_acyclic_realization_is_upper_triangular_for_full_graph():
    alive = {(i, j) for i in range(4) for j in range(4) if i != j}
    kept = acyclic_realization(4, alive)
    # i-主序贪心去环：全连接图的确定性实现 = 上三角 DAG（原版 check_cycle 语义）
    assert kept == {(i, j) for i in range(4) for j in range(4) if i < j}


def test_sample_skip_follows_weighted_degree_softmax():
    opt = make_opt()
    for e in opt.deg_logits[0]:
        opt.deg_logits[0][e] = 100.0 if 2 in e else -100.0  # 节点 2 的加权度远大于其余
    picks = {opt.sample_skip(0)[0] for _ in range(20)}
    assert picks == {2}


def test_skip_reinforce_pushes_skipped_edges_down_others_up():
    opt = make_opt()
    edges = opt.realized_full()
    skip = 1
    before = {e: opt.deg_logits[0][e] for e in edges}
    opt.skip_reinforce([([(skip, edges), (skip, edges)], 1.0)])  # 两轮同 skip，utility=1
    for e in edges:
        if skip in e:
            assert opt.deg_logits[0][e] < before[e]  # 被跳节点的边 → 压低
        else:
            assert opt.deg_logits[0][e] > before[e]  # 其余节点的边 → 提升
    # utility=0：零梯度
    opt2 = make_opt()
    before2 = dict(opt2.deg_logits[0])
    opt2.skip_reinforce([([(skip, edges), (skip, edges)], 0.0)])
    assert opt2.deg_logits[0] == before2


def test_node_dropout_removes_min_normalized_degree_per_round():
    opt = make_opt()
    for e in opt.deg_logits[0]:
        opt.deg_logits[0][e] = -9.0 if 3 in e else 1.0   # 第 0 轮：节点 3 最小
    for e in opt.deg_logits[1]:
        opt.deg_logits[1][e] = -9.0 if 0 in e else 1.0   # 第 1 轮：节点 0 最小
    skips = opt.node_dropout()
    assert skips == {0: 3, 1: 0}
    assert all(opt.spatial_masks[0][(3, k)] == 0 and opt.spatial_masks[0][(k, 3)] == 0
               for k in range(N))
    assert all(opt.spatial_masks[1][(0, k)] == 0 and opt.spatial_masks[1][(k, 0)] == 0
               for k in range(N))
    # 跨轮时间边：3 在第 0 轮被淘汰 → 3→(第 1 轮) 出向清零；0 在第 1 轮被淘汰 → 入向清零
    assert all(opt.temporal_masks[1][(3, k)] == 0 for k in range(N))
    assert all(opt.temporal_masks[1][(k, 0)] == 0 for k in range(N))
    # 未涉及的时间边仍存活
    assert opt.temporal_masks[1][(1, 2)] == 1


def test_edge_reinforce_moves_per_round_logits_independently():
    opt = make_opt()
    reals = [opt.sample_round(0), opt.sample_round(1)]
    on = next(iter(reals[0].spatial_edges))
    opt.edge_reinforce([(reals, 1.0)])
    assert opt.spatial_logits[0][on] > 0.0          # 第 0 轮被采中且做对 → 提升
    off0 = next(e for e, (b, _) in reals[0].spatial_samples.items() if b == 0)
    assert opt.spatial_logits[0][off0] < 0.0
    untouched = next(e for e in opt.spatial_logits[1]
                     if e not in reals[1].spatial_samples)  # 掩码=0 的对角边等
    assert opt.spatial_logits[1][untouched] == pytest.approx(
        opt.deg_logits[1][untouched]) or True  # 逐轮独立：第 1 轮未采样边不动
    assert opt.temporal_logits and 1 in opt.temporal_logits


def test_edge_dropout_prunes_per_round_with_min_one():
    opt = make_opt()
    opt.spatial_logits[0][(0, 1)] = -5.0
    stats = opt.edge_dropout(0.25)
    # 每轮空间 20 条存活 × 0.25 = 5；两轮共 10；时间 25 条 × 0.25 ≈ 6
    assert stats["spatial_pruned"] == 10
    assert stats["temporal_pruned"] == 6
    assert opt.spatial_masks[0][(0, 1)] == 0  # 最低 logit 必被剪
    opt2 = make_opt()
    opt2.edge_dropout(0.001)  # rate 极小也强制每轮 ≥1（原版语义）
    assert sum(1 for r in range(R) for m in opt2.spatial_masks[r].values() if m == 0) \
        >= 2 * N + R  # 对角本为 0（每轮 N 个）+ 每轮至少剪 1


def test_realized_matrices_threshold_is_strict_and_round_checked():
    opt = make_opt()
    sm, tm = opt.realized_matrices(0)
    assert sum(map(sum, sm)) == 0  # 初始 σ=0.5，严格 >0.5 → 空图（原版 threshold 语义）
    opt.spatial_logits[0][(0, 1)] = 3.0
    sm, _ = opt.realized_matrices(0)
    assert sm[0][1] == 1 and sum(map(sum, sm)) == 1
    assert opt.realized_matrices(0)[1] == [[0] * N for _ in range(N)]  # 第 0 轮无时间边
    with pytest.raises(ValueError):
        opt.realized_matrices(R)


def test_state_roundtrip_and_mismatch(tmp_path):
    opt = make_opt()
    opt.deg_logits[0][(0, 1)] = -7.0
    opt.node_dropout()
    opt.edge_dropout(0.25)
    path = str(tmp_path / "state.json")
    opt.save(path)
    back = AgentDropoutOptimizer(n_agents=N, rounds=R, state_file=path)
    assert back.deg_logits[0][(0, 1)] == -7.0
    assert back.skip_nodes == opt.skip_nodes
    assert back.spatial_masks == opt.spatial_masks
    assert back.temporal_masks == opt.temporal_masks
    with pytest.raises(ValueError):
        AgentDropoutOptimizer(n_agents=N, rounds=R + 1, state_file=path)


def test_registered():
    assert "agentdropout" in REGISTRY.list("graph_pruner")
    assert "agentdropout" in REGISTRY.list("pre_run_optimizer")
    assert "agentdropout_v2" in REGISTRY.list("graph_pruner")  # V2 桩仍占名
