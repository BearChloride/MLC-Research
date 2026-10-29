"""methods/prerun/graphops.py 抽取测试：原语归属变更后旧 import 路径零回归。

覆盖：`agentprune` 的 re-export 是**同一对象**（不是副本）、旧路径可用、插件层与
方法层取同一实现、`Adam` 去私有化后行为与 torch.optim.Adam 默认超参一致。
"""
from __future__ import annotations

import math

import pytest
from lychee_mas.methods.prerun import agentprune, graphops
from lychee_mas.methods.prerun.agentdropout import optimizer as ad_optimizer

# ------------------------------ re-export 身份 ------------------------------

@pytest.mark.parametrize("name", ["Edge", "Realization", "full_connected_masks",
                                  "topological_order"])
def test_agentprune_reexports_same_objects(name):
    """旧路径取到的就是 graphops 里的同一对象（兼容层，非拷贝）。"""
    assert getattr(agentprune, name) is getattr(graphops, name)
    assert name in agentprune.__all__


def test_private_adam_alias_kept():
    """`_Adam` 私有名继续可用，且指向去私有化后的 `graphops.Adam`。"""
    assert agentprune._Adam is graphops.Adam
    assert "_Adam" in agentprune.__all__


def test_consumers_share_one_implementation():
    """方法层（agentdropout）与插件层（graphview）都取 graphops 的同一实现。"""
    from lychee_mas.plugins.prerun import graphview

    assert ad_optimizer.Adam is graphops.Adam
    assert graphview.topological_order is graphops.topological_order
    # agentdropout 不再从 agentprune 借私有名
    assert ad_optimizer.Edge is graphops.Edge
    assert ad_optimizer.Realization is graphops.Realization
    assert ad_optimizer.full_connected_masks is graphops.full_connected_masks


def test_old_import_paths_still_work():
    from lychee_mas.methods.prerun.agentprune import (  # noqa: F401
        Adam,
        Edge,
        Realization,
        _Adam,
        full_connected_masks,
        topological_order,
    )


# --------------------------------- 行为契约 ---------------------------------

def test_topological_order_breaks_cycle_deterministically():
    order, preds = graphops.topological_order(3, {(0, 1), (1, 2), (2, 0)})
    assert order == [0, 1, 2]            # 无零入度 → 强制最小序号，逐轮推进
    assert preds == {0: set(), 1: {0}, 2: {1}}   # 破环边 2→0 被丢弃
    order, preds = graphops.topological_order(3, {(0, 1), (1, 2)})
    assert order == [0, 1, 2] and preds[2] == {1}


def test_full_connected_masks_match_reference_shapes():
    spatial, temporal = graphops.full_connected_masks(4)
    assert all(spatial[i][i] == 0 for i in range(4))            # 对角 0（无自环）
    assert sum(map(sum, spatial)) == 12 and sum(map(sum, temporal)) == 16


def test_adam_matches_reference_update():
    """首步 = -lr * sign(g)（偏差校正后 mhat/sqrt(vhat) = ±1），lr 生效。"""
    adam = graphops.Adam(lr=0.1)
    params, grads = {"a": 0.0, "b": 0.0}, {"a": 2.0, "b": -4.0}
    adam.step(params, grads)
    assert params["a"] == pytest.approx(-0.1) and params["b"] == pytest.approx(0.1)
    for _ in range(50):
        adam.step({"a": params["a"], "b": params["b"]}, {"a": 1.0, "b": 100.0})
    assert math.isfinite(params["a"]) and math.isfinite(params["b"])


def test_realization_log_prob_is_sum_of_bernoulli_logs():
    re = graphops.Realization(
        spatial_edges={(0, 1)}, temporal_edges=set(),
        spatial_samples={(0, 1): (1, 0.8)}, temporal_samples={(0, 2): (0, 0.25)})
    assert re.log_prob() == pytest.approx(math.log(0.8) + math.log(0.75))
