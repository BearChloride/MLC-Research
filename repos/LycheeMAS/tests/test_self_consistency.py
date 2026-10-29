"""self_consistency：多数投票（纯标准库可跑组件）。"""
from __future__ import annotations

from lychee_mas.core.registry import REGISTRY
from lychee_mas.core.types import Answer, Trajectory


def _agg():
    return REGISTRY.create("aggregator", "self_consistency")


def test_majority_vote_picks_mode():
    answers = [Answer(content="42"), Answer(content="42"), Answer(content="7")]
    out = _agg().aggregate(answers)
    assert out.content == "42"
    assert out.meta["votes"] == 2
    assert abs(out.confidence - 2 / 3) < 1e-6


def test_vote_is_case_and_space_insensitive():
    answers = [Answer(content="Yes"), Answer(content=" yes "), Answer(content="no")]
    out = _agg().aggregate(answers)
    assert out.content in ("Yes", " yes ")  # 归一化后同一票，取首个原始代表
    assert out.meta["votes"] == 2


def test_vote_over_trajectories():
    t1 = Trajectory(final_answer=Answer(content="A"))
    t2 = Trajectory(final_answer=Answer(content="A"))
    t3 = Trajectory(final_answer=Answer(content="B"))
    out = _agg().aggregate([t1, t2, t3])
    assert out.content == "A"


def test_empty_input():
    out = _agg().aggregate([])
    assert out.content == "" and out.meta["votes"] == 0
