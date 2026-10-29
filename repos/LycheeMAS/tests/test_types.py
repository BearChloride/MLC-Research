"""公共类型：Trajectory.total_tokens / num_rounds 等记账。"""
from __future__ import annotations

from lychee_mas.core.types import Answer, Message, TaskQuery, Trajectory


def test_message_tokens():
    m = Message(sender="a", content="hi", prompt_tokens=10, completion_tokens=5)
    assert m.tokens == 15


def test_trajectory_total_tokens_and_rounds():
    traj = Trajectory(task_id="t1")
    traj.add(Message(sender="manager", content="x", round=0, prompt_tokens=3, completion_tokens=2))
    traj.add(Message(sender="worker", content="y", round=1, prompt_tokens=4, completion_tokens=1))
    traj.add(Message(sender="verifier", content="z", round=2, prompt_tokens=0, completion_tokens=0))
    assert traj.total_tokens == 3 + 2 + 4 + 1
    assert traj.num_rounds == 3  # 三个不同 round 编号


def test_empty_trajectory_rounds():
    traj = Trajectory(task_id="t0")
    assert traj.num_rounds == 0
    assert traj.total_tokens == 0


def test_answer_and_query_defaults():
    a = Answer(content="42", source="solver")
    assert a.content == "42" and a.source == "solver"
    q = TaskQuery(question="q?", gold="42")
    assert q.gold == "42" and q.id  # 自动生成 id
