"""processing 层：serial（跑 1 次）+ parallel（并发跑 K 次 + 聚合）两子模块。"""
from __future__ import annotations

import asyncio
import itertools

from lychee_mas.core.registry import REGISTRY
from lychee_mas.core.types import Answer, Trajectory


def _runner_from(answers):
    """构造一个 runner：每次调用返回下一条 canned Trajectory（模拟每跑一次产一条轨迹）。"""
    it = iter(answers)

    async def runner():
        return Trajectory(final_answer=Answer(content=next(it)))

    return runner


def test_both_processors_registered():
    snap = REGISTRY.snapshot()
    assert set(snap["processor"]) >= {"serial", "parallel"}
    # 并行归约策略仍在 aggregator 类别（零破坏）
    assert "self_consistency" in snap["aggregator"]
    assert "dynamicagg" in snap["aggregator"]


def test_serial_runs_once_and_returns_that_answer():
    calls = itertools.count()

    async def runner():
        next(calls)
        return Trajectory(final_answer=Answer(content="42"))

    proc = REGISTRY.create("processor", "serial")
    res = asyncio.run(proc.run(runner))
    assert res.answer.content == "42"
    assert len(res.trajectories) == 1          # 只产一条轨迹
    assert next(calls) == 1                     # runner 只被调用一次（0 已被消费）


def test_parallel_runs_k_times_and_aggregates():
    # 3 条轨迹：9,9,4 → 多数投票 = 9
    proc = REGISTRY.create("processor", "parallel", k=3, aggregator="self_consistency")
    res = asyncio.run(proc.run(_runner_from(["9", "9", "4"])))
    assert res.answer.content == "9"
    assert len(res.trajectories) == 3          # 产 K=3 条轨迹
    assert res.answer.meta["votes"] == 2


def test_parallel_forwards_aggregator_kwargs():
    # aggregator_kwargs 透传给聚合器构造器：k=1 → 只取第一条终答（9），而非众数 4。
    # 若无透传，REGISTRY.create 会拿到默认 k=None，结果应是 "4"——本测试即转发证据。
    proc = REGISTRY.create(
        "processor", "parallel", k=3, aggregator="self_consistency",
        aggregator_kwargs={"k": 1})
    res = asyncio.run(proc.run(_runner_from(["9", "4", "4"])))
    assert res.answer.content == "9"
    assert res.answer.meta["votes"] == 1       # 只有 1 票参与（answers[:1]）


def test_processing_protocol_importable():
    from lychee_mas.methods.processing import ProcessingResult, Processor
    assert hasattr(Processor, "run")
    assert hasattr(ProcessingResult, "__dataclass_fields__")
