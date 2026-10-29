"""postrun 接缝 optimize_postrun 入口的离线测试（需要 langgraph，未装则整文件跳过；LLM 全脚本化）。

覆盖：REGISTRY 类别与桩可见性、桩实现显式报错、统一入口分发（未知 method
显式 KeyError）、图/轨迹/返回值三处校验（非法输入显式 TypeError、空轨迹合法）。
"""
from __future__ import annotations

from typing import Any, Dict, TypedDict

import pytest

pytest.importorskip("langgraph")

from langgraph.graph import END, START, StateGraph  # noqa: E402
from lychee_mas.core.registry import REGISTRY  # noqa: E402
from lychee_mas.core.types import AgentSpec, Message, Trajectory  # noqa: E402
from lychee_mas.plugins.postrun import optimize_postrun  # noqa: E402


class ChainState(TypedDict):
    question: str
    outputs: Dict[str, str]


def make_node(spec: AgentSpec, record: list[str]):
    """契约节点函数（与 test_prerun 同款：运行时从 spec 读模板与前驱）。"""

    async def node(state: ChainState) -> Dict[str, Any]:
        context = "\n---\n".join(
            state["outputs"][p] for p in spec.meta.get("predecessors", [])
            if p in state["outputs"])
        record.append(f"{spec.name}:{context}")
        return {"outputs": {**state["outputs"], spec.name: f"out-of-{spec.name}"}}

    return node


def make_graph(record: list[str] | None = None) -> StateGraph:
    """predictor -> reflector 最小链，按 graphview 节点契约建图（唯一终端 = reflector）。"""
    record = record if record is not None else []
    sg = StateGraph(ChainState)
    names = ["predictor", "reflector"]
    for i, name in enumerate(names):
        spec = AgentSpec(name=name, role=name,
                         system_prompt="Answer {question} with {context}",
                         meta={"predecessors": names[:i][-1:]})
        sg.add_node(name, make_node(spec, record), metadata={"agent_spec": spec})
    sg.add_edge(START, "predictor")
    sg.add_edge("predictor", "reflector")
    sg.add_edge("reflector", END)
    return sg


def make_trajectory(task_id: str = "t1") -> Trajectory:
    traj = Trajectory(task_id=task_id)
    traj.add(Message(sender="a", content="x"))
    return traj


# ------------------------------ 类别与桩可见性 ------------------------------


def test_registry_category_listed():
    assert "post_run_optimizer" in REGISTRY.snapshot()
    assert REGISTRY.list("post_run_optimizer") == ["attribution", "train"]


def test_stub_optimizers_raise_not_implemented():
    for name in ("attribution", "train"):
        opt = REGISTRY.create("post_run_optimizer", name)
        with pytest.raises(NotImplementedError, match="not wired yet"):
            opt.optimize(make_graph(), [make_trajectory()])


# ------------------------------ 统一入口分发与校验 ------------------------------


def test_unknown_method_keyerror_lists_available():
    with pytest.raises(KeyError, match="post_run_optimizer/unknown"):
        optimize_postrun(make_graph(), [make_trajectory()], method="unknown")


def test_input_must_be_uncompiled_state_graph():
    app = make_graph().compile()
    with pytest.raises(TypeError, match="未编译"):
        optimize_postrun(app, [make_trajectory()], method="attribution")


def test_input_must_be_state_graph():
    with pytest.raises(TypeError, match="StateGraph"):
        optimize_postrun(object(), [make_trajectory()], method="attribution")


def test_trajectories_must_be_trajectory_sequence():
    with pytest.raises(TypeError, match="Trajectory"):
        optimize_postrun(make_graph(), "not-trajectories", method="attribution")
    with pytest.raises(TypeError, match="Trajectory"):
        optimize_postrun(make_graph(), [object()], method="attribution")


def test_empty_trajectories_shape_valid_but_stub_still_raises():
    # 空轨迹序列在接缝层合法（apply 模式可能不消费轨迹）；桩实现仍显式报错
    with pytest.raises(NotImplementedError, match="not wired yet"):
        optimize_postrun(make_graph(), [], method="attribution")


def test_return_value_must_be_state_graph():
    @REGISTRY.register("post_run_optimizer", "_bad_return")
    class _BadReturn:
        def optimize(self, graph: Any, trajectories: Any) -> Any:
            return "not-a-graph"

    try:
        with pytest.raises(TypeError, match="_bad_return"):
            optimize_postrun(make_graph(), [make_trajectory()], method="_bad_return")
    finally:
        del REGISTRY._items["post_run_optimizer"]["_bad_return"]  # 清理，不污染快照
