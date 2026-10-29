"""运行后接缝（postrun）—— 第五模块（归因训练）的三入口。

- **读侧** ``analyze_run(trajectory, score, method, ...)``：逐次运行的归因 → 信用，
  写回 ``trajectory.meta``（attributor + credit_assigner，实现/桩在 ``methods/postrun/``）。
- **图闭环** ``optimize_postrun(sg, trajectories, method, ...)``：与 prerun 对称的
  **批量离线**运行后优化——图 + 一批执行轨迹 τ → 优化 → 图（归因 → 信用 → 更新写回；
  post_run_optimizer 类别，桩在 ``methods/postrun/optimizers.py``）。
- **写侧** ``train_from_runs(method, ...)``：离线消费轨迹与信用信号产训练产物
  （trainer 类别，当前无实现——占名待接 RL 线）；产物统一经 prerun 的 apply 挂载回图。

方法自带训练（MASPO/GEPA/AgentPrune 的 optimize 模式）不走本接缝——训练素材自给的方法，
训练代码跟方法走（methods/prerun/）；素材来自执行轨迹的才在这里。
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Optional, Protocol, runtime_checkable

from ...core.registry import REGISTRY
from ...core.types import Trajectory
from ...methods.postrun.base import (  # noqa: F401  协议 re-export
    Attribution,
    CreditAssigner,
    FailureAttributor,
)
from ...methods.postrun.store import TraceStore  # noqa: F401


@runtime_checkable
class Trainer(Protocol):
    """离线训练器协议（写侧）：消费信用与轨迹，产出新参数/提示（RL 库只在实现里依赖）。"""

    def train(self, credits: dict[str, float], trajectories: list[Trajectory],
              **kwargs: Any) -> Any: ...


def analyze_run(trajectory: Trajectory, score: Optional[float] = None,
                method: str = "all_at_once", credit_assigner: str = "attribution_guided",
                trace_store: Any = None, **kwargs: Any) -> dict[str, float]:
    """读侧：归因（attributor/<method>）→ 信用（credit_assigner）→ 写回轨迹与 TraceStore。

    桩组件的 NotImplementedError 如实上抛（显式错误原则）。返回 per-agent 信用。
    """
    attributor = REGISTRY.create("attributor", method, **kwargs)
    assigner = REGISTRY.create("credit_assigner", credit_assigner)
    attributions = attributor.attribute(trajectory, context=None)
    credits = assigner.credits(attributions, float(score) if score is not None else 0.0)
    trajectory.meta["attribution"] = [asdict(a) for a in attributions]
    trajectory.meta["credits"] = dict(credits)
    if trace_store is not None:
        trace_store.log_decision({
            "type": "attribution", "trajectory_id": trajectory.id,
            "task_id": trajectory.task_id, "score": score, "credits": dict(credits),
        })
    return dict(credits)


@runtime_checkable
class PostRunOptimizer(Protocol):
    """运行后优化器协议：传入未编译 StateGraph + 轨迹批，返回优化后的 StateGraph。

    与 PreRunOptimizer 对称，语义差异在输入多一份执行轨迹（消费 τ 反哺图）。
    实现类经 ``@REGISTRY.register("post_run_optimizer", <method>)`` 注册；超参与运行
    素材走构造参数，``optimize`` 保持「图 + 轨迹 → 图」的最小签名。
    """

    def optimize(self, graph: Any, trajectories: Any) -> Any: ...


def _require_trajectories(obj: Any, where: str) -> list[Trajectory]:
    """校验 obj 是 Trajectory 序列（可为空——apply 模式可能不消费轨迹）。非法显式 TypeError。"""
    if isinstance(obj, (list, tuple)) and all(isinstance(t, Trajectory) for t in obj):
        return list(obj)
    raise TypeError(
        f"{where} 需要 Trajectory 序列（list/tuple，可空——apply 模式可能不消费轨迹），"
        f"得到 {type(obj).__name__}")


def optimize_postrun(graph: Any, trajectories: Any, method: str, **kwargs: Any) -> Any:
    """图闭环入口：按 ``method`` 从 REGISTRY 取运行后优化器，消费轨迹、优化图。

    用法示例::

        sg = optimize_postrun(sg, taus, method="attribution",
                              mode="apply", prompt_file="p.json")
        sg = optimize_postrun(sg, taus, method="attribution", mode="optimize",
                              attributor="all_at_once", evaluator_llm=...)
        app = sg.compile()

    未知 method 由 REGISTRY 显式 KeyError（并列出可用名）；图与轨迹的非法输入、
    以及非 StateGraph 的返回值均显式 TypeError（与 optimize_langgraph 同款约定）。
    """
    from ..prerun.base import _require_state_graph  # 两接缝共用「未编译 StateGraph」校验

    _require_state_graph(graph, "optimize_postrun")
    taus = _require_trajectories(trajectories, "optimize_postrun")
    optimizer = REGISTRY.create("post_run_optimizer", method, **kwargs)
    out = optimizer.optimize(graph, taus)
    _require_state_graph(out, f"post_run_optimizer/{method}.optimize 的返回值")
    return out


def train_from_runs(method: str, **kwargs: Any) -> Any:
    """写侧：按名取 trainer 执行训练（当前无实现，占名待接 RL 线）。"""
    trainer = REGISTRY.create("trainer", method, **kwargs)  # 未注册 → 显式 KeyError
    return trainer
