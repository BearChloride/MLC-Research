"""五接缝离线端到端 demo（make demo）：build → prerun(apply) → compile → processing。

零 GPU / 零 API：LLM 为脚本化假后端；需要 langgraph（`.[langgraph]` extra），
缺失时显式报错（不静默降级）。展示——
  ① build_langgraph(method="static")     构建层产契约图（节点工厂注入）
  ② optimize_langgraph(method="maspo")   运行前挂载优化提示（apply 即插即用）
  ③ sg.compile() + 每题跑图组装轨迹
  ④ run_processed(method="parallel")     执行层并发 K 次 + self_consistency 归约
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from typing import Any, Dict, TypedDict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "src"))

from lychee_mas.core.registry import REGISTRY  # noqa: E402
from lychee_mas.core.types import Answer, Message, Trajectory  # noqa: E402
from lychee_mas.plugins import (  # noqa: E402
    build_langgraph,
    optimize_langgraph,
    run_processed,
)

try:
    import langgraph  # noqa: F401
except ImportError as exc:  # 显式报错：demo 依赖 langgraph extra
    raise SystemExit("demo 需要 langgraph：uv pip install -e '.[langgraph]'") from exc


class DemoState(TypedDict):
    question: str
    outputs: Dict[str, str]


def make_node(spec: Any, is_terminal: bool):
    """契约节点：脚本化「生成」——回显提示模板首词 + 终端给出 APPROVE 答案。"""

    async def node(state: DemoState) -> Dict[str, Any]:
        text = (f"[{spec.name}] prompt_head={spec.system_prompt.split()[0]!r} "
                + ("APPROVE: 4" if is_terminal else "thinking..."))
        return {"outputs": {**state["outputs"], spec.name: text}}

    return node


async def main() -> None:
    # ① 构建：team 模板 → 契约 StateGraph
    sg = build_langgraph(method="static", node_factory=make_node,
                         state_schema=DemoState, team="default")

    # ② 运行前：MASPO apply 挂载「优化后」的提示（demo 用手写 prompt 文件示意即插即用）
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        names = [n for n in sg.nodes]
        json.dump({"prompts": {names[0]: "OPTIMIZED. {question} {context}"}}, f)
    sg = optimize_langgraph(sg, method="maspo", mode="apply", prompt_file=f.name)
    os.unlink(f.name)

    # ③ 编译执行 + 轨迹组装（runner：跑一次产一条 Trajectory）
    app = sg.compile()
    terminal = [n for n in sg.nodes][-1]

    async def runner() -> Trajectory:
        result = await app.ainvoke({"question": "2+2=?", "outputs": {}})
        traj = Trajectory(task_id="demo")
        for name, text in result["outputs"].items():
            traj.add(Message(sender=name, content=text))
        final = result["outputs"][terminal].split("APPROVE:")[-1].strip()
        traj.final_answer = Answer(content=final, source=terminal)
        return traj

    # ④ 执行层：并发 3 次 + 自洽投票归约
    out = await run_processed(runner, method="parallel", k=3,
                              aggregator="self_consistency")
    print(f"final answer = {out.answer.content!r}  (k={len(out.trajectories)} 条轨迹投票)")
    print("registry snapshot:")
    for cat, names in sorted(REGISTRY.snapshot().items()):
        print(f"  {cat}: {names}")


if __name__ == "__main__":
    asyncio.run(main())
