"""AggAgent 接入可行性实验（Step 1：离线机制验证，LLM 决策脚本化、工具执行真实）。

AggAgent（arXiv:2604.11753, COLM 2026, princeton-pli/AggAgent）——「Agentic
Aggregation for Parallel Scaling of Long-Horizon Agentic Tasks」：并行跑 K 条轨迹后，
把聚合本身做成 agentic 任务——一个聚合 agent 把 K 条轨迹当作可检索环境，用四个轻量工具
跨轨迹推理出最终答案：

    get_solution(traj_id?)     取轨迹最后一步的最终解（不传 = 全部 K 个解，看共识/分歧）
    search_trajectory(traj, q) 单条轨迹内关键词检索 top-k 步（先粗定位）
    get_segment(traj, s, e)    读连续段完整内容（≤5 步，细读上下文验证）
    finish(solution, reason)   提交 <explanation>/<answer> XML 格式的最终答案

论文核心原则（本实验要演示的机制）：**数证据不数轨迹数**——单条带工具观测支撑的
证据，强于多条仅靠推理、无支撑的多数一致；冲突时信工具观测而非 agent 推理。

本脚本验证「能否接入 LycheeMAS」：
- 接入对象：LycheeMAS 的 ``Trajectory``（messages: sender/round/content）——缺 step 级
  结构，先做一层「轨迹 → 分段检索索引」适配（SegIndex/TrajEnv，真实实现）；
- 聚合循环：简化 agentic 循环（ACTION/FINISH 文本协议），工具执行器真实执行，
  只把「LLM 每一步决定调用什么」脚本化为数据驱动规则（读真实观察再决定）；
- 对拍基线：现成组件 ``aggregator/self_consistency``（REGISTRY 直取，多数投票）；
- 两个合成场景：①minority-correct（正确线索只藏在 1/5 轨迹的中间步、终答多数是错答，
  论文核心卖点）②majority-ok（多数终答正确，验证 AggAgent 不退化）。

离线、确定性、零重依赖（纯标准库 + core.types + REGISTRY 的 self_consistency）。

运行：PYTHONPATH=src python scripts/aggagent_feasibility.py

Step 2（真实接入，未做）：真实执行（LangGraph 图 × K 并行 → 收集 K 条 Trajectory）→
把 ``aggregate_agentic`` 的 ``llm_respond`` 换成真实 AsyncLLM 回调（prerun 同款注入
约定）→ 与 self_consistency 同配置对拍。核心函数与 TrajEnv 可原样迁往
``aggregator/aggagent``（届时按六步配方 + config + 测试收尾）。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Optional

from lychee_mas.core.registry import REGISTRY
from lychee_mas.core.types import Answer, Message, Trajectory

# ------------------------------ 轨迹检索化适配层 ------------------------------
# 未来迁移点：`methods/processing/aggagent/`（已落位）
# （论文原文按 step 检索；我们的 Trajectory 是 messages，按 (sender, round) 分段即可对齐）


@dataclass
class Segment:
    """一条轨迹里的一个可检索分段（对齐论文的一步）。"""

    traj_id: str
    step: int            # 段内连续编号（0, 1, 2, ...）
    sender: str
    round: int
    content: str

    @property
    def label(self) -> str:
        return f"{self.traj_id}:step{self.step}({self.sender})"


@dataclass
class SegIndex:
    """K 条轨迹的分段索引（构造即定，只读）。"""

    trajs: list[Trajectory] = field(default_factory=list)
    segments: list[Segment] = field(default_factory=list)

    @classmethod
    def build(cls, trajs: list[Trajectory]) -> "SegIndex":
        segs: list[Segment] = []
        for t in trajs:
            for i, m in enumerate(sorted(t.messages, key=lambda x: (x.round, x.sender))):
                segs.append(Segment(traj_id=t.id, step=i, sender=m.sender,
                                    round=m.round, content=m.content))
        return cls(trajs=list(trajs), segments=segs)

    def of(self, traj_id: str) -> list[Segment]:
        return [s for s in self.segments if s.traj_id == traj_id]

    def solution_of(self, traj_id: str) -> str:
        """轨迹的最终解 = 其 final_answer（无则取最后一条 message 的候选/内容）。"""
        t = next(x for x in self.trajs if x.id == traj_id)
        if t.final_answer is not None:
            return t.final_answer.content
        return self.of(traj_id)[-1].content if self.of(traj_id) else "(no solution)"


class TrajEnv:
    """聚合 agent 的检索环境（论文四工具的确定性实现；工具输出即 ground truth）。

    简化：search 用大小写不敏感子串命中计数排序（论文为 ROUGE-L，机制等价，换掉即升级）。
    """

    def __init__(self, trajs: list[Trajectory]):
        self.idx = SegIndex.build(trajs)

    # ---- 工具 1：get_solution ----
    def get_solution(self, traj_id: Optional[str] = None) -> str:
        if traj_id is not None:
            return f"[{traj_id}] {self.idx.solution_of(traj_id)}"
        return "\n".join(f"[{t.id}] {self.idx.solution_of(t.id)}" for t in self.idx.trajs)

    # ---- 工具 2：search_trajectory（先粗定位） ----
    def search_trajectory(self, traj_id: str, query: str, top_k: int = 3) -> str:
        q = query.lower()
        hits = sorted(
            (s for s in self.idx.of(traj_id) if q in s.content.lower()),
            key=lambda s: s.content.lower().count(q), reverse=True)
        if not hits:
            return f"search '{query}' in {traj_id}: no hits"
        return "\n".join(
            f"  hit {s.label}: ...{s.content[max(0, s.content.lower().find(q) - 40):s.content.lower().find(q) + 60]}..."
            for s in hits[:top_k])

    # ---- 工具 3：get_segment（细读上下文验证，≤5 步，对齐论文约束） ----
    def get_segment(self, traj_id: str, start: int, end: int) -> str:
        segs = self.idx.of(traj_id)
        lo, hi = max(0, start), min(len(segs) - 1, end)
        if end - start + 1 > 5:  # 论文：最多读 5 步
            hi = min(hi, lo + 4)
        if lo >= len(segs) or lo > hi:
            return f"get_segment({traj_id}, {start}, {end}): out of range"
        return "\n".join(f"  {s.label}: {s.content}" for s in segs[lo:hi + 1])

    def __repr__(self) -> str:
        return "TrajEnv(" + ", ".join(
            f"{t.id}:{len(self.idx.of(t.id))}steps" for t in self.idx.trajs) + ")"


# ------------------------------ 简化 agentic 聚合循环 ------------------------------

SYSTEM_PROMPT = """You are an aggregation agent. K parallel trajectories of a multi-agent run \
are available as a searchable environment. Survey (get_solution), verify claims against tool \
observations (search_trajectory / get_segment), cross-check, then submit the final answer. \
Tool observations are ground truth; agent reasoning is not. Count evidence, not trajectories. \
Reply with one line: ACTION: tool_name(args) | FINISH: <answer>...</answer>"""

_ACTION_RE = re.compile(r"^ACTION:\s*(\w+)\((.*)\)\s*$", re.S)
_FINISH_RE = re.compile(r"^FINISH:\s*(.*)$", re.S)
_ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.S)


def _call_tool(env: TrajEnv, name: str, raw_args: str) -> str:
    """工具执行器：名字分发 + 显式报错（模拟 tool-calling 的真实执行侧）。"""
    args = [a.strip().strip('"\'') for a in raw_args.split(",")] if raw_args.strip() else []
    if name == "get_solution":
        return env.get_solution(args[0] if args else None)
    if name == "search_trajectory":
        if len(args) < 2:
            raise ValueError(f"search_trajectory 需要 (traj_id, query)，得到 {args}")
        return env.search_trajectory(args[0], ",".join(args[1:]))
    if name == "get_segment":
        if len(args) != 3:
            raise ValueError(f"get_segment 需要 (traj_id, start, end)，得到 {args}")
        return env.get_segment(args[0], int(args[1]), int(args[2]))
    raise ValueError(f"未知工具 {name!r}（可用: get_solution/search_trajectory/get_segment）")


def aggregate_agentic(trajectories: list[Trajectory],
                      llm_respond: Callable[[str], str],
                      max_steps: int = 12) -> Answer:
    """简化 agentic 聚合：轮询 llm_respond(完整历史) -> ACTION/FINISH，工具执行真实。

    ``llm_respond`` 只做「读观察、决定下一步」（脚本化 = 离线验证；真实 = AsyncLLM
    回调，prerun 同款注入约定）。预算用尽未 finish 显式 RuntimeError。
    """
    env = TrajEnv(trajectories)
    history = [SYSTEM_PROMPT, f"ENVIRONMENT: {env}", f"QUESTION: {trajectories[0].task_id}"]
    for _ in range(max_steps):
        decision = llm_respond("\n\n".join(history)).strip()
        m = _ACTION_RE.match(decision)
        if m:
            observation = _call_tool(env, m.group(1), m.group(2))
            history += [f"DECISION: {decision}", f"OBSERVATION:\n{observation}"]
            continue
        m = _FINISH_RE.match(decision)
        if m:
            ans = _ANSWER_RE.search(m.group(1))
            if ans is None:
                raise ValueError(f"finish 缺少 <answer> 块：{decision!r}")
            return Answer(content=ans.group(1).strip(), source="aggagent",
                          meta={"trajectories": [t.id for t in trajectories]})
        raise ValueError(f"LLM 输出了无法解析的决策：{decision!r}")
    raise RuntimeError(f"聚合 agent 步数预算 {max_steps} 耗尽仍未 finish")


# ------------------------------ 合成场景（确定性、零依赖） ------------------------------

QUESTION = "A vault's access code was leaked in the project archive. What is the vault code?"


def _make_traj(traj_id: str, researcher_out: str, final_answer: str) -> Trajectory:
    """3 步轨迹（planner → researcher → answerer），answerer 内容与 final_answer 一致。"""
    t = Trajectory(task_id=QUESTION, id=traj_id)
    t.add(Message(sender="planner", content="Search the archive for the vault code.", round=0))
    t.add(Message(sender="researcher", content=researcher_out, round=1))
    t.add(Message(sender="answerer", content=final_answer, round=2))
    t.final_answer = Answer(content=final_answer, source=f"answerer/{traj_id}")
    return t


def make_minority_correct() -> list[Trajectory]:
    """场景 ①：5 条轨迹中只有 traj_4 的 researcher 段含正确证据（7362），
    但其 answerer 漏写（终答 'no code found'）；其余 4 条终答都是错的 7342。
    → 多数投票 7342（错）；聚合 agent 应凭工具观测找到 7362。"""
    out = []
    for i in range(4):
        out.append(_make_traj(f"traj_{i}",
                              f"Retrieved candidate code 7342 (rejected entry in archive).",
                              "7342"))
    out.append(_make_traj("traj_4",
                          "log line: vault code is 7362 (confirmed by two operators)",
                          "no code found"))
    return out


def make_majority_ok() -> list[Trajectory]:
    """场景 ②：5 条轨迹全部检索到 7362 且终答一致 → SC 与 AggAgent 都应答 7362。"""
    return [_make_traj(f"traj_{i}", "log line: vault code is 7362 (confirmed by two operators)",
                       "7362") for i in range(5)]


def scripted_llm(decision_log: list[str]) -> Callable[[str], str]:
    """数据驱动脚本策略（模拟「从粗到细 + 证据优先」的聚合 agent，决策基于真实观察）。

    状态机（只依据上一次决策 + 上一条观察，显式队列，不反推历史文本）：
      survey → get_solution()（概览全解）
      search → 按队列逐条 search_trajectory(traj, 'vault code is')：
               命中 → get_segment(命中步, 命中步) 读原文验证（论文 REQUIRED PROCEDURE）
               未命中 → 队列下一条；队列空 → 回到概览（无可检索证据）
      verify → 段原文含 'is NNNN' → FINISH 该码（证据优先：单条工具观测推翻无支撑多数）
    """
    state = {"queue": [f"traj_{i}" for i in range(5)]}

    def respond(history: str) -> str:
        last = decision_log[-1] if decision_log else None
        obs = history.split("OBSERVATION:")[-1] if "OBSERVATION:" in history else ""
        if last is None:                              # ① 概览
            decision = "ACTION: get_solution()"
        elif last == "ACTION: get_solution()":        # ② 从粗到细：开始逐条检索
            decision = (f"ACTION: search_trajectory({state['queue'].pop(0)}, 'vault code is')")
        elif "search_trajectory" in last:             # ③ 看上一步检索结果
            if "no hits" in obs:
                decision = (f"ACTION: search_trajectory({state['queue'].pop(0)}, 'vault code is')"
                            if state["queue"] else "ACTION: get_solution()")
            else:                                     # 命中：定位步号去读原文
                hit = re.search(r"(traj_\d):step(\d+)", obs)
                if hit is None:
                    raise AssertionError(f"命中但无法定位步号：{obs!r}")
                decision = f"ACTION: get_segment({hit.group(1)}, {hit.group(2)}, {hit.group(2)})"
        elif "get_segment" in last:                   # ④ 原文到手：验证并提交
            code = re.search(r"is (\d{4})", obs)
            if code is None:
                raise AssertionError(f"期望段原文含 'is NNNN'，得到：{obs!r}")
            decision = (f"FINISH: <explanation>found via tool observation</explanation>"
                        f"<answer>{code.group(1)}</answer>")
        else:
            raise AssertionError(f"脚本策略遇到未预期状态：{last!r}")
        decision_log.append(decision)
        return decision

    return respond


# ------------------------------ 对拍与断言 ------------------------------


def _run_case(trajs: list[Trajectory], name: str) -> None:
    decision_log: list[str] = []
    agg = aggregate_agentic(trajs, scripted_llm(decision_log))
    sc = REGISTRY.create("aggregator", "self_consistency").aggregate(trajs)
    print(f"\n=== {name}（{len(trajs)} 条轨迹 × 3 步）===")
    print(f"  self_consistency(多数投票) -> {sc.content!r}（confidence={sc.confidence:.2f}）")
    print(f"  aggagent(agentic 聚合)     -> {agg.content!r}")
    print(f"  工具调用序列: {[d.replace('ACTION: ', '') for d in decision_log]}")
    if name.startswith("①"):
        assert sc.content == "7342", f"场景设计错误：多数投票应选错答 7342，得到 {sc.content}"
        assert agg.content == "7362", f"minority-correct 场景聚合应找出 7362，得到 {agg.content}"
        print("  ✓ 少数正确证据被检索找回（数证据不数轨迹数）")
    else:
        assert agg.content == sc.content == "7362", "majority-ok 场景两法都应答 7362"
        print("  ✓ 多数正确时与 self_consistency 一致（不退化）")


def main() -> None:
    print("AggAgent 接入可行性 Step 1：离线机制验证（LLM 决策脚本化，工具执行真实）")
    _run_case(make_minority_correct(), "①minority-correct（正确证据只在 1/5 轨迹的中间步）")
    _run_case(make_majority_ok(), "②majority-ok（多数终答正确）")
    print("\n结论：聚合 agent 循环 + 检索环境可以在 LycheeMAS 的 Trajectory 上真实跑通，")
    print("     并能复现论文核心机制（凭工具观测找回被多数压过的正确证据）。")
    print("Step 2 接入路径：真实 LangGraph 执行 × K 并行 → 收集 Trajectory → 把")
    print("     aggregate_agentic 的 llm_respond 换成真实 AsyncLLM（prerun 同款注入）。")


if __name__ == "__main__":
    main()
