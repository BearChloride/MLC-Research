"""离线端到端示例 02：aggregator/aggagent —— agentic 检索式聚合（AggAgent 移植）。

零重依赖、纯离线可跑（与 examples/01 同约束，无 GPU / 无 API key）：
  PYTHONPATH=src python examples/02_aggagent_e2e.py

演示什么：
1. 造 3 条合成 Trajectory（minority-correct 场景）：2 条的"工具观测"记录候选码 7342
   被归档拒绝、终答仍是 7342（多数）；1 条的工具观测实录 "vault code is 7362" 真值，
   但这条轨迹自己的终答是 "no code found"（少数且"答不出来"）——只有 role="tool"
   消息里藏着正确证据；
2. 注入一个确定性的"聚合 LLM"（DemoAggClient，本文件内实现）：概览全部解 → 有分歧就
   逐轨迹只搜 role="tool" 观测 → 命中带真值的步骤才读段细看 → finish 提交证据答案。
   工具观测即 ground truth、不数轨迹数的决策路径与原版 AggAgent 的 system prompt
   一致；真实实验把 client= 换成 model + api_base（HttpAggClient）即可，
   配置见 configs/aggregator/aggagent.yaml；
3. 同一批轨迹并排对照 aggregator/self_consistency 多数投票 → 多数投票取 7342（错），
   aggagent 检索工具观测取 7362（对）——"数证据不数轨迹数"。

打印：两路聚合结果对照、聚合循环的决策日志与统计（轮次 / 工具账 / 每轮 token）、
以及 aggregator 类别的 REGISTRY 快照。
"""
from __future__ import annotations

import json
import re
import pprint

from lychee_mas import REGISTRY
from lychee_mas.core.types import Answer, Message, Trajectory

QUESTION = "A vault's access code was leaked in the project archive. What is the vault code?"
# role="tool" 观测里出现 `is NNNN` 才算真值证据（检索命中只信这种内容，不信推理步）
_ANSWER_RE = re.compile(r"\bis (\d{4})\b")


# ============================== 场景构造 ==============================


def make_scenario() -> list[Trajectory]:
    """3 条轨迹：两条"观测到候选码被拒、终答错写 7342"，一条持有真值观测但终答漏写。"""
    out = []
    for i in range(2):
        t = Trajectory(task_id=QUESTION, id=f"traj_{i}")
        t.add(Message(sender="planner", content="Search the archive for the vault code.", round=0))
        t.add(Message(sender="researcher", role="tool",
                      content="Retrieved candidate code 7342 (rejected entry in archive).",
                      round=1))
        t.add(Message(sender="answerer", content="7342", round=2))
        t.final_answer = Answer(content="7342", source=f"answerer/traj_{i}")
        out.append(t)
    t = Trajectory(task_id=QUESTION, id="traj_2")
    t.add(Message(sender="planner", content="Search the archive for the vault code.", round=0))
    t.add(Message(sender="researcher", role="tool",
                  content="log line: vault code is 7362 (confirmed by two operators)", round=1))
    t.add(Message(sender="answerer", content="no code found", round=2))
    t.final_answer = Answer(content="no code found", source="answerer/traj_2")
    out.append(t)
    return out


# ============================== 脚本化聚合 LLM ==============================


def _tool_step(tool_name: str, arguments: dict, call_no: int) -> dict:
    """一条带单个 tool_call 的 assistant 步（OpenAI 消息形状，engine 消费它）。"""
    return {
        "role": "assistant",
        "content": "",
        "reasoning_content": f"reasoning for {tool_name} #{call_no}",
        "tool_calls": [{
            "id": f"call_{call_no}",
            "type": "function",
            "function": {"name": tool_name, "arguments": json.dumps(arguments)},
        }],
    }


def _last_tool_obs(messages: list[dict]):
    """取对话最后一条 role='tool' 观测并 json.loads（非 JSON 内容原样返回）。"""
    for msg in reversed(messages):
        if msg.get("role") == "tool":
            try:
                return json.loads(msg["content"])
            except (ValueError, TypeError):
                return msg["content"]
    return None


class DemoAggClient:
    """确定性"聚合 agent"：概览 → 有分歧逐轨迹检索工具观测 → 证据读段 → finish。

    决策只看最后一条工具观测（真实 LLM 同款输入），永不翻历史反推。完整协议见
    aggagent/client.py 的 AggClient。真实实验替换为 HttpAggClient 即可。
    """

    def __init__(self) -> None:
        self.log: list[str] = []              # 决策日志（演示打印用）
        self._phase = "survey"
        self._queue = [1, 2, 3]               # 待检索的 trajectory_id（1 起）
        self._traj_id = 0
        self._answer = ""
        self._call_no = 0

    def complete(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        self._call_no += 1
        obs = _last_tool_obs(messages)
        name, args = self._decide(obs)
        self.log.append(f"call#{self._call_no}: {name} {args}")
        return _tool_step(name, args, self._call_no)

    def _decide(self, obs) -> tuple[str, dict]:
        if self._phase == "survey":
            # 先概览全部终答（get_solution 不传 trajectory_id = 全取）
            self._phase = "search"
            return "get_solution", {}
        if self._phase == "search":
            # 按序挑一条还没查过的轨迹，只搜 role="tool" 的真实观测
            self._traj_id = self._queue.pop(0)
            self._phase = "search_result"
            return "search_trajectory", {
                "trajectory_id": self._traj_id, "query": "vault code",
                "role": "tool", "k": 5}
        if self._phase == "search_result":
            if isinstance(obs, str):          # "No matches found ..." → 下一条轨迹
                self._phase = "search"
                return self._decide(None)
            for hit in obs:
                # ROUGE-L 会带回弱相关步：只信内容含 `is NNNN` 的命中，其余跳过
                if _ANSWER_RE.search(hit.get("content", "")):
                    self._phase = "segment"
                    return "get_segment", {
                        "trajectory_id": self._traj_id,
                        "start_step": hit["step"], "end_step": hit["step"]}
            self._phase = "search"
            return self._decide(None)
        if self._phase == "segment":
            # 段原文验证：真值证据 = 工具观测内容里的数字
            match = _ANSWER_RE.search(obs[0].get("content", ""))
            assert match, f"段原文应含 'is NNNN'：{obs!r}"
            self._answer = match.group(1)
            self._phase = "done"
            return "finish", {
                "solution": (f"<explanation>Tool observation (role='tool') records "
                             f"the vault code.</explanation><answer>{self._answer}</answer>"),
                "reason": "verified in tool log of the trajectory with the real observation",
            }
        raise AssertionError(f"未预期状态 {self._phase!r}")


# ============================== 主流程 ==============================


def main() -> None:
    trajs = make_scenario()
    print("=== 场景：3 条并行轨迹（vault code 检索任务）===")
    for t in trajs:
        for m in t.messages:
            kind = m.role or "assistant"
            print(f"  traj {t.id:<7} [{m.round}] {m.sender:<10} ({kind}): {m.content}")

    # 对照 1：多数投票（只看终答）——被 2 条错答带偏
    voted = REGISTRY.create("aggregator", "self_consistency").aggregate(trajs)
    print("\n=== 对照：aggregator/self_consistency（多数投票，只看终答）===")
    print(f"answer    : {voted.content!r}   votes: {voted.meta['votes']}/{voted.meta['total']}")

    # 对照 2：aggagent（agentic 聚合）——REGISTRY 按名取组件（与 01 取
    # self_consistency 同款，不 import 具体类），注入确定性 LLM 作为聚合 client
    client = DemoAggClient()
    agg = REGISTRY.create("aggregator", "aggagent", question=QUESTION, client=client)
    answer = agg.aggregate(trajs)
    stats = answer.meta["stats"]
    print("\n=== aggregator/aggagent（agentic 检索式聚合）===")
    print(f"answer    : {answer.content!r}")
    print(f"variant   : {answer.meta['variant']!r}   n_trajectories: {answer.meta['n_trajectories']}")
    print("decision  log:")
    for line in client.log:
        print(f"  {line}")
    print(f"iterations: {stats['iterations']}   tool ledger: {stats['tool_calls']}"
          f"   tool_call_errors: {stats['tool_call_errors']}")
    print(f"per-step  tokens: {stats['token_usage_each_step']}")
    print(f"meta.reason     : {answer.meta['reason']}")

    print("\n=== 结论：数证据不数轨迹数 ===")
    print(f"多数投票取 {voted.content}（错——那只是被拒候选）；aggagent 检索 role='tool' "
          f"观测取 {answer.content}（对——工具观测是 ground truth）。")

    print("\n=== aggregator 类别组件 ===")
    print("available:", REGISTRY.list("aggregator"))
    print("\n真实实验：把上面注入的 client 换成 model+api_base（OpenAI 兼容 tool-calling")
    print("端点，聚合模型需支持 tools）即可，配置模板：configs/aggregator/aggagent.yaml；")
    print("或经 processor/parallel 的 aggregator + aggregator_kwargs 全链路聚合。")
    
    print("\n=== REGISTRY.snapshot() ===")
    pprint.pprint(REGISTRY.snapshot())


if __name__ == "__main__":
    main()
