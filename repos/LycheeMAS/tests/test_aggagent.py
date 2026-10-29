"""aggregator/aggagent —— 离线 mock 测试（LLM 全脚本化，确定性，零重依赖）。

被测对象 = princeton-pli/AggAgent 移植（见 src/.../parallel/aggagent/NOTICE.md）：
- 适配层 trajectory_to_steps / extract_answer_text（__init__.py）
- 四工具语义（tools.py：schema、clamp、role 过滤、finish 格式校验）
- 聚合 agent 循环（engine.py：client 注入接缝，脚本化客户端 = 论文 REQUIRED
  PROCEDURE 的确定性镜像：概览 → role='tool' 检索 → 命中读段 → finish）
- 显式报错路径（空输入 / 缺题面 / 缺 LLM / 迭代预算耗尽 / 上下文超限强制 finish）

两个合成场景（与 feasibility 实验同构，验证论文核心机制可复现）：
- minority-correct：正确证据只藏在 1/5 轨迹的工具观测步里、终答多数是错答 → 多数
  投票错、aggagent 应凭工具观测找回 7362（数证据不数轨迹数）；
- majority-ok：多数终答正确 → 与 self_consistency 一致，不退化。

运行：PYTHONPATH=src pytest tests/test_aggagent.py -q
"""
from __future__ import annotations

import io
import json
import re

import pytest

from lychee_mas.core.registry import REGISTRY
from lychee_mas.core.types import Answer, Message, Trajectory
from lychee_mas.methods.processing.aggagent import (
    AggAgentAggregator,
    extract_answer_text,
    trajectory_to_steps,
)
from lychee_mas.methods.processing.aggagent.client import HttpAggClient
from lychee_mas.methods.processing.aggagent.engine import run_aggregation
from lychee_mas.methods.processing.aggagent.prompts import FINAL_MESSAGE
from lychee_mas.methods.processing.aggagent.tools import (
    FinishTool,
    GetSegmentTool,
    GetSolutionTool,
    SearchTrajectoriesTool,
    _rouge_l_recall,
    format_metadata,
)

QUESTION = "A vault's access code was leaked in the project archive. What is the vault code?"
EVIDENCE = "log line: vault code is 7362 (confirmed by two operators)"


# ============================== 场景构造 ==============================


def _make_traj(traj_id: str, observation: str, final_text: str) -> Trajectory:
    """3 步轨迹：planner(assistant) → researcher(tool 观测) → answerer(assistant 终答)。

    末条消息 content 与 final_answer 一致（对齐 get_solution 取 last step 的语义；
    final_answer 仅供 self_consistency 对拍时取票）。
    """
    t = Trajectory(task_id=QUESTION, id=traj_id)
    t.add(Message(sender="planner", content="Search the archive for the vault code.", round=0))
    t.add(Message(sender="researcher", role="tool", content=observation, round=1))
    t.add(Message(sender="answerer", content=final_text, round=2))
    t.final_answer = Answer(content=final_text, source=f"answerer/{traj_id}")
    return t


def make_minority_correct() -> list[Trajectory]:
    """4 条终答错 7342 + 1 条证据在工具观测步但终答漏写（no code found）。"""
    out = []
    for i in range(4):
        out.append(_make_traj(f"traj_{i}",
                              "Retrieved candidate code 7342 (rejected entry in archive).",
                              "7342"))
    out.append(_make_traj("traj_4", EVIDENCE, "no code found"))
    return out


def make_majority_ok() -> list[Trajectory]:
    return [_make_traj(f"traj_{i}", EVIDENCE, "7362") for i in range(5)]


# ============================== 脚本化客户端 ==============================


def _assistant(tool_name: str, arguments: dict, call_no: int) -> dict:
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
    """取对话最后一条 role='tool' 消息并 json.loads 其内容（非 JSON 内容原样返回 str）。"""
    for msg in reversed(messages):
        if msg.get("role") == "tool":
            try:
                return json.loads(msg["content"])
            except (ValueError, TypeError):
                return msg["content"]
    return None


class ScriptedEvidenceClient:
    """确定性镜像「聚合 agent」：概览 → 逐轨迹 role='tool' 检索 → 命中读段 → finish。

    决策只看最后一条工具观测（真实 LLM 同款输入），永不翻历史反推 —— 工具观测即
    ground truth。参数控制输出变体与「首轮 finish 故意格式错误」等故障注入。
    """

    _ANSWER_RE = re.compile(r"\bis (\d{4})\b")

    def __init__(self, *, style: str = "xml", search_role: str = "tool",
                 fail_first_finish: bool = False, forced_answer: str = "7362",
                 query: str = "vault code is", traj_count: int = 5):
        self.style = style                    # xml | qwen | report
        self.search_role = search_role
        self.fail_first_finish = fail_first_finish
        self.forced_answer = forced_answer    # 上下文预算路径无检索线索时的终答
        self.query = query
        self._phase = "survey"
        self._queue: list[int] = list(range(1, traj_count + 1))  # 待检索的 trajectory_id
        self._call_no = 0
        self._traj_id: int | None = None
        self._step: int | None = None
        self._answer: str | None = None       # 段原文验证出的答案（finish 重试也复用它）
        self._finish_tried = False
        self.decisions: list[str] = []        # 审计用的决策轨迹

    # -- AggClient 协议 ---------------------------------------------------

    def complete(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        self._call_no += 1
        # 上下文预算强制 finish（engine 追加 FINAL_MESSAGE 后只传 finish 工具）
        if any(m.get("content") == FINAL_MESSAGE for m in messages[-4:]):
            self.decisions.append("forced_finish")
            return _assistant("finish", self._finish_args(self.forced_answer), self._call_no)

        obs = _last_tool_obs(messages)
        decision = self._decide(obs)          # ("tool_name", args) 二元组
        self.decisions.append(decision[0])
        return _assistant(decision[0], decision[1], self._call_no)

    # -- 状态机 -----------------------------------------------------------

    def _decide(self, obs) -> tuple[str, dict]:
        if self._phase == "survey":
            self._phase = "search"
            return "get_solution", {}
        if self._phase == "search":
            if not self._queue:
                # 全部轨迹无命中：显式失败而非臆造（测试场景不该走到这里）
                raise AssertionError("所有轨迹都检索无命中，脚本状态机无处可去")
            self._traj_id = self._queue.pop(0)
            self._phase = "search_result"
            return "search_trajectory", {
                "trajectory_id": self._traj_id, "query": self.query,
                "role": self.search_role, "k": 5}
        if self._phase == "search_result":
            if isinstance(obs, str) and "No matches found" in obs:
                self._phase = "search"  # 未命中 → 下一条轨迹
                return self._decide(None)
            # ROUGE-L 会带回弱相关步：只采信「内容含答案模式」的命中（工具观测即
            # ground truth）；本条轨迹无证据则换下一条，不臆造。
            for m in obs:
                if isinstance(m, dict) and self._ANSWER_RE.search(m.get("content", "")):
                    self._step = m["step"]
                    self._phase = "segment_result"
                    return "get_segment", {
                        "trajectory_id": self._traj_id,
                        "start_step": self._step, "end_step": self._step}
            self._phase = "search"
            return self._decide(None)
        if self._phase == "segment_result":
            answer = self._ANSWER_RE.search(obs[0].get("content", ""))
            assert answer, f"段原文应含 'is NNNN'：{obs!r}"
            self._answer = answer.group(1)
            if self.fail_first_finish and not self._finish_tried:
                self._finish_tried = True
                self._phase = "finish_retry"
                # 故意缺 <answer> 段：finish 工具应返回格式错误串，循环继续
                return "finish", {"solution": "<explanation>draft</explanation>", "reason": "draft"}
            return "finish", self._finish_args(self._answer or self.forced_answer)
        if self._phase == "finish_retry":
            return "finish", self._finish_args(self._answer or self.forced_answer)
        raise AssertionError(f"未预期状态 {self._phase!r}")

    def _finish_args(self, answer: str) -> dict:
        if self.style == "qwen":
            solution = f"Explanation: Found via tool observation.\nExact Answer: {answer}"
        elif self.style == "report":
            return {"solution_report": f"# Vault code\nBased on operator logs the vault code is {answer}.",
                    "reason": "log evidence"}
        else:
            solution = (f"<explanation>Tool observation supports the code.</explanation>"
                        f"<answer>{answer}</answer>")
        return {"solution": solution, "reason": "verified in tool log"}


class FinishNowClient:
    """无检索直接 finish 的客户端（question 覆盖 / 极小上下文预算等单步场景用）。"""

    def __init__(self, answer: str = "7362", style: str = "xml"):
        self.answer = answer
        self.style = style
        self.calls = 0

    def complete(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        self.calls += 1
        if self.style == "qwen":
            solution = f"Explanation: x\nExact Answer: {self.answer}"
        elif self.style == "report":
            return _assistant("finish", {"solution_report": f"Report says {self.answer}.",
                                         "reason": "r"}, self.calls)
        else:
            solution = f"<explanation>x</explanation><answer>{self.answer}</answer>"
        return _assistant("finish", {"solution": solution, "reason": "r"}, self.calls)


# ============================== 注册 / 显式报错 ==============================


def _agg(**kwargs) -> AggAgentAggregator:
    return REGISTRY.create("aggregator", "aggagent", **kwargs)


def test_registered_and_listed():
    assert "aggagent" in REGISTRY.list("aggregator")
    assert _agg().name == "aggagent"


def test_empty_trajectories_raises():
    with pytest.raises(ValueError, match="≥1 条轨迹"):
        _agg(client=FinishNowClient()).aggregate([])


def test_non_trajectory_input_raises():
    with pytest.raises(TypeError, match="list\\[Trajectory\\]"):
        _agg(client=FinishNowClient()).aggregate([Answer(content="x")])


def test_question_missing_raises():
    trajs = make_majority_ok()
    trajs[0].task_id = ""  # 其余轨迹 task_id 非空，但题面取第一条 → 缺题面应显式报错
    with pytest.raises(ValueError, match="缺少题面"):
        _agg(client=FinishNowClient()).aggregate(trajs)


def test_question_override_when_task_id_empty():
    trajs = make_majority_ok()
    for t in trajs:
        t.task_id = ""
    out = _agg(client=FinishNowClient(), question=QUESTION).aggregate(trajs)
    assert out.content == "7362"
    assert out.meta["question"] == QUESTION


def test_missing_llm_raises():
    trajs = make_majority_ok()
    with pytest.raises(ValueError, match="未配置聚合模型"):
        _agg().aggregate(trajs)


def test_client_without_complete_raises():
    with pytest.raises(TypeError, match="complete"):
        _agg(client=object()).aggregate(make_majority_ok())


# ============================== 适配层 ==============================


def test_trajectory_to_steps_order_and_fields():
    t = Trajectory(task_id=QUESTION, id="t1")
    t.add(Message(sender="b", content="round2", round=2))
    t.add(Message(sender="a", content="r0a", round=0))
    t.add(Message(sender="b", content="r0b", round=0))          # 同轮：按 sender 排
    t.add(Message(sender="tool", role="tool", content="obs",
                  meta={"reasoning": "think", "tool_calls": [{"x": 1}]}, round=1))
    steps = trajectory_to_steps(t)
    assert [s["content"] for s in steps] == ["r0a", "r0b", "obs", "round2"]
    assert steps[2]["role"] == "tool" and steps[0]["role"] == "assistant"
    assert steps[2]["reasoning_content"] == "think"
    assert steps[2]["tool_calls"] == [{"x": 1}]
    assert "tool_calls" not in steps[0]


def test_trajectory_to_steps_empty_and_plain():
    assert trajectory_to_steps(Trajectory(task_id="q")) == []
    t = Trajectory(task_id="q")
    t.add(Message(sender="s", content="hi", round=3))
    t.add(Message(sender="s", content="lo", round=1))
    assert trajectory_to_steps(t) == [
        {"role": "assistant", "content": "lo"},
        {"role": "assistant", "content": "hi"},
    ]


def test_extract_answer_text_variants():
    assert extract_answer_text("<explanation>e</explanation><answer> 42 </answer>",
                               variant="", model="") == "42"
    assert extract_answer_text("Explanation: e\nExact Answer: 7", variant="", model="Qwen3-30B") == "7"
    report = "Full report body."
    assert extract_answer_text(report, variant="long_form", model="") == report
    with pytest.raises(RuntimeError, match="无法从 solution"):
        extract_answer_text("<explanation>no answer tag</explanation>", variant="", model="")


# ============================== 工具单元 ==============================


def _steps(*entries: dict) -> list[dict]:
    return list(entries)


STEPS_ALL = [
    {"role": "user", "content": "question"},
    {"role": "assistant", "content": "", "tool_calls": [
        {"id": "c1", "function": {"name": "search", "arguments": "{}"}}]},
    {"role": "tool", "content": "log line: vault code is 7362 (confirmed by two operators)"},
    {"role": "assistant", "content": "final 7362"},
]


def test_get_solution_all_one_and_oob():
    t = GetSolutionTool()
    all_sol = t.call({}, trajectories=[STEPS_ALL, [{"role": "assistant", "content": "other"}]])
    assert all_sol == [
        {"trajectory_id": 1, "content": "final 7362"},
        {"trajectory_id": 2, "content": "other"},
    ]
    one = t.call({"trajectory_id": 2}, trajectories=[STEPS_ALL, STEPS_ALL])
    assert one == [{"trajectory_id": 2, "content": "final 7362"}]
    assert "must be 1-2" in t.call({"trajectory_id": 3}, trajectories=[STEPS_ALL, STEPS_ALL])
    assert t.call({}, trajectories=[[]]) == [{"trajectory_id": 1, "content": ""}]


def test_get_segment_clamp_and_max5():
    traj = [{"role": "assistant", "content": f"step{i}"} for i in range(10)]
    t = GetSegmentTool()
    seg = t.call({"trajectory_id": 1, "start_step": 2, "end_step": 9}, trajectories=[traj])
    assert [s["step"] for s in seg] == [2, 3, 4, 5, 6]          # 跨度 >5 → 截到 5 步
    assert seg[0]["content"] == "step1"
    seg2 = t.call({"trajectory_id": 1, "start_step": 100, "end_step": 100}, trajectories=[traj])
    assert [s["step"] for s in seg2] == [10]                     # 越界 → clamp 到末步
    seg3 = t.call({"trajectory_id": 1, "start_step": 8, "end_step": 3}, trajectories=[traj])
    assert [s["step"] for s in seg3] == [3]                      # start>end → 收敛到 end（原版语义）
    assert "must be 1-1" in t.call({"trajectory_id": 0, "start_step": 1, "end_step": 1},
                                   trajectories=[traj])


def test_search_trajectory_role_filter_and_rank():
    # 注意：ROUGE-L 召回是 LCS 部分匹配——'vault code is' 也会弱命中只含 'code'
    # 或 'vault' 的步（recall 1/3），排名按 recall 降序稳定排。
    traj = [
        {"role": "tool", "content": EVIDENCE},
        {"role": "assistant", "content": "no vault here"},
        {"role": "assistant", "content": "vault code is 7342 (agent guess, unsupported)"},
    ]
    t = SearchTrajectoriesTool()
    only_tool = t.call({"trajectory_id": 1, "query": "vault code is", "role": "tool"},
                       trajectories=[traj])
    assert [m["step"] for m in only_tool] == [1]                 # 该轨迹只有一步 tool 观测
    assert only_tool[0]["role"] == "tool"
    all_steps = t.call({"trajectory_id": 1, "query": "vault code is"}, trajectories=[traj])
    assert [m["step"] for m in all_steps] == [1, 3, 2]           # 1.0 > 0.333，稳定序
    assert all_steps[0]["score"] >= all_steps[-1]["score"] > 0
    only_assistant = t.call({"trajectory_id": 1, "query": "vault code is", "role": "assistant"},
                            trajectories=[traj])
    assert [m["step"] for m in only_assistant] == [3, 2]         # 整句命中排在弱命中前
    nomatch = t.call({"trajectory_id": 1, "query": "zzz not present", "role": "tool"},
                     trajectories=[traj])
    assert isinstance(nomatch, str) and "No matches found" in nomatch


def test_search_trajectory_k_cap_and_oob():
    traj = [{"role": "assistant", "content": f"share the word token{i}"} for i in range(20)]
    t = SearchTrajectoriesTool()
    hits = t.call({"trajectory_id": 1, "query": "share the word", "k": 99}, trajectories=[traj])
    assert len(hits) == 10                                        # k 封顶 10
    assert "must be 1-1" in t.call({"trajectory_id": 2, "query": "x"}, trajectories=[traj])
    assert "required" in t.call({"query": "x"}, trajectories=[traj])  # 缺 trajectory_id


def test_finish_default_xml_validation():
    f = FinishTool()
    good = f.call({"solution": "<explanation>e</explanation><answer>42</answer>", "reason": "r"},
                  trajectories=[])
    assert good == {"solution": "<explanation>e</explanation><answer>42</answer>", "reason": "r"}
    missing_answer = f.call({"solution": "<explanation>e</explanation>", "reason": "r"},
                            trajectories=[])
    assert isinstance(missing_answer, str) and "<answer>" in missing_answer
    missing_field = f.call({"solution": "<explanation>e</explanation><answer>1</answer>"},
                           trajectories=[])
    assert isinstance(missing_field, str) and "reason" in missing_field
    # 缺 <explanation> 校验
    bad_exp = f.call({"solution": "<answer>1</answer>", "reason": "r"}, trajectories=[])
    assert isinstance(bad_exp, str) and "<explanation>" in bad_exp


def test_finish_qwen_and_long_form():
    fq = FinishTool(model="Qwen3-235B-A22B")
    good = fq.call({"solution": "Explanation: because\nExact Answer: 7", "reason": "r"},
                   trajectories=[])
    assert good["solution"].startswith("Explanation:")
    bad = fq.call({"solution": "<explanation>e</explanation><answer>7</answer>", "reason": "r"},
                  trajectories=[])
    assert isinstance(bad, str) and "Explanation:" in bad

    fr = FinishTool(variant="long_form")
    rep = fr.call({"solution_report": "Report.", "reason": "r"}, trajectories=[])
    assert rep == {"solution": "Report.", "reason": "r"}
    empty = fr.call({"solution_report": "   ", "reason": "r"}, trajectories=[])
    assert isinstance(empty, str) and "solution_report" in empty
    nokey = fr.call({"reason": "r"}, trajectories=[])
    assert isinstance(nokey, str) and "solution_report" in nokey


def test_tool_definitions_schema():
    f = FinishTool()
    schema = f.get_tool_definitions()
    assert schema["type"] == "function" and schema["strict"] is True
    assert schema["function"]["name"] == "finish"
    assert schema["function"]["parameters"]["additionalProperties"] is False
    assert schema["function"]["parameters"]["required"] == ["solution", "reason"]
    assert FinishTool(variant="long_form").get_tool_definitions()["function"]["parameters"]["required"] \
        == ["solution_report", "reason"]


def test_rouge_l_recall_sanity():
    assert abs(_rouge_l_recall("vault code is", EVIDENCE) - 1.0) < 1e-9   # 整句含于证据
    assert 0.0 < _rouge_l_recall("two operators confirm", EVIDENCE) < 1.0  # 部分词序命中
    assert _rouge_l_recall("qwerty nonsense", EVIDENCE) == 0.0
    assert _rouge_l_recall("", EVIDENCE) == 0.0
    assert _rouge_l_recall("vault code is", "") == 0.0


def test_format_metadata():
    md = format_metadata([STEPS_ALL, [{"role": "assistant", "content": "solo"}]])
    assert "Trajectory 1: 4 steps" in md
    assert "search×1" in md                      # assistant 步的 tool_calls 记账
    assert "Trajectory 2: 1 steps" in md
    assert "tools: none" in md.split("\n\n")[1]


# ============================== 聚合流程（脚本化 LLM） ==============================


def test_minority_correct_evidence_beats_majority():
    """核心机制：正确证据只在 1/5 轨迹的工具观测步 → aggagent 找回，多数投票错。"""
    trajs = make_minority_correct()
    agg = _agg(client=ScriptedEvidenceClient())
    out = agg.aggregate(trajs)

    sc = REGISTRY.create("aggregator", "self_consistency").aggregate(trajs)
    assert sc.content == "7342"                    # 基线：多数终答是错答
    assert out.content == "7362"                   # aggagent：信工具观测
    assert out.source == "aggagent"
    assert out.meta["reason"].startswith("verified")
    assert "search_trajectory" in out.meta["stats"]["tool_calls"]
    assert "get_segment" in out.meta["stats"]["tool_calls"]
    assert out.meta["variant"] == ""
    assert out.meta["n_trajectories"] == 5


def test_majority_ok_no_regression():
    trajs = make_majority_ok()
    out = _agg(client=ScriptedEvidenceClient()).aggregate(trajs)
    sc = REGISTRY.create("aggregator", "self_consistency").aggregate(trajs)
    assert sc.content == out.content == "7362"
    assert out.meta["stats"]["tool_calls"].get("search_trajectory", 0) == 1  # 首条即命中


def test_aggagent_through_parallel_processor_kwargs():
    """processor/parallel + aggregator_kwargs 注入 aggagent：配置换名字即对拍的接线路径。

    之前 ParallelProcessor 构造聚合器不带任何超参，aggagent 无 client/model 会在
    aggregate 时 ValueError；透传后这条路径（configs/processor/parallel.yaml 的
    aggregator + aggregator_kwargs）应离线端到端可用。
    """
    import asyncio
    import itertools

    client = ScriptedEvidenceClient()
    proc = REGISTRY.create(
        "processor", "parallel", k=3, aggregator="aggagent",
        aggregator_kwargs={"client": client})
    canned = itertools.cycle(make_majority_ok())

    async def runner():
        return next(canned)

    res = asyncio.run(proc.run(runner))
    assert res.answer.content == "7362"        # 检索工具观测得到的答案
    assert res.answer.source == "aggagent"
    assert len(res.trajectories) == 3
    assert client.decisions[0] == "get_solution"   # 聚合 agent 真实走完了循环
    assert "finish" in client.decisions


def test_finish_format_error_then_retry():
    """首轮 finish 缺 <answer> → 工具返回格式错误串（非崩溃）→ 循环继续 → 修正后成功。"""
    trajs = make_minority_correct()
    client = ScriptedEvidenceClient(fail_first_finish=True)
    out = _agg(client=client).aggregate(trajs)
    assert out.content == "7362"
    assert client.decisions.count("finish") == 2
    assert out.meta["stats"]["tool_call_errors"] == 0   # 格式错误是工具返回值，非异常


def test_qwen_solution_variant_end_to_end():
    trajs = make_minority_correct()
    out = _agg(client=ScriptedEvidenceClient(style="qwen"), model="Qwen3-235B-A22B").aggregate(trajs)
    assert out.content == "7362"
    assert out.meta["solution"].startswith("Explanation:")


def test_long_form_report_variant_end_to_end():
    trajs = make_minority_correct()
    out = _agg(client=ScriptedEvidenceClient(style="report"), task="researchrubrics").aggregate(trajs)
    assert out.content == "# Vault code\nBased on operator logs the vault code is 7362."
    assert out.meta["variant"] == "long_form"
    assert out.meta["reason"] == "log evidence"


def test_context_limit_forced_finish():
    """小上下文预算：首轮工具调用后即超限 → FINAL_MESSAGE 强制 finish 仍产出答案。"""
    trajs = make_majority_ok()
    client = ScriptedEvidenceClient()
    out = _agg(client=client, max_context_tokens=1).aggregate(trajs)
    assert out.content == "7362"
    assert out.meta["stats"]["context_limit_reached"] is True
    assert "forced_finish" in client.decisions


def test_max_iterations_exhausted_raises():
    class NeverFinish:
        """永远只调 get_solution（绝不 finish）→ 迭代预算耗尽须显式 RuntimeError。"""

        def __init__(self):
            self.calls = 0

        def complete(self, messages, tools=None):
            self.calls += 1
            return _assistant("get_solution", {}, self.calls)

    trajs = make_majority_ok()
    with pytest.raises(RuntimeError, match="迭代预算 3 耗尽"):
        _agg(client=NeverFinish(), max_iterations=3).aggregate(trajs)


def test_server_error_sentinel_raises_after_budget():
    class AlwaysDown:
        def complete(self, messages, tools=None):
            return "Server error"

    trajs = make_majority_ok()
    with pytest.raises(RuntimeError, match="迭代预算 2 耗尽"):
        _agg(client=AlwaysDown(), max_iterations=2).aggregate(trajs)


def test_context_length_sentinel_raises():
    class ContextRefused:
        def complete(self, messages, tools=None):
            return "ContextLengthError"

    trajs = make_majority_ok()
    with pytest.raises(RuntimeError, match="ContextLengthError"):
        _agg(client=ContextRefused()).aggregate(trajs)


def test_unknown_tool_name_is_observation_not_crash():
    """模型幻觉工具名 → 错误串作为观测回灌，agent 可继续（原版 custom_call_tool 行为）。"""
    trajs = make_majority_ok()

    class HallucinatesOnce:
        def __init__(self):
            self.inner = ScriptedEvidenceClient()
            self.done = False

        def complete(self, messages, tools=None):
            if not self.done:
                self.done = True
                return _assistant("frobnicate_tool", {"x": 1}, 1)
            return self.inner.complete(messages, tools)

    out = _agg(client=HallucinatesOnce()).aggregate(trajs)
    assert out.content == "7362"
    assert "frobnicate_tool" in out.meta["stats"]["tool_calls"]
    assert out.meta["stats"]["tool_call_errors"] == 0


def test_answer_meta_records_usage_and_tool_ledger():
    trajs = make_majority_ok()
    out = _agg(client=ScriptedEvidenceClient()).aggregate(trajs)
    stats = out.meta["stats"]
    assert stats["iterations"] >= 3                       # survey/search/segment/finish 至少 3 轮
    assert stats["tool_calls"]["get_solution"] == 1
    assert stats["tool_calls"]["finish"] == 1
    assert all(e["iteration"] >= 1 for e in stats["token_usage_each_step"])


def test_engine_returns_contract_on_success():
    trajs = [trajectory_to_steps(t) for t in make_majority_ok()]
    out = run_aggregation(QUESTION, trajs, FinishNowClient(), max_iterations=5)
    assert out["error"] is None
    assert out["result"]["solution"].startswith("<explanation>")
    assert "finish" in out["stats"]["tool_calls"]


# ============================== HttpAggClient（假 urlopen） ==============================


class _FakeResp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeUrlopen:
    """记录请求并返回预设 payload；也可注入每次抛出的异常。"""

    def __init__(self, payload: dict | None = None, error=None):
        self.payload = payload
        self.error = error
        self.requests: list[dict] = []

    def __call__(self, request, timeout=None):
        body = json.loads(request.data.decode("utf-8"))
        self.requests.append(body)
        if self.error is not None:
            raise self.error
        return _FakeResp(json.dumps(self.payload).encode("utf-8"))


def test_http_client_sends_openai_body_and_parses_tool_calls(monkeypatch):
    payload = {
        "choices": [{"message": {
            "content": "",
            "reasoning_content": "think",
            "tool_calls": [{"id": "call_1", "type": "function",
                            "function": {"name": "get_solution", "arguments": "{}"}}],
        }}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20},
    }
    fake = _FakeUrlopen(payload=payload)
    monkeypatch.setattr("urllib.request.urlopen", fake)

    client = HttpAggClient(model="glm-test", api_base="http://localhost:6000/v1", max_tries=1)
    out = client.complete([{"role": "user", "content": "hi"}], tools=[{"type": "function"}])

    assert fake.requests[0]["model"] == "glm-test"
    assert fake.requests[0]["tools"] == [{"type": "function"}]
    assert fake.requests[0]["temperature"] == 1.0 and fake.requests[0]["max_tokens"] == 10000
    assert out["tool_calls"][0]["function"]["name"] == "get_solution"
    assert out["reasoning_content"] == "think"
    assert out["usage"]["prompt_tokens"] == 100
    # api_base 拼 /chat/completions
    assert fake.requests and "chat/completions" in client._url


def test_http_client_context_length_sentinel(monkeypatch):
    import urllib.error
    err = urllib.error.HTTPError(
        "http://x/chat/completions", 400, "Bad Request", {},
        _FakeResp(b'{"error": {"message": "maximum context length exceeded"}}'))
    fake = _FakeUrlopen(error=err)
    monkeypatch.setattr("urllib.request.urlopen", fake)

    client = HttpAggClient(model="m", api_base="http://x/v1", max_tries=5)
    assert client.complete([{"role": "user", "content": "q"}]) == "ContextLengthError"


def test_http_client_server_error_after_tries(monkeypatch):
    import urllib.error
    err = urllib.error.HTTPError("http://x/chat/completions", 500, "boom", {}, _FakeResp(b"{}"))
    fake = _FakeUrlopen(error=err)
    monkeypatch.setattr("urllib.request.urlopen", fake)

    client = HttpAggClient(model="m", api_base="http://x/v1", max_tries=1)  # 1 次 → 无退避
    assert client.complete([{"role": "user", "content": "q"}]) == "Server error"


def test_http_client_requires_model_and_base():
    with pytest.raises(ValueError, match="model 与 api_base"):
        HttpAggClient(model="", api_base="http://x/v1")
    with pytest.raises(ValueError, match="model 与 api_base"):
        HttpAggClient(model="m", api_base="")
