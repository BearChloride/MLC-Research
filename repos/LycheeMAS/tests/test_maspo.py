"""MASPO 算法核心的离线测试（纯标准库，LLM 全部脚本化，不需要 langgraph）。

GraphView 用手工构造（extract_view 的 langgraph 依赖在 tests/test_prerun.py 覆盖）。
"""
from __future__ import annotations

import asyncio
import json

import pytest
from lychee_mas.core.registry import REGISTRY
from lychee_mas.core.types import AgentSpec
from lychee_mas.plugins.prerun.graphview import GraphView
from lychee_mas.methods.prerun.maspo.executor import CachedExecutor
from lychee_mas.methods.prerun.maspo.optimizer import MASPOOptimizer
from lychee_mas.methods.prerun.maspo.prompts import (
    AGENT_TEMPLATES,
    role_description,
    seed_template,
)
from lychee_mas.methods.prerun.maspo.textops import (
    extract_answer,
    extract_prompt_tag,
    parse_comparison_result,
    sanitize_prompt,
)

OLD_P = "Solve it.\nQuestion: {question}\nContext: {context}"


# ------------------------------- textops -------------------------------

def test_sanitize_prompt_keeps_legal_placeholders():
    p = "Think hard.\nQuestion: {question}\nContext: {context}"
    assert sanitize_prompt(p, OLD_P) == p


def test_sanitize_prompt_rejects_unknown_placeholder():
    assert sanitize_prompt("Use {tools}!\nQ: {question}\nC: {context}", OLD_P) == OLD_P


def test_sanitize_prompt_appends_missing_question_and_context():
    out = sanitize_prompt("Just answer.", OLD_P)
    assert "{question}" in out and "{context}" in out  # 旧提示带 context 则补齐


def test_sanitize_prompt_rejects_duplicates_and_normalizes_braces():
    assert sanitize_prompt("{question} {question}", OLD_P) == OLD_P
    out = sanitize_prompt("{{question}} only", "Q {question}")  # 双花括号归一化后合法
    assert out.count("{question}") == 1


def test_sanitize_prompt_escapes_numeric_and_empty_braces():
    out = sanitize_prompt("set {0} and {} then {question}", "Q {question}")
    assert "{{0}}" in out and "{{}}" in out  # 转义为字面量，format 不再吃掉


def test_parse_comparison_result():
    assert parse_comparison_result("<analyse>x</analyse><choose>A</choose>") is True
    assert parse_comparison_result("<choose>B</choose>") is False
    assert parse_comparison_result("the better one is B") is False
    assert parse_comparison_result("A") is True
    assert parse_comparison_result("...") is True  # 原版语义：解析不出默认 A


def test_extract_answer_priority():
    assert extract_answer("blah <answer>42</answer> tail <answer>7</answer>") == "7"
    assert extract_answer(r"so \boxed{\frac{14}{3}} done") == r"\frac{14}{3}"
    assert extract_answer("line1\nthe answer is 5") == "the answer is 5"


def test_extract_prompt_tag():
    assert extract_prompt_tag("x<prompt> New P </prompt>y") == "New P"
    assert extract_prompt_tag("no tags here") is None


def test_prompt_assets_explicit_errors():
    assert "{question}" in seed_template("predictor")
    assert role_description("reflector")
    with pytest.raises(KeyError, match="aggregator"):
        seed_template("aggregator")
    with pytest.raises(KeyError):
        role_description("debator")


# --------------------------- 手工 GraphView + 脚本化 LLM ---------------------------

def make_view() -> GraphView:
    """predictor -> reflector 的 reflect 最小图（MASPO 主实验拓扑）。"""
    specs = {
        "predictor": AgentSpec(name="predictor", role="predictor",
                               system_prompt=AGENT_TEMPLATES["predictor"]),
        "reflector": AgentSpec(name="reflector", role="reflector",
                               system_prompt=AGENT_TEMPLATES["reflector"]),
    }
    return GraphView(names=["predictor", "reflector"], specs=specs,
                     predecessors={"predictor": [], "reflector": ["predictor"]},
                     terminal="reflector")


class ScriptedAgentLLM:
    """执行端脚本：记录所有收到的 prompt；压缩请求回摘要，其余回带 <answer> 的解答。"""

    def __init__(self, answer: str = "42"):
        self.answer = answer
        self.calls: list[str] = []

    async def __call__(self, prompt: str) -> str:
        self.calls.append(prompt)
        if prompt.startswith("Below is a solution"):  # COMPRESS_PROMPT
            return "summary: steps then answer"
        return f"reasoning...<answer>{self.answer}</answer>"


class ScriptedEvaluatorLLM:
    """评估/反思端脚本：按模板特征分派——反思回 <prompt>，比较按预设 A/B 回答。"""

    def __init__(self, proposal: str = "Improved.\nQuestion: {question}\nContext: {context}",
                 local: str = "A", global_: str = "A", next_local: str = "A",
                 terminal: str = "A"):
        self.proposal = proposal
        self.replies = {"local": local, "global": global_, "next_local": next_local,
                        "terminal": terminal}
        self.calls: list[str] = []

    async def __call__(self, prompt: str) -> str:
        self.calls.append(prompt)
        if "optimizing a prompt" in prompt:
            return f"<analyse>ok</analyse><prompt>{self.proposal}</prompt>"
        if "more conducive to obtaining the correct final answer" in prompt:
            # intermediate 比较：本节点与后继共用模板，靠 Output A 内容无法区分——统一回复
            return self.replies["local"]
        if "SOLE task" in prompt:
            return self.replies["global"]
        if "evaluating two outputs" in prompt:
            return self.replies["terminal"]
        raise AssertionError(f"未预期的评估请求: {prompt[:80]}")


def make_opt(agent, evaluator, **kw) -> MASPOOptimizer:
    defaults = dict(mode="optimize", trainset=["q1", "q2", "q3"], agent_llm=agent,
                    evaluator_llm=evaluator, max_total_depth=1, rounds_per_turn=1,
                    beam_width=2, eval_batch=2, seed=0, verbose=False)
    defaults.update(kw)
    return MASPOOptimizer(**defaults)


def run_with_limits(opt: MASPOOptimizer, coro_factory):
    """在带限流包装的事件循环里跑一个优化器内部协程（测试内部方法用）。"""

    async def runner():
        import asyncio as aio

        sem = aio.Semaphore(opt.max_concurrency)
        from lychee_mas.methods.prerun.maspo.optimizer import _limited

        opt.agent_llm = _limited(opt._raw_agent_llm, sem)
        opt.evaluator_llm = _limited(opt._raw_evaluator_llm, sem)
        opt.proposer_llm = _limited(opt._raw_proposer_llm or opt._raw_evaluator_llm, sem)
        return await coro_factory()

    return asyncio.run(runner())


# ------------------------------- executor -------------------------------

def test_cached_executor_run_and_partial_rerun():
    view = make_view()
    agent = ScriptedAgentLLM()
    ex = CachedExecutor(view, agent)
    final, cache = asyncio.run(ex.run("what is 6x7?"))
    assert final == "42"
    # predictor 生成 + predictor 压缩 + reflector 生成（终端不压缩）= 3 次调用
    assert len(agent.calls) == 3
    assert cache.node_outputs_short["reflector"] == "42"  # 终端 short = 抽取答案
    assert "summary: steps then answer" in cache.node_inputs["reflector"]  # context 用 short

    # 局部重执行：换 reflector 提示只重跑 reflector，predictor 读缓存
    n_before = len(agent.calls)
    _, cache2 = asyncio.run(ex.run_from_node(
        "reflector", cache, "New reflector.\nQ:{question}\nS:{context}\n"))
    assert len(agent.calls) == n_before + 1  # 只多一次生成
    assert cache2.node_outputs_raw["predictor"] == cache.node_outputs_raw["predictor"]
    assert agent.calls[-1].startswith("New reflector.")
    assert ex.prompt_map == {}  # 临时提示不污染 executor


def test_cached_executor_unknown_start_raises():
    ex = CachedExecutor(make_view(), ScriptedAgentLLM())
    _, cache = asyncio.run(ex.run("q"))
    with pytest.raises(KeyError, match="ghost"):
        asyncio.run(ex.run_from_node("ghost", cache))


# ---------------------------- 评分聚合与错位统计 ----------------------------

def eval_candidate(local, global_, next_local, name="predictor", use_lookahead=True):
    view = make_view()
    agent = ScriptedAgentLLM()
    evaluator = ScriptedEvaluatorLLM(local=local, global_=global_, next_local=next_local,
                                     terminal=global_)
    opt = make_opt(agent, evaluator)

    async def go():
        caches = await opt._build_baseline_caches(
            view, ["q1", "q2"], {n: view.specs[n].system_prompt for n in view.names})
        return await opt._evaluate_candidate(
            view, "Cand.\nQuestion: {question}\nContext: {context}", name,
            ["q1", "q2"], caches, {n: view.specs[n].system_prompt for n in view.names},
            use_lookahead=use_lookahead)

    return run_with_limits(opt, go)


def test_evaluate_candidate_all_wins_gives_half():
    info = eval_candidate("A", "A", "A")
    assert info["score"] == pytest.approx(0.5)  # 全赢：加权 win_rate=1 → 1-0.5
    assert info["bad_cases"] == [] and info["misalignment_rate"] == 0.0


def test_evaluate_candidate_all_losses_and_bad_cases():
    info = eval_candidate("B", "B", "B")
    assert info["score"] == pytest.approx(-0.5)
    assert len(info["bad_cases"]) == 2  # 每题 local 输 → 收 bad case


def test_evaluate_candidate_misalignment_local_win_global_lose():
    # 注意：本脚本无法区分 local 与 next_local（同模板），故两者同赢；只输 global
    info = eval_candidate("A", "B", "A")
    w_l, w_n, _w_g = (0.4, 0.4, 0.2)
    assert info["score"] == pytest.approx(w_l + w_n - 0.5)
    assert info["misalignment_rate"] == 1.0  # Local-Win / Global-Lose
    assert len(info["misleading_cases"]) == 2  # priority=2（global lose）


def test_evaluate_candidate_no_lookahead_weighting():
    info = eval_candidate("A", "B", "A", use_lookahead=False)
    assert info["score"] == pytest.approx(0.7 - 0.5)  # 0.7·local + 0.3·global


def test_evaluate_candidate_terminal_uses_answer_evaluate():
    info = eval_candidate("A", "A", "A", name="reflector")
    assert info["score"] == pytest.approx(0.5)  # 终端：单路 terminal 比较


# ------------------------------ beam 一步与主循环 ------------------------------

def test_process_single_node_keeps_winner():
    view = make_view()
    agent = ScriptedAgentLLM()
    evaluator = ScriptedEvaluatorLLM()  # 候选恒赢
    opt = make_opt(agent, evaluator)
    states = {n: __import__("lychee_mas.methods.prerun.maspo.optimizer",
                            fromlist=["AgentOptState"]).AgentOptState.seeded(
                                n, view.specs[n].system_prompt) for n in view.names}
    node = dict(states["predictor"].current_beam[0])

    res = run_with_limits(opt, lambda: opt._process_single_node(
        view, node, "predictor", states))
    assert res["best_prompt"].startswith("Improved.")
    assert res["best_cumulative"] > 0
    assert all(n["cumulative_score"] > 0 for n in res["nodes"])  # 赢者带累计分入 beam


def test_process_single_node_loser_keeps_old_node():
    view = make_view()
    evaluator = ScriptedEvaluatorLLM(local="B", global_="B", next_local="B", terminal="B")
    opt = make_opt(ScriptedAgentLLM(), evaluator)
    states = {n: __import__("lychee_mas.methods.prerun.maspo.optimizer",
                            fromlist=["AgentOptState"]).AgentOptState.seeded(
                                n, view.specs[n].system_prompt) for n in view.names}
    node = dict(states["predictor"].current_beam[0])
    res = run_with_limits(opt, lambda: opt._process_single_node(
        view, node, "predictor", states))
    assert res["best_prompt"] == node["prompt"]  # 没有更优者：保留原提示
    assert res["best_cumulative"] == pytest.approx(0.0)


def test_optimize_all_fixed_rounds_end_to_end(tmp_path):
    view = make_view()
    agent = ScriptedAgentLLM()
    evaluator = ScriptedEvaluatorLLM()
    opt = make_opt(agent, evaluator, prompt_file=str(tmp_path / "p.json"))
    prompt_map, stats = asyncio.run(opt._optimize_with_limits(view))
    assert set(prompt_map) == {"predictor", "reflector"}
    assert prompt_map["predictor"].startswith("Improved.")
    assert "misalignment_rates" in stats and "final_scores" in stats


# ------------------------------ 模式与显式报错 ------------------------------

def test_constructor_explicit_errors():
    with pytest.raises(ValueError, match="apply|optimize"):
        MASPOOptimizer(mode="train")
    with pytest.raises(ValueError, match="lookahead_weights"):
        MASPOOptimizer(lookahead_weights=(0.0, 0.0))


def test_load_prompt_file_formats(tmp_path):
    opt = MASPOOptimizer(mode="apply", prompt_file=str(tmp_path / "p.json"))
    with open(opt.prompt_file, "w") as f:
        json.dump({"prompts": {"a": "P {question}"}}, f)
    assert opt._load_prompt_file() == {"a": "P {question}"}
    with open(opt.prompt_file, "w") as f:
        json.dump({"a": "P {question}"}, f)  # 兼容裸映射
    assert opt._load_prompt_file() == {"a": "P {question}"}
    with open(opt.prompt_file, "w") as f:
        json.dump({"prompts": {"a": 1}}, f)  # 非法值显式报错
    with pytest.raises(ValueError, match="无法解析"):
        opt._load_prompt_file()


def test_apply_without_prompt_file_raises():
    with pytest.raises(ValueError, match="prompt_file"):
        MASPOOptimizer(mode="apply")._load_prompt_file()


def test_registered():
    assert "maspo" in REGISTRY.list("pre_run_optimizer")
    assert "agentprune" in REGISTRY.list("pre_run_optimizer")


def test_proposer_llm_receives_proposals_evaluator_receives_comparisons():
    """反思提议走 proposer_llm（原版 temp 0.7 分工），三路比较走 evaluator_llm。"""
    view = make_view()
    proposer_calls, evaluator_calls = [], []

    async def proposer(prompt):
        proposer_calls.append(prompt)
        return "<prompt>P-NEW.\nQuestion: {question}\nContext: {context}</prompt>"

    async def evaluator(prompt):
        evaluator_calls.append(prompt)
        return "A"

    opt = make_opt(ScriptedAgentLLM(), evaluator, proposer_llm=proposer)
    prompt_map, _stats = asyncio.run(opt._optimize_with_limits(view))
    assert prompt_map["predictor"].startswith("P-NEW.")
    assert all("optimizing a prompt" in c for c in proposer_calls)  # 提议全走 proposer
    assert not any("optimizing a prompt" in c for c in evaluator_calls)  # 比较端无提议请求
    assert proposer_calls and evaluator_calls


def test_proposer_llm_defaults_to_evaluator():
    view = make_view()
    evaluator = ScriptedEvaluatorLLM()
    opt = make_opt(ScriptedAgentLLM(), evaluator)  # 不传 proposer_llm
    prompt_map, _ = asyncio.run(opt._optimize_with_limits(view))
    assert prompt_map["predictor"].startswith("Improved.")  # 回退 evaluator 承担提议


def test_checkpoint_resume_after_crash(tmp_path):
    """外部故障中断后从 checkpoint 恢复：已探索层数保留，完成后现场文件删除。"""
    import os
    view = make_view()
    pf = str(tmp_path / "p.json")

    class FlakyEvaluator(ScriptedEvaluatorLLM):
        def __init__(self, die_after):
            super().__init__()
            self.die_after = die_after

        async def __call__(self, prompt):
            if len(self.calls) >= self.die_after:
                raise RuntimeError("simulated quota error")
            return await super().__call__(prompt)

    # 第一段：predictor 跑完 1 层后（rounds_per_turn=1）评估端开始报错 → 中断
    flaky = FlakyEvaluator(die_after=8)
    opt = make_opt(ScriptedAgentLLM(), flaky, prompt_file=pf,
                   max_total_depth=2, rounds_per_turn=1)
    with pytest.raises(RuntimeError, match="simulated"):
        asyncio.run(opt._optimize_with_limits(view))
    ckpt = str(tmp_path / "p_ckpt.json")
    assert os.path.isfile(ckpt)  # 至少一个 agent 轮次已落盘
    saved = json.load(open(ckpt))
    assert saved["states"]["predictor"]["total_layers_explored"] >= 1

    # 第二段：新实例 + 正常评估端 → 从现场恢复并跑完，checkpoint 删除
    opt2 = make_opt(ScriptedAgentLLM(), ScriptedEvaluatorLLM(), prompt_file=pf,
                    max_total_depth=2, rounds_per_turn=1)
    prompt_map, _ = asyncio.run(opt2._optimize_with_limits(view))
    assert set(prompt_map) == {"predictor", "reflector"}
    assert not os.path.isfile(ckpt)


def test_checkpoint_node_mismatch_raises(tmp_path):
    import os
    pf = str(tmp_path / "p.json")
    ckpt = str(tmp_path / "p_ckpt.json")
    with open(ckpt, "w") as f:
        json.dump({"states": {"ghost": {}}}, f)
    opt = make_opt(ScriptedAgentLLM(), ScriptedEvaluatorLLM(), prompt_file=pf)
    with pytest.raises(ValueError, match="不符"):
        asyncio.run(opt._optimize_with_limits(make_view()))
    os.remove(ckpt)
