"""LangGraph 运行脚本（构建层版）：网络构建用框架封装的构建函数，其余不接任何套件模块。

与 `run_langgraph_baseline.py` 的唯一区别在**网络构建**环节：
  - baseline    ：agent 画像内联在脚本里（脚本自包含）
  - 本脚本      ：用构建层封装函数产队伍——`methods.build.templates.team_to_agentspecs`
                  按 team 模板产出 AgentSpec 顺序链（MASGraph），team 名默认由
                  `eval.task_config.team_name_for_task(task)` 按任务选择，可 --team 覆盖

除构建层与评测模块外**不引入其他任何模块**：不用 runtime 后端 / 注入引擎 / 记忆 / 路由 /
插件 / Orchestrator / REGISTRY——执行仍是原生 StateGraph + transformers 直连。

五个环节：
  ① Benchmark 选择   eval.benchmarks.load(task, n)
  ② 网络构建         team_to_agentspecs(team) -> list[AgentSpec]（框架封装函数）
  ③ 执行             原生 StateGraph（每个 AgentSpec 一节点；APPROVE / 轮数上限终止）
  ④ 评测             eval.task_config.extractor_for_task + eval.metrics.score
  ⑤ 落盘             outputs.jsonl + metrics.json + config.yaml（与 run_mas 同口径）

注意：team 模板 prompt 中引用的「上一个 agent 输出」来源标志由记忆套件注入，本脚本不注入
——上一个 agent 的输出以普通对话消息形式对下游可见。含工具型节点的队伍（gaia/human_eval
等）本脚本显式拒绝（无工具执行环境，不静默降级）。

用法：
  CDM_DATA_ROOT=/data/.../raw CUDA_VISIBLE_DEVICES=0 \
      python scripts/run_langgraph_construct.py --task aime_2024 --n 3 \
      --model-path /data/mxy/Models/Qwen/Qwen3-4B          # team 自动 = aime
  python scripts/run_langgraph_construct.py --task gsm8k --n 5 --team single
"""
from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, TypedDict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "src"))

from lychee_mas.eval import metrics as M  # noqa: E402
from lychee_mas.eval.benchmarks import LOADERS  # noqa: E402
from lychee_mas.eval.benchmarks import load as load_benchmark  # noqa: E402
from lychee_mas.eval.task_config import extractor_for_task, team_name_for_task  # noqa: E402
from lychee_mas.methods.build.templates import TEAMS, team_to_agentspecs  # noqa: E402


class RunState(TypedDict):
    """LangGraph 状态：history 条目 = {source, content, prompt_tokens, completion_tokens}。"""

    history: List[Dict[str, Any]]
    agent_turns: int
    done: bool


# ================== 生成后端：transformers 直连（不经 lychee_mas runtime） ==================

@dataclass
class GenOut:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float


class HFChat:
    """最小 HF 聊天封装：chat 模板 + 贪心生成 + token/时延记账。"""

    def __init__(self, model_path: str, device: str = "cuda:0", dtype: str = "bfloat16"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=getattr(torch, dtype)).to(device).eval()
        self.device = device

    def _chat_ids(self, messages: List[Dict[str, str]]):
        try:  # Qwen3 支持 enable_thinking；其他模型不认这个参数则回退
            text = self.tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            text = self.tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        return self.tok(text, return_tensors="pt").to(self.device)

    def generate(self, messages: List[Dict[str, str]], max_new_tokens: int) -> GenOut:
        import torch

        inputs = self._chat_ids(messages)
        n_prompt = int(inputs["input_ids"].shape[1])
        t0 = time.time()
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=self.tok.eos_token_id)
        latency = time.time() - t0
        gen_ids = out[0][n_prompt:]
        text = self.tok.decode(gen_ids, skip_special_tokens=True)
        return GenOut(text=text.strip(), prompt_tokens=n_prompt,
                      completion_tokens=int(gen_ids.shape[0]), latency_s=round(latency, 3))


# ====================== ② 网络构建：框架封装函数产图 → 原生 StateGraph ======================

def build_team(task: str, team: str | None) -> tuple[str, list]:
    """用构建层封装函数产 AgentSpec 列表：team 名默认按任务选，显式 --team 覆盖。"""
    profile = team or team_name_for_task(task)
    if profile not in TEAMS:
        raise SystemExit(f"未知 team {profile!r}；可用：{sorted(TEAMS)}")
    specs = team_to_agentspecs(profile)
    for node in specs:  # 无工具执行环境：工具型节点显式拒绝，不静默降级
        kind = str(node.meta.get("agent_type") or node.meta.get("type") or "assistant").lower()
        if kind != "assistant" or node.tools or node.meta.get("tools"):
            raise SystemExit(
                f"team {profile!r} 含工具型节点 {node.name!r}（{kind}）；"
                "本脚本仅支持纯文本团队，请换 --team")
    return profile, specs


def build_app(specs: list, chat: HFChat, max_turns: int, max_new_tokens: int) -> Any:
    """把 AgentSpec 列表连成顺序轮转的 StateGraph（原生 API）。"""
    from langgraph.graph import END, START, StateGraph

    names = [s.name for s in specs]

    def make_node(name: str, system_prompt: str) -> Callable[[RunState],
                                                             Awaitable[Dict[str, Any]]]:
        async def node(state: RunState) -> Dict[str, Any]:
            msgs: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
            for entry in state["history"]:
                role = "assistant" if entry["source"] == name else "user"
                msgs.append({"role": role, "content": entry["content"]})
            g = chat.generate(msgs, max_new_tokens=max_new_tokens)
            entry = {"source": name, "content": g.text,
                     "prompt_tokens": g.prompt_tokens, "completion_tokens": g.completion_tokens,
                     "latency_s": g.latency_s}
            return {"history": state["history"] + [entry],
                    "agent_turns": state["agent_turns"] + 1,
                    "done": state["done"] or ("APPROVE" in g.text)}

        return node

    def route(state: RunState) -> str:
        return "end" if (state["done"] or state["agent_turns"] >= max_turns) else "continue"

    sg = StateGraph(RunState)
    for spec in specs:
        sg.add_node(spec.name, make_node(spec.name, spec.system_prompt))
    sg.add_edge(START, names[0])
    for i, name in enumerate(names):
        sg.add_conditional_edges(name, route,
                                 {"continue": names[(i + 1) % len(names)], "end": END})
    return sg.compile()


# ====================== ③ 执行 + ④ 评测（单 case） ======================

async def run_case(app: Any, task: str, record: Dict[str, Any], case_id: str,
                   max_turns: int, team: str) -> Dict[str, Any]:
    question = record["question"]
    if record.get("context"):
        question = f"{record['context']}\n\n{question}"  # 无记忆套件：上下文直接并入任务文本
    state: RunState = {
        "history": [{"source": "user", "content": question,
                     "prompt_tokens": 0, "completion_tokens": 0, "latency_s": 0.0}],
        "agent_turns": 0,
        "done": False,
    }
    t0 = time.time()
    final = await app.ainvoke(state, config={"recursion_limit": 2 * max_turns + 10})
    wall = time.time() - t0

    history = final["history"]
    prediction = extractor_for_task(task)(history)
    kind = record.get("kind") or "exact"
    score = M.score(kind, prediction, record.get("gold"))

    return {
        "case_id": case_id,
        "task": task,
        "kind": kind,
        "method": f"langgraph_construct_{team}",
        "question": record["question"],
        "gold": record.get("gold"),
        "prediction": prediction,
        "score": score,
        "is_correct": bool(score == 1.0) if M.is_binary_scorer(kind) else None,
        "stop_reason": "APPROVE" if final["done"] else "max_turns",
        "model_call_count": final["agent_turns"],
        "message_count": len(history),
        "input_positions_total": sum(int(e.get("prompt_tokens", 0)) for e in history),
        "gen_tokens": sum(int(e.get("completion_tokens", 0)) for e in history),
        "latency_s": round(sum(float(e.get("latency_s", 0.0)) for e in history), 3),
        "case_wall_time_s": round(wall, 3),
        "messages": [{"source": e["source"], "content": e["content"]} for e in history],
    }


# ====================== ⑤ 主流程：选择 → 构建 → 执行 → 评测 → 落盘 ======================

def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)), text=True).strip()
    except Exception:
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--task", required=True, choices=sorted(LOADERS),
                    help="benchmark 名（eval/benchmarks 已接入的 19 个）")
    ap.add_argument("--n", type=int, default=3, help="样本数（0=全量）")
    ap.add_argument("--team", default=None,
                    help="构建层 team 模板名（默认按任务选：task_config.team_name_for_task）")
    ap.add_argument("--model-path", default=os.environ.get("LYCHEE_HF_MODEL"),
                    help="HF 模型路径（默认 $LYCHEE_HF_MODEL）")
    ap.add_argument("--model-tag", default=None, help="落盘目录用的模型标签（默认取路径末段）")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--max-rounds", type=int, default=2, help="每 agent 最多发言轮数")
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--out-root", default=None,
                    help="结果根目录（默认 eval.benchmarks.common.runs_root()）")
    args = ap.parse_args()

    if not args.model_path:
        raise SystemExit("需要 --model-path 或环境变量 LYCHEE_HF_MODEL（不做静默兜底）")

    # ① Benchmark 选择
    records = load_benchmark(args.task, n=(args.n or None))
    if not records:
        raise SystemExit(f"benchmark {args.task!r} 加载为空：请先准备数据（benchmarks.prepare）")

    # ② 网络构建（框架封装函数）
    team, specs = build_team(args.task, args.team)
    names = [s.name for s in specs]
    max_turns = len(names) * args.max_rounds
    print(f"[construct] task={args.task} cases={len(records)} "
          f"team={team} agents={names} max_turns={max_turns}")

    chat = HFChat(args.model_path, device=args.device, dtype=args.dtype)
    app = build_app(specs, chat, max_turns=max_turns, max_new_tokens=args.max_new_tokens)

    # ③ 执行 + ④ 评测
    samples: List[Dict[str, Any]] = []
    for i, record in enumerate(records):
        case_id = str(record.get("id", i))
        sample = asyncio.run(run_case(app, args.task, record, case_id, max_turns, team))
        samples.append(sample)
        print(f"  [{i + 1}/{len(records)}] score={sample['score']:.3f} "
              f"stop={sample['stop_reason']} calls={sample['model_call_count']} "
              f"pred={sample['prediction'][:60]!r}")

    # ⑤ 落盘
    model_tag = args.model_tag or os.path.basename(os.path.normpath(args.model_path))
    method = f"langgraph_construct_{team}"
    out_dir = M.result_dir(model_tag, method, args.task, root=args.out_root)
    run_info = {"task": args.task, "method": method,
                "scorer_kind": samples[0]["kind"], "model": model_tag, "team": team}
    summary = M.aggregate_samples(samples, run_info)
    config = {
        "script": "run_langgraph_construct.py",
        "git_sha": git_sha(),
        "task": args.task, "n": len(samples),
        "team": team, "agents": names,
        "model_path": args.model_path, "model_tag": model_tag,
        "device": args.device, "dtype": args.dtype,
        "max_rounds": args.max_rounds, "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
    }
    M.write_results(out_dir, samples, summary, config)

    acc = summary.get("accuracy", summary.get("mean_score"))
    print(f"[construct] done: accuracy={acc} out_dir={out_dir}")
    print("[construct] artifacts: outputs.jsonl / metrics.json / config.yaml")


if __name__ == "__main__":
    main()
