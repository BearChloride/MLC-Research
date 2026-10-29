"""复现 MASPO 的 MATH-500 实验（LangGraph 执行 + 本框架 prerun 运行前优化统一接口）。

对标：MASPO: Joint Prompt Optimization for LLM-based Multi-Agent Systems（ICML 2026，
arXiv:2605.06623）；参考实现 https://github.com/wangzx1219/MASPO 的 run_maspo.py
（reflect 拓扑 + `--optimize --fixed-rounds --beam-refresh --lookahead-score
--misleading-sampling` 论文模式）。

与原版逐项对应：
  拓扑         reflect：predictor→reflector 链（--nr 轮：p0→r0→p1→r1→...，终端=末位 reflector）；
               提示模板/压缩/比较/反思提示词逐字 vendored（plugins/prerun/maspo/prompts.py）
  执行语义     每节点单条 user 消息 = template.format(question, context)；
               context = 前驱**压缩短输出**以 "\\n---\\n" 拼接；非终端节点输出过 COMPRESS 压缩，
               终端节点 short = extract_answer(raw)（原版 arun_with_cache / arun_full）
  模型         执行 LLM = Qwen3-8B（本地 HF，原版同款模型经 API）；
               评估/反思 LLM = gemini-2.5-pro（OpenAI 兼容端点，原版同款）
  优化         fixed-rounds 坐标上升（depth=9, rounds_per_turn=3, beam=2）+ Beam Refresh
               + Lookahead(0.4/0.4/0.2) + Misleading Sampling；训练问题采样自同一数据集
               （免 gold 标注，与原版 load_train_for_opt 协议一致）
  评测         MATH-500（benchmark/math500），kind="aime"（数值+sympy 符号等价）打分；
               对比对象 = 同模型下「原始提示 vs MASPO 优化提示」的 accuracy 与 token 成本

与原版的声明差异：
  - 执行引擎为 LangGraph StateGraph（节点按 graphview 契约挂 AgentSpec 元数据）；
    优化经**本框架统一接口** optimize_langgraph(sg, method="maspo", ...) 完成——
    optimize 阶段落 prompt JSON，eval 阶段 mode="apply" 即插即用挂载。
  - 打分用本框架 aime 等价打分（原版 normalize_answer 字符串比对）；采样 seeded 可复现。

用法（三条腿：baseline 基线 → optimize 优化 → eval 挂载评测；both = optimize+eval）：
  CUDA_VISIBLE_DEVICES=0 EVALUATOR_API_KEY=... \\
      python scripts/run_maspo_langgraph.py --phase baseline --eval-n 100
  CUDA_VISIBLE_DEVICES=0 EVALUATOR_API_KEY=... \\
      python scripts/run_maspo_langgraph.py --phase both --train-n 50 --eval-n 100 \\
      --evaluator-base-url https://<openai-compatible>/v1
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, TypedDict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "src"))

from lychee_mas.core.types import AgentSpec  # noqa: E402
from lychee_mas.eval import metrics as M  # noqa: E402
from lychee_mas.eval.benchmarks import load as load_benchmark  # noqa: E402
from lychee_mas.plugins.prerun import optimize_langgraph  # noqa: E402
from lychee_mas.methods.prerun.maspo.executor import (  # noqa: E402
    CONTEXT_JOINER,
    format_agent_prompt,
)
from lychee_mas.methods.prerun.maspo.prompts import COMPRESS_PROMPT, seed_template  # noqa: E402
from lychee_mas.methods.prerun.maspo.textops import extract_answer  # noqa: E402

DEFAULT_MODEL = "/data/mxy/Models/Qwen/Qwen3-8B"       # MASPO 执行模型（本地权重）
DEFAULT_EVALUATOR_MODEL = "gemini-2.5-pro"             # MASPO 评估/反思模型


class ReflectState(TypedDict):
    """LangGraph 状态：raw[name]=节点原始输出；short[name]=压缩短输出（下游 context 用）。"""

    question: str
    raw: Dict[str, str]
    short: Dict[str, str]


@dataclass
class Stats:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model_calls: int = 0
    latency_s: float = 0.0

    def add(self, prompt_tokens: int, completion_tokens: int, latency_s: float) -> None:
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.model_calls += 1
        self.latency_s += latency_s

    def snapshot(self) -> Dict[str, Any]:
        return {"prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "model_calls": self.model_calls, "latency_s": round(self.latency_s, 3)}


# ================== 执行 LLM：transformers 直连（同级脚本同款，单条 user 消息） ==================

class HFChat:
    def __init__(self, model_path: str, device: str = "cuda:0", dtype: str = "bfloat16"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=getattr(torch, dtype)).to(device).eval()
        self.device = device

    def generate_user(self, prompt: str, max_new_tokens: int, stats: Stats,
                      temperature: float = 0.0) -> str:
        import torch

        messages = [{"role": "user", "content": prompt}]  # 原版 async_call_llm 同款
        try:
            text = self.tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            text = self.tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tok(text, return_tensors="pt").to(self.device)
        n_prompt = int(inputs["input_ids"].shape[1])
        # temperature=0 贪心；>0 采样（反思提议端 0.7；top_p/top_k 取 Qwen3 推荐值）
        sample_kwargs = ({"do_sample": True, "temperature": float(temperature),
                          "top_p": 0.95, "top_k": 20}
                         if temperature > 0 else {"do_sample": False})
        t0 = time.time()
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                pad_token_id=self.tok.eos_token_id, **sample_kwargs)
        gen_ids = out[0][n_prompt:]
        stats.add(n_prompt, int(gen_ids.shape[0]), time.time() - t0)
        return self.tok.decode(gen_ids, skip_special_tokens=True).strip()


# ====================== 建图（graphview 节点契约：metadata 挂 AgentSpec） ======================

def make_node(spec: AgentSpec, chat: HFChat, stats: Stats, is_terminal: bool,
              max_new_tokens: int):
    """契约节点函数：运行时从 spec 读模板/前驱（优化器换 spec.system_prompt 即换行为）。"""

    async def node(state: ReflectState) -> Dict[str, Any]:
        context = CONTEXT_JOINER.join(
            state["short"][p] for p in spec.meta.get("predecessors", ())
            if p in state["short"])
        raw = chat.generate_user(
            format_agent_prompt(spec.system_prompt, state["question"], context),
            max_new_tokens, stats)
        if is_terminal:
            short = extract_answer(raw)   # 原版：终端 short = 抽取的最终答案
        else:
            short = chat.generate_user(COMPRESS_PROMPT.format(raw=raw), max_new_tokens, stats)
        return {"raw": {**state["raw"], spec.name: raw},
                "short": {**state["short"], spec.name: short}}

    return node


def build_reflect_graph(chat: HFChat, nr: int, stats: Stats, max_new_tokens: int) -> Any:
    """reflect 拓扑（原版 GraphType.REFLECT）：p0→r0→p1→r1→...，终端=末位 reflector。"""
    from langgraph.graph import END, START, StateGraph

    names: List[str] = []
    for i in range(nr):
        names += [f"predictor_{i}", f"reflector_{i}"]
    sg = StateGraph(ReflectState)
    for idx, name in enumerate(names):
        role = name.split("_")[0]
        spec = AgentSpec(name=name, role=role, system_prompt=seed_template(role),
                         meta={"predecessors": names[idx - 1:idx]})
        sg.add_node(name, make_node(spec, chat, stats, idx == len(names) - 1, max_new_tokens),
                    metadata={"agent_spec": spec})
    sg.add_edge(START, names[0])
    for a, b in zip(names, names[1:]):
        sg.add_edge(a, b)
    sg.add_edge(names[-1], END)
    return sg


# ============ 两路 LLM 回调（统一接口的 agent_llm / evaluator_llm，MASPO 同款分工） ============

def make_agent_llm(chat: HFChat, stats: Stats, max_new_tokens: int):
    async def agent_llm(prompt: str) -> str:
        return chat.generate_user(prompt, max_new_tokens, stats)  # 本地 HF：天然串行

    return agent_llm


def make_evaluator_llms(args: argparse.Namespace, stats: Stats):
    """返回 (evaluator_llm, proposer_llm)：比较端 temperature=0.0 / 提议端 0.7。

    --evaluator-model-path 给出时走本地 HF 大模型（如 Qwen3-32B，独占一张卡，
    与执行端 8B 分卡；只加载一份权重，两路温度共享）；否则走 OpenAI 兼容 API。
    """
    if args.evaluator_model_path:
        chat = HFChat(args.evaluator_model_path, device=args.evaluator_device,
                      dtype=args.evaluator_dtype)

        def make(temp: float):
            async def local_llm(prompt: str) -> str:
                return chat.generate_user(prompt, args.evaluator_max_tokens, stats,
                                          temperature=temp)
            return local_llm

        return make(0.0), make(args.proposer_temperature)
    return (make_evaluator_llm(args, stats, temperature=0.0),
            make_evaluator_llm(args, stats, temperature=args.proposer_temperature))


def make_evaluator_llm(args: argparse.Namespace, stats: Stats, temperature: float = 0.0):
    """gemini-2.5-pro 经 OpenAI 兼容端点（复用框架 API 后端；to_thread 提供并发）。

    temperature：比较端 0.0 / 反思提议端 0.7（原版 _propose_new_prompt 的分工）。
    """
    from lychee_mas.backends.openai_api_backend import OpenAICompatibleBackend

    backend = OpenAICompatibleBackend(
        args.evaluator_model, base_url=args.evaluator_base_url,
        api_key_env=args.evaluator_api_key_env, timeout=args.evaluator_timeout,
        temperature=temperature)

    def call(prompt: str) -> str:
        # 原版 utils.async_retry 同款语义：5 次指数退避；额度类错误重试也救不了，
        # 最终如实上抛（断点续跑靠 optimizer 的 checkpoint）
        delay = 0.5
        for attempt in range(5):
            try:
                g = backend.generate_chat([{"role": "user", "content": prompt}],
                                          max_new_tokens=args.evaluator_max_tokens)
                stats.add(g.n_prompt_pos, g.n_gen_tokens, g.latency_s)
                return g.text
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(min(30.0, delay))
                delay *= 2
        raise RuntimeError("unreachable")

    async def evaluator_llm(prompt: str) -> str:
        return await asyncio.to_thread(call, prompt)

    return evaluator_llm


# ===================================== 评测 =====================================

def run_eval(graph: Any, records: List[Dict[str, Any]], method: str,
             args: argparse.Namespace, stats: Stats, extra_config: Dict[str, Any]) -> None:
    app = graph.compile()
    terminal = [n for n in graph.nodes][-1]
    samples: List[Dict[str, Any]] = []
    for i, rec in enumerate(records):
        before = stats.snapshot()
        result = asyncio.run(app.ainvoke(
            {"question": rec["question"], "raw": {}, "short": {}}))
        pred = extract_answer(result["raw"][terminal])
        score = M.score(rec["kind"], pred, rec["gold"])
        after = stats.snapshot()
        samples.append({
            "case_id": str(i), "task": rec["task"], "kind": rec["kind"], "method": method,
            "question": rec["question"], "gold": rec["gold"], "prediction": pred,
            "score": score, "is_correct": bool(score == 1.0),
            "model_call_count": after["model_calls"] - before["model_calls"],
            "input_positions_total": after["prompt_tokens"] - before["prompt_tokens"],
            "gen_tokens": after["completion_tokens"] - before["completion_tokens"],
            "latency_s": round(after["latency_s"] - before["latency_s"], 3),
        })
        print(f"  [{i + 1}/{len(records)}] score={score:.0f} pred={pred[:40]!r} "
              f"gold={rec['gold'][:40]!r}", flush=True)

    model_tag = args.model_tag or os.path.basename(os.path.normpath(args.model_path))
    out_dir = M.result_dir(model_tag, method, args.task, root=args.out_root)
    summary = M.aggregate_samples(samples, {"task": args.task, "method": method,
                                            "scorer_kind": "aime", "model": model_tag})
    config = {"script": "run_maspo_langgraph.py", "method": method, "task": args.task,
              "nr": args.nr, "eval_n": len(samples), "seed": args.seed,
              "model_path": args.model_path, "max_new_tokens": args.max_new_tokens,
              "evaluator_model": args.evaluator_model_path or args.evaluator_model,
              "reference": "https://github.com/wangzx1219/MASPO (ICML 2026)",
              **extra_config}
    M.write_results(out_dir, samples, summary, config)
    acc = summary.get("accuracy", summary.get("mean_score"))
    print(f"[eval:{method}] accuracy={acc} out_dir={out_dir}", flush=True)


# ===================================== 主流程 =====================================

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--phase", default="both",
                    choices=("baseline", "optimize", "eval", "both"),
                    help="baseline=原始提示跑分；optimize=联合优化落 prompt JSON；"
                         "eval=挂载优化提示跑分；both=optimize+eval")
    ap.add_argument("--task", default="math500", help="benchmark 注册名（默认 math500）")
    ap.add_argument("--nr", type=int, default=1, help="reflect 轮数（原版 --nr）")
    ap.add_argument("--train-n", type=int, default=50,
                    help="优化用训练问题数（原版 load_train_for_opt k=50，采样自同一数据集）")
    ap.add_argument("--eval-n", type=int, default=100, help="评测题数（0=全集）")
    ap.add_argument("--prompt-file", default=None,
                    help="优化产物 JSON（optimize 写 / eval 读）；"
                         "默认 <out-root>/maspo_<task>_prompts.json")
    # 优化超参（默认=原版论文模式）
    ap.add_argument("--depth", type=int, default=9, help="max_total_depth（原版 9）")
    ap.add_argument("--rounds-per-turn", type=int, default=3)
    ap.add_argument("--beam-width", type=int, default=2)
    ap.add_argument("--eval-batch", type=int, default=10)
    ap.add_argument("--lookahead-weights", default="4:4:2")
    ap.add_argument("--no-beam-refresh", action="store_true")
    ap.add_argument("--no-lookahead-score", action="store_true")
    ap.add_argument("--no-misleading-sampling", action="store_true")
    ap.add_argument("--feedback", action="store_true", help="下游反馈（原版论文模式未开）")
    # 执行 LLM（MASPO 同款 Qwen3-8B，本地权重）
    ap.add_argument("--model-path", default=os.environ.get("LYCHEE_HF_MODEL", DEFAULT_MODEL))
    ap.add_argument("--model-tag", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--max-new-tokens", type=int, default=4096,
                    help="执行端生成上限（原版 async_call_llm max_tokens=4096）")
    # 评估/反思 LLM（MASPO 同款 gemini-2.5-pro，OpenAI 兼容端点）
    ap.add_argument("--evaluator-model", default=DEFAULT_EVALUATOR_MODEL)
    ap.add_argument("--evaluator-model-path", default=None,
                    help="本地 HF 评估/反思模型路径（如 Qwen3-32B）；给出则不走 API")
    ap.add_argument("--evaluator-device", default="cuda:1",
                    help="本地评估模型的设备（与执行端 --device 分卡）")
    ap.add_argument("--evaluator-dtype", default="bfloat16")
    ap.add_argument("--evaluator-base-url",
                    default=os.environ.get("EVALUATOR_BASE_URL"))
    ap.add_argument("--evaluator-api-key-env", default="EVALUATOR_API_KEY")
    ap.add_argument("--evaluator-max-tokens", type=int, default=16384,
                    help="原版评估端 max_tokens=16384")
    ap.add_argument("--evaluator-timeout", type=float, default=180.0)
    ap.add_argument("--proposer-temperature", type=float, default=0.7,
                    help="反思提议端采样温度（原版 0.7；比较端恒 0.0）")
    ap.add_argument("--max-concurrency", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-root", default=None)
    args = ap.parse_args()

    if not args.model_path:
        raise SystemExit("需要 --model-path 或环境变量 LYCHEE_HF_MODEL（不做静默兜底）")
    prompt_file = args.prompt_file or os.path.join(
        args.out_root or os.path.join("runs", "maspo"), f"maspo_{args.task}_prompts.json")
    try:
        w = [float(x) for x in args.lookahead_weights.split(":")]
    except ValueError as exc:
        raise SystemExit(f"--lookahead-weights 需形如 4:4:2，得到 {args.lookahead_weights!r}"
                         f"（{exc}）")

    if args.phase in ("optimize", "both") and args.evaluator_model_path \
            and not os.path.isdir(args.evaluator_model_path):
        raise SystemExit(f"--evaluator-model-path 不存在: {args.evaluator_model_path}")
    if args.phase in ("optimize", "both") and not args.evaluator_model_path:
        # 快速失败（API 路径）：评估/反思端点缺配置就别等 8B 模型加载完才报错
        if not os.environ.get(args.evaluator_api_key_env):
            raise SystemExit(f"phase={args.phase} 需要评估端 API key：请设环境变量 "
                             f"{args.evaluator_api_key_env}（gemini-2.5-pro 端点）")
        if not args.evaluator_base_url:
            raise SystemExit("phase 含 optimize 需要 --evaluator-base-url "
                             "或环境变量 EVALUATOR_BASE_URL（OpenAI 兼容端点）")

    records = load_benchmark(args.task, n=None)
    eval_records = records[: args.eval_n] if args.eval_n else records
    import torch as _torch

    _torch.manual_seed(args.seed)  # 反思端采样（temperature>0）的进程级种子
    chat = HFChat(args.model_path, device=args.device, dtype=args.dtype)

    if args.phase == "baseline":
        stats = Stats()
        graph = build_reflect_graph(chat, args.nr, stats, args.max_new_tokens)
        run_eval(graph, eval_records, "maspo_original", args, stats,
                 {"prompts": "seed (vendored MASPO defaults)"})
        return

    if args.phase in ("optimize", "both"):
        import random as _random

        rng = _random.Random(args.seed)
        trainset = rng.sample([r["question"] for r in records],
                              min(args.train_n, len(records)))
        agent_stats, eval_stats = Stats(), Stats()
        evaluator_llm, proposer_llm = make_evaluator_llms(args, eval_stats)
        graph = build_reflect_graph(chat, args.nr, agent_stats, args.max_new_tokens)
        t0 = time.time()
        # ★ 本框架运行前优化统一接口：在 LangGraph 图上跑 MASPO 联合优化，产物落 prompt JSON
        optimize_langgraph(
            graph, method="maspo", mode="optimize", trainset=trainset,
            agent_llm=make_agent_llm(chat, agent_stats, args.max_new_tokens),
            evaluator_llm=evaluator_llm, proposer_llm=proposer_llm,
            prompt_file=prompt_file, max_total_depth=args.depth,
            rounds_per_turn=args.rounds_per_turn, beam_width=args.beam_width,
            eval_batch=args.eval_batch, lookahead_weights=tuple(w),
            use_beam_refresh=not args.no_beam_refresh,
            use_lookahead_score=not args.no_lookahead_score,
            use_misleading_sampling=not args.no_misleading_sampling,
            use_feedback=args.feedback, seed=args.seed,
            max_concurrency=args.max_concurrency)
        print(f"[optimize] done in {time.time() - t0:.0f}s -> {prompt_file}\n"
              f"  agent LLM:     {agent_stats.snapshot()}\n"
              f"  evaluator LLM: {eval_stats.snapshot()}", flush=True)

    if args.phase in ("eval", "both"):
        if not os.path.isfile(prompt_file):
            raise SystemExit(f"eval 需要优化产物 {prompt_file}（先跑 --phase optimize）")
        stats = Stats()
        graph = build_reflect_graph(chat, args.nr, stats, args.max_new_tokens)
        # ★ 即插即用：mode="apply" 加载 prompt JSON 注入图，再照常编译评测
        graph = optimize_langgraph(graph, method="maspo", mode="apply",
                                   prompt_file=prompt_file)
        run_eval(graph, eval_records, "maspo_optimized", args, stats,
                 {"prompt_file": prompt_file})


if __name__ == "__main__":
    main()
