"""复现 AgentPrune 的 GSM8K 实验（LangGraph 执行 + 本框架运行前组件挂载）。

对标：Cut the Crap: An Economical Communication Pipeline for LLM-based Multi-Agent
Systems（ICLR 2025）；参考实现 https://github.com/yanweiyue/AgentPrune 的
experiments/run_gsm8k.py（FullConnected 模式，4 个 MathSolver 节点 + FinalRefer 决策）。

与原版逐项对应：
  agent 配置    4 agents，角色轮转 [Math Solver, Mathematical Analyst, Programming
                Expert, Inspector]；prompt/few-shot 逐字取自原仓库
                （vendored 于 scripts/agentprune_gsm8k_prompts.py）
  通信图        FullConnected：空间边 i≠j 全 1（可训练），时间边全 1；每 query 伯努利
                采样实现图，按拓扑序执行（graph_pruner/agentprune 实现同款机制）
  训练          REINFORCE：loss = -log_prob × utility（utility=该题对错），Adam lr=0.1，
                batch_size=4；每 imp_per_iterations=5 个 batch one-shot 剪枝
                pruning_rate=0.25；共 num_iterations=10 个 batch 训练
  决策          FinalRefer：汇总全部 agent 输出出最终答案（system/user 拼法逐字复刻）
  打分          gsm_get_predict（原版抽取）+ float 相等比较

与原版的声明差异：
  - LLM 为本地 HF 模型（原版 gpt-4-1106-preview）；绝对准确率不可比，
    对比对象是「同模型下 FullConnected vs 剪枝后」的准确率与 token 成本。
  - 执行引擎为 LangGraph StateGraph（按采样实现图的拓扑序连线性链 + 决策节点）。
  - 评测阶段直接加载训练好的 `methods/prerun/agentprune`（state_file → threshold
    确定性实现）驱动 LangGraph 执行；图级统一接口挂载（optimize_langgraph(
    method="agentprune")）见 run_maspo_langgraph.py 与 tests/test_prerun.py。
    --eval-mode sample 可切回原版的采样式推理。

用法（train 落 pruner 状态 → eval 挂插件跑分；也可分阶段跑）：
  CDM_DATA_ROOT=/data/.../raw CUDA_VISIBLE_DEVICES=0 \
      python scripts/run_agentprune_gsm8k.py --phase both \
      --model-path /data/mxy/Models/Qwen/Qwen3-4B \
      --train-n 40 --eval-n 40
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, TypedDict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import agentprune_gsm8k_prompts as P  # noqa: E402  vendored 原版 prompt 资产
from lychee_mas.core.types import AgentSpec  # noqa: E402
from lychee_mas.eval import metrics as M  # noqa: E402
from lychee_mas.eval.benchmarks import load as load_benchmark  # noqa: E402
from lychee_mas.methods.prerun.agentprune import AgentPrunePruner  # noqa: E402
from lychee_mas.methods.prerun.graphops import (  # noqa: E402
    Realization,
    topological_order,
)

QUESTION_SUFFIX = "\nGive the final numeric answer."  # 本框架 loader 附加，复现时剥掉
N_AGENTS = 4
AGENT_ROLES = list(P.GSM8K_ROLES)  # 原版 roles cycle，N=4 恰好各一个


class ChainState(TypedDict):
    """LangGraph 状态：outputs[i] = agent i 本轮输出；final = FinalRefer 输出。"""

    task: str
    outputs: Dict[int, str]
    final: str


# ================== 生成后端：transformers 直连（同级脚本同款） ==================

@dataclass
class GenOut:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float


class HFChat:
    def __init__(self, model_path: str, device: str = "cuda:0", dtype: str = "bfloat16"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=getattr(torch, dtype)).to(device).eval()
        self.device = device

    def generate(self, messages: List[Dict[str, str]], max_new_tokens: int) -> GenOut:
        import torch

        try:
            text = self.tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            text = self.tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tok(text, return_tensors="pt").to(self.device)
        n_prompt = int(inputs["input_ids"].shape[1])
        t0 = time.time()
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=self.tok.eos_token_id)
        latency = time.time() - t0
        gen_ids = out[0][n_prompt:]
        return GenOut(text=self.tok.decode(gen_ids, skip_special_tokens=True).strip(),
                      prompt_tokens=n_prompt, completion_tokens=int(gen_ids.shape[0]),
                      latency_s=round(latency, 3))


def execute_python_code(code: str, timeout: int = 10) -> str:
    """原版 Programming Expert 路径：跑生成代码取 answer 变量（子进程 + 超时）。"""
    payload = code + "\nprint(repr(answer))\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(payload)
        path = f.name
    try:
        proc = subprocess.run([sys.executable, path], capture_output=True, text=True,
                              timeout=timeout)
        if proc.returncode != 0:
            return f"<execution error: {proc.stderr.strip()[:200]}>"
        return proc.stdout.strip()
    except subprocess.TimeoutExpired:
        return "<execution timeout>"
    finally:
        os.unlink(path)


# ================== 消息拼装（逐字复刻 math_solver.py / final_decision.py） ==================

def agent_messages(role: str, question: str, spatial_info: Dict[str, Dict[str, str]],
                   temporal_info: Dict[str, Dict[str, str]]) -> List[Dict[str, str]]:
    system_prompt = P.ROLE_DESCRIPTION[role]
    user_prompt = P.get_answer_prompt(question=question, role=role)
    if role == "Math Solver":
        user_prompt += "(Hint: The answer is near to"
        for _id, info in spatial_info.items():
            user_prompt += " " + P.gsm_get_predict(info["output"])
        for _id, info in temporal_info.items():
            user_prompt += " " + P.gsm_get_predict(info["output"])
        user_prompt += ")."
    else:
        spatial_str = ""
        temporal_str = ""
        for aid, info in spatial_info.items():
            spatial_str += (f"Agent {aid} as a {info['role']} his answer to this question "
                            f"is:\n\n{info['output']}\n\n")
        for aid, info in temporal_info.items():
            temporal_str += (f"Agent {aid} as a {info['role']} his answer to this question "
                             f"was:\n\n{info['output']}\n\n")
        if spatial_str:
            user_prompt += ("At the same time, there are the following responses to the same "
                            f"question for your reference:\n\n{spatial_str} \n\n")
        if temporal_str:
            user_prompt += ("In the last round of dialogue, there were the following responses "
                            f"to the same question for your reference: \n\n{temporal_str}")
    return [{"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}]


def decision_messages(question: str, outputs: Dict[int, str]) -> List[Dict[str, str]]:
    system_prompt = f"{P.DECISION_ROLE}.\n {P.DECISION_CONSTRAINT}"
    spatial_str = ""
    for idx in sorted(outputs):
        spatial_str += f"A{idx}: {outputs[idx]}\n\n"
    user_prompt = (f"{P.DECISION_FEW_SHOT} The task is:\n\n {question}.\n At the same time, "
                   f"the output of other agents is as follows:\n\n{spatial_str}")
    return [{"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}]


# ================== LangGraph 执行：实现图 → 线性链（拓扑序）+ 决策节点 ==================

@dataclass
class QueryStats:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model_calls: int = 0
    latency_s: float = 0.0


def build_round_app(chat: HFChat, order: List[int], final_preds: Dict[int, set],
                    prev_outputs: Dict[int, str], temporal_edges: set,
                    max_new_tokens: int, stats: QueryStats) -> Any:
    """一轮 = 按拓扑序的线性链 + FinalRefer 决策节点（LangGraph StateGraph）。"""
    from langgraph.graph import END, START, StateGraph

    def make_agent_node(idx: int):
        role = AGENT_ROLES[idx]

        async def node(state: ChainState) -> Dict[str, Any]:
            spatial_info = {f"A{j}": {"role": AGENT_ROLES[j], "output": state["outputs"][j]}
                            for j in sorted(final_preds[idx]) if j in state["outputs"]}
            temporal_info = {f"A{j}": {"role": AGENT_ROLES[j], "output": prev_outputs[j]}
                             for j in sorted({s for (s, d) in temporal_edges if d == idx})
                             if j in prev_outputs}
            msgs = agent_messages(role, state["task"], spatial_info, temporal_info)
            g = chat.generate(msgs, max_new_tokens=max_new_tokens)
            text = g.text
            if role == "Programming Expert":
                # 原版：跑生成代码，把返回值以 "the answer is X" 附在响应后
                code = text.lstrip("```python\n").rstrip("\n```")
                answer = execute_python_code(code)
                text += f"\nthe answer is {answer}"
            stats.prompt_tokens += g.prompt_tokens
            stats.completion_tokens += g.completion_tokens
            stats.model_calls += 1
            stats.latency_s += g.latency_s
            return {"outputs": {**state["outputs"], idx: text}}

        return node

    async def decision_node(state: ChainState) -> Dict[str, Any]:
        msgs = decision_messages(state["task"], state["outputs"])
        g = chat.generate(msgs, max_new_tokens=max_new_tokens)
        stats.prompt_tokens += g.prompt_tokens
        stats.completion_tokens += g.completion_tokens
        stats.model_calls += 1
        stats.latency_s += g.latency_s
        return {"final": g.text}

    sg = StateGraph(ChainState)
    names = [f"agent_{i}" for i in order]
    for i in order:
        sg.add_node(f"agent_{i}", make_agent_node(i))
    sg.add_node("final_refer", decision_node)
    sg.add_edge(START, names[0])
    for a, b in zip(names, names[1:]):
        sg.add_edge(a, b)
    sg.add_edge(names[-1], "final_refer")
    sg.add_edge("final_refer", END)
    return sg.compile()


async def run_query(chat: HFChat, question: str, realizations: List[Realization],
                    max_new_tokens: int) -> Tuple[str, Dict[int, str], QueryStats]:
    """按每轮的实现图执行 num_rounds 轮，末轮后过 FinalRefer。"""
    stats = QueryStats()
    prev_outputs: Dict[int, str] = {}
    final = ""
    outputs: Dict[int, str] = {}
    for r, real in enumerate(realizations):
        order, final_preds = topological_order(N_AGENTS, real.spatial_edges)
        is_last = r == len(realizations) - 1
        app = build_round_app(chat, order, final_preds, prev_outputs,
                              real.temporal_edges, max_new_tokens, stats)
        state: ChainState = {"task": question, "outputs": {}, "final": ""}
        if not is_last:
            # 非末轮：跑到最后一个 agent 即可（决策节点只在末轮有意义）；
            # 为保持图结构一致仍执行决策节点，但丢弃其输出（原版每轮都产 decision，
            # 只取最后一轮）。
            pass
        result = await app.ainvoke(state, config={"recursion_limit": 4 * N_AGENTS + 10})
        outputs = result["outputs"]
        final = result["final"]
        prev_outputs = dict(outputs)
    return final, outputs, stats


def utility_of(final_answer: str, gold: str) -> float:
    pred = P.gsm_get_predict(final_answer)
    try:
        return float(float(pred) == float(gold))
    except (TypeError, ValueError):
        return 0.0  # 原版 float() 失败即算错（不抛：pred 可能为空串）


def load_gsm8k_records(n: int, skip: int = 0) -> List[Dict[str, Any]]:
    records = load_benchmark("gsm8k", n=(skip + n) or None)[skip:]
    if not records:
        raise SystemExit("gsm8k 加载为空：请先准备数据（benchmarks.prepare('gsm8k')）")
    for rec in records:
        rec["question"] = rec["question"].removesuffix(QUESTION_SUFFIX)
    return records


# ====================== 训练阶段（REINFORCE + one-shot 剪枝） ======================

def train(args: argparse.Namespace, chat: HFChat) -> str:
    records = load_gsm8k_records(args.train_n)
    pruner = AgentPrunePruner(
        n_agents=N_AGENTS, lr=args.lr, seed=args.seed,
        optimized_spatial=True, optimized_temporal=(args.num_rounds > 1))
    alive0 = sum(v for (i, j), v in pruner.spatial_masks.items() if i != j)
    num_batches = len(records) // args.batch_size
    if num_batches < 1:
        raise SystemExit(f"训练数据不足一个 batch（{len(records)} < {args.batch_size}）")
    log: List[Dict[str, Any]] = []
    solved = 0

    for i_batch in range(num_batches):
        batch = records[i_batch * args.batch_size:(i_batch + 1) * args.batch_size]
        grad_batch: List[Tuple[List[Realization], float]] = []
        for rec in batch:
            reals = [pruner.sample_realization(include_temporal=(r > 0))
                     for r in range(args.num_rounds)]
            final, _outputs, stats = asyncio.run(
                run_query(chat, rec["question"], reals, args.max_new_tokens))
            u = utility_of(final, rec["gold"])
            solved += int(u)
            grad_batch.append((reals, u))
            log.append({"phase": "train", "batch": i_batch, "utility": u,
                        "prompt_tokens": stats.prompt_tokens,
                        "completion_tokens": stats.completion_tokens,
                        "pred": P.gsm_get_predict(final), "gold": rec["gold"]})
        pruner.reinforce(grad_batch)
        if (i_batch + 1) % args.imp_per_iterations == 0:
            ks, kt = pruner.update_masks(args.pruning_rate)
            alive = sum(v for (i, j), v in pruner.spatial_masks.items() if i != j)
            print(f"[train] batch {i_batch + 1}: pruned spatial={ks} temporal={kt} "
                  f"alive_spatial={alive}/{alive0}")
        acc = solved / len(log)
        print(f"[train] batch {i_batch + 1}/{num_batches} running_acc={acc:.3f}")

    state_path = os.path.join(args.out_root or ".", "agentprune_gsm8k_state.json")
    os.makedirs(os.path.dirname(state_path) or ".", exist_ok=True)
    pruner.save(state_path)
    with open(state_path.replace("_state.json", "_train_log.json"), "w") as f:
        json.dump(log, f, indent=2)
    alive = sum(v for (i, j), v in pruner.spatial_masks.items() if i != j)
    print(f"[train] done: state -> {state_path}  alive_spatial={alive}/{alive0}")
    return state_path


# ====================== 评测阶段（运行前插件挂载剪枝图） ======================

def evaluate(args: argparse.Namespace, chat: HFChat, state_file: Optional[str]) -> None:
    records = load_gsm8k_records(args.eval_n, skip=args.train_n)

    if state_file:
        # ★ 加载训练产物：threshold 确定性实现 / sample 采样式（原版）
        pruner = AgentPrunePruner(n_agents=N_AGENTS, state_file=state_file,
                                  optimized_spatial=True,
                                  optimized_temporal=(args.num_rounds > 1))
        method = "agentprune_pruned"
    else:
        pruner = None  # 对照：FullConnected 全图
        method = "agentprune_full"

    samples: List[Dict[str, Any]] = []
    for i, rec in enumerate(records):
        if pruner is not None:
            sm, tm = pruner.realized_matrices("threshold")
            if args.eval_mode == "threshold":
                spatial = {(a, b) for a in range(N_AGENTS) for b in range(N_AGENTS)
                           if sm[a][b]}
                temporal = {(a, b) for a in range(N_AGENTS) for b in range(N_AGENTS)
                            if tm[a][b]}
                reals = [Realization(spatial_edges=spatial,
                                     temporal_edges=(temporal if r > 0 else set()))
                         for r in range(args.num_rounds)]
            else:  # sample：原版式采样推理（用训练后的 logits/masks）
                reals = [pruner.sample_realization(include_temporal=(r > 0))
                         for r in range(args.num_rounds)]
        else:
            full = {(a, b) for a in range(N_AGENTS) for b in range(N_AGENTS) if a != b}
            allt = {(a, b) for a in range(N_AGENTS) for b in range(N_AGENTS)}
            reals = [Realization(spatial_edges=set(full),
                                 temporal_edges=(allt if r > 0 else set()))
                     for r in range(args.num_rounds)]

        final, _outputs, stats = asyncio.run(
            run_query(chat, rec["question"], reals, args.max_new_tokens))
        pred = P.gsm_get_predict(final)
        score = utility_of(final, rec["gold"])
        samples.append({
            "case_id": str(i), "task": "gsm8k", "kind": "exact", "method": method,
            "question": rec["question"], "gold": rec["gold"], "prediction": pred,
            "score": score, "is_correct": bool(score == 1.0),
            "spatial_edges": sorted(list(reals[-1].spatial_edges)),
            "model_call_count": stats.model_calls,
            "input_positions_total": stats.prompt_tokens,
            "gen_tokens": stats.completion_tokens,
            "latency_s": round(stats.latency_s, 3),
        })
        print(f"  [{i + 1}/{len(records)}] score={score:.0f} "
              f"edges={len(reals[-1].spatial_edges)} calls={stats.model_calls} "
              f"prompt_toks={stats.prompt_tokens} pred={pred!r}")

    model_tag = args.model_tag or os.path.basename(os.path.normpath(args.model_path))
    out_dir = M.result_dir(model_tag, method, "gsm8k", root=args.out_root)
    summary = M.aggregate_samples(samples, {"task": "gsm8k", "method": method,
                                            "scorer_kind": "exact", "model": model_tag})
    config = {
        "script": "run_agentprune_gsm8k.py", "method": method,
        "state_file": state_file, "eval_mode": args.eval_mode,
        "n_agents": N_AGENTS, "roles": AGENT_ROLES,
        "num_rounds": args.num_rounds, "batch_size": args.batch_size, "lr": args.lr,
        "imp_per_iterations": args.imp_per_iterations, "pruning_rate": args.pruning_rate,
        "train_n": args.train_n, "eval_n": len(samples), "seed": args.seed,
        "model_path": args.model_path, "max_new_tokens": args.max_new_tokens,
        "reference": "https://github.com/yanweiyue/AgentPrune (ICLR 2025)",
    }
    M.write_results(out_dir, samples, summary, config)
    acc = summary.get("accuracy", summary.get("mean_score"))
    mean_prompt = sum(s["input_positions_total"] for s in samples) / len(samples)
    print(f"[eval:{method}] accuracy={acc} mean_prompt_tokens={mean_prompt:.0f} "
          f"out_dir={out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--phase", default="both", choices=("train", "eval", "both"))
    ap.add_argument("--state-file", default=None,
                    help="eval 阶段加载的 pruner 状态（train 阶段产出；不给且 phase=eval "
                         "则跑 FullConnected 对照）")
    ap.add_argument("--train-n", type=int, default=40, help="训练用题数（原版 10 batch × 4）")
    ap.add_argument("--eval-n", type=int, default=40, help="评测题数（取训练集之后的题）")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--imp-per-iterations", type=int, default=5)
    ap.add_argument("--pruning-rate", type=float, default=0.25)
    ap.add_argument("--num-rounds", type=int, default=1)
    ap.add_argument("--eval-mode", default="threshold", choices=("threshold", "sample"),
                    help="threshold=确定性剪枝图（插件 before_run 产物）；sample=原版采样式")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model-path", default=os.environ.get("LYCHEE_HF_MODEL"))
    ap.add_argument("--model-tag", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--out-root", default=None)
    args = ap.parse_args()

    if not args.model_path:
        raise SystemExit("需要 --model-path 或环境变量 LYCHEE_HF_MODEL（不做静默兜底）")

    chat = HFChat(args.model_path, device=args.device, dtype=args.dtype)
    state_file = args.state_file
    if args.phase in ("train", "both"):
        state_file = train(args, chat)
    if args.phase in ("eval", "both"):
        evaluate(args, chat, state_file)


if __name__ == "__main__":
    main()
