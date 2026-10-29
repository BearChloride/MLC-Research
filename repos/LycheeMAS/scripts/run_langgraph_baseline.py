"""基于原生 LangGraph 的独立基线运行脚本（不使用 lychee_mas 运行时/记忆/插件套件）。

只复用**评测模块**（`lychee_mas.eval`：benchmark 加载 + 答案抽取 + 评分 + 落盘），
执行侧全部用原生 LangGraph API + transformers 直连实现，作为框架无关的对照基线。

五个环节（对应本文件五个区块）：
  ① Benchmark 选择   eval.benchmarks.load(task, n)（19 个已接入任务按名选择）
  ② 网络构建         原生 StateGraph：单 agent 或 planner→solver→verifier 顺序链，
                     条件边终止（输出含 "APPROVE" 或达 len(agents)×max_rounds）
  ③ 执行             app.ainvoke 逐 case 跑（deterministic：do_sample=False）
  ④ 评测             eval.task_config.extractor_for_task 抽答案 + eval.metrics.score 打分
  ⑤ 落盘             outputs.jsonl + metrics.json + config.yaml
                     （eval.metrics.write_results / aggregate_samples，字段与 run_mas 同口径）

用法：
  CDM_DATA_ROOT=/data/.../raw CUDA_VISIBLE_DEVICES=0 \
      python scripts/run_langgraph_baseline.py --task gsm8k --n 5 \
      --model-path /data/mxy/Models/Qwen/Qwen3-4B
  python scripts/run_langgraph_baseline.py --task aime_2024 --n 3 --team single

依赖：langgraph + torch/transformers（conda env CDM）；无 GPU/数据时显式报错，不静默降级。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, TypedDict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "src"))

from lychee_mas.eval import metrics as M  # noqa: E402
from lychee_mas.eval.benchmarks import LOADERS  # noqa: E402
from lychee_mas.eval.benchmarks import load as load_benchmark
from lychee_mas.eval.sampling import sample_records  # noqa: E402
from lychee_mas.eval.task_config import extractor_for_task  # noqa: E402

# ====================== ② 网络构建：agent 画像（脚本内自包含，不用套件模板） ======================

APPROVE_RULE = (
    'When you are confident in the final answer, end your reply with a line '
    '"APPROVE: <final answer>" (for math, put the answer inside \\boxed{...}).'
)

CHAIN3_AGENTS: List[tuple[str, str]] = [
    ("planner",
     "You are the planner. Restate the problem precisely and lay out a short, "
     "numbered solution plan. Do not compute the final answer yourself."),
    ("solver",
     "You are the solver. Follow the planner's plan step by step and derive the answer. "
     "Show the key computations compactly."),
    ("verifier",
     "You are the verifier. Check the solver's derivation for errors. If it is correct, "
     "confirm the answer; if not, correct it. " + APPROVE_RULE),
]

SINGLE_AGENT: List[tuple[str, str]] = [
    ("solver",
     "You are a careful problem solver. Solve the task step by step, then conclude. "
     + APPROVE_RULE),
]

TEAMS: Dict[str, List[tuple[str, str]]] = {"chain3": CHAIN3_AGENTS, "single": SINGLE_AGENT}


class BaselineState(TypedDict):
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
    hit_token_limit: bool


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
        hit_token_limit = (
            int(gen_ids.shape[0]) >= max_new_tokens
            and (self.tok.eos_token_id is None
                 or int(gen_ids[-1]) != int(self.tok.eos_token_id))
        )
        return GenOut(text=text.strip(), prompt_tokens=n_prompt,
                      completion_tokens=int(gen_ids.shape[0]), latency_s=round(latency, 3),
                      hit_token_limit=hit_token_limit)


class APIChat:
    """OpenAI-compatible chat backend（仅用标准库 urllib，不新增依赖）。

    面向组内 vLLM 推理服务，例如:
        base_url = http://10.249.45.207:6666/v1
        model    = Qwen3ForCausalLM
        api_key  = EMPTY

    Qwen3 系列默认输出 ``<think>`` 推理块，评测基线需要最终答案，故默认在请求里传
    ``chat_template_kwargs={"enable_thinking": False}`` 关闭思维链；对不支持该参数的
    服务可改用 ``--api-enable-thinking`` 关闭这个开关（用原生行为）。
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: str = "EMPTY",
        timeout: float = 180.0,
        enable_thinking: bool = False,
    ) -> None:
        self.endpoint = self._endpoint_url(base_url)
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.enable_thinking = enable_thinking

    @staticmethod
    def _endpoint_url(base_url: str) -> str:
        url = base_url.rstrip("/")
        if not url.endswith("/v1"):
            url += "/v1"
        return url + "/chat/completions"

    def generate(self, messages: List[Dict[str, str]], max_new_tokens: int) -> GenOut:
        import urllib.error
        import urllib.request

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_new_tokens,
            "temperature": 0.0,
        }
        if not self.enable_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        t0 = time.time()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")
            raise RuntimeError(
                f"API request failed with HTTP {exc.code}: {detail[:500]}"
            ) from exc
        latency = time.time() - t0

        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        text = (message.get("content") or "").strip()
        if not text and message.get("reasoning_content"):
            text = str(message["reasoning_content"]).strip()
        usage = data.get("usage") or {}
        finish_reason = choice.get("finish_reason")
        return GenOut(
            text=text,
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            latency_s=round(latency, 3),
            hit_token_limit=(finish_reason == "length"),
        )


# ====================== ② 网络构建：原生 StateGraph ======================

def build_app(agents: List[tuple[str, str]], chat: HFChat, max_turns: int,
              max_new_tokens: int) -> Any:
    """把 agent 列表连成顺序轮转的 StateGraph；APPROVE 或轮数上限终止。"""
    from langgraph.graph import END, START, StateGraph

    names = [name for name, _ in agents]

    def make_node(name: str, system_prompt: str) -> Callable[[BaselineState],
                                                             Awaitable[Dict[str, Any]]]:
        async def node(state: BaselineState) -> Dict[str, Any]:
            msgs: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
            for entry in state["history"]:
                role = "assistant" if entry["source"] == name else "user"
                msgs.append({"role": role, "content": entry["content"]})
            g = chat.generate(msgs, max_new_tokens=max_new_tokens)
            entry = {"source": name, "content": g.text,
                     "prompt_tokens": g.prompt_tokens, "completion_tokens": g.completion_tokens,
                     "latency_s": g.latency_s,
                     "generation_stop_reason": "length" if g.hit_token_limit else "eos_or_other"}
            return {"history": state["history"] + [entry],
                    "agent_turns": state["agent_turns"] + 1,
                    "done": state["done"] or ("APPROVE" in g.text)}

        return node

    def route(state: BaselineState) -> str:
        return "end" if (state["done"] or state["agent_turns"] >= max_turns) else "continue"

    graph = StateGraph(BaselineState)
    for name, prompt in agents:
        graph.add_node(name, make_node(name, prompt))
    graph.add_edge(START, names[0])
    for i, name in enumerate(names):
        graph.add_conditional_edges(name, route,
                                    {"continue": names[(i + 1) % len(names)], "end": END})
    return graph.compile()


# ====================== ③ 执行 + ④ 评测（单 case） ======================

async def run_case(app: Any, task: str, record: Dict[str, Any], case_id: str,
                   max_turns: int, method_label: str = "langgraph_baseline") -> Dict[str, Any]:
    question = record["question"]
    if record.get("context"):
        question = f"{record['context']}\n\n{question}"  # 无记忆套件：上下文直接并入任务文本
    state: BaselineState = {
        "history": [{"source": "user", "content": question,
                     "prompt_tokens": 0, "completion_tokens": 0, "latency_s": 0.0}],
        "agent_turns": 0,
        "done": False,
    }
    t0 = time.time()
    final = await app.ainvoke(state, config={"recursion_limit": 2 * max_turns + 10})
    wall = time.time() - t0

    history = final["history"]
    prediction = extractor_for_task(task)(history)  # eval 模块的答案抽取（兼容 dict 消息）
    kind = record.get("kind") or "exact"
    score = M.score(kind, prediction, record.get("gold"))

    return {
        # 与 run_mas / aggregate_samples 同口径的字段名
        "case_id": case_id,
        "task": task,
        "kind": kind,
        "method": method_label,
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
        "length_truncation_calls": sum(
            e.get("generation_stop_reason") == "length" for e in history
        ),
        "length_truncated": any(
            e.get("generation_stop_reason") == "length" for e in history
        ),
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


def runtime_metadata(weights_path: str | None) -> Dict[str, Any]:
    """收集解释器/框架/GPU/权重信息，API 模式或缺少 torch 时不崩溃。"""
    meta: Dict[str, Any] = {
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
    }
    for package in ("torch", "transformers"):
        try:
            module = __import__(package)
            meta[f"{package}_version"] = str(getattr(module, "__version__", "unknown"))
        except Exception:
            meta[f"{package}_version"] = None
    try:
        import torch

        meta["torch_cuda_version"] = torch.version.cuda
        meta["cuda_available"] = bool(torch.cuda.is_available())
        meta["gpu_name"] = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        )
    except Exception:
        meta["torch_cuda_version"] = None
        meta["cuda_available"] = None
        meta["gpu_name"] = None
    meta["weights_path"] = weights_path
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--task", required=True, choices=sorted(LOADERS),
                    help="benchmark 名（eval/benchmarks 已接入的 19 个）")
    ap.add_argument("--n", type=int, default=3, help="样本数（0=全量）")
    ap.add_argument("--sample-seed", type=int, default=0,
                    help="固定种子随机抽样（对齐论文「随机抽 N 题」口径；0=全量时不生效）")
    ap.add_argument("--team", default="chain3", choices=sorted(TEAMS),
                    help="网络形态：single=单 agent；chain3=planner→solver→verifier")
    ap.add_argument("--model-path", default=os.environ.get("LYCHEE_HF_MODEL"),
                    help="HF 模型路径（默认 $LYCHEE_HF_MODEL）")
    ap.add_argument("--api-base-url", default=os.environ.get("LYCHEE_API_BASE_URL"),
                    help="OpenAI 兼容 API 地址（如 http://10.249.45.207:6666/v1）；"
                         "给定后走远程 API，不再要求 --model-path")
    ap.add_argument("--api-model", default=os.environ.get("LYCHEE_API_MODEL"),
                    help="API 模型名（默认 $LYCHEE_API_MODEL）")
    ap.add_argument("--api-key", default=os.environ.get("LYCHEE_API_KEY", "EMPTY"),
                    help="API key（默认 $LYCHEE_API_KEY，缺省为 EMPTY）")
    ap.add_argument("--api-enable-thinking", action="store_true",
                    help="保留 Qwen3 思维链（默认关闭，走 chat_template_kwargs）")
    ap.add_argument("--model-tag", default=None, help="落盘目录用的模型标签（默认取路径末段）")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--max-rounds", type=int, default=2, help="每 agent 最多发言轮数")
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--out-root", default=None,
                    help="结果根目录（默认 eval.benchmarks.common.runs_root()）")
    ap.add_argument("--case-ids", default=None, help="JSON array of benchmark pool indices")
    args = ap.parse_args()

    use_api = bool(args.api_base_url)
    if use_api:
        if not args.api_model:
            raise SystemExit("使用 API 模式需要 --api-model 或环境变量 LYCHEE_API_MODEL")
    elif not args.model_path:
        raise SystemExit("需要 --model-path（或设置 --api-base-url 走远程 API）")

    # ① Benchmark 选择（全量加载 -> 固定种子随机抽样，保证子集可复现）
    pool = load_benchmark(args.task, n=None)
    if not pool:
        raise SystemExit(f"benchmark {args.task!r} 加载为空：请先准备数据（benchmarks.prepare）")
    if args.case_ids:
        with open(args.case_ids, encoding="utf-8") as stream:
            pool_indices = [int(x) for x in json.load(stream)]
        if len(pool_indices) != args.n or len(set(pool_indices)) != len(pool_indices):
            ap.error("case-ids must contain exactly --n unique pool indices")
        if any(i < 0 or i >= len(pool) for i in pool_indices):
            ap.error("case-ids contains an index outside the benchmark pool")
        records = [pool[i] for i in pool_indices]
        sample_meta = {"sample_seed": args.sample_seed, "pool_size": len(pool),
                       "sampled": len(records), "sampled_indices": pool_indices,
                       "subset_source": "explicit_case_ids"}
    else:
        records, sample_meta = sample_records(pool, n=args.n or None, seed=args.sample_seed)
    print(f"[baseline] task={args.task} cases={len(records)} team={args.team}")
    print(f"[baseline] sampling: {len(records)}/{sample_meta['pool_size']} "
          f"seed={sample_meta['sample_seed']} "
          f"indices={'full' if sample_meta['sampled_indices'] is None else 'frozen'}")

    # ② 网络构建
    agents = TEAMS[args.team]
    max_turns = len(agents) * args.max_rounds
    if use_api:
        chat = APIChat(
            args.api_base_url,
            args.api_model,
            api_key=args.api_key,
            enable_thinking=args.api_enable_thinking,
        )
    else:
        chat = HFChat(args.model_path, device=args.device, dtype=args.dtype)
    app = build_app(agents, chat, max_turns=max_turns, max_new_tokens=args.max_new_tokens)

    # ③ 执行 + ④ 评测（逐 case）
    # 结果目录带 team：否则不同队伍（single / chain3）会写进同一目录互相覆盖
    method_label = f"langgraph_baseline_{args.team}"
    # case_id 用「在全量池中的下标」而非子集内序号：同一 seed 下不同 team 的 case_id 可直接配对
    pool_indices = sample_meta["sampled_indices"] or list(range(len(records)))
    if args.model_tag:
        model_tag = args.model_tag
    elif use_api:
        model_tag = args.api_model
    else:
        model_tag = os.path.basename(os.path.normpath(args.model_path))
    out_dir = M.result_dir(model_tag, method_label, args.task, root=args.out_root)
    os.makedirs(out_dir, exist_ok=True)
    progress_path = os.path.join(out_dir, "case_progress.jsonl")
    completed: Dict[str, Dict[str, Any]] = {}
    if os.path.isfile(progress_path):
        with open(progress_path, encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                completed[str(row["case_id"])] = row
    samples: List[Dict[str, Any]] = []
    for i, (pool_index, record) in enumerate(zip(pool_indices, records)):
        case_id = str(pool_index)
        if case_id in completed:
            samples.append(completed[case_id])
            continue
        sample = asyncio.run(
            run_case(app, args.task, record, case_id, max_turns, method_label=method_label)
        )
        sample["pool_index"] = pool_index
        samples.append(sample)
        with open(progress_path, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(sample, ensure_ascii=False) + "\n")
            stream.flush()
        print(f"  [{i + 1}/{len(records)}] score={sample['score']:.3f} "
              f"stop={sample['stop_reason']} calls={sample['model_call_count']} "
              f"pred={sample['prediction'][:60]!r}")

    # ⑤ 落盘：outputs.jsonl + metrics.json + config.yaml（评测模块的统一产物）
    samples = [
        completed.get(
            str(index),
            next((s for s in samples if str(s.get("case_id")) == str(index)), {}),
        )
        for index in pool_indices
    ]
    total_calls = sum(int(sample.get("model_call_count", 0)) for sample in samples)
    truncated_calls = sum(int(sample.get("length_truncation_calls", 0)) for sample in samples)
    truncated_cases = sum(bool(sample.get("length_truncated", False)) for sample in samples)
    run_info = {"task": args.task, "method": method_label,
                "scorer_kind": samples[0]["kind"], "model": model_tag}
    summary = M.aggregate_samples(samples, run_info)
    config = {
        "script": "run_langgraph_baseline.py",
        "git_sha": git_sha(),
        "task": args.task, "n": len(samples), "team": args.team,
        "agents": [name for name, _ in agents],
        "model_path": None if use_api else args.model_path,
        "api_base_url": args.api_base_url if use_api else None,
        "api_model": args.api_model if use_api else None,
        "api_enable_thinking": args.api_enable_thinking if use_api else None,
        "model_tag": model_tag,
        "device": args.device, "dtype": args.dtype,
        "max_rounds": args.max_rounds, "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "runtime": runtime_metadata(
            None if use_api else os.path.abspath(args.model_path)
        ),
        # 抽样快照：种子 + 全量池大小 + 被抽中下标（可直接复现同一子集）
        "sample_seed": sample_meta["sample_seed"],
        "pool_size": sample_meta["pool_size"],
        "sampled_indices": sample_meta["sampled_indices"],
        "subset_source": sample_meta.get("subset_source", "seeded_random_sample"),
        "generation_summary": {
            "calls": total_calls,
            "length_truncation_calls": truncated_calls,
            "length_truncation_call_fraction": (
                truncated_calls / total_calls if total_calls else 0.0),
            "cases_with_length_truncation": truncated_cases,
            "length_truncation_case_fraction": (
                truncated_cases / len(samples) if samples else 0.0),
        },
    }
    M.write_results(out_dir, samples, summary, config)

    acc = summary.get("accuracy", summary.get("mean_score"))
    print(f"[baseline] done: accuracy={acc} out_dir={out_dir}")
    print("[baseline] artifacts: outputs.jsonl / metrics.json / config.yaml")


if __name__ == "__main__":
    main()
