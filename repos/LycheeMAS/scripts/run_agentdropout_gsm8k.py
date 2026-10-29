"""复现 AgentDropout 的 GSM8K 实验（LangGraph 执行 + 本框架 prerun 组件两阶段训练）。

对标：AgentDropout: Dynamic Agent Elimination for Token-Efficient and
High-Performance LLM-Based Multi-Agent Collaboration（ACL 2025, arXiv:2503.18891）；
参考实现 https://github.com/wangzx1219/AgentDropout 的 experiments/run_gsm8k.py 与
graph/graph.py（arun / update_masks_dec / update_masks_diff）——机制与消息语义逐一
对齐；prompt/few-shot 资产逐字 vendored 于 scripts/agentdropout_gsm8k_prompts.py。

与原版逐项对应（代码事实路径；configs/prerun/agentdropout.yaml 有同款注释）：
  agent 配置    5 agents，角色按 [Math Solver, Mathematical Analyst, Programming
                Expert, Inspector] 循环（idx mod 4）；图 FullConnected
  README 调用   python run_gsm8k.py --agent_nums 5 --num_rounds 2 --FullConnected
                --optimized_spatial --optimized_temporal --diff --dec
                --pruning_rate 0.10（README/论文口径；argparse 默认实为 0.25）
 阶段一 Node    dec 期 optimized=False：每轮固定掩码全图的确定性无环实现上执行；
   Dropout      opt.sample_skip(r)（加权度 softmax 多项式采样）选一轮被跳节点，
                其输出 'None.' 且不调 LLM；2 batch × 20 题，批末 opt.skip_reinforce
                （loss=mean_batch(−utility×log_prob)，Adam lr=0.1）；两批后
                opt.node_dropout()（每轮淘汰「全行列 logits 和 / 固定度数」最小者，
                空间行列 + 跨轮时间边清零 → skip_nodes）
 阶段二 Edge    opt.sample_round(r) 逐轮伯努利实现执行（skip_nodes 节点仍 'None.'）；
   Dropout      4 batch × 10 题（与阶段一同用 train 前 40 题），批末
                opt.edge_reinforce；batch idx {1,3} 后各 opt.edge_dropout(rate)
                一次（原版 if (i_batch+1)%2==0 and i_batch<4）；结束 save 落 state
                （两阶段日程在 methods/prerun/agentdropout/trainer.py::AgentDropoutTrainer；
                 本脚本只提供题池、rollout / reward / predict 注入与评测，经统一接口
                 optimize_langgraph(mode="optimize") 驱动——训练循环跟方法走）
 决策          FinalRefer：全部轮次执行完后只跑一次（原版 arun 语义），system/user
                拼法逐字复刻，汇总 5 个 agent 末轮输出（含被淘汰节点的 'None.'）
 打分          gsm_get_predict（原版抽取）+ float 相等比较

与原版的声明差异：
  - LLM 为本地 HF 模型（原版 gpt-4-1106-preview）；绝对准确率不可比，对比对象是
    「同模型下 FullConnected vs 两阶段淘汰后」的准确率与 token 成本。
  - 执行引擎为 LangGraph StateGraph（graphview 节点契约，与 run_maspo_langgraph.py
    同款建图）：每轮把 realized 空间边按拓扑序实例化成契约链图——agent 节点挂
    AgentSpec 元数据（name=agent_{i}、meta.predecessors 通信前驱、meta.dropped 本轮
    被淘汰节点），state 的 outputs 以节点名作 key，A{i} 标识与节点名经 meta.label
    解耦；成环时确定性破环（methods/prerun/graphops.topological_order，与
    run_agentprune_gsm8k.py 同款）。eval threshold 的确定性轮经本框架 prerun 统一
    接口 optimize_langgraph(method="agentdropout", mode="apply", state_file=…, round=r) 逐轮挂载
    （methods/prerun/agentdropout/optimizer.py：realized_matrices(threshold) + meta.dropped
    + rebuild 线性链）——过保真闸门（无终端出边、Kahn 链尾=终端）才走此路径，其余
    轮直构契约链图；两条途径共用同一拓扑排序，消息文本与调用序一致。
    FinalRefer 决策在全部轮次执行完后单独执行一次（不嵌进契约图——挂载器假定图
    节点即 agent 节点）。
  - agent 标识用 A{i} 代替原版 shortuuid（消息文本其余逐字）；角色按图内 idx 静态
    映射，不复刻原版模块级全局 itertools.cycle（同进程多图会互相污染）。
  - 数据集用本框架 gsm8k benchmark loader 单文件分流：train 用前 40 题（两阶段共用，
    与代码事实一致），eval 取 train 之后；原版 train.jsonl/test 分开。
  - 评测 --state-file：threshold 模式 = σ(logit)>0.5 且未被剪的确定性实现
    （methods/prerun/agentdropout/optimizer.py 的 mode="apply" 挂载口径）；sample 模式 = 原版式
    每 query 伯努利采样（原版评测 loop 采样执行）。无 state-file = FullConnected
    对照（确定性全图无环实现、无淘汰），method=agentdropout_full。

用法（train 落 state → eval 挂状态跑分；也可分阶段跑）：
  CDM_DATA_ROOT=/data/.../raw CUDA_VISIBLE_DEVICES=0 \
      python scripts/run_agentdropout_gsm8k.py --phase both \
      --model-path /data/cmz/models/Qwen/Qwen3.5-4B \
      --train-n 40 --eval-n 40 --pruning-rate 0.10
"""

from __future__ import annotations

import argparse
import asyncio
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

import agentdropout_gsm8k_prompts as P  # noqa: E402  vendored 原版 prompt 资产
from lychee_mas.core.types import AgentSpec, TaskQuery  # noqa: E402  节点契约（graphview）
from lychee_mas.eval import metrics as M  # noqa: E402
from lychee_mas.eval.benchmarks import load as load_benchmark  # noqa: E402
from lychee_mas.methods.prerun.agentdropout import (  # noqa: E402
    AgentDropoutOptimizer,
    acyclic_realization,
)
from lychee_mas.methods.prerun.agentdropout.trainer import (  # noqa: E402
    PHASE2_BATCH,
    PRUNE_BATCHES,
    RolloutStats,
    RoundPlan,
    full_temporal_edges,
)
from lychee_mas.methods.prerun.graphops import topological_order  # noqa: E402

# prerun 统一接缝（import 副作用即触发 pre_run_optimizer 注册；run_maspo_langgraph 同款）
from lychee_mas.plugins.prerun import optimize_langgraph  # noqa: E402
from lychee_mas.plugins.prerun.graphview import extract_view  # noqa: E402

QUESTION_SUFFIX = "\nGive the final numeric answer."  # 本框架 loader 附加，复现时剥掉
N_AGENTS = 5                      # README 复现口径：--agent_nums 5
AGENT_ROLES = list(P.GSM8K_ROLES)  # 原版 roles cycle，agent i → AGENT_ROLES[i % 4]
# 两阶段日程的常量（PHASE1_BATCH/PHASE2_BATCH/PRUNE_BATCHES 与批次数）跟训练循环同在
# methods/prerun/agentdropout/trainer.py；本脚本只按需 import（config 快照要用其字面值）。


class ChainState(TypedDict):
    """LangGraph 状态：outputs[agent_{i}] = 本轮输出（契约图以节点名作 key）；
    final 保留占位（FinalRefer 决策在契约图外独立执行，见 decision_step）。"""

    task: str
    outputs: Dict[str, str]
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


# ============ 消息拼装（逐字复刻 ref math_solver.py / final_decision.py） ============

def agent_messages(role: str, question: str, spatial_info: Dict[str, Dict[str, str]],
                   temporal_info: Dict[str, Dict[str, str]]) -> List[Dict[str, str]]:
    system_prompt = P.ROLE_DESCRIPTION[role]
    user_prompt = P.get_answer_prompt(question=question, role=role)
    if role == "Math Solver":
        # 原版 hint 分支：把各前驱输出的抽取数字拼进 "(Hint: The answer is near to ...)."
        # （被淘汰节点的 'None.' 会被 gsm_get_predict 抽成 '0'——原版未过滤，忠实保留）
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


# ========== LangGraph 执行：契约链图（graphview 节点契约）→ 逐轮 ainvoke + 末轮后决策 ==========
# RoundPlan / RolloutStats / full_temporal_edges 由 methods/prerun/agentdropout/trainer.py
# 提供（与训练日程同源，两边口径一致）。

def make_agent_specs(n_agents: int) -> List[AgentSpec]:
    """契约图节点画像（graphview 节点契约，run_maspo_langgraph.py 同款）。

    name=agent_{i}（= state 的 outputs key）；role 按 idx mod 4 静态映射；
    system_prompt = 角色描述（vendored，契约的"可变异提示模板"位——AgentDropout
    不改提示，仅承载身份）；meta 存 idx / label（A{i}，与节点名解耦）/ dropped /
    predecessors。n_agents ≤ 9：rebuild 写回 predecessors 按字典序 = 下标升序，
    是 A{j} 消息序与论文逐字一致的前提。
    """
    if not 2 <= n_agents <= 9:
        raise SystemExit(f"契约图要求 2 ≤ n_agents ≤ 9（字典序=下标升序，保 A{{j}} "
                         f"消息序），得到 {n_agents}")
    specs: List[AgentSpec] = []
    for i in range(n_agents):
        role = AGENT_ROLES[i % len(AGENT_ROLES)]
        specs.append(AgentSpec(
            name=f"agent_{i}", role=role, system_prompt=P.ROLE_DESCRIPTION[role],
            meta={"idx": i, "label": f"A{i}", "dropped": False, "predecessors": []}))
    return specs


def build_contract_app(chat: HFChat, specs: List[AgentSpec], order: List[int],
                       preds: Dict[int, set], prev_outputs: Dict[str, str],
                       temporal_edges: set, skip_idx: Optional[int],
                       max_new_tokens: int, stats: RolloutStats) -> Any:
    """直构：把一轮 (order, preds) 实例化成契约链图并编译。

    preds/skip 写进 spec.meta（predecessors 按 idx 升序 = 旧 sorted(j) 序 → 消息
    A{j} 顺序逐字一致）。每 (query, round) 必须用新鲜 specs：节点闭包捕获
    per-query stats 与 prev_outputs，复用 spec 对象会串账。
    """
    for i in order:
        spec = specs[i]
        spec.meta["predecessors"] = [f"agent_{j}" for j in sorted(preds[i])]
        spec.meta["dropped"] = (i == skip_idx)
    return make_base_graph(chat, specs, order, prev_outputs, temporal_edges,
                           max_new_tokens, stats).compile()


def make_base_graph(chat: HFChat, specs: List[AgentSpec], order: List[int],
                    prev_outputs: Dict[str, str], temporal_edges: set,
                    max_new_tokens: int, stats: RolloutStats) -> Any:
    """契约基图（未编译）：线性链 order[0] → … → order[-1] → END，agent 节点挂
    metadata["agent_spec"]（graphview 节点契约）。节点函数运行时从共享 spec 读
    meta.predecessors（空间前驱，本轮已执行）与 meta.dropped（'None.' 不调 LLM）；
    temporal 前驱不进契约（单轮快照，挂载器只写空间邻接）——由闭包捕获的
    prev_outputs / temporal_edges 提供，随逐轮驱动循环传入。
    langgraph 惰性导入（黄金法则 2）。
    """
    from langgraph.graph import END, START, StateGraph

    by_idx = {spec.meta["idx"]: spec for spec in specs}
    by_name = {spec.name: spec for spec in specs}

    def make_agent_node(spec: AgentSpec):
        async def node(state: ChainState) -> Dict[str, Any]:
            if spec.meta["dropped"]:
                # 原版 arun：被淘汰/被采样跳过的节点不调 LLM，输出 'None.' 传给后继
                return {"outputs": {**state["outputs"], spec.name: "None."}}
            spatial_info = {by_name[p].meta["label"]: {"role": by_name[p].role,
                                                       "output": state["outputs"][p]}
                            for p in spec.meta["predecessors"] if p in state["outputs"]}
            temporal_info = {}
            for s in sorted({s for (s, d) in temporal_edges if d == spec.meta["idx"]}):
                p = by_idx[s].name
                if p in prev_outputs:
                    temporal_info[by_idx[s].meta["label"]] = {"role": by_idx[s].role,
                                                              "output": prev_outputs[p]}
            msgs = agent_messages(spec.role, state["task"], spatial_info, temporal_info)
            g = chat.generate(msgs, max_new_tokens=max_new_tokens)
            text = g.text
            if spec.role == "Programming Expert":
                # 原版：跑生成代码，把返回值以 "the answer is X" 附在响应后
                code = text.lstrip("```python\n").rstrip("\n```")
                answer = execute_python_code(code)
                text += f"\nthe answer is {answer}"
            stats.prompt_tokens += g.prompt_tokens
            stats.completion_tokens += g.completion_tokens
            stats.model_calls += 1
            stats.latency_s += g.latency_s
            return {"outputs": {**state["outputs"], spec.name: text}}

        return node

    sg = StateGraph(ChainState)
    for i in order:
        spec = specs[i]
        sg.add_node(spec.name, make_agent_node(spec), metadata={"agent_spec": spec})
    names = [specs[i].name for i in order]
    sg.add_edge(START, names[0])
    for a, b in zip(names, names[1:]):
        sg.add_edge(a, b)
    sg.add_edge(names[-1], END)
    return sg


def decision_step(chat: HFChat, question: str, specs: List[AgentSpec],
                  outputs: Dict[str, str], max_new_tokens: int,
                  stats: RolloutStats) -> str:
    """FinalRefer 收尾：全部轮次执行完后只跑一次（原版 arun 语义）。

    决策节点不进契约图（挂载器假定图节点即 agent 节点、names 序 = mask 下标）——
    作为独立一步读末轮全部 agent 输出（含被淘汰节点的 'None.'），按 A{idx} 升序
    渲染（decision_messages 逐字拼装，与链内执行时的消息文本一致）。
    """
    by_idx = {spec.meta["idx"]: spec for spec in specs}
    ordered = {i: outputs[by_idx[i].name] for i in sorted(by_idx)}
    msgs = decision_messages(question, ordered)
    g = chat.generate(msgs, max_new_tokens=max_new_tokens)
    stats.prompt_tokens += g.prompt_tokens
    stats.completion_tokens += g.completion_tokens
    stats.model_calls += 1
    stats.latency_s += g.latency_s
    return g.text


def safe_preds_for_rebuild(edges: set, n: int) -> Optional[Dict[int, set]]:
    """保真闸门：rebuild/挂载器与直构（旧执行）逐字同语义 ⇔ 边集不含 (n-1,·) 出边
    （rebuild 无条件剔除终端出边 → 消息文本会变）且 Kahn 链尾 = n-1（rebuild 链尾
    强制为终端）。过闸门返回 topological_order 的 final_preds（= rebuild 将给出的
    同一结果，同函数同输入）；不过返回 None（该轮只能直构）。
    """
    if any(a == n - 1 for (a, _b) in edges):
        return None
    order, preds = topological_order(n, edges)
    return preds if order[-1] == n - 1 else None


def assert_contract_matches(graph: Any, n_agents: int, expected_preds: Dict[int, set],
                            skip_idx: Optional[int]) -> None:
    """挂载器（optimize_langgraph round 路径）产图的契约断言：predecessors / dropped
    / terminal 必须与直构预期（topological_order final_preds）逐点一致——不符即显式
    raise（不静默降级）。"""
    view = extract_view(graph)
    for i in range(n_agents):
        name = f"agent_{i}"
        expect = [f"agent_{j}" for j in sorted(expected_preds[i])]
        if view.predecessors[name] != expect:
            raise AssertionError(
                f"optimize_langgraph(round={view.specs[name].meta.get('idx')}) 产图 "
                f"节点 {name} 前驱 {view.predecessors[name]} != 直构预期 {expect}")
        if view.specs[name].meta["dropped"] != (i == skip_idx):
            raise AssertionError(
                f"optimize_langgraph 产图节点 {name} meta.dropped 与预期不符")
    if view.terminal != f"agent_{n_agents - 1}":
        raise AssertionError(
            f"optimize_langgraph 产图终端 {view.terminal!r} 非 agent_{n_agents - 1}")


async def run_query(chat: HFChat, n_agents: int, question: str, plans: List[RoundPlan],
                    max_new_tokens: int, *,
                    plugin_state_file: Optional[str] = None
                    ) -> Tuple[str, Dict[str, str], RolloutStats]:
    """按每轮 plan 执行 num_rounds 轮；全部轮后 FinalRefer 决策一次（原版 arun 语义）。

    每 (query, round) 把该轮 realized 空间边实例化成契约链图执行：
      - plugin_state_file（eval threshold）且过保真闸门 → 经本框架 prerun 统一入口
        optimize_langgraph(method="agentdropout", mode="apply", state_file=…, round=r,
        rounds=…) 挂载（methods/prerun/agentdropout/optimizer.py：threshold 矩阵 + meta.dropped +
        rebuild 线性链），随即契约断言锁等价性（不符显式失败）；
      - 否则直构契约链图（build_contract_app：topological_order 定序 + spec.meta
        写 preds/dropped + 线性链）。
    两条途径共用 methods/prerun/agentprune.topological_order，消息文本与 LLM 调用
    序逐字一致；temporal 跨轮通道（prev_outputs）随循环闭包传递。每轮每 query 的
    specs 都是新造的（串扰防护）；compile 次数 = 每 (query, round) 一次，与旧版持平。
    """
    stats = RolloutStats()
    prev_outputs: Dict[str, str] = {}
    final = ""
    outputs: Dict[str, str] = {}
    specs: List[AgentSpec] = []
    for r, plan in enumerate(plans):
        specs = make_agent_specs(n_agents)
        expected_preds = (safe_preds_for_rebuild(plan.spatial_edges, n_agents)
                          if plugin_state_file else None)
        if expected_preds is not None:
            base = make_base_graph(chat, specs, list(range(n_agents)), prev_outputs,
                                   plan.temporal_edges, max_new_tokens, stats)
            graph = optimize_langgraph(base, method="agentdropout", mode="apply",
                                       state_file=plugin_state_file,
                                       round=r, rounds=len(plans))
            assert_contract_matches(graph, n_agents, expected_preds, plan.skip_idx)
            app = graph.compile()
        else:
            order, preds = topological_order(n_agents, plan.spatial_edges)
            app = build_contract_app(chat, specs, order, preds, prev_outputs,
                                     plan.temporal_edges, plan.skip_idx,
                                     max_new_tokens, stats)
        state: ChainState = {"task": question, "outputs": {}, "final": ""}
        result = await app.ainvoke(state, config={"recursion_limit": 4 * n_agents + 10})
        outputs = result["outputs"]
        prev_outputs = dict(outputs)
    final = decision_step(chat, question, specs, outputs, max_new_tokens, stats)
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


def make_optimizer(args: argparse.Namespace, state_file: Optional[str] = None,
                   seed: Optional[int] = None) -> AgentDropoutOptimizer:
    return AgentDropoutOptimizer(n_agents=args.num_agents, rounds=args.num_rounds,
                                 lr=args.lr, seed=args.seed if seed is None else seed,
                                 state_file=state_file)


# ============ 训练：阶段一 Node Dropout → node_dropout；阶段二 Edge Dropout ============

def _train_pool(records: List[Dict[str, Any]], args: argparse.Namespace) -> List[Dict[str, Any]]:
    """两阶段共用的 40 题池（代码事实：dec 2×20 与 diff 4×10 都取自 train 前 40）。"""
    if len(records) < 40:
        raise SystemExit(f"训练需要 ≥40 题（阶段一 2×20 + 阶段二 4×10 同池），得到 {len(records)}")
    return records[:40]


def train(args: argparse.Namespace, chat: HFChat) -> str:
    """两阶段训练（循环在方法侧：methods/prerun/agentdropout/trainer.py）。

    本脚本只提供「数据 + 生成后端」：题池（train 前 40 题）→ TaskQuery；``rollout``
    （契约链图逐轮执行 + FinalRefer 决策 + token/调用记账）；``reward``（gsm_get_predict
    + float 相等）；``predict``（train_log 的 pred 字段抽取器）。日程、RNG 消费序、落盘
    与进度打印全在 ``AgentDropoutTrainer``；本函数经统一入口 ``optimize_langgraph
    (mode="optimize")`` 驱动（methods/prerun/agentdropout/optimizer.py 的 AgentDropoutLG，
    与 run_maspo_langgraph.py 同款路径），返回 state 产物路径。
    """
    records = load_gsm8k_records(args.train_n)
    pool = _train_pool(records, args)
    trainset = [TaskQuery(question=rec["question"], gold=rec["gold"]) for rec in pool]
    state_path = os.path.join(args.out_root or ".", "agentdropout_gsm8k_state.json")

    async def rollout(question: str, plans: List[RoundPlan]) -> Tuple[str, RolloutStats]:
        final, _outputs, stats = await run_query(
            chat, args.num_agents, question, plans, args.max_new_tokens)
        return final, stats

    # 载体图：训练不消费图（rollout 自行跑图），只需按契约图取 n_agents（节点不执行）
    carrier = make_base_graph(chat, make_agent_specs(args.num_agents),
                              list(range(args.num_agents)), {}, set(),
                              args.max_new_tokens, RolloutStats())
    optimize_langgraph(carrier, method="agentdropout", mode="optimize",
                       state_file=state_path, trainset=trainset, rollout=rollout,
                       reward=utility_of, predict=P.gsm_get_predict,
                       lr=args.lr, seed=args.seed, rounds=args.num_rounds,
                       pruning_rate=args.pruning_rate)
    return state_path


# ================== 评测阶段（state 驱动 / FullConnected 对照） ==================

def evaluate(args: argparse.Namespace, chat: HFChat, state_file: Optional[str]) -> None:
    records = load_gsm8k_records(args.eval_n, skip=args.train_n)
    opt = make_optimizer(args, state_file=state_file) if state_file else None
    method = "agentdropout_pruned" if state_file else "agentdropout_full"

    samples: List[Dict[str, Any]] = []
    for i, rec in enumerate(records):
        if opt is not None:
            if args.eval_mode == "threshold":
                # ★ 确定性：σ>0.5 且未被剪（plugins apply 挂载口径）；skip_nodes 本轮 'None.'
                plans = []
                for r in range(opt.rounds):
                    sm, tm = opt.realized_matrices(r, "threshold")
                    spatial = {(a, b) for a in range(opt.n) for b in range(opt.n)
                               if sm[a][b]}
                    temporal = {(a, b) for a in range(opt.n) for b in range(opt.n)
                                if tm[a][b]}
                    plans.append(RoundPlan(spatial_edges=spatial, temporal_edges=temporal,
                                           skip_idx=opt.skip_nodes.get(r)))
            else:  # sample：原版式采样式推理（评测 loop 即逐 query 伯努利实现）
                plans = [RoundPlan(spatial_edges=re.spatial_edges,
                                   temporal_edges=re.temporal_edges,
                                   skip_idx=opt.skip_nodes.get(r))
                         for r, re in enumerate([opt.sample_round(r)
                                                 for r in range(opt.rounds)])]
        else:
            # FullConnected 对照：确定性全图无环实现，无淘汰（原版 optimized=False 的构造）
            full_edges = acyclic_realization(
                args.num_agents,
                {(a, b) for a in range(args.num_agents) for b in range(args.num_agents)
                 if a != b})
            plans = [RoundPlan(spatial_edges=full_edges,
                               temporal_edges=(full_temporal_edges(args.num_agents)
                                               if r >= 1 else set()),
                               skip_idx=None)
                     for r in range(args.num_rounds)]

        final, _outputs, stats = asyncio.run(
            run_query(chat, args.num_agents, rec["question"], plans, args.max_new_tokens,
                      plugin_state_file=(state_file if args.eval_mode == "threshold"
                                         else None)))
        pred = P.gsm_get_predict(final)
        score = utility_of(final, rec["gold"])
        samples.append({
            "case_id": str(i), "task": "gsm8k", "kind": "exact", "method": method,
            "question": rec["question"], "gold": rec["gold"], "prediction": pred,
            "score": score, "is_correct": bool(score == 1.0),
            "spatial_edges": sorted(list(plans[-1].spatial_edges)),
            "skip_nodes": {str(r): p.skip_idx for r, p in enumerate(plans)
                           if p.skip_idx is not None},
            "model_call_count": stats.model_calls,
            "input_positions_total": stats.prompt_tokens,
            "gen_tokens": stats.completion_tokens,
            "latency_s": round(stats.latency_s, 3),
        })
        print(f"  [{i + 1}/{len(records)}] score={score:.0f} "
              f"edges={len(plans[-1].spatial_edges)} calls={stats.model_calls} "
              f"prompt_toks={stats.prompt_tokens} pred={pred!r}")

    model_tag = args.model_tag or os.path.basename(os.path.normpath(args.model_path))
    out_dir = M.result_dir(model_tag, method, "gsm8k", root=args.out_root)
    summary = M.aggregate_samples(samples, {"task": "gsm8k", "method": method,
                                            "scorer_kind": "exact", "model": model_tag})
    config = {
        "script": "run_agentdropout_gsm8k.py", "method": method,
        "state_file": state_file, "eval_mode": args.eval_mode,
        "n_agents": args.num_agents, "roles": AGENT_ROLES,
        "num_rounds": args.num_rounds, "lr": args.lr,
        "phase1": "2 batch x 20 -> node_dropout()", "phase2_batch_size": PHASE2_BATCH,
        "prune_batch_idx": list(PRUNE_BATCHES), "pruning_rate": args.pruning_rate,
        "train_n": args.train_n, "eval_n": len(samples), "seed": args.seed,
        "model_path": args.model_path, "max_new_tokens": args.max_new_tokens,
        "reference": "https://github.com/wangzx1219/AgentDropout (ACL 2025, "
                     "arXiv:2503.18891)",
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
                    help="eval 阶段加载的两阶段训练产物（train 产出；不给且 phase=eval "
                         "则跑 FullConnected 对照）")
    ap.add_argument("--num-agents", type=int, default=N_AGENTS,
                    help="agent 数（README 复现口径 5）")
    ap.add_argument("--num-rounds", type=int, default=2,
                    help="每 query 轮数（README 复现口径 2）")
    ap.add_argument("--train-n", type=int, default=40,
                    help="训练数据取前 N 题（两阶段共用前 40 题为代码事实，需 ≥40）")
    ap.add_argument("--eval-n", type=int, default=40, help="评测题数（取训练集之后的题）")
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--pruning-rate", type=float, default=0.10,
                    help="阶段二剪边率（README/论文口径 0.10；原版 argparse 默认 0.25）")
    ap.add_argument("--eval-mode", default="threshold", choices=("threshold", "sample"),
                    help="threshold=确定性实现（σ>0.5 且未被剪，插件 apply 口径）；"
                         "sample=原版式采样式")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model-path", default=os.environ.get("LYCHEE_HF_MODEL"))
    ap.add_argument("--model-tag", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--out-root", default=None)
    args = ap.parse_args()

    if args.train_n < 40:
        raise SystemExit("--train-n 需 ≥40（阶段一 2×20 + 阶段二 4×10 同池；原版硬编码）")
    if args.num_agents < 2:
        raise SystemExit("需要 ≥2 个 agent（--num-agents）")
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
