"""pre_run_optimizer/maspo —— MASPO 联合提示优化（LangGraph 统一接口版）。

复现自：MASPO: Joint Prompt Optimization for LLM-based Multi-Agent Systems（ICML 2026，
arXiv:2605.06623）；参考实现 https://github.com/wangzx1219/MASPO（optimizers.py 的
fixed-rounds 主线，与 run_maspo.py 的 `--optimize --fixed-rounds --beam-refresh
--lookahead-score --misleading-sampling` 论文模式逐一对应）：

- **多粒度联合评估**：候选提示 vs 基线的三路 LLM 成对比较——Local（本节点中间输出）、
  Lookahead（直接后继输出）、Global（终端答案）；win_rate 按 lookahead_weights
  （默认 0.4/0.4/0.2）加权，无 lookahead 时退化为 0.7·local + 0.3·global；
  score = win_rate − 0.5，全程**免 gold 标注**。
- **错位驱动采样**：Local-Win 但 Next/Global-Lose 的样本为 misleading cases（both-lose >
  next-lose > global-lose 优先），注入后续采样池作 hard negatives。
- **进化 beam search + 自适应调度**：每节点每层 2 个候选、beam_width=2、score>0 才入 beam、
  累计分排序；坐标上升式 fixed-rounds 调度（每 agent 轮流 rounds_per_turn=3 层直到
  max_total_depth=9）；Beam Refresh 在重访 agent 时按队友新提示重打 beam 分并可切换锚点。

与原版的声明差异（不影响方法语义）：
1. LLM 经注入的 async 回调触达（agent_llm=执行端 / evaluator_llm=比较端 /
   proposer_llm=反思提议端，对应原版比较 temperature=0、提议 temperature=0.7 的分工），
   非原版写死的 API 端点；2. 系统表示为 GraphView（自 StateGraph 元数据提取），产物 prompt_map
   以**节点名**为键（原版为下标）；3. 采样用实例内 seeded RNG（原版全局 random，不可复现）；
4. round-robin 调度与 dynamic-switching / stochastic-sampling 开关未移植（论文主实验未用）。

产物文件格式：``{"prompts": {节点名: 提示模板}, "meta": {...}}``（apply 模式亦兼容
裸 ``{节点名: 提示}`` 映射）；统计落同名 ``*_stats.json``。纯标准库。
"""
from __future__ import annotations

import asyncio
import json
import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ....core.registry import REGISTRY
from .executor import AsyncLLM, CachedExecutor, InferenceCache
from .prompts import (
    ANSWER_EVALUATE_TEMPLATE,
    FINAL_ANSWER_COMPARE_TEMPLATE,
    INTERMEDIATE_COMPARE_TEMPLATE,
    OPTIMIZATION_REQUIREMENT,
    PROMPT_OPTIMIZE_TEMPLATE,
    role_description,
)
from .textops import extract_prompt_tag, parse_comparison_result, sanitize_prompt

# 全角色统一前缀（原版 full_requirement 的固定句）
_REQUIREMENT_PREFIX = ("Ensure the agent's role, responsibilities, and input format "
                       "remain consistent. ")


@dataclass
class AgentOptState:
    """单 agent 的优化状态（原版 AgentOptState，键改节点名）。beam 节点为
    ``{"prompt", "cumulative_score", "path"}`` 字典（与原版同构）。"""

    name: str
    current_beam: List[Dict[str, Any]]
    best_overall_node: Dict[str, Any]
    total_layers_explored: int = 0
    recent_bad_cases: List[Dict[str, Any]] = field(default_factory=list)
    misleading_cases: List[Dict[str, Any]] = field(default_factory=list)
    misalignment_rates_per_depth: List[float] = field(default_factory=list)
    beam_refresh_kendall_scores: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def seeded(cls, name: str, prompt: str) -> "AgentOptState":
        node = {"prompt": prompt, "cumulative_score": 0.0, "path": [prompt]}
        return cls(name=name, current_beam=[node], best_overall_node=dict(node))

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "current_beam": self.current_beam,
                "best_overall_node": self.best_overall_node,
                "total_layers_explored": self.total_layers_explored,
                "recent_bad_cases": self.recent_bad_cases,
                "misleading_cases": self.misleading_cases,
                "misalignment_rates_per_depth": self.misalignment_rates_per_depth,
                "beam_refresh_kendall_scores": self.beam_refresh_kendall_scores}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AgentOptState":
        return cls(name=d["name"], current_beam=list(d["current_beam"]),
                   best_overall_node=dict(d["best_overall_node"]),
                   total_layers_explored=int(d["total_layers_explored"]),
                   recent_bad_cases=list(d["recent_bad_cases"]),
                   misleading_cases=list(d["misleading_cases"]),
                   misalignment_rates_per_depth=list(d["misalignment_rates_per_depth"]),
                   beam_refresh_kendall_scores=list(d["beam_refresh_kendall_scores"]))


def _limited(llm: AsyncLLM, sem: asyncio.Semaphore) -> AsyncLLM:
    async def call(prompt: str) -> str:
        async with sem:
            return await llm(prompt)
    return call


@REGISTRY.register("pre_run_optimizer", "maspo")
class MASPOOptimizer:
    """MASPO 联合提示优化器（PreRunOptimizer 协议）。

    - ``mode="apply"``：从 ``prompt_file`` 读优化后的 prompt_map，写回图（轻量即插即用）。
    - ``mode="optimize"``：在图上跑完整 fixed-rounds 联合优化（需 ``trainset`` 问题列表 +
      ``agent_llm``/``evaluator_llm`` 回调），产物写 ``prompt_file``（若给了），再写回图。
    """

    name = "maspo"

    def __init__(self, mode: str = "apply", prompt_file: Optional[str] = None,
                 trainset: Optional[Sequence[str]] = None,
                 agent_llm: Optional[AsyncLLM] = None,
                 evaluator_llm: Optional[AsyncLLM] = None,
                 proposer_llm: Optional[AsyncLLM] = None,
                 requirement: str = OPTIMIZATION_REQUIREMENT,
                 max_total_depth: int = 9, rounds_per_turn: int = 3, beam_width: int = 2,
                 eval_batch: int = 10, misleading_max: int = 5,
                 lookahead_weights: Tuple[float, float, float] = (0.4, 0.4, 0.2),
                 use_beam_refresh: bool = True, use_lookahead_score: bool = True,
                 use_misleading_sampling: bool = True, use_feedback: bool = False,
                 seed: int = 0, max_concurrency: int = 8, verbose: bool = True) -> None:
        if mode not in ("apply", "optimize"):
            raise ValueError(f"maspo mode 必须是 apply|optimize，得到 {mode!r}")
        self.mode = mode
        self.prompt_file = prompt_file
        self.trainset = list(trainset or [])
        self.max_concurrency = int(max_concurrency)
        # 原始回调；并发限流包装在每次 optimize 的事件循环内做（Semaphore 不能跨 loop 复用）
        self._raw_agent_llm = agent_llm
        self._raw_evaluator_llm = evaluator_llm
        # 反思提议端与比较端分离：原版 _propose_new_prompt 用 temperature=0.7（探索），
        # 三路比较用 0.0；不传 proposer_llm 时回退 evaluator_llm
        self._raw_proposer_llm = proposer_llm
        self.agent_llm: Optional[AsyncLLM] = None
        self.evaluator_llm: Optional[AsyncLLM] = None
        self.proposer_llm: Optional[AsyncLLM] = None
        self.requirement = requirement
        self.max_total_depth = int(max_total_depth)
        self.rounds_per_turn = int(rounds_per_turn)
        self.beam_width = int(beam_width)
        self.eval_batch = int(eval_batch)
        self.misleading_max = int(misleading_max)
        w = tuple(float(x) for x in lookahead_weights)
        if len(w) != 3 or sum(w) <= 0:
            raise ValueError(f"lookahead_weights 需要 3 个非全零权重，得到 {lookahead_weights!r}")
        self.lookahead_weights = tuple(x / sum(w) for x in w)
        self.use_beam_refresh = bool(use_beam_refresh)
        self.use_lookahead_score = bool(use_lookahead_score)
        self.use_misleading_sampling = bool(use_misleading_sampling)
        self.use_feedback = bool(use_feedback)
        self.seed = int(seed)
        self.rng = random.Random(seed)
        self.verbose = bool(verbose)
        # 运行产物（脚本读取用）
        self.last_prompt_map: Optional[Dict[str, str]] = None
        self.last_statistics: Optional[Dict[str, Any]] = None

    # ------------------------------ 统一入口 ------------------------------

    def optimize(self, graph: Any) -> Any:
        from ....plugins.prerun.graphview import extract_view, rebuild  # 函数内导入避免包内环

        view = extract_view(graph)
        if self.mode == "apply":
            prompt_map = self._load_prompt_file()
        else:
            if not self.trainset:
                raise ValueError("maspo mode=optimize 需要非空 trainset（训练问题列表）")
            if self._raw_agent_llm is None or self._raw_evaluator_llm is None:
                raise ValueError("maspo mode=optimize 需要 agent_llm 与 evaluator_llm 回调")
            for name in view.names:
                if not view.specs[name].system_prompt.strip():
                    raise ValueError(f"节点 {name!r} 的种子提示为空：无法作为优化起点")
            prompt_map, statistics = asyncio.run(self._optimize_with_limits(view))
            self.last_statistics = statistics
            if self.prompt_file:
                self._save(prompt_map, statistics)
        self.last_prompt_map = dict(prompt_map)
        return rebuild(graph, prompts=prompt_map)

    def _load_prompt_file(self) -> Dict[str, str]:
        if not self.prompt_file:
            raise ValueError("maspo mode=apply 需要 prompt_file（optimize 阶段的产物）")
        with open(self.prompt_file, encoding="utf-8") as f:
            data = json.load(f)
        raw = data.get("prompts", data) if isinstance(data, dict) else None
        if not isinstance(raw, dict) or not raw or \
                not all(isinstance(k, str) and isinstance(v, str) for k, v in raw.items()):
            raise ValueError(
                f"{self.prompt_file}: 期望 {{'prompts': {{节点名: 提示}}}} 或裸映射，无法解析")
        return dict(raw)

    def _save(self, prompt_map: Dict[str, str], statistics: Dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(self.prompt_file) or ".", exist_ok=True)
        payload = {"prompts": prompt_map,
                   "meta": {"method": "maspo", "max_total_depth": self.max_total_depth,
                            "rounds_per_turn": self.rounds_per_turn,
                            "beam_width": self.beam_width, "eval_batch": self.eval_batch,
                            "lookahead_weights": list(self.lookahead_weights),
                            "use_beam_refresh": self.use_beam_refresh,
                            "use_lookahead_score": self.use_lookahead_score,
                            "use_misleading_sampling": self.use_misleading_sampling,
                            "use_feedback": self.use_feedback,
                            "trainset_size": len(self.trainset),
                            "reference": "https://github.com/wangzx1219/MASPO (ICML 2026)"}}
        with open(self.prompt_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        stem, _ext = os.path.splitext(self.prompt_file)
        with open(stem + "_stats.json", "w", encoding="utf-8") as f:
            json.dump(statistics, f, indent=2, ensure_ascii=False)

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    # --------------------- 三路成对比较（原版 _compare*） ---------------------

    async def _compare_terminal(self, question: str, cand: str, base: str, role: str) -> bool:
        prompt = ANSWER_EVALUATE_TEMPLATE.format(
            agent_type=role, role_description=role_description(role),
            question=question.strip(), requirement=self.requirement,
            Answer_A=cand, Answer_B=base)
        return parse_comparison_result(await self.evaluator_llm(prompt))

    async def _compare_intermediate(self, question: str, cand: str, base: str) -> bool:
        prompt = INTERMEDIATE_COMPARE_TEMPLATE.format(
            question=question.strip(), output_a=cand.strip() or "(empty)",
            output_b=base.strip() or "(empty)")
        return parse_comparison_result(await self.evaluator_llm(prompt))

    async def _compare_final(self, question: str, cand: str, base: str) -> bool:
        prompt = FINAL_ANSWER_COMPARE_TEMPLATE.format(
            question=question.strip(), requirement=self.requirement,
            Answer_A=cand.strip(), Answer_B=base.strip())
        return parse_comparison_result(await self.evaluator_llm(prompt))

    # ----------------------- 反思变异（原版 _propose_new_prompt） -----------------------

    async def _propose_new_prompt(self, old_p: str, qa: Dict[str, Dict[str, str]],
                                  role: str, successor_info: str = "") -> str:
        samples_block = "\n\n".join(
            f"Problem {i + 1}:\n{q.strip()}\n\nContext:\n"
            f"{data['context'].strip() or '(no context)'}\n\nAgent Output:\n"
            f"{data['output'].strip()}"
            for i, (q, data) in enumerate(qa.items()))
        full_requirement = _REQUIREMENT_PREFIX + self.requirement
        if successor_info:
            full_requirement += (
                "\n\n[DOWNSTREAM CONSTRAINT & FEEDBACK]\n"
                "The output of this agent serves as INPUT for a downstream agent.\n"
                f"{successor_info}\n"
                "**Optimization Goal**: Crucially, modify the prompt so the output addresses "
                "the issues above and strictly adheres to constraints to help the downstream "
                "agent succeed.")
        prompt = PROMPT_OPTIMIZE_TEMPLATE.format(
            agent_type=role, role_description=role_description(role),
            requirements=full_requirement, prompt=old_p, samples=samples_block)
        raw = await self.proposer_llm(prompt)
        new_p = extract_prompt_tag(raw)
        if new_p is None:
            self._log("  [maspo] 反思输出缺 <prompt> 标签，保留旧提示")
            return old_p
        return sanitize_prompt(new_p, old_p)

    # ------------------- 基线缓存 / 候选评估（原版 _evaluate_candidate） -------------------

    async def _build_baseline_caches(self, view: Any, questions: Sequence[str],
                                     prompt_map: Dict[str, str]) -> Dict[str, InferenceCache]:
        ex = CachedExecutor(view, self.agent_llm, prompt_map)

        async def run_one(q: str) -> Tuple[str, InferenceCache]:
            _, cache = await ex.run(q)
            return q, cache

        return dict(await asyncio.gather(*[run_one(q) for q in questions]))

    async def _evaluate_candidate(self, view: Any, cand_prompt: str, name: str,
                                  eval_samples: Sequence[str],
                                  baseline_caches: Dict[str, InferenceCache],
                                  temp_prompt_map: Dict[str, str],
                                  use_lookahead: bool,
                                  random_sample_set: Optional[set] = None) -> Dict[str, Any]:
        """候选 vs 基线的多粒度联合评估（返回 score / wins / bad & misleading cases）。"""
        is_terminal = name == view.terminal
        successors = view.successors(name) if (use_lookahead and not is_terminal) else []
        n_eval = len(eval_samples)

        async def process(q: str):
            base_cache = baseline_caches[q]
            ex = CachedExecutor(view, self.agent_llm, temp_prompt_map)
            _, cand_cache = await ex.run_from_node(name, base_cache, cand_prompt)
            cand_out = cand_cache.node_outputs_raw.get(name, "")
            base_out = base_cache.node_outputs_raw.get(name, "")
            info = {"question": q,
                    "context": base_cache.context_for(view.predecessors[name]),
                    "output": cand_out}
            tasks, metas = [], []
            if is_terminal:
                tasks.append(self._compare_terminal(q, cand_out, base_out,
                                                    view.specs[name].role))
                metas.append(("terminal", info))
            else:
                cand_final = cand_cache.node_outputs_raw.get(view.terminal, "")
                base_final = base_cache.node_outputs_raw.get(view.terminal, "")
                tasks.append(self._compare_final(q, cand_final, base_final))
                metas.append(("global", info))
                tasks.append(self._compare_intermediate(q, cand_out, base_out))
                metas.append(("local", info))
                for succ in successors:
                    sc, sb = (cand_cache.node_outputs_raw.get(succ, ""),
                              base_cache.node_outputs_raw.get(succ, ""))
                    if sc and sb:
                        tasks.append(self._compare_intermediate(q, sc, sb))
                        metas.append(("next_local", info))
            return list(zip(await asyncio.gather(*tasks), metas))

        flat = [item for sub in await asyncio.gather(*[process(q) for q in eval_samples])
                for item in sub]

        wins = {"terminal": 0, "global": 0, "local": 0, "next_local": 0}
        count_next = 0
        case_map: Dict[str, Dict[str, Any]] = {}
        bad_candidates: Dict[str, Dict[str, Any]] = {}
        for is_win, (kind, info) in flat:
            q = info["question"]
            case = case_map.setdefault(q, {"info": info})
            if kind == "next_local":
                count_next += 1
                case["next_local"] = case.get("next_local", True) and is_win
            elif kind in ("global", "terminal"):
                case["global"] = is_win
            else:
                case["local"] = is_win
            wins[kind] += int(is_win)
            if not is_win and kind in ("local", "terminal"):
                bad_candidates[q] = info

        # 错位统计与 misleading 收集（Local-Win 而 Next/Global-Lose；原版优先级 0/1/2）
        misalignment_count, total_evaluable = 0, 0
        prioritized: List[Tuple[int, Dict[str, Any]]] = []
        for q, case in case_map.items():
            if "local" not in case:
                continue
            in_random = random_sample_set is None or q in random_sample_set
            if not case["local"]:
                total_evaluable += int(in_random)
                continue
            g, nx = case.get("global", True), case.get("next_local", True)
            if in_random:
                total_evaluable += 1
                misalignment_count += int((not nx) or (not g))
            priority = 0 if (not nx and not g) else 1 if not nx else 2 if not g else -1
            if priority != -1:
                prioritized.append((priority, case["info"]))
        prioritized.sort(key=lambda x: x[0])

        if is_terminal:
            win_rate = wins["terminal"] / n_eval if n_eval else 0.0
        else:
            rate_local = wins["local"] / n_eval if n_eval else 0.0
            rate_global = wins["global"] / n_eval if n_eval else 0.0
            if use_lookahead and count_next > 0:
                w_l, w_n, w_g = self.lookahead_weights
                win_rate = (rate_local * w_l + (wins["next_local"] / count_next) * w_n
                            + rate_global * w_g)
            else:
                win_rate = rate_global * 0.3 + rate_local * 0.7

        return {"score": win_rate - 0.5,
                "wins_global": wins["global"] + wins["terminal"], "wins_local": wins["local"],
                "bad_cases": list(bad_candidates.values())[:3],
                "misleading_cases": [info for _p, info in prioritized[:3]],
                "misalignment_rate": (misalignment_count / total_evaluable
                                      if total_evaluable else 0.0)}

    # ---------------- 单 beam 节点扩展（原版 process_single_node） ----------------

    async def _process_single_node(self, view: Any, node: Dict[str, Any], name: str,
                                   states: Dict[str, AgentOptState]) -> Dict[str, Any]:
        # ① 采样池：上游 misleading cases 注入（hard negatives）+ 随机补齐 eval_batch 题
        injected: set = set()
        if self.use_misleading_sampling:
            for pred in view.predecessors[name]:
                if pred in states:
                    injected |= {c["question"] for c in states[pred].misleading_cases}
        injected_list = list(injected)
        if len(injected_list) > self.misleading_max:
            injected_list = self.rng.sample(injected_list, self.misleading_max)
        pool = [q for q in self.trainset if q not in injected]
        needed = max(0, min(self.eval_batch, len(pool) + len(injected_list)) - len(injected_list))
        random_samples = (self.rng.sample(pool, needed) if len(pool) >= needed
                          else list(pool))
        samples = injected_list + random_samples
        mid = len(samples) // 2

        # ② 该 beam 节点的基线：全局最优 prompt_map + 本节点换成 node 的 prompt
        base_map = {n: states[n].best_overall_node["prompt"] if n in states
                    else view.specs[n].system_prompt for n in view.names}
        base_map[name] = node["prompt"]
        baseline_caches = await self._build_baseline_caches(view, samples, base_map)

        def qa_of(part: Sequence[str]) -> Dict[str, Dict[str, str]]:
            return {q: {"context": baseline_caches[q].context_for(view.predecessors[name]),
                        "output": baseline_caches[q].node_outputs_raw.get(name, "")}
                    for q in part}

        # ③ 下游反馈（use_feedback：后继 agent 的 bad cases 作为上游约束）
        successor_info = ""
        if self.use_feedback:
            msgs = []
            for succ in view.successors(name):
                cases = states[succ].recent_bad_cases if succ in states else []
                if cases:
                    cases_str = "\n".join(
                        f"  - Case {i + 1}:\n    [Problem]: \"{c['question']}\"\n"
                        f"    [Context Snippet]: \"{c['context'][:200]}\"\n"
                        f"    The downstream agent ({view.specs[succ].role}) failed to "
                        "produce a correct/better answer."
                        for i, c in enumerate(cases[:2]))
                    msgs.append(f"Feedback from downstream Agent-{succ} "
                                f"({view.specs[succ].role}):\nYour previous outputs led to "
                                f"failures in the downstream task in the following cases:\n"
                                f"{cases_str}")
            successor_info = "\n\n".join(msgs)

        # ④ 两半样本各提一个候选（原版 2 proposals/层），去重后逐个评估
        role = view.specs[name].role
        cand_prompts = await asyncio.gather(
            self._propose_new_prompt(node["prompt"], qa_of(samples[:mid]), role, successor_info),
            self._propose_new_prompt(node["prompt"], qa_of(samples[mid:]), role, successor_info))
        candidates = sorted(set(cand_prompts))  # sorted：去重后顺序确定，保证可复现

        async def evaluate(cand: str) -> Tuple[str, Dict[str, Any]]:
            if cand == node["prompt"]:  # 变异失败/回退：中性分，不产生新节点
                return cand, {"score": 0.0, "bad_cases": [], "misleading_cases": [],
                              "misalignment_rate": 0.0}
            info = await self._evaluate_candidate(
                view, cand, name, samples, baseline_caches, base_map,
                use_lookahead=self.use_lookahead_score,
                random_sample_set=set(random_samples))
            return cand, info

        results = dict(await asyncio.gather(*[evaluate(c) for c in candidates]))

        # ⑤ score>0 者入 beam（否则保留原节点占位，原版语义）；记录本层最优
        new_nodes: List[Dict[str, Any]] = []
        local_best = {"prompt": node["prompt"], "score": 0.0,
                      "bad_cases": [], "misleading_cases": []}
        rates = []
        for cand in candidates:
            info = results[cand]
            rates.append(info["misalignment_rate"])
            if info["score"] > local_best["score"]:
                local_best = {"prompt": cand, "score": info["score"],
                              "bad_cases": info["bad_cases"],
                              "misleading_cases": info["misleading_cases"]}
            if info["score"] > 0:
                new_nodes.append({"prompt": cand,
                                  "cumulative_score": node["cumulative_score"] + info["score"],
                                  "path": node["path"] + [cand]})
            else:
                new_nodes.append(dict(node))
        return {"nodes": new_nodes,
                "best_prompt": local_best["prompt"],
                "best_cumulative": node["cumulative_score"] + max(0.0, local_best["score"]),
                "best_bad_cases": local_best["bad_cases"],
                "best_misleading_cases": local_best["misleading_cases"],
                "avg_misalignment_rate": sum(rates) / len(rates) if rates else 0.0}

    # --------------- Beam Refresh（原版 _refresh_beam_scores） ---------------

    async def _refresh_beam_scores(self, view: Any, state: AgentOptState,
                                   states: Dict[str, AgentOptState]) -> None:
        if not state.current_beam:
            return
        self._log(f"  [maspo] Beam Refresh: 重评 {state.name} 的 {len(state.current_beam)} 个节点")
        ranking_before = [n["prompt"] for n in sorted(
            state.current_beam, key=lambda x: x["cumulative_score"], reverse=True)]
        samples = self.rng.sample(self.trainset, min(self.eval_batch, len(self.trainset)))
        global_map = {n: states[n].best_overall_node["prompt"] if n in states
                      else view.specs[n].system_prompt for n in view.names}
        baseline_caches = await self._build_baseline_caches(view, samples, global_map)

        async def re_eval(node: Dict[str, Any]) -> Dict[str, Any]:
            if node["prompt"] == global_map[state.name]:
                node["cumulative_score"] = 0.0
                return node
            temp = dict(global_map)
            temp[state.name] = node["prompt"]
            info = await self._evaluate_candidate(  # 无 lookahead：0.3 global + 0.7 local（原版）
                view, node["prompt"], state.name, samples, baseline_caches, temp,
                use_lookahead=False)
            node["cumulative_score"] = info["score"]
            return node

        new_beam = await asyncio.gather(*[re_eval(n) for n in state.current_beam])
        state.current_beam = sorted(new_beam, key=lambda x: x["cumulative_score"], reverse=True)
        ranking_after = [n["prompt"] for n in state.current_beam]
        state.beam_refresh_kendall_scores.append({
            "depth": state.total_layers_explored,
            "kendall_top2_overlap": 1.0 if (ranking_before and ranking_after
                                            and ranking_before[0] == ranking_after[0]) else 0.0,
        })
        top = state.current_beam[0]
        state.best_overall_node = dict(top)
        if top["prompt"] != global_map[state.name]:
            self._log(f"  [maspo] Beam Refresh: {state.name} 锚点切换，"
                      f"新分 {top['cumulative_score']:.3f}")

    # ------------- fixed-rounds 主循环（原版 optimize_all_fixed_rounds） -------------

    async def _optimize_agent_turn(self, view: Any, state: AgentOptState,
                                   states: Dict[str, AgentOptState], rounds: int) -> None:
        if self.use_beam_refresh and state.total_layers_explored > 0 and state.current_beam:
            await self._refresh_beam_scores(view, state, states)
        for _ in range(rounds):
            if state.total_layers_explored >= self.max_total_depth:
                break
            if not state.current_beam:
                state.current_beam = [dict(state.best_overall_node)]
            node_results = await asyncio.gather(*[
                self._process_single_node(view, node, state.name, states)
                for node in state.current_beam])

            all_next: List[Dict[str, Any]] = []
            best = dict(state.best_overall_node)
            best_bad: List[Dict[str, Any]] = []
            best_misleading: List[Dict[str, Any]] = []
            for res in node_results:
                all_next.extend(res["nodes"])
                if res["best_cumulative"] > best["cumulative_score"]:
                    best = {"prompt": res["best_prompt"],
                            "cumulative_score": res["best_cumulative"], "path": []}
                    best_bad = res["best_bad_cases"]
                    best_misleading = res["best_misleading_cases"]
            all_next.sort(key=lambda x: x["cumulative_score"], reverse=True)
            state.current_beam = all_next[: self.beam_width]
            state.misalignment_rates_per_depth.append(
                sum(r["avg_misalignment_rate"] for r in node_results) / len(node_results))

            if best["cumulative_score"] > state.best_overall_node["cumulative_score"] + 1e-6:
                state.best_overall_node = best
                if best_bad:
                    state.recent_bad_cases = best_bad
                if best_misleading:
                    state.misleading_cases = best_misleading
            state.total_layers_explored += 1
            self._log(f"  [maspo] {state.name}: depth "
                      f"{state.total_layers_explored}/{self.max_total_depth} "
                      f"best={state.best_overall_node['cumulative_score']:.3f}")

    async def _optimize_with_limits(self, view: Any) -> Tuple[Dict[str, str], Dict[str, Any]]:
        """在当前事件循环内包并发限流（Semaphore 绑定 loop，不能建在循环外）。"""
        sem = asyncio.Semaphore(self.max_concurrency)
        self.agent_llm = _limited(self._raw_agent_llm, sem)
        self.evaluator_llm = _limited(self._raw_evaluator_llm, sem)
        self.proposer_llm = _limited(self._raw_proposer_llm or self._raw_evaluator_llm, sem)
        try:
            return await self._optimize_all_fixed_rounds(view)
        finally:
            self.agent_llm = None
            self.evaluator_llm = None
            self.proposer_llm = None

    def _ckpt_path(self) -> Optional[str]:
        if not self.prompt_file:
            return None
        stem, _ext = os.path.splitext(self.prompt_file)
        return stem + "_ckpt.json"

    def _load_or_seed_states(self, view: Any) -> Dict[str, AgentOptState]:
        """有 checkpoint（上次中断的现场）则恢复，否则播种。

        注意：恢复点之后的采样序列与一次性跑完不逐位一致（RNG 状态不落盘），方法语义不变。
        节点名与 checkpoint 不符时显式报错（删除 *_ckpt.json 可重新开始）。
        """
        path = self._ckpt_path()
        if not path or not os.path.isfile(path):
            return {name: AgentOptState.seeded(name, view.specs[name].system_prompt)
                    for name in view.names}
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if sorted(data.get("states", {})) != sorted(view.names):
            raise ValueError(
                f"{path}: checkpoint 的节点集与当前图不符"
                f"（{sorted(data.get('states', {}))} vs {sorted(view.names)}）；"
                "若是过期现场请删除该文件后重跑")
        states = {n: AgentOptState.from_dict(d) for n, d in data["states"].items()}
        done = {n: s.total_layers_explored for n, s in states.items()}
        self._log(f"  [maspo] 从 checkpoint 恢复：{path}（各 agent 已探索层数 {done}）")
        return states

    def _save_ckpt(self, states: Dict[str, AgentOptState]) -> None:
        path = self._ckpt_path()
        if not path:
            return
        payload = {"states": {n: s.to_dict() for n, s in states.items()},
                   "meta": {"max_total_depth": self.max_total_depth,
                            "trainset_size": len(self.trainset), "seed": self.seed}}
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        os.replace(tmp, path)

    async def _optimize_all_fixed_rounds(self, view: Any
                                         ) -> Tuple[Dict[str, str], Dict[str, Any]]:
        states = self._load_or_seed_states(view)
        while any(s.total_layers_explored < self.max_total_depth for s in states.values()):
            for name in view.names:  # 拓扑序坐标上升（原版 topo_order 轮转）
                state = states[name]
                remaining = self.max_total_depth - state.total_layers_explored
                if remaining <= 0:
                    continue
                await self._optimize_agent_turn(view, state, states,
                                                min(self.rounds_per_turn, remaining))
                self._save_ckpt(states)  # 每个 agent 轮次落盘：外部故障可断点续跑
        ckpt = self._ckpt_path()
        if ckpt and os.path.isfile(ckpt):
            os.remove(ckpt)  # 正常完成：现场文件功成身退
        prompt_map = {name: states[name].best_overall_node["prompt"] for name in view.names}
        statistics = {
            "misalignment_rates": {
                "per_agent": {n: s.misalignment_rates_per_depth
                              for n, s in states.items() if n != view.terminal},
            },
            "kendall_scores": {
                "per_agent": {n: [r["kendall_top2_overlap"]
                                  for r in s.beam_refresh_kendall_scores]
                              for n, s in states.items()},
            },
            "final_scores": {n: s.best_overall_node["cumulative_score"]
                             for n, s in states.items()},
        }
        return prompt_map, statistics
