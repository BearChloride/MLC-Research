"""评分 + 结果落盘（迁移自 benchs/metrics.py + math_parsing_util.py）。

评分类型（kind）：
  - "mc":    选择题；预测里出现 gold 选项字母 或 gold 文本即算对。用于 medqa/arc/openbookqa。
  - "exact": 数值/短答规整后精确匹配。用于 gsm8k。
  - "aime":  数学答案等价比较（数值 + sympy 符号），支持整数/分式/根式/带符号式子。用于 aime。
  - "f1":    token 级 F1（对 gold 列表取最大），MemGAS 口径。用于 locomo/longmemeval。
  - "human_eval": HumanEval 代码单测。
  - "gaia": GAIA 官方风格短答案规整。
  - "mas_audit": AFTraj safe/unsafe + decisive step/agent audit。
  - "mas_failure_taxonomy": MAST taxonomy label F1。
  - "mas_deviation": Open Agent Traces deviation detection。
  - "mas_instruction_decay" / "mas_tracer_durability" /
    "mas_consensus_pollution" / "mas_context_leakage": AgentCollabBench metrics。

CLAUDE.md §9：所有跑分必须同时报性能与成本——`result_dir`/`write_results` 落 metrics+outputs+config。

⚠️ 惰性导入：`score_aime` 用到的 math_parsing_util
（依赖 sympy/regex/latex2sympy2-extended/word2number）与 `write_results` 用到的 yaml
都在函数内部 import，保证本模块在无这些库时也可被 import（黄金法则 4）。
"""

from __future__ import annotations

import json
import os
import re
import string
from collections import Counter
from typing import Any, List, Optional

BINARY_SCORERS = {
    "exact",
    "aime",
    "mc",
    "human_eval",
    "gaia",
    "mas_audit",
    "mas_deviation",
    "mas_instruction_decay",
    "mas_tracer_durability",
    "mas_consensus_pollution",
    "mas_context_leakage",
}


def is_binary_scorer(kind: str | None) -> bool:
    return bool(kind in BINARY_SCORERS)


# ---- 规整 / 评分 ----
def _norm(s: str) -> str:
    # 统一小写、去标点、去冠词(a/an/the)、压空白——SQuAD 式规整，使匹配更鲁棒
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def _last_number(s: str) -> Optional[str]:
    # 抽取字符串里最后一个数字（去千分位逗号）；数学题答案通常在末尾
    nums = re.findall(r"-?\d[\d,]*\.?\d*", s.replace(",", ""))
    return nums[-1].rstrip(".") if nums else None


def score_exact(pred: str, gold: str) -> float:
    g = _last_number(gold) or _norm(gold)
    p = _last_number(pred)
    if p is not None and g is not None and p == g:
        return 1.0  # 数值匹配优先
    return 1.0 if _norm(gold) and _norm(gold) in _norm(pred) else 0.0  # 否则退化为子串包含


def score_mc(pred: str, gold_text: str, gold_letter: Optional[str]) -> float:
    np_ = _norm(pred)
    if gold_letter:
        # 在预测开头 40 字符内匹配独立的选项字母，如 "A" / "(A)" / "answer: b"
        if re.search(rf"\b{re.escape(gold_letter.lower())}\b", np_[:40]):
            return 1.0
    return 1.0 if _norm(gold_text) and _norm(gold_text) in np_ else 0.0  # 或选项文本出现在预测里


def score_f1(pred: str, golds: List[str]) -> float:
    # token 级 F1，对多个 gold 取最大值（locomo 一题可有多个可接受答案）
    pt = _norm(pred).split()
    best = 0.0
    for g in golds:
        gt = _norm(g).split()
        if not pt or not gt:
            continue
        common = Counter(pt) & Counter(gt)  # 词频交集 = 命中词数
        n = sum(common.values())
        if n == 0:
            continue
        prec, rec = n / len(pt), n / len(gt)
        best = max(best, 2 * prec * rec / (prec + rec))
    return best


def score_aime(pred: str, gold: str) -> float:
    """AIME/数学评分：两边各自归一化后做数学等价比较（数值 + sympy 符号）。

    math_parsing_util 在此惰性导入（它依赖 sympy/regex/latex2sympy2-extended/word2number）。
    """
    # 数学答案抽取/等价比较工具（math_parsing_util 里函数名无 math_ 前缀）
    from .math_parsing_util import extract_answer, math_equal, strip_answer_string

    def _math_norm(s: str) -> str:
        r"""归一化数学答案。

        含 \boxed/"answer is" 标记时用 extract_answer 抽取（它能正确取出 \boxed{} 里的表达式）；
        否则只做 strip_answer_string 归一化。aime 提取器已剥掉 \boxed，pred 多是裸表达式，故走
        strip 分支。
        """
        s = str(s)
        return extract_answer(s) if "boxed" in s else strip_answer_string(s)

    p = _math_norm(pred)
    g = _math_norm(gold)
    return 1.0 if math_equal(p, g) else 0.0


def score_gaia(pred: str, gold: str) -> float:
    from .benchmarks.gaia import gaia_question_scorer

    return 1.0 if gaia_question_scorer(str(pred), str(gold)) else 0.0


def score_human_eval(pred: str, gold) -> float:
    from .benchmarks.human_eval import evaluate_answer

    if not isinstance(gold, dict) or "test" not in gold or "entry_point" not in gold:
        return 0.0
    return 1.0 if evaluate_answer(pred, gold).get("success") else 0.0


def score_mas_audit(pred: str, gold) -> float:
    from .benchmarks.aftraj import score_audit

    return score_audit(pred, gold if isinstance(gold, dict) else {})


def score_mas_failure_taxonomy(pred: str, gold) -> float:
    from .benchmarks.mast_data import score_taxonomy

    return score_taxonomy(pred, gold if isinstance(gold, dict) else {"labels": gold})


def score_mas_deviation(pred: str, gold) -> float:
    from .benchmarks.open_agent_traces import score_deviation

    return score_deviation(pred, gold if isinstance(gold, dict) else {})


def score_agent_collab(pred: str, gold, metric: str | None = None) -> float:
    from .benchmarks.agent_collab import score_agent_collab as _score_agent_collab

    return _score_agent_collab(
        pred, gold if isinstance(gold, dict) else {"expected": gold}, metric=metric
    )


def score(kind: str, pred: str, gold) -> float:
    # 统一评分入口：按 kind 分发；gold 可能是 str / (text, letter) / list
    if kind == "exact":
        return score_exact(pred, gold if isinstance(gold, str) else gold[0])
    if kind == "aime":
        return score_aime(pred, gold if isinstance(gold, str) else gold[0])
    if kind == "mc":
        gt, gl = gold if isinstance(gold, (list, tuple)) else (gold, None)  # (选项文本, 选项字母)
        return score_mc(pred, gt, gl)
    if kind == "f1":
        return score_f1(pred, gold if isinstance(gold, list) else [gold])
    if kind == "gaia":
        return score_gaia(pred, gold if isinstance(gold, str) else gold[0])
    if kind == "human_eval":
        return score_human_eval(pred, gold)
    if kind == "mas_audit":
        return score_mas_audit(pred, gold)
    if kind == "mas_failure_taxonomy":
        return score_mas_failure_taxonomy(pred, gold)
    if kind == "mas_deviation":
        return score_mas_deviation(pred, gold)
    if kind == "mas_instruction_decay":
        return score_agent_collab(pred, gold, metric="IDR")
    if kind == "mas_tracer_durability":
        return score_agent_collab(pred, gold, metric="RTD")
    if kind == "mas_consensus_pollution":
        return score_agent_collab(pred, gold, metric="CPR")
    if kind == "mas_context_leakage":
        return score_agent_collab(pred, gold, metric="CLC")
    raise ValueError(kind)


def score_details(kind: str, pred: str, gold) -> dict:
    """Return a structured score breakdown when a benchmark supports one."""
    if kind == "mas_audit":
        from .benchmarks.aftraj import audit_score_details

        return audit_score_details(pred, gold if isinstance(gold, dict) else {})
    if kind == "mas_failure_taxonomy":
        from .benchmarks.mast_data import taxonomy_score_details

        return taxonomy_score_details(pred, gold if isinstance(gold, dict) else {"labels": gold})
    if kind == "mas_deviation":
        from .benchmarks.open_agent_traces import deviation_score_details

        return deviation_score_details(pred, gold if isinstance(gold, dict) else {})
    if kind == "mas_instruction_decay":
        from .benchmarks.agent_collab import agent_collab_score_details

        return agent_collab_score_details(
            pred, gold if isinstance(gold, dict) else {"expected": gold}, metric="IDR"
        )
    if kind == "mas_tracer_durability":
        from .benchmarks.agent_collab import agent_collab_score_details

        return agent_collab_score_details(
            pred, gold if isinstance(gold, dict) else {"expected": gold}, metric="RTD"
        )
    if kind == "mas_consensus_pollution":
        from .benchmarks.agent_collab import agent_collab_score_details

        return agent_collab_score_details(
            pred, gold if isinstance(gold, dict) else {"expected": gold}, metric="CPR"
        )
    if kind == "mas_context_leakage":
        from .benchmarks.agent_collab import agent_collab_score_details

        return agent_collab_score_details(
            pred, gold if isinstance(gold, dict) else {"expected": gold}, metric="CLC"
        )
    value = score(kind, pred, gold)
    return {"score": value}


# ---- run-level 汇总 ----
def _sample_number(sample: dict[str, Any], *names: str, default=0):
    for name in names:
        value = sample.get(name)
        if value is not None:
            return value
    return default


def _sample_count(sample: dict[str, Any], *names: str, list_field: str | None = None) -> int:
    value = _sample_number(sample, *names, default=None)
    if value is not None:
        return int(value)
    if list_field:
        items = sample.get(list_field)
        if isinstance(items, list):
            return len(items)
    return 0


def aggregate_samples(samples: List[dict], run_info: Optional[dict] = None) -> dict:
    """Aggregate case-level ``outputs.jsonl`` records into run-level metrics.

    This is intentionally independent of model execution, so partial or merged
    ``outputs.jsonl`` files can be analyzed without re-running inference.
    """
    run_info = run_info or {}
    n = len(samples)
    first = samples[0] if samples else {}
    kind = run_info.get("scorer_kind") or first.get("scorer_kind") or first.get("kind")
    task = run_info.get("task") or first.get("task")
    method = run_info.get("method") or first.get("method")

    score_sum = 0.0
    num_correct = 0
    model_call_count_sum = 0
    message_count_sum = 0
    input_positions_sum = 0
    text_input_tokens_sum = 0
    latent_prefix_positions_sum = 0
    output_tokens_sum = 0
    generation_latency_sum = 0.0
    case_wall_time_sum = 0.0
    tool_call_count_sum = 0
    tool_error_count_sum = 0

    for sample in samples:
        score_value = float(sample.get("score", sample.get("correct", 0.0)) or 0.0)
        score_sum += score_value
        if is_binary_scorer(kind):
            is_correct = sample.get("is_correct")
            num_correct += int(bool(is_correct) if is_correct is not None else score_value == 1.0)

        model_call_count_sum += _sample_count(
            sample, "num_model_calls", "model_call_count", list_field="model_calls"
        )
        message_count_sum += _sample_count(
            sample, "num_messages", "message_count", "n_messages", list_field="messages"
        )
        input_positions_sum += int(
            _sample_number(
                sample, "sum_input_total_positions", "input_positions_total", "cost_prompt_pos"
            )
        )
        text_input_tokens_sum += int(
            _sample_number(sample, "sum_input_text_tokens", "text_input_tokens_total")
        )
        latent_prefix_positions_sum += int(
            _sample_number(sample, "sum_input_latent_positions", "latent_prefix_positions_total")
        )
        output_tokens_sum += int(
            _sample_number(sample, "sum_output_text_tokens", "output_tokens_total", "gen_tokens")
        )
        generation_latency_sum += float(
            _sample_number(
                sample, "sum_model_latency_s", "model_generation_latency_s_total", "latency_s"
            )
        )
        case_wall_time_sum += float(_sample_number(sample, "case_wall_time_s"))
        tool_call_count_sum += _sample_count(
            sample, "num_tool_calls", "tool_call_count", list_field="tool_calls"
        )
        tool_error_count_sum += int(_sample_number(sample, "num_tool_errors", "tool_error_count"))

    score_mean = round(score_sum / n, 4) if n else None
    metrics = {
        "schema_version": 2,
        "model": run_info.get("model"),
        "backend_provider": run_info.get("backend_provider"),
        "method": method,
        "team": run_info.get("team"),
        "task": task,
        "probe": run_info.get("probe", "mas"),
        "memory": run_info.get("memory"),
        "router": run_info.get("router"),
        "summary_scope": run_info.get("summary_scope", "run"),
        "aggregation_scope": "case_aggregates",
        "num_cases": n,
        "case_count": n,
        "scorer_kind": kind,
        "mean_score": score_mean,
        "score_mean": score_mean,
        "accuracy": score_mean if is_binary_scorer(kind) else None,
        "num_correct": num_correct if is_binary_scorer(kind) else None,
        "mean_model_calls_per_case": round(model_call_count_sum / n, 2) if n else 0,
        "mean_messages_per_case": round(message_count_sum / n, 2) if n else 0,
        "mean_input_total_positions_per_case": round(input_positions_sum / n, 1) if n else 0,
        "mean_input_text_tokens_per_case": round(text_input_tokens_sum / n, 1) if n else 0,
        "mean_input_latent_positions_per_case": round(latent_prefix_positions_sum / n, 1)
        if n
        else 0,
        "mean_output_text_tokens_per_case": round(output_tokens_sum / n, 1) if n else 0,
        "mean_model_latency_s_per_case": round(generation_latency_sum / n, 3) if n else 0,
        "mean_model_latency_s_per_call": round(generation_latency_sum / model_call_count_sum, 3)
        if model_call_count_sum
        else 0,
        "mean_case_wall_time_s": round(case_wall_time_sum / n, 3) if n else 0,
        "mean_tool_calls_per_case": round(tool_call_count_sum / n, 2) if n else 0,
        "mean_tool_errors_per_case": round(tool_error_count_sum / n, 2) if n else 0,
        "total_model_calls": model_call_count_sum,
        "total_messages": message_count_sum,
        "total_tool_calls": tool_call_count_sum,
        "total_tool_errors": tool_error_count_sum,
        "total_input_total_positions": input_positions_sum,
        "total_input_text_tokens": text_input_tokens_sum,
        "total_input_latent_positions": latent_prefix_positions_sum,
        "total_output_text_tokens": output_tokens_sum,
        "total_model_latency_s": round(generation_latency_sum, 3),
        "total_case_wall_time_s": round(case_wall_time_sum, 3),
        "tool_call_count": tool_call_count_sum,
        "tool_error_count": tool_error_count_sum,
        "model_calls_per_case_mean": round(model_call_count_sum / n, 2) if n else 0,
        "messages_per_case_mean": round(message_count_sum / n, 2) if n else 0,
        "tool_calls_per_case_mean": round(tool_call_count_sum / n, 2) if n else 0,
        "tool_errors_per_case_mean": round(tool_error_count_sum / n, 2) if n else 0,
        "input_positions_per_case_mean": round(input_positions_sum / n, 1) if n else 0,
        "text_input_tokens_per_case_mean": round(text_input_tokens_sum / n, 1) if n else 0,
        "latent_prefix_positions_per_case_mean": round(latent_prefix_positions_sum / n, 1)
        if n
        else 0,
        "output_tokens_per_case_mean": round(output_tokens_sum / n, 1) if n else 0,
        "model_generation_latency_s_per_case_mean": round(generation_latency_sum / n, 3)
        if n
        else 0,
        "model_generation_latency_s_per_call_mean": round(
            generation_latency_sum / model_call_count_sum, 3
        )
        if model_call_count_sum
        else 0,
        # Backward-compatible aliases for older result readers.
        "n": n,
        "quality": score_mean,
        "quality_metric": kind,
        "cost_prompt_pos_mean": round(input_positions_sum / n, 1) if n else 0,
        "gen_tokens_mean": round(output_tokens_sum / n, 1) if n else 0,
        "latency_s_mean": round(generation_latency_sum / n, 3) if n else 0,
        "messages_mean": round(message_count_sum / n, 2) if n else 0,
    }

    # ---- pass@K：当某些 case 有多份采样（run_mas --samples K）时，按 case 分组补充 ----
    # 纯附加、不改上面任一字段；仅当检测到「样本数 > 去重 case 数」时才写入，K=1 运行完全不变。
    # pass@1 = 各 case 内 K 份得分均值再对 case 求平均；pass@K = 各 case best-of-K 得分再平均
    # （二值打分下即「命中率 / 任一正确率」，f1 等连续打分下为最优采样均值）。
    by_case: dict = {}
    for sample in samples:
        cid = str(sample.get("case_id", ""))
        sv = float(sample.get("score", sample.get("correct", 0.0)) or 0.0)
        by_case.setdefault(cid, []).append(sv)
    distinct_cases = len(by_case)
    if distinct_cases and n > distinct_cases:
        k_max = max(len(v) for v in by_case.values())
        pass1 = round(sum(sum(v) / len(v) for v in by_case.values()) / distinct_cases, 4)
        passk = round(sum(max(v) for v in by_case.values()) / distinct_cases, 4)
        passk_block = {
            "num_distinct_cases": distinct_cases,
            "samples_per_case": round(n / distinct_cases, 2),
            "max_samples_per_case": k_max,
            "pass@1": pass1,
            f"pass@{k_max}": passk,
        }
        metrics.update(passk_block)
        metrics["pass_at_k"] = passk_block
    return metrics


# ---- 落盘 ----
def result_dir(model: str, method: str, task: str, root: Optional[str] = None) -> str:
    # 按 CLAUDE.md §9 布局拼出叶子目录并创建：root/[model]/[method]/[task]/
    if root is None:
        from .benchmarks.common import runs_root

        root = runs_root()
    d = os.path.join(root, model, method, task)
    os.makedirs(d, exist_ok=True)
    return d


def read_outputs_jsonl(path: str) -> List[dict]:
    samples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def write_outputs(out_dir: str, samples: List[dict]) -> None:
    with open(os.path.join(out_dir, "outputs.jsonl"), "w", encoding="utf-8") as f:
        for s in samples:
            # 每行一个样本（含 routing_trace+transcript）
            f.write(json.dumps(s, ensure_ascii=False) + "\n")


def write_metrics(out_dir: str, metrics: dict) -> None:
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)  # Pareto 主图的数据源


def write_config(out_dir: str, config: dict) -> None:
    _dump_config(os.path.join(out_dir, "config.yaml"), config)


def write_results(out_dir: str, samples: List[dict], metrics: dict, config: dict):
    # 三个产物：每样本明细 / 汇总指标 / 可复现配置。yaml 惰性导入（写 config 时才需要）。
    write_outputs(out_dir, samples)
    write_metrics(out_dir, metrics)
    write_config(out_dir, config)


def _dump_config(path: str, config: dict) -> None:
    """落配置快照：优先 yaml（可读），无 yaml 时退回 JSON（保证离线也能落盘）。"""
    try:
        import yaml  # 惰性导入

        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)
    except ImportError:
        with open(path + ".json", "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
