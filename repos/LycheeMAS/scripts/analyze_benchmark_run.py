"""Analyze or score an existing LycheeMAS benchmark run without inference."""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any


_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from lychee_mas.eval import metrics as M  # noqa: E402


def _get(cfg: dict, dotted: str, default=None):
    cur = cfg
    for key in dotted.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _load_config(path: str | None) -> dict[str, Any]:
    if not path or not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        text = f.read()
    try:
        import yaml

        return yaml.safe_load(text) or {}
    except Exception:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {}


def _case_id(item: dict, index: int) -> str:
    metadata = item.get("metadata") or {}
    return str(metadata.get("task_id") or metadata.get("safe_task_id") or index)


def _run_info(snapshot: dict[str, Any], samples: list[dict], args) -> dict[str, Any]:
    resolved = snapshot.get("resolved") if isinstance(snapshot, dict) else {}
    resolved = resolved if isinstance(resolved, dict) else {}
    cfg = snapshot.get("config") if isinstance(snapshot, dict) else {}
    cfg = cfg if isinstance(cfg, dict) else {}
    first = samples[0] if samples else {}
    return {
        "model": resolved.get("model") or resolved.get("model_tag") or _get(cfg, "backend.model_tag"),
        "backend_provider": resolved.get("backend_provider") or _get(cfg, "backend.provider"),
        "method": resolved.get("method") or first.get("method") or _get(cfg, "router.method"),
        "team": resolved.get("team") or _get(cfg, "run.team"),
        "task": args.task or resolved.get("task") or first.get("task") or _get(cfg, "run.task"),
        "probe": "mas",
        "memory": resolved.get("memory") or _get(cfg, "memory.name") or "cdm",
        "router": resolved.get("router"),
        "scorer_kind": args.kind or resolved.get("scorer_kind") or first.get("scorer_kind"),
        "start_index": args.start_index if args.start_index is not None else resolved.get("start_index", 0),
        "expected_num_cases": resolved.get("n"),
    }


def _load_gold_items(task: str, start_index: int, count: int) -> tuple[list[dict], dict[str, dict]]:
    from lychee_mas.eval.benchmarks import load as load_task

    data = load_task(task, n=start_index + count)
    data = data[start_index:start_index + count]
    by_id = {_case_id(item, start_index + offset): item for offset, item in enumerate(data)}
    return data, by_id


def _score_predictions(
    predictions: list[dict],
    gold_data: list[dict],
    gold_by_id: dict[str, dict],
    *,
    kind: str | None,
) -> list[dict]:
    scored = []
    for offset, pred in enumerate(predictions):
        item = gold_by_id.get(str(pred.get("case_id")))
        if item is None and offset < len(gold_data):
            item = gold_data[offset]
        if item is None:
            raise SystemExit(f"Cannot find gold item for prediction case_id={pred.get('case_id')!r}")
        scorer_kind = kind or pred.get("scorer_kind") or item.get("kind")
        details = M.score_details(scorer_kind, pred.get("final_answer", ""), item.get("gold"))
        score = float(details.get("score", 0.0))
        sample = dict(pred)
        sample.update({
            "scorer_kind": scorer_kind,
            "gold": item.get("gold"),
            "score": score,
            "score_details": details,
            "is_correct": bool(score == 1.0) if M.is_binary_scorer(scorer_kind) else None,
            "correct": score,
        })
        scored.append(sample)
    return scored


def _rescore_outputs(samples: list[dict]) -> None:
    for sample in samples:
        kind = sample.get("scorer_kind")
        if not kind:
            continue
        details = M.score_details(kind, sample.get("final_answer", ""), sample.get("gold"))
        score = float(details.get("score", 0.0))
        sample["score_details"] = details
        sample["score"] = score
        sample["is_correct"] = bool(score == 1.0) if M.is_binary_scorer(kind) else None
        sample["correct"] = score


def _with_partial_flags(metrics: dict, run_info: dict, sample_count: int) -> dict:
    expected_n = run_info.get("expected_num_cases")
    if isinstance(expected_n, int):
        metrics["expected_num_cases"] = expected_n
        metrics["is_partial"] = sample_count < expected_n
    return metrics


def _resolve_run_dir(args) -> str | None:
    return args.run_dir_opt or args.run_dir


def _score_prediction_mode(args, run_dir: str, config_path: str) -> dict:
    predictions_path = args.predictions or os.path.join(run_dir, "predictions.jsonl")
    predictions_path = os.path.abspath(predictions_path)
    if not os.path.exists(predictions_path):
        raise SystemExit(f"predictions.jsonl not found: {predictions_path}")

    predictions = M.read_outputs_jsonl(predictions_path)
    snapshot = _load_config(config_path)
    run_info = _run_info(snapshot, predictions, args)
    task = run_info.get("task")
    kind = run_info.get("scorer_kind")
    if not task:
        raise SystemExit("Cannot infer task; pass --task")
    start_index = int(run_info.get("start_index") or 0)

    # pass@K：一个 case 可能对应多份 prediction（k_index），按去重 case 数加载 gold 对齐
    num_distinct = len({str(p.get("case_id")) for p in predictions}) or len(predictions)
    gold_data, gold_by_id = _load_gold_items(task, start_index, num_distinct)
    scored = _score_predictions(predictions, gold_data, gold_by_id, kind=kind)
    metrics = _with_partial_flags(
        M.aggregate_samples(scored, run_info=run_info),
        run_info,
        len(scored),
    )
    if not args.no_write:
        M.write_outputs(run_dir, scored)
        M.write_metrics(run_dir, metrics)
    return metrics


def _analyze_outputs_mode(args, run_dir: str, config_path: str) -> dict:
    outputs_path = args.outputs or os.path.join(run_dir, "outputs.jsonl")
    outputs_path = os.path.abspath(outputs_path)
    if not os.path.exists(outputs_path):
        prediction_hint = os.path.join(run_dir, "predictions.jsonl")
        if os.path.exists(prediction_hint):
            raise SystemExit(
                f"outputs.jsonl not found: {outputs_path}; use --score-predictions first"
            )
        raise SystemExit(f"outputs.jsonl not found: {outputs_path}")

    samples = M.read_outputs_jsonl(outputs_path)
    if args.rescore:
        _rescore_outputs(samples)
        if args.rewrite_outputs:
            M.write_outputs(run_dir, samples)

    run_info = _run_info(_load_config(config_path), samples, args)
    metrics = _with_partial_flags(
        M.aggregate_samples(samples, run_info=run_info),
        run_info,
        len(samples),
    )
    if args.write:
        M.write_metrics(run_dir, metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze a benchmark run: score predictions or summarize outputs."
    )
    parser.add_argument("run_dir", nargs="?", help="Run directory containing predictions.jsonl or outputs.jsonl")
    parser.add_argument("--run-dir", dest="run_dir_opt", default=None)
    parser.add_argument("--predictions", default=None, help="Explicit predictions.jsonl path")
    parser.add_argument("--outputs", default=None, help="Explicit outputs.jsonl path")
    parser.add_argument("--config", default=None, help="Explicit config.yaml/config.json path")
    parser.add_argument("--score-predictions", action="store_true",
                        help="Score predictions.jsonl with benchmark gold and write outputs/metrics")
    parser.add_argument("--task", default=None, help="Override task used to load gold data")
    parser.add_argument("--kind", default=None, help="Override scorer kind")
    parser.add_argument("--start-index", type=int, default=None,
                        help="Override dataset start index for gold alignment")
    parser.add_argument("--write", action="store_true",
                        help="When analyzing outputs, write regenerated metrics.json")
    parser.add_argument("--no-write", action="store_true",
                        help="When scoring predictions, only print metrics")
    parser.add_argument("--rescore", action="store_true",
                        help="When analyzing outputs, recompute per-case score/score_details")
    parser.add_argument("--rewrite-outputs", action="store_true",
                        help="With --rescore, rewrite outputs.jsonl with refreshed scores")
    args = parser.parse_args()

    run_dir = _resolve_run_dir(args)
    if not run_dir:
        if args.predictions:
            run_dir = os.path.dirname(os.path.abspath(args.predictions))
        elif args.outputs:
            run_dir = os.path.dirname(os.path.abspath(args.outputs))
    if not run_dir:
        raise SystemExit("需要提供 run_dir、--predictions 或 --outputs")
    run_dir = os.path.abspath(run_dir)
    config_path = args.config or os.path.join(run_dir, "config.yaml")

    if args.score_predictions:
        metrics = _score_prediction_mode(args, run_dir, config_path)
        verb = "score"
    else:
        metrics = _analyze_outputs_mode(args, run_dir, config_path)
        verb = "analyze"

    passk = metrics.get("pass_at_k")
    passk_str = ""
    if passk:
        k_max = passk["max_samples_per_case"]
        passk_str = (
            f" | pass@1={passk['pass@1']} pass@{k_max}={passk[f'pass@{k_max}']} "
            f"(cases={passk['num_distinct_cases']}×{passk['samples_per_case']})"
        )
    print(
        f"[{verb}] cases={metrics['num_cases']} score_mean={metrics['score_mean']} "
        f"model_calls/case={metrics['mean_model_calls_per_case']} "
        f"tool_calls/case={metrics['mean_tool_calls_per_case']}{passk_str} -> {run_dir}",
        flush=True,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
