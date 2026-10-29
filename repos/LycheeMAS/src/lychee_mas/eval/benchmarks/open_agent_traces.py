"""Open Agent Traces loader for trace-level MAS observability evaluation."""

from __future__ import annotations

import fnmatch
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

from .common import (
    copy_raw_to_prepared,
    download_hf_files,
    list_hf_dataset_files,
    log_download_source,
    prepared_benchmark_dir,
    raw_source_candidates,
    raw_source_dir,
    remove_path,
    restore_prepared_from_raw,
)
from .source_catalog import fallback_specs, provider_ids, source_backend_order

_FALLBACK = fallback_specs("open_agent_traces")[0]
REPO_ID = provider_ids("open_agent_traces", "huggingface")[0]
ALLOW_PATTERNS = ["data/**", "ocel/**", "README*", ".gitattributes"]


def source_dir(root: str | Path | None = None) -> Path:
    return Path(root) if root is not None else prepared_benchmark_dir("open_agent_traces")


def _download_huggingface(src: Path) -> Path:
    from huggingface_hub import snapshot_download

    raw = raw_source_dir("open_agent_traces", "huggingface", REPO_ID)
    log_download_source("open_agent_traces", "huggingface", REPO_ID, raw)
    try:
        snapshot_download(
            repo_id=REPO_ID, repo_type="dataset", local_dir=str(raw), allow_patterns=ALLOW_PATTERNS
        )
    except Exception:
        files = _matching_hf_files()
        if not files:
            raise RuntimeError("no Open Agent Traces parquet files found on HuggingFace")
        download_hf_files(REPO_ID, files, str(raw))
    copy_raw_to_prepared(raw, src)
    return src


def _matching_hf_files() -> list[str]:
    files = list_hf_dataset_files(REPO_ID)
    patterns = tuple(_FALLBACK["patterns"])
    return [path for path in files if any(fnmatch.fnmatch(path, pattern) for pattern in patterns)]


def _download_modelscope(src: Path) -> Path:
    dataset_ids = provider_ids("open_agent_traces", "modelscope")
    if not any(dataset_ids):
        raise RuntimeError(
            "no known ModelScope mirror for Open Agent Traces; set "
            "LYCHEE_OPENAGENTTRACES_MODELSCOPE_ID or use source=huggingface"
        )
    errors: list[str] = []
    for dataset_id in [x for x in dataset_ids if x]:
        try:
            from modelscope.hub.snapshot_download import snapshot_download

            raw = raw_source_dir("open_agent_traces", "modelscope", dataset_id)
            log_download_source("open_agent_traces", "modelscope", dataset_id, raw)
            snapshot_download(
                dataset_id, repo_type="dataset", local_dir=str(raw), allow_patterns=ALLOW_PATTERNS
            )
            copy_raw_to_prepared(raw, src)
            return src
        except Exception as exc:  # pragma: no cover - provider/network dependent
            errors.append(f"{dataset_id}: {exc}")
    raise RuntimeError("ModelScope Open Agent Traces download failed; " + " | ".join(errors))


def _has_source(src: Path) -> bool:
    return bool(_parquet_files(src))


def ensure_source(
    root: str | Path | None = None,
    force_download: bool = False,
    source: str | None = None,
) -> Path:
    src = source_dir(root)
    if _has_source(src) and not force_download:
        return src
    if not force_download:
        restored = restore_prepared_from_raw(
            "open_agent_traces",
            src,
            raw_source_candidates("open_agent_traces", _backend_order(source)),
            ready=_has_source,
        )
        if restored is not None:
            return restored
    if force_download and src.exists():
        remove_path(src)
    src.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    for backend in _backend_order(source):
        try:
            return (
                _download_modelscope(src) if backend == "modelscope" else _download_huggingface(src)
            )
        except Exception as exc:
            errors.append(f"{backend}: {exc}")
    raise RuntimeError("Failed to prepare Open Agent Traces; " + " | ".join(errors))


def _backend_order(source: Optional[str]) -> list[str]:
    return source_backend_order("open_agent_traces", source)


def _parquet_files(src: Path) -> list[Path]:
    data_dir = src / "data"
    return sorted(path for path in data_dir.glob("**/*.parquet") if path.is_file())


def load_source_events(
    *,
    root: str | Path | None = None,
    force_download: bool = False,
) -> list[dict[str, Any]]:
    src = ensure_source(root, force_download=force_download)
    files = _parquet_files(src)
    if not files:
        raise FileNotFoundError(f"No Open Agent Traces parquet files under {src / 'data'}")
    from datasets import load_dataset

    events: list[dict[str, Any]] = []
    for path in files:
        dataset = load_dataset("parquet", data_files={"train": [str(path)]}, split="train")
        for row in dataset:
            event = dict(row)
            event.setdefault("domain", _domain_for_event_file(src, event, path))
            events.append(event)
    return events


def _domain_for_event_file(
    src: Path, event: dict[str, Any], path: Path | None = None
) -> str | None:
    value = event.get("domain")
    if value:
        return str(value)
    if path is not None:
        try:
            rel = path.relative_to(src / "data")
            if len(rel.parts) > 1:
                return rel.parts[0]
        except ValueError:
            pass
    return None


def _group_runs(events: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        run_id = str(
            event.get("run_id")
            or event.get("case_id")
            or event.get("workflow_id")
            or event.get("trace_id")
            or event.get("process_instance_id")
            or event.get("ocel_id")
            or "unknown"
        )
        grouped[run_id].append(event)
    runs = list(grouped.values())
    for run in runs:
        run.sort(key=lambda row: row.get("event_index") or row.get("timestamp") or 0)
    return runs


def _event_line(event: dict[str, Any]) -> str:
    event_type = event.get("event_type") or event.get("activity") or event.get("type") or "event"
    role = event.get("agent_role") or event.get("agent") or event.get("agent_name") or ""
    tool = event.get("tool_name") or ""
    prompt = event.get("prompt") or event.get("message_content") or event.get("reasoning") or ""
    completion = event.get("completion") or event.get("output") or ""
    parts = [str(event_type)]
    if role:
        parts.append(f"agent={role}")
    if tool:
        parts.append(f"tool={tool}")
    if prompt:
        parts.append(f"prompt={str(prompt)[:220]}")
    if completion:
        parts.append(f"completion={str(completion)[:220]}")
    if _as_bool(event.get("is_deviation")):
        parts.append(f"deviation={_deviation_type(event) or 'true'}")
    return " | ".join(parts)


def _run_text(run: list[dict[str, Any]], max_events: int = 80) -> str:
    lines = [_event_line(event) for event in run[:max_events]]
    if len(run) > max_events:
        lines.append(f"... {len(run) - max_events} more events omitted ...")
    return "\n".join(lines)


def _gold(run: list[dict[str, Any]]) -> dict[str, Any]:
    deviation_events = [event for event in run if _as_bool(event.get("is_deviation"))]
    deviation_types = sorted(
        {
            str(_deviation_type(event))
            for event in deviation_events
            if _deviation_type(event) not in (None, "")
        }
    )
    agents = sorted(
        {
            str(event.get("agent_role") or event.get("agent") or event.get("agent_name"))
            for event in deviation_events
            if event.get("agent_role") or event.get("agent") or event.get("agent_name")
        }
    )
    return {
        "is_deviation": bool(deviation_events),
        "deviation_types": deviation_types,
        "responsible_agents": agents,
    }


def _to_record(run: list[dict[str, Any]]) -> dict[str, Any]:
    first = run[0] if run else {}
    question = (
        "Analyze this multi-agent workflow trace. Determine whether the run "
        "contains a process deviation/anomaly. Return JSON only with keys: "
        "is_deviation, deviation_types, responsible_agents, reason.\n\n"
        f"Trace:\n{_run_text(run)}"
    )
    return {
        "task": "open_agent_traces",
        "kind": "mas_deviation",
        "question": question,
        "gold": _gold(run),
        "context": None,
        "metadata": {
            "source": "Open Agent Traces",
            "run_id": first.get("run_id") or first.get("workflow_id") or first.get("trace_id"),
            "domain": first.get("domain"),
            "event_count": len(run),
        },
    }


def load_open_agent_traces(n: Optional[int] = None) -> list[dict[str, Any]]:
    out = [_to_record(run) for run in _group_runs(load_source_events())]
    return out[:n] if n else out


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {
        "true",
        "1",
        "yes",
        "y",
        "deviation",
        "anomaly",
        "anomalous",
    }


def _deviation_type(event: dict[str, Any]) -> Any:
    return (
        event.get("deviation_type")
        or event.get("deviation_label")
        or event.get("anomaly_type")
        or event.get("anomaly_label")
    )


def parse_deviation_answer(pred: str) -> dict[str, Any]:
    text = str(pred or "")
    data: dict[str, Any] = {}
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            loaded = json.loads(match.group(0))
            if isinstance(loaded, dict):
                data.update(loaded)
        except json.JSONDecodeError:
            pass
    low = text.lower()
    if "is_deviation" not in data:
        if re.search(r"\b(no deviation|conformant|normal|not anomal)", low):
            data["is_deviation"] = False
        elif re.search(r"\b(deviation|anomaly|anomalous|non[- ]?conform)", low):
            data["is_deviation"] = True
    types = data.get("deviation_types") or data.get("deviation_type") or []
    if isinstance(types, str):
        types = [types]
    data["deviation_types"] = [str(item).lower() for item in types]
    agents = data.get("responsible_agents") or data.get("responsible_agent") or []
    if isinstance(agents, str):
        agents = [agents]
    data["responsible_agents"] = [str(item).lower() for item in agents]
    return data


def deviation_score_details(pred: str, gold: dict[str, Any]) -> dict[str, Any]:
    parsed = parse_deviation_answer(pred)
    expected = bool(gold.get("is_deviation"))
    details = {
        "is_deviation_expected": expected,
        "is_deviation_pred": bool(parsed.get("is_deviation")),
        "deviation_decision_correct": bool(parsed.get("is_deviation")) == expected,
        "type_overlap": [],
        "score": 0.0,
    }
    if bool(parsed.get("is_deviation")) != expected:
        return details
    if not expected:
        details["score"] = 1.0
        return details
    gold_types = {str(item).lower() for item in gold.get("deviation_types") or []}
    if not gold_types:
        details["score"] = 1.0
        return details
    pred_types = set(parsed.get("deviation_types") or [])
    details["type_overlap"] = sorted(pred_types & gold_types)
    details["score"] = 1.0 if pred_types & gold_types else 0.5
    return details


def score_deviation(pred: str, gold: dict[str, Any]) -> float:
    return float(deviation_score_details(pred, gold)["score"])
