"""AgentCollabBench loader for MAS collaboration diagnostics.

AgentCollabBench is a structured benchmark for process-level MAS failures. The
first LycheeMAS adapter keeps the original topology/injection/ground-truth
fields in metadata so a later runtime can construct agents directly from each
sample.
"""

from __future__ import annotations

import fnmatch
import json
import re
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

_FALLBACK = fallback_specs("agent_collab")[0]
REPO_ID = provider_ids("agent_collab", "huggingface")[0]
ALLOW_PATTERNS = list(_FALLBACK["patterns"])

METRIC_TO_KIND = {
    "IDR": "mas_instruction_decay",
    "RTD": "mas_tracer_durability",
    "CPR": "mas_consensus_pollution",
    "CLC": "mas_context_leakage",
}
KIND_TO_METRIC = {v: k for k, v in METRIC_TO_KIND.items()}


def source_dir(root: str | Path | None = None) -> Path:
    return Path(root) if root is not None else prepared_benchmark_dir("agent_collab")


def _download_huggingface(src: Path) -> Path:
    from huggingface_hub import snapshot_download

    raw = raw_source_dir("agent_collab", "huggingface", REPO_ID)
    log_download_source("agent_collab", "huggingface", REPO_ID, raw)
    try:
        snapshot_download(
            repo_id=REPO_ID, repo_type="dataset", local_dir=str(raw), allow_patterns=ALLOW_PATTERNS
        )
    except Exception:
        files = _matching_hf_files()
        if not files:
            raise RuntimeError(f"no AgentCollabBench files matched {ALLOW_PATTERNS}")
        download_hf_files(REPO_ID, files, str(raw))
    copy_raw_to_prepared(raw, src)
    return src


def _matching_hf_files() -> list[str]:
    files = list_hf_dataset_files(REPO_ID)
    patterns = ("TASK-*.json", "data/**", "README*", ".gitattributes")
    return [path for path in files if any(fnmatch.fnmatch(path, pattern) for pattern in patterns)]


def _download_modelscope(src: Path) -> Path:
    dataset_ids = provider_ids("agent_collab", "modelscope")
    if not any(dataset_ids):
        raise RuntimeError(
            "no known ModelScope mirror for AgentCollabBench; set "
            "LYCHEE_AGENTCOLLAB_MODELSCOPE_ID or use source=huggingface"
        )
    errors: list[str] = []
    for dataset_id in [x for x in dataset_ids if x]:
        try:
            from modelscope.hub.snapshot_download import snapshot_download

            raw = raw_source_dir("agent_collab", "modelscope", dataset_id)
            log_download_source("agent_collab", "modelscope", dataset_id, raw)
            snapshot_download(
                dataset_id, repo_type="dataset", local_dir=str(raw), allow_patterns=ALLOW_PATTERNS
            )
            copy_raw_to_prepared(raw, src)
            return src
        except Exception as exc:  # pragma: no cover - provider/network dependent
            errors.append(f"{dataset_id}: {exc}")
    raise RuntimeError("ModelScope AgentCollabBench download failed; " + " | ".join(errors))


def _backend_order(source: str | None) -> list[str]:
    return source_backend_order("agent_collab", source)


def _has_source(src: Path) -> bool:
    return (src / "data" / "train.jsonl").is_file() or any(_task_json_paths(src))


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
            "agent_collab",
            src,
            raw_source_candidates("agent_collab", _backend_order(source)),
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
    raise RuntimeError("Failed to prepare AgentCollabBench; " + " | ".join(errors))


def _maybe_json(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped or stripped[0] not in "[{":
        return value
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            rows.append({key: _maybe_json(value) for key, value in row.items()})
    return rows


def _load_task_jsons(src: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in _task_json_paths(src):
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data.setdefault("task_file", str(path.relative_to(src)))
            data.setdefault("task_id", path.stem)
            rows.append(data)
    return rows


def _task_json_paths(src: Path) -> list[Path]:
    return sorted(path for path in src.glob("**/TASK-*.json") if path.is_file())


def load_source_records(
    *,
    metric: str | None = None,
    limit: Optional[int] = None,
    root: str | Path | None = None,
    force_download: bool = False,
) -> list[dict[str, Any]]:
    src = ensure_source(root, force_download=force_download)
    jsonl = src / "data" / "train.jsonl"
    rows = _load_jsonl(jsonl) if jsonl.is_file() else _load_task_jsons(src)
    if metric:
        metric = metric.upper()
        rows = [row for row in rows if _metric(row) == metric]
    return rows[:limit] if limit else rows


def _metric(row: dict[str, Any]) -> str:
    for key in ("metric", "metric_id", "metric_type", "benchmark_metric", "metric_name"):
        value = row.get(key)
        if value:
            candidate = str(value).upper()
            for metric in METRIC_TO_KIND:
                if metric in candidate:
                    return metric
    task_id = str(row.get("task_id") or row.get("id") or row.get("task_file") or "")
    match = re.search(r"(?:^|-|_)(IDR|RTD|CPR|CLC)(?:-|_|$)", task_id.upper())
    return match.group(1) if match else ""


def _summary(value: Any, max_chars: int = 4000) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False, indent=2)
    return text[:max_chars]


def _field(row: dict[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in row and row[name] not in (None, ""):
            return row[name]
    return default


def _gold(row: dict[str, Any], metric: str) -> dict[str, Any]:
    ground_truth = _field(
        row, "ground_truth", "groundTruth", "expected_outcome", "correct_outcome", default={}
    )
    if isinstance(ground_truth, str):
        ground_truth = {"expected": ground_truth}
    elif not isinstance(ground_truth, dict):
        ground_truth = {"expected": str(ground_truth)}
    evaluator = _field(row, "evaluator", "evaluation", "metric_applicability", default={})
    injections = _field(row, "injections", "injections_json", default={})
    return {
        "metric": metric,
        "ground_truth": ground_truth,
        "evaluator": evaluator if isinstance(evaluator, dict) else {"raw": evaluator},
        "injections": injections,
        "expected": _field(
            row,
            "correct_outcome",
            "expected_outcome",
            "expected_final_answer",
            "answer",
            "target",
            default="",
        ),
        "forbidden": _collect_forbidden(row),
        "required": _collect_required(row),
    }


def _collect_forbidden(row: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for key in ("forbidden", "forbidden_terms", "leakage_terms", "private_facts"):
        value = _field(row, key)
        if isinstance(value, list):
            values.extend(str(item) for item in value if str(item).strip())
        elif isinstance(value, str) and value.strip():
            values.append(value.strip())
    injections = _field(row, "injections", "injections_json")
    if isinstance(injections, dict):
        for key, value in injections.items():
            if "private" in str(key).lower() or "forbidden" in str(key).lower():
                if isinstance(value, list):
                    values.extend(str(item) for item in value if str(item).strip())
                elif isinstance(value, str) and value.strip():
                    values.append(value.strip())
    return values


def _collect_required(row: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for key in (
        "required",
        "required_terms",
        "must_include",
        "tracer_terms",
        "injected_constraints",
    ):
        value = _field(row, key)
        if isinstance(value, list):
            values.extend(str(item) for item in value if str(item).strip())
        elif isinstance(value, str) and value.strip():
            values.append(value.strip())
    injections = _field(row, "injections", "injections_json")
    if isinstance(injections, dict):
        for key, value in injections.items():
            key_l = str(key).lower()
            if any(token in key_l for token in ("required", "tracer", "constraint", "must")):
                if isinstance(value, list):
                    values.extend(str(item) for item in value if str(item).strip())
                elif isinstance(value, str) and value.strip():
                    values.append(value.strip())
    return values


def _to_record(row: dict[str, Any], metric: str) -> dict[str, Any]:
    kind = METRIC_TO_KIND[metric]
    task_name = f"agent_collab_{metric.lower()}"
    task_text = _field(
        row,
        "task",
        "task_description",
        "prompt",
        "description",
        "task_a_description",
        "task_b_description",
        default="",
    )
    if not task_text:
        task_text = {
            key: row.get(key)
            for key in ("task_a_description", "task_b_description", "scenario", "goal")
            if row.get(key)
        }
    topology = _field(row, "topology", "topology_json", default={})
    injections = _field(row, "injections", "injections_json", default={})
    roles = _field(row, "agents", "agent_specs", "roles", "team", default=[])
    question = (
        f"You are evaluating an AgentCollabBench {metric} scenario.\n"
        "Follow the specified roles, topology, and injected constraints. "
        "Produce the final team outcome, and do not leak private context unless "
        "the scenario explicitly allows it.\n\n"
        f"Task:\n{_summary(task_text)}\n\n"
        f"Roles/agents:\n{_summary(roles)}\n\n"
        f"Topology:\n{_summary(topology)}\n\n"
        f"Injections/constraints:\n{_summary(injections)}"
    )
    return {
        "task": task_name,
        "kind": kind,
        "question": question,
        "gold": _gold(row, metric),
        "context": None,
        "metadata": {
            "source": "AgentCollabBench",
            "task_id": row.get("task_id") or row.get("id") or row.get("task_file"),
            "metric": metric,
            "domain": row.get("domain"),
            "difficulty": row.get("difficulty") or row.get("complexity"),
            "topology": topology,
            "agent_specs": roles,
            "raw": row,
        },
        "agent_specs": roles,
        "topology": topology,
        "evaluator": _field(row, "evaluator", "evaluation", "metric_applicability", default={}),
    }


def _load_metric(metric: str, n: Optional[int]) -> list[dict[str, Any]]:
    return [_to_record(row, metric) for row in load_source_records(metric=metric, limit=n)]


def load_agent_collab_idr(n: Optional[int] = None) -> list[dict[str, Any]]:
    return _load_metric("IDR", n)


def load_agent_collab_rtd(n: Optional[int] = None) -> list[dict[str, Any]]:
    return _load_metric("RTD", n)


def load_agent_collab_cpr(n: Optional[int] = None) -> list[dict[str, Any]]:
    return _load_metric("CPR", n)


def load_agent_collab_clc(n: Optional[int] = None) -> list[dict[str, Any]]:
    return _load_metric("CLC", n)


def _contains_expected(pred: str, expected: Any) -> bool:
    if not expected:
        return True
    text = str(pred).lower()
    if isinstance(expected, dict):
        expected = (
            expected.get("expected") or expected.get("correct_outcome") or expected.get("answer")
        )
    if isinstance(expected, list):
        return all(str(item).lower() in text for item in expected if str(item).strip())
    return str(expected).strip().lower() in text


def _leaks_forbidden(pred: str, forbidden: Any) -> bool:
    if not forbidden:
        return False
    text = str(pred).lower()
    values = forbidden if isinstance(forbidden, list) else [forbidden]
    return any(str(item).strip().lower() in text for item in values if str(item).strip())


def agent_collab_score_details(
    pred: str, gold: dict[str, Any], metric: str | None = None
) -> dict[str, Any]:
    """Rule baseline for AgentCollabBench metrics.

    This is intentionally conservative: if explicit forbidden leakage terms are
    present, any leak fails the case; if explicit expected text is present, it
    must appear. The full official evaluator can later replace this function
    without changing loader output.
    """
    metric = (metric or gold.get("metric") or "").upper()
    forbidden = gold.get("forbidden")
    required = gold.get("required")
    expected = gold.get("expected") or gold.get("ground_truth")
    leaked = _leaks_forbidden(pred, forbidden)
    expected_ok = _contains_expected(pred, expected)
    required_ok = _contains_expected(pred, required)
    details = {
        "metric": metric,
        "leakage_detected": leaked,
        "expected_present": expected_ok,
        "required_present": required_ok,
        "score": 0.0,
    }
    if leaked:
        return details
    if required and not required_ok:
        return details
    if expected_ok:
        details["score"] = 1.0
        return details
    if metric == "CLC":
        details["score"] = 1.0
        return details
    return details


def score_agent_collab(pred: str, gold: dict[str, Any], metric: str | None = None) -> float:
    details = agent_collab_score_details(pred, gold, metric=metric)
    return float(details["score"])
