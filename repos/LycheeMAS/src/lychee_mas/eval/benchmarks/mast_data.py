"""MAST/MAD failure-taxonomy benchmark loader."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Optional

from .common import (
    copy_raw_to_prepared,
    download_hf_files,
    json_file_ready,
    log_download_source,
    prepared_benchmark_dir,
    raw_source_candidates,
    raw_source_dir,
    remove_path,
    restore_prepared_from_raw,
)
from .source_catalog import fallback_specs, provider_ids, source_backend_order

_FALLBACK = fallback_specs("mast_data")[0]
REPO_ID = provider_ids("mast_data", "huggingface")[0]
ALLOW_PATTERNS = ["*.json", "README*", ".gitattributes"]
DEFAULT_FILES = list(_FALLBACK["files"])
FULL_FILES = list(_FALLBACK.get("optional_full_files", []))


def source_dir(root: str | Path | None = None) -> Path:
    return Path(root) if root is not None else prepared_benchmark_dir("mast_data")


def _download_huggingface(src: Path) -> Path:
    from huggingface_hub import snapshot_download

    raw = raw_source_dir("mast_data", "huggingface", REPO_ID)
    log_download_source("mast_data", "huggingface", REPO_ID, raw)
    try:
        snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            local_dir=str(raw),
            allow_patterns=_download_patterns(),
        )
    except Exception:
        download_hf_files(REPO_ID, _download_files(), str(raw))
    copy_raw_to_prepared(raw, src)
    return src


def _download_modelscope(src: Path) -> Path:
    dataset_ids = provider_ids("mast_data", "modelscope")
    if not any(dataset_ids):
        raise RuntimeError(
            "no known ModelScope mirror for MAST-Data; set LYCHEE_MAST_MODELSCOPE_ID "
            "or use source=huggingface"
        )
    errors: list[str] = []
    for dataset_id in [x for x in dataset_ids if x]:
        try:
            from modelscope.hub.snapshot_download import snapshot_download

            raw = raw_source_dir("mast_data", "modelscope", dataset_id)
            log_download_source("mast_data", "modelscope", dataset_id, raw)
            snapshot_download(
                dataset_id,
                repo_type="dataset",
                local_dir=str(raw),
                allow_patterns=_download_patterns(),
            )
            copy_raw_to_prepared(raw, src)
            return src
        except Exception as exc:  # pragma: no cover - provider/network dependent
            errors.append(f"{dataset_id}: {exc}")
    raise RuntimeError("ModelScope MAST-Data download failed; " + " | ".join(errors))


def _download_full_dataset() -> bool:
    return str(os.environ.get("LYCHEE_MAST_DOWNLOAD_FULL", "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
    }


def _download_files() -> list[str]:
    return [*DEFAULT_FILES, *(FULL_FILES if _download_full_dataset() else [])]


def _download_patterns() -> list[str]:
    return [*_download_files(), "README*", ".gitattributes"]


def _has_source(src: Path) -> bool:
    return any(json_file_ready(path) for path in src.glob("MAD_*dataset.json"))


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
            "mast_data",
            src,
            raw_source_candidates("mast_data", _backend_order(source)),
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
    raise RuntimeError("Failed to prepare MAST-Data; " + " | ".join(errors))


def _backend_order(source: Optional[str]) -> list[str]:
    return source_backend_order("mast_data", source)


def _dataset_path(src: Path) -> Path:
    preferred = [
        src / "MAD_human_labelled_dataset.json",
        src / "MAD_full_dataset.json",
        src / "MAD_annotated_dataset.json",
    ]
    for path in preferred:
        if path.is_file():
            return path
    matches = sorted(src.glob("MAD_*dataset.json"))
    if not matches:
        raise FileNotFoundError(f"No MAST json dataset found under {src}")
    return matches[0]


def _load_json_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if stripped.startswith("["):
        data = json.loads(text)
        return [row for row in data if isinstance(row, dict)]
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if isinstance(row, dict):
            rows.append(row)
    return rows


def load_source_records(
    *,
    root: str | Path | None = None,
    force_download: bool = False,
) -> list[dict[str, Any]]:
    src = ensure_source(root, force_download=force_download)
    return _load_json_records(_dataset_path(src))


def _field(row: dict[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in row and row[name] not in (None, ""):
            return row[name]
    return default


def extract_taxonomy_labels(annotation: Any) -> list[str]:
    if isinstance(annotation, str):
        try:
            annotation = json.loads(annotation)
        except json.JSONDecodeError:
            return sorted(set(re.findall(r"\b[1-3]\.\d+\b", annotation)))
    if isinstance(annotation, dict):
        labels: list[str] = []
        for key, value in annotation.items():
            if re.fullmatch(r"[1-3]\.\d+", str(key)) and bool(value):
                labels.append(str(key))
            elif isinstance(value, dict):
                labels.extend(extract_taxonomy_labels(value))
            elif isinstance(value, list):
                for item in value:
                    labels.extend(extract_taxonomy_labels(item))
            elif isinstance(value, str):
                labels.extend(re.findall(r"\b[1-3]\.\d+\b", value))
        return sorted(set(labels))
    if isinstance(annotation, list):
        labels: list[str] = []
        for item in annotation:
            labels.extend(extract_taxonomy_labels(item))
        return sorted(set(labels))
    return []


def _trace_text(row: dict[str, Any], max_chars: int = 12000) -> str:
    trace = _field(
        row,
        "trajectory",
        "trace",
        "raw_trace",
        "conversation",
        "messages",
        "history",
        "dialogue",
        default="",
    )
    if not isinstance(trace, str):
        trace = json.dumps(trace, ensure_ascii=False, indent=2)
    return trace[:max_chars]


def _to_record(row: dict[str, Any]) -> dict[str, Any]:
    annotation = _field(row, "mast_annotation", "annotation", "annotations", "taxonomy", default={})
    labels = extract_taxonomy_labels(annotation)
    task_name = _field(row, "benchmark_task", "task", "scenario", "task_description", default="")
    question = (
        "Classify the failure modes in this multi-agent execution trace using "
        "the MAST taxonomy labels. Return JSON only with key labels, whose value "
        'is a list such as ["1.1", "2.3"].\n\n'
        f"MAS/framework: {_field(row, 'mas', 'framework', 'source_project', default='unknown')}\n"
        f"LLM: {_field(row, 'llm', 'model', default='unknown')}\n"
        f"Task: {task_name}\n\n"
        f"Trace:\n{_trace_text(row)}"
    )
    return {
        "task": "mast_failure",
        "kind": "mas_failure_taxonomy",
        "question": question,
        "gold": {"labels": labels, "annotation": annotation},
        "context": None,
        "metadata": {
            "source": "MAST-Data",
            "index": row.get("index"),
            "key": row.get("key"),
            "mas": _field(row, "mas", "framework", "source_project"),
            "model": _field(row, "llm", "model"),
            "benchmark_task": task_name,
            "label_count": len(labels),
        },
    }


def load_mast_failure(n: Optional[int] = None) -> list[dict[str, Any]]:
    out = [_to_record(row) for row in load_source_records()]
    return out[:n] if n else out


def parse_taxonomy_answer(pred: str) -> list[str]:
    text = str(pred or "")
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            if isinstance(data, dict):
                labels = data.get("labels") or data.get("label") or data.get("taxonomy_labels")
                if isinstance(labels, str):
                    return sorted(set(re.findall(r"\b[1-3]\.\d+\b", labels)))
                if isinstance(labels, list):
                    return sorted(
                        set(str(x) for x in labels if re.fullmatch(r"[1-3]\.\d+", str(x)))
                    )
        except json.JSONDecodeError:
            pass
    return sorted(set(re.findall(r"\b[1-3]\.\d+\b", text)))


def score_taxonomy(pred: str, gold: dict[str, Any]) -> float:
    return float(taxonomy_score_details(pred, gold)["score"])


def taxonomy_score_details(pred: str, gold: dict[str, Any]) -> dict[str, Any]:
    pred_labels = set(parse_taxonomy_answer(pred))
    gold_labels = set(gold.get("labels") or [])
    details = {
        "pred_labels": sorted(pred_labels),
        "gold_labels": sorted(gold_labels),
        "true_positive_labels": sorted(pred_labels & gold_labels),
        "score": 0.0,
    }
    if not pred_labels and not gold_labels:
        details["score"] = 1.0
        return details
    if not pred_labels or not gold_labels:
        return details
    tp = len(pred_labels & gold_labels)
    details["score"] = (2 * tp) / (len(pred_labels) + len(gold_labels))
    return details
