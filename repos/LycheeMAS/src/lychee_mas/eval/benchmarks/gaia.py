"""GAIA loader + official-style answer scoring.

GAIA is cached under the benchmark raw root by default. Only the validation
split is registered for scoring because it contains public gold answers; the
official test split is for leaderboard submission and does not fit LycheeMAS'
local ``gold``-based evaluator.
"""

from __future__ import annotations

import json
import re
import string
from pathlib import Path
from typing import Any, Optional

from .common import (
    copy_raw_to_prepared,
    log_download_source,
    prepared_benchmark_dir,
    raw_source_candidates,
    raw_source_dir,
    restore_prepared_from_raw,
)
from .source_catalog import provider_ids, source_backend_order

REPO_ID = provider_ids("gaia", "huggingface")[0]
VALIDATION_ALLOW_PATTERNS = [
    "README*",
    ".gitattributes",
    "2023/validation/**",
]


def source_dir(root: str | Path | None = None) -> Path:
    return Path(root) if root is not None else prepared_benchmark_dir("gaia")


def safe_task_id(task_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", task_id).strip("_")


def _download_huggingface(src: Path, allow_patterns: list[str] | None = None) -> Path:
    from huggingface_hub import snapshot_download

    kwargs = {"repo_id": REPO_ID, "repo_type": "dataset", "local_dir": str(src)}
    if allow_patterns is not None:
        kwargs["allow_patterns"] = allow_patterns
    log_download_source("gaia", "huggingface", REPO_ID, src)
    snapshot_download(**kwargs)
    return src


def _download_modelscope(
    src: Path, allow_patterns: list[str] | None = None, dataset_id: str | None = None
) -> Path:
    dataset_ids = [dataset_id] if dataset_id else provider_ids("gaia", "modelscope")
    errors: list[str] = []
    for dataset_id in [x for x in dataset_ids if x]:
        try:
            from modelscope.hub.snapshot_download import snapshot_download

            kwargs = {"repo_type": "dataset", "local_dir": str(src)}
            if allow_patterns is not None:
                kwargs["allow_patterns"] = allow_patterns
            log_download_source("gaia", "modelscope", dataset_id, src)
            snapshot_download(dataset_id, **kwargs)
            return src
        except Exception as exc:
            errors.append(f"{dataset_id}: {exc}")
    raise RuntimeError("ModelScope GAIA download failed; " + " | ".join(errors))


def _backend_order(source: str | None) -> list[str]:
    return source_backend_order("gaia", source)


def _has_metadata(split_dir: Path) -> bool:
    try:
        return bool(_load_metadata(split_dir))
    except Exception:
        return False


def _download_with_backends(
    prepared: Path,
    source: str | None,
    *,
    allow_patterns: list[str] | None,
    label: str,
) -> Path:
    prepared.parent.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    for backend in _backend_order(source):
        try:
            if backend == "modelscope":
                for dataset_id in [x for x in provider_ids("gaia", "modelscope") if x]:
                    raw = raw_source_dir("gaia", "modelscope", dataset_id)
                    try:
                        _download_modelscope(
                            raw, allow_patterns=allow_patterns, dataset_id=dataset_id
                        )
                        return copy_raw_to_prepared(raw, prepared)
                    except Exception as exc:
                        errors.append(f"modelscope/{dataset_id}: {exc}")
                continue
            raw = raw_source_dir("gaia", "huggingface", REPO_ID)
            _download_huggingface(raw, allow_patterns=allow_patterns)
            return copy_raw_to_prepared(raw, prepared)
        except Exception as exc:
            errors.append(f"{backend}: {exc}")
    raise RuntimeError(f"Failed to prepare {label}; " + " | ".join(errors))


def ensure_validation_source(
    root: str | Path | None = None,
    force_download: bool = False,
    source: str | None = None,
) -> Path:
    src = source_dir(root)
    validation_dir = src / "2023" / "validation"
    if _has_metadata(validation_dir) and not force_download:
        return src
    if not force_download:
        restored = restore_prepared_from_raw(
            "gaia",
            src,
            raw_source_candidates("gaia", _backend_order(source)),
            ready=lambda path: _has_metadata(path / "2023" / "validation"),
        )
        if restored is not None:
            return restored
    return _download_with_backends(
        src,
        source,
        allow_patterns=VALIDATION_ALLOW_PATTERNS,
        label="GAIA validation",
    )


def ensure_full_source(
    root: str | Path | None = None,
    force_download: bool = False,
    source: str | None = None,
) -> Path:
    src = source_dir(root)
    validation_dir = src / "2023" / "validation"
    test_dir = src / "2023" / "test"
    if _has_metadata(validation_dir) and _has_metadata(test_dir) and not force_download:
        return src
    if not force_download:
        restored = restore_prepared_from_raw(
            "gaia",
            src,
            raw_source_candidates("gaia", _backend_order(source)),
            ready=lambda path: (
                _has_metadata(path / "2023" / "validation")
                and _has_metadata(path / "2023" / "test")
            ),
        )
        if restored is not None:
            return restored
    return _download_with_backends(src, source, allow_patterns=None, label="GAIA")


def ensure_source(
    root: str | Path | None = None,
    force_download: bool = False,
    source: str | None = None,
) -> Path:
    """Backward-compatible alias for the locally scored validation split."""
    return ensure_validation_source(root, force_download=force_download, source=source)


def _split_dir(split: str = "validation", root: str | Path | None = None) -> Path:
    if split != "validation":
        raise ValueError("Only GAIA validation is supported for local scoring.")
    return source_dir(root) / "2023" / split


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _metadata_path(split_dir: Path) -> Path:
    jsonl = split_dir / "metadata.jsonl"
    if jsonl.is_file():
        return jsonl
    return split_dir / "metadata.parquet"


def _load_metadata(split_dir: Path) -> list[dict[str, Any]]:
    path = _metadata_path(split_dir)
    if path.suffix == ".jsonl":
        return _load_jsonl(path)
    if path.suffix == ".parquet" and path.is_file():
        from datasets import load_dataset

        dataset = load_dataset(
            "parquet", data_files={"validation": [str(path)]}, split="validation"
        )
        return [dict(row) for row in dataset]
    raise FileNotFoundError(f"GAIA metadata not found under {split_dir}")


def load_source_records(
    level: Optional[int] = None,
    root: str | Path | None = None,
    force_download: bool = False,
) -> list[dict[str, Any]]:
    ensure_validation_source(root, force_download=force_download)
    rows = _load_metadata(_split_dir("validation", root))
    rows = [row for row in rows if row.get("task_id") != "0-0-0-0-0"]
    if level is not None:
        rows = [row for row in rows if int(row["Level"]) == int(level)]
    return rows


def _attachment_path(task: dict[str, Any], root: str | Path | None = None) -> Optional[Path]:
    file_name = str(task.get("file_name", "") or "").strip()
    if not file_name:
        return None
    path = _split_dir("validation", root) / file_name
    return path if path.exists() else None


def _format_question(
    task: dict[str, Any], root: str | Path | None = None
) -> tuple[str, Optional[str]]:
    question = str(task.get("Question", "")).strip()
    attachment = _attachment_path(task, root)
    if attachment is None:
        return question, None
    context = f"Referenced file path: {attachment}"
    return f"{question}\n\n{context}", context


def _load_gaia(level: Optional[int], n: Optional[int]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    task_name = "gaia_validation" + (f"_level_{level}" if level else "")
    for task in load_source_records(level=level):
        question, context = _format_question(task)
        raw_task_id = task["task_id"]
        out.append(
            {
                "task": task_name,
                "kind": "gaia",
                "question": question,
                "gold": str(task.get("Final answer", "")).strip(),
                "context": context,
                "metadata": {
                    "task_id": raw_task_id,
                    "safe_task_id": safe_task_id(raw_task_id),
                    "level": task.get("Level"),
                    "file_name": str(task.get("file_name", "") or "").strip(),
                },
            }
        )
        if n and len(out) >= n:
            break
    return out


def load_gaia_validation(n: Optional[int] = None) -> list[dict[str, Any]]:
    return _load_gaia(None, n)


def load_gaia_validation_level_1(n: Optional[int] = None) -> list[dict[str, Any]]:
    return _load_gaia(1, n)


def load_gaia_validation_level_2(n: Optional[int] = None) -> list[dict[str, Any]]:
    return _load_gaia(2, n)


def load_gaia_validation_level_3(n: Optional[int] = None) -> list[dict[str, Any]]:
    return _load_gaia(3, n)


def normalize_number_str(number_str: str) -> float:
    for char in ["$", "%", ","]:
        number_str = number_str.replace(char, "")
    try:
        return float(number_str)
    except ValueError:
        return float("inf")


def split_string(value: str, char_list: list[str] | None = None) -> list[str]:
    chars = char_list or [",", ";"]
    pattern = f"[{''.join(chars)}]"
    return re.split(pattern, value)


def normalize_str(input_str: str, remove_punct: bool = True) -> str:
    no_spaces = re.sub(r"\s", "", input_str)
    if remove_punct:
        translator = str.maketrans("", "", string.punctuation)
        return no_spaces.lower().translate(translator)
    return no_spaces.lower()


def is_float(element: object) -> bool:
    try:
        float(str(element))
        return True
    except ValueError:
        return False


def gaia_question_scorer(model_answer: str, ground_truth: str) -> bool:
    if is_float(ground_truth):
        normalized_answer = normalize_number_str(model_answer)
        return normalized_answer == float(ground_truth)

    if any(char in ground_truth for char in [",", ";"]):
        gt_elems = split_string(ground_truth)
        ma_elems = split_string(model_answer)
        if len(gt_elems) != len(ma_elems):
            return False

        comparisons = []
        for ma_elem, gt_elem in zip(ma_elems, gt_elems):
            if is_float(gt_elem):
                normalized_ma_elem = normalize_number_str(ma_elem)
                comparisons.append(normalized_ma_elem == float(gt_elem))
            else:
                comparisons.append(
                    normalize_str(ma_elem, remove_punct=False)
                    == normalize_str(gt_elem, remove_punct=False)
                )
        return all(comparisons)

    return normalize_str(model_answer) == normalize_str(ground_truth)


def extract_final_answer(console_log: str) -> str | None:
    matches = re.findall(r"FINAL ANSWER:(.*?)(?:\n|$)", console_log, re.DOTALL)
    if not matches:
        return None
    return matches[-1].strip()
